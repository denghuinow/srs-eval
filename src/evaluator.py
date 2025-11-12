"""评估模块"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import Config, load_config
from src.document_parser import DocumentParser


class CheckpointResult:
    """检查项结果"""

    def __init__(self, checkpoint: str, passed: bool):
        self.checkpoint = checkpoint
        self.passed = passed


class EvaluationResult:
    """单次评估结果"""

    def __init__(
        self,
        point_id: str,
        exists: bool,
        checkpoint_results: list[CheckpointResult] | None = None,
        accuracy: float | None = None,  # 向后兼容，如果提供则使用，否则从checkpoint_results计算
        explanation: str = "",
    ):
        self.point_id = point_id
        self.exists = exists
        self.checkpoint_results = checkpoint_results or []
        self.explanation = explanation

        # 计算准确性：如果提供了accuracy则使用，否则基于检查项通过率计算
        if accuracy is not None:
            self.accuracy = accuracy
        elif self.checkpoint_results:
            passed_count = sum(1 for cp in self.checkpoint_results if cp.passed)
            total_count = len(self.checkpoint_results)
            self.accuracy = (passed_count / total_count) if total_count > 0 else 0.0
        else:
            # 向后兼容：如果没有检查项结果，使用默认值
            self.accuracy = 0.0


class DocumentEvaluation:
    """文档评估结果"""

    def __init__(
        self,
        target_document: str,
        points: list[dict[str, Any]],
        evaluations: list[EvaluationResult],
    ):
        self.target_document = target_document
        self.points = points
        self.evaluations = evaluations

        # 计算分数
        total_points = len(points)
        if total_points == 0:
            self.completeness = 0.0
            self.accuracy = 0.0
            self.comprehensive = 0.0
        else:
            # 完整性：存在的要点数 / 总要点数
            existing_count = sum(1 for e in evaluations if e.exists)
            self.completeness = (existing_count / total_points) * 100

            # 准确性：基于检查项通过率计算
            # 方法1：如果有点的检查项，计算所有检查项的通过率
            total_checkpoints = 0
            passed_checkpoints = 0
            for point in points:
                point_id = point.get("id", "")
                # 找到对应的评估结果
                eval_result = next(
                    (e for e in evaluations if e.point_id == point_id), None
                )
                if eval_result:
                    checkpoints = point.get("checkpoints", [])
                    total_checkpoints += len(checkpoints)
                    if eval_result.checkpoint_results:
                        passed_checkpoints += sum(
                            1 for cp in eval_result.checkpoint_results if cp.passed
                        )
                    else:
                        # 向后兼容：如果没有检查项结果，使用accuracy字段
                        passed_checkpoints += eval_result.accuracy * len(checkpoints)

            if total_checkpoints > 0:
                self.accuracy = (passed_checkpoints / total_checkpoints) * 100
            else:
                # 向后兼容：如果没有检查项，使用旧的准确性计算方法
                accuracy_sum = sum(e.accuracy for e in evaluations)
                self.accuracy = (accuracy_sum / total_points) * 100

            # 综合分数：完整性 × 0.5 + 准确性 × 0.5
            self.comprehensive = (
                self.completeness * 0.5 + self.accuracy * 0.5
            )


class Evaluator:
    """文档评估器"""

    def __init__(self, config: Config | None = None):
        """
        初始化评估器

        Args:
            config: 配置对象，如果为None则自动加载
        """
        self.config = config or load_config()
        self.client = OpenAI(
            api_key=self.config.openai.api_key,
            base_url=self.config.openai.base_url,
        )
        self.parser = DocumentParser()

    def _load_prompt_template(self, template_name: str) -> str:
        """
        加载prompt模板

        Args:
            template_name: 模板文件名

        Returns:
            模板内容
        """
        template_path = (
            Path(__file__).parent.parent / "prompts" / template_name
        )
        if not template_path.exists():
            raise FileNotFoundError(f"模板文件不存在: {template_path}")
        return template_path.read_text(encoding="utf-8")

    def evaluate_single_run(
        self,
        points: list[dict[str, Any]],
        target_document_path: str | Path,
    ) -> DocumentEvaluation:
        """
        单次评估运行

        Args:
            points: 要点清单
            target_document_path: 待评估文档路径

        Returns:
            评估结果
        """
        # 读取待评估文档内容
        target_content = self.parser.read_markdown(target_document_path)

        # 加载prompt模板
        template = self._load_prompt_template("evaluate_points.txt")

        # 准备要点JSON
        points_json = json.dumps({"points": points}, ensure_ascii=False, indent=2)

        # 填充模板
        prompt = template.format(
            points_json=points_json, target_content=target_content
        )

        # 调用OpenAI API
        # 尝试使用response_format，如果不支持则回退
        api_params = {
            "model": self.config.openai.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的需求文档评估专家。严格按照JSON格式输出，不要有任何其他文字。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.openai.temperature,
            "max_tokens": self.config.openai.max_tokens,
        }
        
        # 尝试添加response_format，如果API不支持会自动失败并回退
        try:
            api_params["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**api_params)
        except Exception as e:
            # 如果response_format不支持，尝试不使用它
            if "response_format" in str(e).lower() or "response" in str(e).lower():
                print("警告: API可能不支持response_format参数，尝试不使用该参数...")
                api_params.pop("response_format", None)
                try:
                    response = self.client.chat.completions.create(**api_params)
                except Exception as e2:
                    raise ValueError(f"API调用失败: {e2}")
            else:
                raise ValueError(f"API调用失败: {e}")

        # 检查响应
        if not response or not response.choices:
            raise ValueError(f"API返回无效响应: {response}")

        # 解析响应
        result_text = response.choices[0].message.content
        if not result_text:
            # 尝试获取更多信息
            error_msg = f"API返回结果为空。响应对象: {response}"
            if hasattr(response, "error"):
                error_msg += f", 错误信息: {response.error}"
            raise ValueError(error_msg)

        try:
            result_json = json.loads(result_text)
            evaluations_data = result_json.get("evaluations", [])

            # 转换为EvaluationResult对象
            evaluations = []
            for eval_data in evaluations_data:
                exists = eval_data.get("exists", False)
                point_id = eval_data.get("point_id", "")
                
                # 找到对应的要点，获取检查项列表
                point = next((p for p in points if p.get("id") == point_id), None)
                expected_checkpoints = point.get("checkpoints", []) if point else []
                
                # 解析检查项结果
                checkpoint_results = []
                checkpoint_results_data = eval_data.get("checkpoint_results", [])
                
                if checkpoint_results_data and expected_checkpoints:
                    # 新格式：基于检查项结果
                    # 创建检查项映射（用于匹配）
                    checkpoint_map = {
                        cp_data.get("checkpoint", ""): cp_data.get("passed", False)
                        for cp_data in checkpoint_results_data
                    }
                    
                    # 按照要点中的检查项顺序创建结果
                    for expected_cp in expected_checkpoints:
                        passed = checkpoint_map.get(expected_cp, False)
                        # 如果要点不存在，所有检查项必须为False
                        if not exists:
                            passed = False
                        checkpoint_results.append(
                            CheckpointResult(checkpoint=expected_cp, passed=passed)
                        )
                elif expected_checkpoints:
                    # 如果API没有返回检查项结果，但要点有检查项，全部标记为未通过
                    for expected_cp in expected_checkpoints:
                        checkpoint_results.append(
                            CheckpointResult(checkpoint=expected_cp, passed=False)
                        )
                
                # 向后兼容：如果没有检查项结果，尝试使用accuracy字段
                accuracy = None
                if not checkpoint_results and "accuracy" in eval_data:
                    accuracy = float(eval_data.get("accuracy", 0.0))
                    if not exists:
                        accuracy = 0.0
                
                evaluations.append(
                    EvaluationResult(
                        point_id=point_id,
                        exists=exists,
                        checkpoint_results=checkpoint_results if checkpoint_results else None,
                        accuracy=accuracy,
                        explanation=eval_data.get("explanation", ""),
                    )
                )

            return DocumentEvaluation(
                target_document=str(target_document_path),
                points=points,
                evaluations=evaluations,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"解析评估结果失败: {e}, 原始响应: {result_text}")

    def evaluate_multiple_runs(
        self,
        points: list[dict[str, Any]],
        target_document_path: str | Path,
        runs: int = 3,
    ) -> DocumentEvaluation:
        """
        多次运行评估并取平均（并行执行）

        Args:
            points: 要点清单
            target_document_path: 待评估文档路径
            runs: 运行次数

        Returns:
            平均评估结果
        """
        results = []
        
        # 并行执行多次评估
        with ThreadPoolExecutor(max_workers=runs) as executor:
            futures = {
                executor.submit(self.evaluate_single_run, points, target_document_path): i
                for i in range(runs)
            }
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    print(f"  [{completed}/{runs}] 评估运行完成")
                except Exception as e:
                    print(f"  [{completed}/{runs}] 评估运行失败: {e}")
                    continue

        # 合并评估结果（基于检查项结果）
        # 注意：不再计算平均值，而是基于合并后的检查项结果重新计算分数
        point_evaluations = {}
        for result in results:
            for eval_result in result.evaluations:
                point_id = eval_result.point_id
                if point_id not in point_evaluations:
                    point_evaluations[point_id] = {
                        "exists_count": 0,
                        "checkpoint_votes": {},  # checkpoint -> {passed_count, total_count}
                        "explanations": [],
                    }
                if eval_result.exists:
                    point_evaluations[point_id]["exists_count"] += 1
                
                # 统计检查项结果
                if eval_result.checkpoint_results:
                    for cp_result in eval_result.checkpoint_results:
                        cp_text = cp_result.checkpoint
                        if cp_text not in point_evaluations[point_id]["checkpoint_votes"]:
                            point_evaluations[point_id]["checkpoint_votes"][cp_text] = {
                                "passed_count": 0,
                                "total_count": 0,
                            }
                        point_evaluations[point_id]["checkpoint_votes"][cp_text]["total_count"] += 1
                        if cp_result.passed:
                            point_evaluations[point_id]["checkpoint_votes"][cp_text]["passed_count"] += 1
                
                point_evaluations[point_id]["explanations"].append(
                    eval_result.explanation
                )

        # 创建平均评估结果
        avg_evaluations = []
        for point_id, data in point_evaluations.items():
            exists = data["exists_count"] > (runs / 2)  # 多数存在则认为存在
            
            # 合并检查项结果：如果多数运行认为通过，则通过
            avg_checkpoint_results = []
            for cp_text, votes in data["checkpoint_votes"].items():
                passed = votes["passed_count"] > (votes["total_count"] / 2)
                # 如果要点不存在，所有检查项必须为False
                if not exists:
                    passed = False
                avg_checkpoint_results.append(
                    CheckpointResult(checkpoint=cp_text, passed=passed)
                )
            
            # 使用最常见的解释
            explanation = max(
                set(data["explanations"]), key=data["explanations"].count
            )
            
            avg_evaluations.append(
                EvaluationResult(
                    point_id=point_id,
                    exists=exists,
                    checkpoint_results=avg_checkpoint_results if avg_checkpoint_results else None,
                    explanation=explanation,
                )
            )

        # 创建最终结果对象
        # DocumentEvaluation 的 __init__ 方法会基于合并后的检查项结果自动计算分数
        # 这样确保总体分数与详细评估结果表格一致
        final_result = DocumentEvaluation(
            target_document=str(target_document_path),
            points=points,
            evaluations=avg_evaluations,
        )

        # 不再使用平均值覆盖，而是基于合并后的检查项结果重新计算
        # 这样确保总体分数与详细评估结果表格完全一致

        return final_result

