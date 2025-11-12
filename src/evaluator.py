"""评估模块"""

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import Config, load_config
from src.document_parser import DocumentParser


class EvaluationResult:
    """单次评估结果"""

    def __init__(
        self,
        point_id: str,
        exists: bool,
        accuracy: float,
        explanation: str,
    ):
        self.point_id = point_id
        self.exists = exists
        self.accuracy = accuracy
        self.explanation = explanation


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

            # 准确性：所有要点准确性得分之和 / 总要点数
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
                evaluations.append(
                    EvaluationResult(
                        point_id=eval_data.get("point_id", ""),
                        exists=eval_data.get("exists", False),
                        accuracy=float(eval_data.get("accuracy", 0.0)),
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
        多次运行评估并取平均

        Args:
            points: 要点清单
            target_document_path: 待评估文档路径
            runs: 运行次数

        Returns:
            平均评估结果
        """
        results = []
        for i in range(runs):
            result = self.evaluate_single_run(points, target_document_path)
            results.append(result)

        # 计算平均值
        avg_completeness = sum(r.completeness for r in results) / len(results)
        avg_accuracy = sum(r.accuracy for r in results) / len(results)
        avg_comprehensive = sum(r.comprehensive for r in results) / len(results)

        # 合并评估结果（取平均准确性）
        point_evaluations = {}
        for result in results:
            for eval_result in result.evaluations:
                point_id = eval_result.point_id
                if point_id not in point_evaluations:
                    point_evaluations[point_id] = {
                        "exists_count": 0,
                        "accuracy_sum": 0.0,
                        "explanations": [],
                    }
                if eval_result.exists:
                    point_evaluations[point_id]["exists_count"] += 1
                point_evaluations[point_id]["accuracy_sum"] += eval_result.accuracy
                point_evaluations[point_id]["explanations"].append(
                    eval_result.explanation
                )

        # 创建平均评估结果
        avg_evaluations = []
        for point_id, data in point_evaluations.items():
            exists = data["exists_count"] > (runs / 2)  # 多数存在则认为存在
            avg_accuracy = data["accuracy_sum"] / runs
            # 使用最常见的解释
            explanation = max(
                set(data["explanations"]), key=data["explanations"].count
            )
            avg_evaluations.append(
                EvaluationResult(
                    point_id=point_id,
                    exists=exists,
                    accuracy=avg_accuracy,
                    explanation=explanation,
                )
            )

        # 创建最终结果对象
        final_result = DocumentEvaluation(
            target_document=str(target_document_path),
            points=points,
            evaluations=avg_evaluations,
        )

        # 使用计算出的平均值覆盖
        final_result.completeness = avg_completeness
        final_result.accuracy = avg_accuracy
        final_result.comprehensive = avg_comprehensive

        return final_result

