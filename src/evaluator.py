"""评估模块"""

import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import Config, load_config
from src.document_parser import DocumentParser

# 配置日志
logger = logging.getLogger(__name__)


class MergeStrategy(str, Enum):
    """合并策略枚举"""
    AVERAGE = "average"  # 当前方法：平均值
    MAJORITY = "majority"  # 多数投票（>50%）
    CONSENSUS_67 = "consensus_67"  # 2/3一致性
    CONSENSUS_75 = "consensus_75"  # 3/4一致性
    MEDIAN = "median"  # 中位数：计算所有评委通过率的中位数
    WEIGHTED = "weighted"  # 加权投票
    CONFIDENCE = "confidence"  # 置信区间法
    TRIMMED_MEAN = "trimmed_mean"  # 截断均值（去除最高和最低分后取平均）
    QUANTILE_WEIGHTED = "quantile_weighted"  # 分位数加权（根据分数分布位置加权）
    BAYESIAN_AVERAGE = "bayesian_average"  # 贝叶斯平均（结合先验信息）


class CheckpointResult:
    """检查项结果"""

    def __init__(self, checkpoint: str, passed: bool, pass_rate: float | None = None):
        self.checkpoint = checkpoint
        self.passed = passed
        # pass_rate: 通过率 (0.0-1.0)，用于加权平均计算
        # 如果未提供，则基于passed计算：passed=True时为1.0，False时为0.0
        if pass_rate is not None:
            self.pass_rate = pass_rate
        else:
            self.pass_rate = 1.0 if passed else 0.0


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

            # 准确性：基于检查项通过情况计算
            # 方法1：如果有点的检查项，计算所有检查项的通过情况
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
                        # 使用passed判断（是否通过）来计算准确性，这样不同策略会产生不同结果
                        passed_checkpoints += sum(
                            1 if cp.passed else 0 for cp in eval_result.checkpoint_results
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

    @staticmethod
    def _extract_json_from_markdown(text: str) -> str:
        """
        从markdown代码块中提取JSON内容
        
        处理以下情况：
        - ```json ... ```
        - ``` ... ```
        - 纯JSON（无代码块标记）
        
        Args:
            text: 可能包含markdown代码块的文本
            
        Returns:
            清理后的JSON文本
        """
        text = text.strip()
        
        # 尝试匹配 ```json ... ``` 或 ``` ... ```
        if text.startswith("```"):
            # 找到第一个换行符（代码块开始标记之后）
            first_newline = text.find("\n")
            if first_newline != -1:
                # 移除开始标记（```json 或 ```）
                text = text[first_newline + 1:]
            
            # 找到最后一个 ```（代码块结束标记）
            last_code_block = text.rfind("```")
            if last_code_block != -1:
                # 移除结束标记
                text = text[:last_code_block]
        
        return text.strip()

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
        logger.info(f"开始评估文档: {target_document_path}")
        logger.debug(f"文档内容长度: {len(target_content)} 字符")

        # 加载prompt模板
        template = self._load_prompt_template("evaluate_points.txt")

        # 准备要点JSON
        points_json = json.dumps({"points": points}, ensure_ascii=False, indent=2)
        logger.debug(f"要点数量: {len(points)}, JSON长度: {len(points_json)} 字符")

        # 填充模板
        prompt = template.format(
            points_json=points_json, target_content=target_content
        )
        logger.debug(f"完整prompt长度: {len(prompt)} 字符")
        # 记录完整Prompt内容
        logger.debug("=" * 80)
        logger.debug("完整Prompt内容:")
        logger.debug("=" * 80)
        logger.debug(prompt)
        logger.debug("=" * 80)

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
        }
        
        # 只有当 max_tokens 配置了值时才添加到参数中
        if self.config.openai.max_tokens is not None:
            api_params["max_tokens"] = self.config.openai.max_tokens
        
        # 记录API调用参数
        logger.info(f"调用API - 模型: {api_params['model']}, temperature: {api_params['temperature']}, "
                   f"max_tokens: {api_params.get('max_tokens', 'None')}")
        logger.debug(f"API参数: {json.dumps({k: v for k, v in api_params.items() if k != 'messages'}, ensure_ascii=False)}")
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 尝试添加response_format，如果API不支持会自动失败并回退
        try:
            api_params["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**api_params)
        except Exception as e:
            # 如果response_format不支持，尝试不使用它
            if "response_format" in str(e).lower() or "response" in str(e).lower():
                logger.warning("API可能不支持response_format参数，尝试不使用该参数...")
                api_params.pop("response_format", None)
                try:
                    response = self.client.chat.completions.create(**api_params)
                except Exception as e2:
                    logger.error(f"API调用失败: {e2}")
                    logger.debug(f"API调用失败详情:", exc_info=True)
                    raise ValueError(f"API调用失败: {e2}")
            else:
                logger.error(f"API调用失败: {e}")
                logger.debug(f"API调用失败详情:", exc_info=True)
                raise ValueError(f"API调用失败: {e}")
        
        # 记录响应时间
        elapsed_time = time.time() - start_time
        logger.info(f"API调用完成，耗时: {elapsed_time:.2f}秒")

        # 检查响应
        if not response or not response.choices:
            logger.error(f"API返回无效响应")
            logger.debug(f"API返回无效响应详情: {response}")
            raise ValueError(f"API返回无效响应: {response}")

        # 记录token使用情况
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            logger.info(f"Token使用 - prompt_tokens: {usage.prompt_tokens}, "
                       f"completion_tokens: {usage.completion_tokens}, "
                       f"total_tokens: {usage.total_tokens}")

        # 解析响应
        result_text = response.choices[0].message.content
        if not result_text:
            # 尝试获取更多信息
            error_msg = f"API返回结果为空"
            logger.error(error_msg)
            logger.debug(f"API返回结果为空详情。响应对象: {response}")
            if hasattr(response, "error"):
                logger.debug(f"错误信息: {response.error}")
            raise ValueError(error_msg)

        # 记录原始响应
        logger.debug(f"原始响应长度: {len(result_text)} 字符")
        # 记录完整原始响应内容
        logger.debug("=" * 80)
        logger.debug("完整原始响应内容:")
        logger.debug("=" * 80)
        logger.debug(result_text)
        logger.debug("=" * 80)

        # 清理markdown代码块标记
        cleaned_text = self._extract_json_from_markdown(result_text)
        logger.debug(f"清理后文本长度: {len(cleaned_text)} 字符 (减少了 {len(result_text) - len(cleaned_text)} 字符)")
        # 记录完整清理后文本内容
        logger.debug("=" * 80)
        logger.debug("完整清理后文本内容:")
        logger.debug("=" * 80)
        logger.debug(cleaned_text)
        logger.debug("=" * 80)

        try:
            result_json = json.loads(cleaned_text)
            evaluations_data = result_json.get("evaluations", [])
            logger.info(f"成功解析JSON，包含 {len(evaluations_data)} 个评估结果")
            logger.debug(f"评估结果要点ID: {[e.get('point_id', '') for e in evaluations_data[:10]]}")

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

            evaluation = DocumentEvaluation(
                target_document=str(target_document_path),
                points=points,
                evaluations=evaluations,
            )
            logger.info(f"评估完成 - 完整性: {evaluation.completeness:.2f}, "
                       f"准确性: {evaluation.accuracy:.2f}, "
                       f"综合: {evaluation.comprehensive:.2f}")
            return evaluation
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # 提供更详细的错误信息
            error_preview = result_text[:500] if len(result_text) > 500 else result_text
            cleaned_preview = cleaned_text[:500] if len(cleaned_text) > 500 else cleaned_text
            logger.error(f"解析评估结果失败: {e}")
            logger.debug(f"解析评估结果失败详情:", exc_info=True)
            logger.debug(f"原始响应（前500字符）: {error_preview}")
            logger.debug(f"清理后文本（前500字符）: {cleaned_preview}")
            logger.debug(f"原始响应完整长度: {len(result_text)} 字符")
            logger.debug(f"清理后文本完整长度: {len(cleaned_text)} 字符")
            # 异常消息只包含简短信息，详细内容已在日志文件中记录
            raise ValueError(f"解析评估结果失败: {e} (详细内容请查看日志文件)")

    def evaluate_multiple_runs(
        self,
        points: list[dict[str, Any]],
        target_document_path: str | Path,
        runs: int = 3,
        merge_strategy: MergeStrategy | str = MergeStrategy.AVERAGE,
    ) -> DocumentEvaluation:
        """
        多次运行评估并合并结果（并行执行）
        
        每个评委独立评估，各自得到一个准确性分数，然后使用指定策略合并这些分数。

        Args:
            points: 要点清单
            target_document_path: 待评估文档路径
            runs: 运行次数（评委数量）
            merge_strategy: 合并策略，可选值：
                - AVERAGE: 平均值（默认）
                - MAJORITY: 多数投票
                - CONSENSUS_67: 2/3一致性
                - CONSENSUS_75: 3/4一致性
                - MEDIAN: 中位数
                - WEIGHTED: 加权投票
                - CONFIDENCE: 置信区间法

        Returns:
            合并后的评估结果
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
                    logger.info(f"  [{completed}/{runs}] 评委评估完成")
                except Exception as e:
                    logger.error(f"  [{completed}/{runs}] 评委评估失败: {e}")
                    logger.debug(f"评委评估失败详情:", exc_info=True)
                    continue

        # 如果只有一个结果，直接返回
        if len(results) == 1:
            return results[0]
        
        if not results:
            raise ValueError("没有成功的评估结果")

        # 收集每个评委的准确性、完整性和综合分数
        accuracy_scores = [r.accuracy for r in results]
        completeness_scores = [r.completeness for r in results]
        comprehensive_scores = [r.comprehensive for r in results]

        # 使用指定策略合并准确性分数
        merged_accuracy = self._merge_accuracy_by_strategy(accuracy_scores, merge_strategy)
        
        # 完整性和综合分数使用平均值（或可以使用相同策略）
        merged_completeness = statistics.mean(completeness_scores)
        merged_comprehensive = statistics.mean(comprehensive_scores)

        # 使用第一个评委的详细评估结果（evaluations字段）
        # 这样可以保留要点的详细评估信息
        first_result = results[0]

        # 创建最终结果对象，手动设置合并后的分数
        final_result = DocumentEvaluation(
            target_document=str(target_document_path),
            points=points,
            evaluations=first_result.evaluations,
        )
        
        # 覆盖计算出的分数，使用合并后的分数
        final_result.accuracy = merged_accuracy
        final_result.completeness = merged_completeness
        final_result.comprehensive = merged_comprehensive

        return final_result

    def merge_evaluation_results(
        self,
        raw_results: list[DocumentEvaluation],
        points: list[dict[str, Any]],
        target_document_path: str | Path,
        merge_strategy: MergeStrategy | str = MergeStrategy.AVERAGE,
    ) -> DocumentEvaluation:
        """
        对原始评估结果应用合并策略
        
        每个评委独立评估，各自得到一个准确性分数，然后使用指定策略合并这些分数。
        
        Args:
            raw_results: 原始评估结果列表（来自多次 evaluate_single_run）
            points: 要点清单
            target_document_path: 待评估文档路径
            merge_strategy: 合并策略
        
        Returns:
            合并后的评估结果
        """
        runs = len(raw_results)
        if runs == 0:
            raise ValueError("原始结果列表不能为空")
        
        # 如果只有一个结果，直接返回
        if runs == 1:
            return raw_results[0]

        # 收集每个评委的准确性、完整性和综合分数
        accuracy_scores = [r.accuracy for r in raw_results]
        completeness_scores = [r.completeness for r in raw_results]
        comprehensive_scores = [r.comprehensive for r in raw_results]

        # 使用指定策略合并准确性分数
        merged_accuracy = self._merge_accuracy_by_strategy(accuracy_scores, merge_strategy)
        
        # 完整性和综合分数使用平均值（或可以使用相同策略）
        merged_completeness = statistics.mean(completeness_scores)
        merged_comprehensive = statistics.mean(comprehensive_scores)

        # 使用第一个评委的详细评估结果（evaluations字段）
        # 这样可以保留要点的详细评估信息
        first_result = raw_results[0]

        # 创建最终结果对象，手动设置合并后的分数
        final_result = DocumentEvaluation(
            target_document=str(target_document_path),
            points=points,
            evaluations=first_result.evaluations,
        )
        
        # 覆盖计算出的分数，使用合并后的分数
        final_result.accuracy = merged_accuracy
        final_result.completeness = merged_completeness
        final_result.comprehensive = merged_comprehensive

        return final_result

    def merge_evaluation_results_by_checkpoints(
        self,
        raw_results: list[DocumentEvaluation],
        points: list[dict[str, Any]],
        target_document_path: str | Path,
        merge_strategy: MergeStrategy | str = MergeStrategy.AVERAGE,
    ) -> DocumentEvaluation:
        """
        对原始评估结果应用检查项合并策略
        
        合并每个检查项的投票结果，然后重新计算准确性分数。
        
        Args:
            raw_results: 原始评估结果列表（来自多次 evaluate_single_run）
            points: 要点清单
            target_document_path: 待评估文档路径
            merge_strategy: 合并策略
        
        Returns:
            合并后的评估结果
        """
        runs = len(raw_results)
        if runs == 0:
            raise ValueError("原始结果列表不能为空")
        
        # 如果只有一个结果，直接返回
        if runs == 1:
            return raw_results[0]

        # 合并评估结果（基于检查项结果）
        point_evaluations = {}
        for result in raw_results:
            for eval_result in result.evaluations:
                point_id = eval_result.point_id
                if point_id not in point_evaluations:
                    point_evaluations[point_id] = {
                        "exists_count": 0,
                        "checkpoint_votes": {},  # checkpoint -> {passed_count, total_count, pass_rates}
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
                                "pass_rates": [],  # 保存每个评委的 pass_rate
                            }
                        point_evaluations[point_id]["checkpoint_votes"][cp_text]["total_count"] += 1
                        if cp_result.passed:
                            point_evaluations[point_id]["checkpoint_votes"][cp_text]["passed_count"] += 1
                        # 保存每个评委的 pass_rate（用于中位数计算）
                        point_evaluations[point_id]["checkpoint_votes"][cp_text]["pass_rates"].append(
                            cp_result.pass_rate
                        )
                
                point_evaluations[point_id]["explanations"].append(
                    eval_result.explanation
                )
        
        # 创建合并后的评估结果
        avg_evaluations = []
        for point_id, data in point_evaluations.items():
            exists = data["exists_count"] > (runs / 2)  # 多数存在则认为存在
            
            # 合并检查项结果：根据策略选择不同算法
            avg_checkpoint_results = []
            for cp_text, votes in data["checkpoint_votes"].items():
                pass_rate, passed = self._merge_checkpoint_by_strategy(
                    votes, runs, exists, merge_strategy
                )
                avg_checkpoint_results.append(
                    CheckpointResult(checkpoint=cp_text, passed=passed, pass_rate=pass_rate)
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
        final_result = DocumentEvaluation(
            target_document=str(target_document_path),
            points=points,
            evaluations=avg_evaluations,
        )
        
        return final_result

    def _merge_accuracy_by_strategy(
        self,
        accuracy_scores: list[float],
        strategy: MergeStrategy | str,
    ) -> float:
        """
        根据策略合并准确性分数列表
        
        支持的策略：
        - AVERAGE: 平均值，适用于评委水平相近、分布正常时
        - MEDIAN: 中位数，适用于存在异常值或评委差异较大时，抗异常值能力强
        - MAJORITY: 多数投票，返回平均值（与AVERAGE相同）
        - CONSENSUS_67: 2/3一致性，如果2/3以上评委给出>50分则返回平均值，否则返回较低值
        - CONSENSUS_75: 3/4一致性，如果3/4以上评委给出>50分则返回平均值，否则返回较低值
        - WEIGHTED: 加权投票，根据标准差自动调整权重，一致性越高权重越大
        - CONFIDENCE: 置信区间法，标准差小时使用平均值，大时使用中位数（更保守）
        - TRIMMED_MEAN: 截断均值，去除最高和最低10%分数后取平均，抗异常值但比中位数更充分利用信息
        - QUANTILE_WEIGHTED: 分位数加权，中间分位数（接近0.5）的分数权重更大，认为中间位置评分更可靠
        - BAYESIAN_AVERAGE: 贝叶斯平均，结合先验信息（默认50分），样本量小时更依赖先验，样本量大时更依赖数据
        
        Args:
            accuracy_scores: 准确性分数列表（0-100之间的值）
            strategy: 合并策略
        
        Returns:
            合并后的准确性分数（0-100）
        """
        if not accuracy_scores:
            return 0.0
        
        # 如果只有一个分数，直接返回
        if len(accuracy_scores) == 1:
            return accuracy_scores[0]
        
        # 处理字符串类型的策略
        if isinstance(strategy, str):
            try:
                strategy = MergeStrategy(strategy)
            except ValueError:
                strategy = MergeStrategy.AVERAGE
        
        if strategy == MergeStrategy.AVERAGE:
            # 平均值
            return statistics.mean(accuracy_scores)
        
        elif strategy == MergeStrategy.MEDIAN:
            # 中位数
            return statistics.median(accuracy_scores)
        
        elif strategy == MergeStrategy.MAJORITY:
            # 多数投票：>50%的评委给出>50分则认为通过，返回平均值
            # 这里我们计算平均值，但逻辑上可以理解为多数评委的平均
            return statistics.mean(accuracy_scores)
        
        elif strategy == MergeStrategy.CONSENSUS_67:
            # 2/3一致性：如果2/3以上的评委给出>50分，返回平均值
            threshold = 2.0 / 3.0
            passing_count = sum(1 for score in accuracy_scores if score > 50.0)
            if passing_count / len(accuracy_scores) >= threshold:
                return statistics.mean(accuracy_scores)
            else:
                # 未达到一致性，返回较低的值（平均值和最低值的加权）
                return statistics.mean(accuracy_scores) * 0.7 + min(accuracy_scores) * 0.3
        
        elif strategy == MergeStrategy.CONSENSUS_75:
            # 3/4一致性：如果3/4以上的评委给出>50分，返回平均值
            threshold = 0.75
            passing_count = sum(1 for score in accuracy_scores if score > 50.0)
            if passing_count / len(accuracy_scores) >= threshold:
                return statistics.mean(accuracy_scores)
            else:
                # 未达到一致性，返回较低的值
                return statistics.mean(accuracy_scores) * 0.7 + min(accuracy_scores) * 0.3
        
        elif strategy == MergeStrategy.WEIGHTED:
            # 加权投票：一致性越高权重越大
            # 计算标准差，标准差越小权重越大
            if len(accuracy_scores) > 1:
                std_dev = statistics.stdev(accuracy_scores)
                mean_score = statistics.mean(accuracy_scores)
                # 归一化标准差（假设最大标准差为50）
                normalized_std = min(std_dev / 50.0, 1.0)
                weight = 1.0 - normalized_std * 0.5  # 权重在0.5-1.0之间
                return mean_score * weight + statistics.median(accuracy_scores) * (1 - weight)
            else:
                return accuracy_scores[0]
        
        elif strategy == MergeStrategy.CONFIDENCE:
            # 置信区间法：只使用高置信度结果（标准差小的结果）
            if len(accuracy_scores) > 1:
                std_dev = statistics.stdev(accuracy_scores)
                mean_score = statistics.mean(accuracy_scores)
                # 如果标准差小（<10），使用平均值；否则保守处理
                if std_dev < 10.0:
                    return mean_score
                else:
                    # 低置信度时使用中位数（更保守）
                    return statistics.median(accuracy_scores)
            else:
                return accuracy_scores[0]
        
        elif strategy == MergeStrategy.TRIMMED_MEAN:
            # 截断均值：去除最高和最低的10%分数（至少1个，最多去除总数的一半）后取平均
            if len(accuracy_scores) <= 2:
                # 如果只有2个或更少，直接返回平均值
                return statistics.mean(accuracy_scores)
            
            sorted_scores = sorted(accuracy_scores)
            n = len(sorted_scores)
            # 计算需要去除的数量：10%，至少1个，最多n//2
            trim_count = max(1, min(int(n * 0.1), n // 2))
            
            # 去除最高和最低的 trim_count 个分数
            trimmed_scores = sorted_scores[trim_count:-trim_count] if trim_count > 0 else sorted_scores
            
            if trimmed_scores:
                return statistics.mean(trimmed_scores)
            else:
                # 如果去除后没有剩余分数，返回中位数
                return statistics.median(accuracy_scores)
        
        elif strategy == MergeStrategy.QUANTILE_WEIGHTED:
            # 分位数加权：根据分数在分布中的位置分配权重，中间分位数权重更大
            if len(accuracy_scores) <= 1:
                return accuracy_scores[0] if accuracy_scores else 0.0
            
            sorted_scores = sorted(accuracy_scores)
            n = len(sorted_scores)
            weighted_sum = 0.0
            total_weight = 0.0
            
            for i, score in enumerate(sorted_scores):
                # 计算分位数位置（0到1之间）
                quantile = (i + 0.5) / n
                # 权重函数：中间分位数（接近0.5）权重最大
                # 使用二次函数，使权重分布更平滑，峰值在0.5
                # weight = 1.0 - (quantile - 0.5) ** 2 * 4
                # 这样中间位置（0.5）权重为1.0，两端（0和1）权重为0
                weight = 1.0 - (quantile - 0.5) ** 2 * 4
                weight = max(0.0, weight)  # 确保权重非负
                weighted_sum += score * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                return statistics.mean(accuracy_scores)
        
        elif strategy == MergeStrategy.BAYESIAN_AVERAGE:
            # 贝叶斯平均：使用先验均值（默认50分）和样本均值加权
            # 样本量小时更依赖先验，样本量大时更依赖数据
            prior_mean = 50.0  # 先验均值（假设平均分数为50）
            prior_weight = 1.0  # 先验权重（相当于1个样本）
            
            sample_mean = statistics.mean(accuracy_scores)
            sample_size = len(accuracy_scores)
            
            # 贝叶斯平均 = (先验均值 * 先验权重 + 样本均值 * 样本量) / (先验权重 + 样本量)
            bayesian_mean = (prior_mean * prior_weight + sample_mean * sample_size) / (prior_weight + sample_size)
            
            return bayesian_mean
        
        else:
            # 默认使用平均值
            return statistics.mean(accuracy_scores)

    def _merge_checkpoint_by_strategy(
        self,
        votes: dict,
        runs: int,
        exists: bool,
        strategy: MergeStrategy | str,
    ) -> tuple[float, bool]:
        """
        根据策略合并检查项结果
        
        Args:
            votes: 投票统计 {"passed_count": int, "total_count": int, "pass_rates": list[float]}
            runs: 运行次数
            exists: 要点是否存在
            strategy: 合并策略
        
        Returns:
            (pass_rate, passed) 元组
        """
        if not exists:
            return 0.0, False
        
        passed_count = votes["passed_count"]
        total_count = votes["total_count"]
        pass_ratio = passed_count / total_count if total_count > 0 else 0.0
        
        # 处理字符串类型的策略
        if isinstance(strategy, str):
            try:
                strategy = MergeStrategy(strategy)
            except ValueError:
                strategy = MergeStrategy.AVERAGE
        
        if strategy == MergeStrategy.AVERAGE:
            # 当前方法：使用通过率
            passed = pass_ratio >= 0.5
            return pass_ratio, passed
        
        elif strategy == MergeStrategy.MAJORITY:
            # 多数投票：>50%
            passed = pass_ratio > 0.5
            return pass_ratio, passed
        
        elif strategy == MergeStrategy.CONSENSUS_67:
            # 2/3一致性
            threshold = 2.0 / 3.0
            passed = pass_ratio >= threshold
            return pass_ratio, passed
        
        elif strategy == MergeStrategy.CONSENSUS_75:
            # 3/4一致性
            threshold = 0.75
            passed = pass_ratio >= threshold
            return pass_ratio, passed
        
        elif strategy == MergeStrategy.MEDIAN:
            # 中位数：计算所有评委 pass_rate 的中位数
            pass_rates = votes.get("pass_rates", [])
            if pass_rates:
                median_rate = statistics.median(pass_rates)
                passed = median_rate >= 0.5
                return median_rate, passed
            else:
                # 如果没有 pass_rates，回退到使用 pass_ratio
                passed = pass_ratio >= 0.5
                return pass_ratio, passed
        
        elif strategy == MergeStrategy.WEIGHTED:
            # 加权投票：一致性越高权重越大
            weight = pass_ratio ** 1.5
            weighted_rate = pass_ratio * weight
            passed = weighted_rate >= 0.5
            return pass_ratio, passed
        
        elif strategy == MergeStrategy.CONFIDENCE:
            # 置信区间法：只使用高置信度结果
            confidence = abs(pass_ratio - 0.5) * 2  # 0.0-1.0
            if confidence > 0.3:  # 置信度>30%
                passed = pass_ratio > 0.5
            else:
                # 低置信度时保守处理
                passed = False
            return pass_ratio, passed
        
        else:
            # 默认使用平均值
            passed = pass_ratio >= 0.5
            return pass_ratio, passed

