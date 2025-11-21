"""评估模块"""

import copy
import csv
import io
import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from openai import OpenAI, APIError, APITimeoutError, InternalServerError

from src.config import Config, load_config
from src.document_parser import DocumentParser
from src.utils.token_counter import calculate_adjusted_max_tokens, count_tokens

# 配置日志
logger = logging.getLogger(__name__)


class MergeStrategy(str, Enum):
    """合并策略枚举"""
    CHECKPOINT_MAJORITY = "checkpoint_majority"  # 检查项合并：每个检查项按多数投票判定，再统计通过项占比
    ACCURACY_AVERAGE = "accuracy_average"  # 准确性合并：对每个评委的通过项数量进行平均计算


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




class DocumentEvaluation:
    """文档评估结果"""

    def __init__(
        self,
        target_document: str,
        checkpoints: list[str],
        checkpoint_results: list[CheckpointResult],
        all_judge_results: list[list[CheckpointResult]] | None = None,
        checkpoint_merge_result: "DocumentEvaluation | None" = None,
        accuracy_merge_result: "DocumentEvaluation | None" = None,
        model_name: str | None = None,
        baseline_document: str | None = None,
    ):
        self.target_document = target_document
        self.checkpoints = checkpoints
        self.checkpoint_results = checkpoint_results
        # 存储所有评委的检查项结果，用于显示每个评委的判断
        # all_judge_results[i] 表示第 i 个评委对所有检查项的评估结果
        self.all_judge_results = all_judge_results
        # 存储两种策略的合并结果
        self.checkpoint_merge_result = checkpoint_merge_result  # 检查项合并策略的结果
        self.accuracy_merge_result = accuracy_merge_result  # 准确性合并策略的结果
        # 评估元信息
        self.model_name = model_name  # 使用的模型名称
        self.baseline_document = baseline_document  # 基准文档路径
        self.evaluation_time: str | None = None  # 评估时间
        self.evaluation_duration: float | None = None  # 评估耗时（秒）

        # 计算分数
        total_checkpoints = len(checkpoints)
        if total_checkpoints == 0:
            self.completeness = 0.0
            self.accuracy = 0.0
            self.comprehensive = 0.0
        else:
            # 准确性：基于检查项通过情况计算
            passed_checkpoints = sum(1 for cp in checkpoint_results if cp.passed)
            self.accuracy = (passed_checkpoints / total_checkpoints) * 100 if total_checkpoints > 0 else 0.0

            # 完整性：与准确性相同（因为不再有points层级）
            self.completeness = self.accuracy

            # 综合分数：完整性 × 0.5 + 准确性 × 0.5（实际上两者相同）
            self.comprehensive = self.accuracy


class Evaluator:
    """文档评估器"""

    def __init__(self, config: Config | None = None, prompt_version: str | None = None):
        """
        初始化评估器

        Args:
            config: 配置对象，如果为None则自动加载
            prompt_version: 提示词版本，如果为None则使用配置中的版本
        """
        self.config = config or load_config()
        self.prompt_version = prompt_version or self.config.prompt_version
        self.client = OpenAI(
            api_key=self.config.openai.api_key,
            base_url=self.config.openai.base_url,
            timeout=self.config.openai.timeout,
        )
        self.parser = DocumentParser()

    @staticmethod
    def _extract_text_from_markdown(text: str) -> str:
        """
        从markdown代码块中提取文本内容
        
        处理以下情况：
        - ```tsv ... ```
        - ```csv ... ``` (兼容旧格式)
        - ``` ... ```
        - 纯文本（无代码块标记）
        
        Args:
            text: 可能包含markdown代码块的文本
            
        Returns:
            清理后的文本
        """
        # 处理 None 或空值的情况
        if text is None:
            return ""
        
        text = text.strip()
        
        # 尝试匹配 ```tsv ... ``` 或 ```csv ... ``` 或 ``` ... ```
        if text.startswith("```"):
            # 找到第一个换行符（代码块开始标记之后）
            first_newline = text.find("\n")
            if first_newline != -1:
                # 移除开始标记（```tsv 或 ```csv 或 ```）
                text = text[first_newline + 1:]
            
            # 找到最后一个 ```（代码块结束标记）
            last_code_block = text.rfind("```")
            if last_code_block != -1:
                # 移除结束标记
                text = text[:last_code_block]
        
        # 清理后的文本可能包含代码块前的说明文字
        # 查找TSV表头（支持两种格式：
        # 1. Index\tPass（新格式，两列）
        # 2. Index\tReason\tResult 或 索引编号\t判定理由\t判定结果（旧格式，三列））
        # 只保留从表头开始的内容
        lines = text.split('\n')
        header_found = False
        header_index = -1
        
        # 查找表头行
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            # 检查是否是TSV表头（新格式：Index\tPass）
            if (line_lower.startswith('index') and '\t' in line and 
                'pass' in line_lower and line_lower.count('\t') == 1):
                header_found = True
                header_index = i
                break
            # 检查是否是TSV表头（旧格式：Index\tReason\tResult）
            elif (line_lower.startswith('index') and '\t' in line and 
                ('reason' in line_lower or 'result' in line_lower)):
                header_found = True
                header_index = i
                break
            # 兼容中文表头
            elif (('索引编号' in line or '索引' in line) and '\t' in line and 
                  ('判定理由' in line or '判定结果' in line)):
                header_found = True
                header_index = i
                break
        
        # 如果找到表头，只保留从表头开始的内容
        if header_found and header_index >= 0:
            text = '\n'.join(lines[header_index:])
        
        return text.strip()
    
    @staticmethod
    def _format_checkpoints_as_csv(checkpoints: list[str]) -> str:
        """
        将检查项列表格式化为TSV表格
        
        Args:
            checkpoints: 检查项列表
            
        Returns:
            TSV格式的字符串，包含表头：Index	Checkpoint（使用制表符分隔）
        """
        output = io.StringIO()
        writer = csv.writer(output, delimiter='\t')
        
        # 写入表头
        writer.writerow(["Index", "Checkpoint"])
        
        # 写入数据行
        for index, checkpoint in enumerate(checkpoints):
            writer.writerow([index, checkpoint])
        
        return output.getvalue()
    
    @staticmethod
    def _validate_parse_result(
        checkpoint_results: list[CheckpointResult], 
        expected_count: int,
        tsv_text: str = "",
        min_valid_ratio: float = 0.3
    ) -> tuple[bool, str]:
        """
        验证解析结果是否有效
        
        Args:
            checkpoint_results: 解析得到的检查项结果列表
            expected_count: 期望的检查项数量
            tsv_text: 原始TSV文本（用于检查是否有数据行）
            min_valid_ratio: 最小有效比例（默认0.3，即至少30%的检查项需要被正确解析）
            
        Returns:
            (is_valid, error_message) 元组，is_valid为True表示解析结果有效，False表示无效
        """
        if len(checkpoint_results) != expected_count:
            return False, f"检查项数量不匹配: 期望 {expected_count} 个，实际得到 {len(checkpoint_results)} 个"
        
        # 检查TSV文本中是否有数据行（除了表头）
        if tsv_text:
            lines = tsv_text.strip().split('\n')
            # 至少应该有表头 + 1行数据
            if len(lines) < 2:
                return False, "TSV文本中没有数据行（只有表头）"
            
            # 检查是否有至少一行包含数字索引的数据行
            has_data_row = False
            for line in lines[1:]:  # 跳过表头
                line = line.strip()
                if not line:
                    continue
                # 检查是否以数字开头（可能是索引）
                parts = line.split('\t')
                if parts and parts[0].strip().isdigit():
                    has_data_row = True
                    break
            
            if not has_data_row:
                return False, "TSV文本中没有有效的数字索引行"
        
        # 检查是否有足够的检查项被正确解析
        # 这里我们检查是否有至少min_valid_ratio比例的检查项有明确的判定结果
        # 由于我们改进了解析逻辑，现在即使Result列为空也会默认判定为False
        # 所以只要数量匹配且有数据行就认为解析有效
        return True, ""
    
    @staticmethod
    def _parse_tsv_result(tsv_text: str, checkpoints: list[str]) -> list[CheckpointResult]:
        """
        解析TSV格式的评估结果

        Args:
            tsv_text: TSV格式的文本
            checkpoints: 检查项列表（用于验证索引）

        Returns:
            检查项结果列表
        """
        # 处理 None 或空值的情况
        if tsv_text is None:
            tsv_text = ""
        
        # 使用csv模块解析TSV（指定delimiter为制表符）
        reader = csv.DictReader(io.StringIO(tsv_text), delimiter='\t')

        # 构建索引到结果的映射
        index_to_result = {}
        missing_result_count = 0
        for row in reader:
            try:
                # 跳过不符合格式的行（索引列不是数字或为空）
                index_str = row.get("Index", row.get("索引编号", ""))
                if not index_str or not index_str.strip():
                    # 可能是说明文字行，跳过
                    continue
                
                try:
                    index = int(index_str)
                except ValueError:
                    # 索引列不是数字，可能是说明文字行，跳过
                    logger.debug(f"跳过非数据行（索引列不是数字）: {row}")
                    continue
                
                # 优先尝试从 Pass 列读取（新格式：Index, Pass）
                result_str_raw = row.get("Pass", "")
                result_str = (result_str_raw or "").strip().lower()
                
                # 如果没有 Pass 列，尝试从 Result 列读取（兼容旧格式：Index, Reason, Result）
                if not result_str or result_str in ["Pass", "pass"]:
                    result_str_raw = row.get("Result", row.get("判定结果", ""))
                    result_str = (result_str_raw or "").strip().lower()
                
                # 如果判定结果列为空，尝试从判定理由列末尾提取（兼容旧格式）
                if not result_str or result_str in ["Reason", "判定理由", "Pass", "pass"]:
                    reason_str = row.get("Reason", row.get("判定理由", ""))
                    if reason_str:
                        # 尝试从判定理由末尾提取 yes/no
                        reason_lower = reason_str.strip().lower()
                        # 检查是否以 yes/no 结尾（可能用制表符分隔）
                        if reason_lower.endswith("\tyes") or reason_lower.endswith(" yes"):
                            result_str = "yes"
                        elif reason_lower.endswith("\tno") or reason_lower.endswith(" no"):
                            result_str = "no"
                        elif reason_lower.endswith("yes"):
                            result_str = "yes"
                        elif reason_lower.endswith("no"):
                            result_str = "no"
                    
                    # 如果仍然没有找到，记录警告
                    if not result_str or result_str in ["Reason", "判定理由", "Pass", "pass"]:
                        missing_result_count += 1
                        if missing_result_count <= 3:  # 只记录前3个，避免日志过多
                            logger.warning(
                                f"检查项 {index} 的判定结果列缺失或格式错误: "
                                f"期望 'yes' 或 'no'，实际得到 '{row.get('Pass', row.get('Result', row.get('判定结果', '')))}'。"
                                f"整行内容: {row}"
                            )
                        # 如果判定结果列缺失或格式错误，默认判定为未通过
                        passed = False
                    else:
                        # 成功从判定理由中提取，记录调试信息
                        logger.debug(f"检查项 {index} 从判定理由中提取到判定结果: {result_str}")
                        passed = result_str in ["y", "yes", "true", "1", "是"]
                else:
                    passed = result_str in ["y", "yes", "true", "1", "是"]

                # 验证索引有效性
                if 0 <= index < len(checkpoints):
                    index_to_result[index] = passed
            except (ValueError, KeyError) as e:
                logger.warning(f"解析TSV行失败: {row}, 错误: {e}")
                continue
        
        # 如果发现大量缺失的判定结果，给出警告
        if missing_result_count > 0:
            logger.warning(
                f"警告: 发现 {missing_result_count} 个检查项的判定结果列缺失或格式错误。"
                f"这些检查项将被默认判定为未通过。"
                f"请检查API返回的TSV格式是否正确，应包含两列：Index	Pass (或三列：Index	Reason	Result，使用制表符分隔)"
            )

        # 构建检查项结果列表
        checkpoint_results = []
        for index, checkpoint in enumerate(checkpoints):
            passed = index_to_result.get(index, False)
            checkpoint_results.append(
                CheckpointResult(checkpoint=checkpoint, passed=passed)
            )

        return checkpoint_results

    def _load_prompt_template(self, template_name: str) -> str:
        """
        加载prompt模板

        Args:
            template_name: 模板文件名

        Returns:
            模板内容
        """
        # 构建文件路径：prompts/{version}/{template_name}
        template_path = (
            Path(__file__).parent.parent / "prompts" / self.prompt_version / template_name
        )
        if not template_path.exists():
            # 如果指定版本不存在，尝试使用v1作为默认版本
            if self.prompt_version != "v1":
                logger.warning(
                    f"提示词文件不存在: {template_path}，尝试使用 v1 版本"
                )
                template_path = (
                    Path(__file__).parent.parent / "prompts" / "v1" / template_name
                )
        
        if not template_path.exists():
            raise FileNotFoundError(
                f"模板文件不存在: {template_path}，请检查提示词版本和文件路径"
            )
        return template_path.read_text(encoding="utf-8")

    def _call_api_with_retry(self, api_params: dict[str, Any]) -> Any:
        """
        带重试机制的API调用方法
        
        Args:
            api_params: API调用参数
            
        Returns:
            API响应对象（如果是流式响应，则返回收集后的完整响应）
            
        Raises:
            ValueError: 所有重试都失败后抛出异常
        """
        max_retries = self.config.openai.max_retries
        retry_delay = self.config.openai.retry_delay
        last_exception = None
        stream = api_params.get("stream", False)
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # 指数退避：延迟时间 = 初始延迟 * 2^(attempt-1)
                    delay = retry_delay * (2 ** (attempt - 1))
                    logger.warning(f"第 {attempt + 1}/{max_retries + 1} 次尝试，等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                
                response = self.client.chat.completions.create(**api_params)
                
                # 如果是流式响应，收集所有数据块
                if stream:
                    from openai.types.chat import ChatCompletion
                    from openai.types.chat.chat_completion import Choice
                    from openai.types.chat.chat_completion_message import ChatCompletionMessage
                    
                    content_parts = []
                    usage_data = None
                    finish_reason = None
                    role = None
                    chunk_id = ""
                    chunk_model = api_params.get("model", "")
                    chunk_created = int(time.time())
                    
                    for chunk in response:
                        if hasattr(chunk, 'id') and chunk.id:
                            chunk_id = chunk.id
                        if hasattr(chunk, 'model') and chunk.model:
                            chunk_model = chunk.model
                        if hasattr(chunk, 'created') and chunk.created:
                            chunk_created = chunk.created
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                content_parts.append(delta.content)
                            if delta.role:
                                role = delta.role
                            if chunk.choices[0].finish_reason:
                                finish_reason = chunk.choices[0].finish_reason
                        if hasattr(chunk, 'usage') and chunk.usage:
                            usage_data = chunk.usage
                    
                    # 构建完整的响应对象
                    full_content = "".join(content_parts)
                    message = ChatCompletionMessage(
                        role=role or "assistant",
                        content=full_content
                    )
                    choice = Choice(
                        index=0,
                        message=message,
                        finish_reason=finish_reason or "stop"
                    )
                    response = ChatCompletion(
                        id=chunk_id,
                        model=chunk_model,
                        choices=[choice],
                        created=chunk_created,
                        object="chat.completion",
                        usage=usage_data
                    )
                
                if attempt > 0:
                    logger.info(f"重试成功（第 {attempt + 1} 次尝试）")
                return response
                
            except (InternalServerError, APITimeoutError) as e:
                # 对于500错误和超时错误，进行重试
                last_exception = e
                error_type = "服务器内部错误" if isinstance(e, InternalServerError) else "请求超时"
                logger.warning(f"API调用失败（{error_type}）: {e}")
                if attempt < max_retries:
                    continue
                else:
                    logger.error(f"所有 {max_retries + 1} 次尝试均失败")
                    raise ValueError(f"API调用失败（{error_type}）: {e}")
                    
            except APIError as e:
                # 对于其他API错误，根据状态码决定是否重试
                last_exception = e
                status_code = getattr(e, 'status_code', None)
                # 5xx错误可以重试，4xx错误（客户端错误）不重试
                if status_code and 500 <= status_code < 600:
                    logger.warning(f"API调用失败（服务器错误 {status_code}）: {e}")
                    if attempt < max_retries:
                        continue
                    else:
                        logger.error(f"所有 {max_retries + 1} 次尝试均失败")
                        raise ValueError(f"API调用失败（服务器错误 {status_code}）: {e}")
                else:
                    # 客户端错误（4xx）不重试，直接抛出
                    logger.error(f"API调用失败（客户端错误）: {e}")
                    raise ValueError(f"API调用失败: {e}")
                    
            except Exception as e:
                # 其他异常（网络错误等）也进行重试
                last_exception = e
                logger.warning(f"API调用失败（未知错误）: {e}")
                if attempt < max_retries:
                    continue
                else:
                    logger.error(f"所有 {max_retries + 1} 次尝试均失败")
                    raise ValueError(f"API调用失败: {e}")
        
        # 如果所有重试都失败，抛出最后一个异常
        if last_exception:
            raise ValueError(f"API调用失败: {last_exception}")

    def _continue_on_truncation(
        self,
        api_params: dict[str, Any],
        accumulated_text: str,
        finish_reason: str | None,
        task_name: str = "文档评估",
    ) -> tuple[str, str | None]:
        """
        如果模型因为达到max_tokens而提前结束，自动请求接续
        
        使用对话前缀续写方式：将已生成的内容作为 assistant 消息，设置 prefix: True，
        让模型从该前缀继续生成，避免重复输出表头和上一行。
        
        除了检查finish_reason == "length"外，还会检测TSV格式的响应是否被截断
        （例如最后一行不完整）。
        """
        # 检查是否需要接续：finish_reason为"length"或检测到TSV格式被截断
        needs_continuation = False
        truncation_reason = None
        
        if finish_reason == "length":
            needs_continuation = True
            truncation_reason = "finish_reason为'length'"
        else:
            # 检测TSV格式是否被截断：检查最后一行是否完整
            # 支持两种格式：
            # 1. Index	Pass（新格式，2列）
            # 2. Index	Reason	Result 或 索引编号	判定理由	判定结果（旧格式，3列）
            cleaned_text = self._extract_text_from_markdown(accumulated_text)
            if cleaned_text:
                lines = cleaned_text.strip().split('\n')
                if len(lines) > 1:  # 至少有表头和数据行
                    # 检测表头格式，确定期望的列数
                    header_line = lines[0].strip().lower()
                    expected_columns = 3  # 默认期望3列（旧格式）
                    pass_column_index = 1  # Pass列在第二列（索引1）
                    result_column_index = 2  # Result列在第三列（索引2）
                    
                    # 检查是否是两列格式（Index\tPass）
                    if header_line.startswith('index') and '\t' in header_line:
                        header_fields = header_line.split('\t')
                        if len(header_fields) == 2 and 'pass' in header_line:
                            expected_columns = 2
                            pass_column_index = 1
                    
                    # 检查最后一行是否完整
                    last_line = lines[-1].strip()
                    if last_line:
                        # 尝试解析最后一行（使用制表符分隔）
                        try:
                            reader = csv.reader([last_line], delimiter='\t')
                            fields = next(reader)
                            # 如果最后一行字段数少于期望列数，可能被截断
                            if len(fields) < expected_columns:
                                needs_continuation = True
                                truncation_reason = f"最后一行字段数不足（{len(fields)}/{expected_columns}）"
                            elif len(fields) >= expected_columns:
                                # 检查Pass列或Result列是否有值
                                if expected_columns == 2:
                                    # 两列格式：检查Pass列
                                    pass_value = fields[pass_column_index].strip().lower() if len(fields) > pass_column_index else ""
                                    if not pass_value or pass_value not in ["yes", "no", "y", "n", "是", "否"]:
                                        needs_continuation = True
                                        truncation_reason = f"最后一行Pass列格式异常: '{pass_value}'"
                                else:
                                    # 三列格式：检查Result列
                                    result_value = fields[result_column_index].strip().lower() if len(fields) > result_column_index else ""
                                    if not result_value or result_value not in ["yes", "no", "y", "n", "是", "否"]:
                                        needs_continuation = True
                                        truncation_reason = f"最后一行Result列格式异常: '{result_value}'"
                        except Exception:
                            # 解析失败，可能格式有问题，尝试其他检测方法
                            # 检查是否以引号开始但未结束（可能被截断）
                            if last_line.count('"') % 2 != 0:
                                needs_continuation = True
                                truncation_reason = "最后一行引号未闭合"
                            # 检查是否以逗号结尾（可能是被截断的字段）
                            elif last_line.endswith(','):
                                needs_continuation = True
                                truncation_reason = "最后一行以逗号结尾，可能被截断"
        
        if not needs_continuation:
            return accumulated_text, finish_reason
        
        logger.warning(
            f"{task_name} 检测到响应可能被截断（{truncation_reason}），"
            f"finish_reason: {finish_reason}，尝试自动接续..."
        )
        
        if self.config.openai.max_continuations <= 0:
            logger.warning(f"{task_name} 输出被截断，但未开启接续功能")
            return accumulated_text, finish_reason
        base_messages = api_params.get("messages")
        if not base_messages:
            logger.warning(f"{task_name} 输出被截断，但缺少原始消息上下文，无法接续")
            return accumulated_text, finish_reason

        combined_text = accumulated_text
        attempt = 0
        total_attempts = self.config.openai.max_continuations

        # 继续接续直到不再需要（finish_reason不是"length"且CSV格式完整）
        while attempt < total_attempts:
            attempt += 1
            logger.warning(
                f"{task_name} 响应达到 max_tokens，自动发送第 {attempt}/{total_attempts} 次接续请求..."
            )
            
            # 记录接续前的内容（最后500字符）
            preview_length = 500
            before_text = combined_text[-preview_length:] if len(combined_text) > preview_length else combined_text
            logger.info(f"接续前内容（最后{len(before_text)}字符，总长度{len(combined_text)}字符）:")
            logger.info("=" * 80)
            logger.info(before_text)
            logger.info("=" * 80)
            
            # 使用对话前缀续写：更新最后一个 assistant 消息，而不是追加新消息
            continuation_messages = copy.deepcopy(base_messages)
            # 如果最后一个消息是 assistant 消息，更新它；否则追加新的
            if continuation_messages and continuation_messages[-1].get("role") == "assistant":
                continuation_messages[-1] = {
                    "role": "assistant",
                    "content": combined_text,
                    "prefix": True,
                    "partial": True
                }
            else:
                continuation_messages.append({
                    "role": "assistant",
                    "content": combined_text,
                    "prefix": True,
                    "partial": True
                })

            continuation_params = {
                k: v for k, v in api_params.items() if k != "messages"
            }
            continuation_params["messages"] = continuation_messages

            start_time = time.time()
            response = self._call_api_with_retry(continuation_params)
            elapsed = time.time() - start_time
            logger.info(f"接续请求 #{attempt} 完成，耗时: {elapsed:.2f}秒")

            if not response or not response.choices:
                logger.warning("接续请求返回无效响应，停止继续尝试")
                break

            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                logger.info(
                    f"接续 Token使用 - prompt_tokens: {usage.prompt_tokens}, "
                    f"completion_tokens: {usage.completion_tokens}, "
                    f"total_tokens: {usage.total_tokens}"
                )

            extra_text = response.choices[0].message.content or ""
            if not extra_text:
                logger.warning("接续响应为空，重新发起原始请求")
                # 重新使用原始api_params发起请求，而不是继续接续
                start_time = time.time()
                response = self._call_api_with_retry(api_params)
                elapsed = time.time() - start_time
                logger.info(f"重新发起原始请求完成，耗时: {elapsed:.2f}秒")
                
                if not response or not response.choices:
                    logger.warning("重新发起的原始请求返回无效响应，停止继续尝试")
                    break
                
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    logger.info(
                        f"重新请求 Token使用 - prompt_tokens: {usage.prompt_tokens}, "
                        f"completion_tokens: {usage.completion_tokens}, "
                        f"total_tokens: {usage.total_tokens}"
                    )
                
                # 使用重新请求的完整响应替换accumulated_text
                new_text = response.choices[0].message.content or ""
                if new_text:
                    combined_text = new_text
                    finish_reason = response.choices[0].finish_reason
                    logger.info(f"重新请求获得响应，长度: {len(new_text)}字符，finish_reason: {finish_reason}")
                    # 重新检查是否需要继续接续
                    if finish_reason == "length":
                        # 如果重新请求后仍然被截断，继续接续循环
                        continue
                    else:
                        # 重新请求后完成，退出循环
                        break
                else:
                    logger.warning("重新发起的原始请求响应仍为空，停止继续尝试")
                    break

            # 记录接续后的内容
            logger.info(f"接续后新增内容（长度: {len(extra_text)}字符）:")
            logger.info("=" * 80)
            logger.info(extra_text)
            logger.info("=" * 80)

            combined_text += extra_text
            finish_reason = response.choices[0].finish_reason
            logger.debug(f"接续后 finish_reason: {finish_reason}")
            
            # 记录合并后的内容（最后500字符，用于验证接续是否连贯）
            after_text = combined_text[-preview_length:] if len(combined_text) > preview_length else combined_text
            logger.info(f"接续后合并内容（最后{len(after_text)}字符，总长度{len(combined_text)}字符）:")
            logger.info("=" * 80)
            logger.info(after_text)
            logger.info("=" * 80)
            
            # 检查接续后是否还需要继续接续
            still_needs_continuation = False
            
            if finish_reason == "length":
                # finish_reason是"length"，需要继续接续
                still_needs_continuation = True
            else:
                # finish_reason不是"length"，但需要检查TSV格式是否完整
                cleaned_text = self._extract_text_from_markdown(combined_text)
                if cleaned_text:
                    lines = cleaned_text.strip().split('\n')
                    if len(lines) > 1:
                        # 检测表头格式，确定期望的列数
                        header_line = lines[0].strip().lower()
                        expected_columns = 3  # 默认期望3列（旧格式）
                        pass_column_index = 1  # Pass列在第二列（索引1）
                        result_column_index = 2  # Result列在第三列（索引2）
                        
                        # 检查是否是两列格式（Index\tPass）
                        if header_line.startswith('index') and '\t' in header_line:
                            header_fields = header_line.split('\t')
                            if len(header_fields) == 2 and 'pass' in header_line:
                                expected_columns = 2
                                pass_column_index = 1
                        
                        last_line = lines[-1].strip()
                        if last_line:
                            try:
                                reader = csv.reader([last_line], delimiter='\t')
                                fields = next(reader)
                                if len(fields) >= expected_columns:
                                    # 检查Pass列或Result列是否有值
                                    if expected_columns == 2:
                                        # 两列格式：检查Pass列
                                        pass_value = fields[pass_column_index].strip().lower() if len(fields) > pass_column_index else ""
                                        if pass_value in ["yes", "no", "y", "n", "是", "否"]:
                                            # TSV格式完整，不需要继续接续
                                            logger.info(f"接续后TSV格式已完整，停止接续")
                                            still_needs_continuation = False
                                        else:
                                            # Pass列格式不对，可能还需要接续
                                            still_needs_continuation = True
                                            logger.debug(f"接续后最后一行Pass列格式仍异常: '{pass_value}'，继续接续")
                                    else:
                                        # 三列格式：检查Result列
                                        result_value = fields[result_column_index].strip().lower() if len(fields) > result_column_index else ""
                                        if result_value in ["yes", "no", "y", "n", "是", "否"]:
                                            # TSV格式完整，不需要继续接续
                                            logger.info(f"接续后TSV格式已完整，停止接续")
                                            still_needs_continuation = False
                                        else:
                                            # Result列格式不对，可能还需要接续
                                            still_needs_continuation = True
                                            logger.debug(f"接续后最后一行Result列格式仍异常: '{result_value}'，继续接续")
                                else:
                                    # 字段数不足，需要继续接续
                                    still_needs_continuation = True
                                    logger.debug(f"接续后最后一行字段数仍不足（{len(fields)}/{expected_columns}），继续接续")
                            except Exception as e:
                                # 解析失败，检查其他截断特征
                                if last_line.count('"') % 2 != 0 or last_line.endswith(','):
                                    still_needs_continuation = True
                                    logger.debug(f"接续后最后一行仍有截断特征，继续接续: {e}")
                                else:
                                    # 无法判断，假设已完成
                                    still_needs_continuation = False
            
            # 如果不需要继续接续，退出循环
            if not still_needs_continuation:
                break

        # 如果达到最大尝试次数后仍然被截断，重新发起原始请求
        if finish_reason == "length":
            logger.warning(
                f"{task_name} 在尝试 {total_attempts} 次后仍被max_tokens截断，重新发起原始请求"
            )
            # 重新使用原始api_params发起请求
            start_time = time.time()
            response = self._call_api_with_retry(api_params)
            elapsed = time.time() - start_time
            logger.info(f"8次接续后重新发起原始请求完成，耗时: {elapsed:.2f}秒")
            
            if response and response.choices:
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage
                    logger.info(
                        f"重新请求 Token使用 - prompt_tokens: {usage.prompt_tokens}, "
                        f"completion_tokens: {usage.completion_tokens}, "
                        f"total_tokens: {usage.total_tokens}"
                    )
                
                new_text = response.choices[0].message.content or ""
                if new_text:
                    combined_text = new_text
                    finish_reason = response.choices[0].finish_reason
                    logger.info(f"重新请求获得响应，长度: {len(new_text)}字符，finish_reason: {finish_reason}")
                    if finish_reason == "length":
                        logger.warning(
                            f"{task_name} 重新请求后仍被max_tokens截断，输出可能不完整"
                        )
                else:
                    logger.warning(f"{task_name} 重新请求后响应为空，输出可能不完整")
            else:
                logger.warning(f"{task_name} 重新请求返回无效响应，输出可能不完整")

        return combined_text, finish_reason

    def evaluate_single_run(
        self,
        checkpoints: list[str],
        target_document_path: str | Path,
    ) -> DocumentEvaluation:
        """
        单次评估运行

        Args:
            checkpoints: 检查项清单
            target_document_path: 待评估文档路径

        Returns:
            评估结果
        """
        # 读取待评估文档内容
        target_content = self.parser.read_markdown(target_document_path)
        logger.info(f"开始评估文档: {target_document_path}")
        logger.debug(f"文档内容长度: {len(target_content)} 字符")

        # 加载prompt模板
        template = self._load_prompt_template("evaluate_points.md")

        # 去除检查项中的重复项（保持顺序）
        original_count = len(checkpoints)
        seen = set()
        unique_checkpoints = []
        for checkpoint in checkpoints:
            if checkpoint not in seen:
                seen.add(checkpoint)
                unique_checkpoints.append(checkpoint)
        
        if original_count != len(unique_checkpoints):
            logger.info(f"去重前检查项数量: {original_count}, 去重后: {len(unique_checkpoints)} (已去除 {original_count - len(unique_checkpoints)} 个重复项)")
        
        # 准备检查项TSV表格
        checkpoints_table = self._format_checkpoints_as_csv(unique_checkpoints)
        logger.debug(f"检查项数量: {len(unique_checkpoints)}, TSV长度: {len(checkpoints_table)} 字符")

        # 填充模板
        prompt = template.format(
            checkpoints_table=checkpoints_table, target_content=target_content
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
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.openai.temperature,
        }
        
        # 只有当 max_tokens 配置了值时才添加到参数中
        if self.config.openai.max_tokens is not None:
            api_params["max_tokens"] = self.config.openai.max_tokens
        
        # 在请求前计算输入 token 并调整 MAX_TOKENS
        max_context_length = Config.get_max_context_length()
        if max_context_length is not None:
            adjusted_max_tokens = calculate_adjusted_max_tokens(
                messages=api_params["messages"],
                max_context_length=max_context_length,
                configured_max_tokens=api_params.get("max_tokens")
            )
            
            if adjusted_max_tokens is not None:
                api_params["max_tokens"] = adjusted_max_tokens
                # 记录 token 计算信息
                input_tokens = count_tokens(api_params["messages"])
                if input_tokens is not None:
                    logger.debug(
                        f"Token计算 - 输入: {input_tokens}, "
                        f"最大上下文: {max_context_length}, "
                        f"调整后MAX_TOKENS: {adjusted_max_tokens}"
                    )
            elif "max_tokens" in api_params:
                # 如果调整后为 None（输入已超过最大上下文长度），移除 max_tokens 参数
                del api_params["max_tokens"]
                logger.warning(
                    "输入token数量可能超过最大上下文长度，移除MAX_TOKENS限制"
                )
        
        # 添加 stream 参数
        api_params["stream"] = self.config.openai.stream
        
        # 记录API调用参数
        logger.info(f"调用API - 模型: {api_params['model']}, temperature: {api_params['temperature']}, "
                   f"max_tokens: {api_params.get('max_tokens', 'None')}")
        logger.debug(f"API参数: {json.dumps({k: v for k, v in api_params.items() if k != 'messages'}, ensure_ascii=False)}")
        
        # 解析重试循环
        max_parse_retries = self.config.openai.max_parse_retries
        parse_attempt = 0
        last_exception = None
        
        while parse_attempt <= max_parse_retries:
            if parse_attempt > 0:
                logger.warning(
                    f"解析失败，第 {parse_attempt}/{max_parse_retries} 次重新请求API..."
                )
                # 指数退避：延迟时间 = 初始延迟 * 2^(attempt-1)
                retry_delay = self.config.openai.retry_delay * (2 ** (parse_attempt - 1))
                time.sleep(retry_delay)
            
            # 记录请求开始时间
            start_time = time.time()
            
            # 调用API（带重试机制）
            try:
                response = self._call_api_with_retry(api_params)
            except ValueError as e:
                # 重试机制已经记录了详细错误信息
                logger.debug(f"API调用失败详情:", exc_info=True)
                # 如果是最后一次尝试，抛出异常
                if parse_attempt >= max_parse_retries:
                    raise
                # 否则继续重试
                last_exception = e
                parse_attempt += 1
                continue
            
            # 记录响应时间
            elapsed_time = time.time() - start_time
            logger.info(f"API调用完成，耗时: {elapsed_time:.2f}秒")
            
            # 检查响应
            if not response or not response.choices:
                logger.error(f"API返回无效响应")
                logger.debug(f"API返回无效响应详情: {response}")
                if parse_attempt >= max_parse_retries:
                    raise ValueError(f"API返回无效响应: {response}")
                last_exception = ValueError(f"API返回无效响应: {response}")
                parse_attempt += 1
                continue
            
            # 记录token使用情况
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                logger.info(f"Token使用 - prompt_tokens: {usage.prompt_tokens}, "
                           f"completion_tokens: {usage.completion_tokens}, "
                           f"total_tokens: {usage.total_tokens}")
            
            # 解析响应
            result_text = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            logger.debug(f"API响应 finish_reason: {finish_reason}")
            if not result_text:
                # 尝试获取更多信息
                error_msg = f"API返回结果为空"
                logger.error(error_msg)
                logger.debug(f"API返回结果为空详情。响应对象: {response}")
                if hasattr(response, "error"):
                    logger.debug(f"错误信息: {response.error}")
                if parse_attempt >= max_parse_retries:
                    raise ValueError(error_msg)
                last_exception = ValueError(error_msg)
                parse_attempt += 1
                continue
            
            result_text, _ = self._continue_on_truncation(
                api_params, result_text, finish_reason, task_name="文档评估"
            )
            
            # 记录原始响应
            logger.debug(f"原始响应长度: {len(result_text)} 字符")
            # 记录完整原始响应内容
            logger.debug("=" * 80)
            logger.debug("完整原始响应内容:")
            logger.debug("=" * 80)
            logger.debug(result_text)
            logger.debug("=" * 80)
            
            # 清理markdown代码块标记
            cleaned_text = self._extract_text_from_markdown(result_text)
            logger.debug(f"清理后文本长度: {len(cleaned_text)} 字符 (减少了 {len(result_text) - len(cleaned_text)} 字符)")
            # 记录完整清理后文本内容
            logger.debug("=" * 80)
            logger.debug("完整清理后文本内容:")
            logger.debug("=" * 80)
            logger.debug(cleaned_text)
            logger.debug("=" * 80)
            
            # 尝试解析TSV格式的结果
            try:
                checkpoint_results = self._parse_tsv_result(cleaned_text, unique_checkpoints)
                
                # 验证解析结果是否有效
                is_valid, error_msg = self._validate_parse_result(
                    checkpoint_results, len(unique_checkpoints), cleaned_text
                )
                if not is_valid:
                    raise ValueError(f"解析结果无效: {error_msg}")
                
                # 统计通过的检查项数量
                total_passed = sum(1 for cp in checkpoint_results if cp.passed)
                logger.info(f"成功解析TSV，共 {total_passed} 个通过的检查项（共 {len(unique_checkpoints)} 个检查项）")
                
                evaluation = DocumentEvaluation(
                    target_document=str(target_document_path),
                    checkpoints=unique_checkpoints,
                    checkpoint_results=checkpoint_results,
                )
                logger.info(f"评估完成 - 准确性: {evaluation.accuracy:.2f}, "
                           f"综合: {evaluation.comprehensive:.2f}")
                
                # 如果之前有重试，记录成功信息
                if parse_attempt > 0:
                    logger.info(f"解析重试成功（第 {parse_attempt + 1} 次尝试）")
                
                return evaluation
                
            except (ValueError, csv.Error) as e:
                # 提供更详细的错误信息
                error_preview = result_text[:500] if len(result_text) > 500 else result_text
                cleaned_preview = cleaned_text[:500] if len(cleaned_text) > 500 else cleaned_text
                logger.warning(f"解析评估结果失败: {e}")
                logger.debug(f"解析评估结果失败详情:", exc_info=True)
                logger.debug(f"原始响应（前500字符）: {error_preview}")
                logger.debug(f"清理后文本（前500字符）: {cleaned_preview}")
                logger.debug(f"原始响应完整长度: {len(result_text)} 字符")
                logger.debug(f"清理后文本完整长度: {len(cleaned_text)} 字符")
                
                # 如果已达到最大重试次数，抛出异常
                if parse_attempt >= max_parse_retries:
                    raise ValueError(f"解析评估结果失败（已重试 {max_parse_retries} 次）: {e} (详细内容请查看日志文件)")
                
                # 否则继续重试
                last_exception = e
                parse_attempt += 1
                continue
        
        # 如果所有重试都失败，抛出最后一个异常
        if last_exception:
            raise ValueError(f"解析评估结果失败（已重试 {max_parse_retries} 次）: {last_exception} (详细内容请查看日志文件)")
        else:
            raise ValueError(f"解析评估结果失败（已重试 {max_parse_retries} 次）")

    def evaluate_multiple_runs(
        self,
        checkpoints: list[str],
        target_document_path: str | Path,
        runs: int = 3,
        merge_strategy: MergeStrategy | str = MergeStrategy.ACCURACY_AVERAGE,
        baseline_document_path: str | Path | None = None,
    ) -> DocumentEvaluation:
        """
        多次运行评估并合并结果（并行执行）
        
        支持两种合并策略：
        1. CHECKPOINT_MAJORITY: 检查项合并 - 每个检查项按多数投票判定，再统计通过项占比
        2. ACCURACY_AVERAGE: 准确性合并 - 对每个评委的通过项数量进行平均计算

        Args:
            checkpoints: 检查项清单
            target_document_path: 待评估文档路径
            runs: 运行次数（评委数量）
            merge_strategy: 合并策略，可选值：
                - CHECKPOINT_MAJORITY: 检查项合并（默认）
                - ACCURACY_AVERAGE: 准确性合并

        Returns:
            合并后的评估结果
        """
        # 记录开始时间
        start_time = time.time()
        evaluation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        results = []
        
        # 并行执行多次评估
        with ThreadPoolExecutor(max_workers=runs) as executor:
            futures = {
                executor.submit(self.evaluate_single_run, checkpoints, target_document_path): i
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

        # 处理字符串类型的策略
        if isinstance(merge_strategy, str):
            try:
                merge_strategy = MergeStrategy(merge_strategy)
            except ValueError:
                merge_strategy = MergeStrategy.ACCURACY_AVERAGE

        # 保存所有评委的检查项结果，用于在报告中显示每个评委的判断
        all_judge_results = [r.checkpoint_results for r in results]

        # 同时计算两种策略的结果
        # 策略1：检查项合并 - 每个检查项按多数投票判定，再统计通过项占比
        checkpoint_merge_result = self.merge_evaluation_results_by_checkpoints(
            results, checkpoints, target_document_path, merge_strategy=MergeStrategy.CHECKPOINT_MAJORITY
        )
        
        # 策略2：准确性合并 - 对每个评委的通过项数量进行平均计算
        accuracy_scores = [r.accuracy for r in results]
        merged_accuracy = statistics.mean(accuracy_scores)
        merged_completeness = statistics.mean([r.completeness for r in results])
        merged_comprehensive = statistics.mean([r.comprehensive for r in results])
        
        first_result = results[0]
        accuracy_merge_result = DocumentEvaluation(
            target_document=str(target_document_path),
            checkpoints=checkpoints,
            checkpoint_results=first_result.checkpoint_results,
            all_judge_results=all_judge_results,
        )
        accuracy_merge_result.accuracy = merged_accuracy
        accuracy_merge_result.completeness = merged_completeness
        accuracy_merge_result.comprehensive = merged_comprehensive

        # 根据指定策略选择返回哪个结果，但保存两种策略的结果
        if merge_strategy == MergeStrategy.CHECKPOINT_MAJORITY:
            final_result = checkpoint_merge_result
        else:
            final_result = accuracy_merge_result
        
        # 在最终结果中保存两种策略的结果
        final_result.checkpoint_merge_result = checkpoint_merge_result
        final_result.accuracy_merge_result = accuracy_merge_result
        
        # 保存评估元信息
        final_result.model_name = self.config.openai.model
        if baseline_document_path:
            final_result.baseline_document = str(baseline_document_path)
        
        # 记录评估时间和耗时
        final_result.evaluation_time = evaluation_time
        final_result.evaluation_duration = time.time() - start_time

        return final_result

    def merge_evaluation_results(
        self,
        raw_results: list[DocumentEvaluation],
        checkpoints: list[str],
        target_document_path: str | Path,
        merge_strategy: MergeStrategy | str = MergeStrategy.ACCURACY_AVERAGE,
    ) -> DocumentEvaluation:
        """
        准确性合并策略：对每个评委的通过项数量进行平均计算
        
        Args:
            raw_results: 原始评估结果列表（来自多次 evaluate_single_run）
            checkpoints: 检查项清单
            target_document_path: 待评估文档路径
            merge_strategy: 合并策略（仅用于兼容性，实际使用平均值）
        
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

        # 使用平均值合并
        merged_accuracy = statistics.mean(accuracy_scores)
        merged_completeness = statistics.mean(completeness_scores)
        merged_comprehensive = statistics.mean(comprehensive_scores)

        # 使用第一个评委的详细评估结果（checkpoint_results字段）
        first_result = raw_results[0]

        # 保存所有评委的检查项结果，用于在报告中显示每个评委的判断
        all_judge_results = [r.checkpoint_results for r in raw_results]

        # 创建最终结果对象，手动设置合并后的分数
        final_result = DocumentEvaluation(
            target_document=str(target_document_path),
            checkpoints=checkpoints,
            checkpoint_results=first_result.checkpoint_results,
            all_judge_results=all_judge_results,
        )
        
        # 覆盖计算出的分数，使用合并后的分数
        final_result.accuracy = merged_accuracy
        final_result.completeness = merged_completeness
        final_result.comprehensive = merged_comprehensive

        return final_result

    def merge_evaluation_results_by_checkpoints(
        self,
        raw_results: list[DocumentEvaluation],
        checkpoints: list[str],
        target_document_path: str | Path,
        merge_strategy: MergeStrategy | str = MergeStrategy.CHECKPOINT_MAJORITY,
    ) -> DocumentEvaluation:
        """
        检查项合并策略：每个检查项按多数投票判定结果是否通过，再统计所有投票通过的项占比
        
        Args:
            raw_results: 原始评估结果列表（来自多次 evaluate_single_run）
            checkpoints: 检查项清单
            target_document_path: 待评估文档路径
            merge_strategy: 合并策略（仅用于兼容性，实际使用多数投票）
        
        Returns:
            合并后的评估结果
        """
        runs = len(raw_results)
        if runs == 0:
            raise ValueError("原始结果列表不能为空")
        
        # 如果只有一个结果，直接返回
        if runs == 1:
            return raw_results[0]

        # 合并评估结果（基于检查项结果）- 使用多数投票
        checkpoint_votes = {}  # checkpoint_index -> {passed_count, total_count}
        for result in raw_results:
            for index, cp_result in enumerate(result.checkpoint_results):
                if index not in checkpoint_votes:
                    checkpoint_votes[index] = {
                        "passed_count": 0,
                        "total_count": 0,
                    }
                checkpoint_votes[index]["total_count"] += 1
                if cp_result.passed:
                    checkpoint_votes[index]["passed_count"] += 1
        
        # 保存所有评委的检查项结果，用于在报告中显示每个评委的判断
        all_judge_results = [r.checkpoint_results for r in raw_results]
        
        # 创建合并后的检查项结果 - 使用多数投票（超过50%认为通过则通过）
        merged_checkpoint_results = []
        for index, checkpoint in enumerate(checkpoints):
            votes = checkpoint_votes.get(index, {
                "passed_count": 0,
                "total_count": runs,
            })
            
            # 多数投票：超过50%认为通过则通过
            passed = votes["passed_count"] > (runs / 2)
            pass_rate = votes["passed_count"] / runs if runs > 0 else 0.0
            
            merged_checkpoint_results.append(
                CheckpointResult(checkpoint=checkpoint, passed=passed, pass_rate=pass_rate)
            )
        
        # 创建最终结果对象
        # DocumentEvaluation 的 __init__ 方法会基于合并后的检查项结果自动计算分数
        final_result = DocumentEvaluation(
            target_document=str(target_document_path),
            checkpoints=checkpoints,
            checkpoint_results=merged_checkpoint_results,
            all_judge_results=all_judge_results,
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

