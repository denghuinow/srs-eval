"""要点提取模块"""

import copy
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI, APIError, APITimeoutError, InternalServerError

from src.config import Config, load_config
from src.document_parser import DocumentParser

# 配置日志
logger = logging.getLogger(__name__)
CONTINUATION_PROMPT = (
    "上次回答因为达到 max_tokens 限制被截断。请从中断处继续完整输出剩余内容，保持相同的结构和格式，不要重复已经输出的内容。"
)


class PointExtractor:
    """从基准文档提取结构化要点清单"""

    def __init__(self, config: Config | None = None):
        """
        初始化要点提取器

        Args:
            config: 配置对象，如果为None则自动加载
        """
        self.config = config or load_config()
        self.client = OpenAI(
            api_key=self.config.openai.api_key,
            base_url=self.config.openai.base_url,
            timeout=self.config.openai.timeout,
        )
        self.parser = DocumentParser()

    @staticmethod
    def _clean_text_output(text: str) -> str:
        """
        清理文本输出，移除markdown代码块标记等
        
        处理以下情况：
        - ``` ... ```
        - 纯文本
        
        Args:
            text: 可能包含markdown代码块的文本
            
        Returns:
            清理后的文本
        """
        text = text.strip()
        
        # 尝试匹配 ``` ... ```
        if text.startswith("```"):
            # 找到第一个换行符（代码块开始标记之后）
            first_newline = text.find("\n")
            if first_newline != -1:
                # 移除开始标记（``` 或 ```xxx）
                text = text[first_newline + 1:]
            
            # 找到最后一个 ```（代码块结束标记）
            last_code_block = text.rfind("```")
            if last_code_block != -1:
                # 移除结束标记
                text = text[:last_code_block]
        
        return text.strip()
    
    @staticmethod
    def _remove_numbering(line: str) -> str:
        """
        移除行首的编号或标记
        
        Args:
            line: 文本行
            
        Returns:
            移除编号后的文本
        """
        line = line.strip()
        # 移除常见的编号格式：1. 2. 3. 或 1) 2) 3) 或 - * • 等
        import re
        # 匹配开头的数字编号（如 "1. ", "1) ", "1、", "1. " 等）
        line = re.sub(r'^\d+[\.\)、]\s*', '', line)
        # 匹配开头的符号（如 "- ", "* ", "• ", "· " 等）
        line = re.sub(r'^[-*•·]\s+', '', line)
        return line.strip()

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

    def _get_cache_path(self, document_path: str | Path) -> Path:
        """
        获取缓存文件路径

        Args:
            document_path: 文档路径

        Returns:
            缓存文件路径
        """
        doc_path = Path(document_path)
        # 基于文档路径生成缓存文件名
        # 使用文档路径的hash来生成唯一文件名
        doc_str = str(doc_path.absolute())
        doc_hash = hashlib.md5(doc_str.encode()).hexdigest()[:12]
        doc_name = doc_path.stem
        
        cache_dir = Path(__file__).parent.parent / ".cache" / "points"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        return cache_dir / f"{doc_name}_{doc_hash}.json"

    def _get_content_hash(self, content: str) -> str:
        """
        计算文档内容的hash值

        Args:
            content: 文档内容

        Returns:
            hash值
        """
        return hashlib.sha256(content.encode()).hexdigest()

    def _save_points_cache(
        self, document_path: str | Path, checkpoints: list[str], content_hash: str
    ) -> None:
        """
        保存检查项清单到缓存文件

        Args:
            document_path: 文档路径
            checkpoints: 检查项清单
            content_hash: 文档内容hash
        """
        cache_path = self._get_cache_path(document_path)
        cache_data = {
            "document_path": str(Path(document_path).absolute()),
            "content_hash": content_hash,
            "checkpoints": checkpoints,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    def _load_points_cache(
        self, document_path: str | Path, content_hash: str | None = None
    ) -> list[str] | None:
        """
        从缓存文件加载检查项清单

        Args:
            document_path: 文档路径
            content_hash: 文档内容hash（用于验证缓存有效性，如果为None则不验证）

        Returns:
            检查项清单，如果缓存不存在或无效则返回None
        """
        cache_path = self._get_cache_path(document_path)
        
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            # 验证文档路径是否匹配
            cached_path = cache_data.get("document_path", "")
            if cached_path != str(Path(document_path).absolute()):
                return None
            
            # 如果提供了content_hash，验证缓存是否有效
            if content_hash is not None:
                cached_hash = cache_data.get("content_hash", "")
                if cached_hash != content_hash:
                    return None
            
            # 读取checkpoints数组
            checkpoints = cache_data.get("checkpoints", [])
            if not checkpoints:
                return None
            
            return checkpoints
        except (json.JSONDecodeError, KeyError, IOError):
            return None

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
                        finish_reason=finish_reason
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
        task_name: str = "要点提取",
        document_path: str | Path | None = None,
    ) -> tuple[str, str | None]:
        """
        当模型因为max_tokens截断时自动发送接续请求

        Args:
            api_params: 首次请求的参数（用于构造后续messages）
            accumulated_text: 当前已生成的文本
            finish_reason: 首次响应的finish_reason
            task_name: 日志中展示的任务名称
            document_path: 文档路径（用于日志记录）

        Returns:
            (完整文本, 最终finish_reason)
        """
        # 构建日志前缀，包含文件信息
        log_prefix = f"[{Path(document_path).name}] " if document_path else ""
        
        if finish_reason != "length":
            return accumulated_text, finish_reason
        if self.config.openai.max_continuations <= 0:
            logger.warning(f"{log_prefix}{task_name} 输出因max_tokens被截断，但未配置接续次数，将直接返回当前结果")
            return accumulated_text, finish_reason
        base_messages = api_params.get("messages")
        if not base_messages:
            logger.warning(f"{log_prefix}{task_name} 输出被截断，但缺少原始消息上下文，无法自动接续")
            return accumulated_text, finish_reason

        combined_text = accumulated_text
        total_attempts = self.config.openai.max_continuations
        attempt = 0

        while finish_reason == "length" and attempt < total_attempts:
            attempt += 1
            logger.warning(
                f"{log_prefix}{task_name} 响应达到 max_tokens，自动发送第 {attempt}/{total_attempts} 次接续请求..."
            )
            continuation_messages = copy.deepcopy(base_messages)
            continuation_messages.append({"role": "assistant", "content": combined_text})
            continuation_messages.append({"role": "user", "content": CONTINUATION_PROMPT})

            continuation_params = {
                k: v for k, v in api_params.items() if k != "messages"
            }
            continuation_params["messages"] = continuation_messages

            start_time = time.time()
            response = self._call_api_with_retry(continuation_params)
            elapsed = time.time() - start_time
            logger.info(f"{log_prefix}接续请求 #{attempt} 完成，耗时: {elapsed:.2f}秒")

            if not response or not response.choices:
                logger.warning(f"{log_prefix}接续请求返回无效响应，停止继续尝试")
                break

            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                logger.info(
                    f"{log_prefix}接续 Token使用 - prompt_tokens: {usage.prompt_tokens}, "
                    f"completion_tokens: {usage.completion_tokens}, "
                    f"total_tokens: {usage.total_tokens}"
                )

            extra_text = response.choices[0].message.content or ""
            if not extra_text:
                logger.warning(f"{log_prefix}接续响应内容为空，停止继续尝试")
                break

            combined_text += extra_text
            finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            logger.warning(
                f"{log_prefix}{task_name} 在尝试 {total_attempts} 次后仍被max_tokens截断，输出可能不完整"
            )

        return combined_text, finish_reason

    def _extract_points_single(
        self, content: str, document_path: str | Path | None = None
    ) -> list[str]:
        """
        单次提取检查项清单（内部方法）

        Args:
            content: 文档内容
            document_path: 文档路径（用于日志记录）

        Returns:
            检查项清单（平铺的checkpoints数组）
        """
        # 构建日志前缀，包含文件信息
        log_prefix = f"[{Path(document_path).name}] " if document_path else ""
        
        # 加载prompt模板
        template = self._load_prompt_template("extract_points.txt")
        logger.info(f"{log_prefix}开始提取要点")
        logger.debug(f"{log_prefix}文档内容长度: {len(content)} 字符")

        # 填充模板
        prompt = template.format(document_content=content)
        logger.debug(f"{log_prefix}完整prompt长度: {len(prompt)} 字符")
        # 记录完整Prompt内容
        logger.debug(f"{log_prefix}{'=' * 80}")
        logger.debug(f"{log_prefix}完整Prompt内容:")
        logger.debug(f"{log_prefix}{'=' * 80}")
        logger.debug(prompt)
        logger.debug(f"{log_prefix}{'=' * 80}")
        
        # 检查prompt长度（可选：如果太长可以截断或警告）
        # 注意：不同模型的token限制不同，这里只是简单检查字符数
        if len(prompt) > 100000:  # 大约25k tokens（粗略估计）
            logger.warning(f"{log_prefix}Prompt内容较长 ({len(prompt)} 字符)，可能需要较长时间处理")

        # 调用OpenAI API
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
        
        # 添加 stream 参数
        api_params["stream"] = self.config.openai.stream
        
        # 记录API调用参数
        logger.info(f"{log_prefix}调用API - 模型: {api_params['model']}, temperature: {api_params['temperature']}, "
                   f"max_tokens: {api_params.get('max_tokens', 'None')}")
        logger.debug(f"{log_prefix}API参数: {json.dumps({k: v for k, v in api_params.items() if k != 'messages'}, ensure_ascii=False)}")
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 调用API（带重试机制）
        try:
            response = self._call_api_with_retry(api_params)
        except ValueError as e:
            # 重试机制已经记录了详细错误信息
            logger.debug(f"{log_prefix}API调用失败详情:", exc_info=True)
            raise
        
        # 记录响应时间
        elapsed_time = time.time() - start_time
        logger.info(f"{log_prefix}API调用完成，耗时: {elapsed_time:.2f}秒")

        # 检查响应
        if not response or not response.choices:
            logger.error(f"{log_prefix}API返回无效响应")
            logger.debug(f"{log_prefix}API返回无效响应详情: {response}")
            raise ValueError(f"API返回无效响应: {response}")

        # 记录token使用情况
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            logger.info(f"{log_prefix}Token使用 - prompt_tokens: {usage.prompt_tokens}, "
                       f"completion_tokens: {usage.completion_tokens}, "
                       f"total_tokens: {usage.total_tokens}")

        # 解析响应
        result_text = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        if not result_text:
            # 尝试获取更多信息
            error_msg = f"API返回结果为空"
            logger.error(f"{log_prefix}{error_msg}")
            logger.debug(f"{log_prefix}API返回结果为空详情。响应对象: {response}")
            if hasattr(response, "error"):
                logger.debug(f"{log_prefix}错误信息: {response.error}")
            raise ValueError(error_msg)

        # 如因max_tokens截断则自动请求接续
        result_text, _ = self._continue_on_truncation(
            api_params, result_text, finish_reason, task_name="要点提取", document_path=document_path
        )

        # 记录原始响应
        logger.debug(f"{log_prefix}原始响应长度: {len(result_text)} 字符")
        # 记录完整原始响应内容
        logger.debug(f"{log_prefix}{'=' * 80}")
        logger.debug(f"{log_prefix}完整原始响应内容:")
        logger.debug(f"{log_prefix}{'=' * 80}")
        logger.debug(result_text)
        logger.debug(f"{log_prefix}{'=' * 80}")

        # 清理文本：移除markdown代码块标记（如果有）
        cleaned_text = self._clean_text_output(result_text)
        logger.debug(f"{log_prefix}清理后文本长度: {len(cleaned_text)} 字符 (减少了 {len(result_text) - len(cleaned_text)} 字符)")
        # 记录完整清理后文本内容
        logger.debug(f"{log_prefix}{'=' * 80}")
        logger.debug(f"{log_prefix}完整清理后文本内容:")
        logger.debug(f"{log_prefix}{'=' * 80}")
        logger.debug(cleaned_text)
        logger.debug(f"{log_prefix}{'=' * 80}")

        try:
            # 按行解析检查项
            checkpoints = []
            lines = cleaned_text.split('\n')
            
            for line in lines:
                line = line.strip()
                # 跳过空行
                if not line:
                    continue
                # 跳过明显的编号或标记（如 "1. ", "- ", "* " 等）
                line = self._remove_numbering(line)
                # 跳过太短的行（可能是格式标记）
                if len(line) < 3:
                    continue
                checkpoints.append(line)
            
            if not checkpoints:
                logger.error(f"{log_prefix}未提取到任何检查项")
                raise ValueError("未提取到任何检查项")
            
            logger.info(f"{log_prefix}成功解析文本，提取到 {len(checkpoints)} 个检查项")
            return checkpoints
        except Exception as e:
            # 提供更详细的错误信息
            error_preview = result_text[:500] if len(result_text) > 500 else result_text
            cleaned_preview = cleaned_text[:500] if len(cleaned_text) > 500 else cleaned_text
            logger.error(f"{log_prefix}解析文本失败: {e}")
            logger.debug(f"{log_prefix}解析文本失败详情:", exc_info=True)
            logger.debug(f"{log_prefix}原始响应（前500字符）: {error_preview}")
            logger.debug(f"{log_prefix}清理后文本（前500字符）: {cleaned_preview}")
            logger.debug(f"{log_prefix}原始响应完整长度: {len(result_text)} 字符")
            logger.debug(f"{log_prefix}清理后文本完整长度: {len(cleaned_text)} 字符")
            # 异常消息只包含简短信息，详细内容已在日志文件中记录
            raise ValueError(f"解析文本失败: {e} (详细内容请查看日志文件)")

    def extract_points(
        self,
        document_path: str | Path,
        force_extract: bool = False,
        extract_runs: int = 1,
    ) -> list[str]:
        """
        从文档中提取检查项清单（支持缓存和多次提取取最优）

        Args:
            document_path: 文档路径
            force_extract: 是否强制重新提取（忽略缓存）
            extract_runs: 提取运行次数，多次提取后选择检查项数量最多的结果（默认1次）

        Returns:
            检查项清单，格式为 ["检查项1", "检查项2", ...]
        """
        # 读取文档内容
        content = self.parser.read_markdown(document_path)
        
        if not content or not content.strip():
            raise ValueError(f"文档内容为空: {document_path}")

        # 计算文档内容hash
        content_hash = self._get_content_hash(content)

        # 尝试从缓存加载
        if not force_extract:
            cached_checkpoints = self._load_points_cache(document_path, content_hash)
            if cached_checkpoints is not None:
                logger.info(f"✓ 从缓存加载检查项清单（{len(cached_checkpoints)} 个检查项）")
                return cached_checkpoints

        # 缓存不存在或强制重新提取，调用API提取
        logger.info(f"开始提取检查项，文档: {document_path}, 提取次数: {extract_runs}, 强制提取: {force_extract}")
        if extract_runs <= 1:
            # 单次提取
            checkpoints = self._extract_points_single(content, document_path=document_path)
            logger.info(f"✓ 成功提取 {len(checkpoints)} 个检查项")
        else:
            # 多次提取，并行执行，选择检查项数量最多的结果
            logger.info(f"正在并行执行 {extract_runs} 次提取，选择检查项数量最多的结果...")
            all_results = []
            
            def extract_with_index(i: int) -> tuple[int, list[str]]:
                """带索引的提取函数，用于并行执行"""
                try:
                    checkpoints = self._extract_points_single(content, document_path=document_path)
                    return (i, checkpoints)
                except Exception as e:
                    logger.error(f"  第 {i+1}/{extract_runs} 次提取失败: {e}")
                    logger.debug(f"提取失败详情:", exc_info=True)
                    return (i, None)
            
            # 并行执行提取
            with ThreadPoolExecutor(max_workers=extract_runs) as executor:
                futures = {
                    executor.submit(extract_with_index, i): i
                    for i in range(extract_runs)
                }
                
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    i, checkpoints = future.result()
                    if checkpoints is not None:
                        all_results.append(checkpoints)
                        logger.info(f"  [{completed}/{extract_runs}] 第 {i+1} 次提取完成: {len(checkpoints)} 个检查项")
            
            if not all_results:
                raise ValueError("所有提取尝试均失败")
            
            # 选择检查项数量最多的结果
            checkpoints = max(all_results, key=len)
            logger.info(f"✓ 选择最优结果：{len(checkpoints)} 个检查项（从 {len(all_results)} 次成功提取中选择）")
            
            # 显示其他结果的统计信息
            if len(all_results) > 1:
                checkpoint_counts = [len(cp) for cp in all_results]
                logger.info(f"  提取结果统计（检查项数量）：最少 {min(checkpoint_counts)} 个，最多 {max(checkpoint_counts)} 个，平均 {sum(checkpoint_counts)/len(checkpoint_counts):.1f} 个")
        
        # 保存到缓存
        self._save_points_cache(document_path, checkpoints, content_hash)
        cache_path = self._get_cache_path(document_path)
        logger.info(f"✓ 检查项清单已保存到缓存: {cache_path}")
        
        return checkpoints
