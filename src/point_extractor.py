"""要点提取模块"""

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import Config, load_config
from src.document_parser import DocumentParser

# 配置日志
logger = logging.getLogger(__name__)


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
        self, document_path: str | Path, points: list[dict[str, Any]], content_hash: str
    ) -> None:
        """
        保存要点清单到缓存文件

        Args:
            document_path: 文档路径
            points: 要点清单
            content_hash: 文档内容hash
        """
        cache_path = self._get_cache_path(document_path)
        cache_data = {
            "document_path": str(Path(document_path).absolute()),
            "content_hash": content_hash,
            "points": points,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    def _load_points_cache(
        self, document_path: str | Path, content_hash: str | None = None
    ) -> list[dict[str, Any]] | None:
        """
        从缓存文件加载要点清单

        Args:
            document_path: 文档路径
            content_hash: 文档内容hash（用于验证缓存有效性，如果为None则不验证）

        Returns:
            要点清单，如果缓存不存在或无效则返回None
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
            
            points = cache_data.get("points", [])
            if not points:
                return None
            
            return points
        except (json.JSONDecodeError, KeyError, IOError):
            return None

    def _extract_points_single(self, content: str) -> list[dict[str, Any]]:
        """
        单次提取要点清单（内部方法）

        Args:
            content: 文档内容

        Returns:
            要点清单
        """
        # 加载prompt模板
        template = self._load_prompt_template("extract_points.txt")
        logger.info("开始提取要点")
        logger.debug(f"文档内容长度: {len(content)} 字符")

        # 填充模板
        prompt = template.format(document_content=content)
        logger.debug(f"完整prompt长度: {len(prompt)} 字符")
        # 记录完整Prompt内容
        logger.debug("=" * 80)
        logger.debug("完整Prompt内容:")
        logger.debug("=" * 80)
        logger.debug(prompt)
        logger.debug("=" * 80)
        
        # 检查prompt长度（可选：如果太长可以截断或警告）
        # 注意：不同模型的token限制不同，这里只是简单检查字符数
        if len(prompt) > 100000:  # 大约25k tokens（粗略估计）
            logger.warning(f"Prompt内容较长 ({len(prompt)} 字符)，可能需要较长时间处理")

        # 调用OpenAI API
        # 尝试使用response_format，如果不支持则回退
        api_params = {
            "model": self.config.openai.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的需求文档分析专家。严格按照JSON格式输出，不要有任何其他文字。",
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
            points = result_json.get("points", [])
            if not points:
                logger.error("未提取到任何要点")
                raise ValueError("未提取到任何要点")
            logger.info(f"成功解析JSON，提取到 {len(points)} 个要点")
            logger.debug(f"要点ID列表: {[p.get('id', '') for p in points[:10]]}")
            return points
        except json.JSONDecodeError as e:
            # 提供更详细的错误信息
            error_preview = result_text[:500] if len(result_text) > 500 else result_text
            cleaned_preview = cleaned_text[:500] if len(cleaned_text) > 500 else cleaned_text
            logger.error(f"解析JSON失败: {e}")
            logger.debug(f"解析JSON失败详情:", exc_info=True)
            logger.debug(f"原始响应（前500字符）: {error_preview}")
            logger.debug(f"清理后文本（前500字符）: {cleaned_preview}")
            logger.debug(f"原始响应完整长度: {len(result_text)} 字符")
            logger.debug(f"清理后文本完整长度: {len(cleaned_text)} 字符")
            # 异常消息只包含简短信息，详细内容已在日志文件中记录
            raise ValueError(f"解析JSON失败: {e} (详细内容请查看日志文件)")

    def extract_points(
        self,
        document_path: str | Path,
        force_extract: bool = False,
        extract_runs: int = 1,
    ) -> list[dict[str, Any]]:
        """
        从文档中提取结构化要点清单（支持缓存和多次提取取最优）

        Args:
            document_path: 文档路径
            force_extract: 是否强制重新提取（忽略缓存）
            extract_runs: 提取运行次数，多次提取后选择要点数量最多的结果（默认1次）

        Returns:
            要点清单，格式为 [{"id": "1", "level": 1, "title": "...", ...}, ...]
        """
        # 读取文档内容
        content = self.parser.read_markdown(document_path)
        
        if not content or not content.strip():
            raise ValueError(f"文档内容为空: {document_path}")

        # 计算文档内容hash
        content_hash = self._get_content_hash(content)

        # 尝试从缓存加载
        if not force_extract:
            cached_points = self._load_points_cache(document_path, content_hash)
            if cached_points is not None:
                logger.info(f"✓ 从缓存加载要点清单（{len(cached_points)} 个要点）")
                return cached_points

        # 缓存不存在或强制重新提取，调用API提取
        logger.info(f"开始提取要点，文档: {document_path}, 提取次数: {extract_runs}, 强制提取: {force_extract}")
        if extract_runs <= 1:
            # 单次提取
            points = self._extract_points_single(content)
            logger.info(f"✓ 成功提取 {len(points)} 个要点")
        else:
            # 多次提取，并行执行，选择要点数量最多的结果
            logger.info(f"正在并行执行 {extract_runs} 次提取，选择要点数量最多的结果...")
            all_results = []
            
            def extract_with_index(i: int) -> tuple[int, list[dict[str, Any]]]:
                """带索引的提取函数，用于并行执行"""
                try:
                    points = self._extract_points_single(content)
                    return (i, points)
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
                    i, points = future.result()
                    if points is not None:
                        all_results.append(points)
                        logger.info(f"  [{completed}/{extract_runs}] 第 {i+1} 次提取完成: {len(points)} 个要点")
            
            if not all_results:
                raise ValueError("所有提取尝试均失败")
            
            # 选择要点数量最多的结果
            points = max(all_results, key=len)
            logger.info(f"✓ 选择最优结果：{len(points)} 个要点（从 {len(all_results)} 次成功提取中选择）")
            
            # 显示其他结果的统计信息
            if len(all_results) > 1:
                point_counts = [len(p) for p in all_results]
                logger.info(f"  提取结果统计：最少 {min(point_counts)} 个，最多 {max(point_counts)} 个，平均 {sum(point_counts)/len(point_counts):.1f} 个")
        
        # 保存到缓存
        self._save_points_cache(document_path, points, content_hash)
        cache_path = self._get_cache_path(document_path)
        logger.info(f"✓ 要点清单已保存到缓存: {cache_path}")
        
        return points

