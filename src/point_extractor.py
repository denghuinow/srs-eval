"""要点提取模块"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import Config, load_config
from src.document_parser import DocumentParser


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

        # 填充模板
        prompt = template.format(document_content=content)
        
        # 检查prompt长度（可选：如果太长可以截断或警告）
        # 注意：不同模型的token限制不同，这里只是简单检查字符数
        if len(prompt) > 100000:  # 大约25k tokens（粗略估计）
            print(f"警告: Prompt内容较长 ({len(prompt)} 字符)，可能需要较长时间处理")

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
            points = result_json.get("points", [])
            if not points:
                raise ValueError("未提取到任何要点")
            return points
        except json.JSONDecodeError as e:
            raise ValueError(f"解析JSON失败: {e}, 原始响应: {result_text}")

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
                print(f"✓ 从缓存加载要点清单（{len(cached_points)} 个要点）")
                return cached_points

        # 缓存不存在或强制重新提取，调用API提取
        if extract_runs <= 1:
            # 单次提取
            points = self._extract_points_single(content)
            print(f"✓ 成功提取 {len(points)} 个要点")
        else:
            # 多次提取，并行执行，选择要点数量最多的结果
            print(f"正在并行执行 {extract_runs} 次提取，选择要点数量最多的结果...")
            all_results = []
            
            def extract_with_index(i: int) -> tuple[int, list[dict[str, Any]]]:
                """带索引的提取函数，用于并行执行"""
                try:
                    points = self._extract_points_single(content)
                    return (i, points)
                except Exception as e:
                    print(f"  第 {i+1}/{extract_runs} 次提取失败: {e}")
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
                        print(f"  [{completed}/{extract_runs}] 第 {i+1} 次提取完成: {len(points)} 个要点")
            
            if not all_results:
                raise ValueError("所有提取尝试均失败")
            
            # 选择要点数量最多的结果
            points = max(all_results, key=len)
            print(f"✓ 选择最优结果：{len(points)} 个要点（从 {len(all_results)} 次成功提取中选择）")
            
            # 显示其他结果的统计信息
            if len(all_results) > 1:
                point_counts = [len(p) for p in all_results]
                print(f"  提取结果统计：最少 {min(point_counts)} 个，最多 {max(point_counts)} 个，平均 {sum(point_counts)/len(point_counts):.1f} 个")
        
        # 保存到缓存
        self._save_points_cache(document_path, points, content_hash)
        cache_path = self._get_cache_path(document_path)
        print(f"✓ 要点清单已保存到缓存: {cache_path}")
        
        return points

