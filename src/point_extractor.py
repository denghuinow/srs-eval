"""要点提取模块"""

import json
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

    def extract_points(self, document_path: str | Path) -> list[dict[str, Any]]:
        """
        从文档中提取结构化要点清单

        Args:
            document_path: 文档路径

        Returns:
            要点清单，格式为 [{"id": "1", "level": 1, "title": "...", ...}, ...]
        """
        # 读取文档内容
        content = self.parser.read_markdown(document_path)
        
        if not content or not content.strip():
            raise ValueError(f"文档内容为空: {document_path}")

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

