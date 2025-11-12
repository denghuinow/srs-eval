"""文档解析模块"""

from pathlib import Path
from typing import Optional


class DocumentParser:
    """Markdown文档解析器"""

    @staticmethod
    def read_markdown(file_path: str | Path) -> str:
        """
        读取Markdown文档内容

        Args:
            file_path: 文档路径

        Returns:
            文档内容字符串

        Raises:
            FileNotFoundError: 文件不存在
            IOError: 文件读取失败
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            raise IOError(f"读取文件失败: {file_path}, 错误: {e}")

    @staticmethod
    def validate_markdown(file_path: str | Path) -> bool:
        """
        验证文件是否为Markdown格式

        Args:
            file_path: 文件路径

        Returns:
            是否为Markdown文件
        """
        path = Path(file_path)
        return path.suffix.lower() in [".md", ".markdown"]

