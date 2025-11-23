"""检查点解析与分类工具函数"""

from __future__ import annotations

from typing import Tuple

# 支持的标准分类及常见别名映射
_CATEGORY_ALIASES: dict[str, str] = {
    "FUNCTIONAL": "FUNCTIONAL",
    "FUNCTION": "FUNCTIONAL",
    "FUNC": "FUNCTIONAL",
    "BUSINESS_FLOW": "BUSINESS_FLOW",
    "FLOW": "BUSINESS_FLOW",
    "PROCESS": "BUSINESS_FLOW",
    "BOUNDARY": "BOUNDARY",
    "LIMIT": "BOUNDARY",
    "EXCEPTION": "EXCEPTION",
    "ERROR": "EXCEPTION",
    "FAILURE": "EXCEPTION",
    "DATA_STATE": "DATA_STATE",
    "DATA": "DATA_STATE",
    "STATE": "DATA_STATE",
    "STATE_MACHINE": "DATA_STATE",
    "CONSISTENCY_RULE": "CONSISTENCY_RULE",
    "CONSISTENCY": "CONSISTENCY_RULE",
    "CONFLICT": "CONSISTENCY_RULE",
}

_DEFAULT_CATEGORY = "FUNCTIONAL"


def _normalize_token(token: str | None) -> str:
    """规范化类别标记"""
    if not token:
        return ""
    token = token.strip().upper().replace("-", "_").replace(" ", "_")
    if token.startswith("[") and token.endswith("]"):
        token = token[1:-1].strip()
    return token


def normalize_category(raw: str | None) -> str:
    """将原始类别字符串映射到标准类别"""
    token = _normalize_token(raw)
    return _CATEGORY_ALIASES.get(token, token or _DEFAULT_CATEGORY)


def split_checkpoint_category(checkpoint: str) -> Tuple[str, str]:
    """
    从检查点字符串中解析出类别和去除类别前缀后的正文

    支持的前缀格式示例：
    - [BOUNDARY] 系统需限制...
    - BOUNDARY: 系统需限制...
    - BOUNDARY | 系统需限制...
    - BUSINESS_FLOW 系统需限制...
    若未提供类别，则返回默认类别。
    """
    if not checkpoint:
        return _DEFAULT_CATEGORY, ""

    text = checkpoint.strip()

    # 1. 方括号前缀
    if text.startswith("[") and "]" in text:
        right_bracket = text.find("]")
        raw_category = text[1:right_bracket]
        content = text[right_bracket + 1 :].lstrip(" :|-")
        return normalize_category(raw_category), content.strip()

    # 2. 以类别开头并跟随分隔符
    separators = [":", "|", "-"]
    for sep in separators:
        if sep in text:
            possible_cat, possible_content = text.split(sep, 1)
            normalized = normalize_category(possible_cat)
            # 确认解析结果看起来像类别，且正文存在
            if normalized != possible_cat.strip() or possible_cat.strip().isupper():
                return normalized, possible_content.strip()

    # 3. 首个词为大写或包含下划线，视为类别
    first_token, *rest_tokens = text.split()
    normalized = normalize_category(first_token)
    if normalized != first_token or first_token.isupper():
        content = " ".join(rest_tokens).strip()
        return normalized, content if content else text

    # 4. 无显式类别，返回默认类别
    return _DEFAULT_CATEGORY, text


def format_checkpoint(category: str, content: str) -> str:
    """将类别与正文组合为统一的展示格式"""
    normalized_category = normalize_category(category)
    clean_content = content.strip()
    if not clean_content:
        return f"[{normalized_category}]"
    return f"[{normalized_category}] {clean_content}"
