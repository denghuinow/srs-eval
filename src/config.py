"""配置管理模块"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class OpenAIConfig(BaseModel):
    """OpenAI API配置"""

    api_key: str = Field(..., description="OpenAI API密钥")
    model: str = Field(default="gpt-4", description="使用的模型")
    base_url: str = Field(
        default="https://api.openai.com/v1", description="API基础URL"
    )
    temperature: Optional[float] = Field(default=None, description="温度参数，None时使用API默认值")
    max_tokens: Optional[int] = Field(default=None, description="最大token数，None时使用API默认值")
    timeout: float = Field(default=1800.0, description="API调用超时时间（秒）")
    max_retries: int = Field(default=5, description="最大重试次数")
    retry_delay: float = Field(default=8.0, description="重试延迟初始值（秒），使用指数退避")
    stream: bool = Field(default=True, description="是否启用流式响应")
    max_continuations: int = Field(
        default=2, description="当因max_tokens导致输出被截断时自动请求接续的次数上限"
    )
    max_parse_retries: int = Field(
        default=2, description="当解析TSV结果失败时，重新请求API的最大重试次数"
    )


class EvalConfig(BaseModel):
    """评估配置"""

    default_runs: int = Field(default=3, description="默认运行次数（用于取平均）")


class Config(BaseModel):
    """全局配置"""

    openai: OpenAIConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)
    prompt_version: str = Field(default="v1", description="提示词版本")
    
    @classmethod
    def get_max_context_length(cls) -> Optional[int]:
        """获取最大上下文长度配置值"""
        max_context_length_str = os.getenv("MAX_CONTEXT_LENGTH")
        if max_context_length_str:
            try:
                return int(max_context_length_str)
            except ValueError:
                return None
        return None


def load_config() -> Config:
    """加载配置"""
    # 加载.env文件
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # 从环境变量读取配置
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY未设置，请在.env文件中配置或设置环境变量"
        )

    # 处理 max_tokens：如果环境变量存在则使用，否则为 None（使用 API 默认值）
    max_tokens_env = os.getenv("MAX_TOKENS")
    max_tokens = int(max_tokens_env) if max_tokens_env else None
    max_continuations = int(os.getenv("MAX_CONTINUATIONS", "2"))
    if max_continuations < 0:
        max_continuations = 0
    max_parse_retries = int(os.getenv("MAX_PARSE_RETRIES", "2"))
    if max_parse_retries < 0:
        max_parse_retries = 0

    # 处理 temperature：如果环境变量存在则使用，否则为 None（使用 API 默认值）
    temperature_env = os.getenv("TEMPERATURE")
    temperature = float(temperature_env) if temperature_env else None

    openai_config = OpenAIConfig(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=float(os.getenv("OPENAI_TIMEOUT", "900")),
        max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
        retry_delay=float(os.getenv("OPENAI_RETRY_DELAY", "1.0")),
        stream=os.getenv("OPENAI_STREAM", "true").lower() in ("true", "1", "yes"),
        max_continuations=max_continuations,
        max_parse_retries=max_parse_retries,
    )

    eval_config = EvalConfig(
        default_runs=int(os.getenv("DEFAULT_RUNS", "3")),
    )

    prompt_version = os.getenv("PROMPT_VERSION", "v1")

    return Config(openai=openai_config, eval=eval_config, prompt_version=prompt_version)
