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
    temperature: float = Field(default=0.0, description="温度参数，固定为0确保可重复性")
    max_tokens: int = Field(default=4000, description="最大token数")


class EvalConfig(BaseModel):
    """评估配置"""

    default_runs: int = Field(default=3, description="默认运行次数（用于取平均）")
    completeness_weight: float = Field(default=0.5, description="完整性权重")
    accuracy_weight: float = Field(default=0.5, description="准确性权重")


class Config(BaseModel):
    """全局配置"""

    openai: OpenAIConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)


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

    openai_config = OpenAIConfig(
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=float(os.getenv("TEMPERATURE", "0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "4000")),
    )

    eval_config = EvalConfig(
        default_runs=int(os.getenv("DEFAULT_RUNS", "3")),
        completeness_weight=float(os.getenv("COMPLETENESS_WEIGHT", "0.5")),
        accuracy_weight=float(os.getenv("ACCURACY_WEIGHT", "0.5")),
    )

    return Config(openai=openai_config, eval=eval_config)

