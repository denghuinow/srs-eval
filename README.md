# 需求文档差异评估系统

基于大模型的需求文档评估工具，以基准文档为真值，评估待评估文档的完整性和准确性。

## 功能特性

- 从基准文档自动提取结构化要点清单
- 逐条核对待评估文档，计算可量化的分数
- 支持完整性、准确性两个维度的评估
- 确保评估结果的可重复性（固定temperature、多次运行取平均）
- 支持JSON、CSV、Markdown三种输出格式

## 安装

使用 [uv](https://github.com/astral-sh/uv) 管理项目环境：

```bash
# 安装uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖
uv sync
```

## 配置

创建 `.env` 文件并填入你的OpenAI API配置：

```bash
# 创建 .env 文件
cat > .env << EOF
# OpenAI API密钥（必填）
OPENAI_API_KEY=your_openai_api_key_here

# 使用的模型（默认：gpt-4）
OPENAI_MODEL=gpt-4

# OpenAI API基础URL（默认：https://api.openai.com/v1）
# 可用于配置代理或使用兼容OpenAI API的其他服务
OPENAI_BASE_URL=https://api.openai.com/v1

# 温度参数（默认：0，确保可重复性）
TEMPERATURE=0

# 默认运行次数（默认：3，用于取平均提高可重复性）
DEFAULT_RUNS=3
EOF
```

### 配置说明

- **OPENAI_API_KEY**: OpenAI API密钥，必填项
- **OPENAI_MODEL**: 使用的模型名称，默认为 `gpt-4`，也可使用 `gpt-3.5-turbo` 等
- **OPENAI_BASE_URL**: API基础URL，默认为官方地址。可用于：
  - 配置代理服务器
  - 使用兼容OpenAI API的其他服务（如本地部署的模型服务）
- **TEMPERATURE**: 温度参数，固定为0以确保评估结果的可重复性
- **DEFAULT_RUNS**: 默认运行次数，用于多次运行取平均值

## 使用方法

使用 `uv run` 运行程序：

### 评估单个文档

```bash
uv run main.py --baseline baseline.md --target target1.md
```

### 批量评估多个文档

```bash
uv run main.py --baseline baseline.md --targets target1.md target2.md target3.md
```

### 多次运行取平均（提高可重复性）

```bash
uv run main.py --baseline baseline.md --target target1.md --runs 3
```

### 指定输出格式

```bash
# JSON格式
uv run main.py --baseline baseline.md --target target1.md --output json

# CSV格式
uv run main.py --baseline baseline.md --target target1.md --output csv

# Markdown格式（默认）
uv run main.py --baseline baseline.md --target target1.md --output markdown

# 所有格式
uv run main.py --baseline baseline.md --target target1.md --output all
```

### 指定输出目录

```bash
uv run main.py --baseline baseline.md --target target1.md --output-dir results
```

## 评分说明

- **完整性分数**：覆盖的要点数 / 总要点数 × 100
- **准确性分数**：所有要点准确性得分之和 / 总要点数 × 100
- **综合分数**：完整性 × 0.5 + 准确性 × 0.5

## 输出格式

- **JSON**: 包含每个要点的详细评估结果、总分、子分数
- **CSV**: 文档名、完整性分数、准确性分数、综合分数（批量评估时生成汇总CSV）
- **Markdown**: 包含要点清单、逐条评估结果、总结报告

所有输出文件默认保存在 `output/` 目录下。

## 可重复性保证

为确保评估结果的可重复性，系统采用以下策略：

1. **固定温度参数**：temperature=0，确保模型输出稳定
2. **结构化Prompt**：使用标准化的prompt模板，减少随机性
3. **多次运行取平均**：默认运行3次，取平均值作为最终结果（可通过 `--runs` 参数调整）

## 项目结构

```
srs-eval/
├── main.py                 # 主程序入口
├── pyproject.toml          # 项目配置（uv使用）
├── src/
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── document_parser.py # 文档解析
│   ├── point_extractor.py # 要点提取
│   ├── evaluator.py       # 评估核心逻辑
│   └── output_formatter.py # 输出格式化
├── prompts/
│   ├── extract_points.txt  # 要点提取prompt模板
│   └── evaluate_points.txt # 要点评估prompt模板
└── output/                 # 输出目录（自动创建）
```

