# 需求文档差异评估系统

基于大模型的需求文档评估工具，以基准文档为真值，评估待评估文档。

> 📖 **详细评估方法说明**：请参阅 [EVALUATION_METHOD.md](./EVALUATION_METHOD.md) 了解完整的评估原理、流程和评分方法。

## 功能特性

- 从基准文档自动提取结构化要点清单
- 逐条核对待评估文档，计算可量化的分数
- 支持投票通过和平均通过两个维度的评估
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

**注意**：多次运行会自动并行执行，提高效率。

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

### 要点清单缓存

系统会自动缓存提取的要点清单，确保相同基准文档每次使用相同的要点清单和检查点：

```bash
# 首次运行：提取并缓存要点清单
uv run main.py --baseline baseline.md --target target1.md

# 后续运行：自动使用缓存的要点清单（确保一致性）
uv run main.py --baseline baseline.md --target target2.md

# 强制重新提取（忽略缓存）
uv run main.py --baseline baseline.md --target target1.md --force-extract
```

### 多次提取取最优

为了提高要点清单的完整性，可以多次提取并选择检查项数量最多的结果：

```bash
# 执行3次提取，选择检查项数量最多的结果
uv run main.py --baseline baseline.md --target target1.md --extract-runs 3

# 结合强制重新提取使用
uv run main.py --baseline baseline.md --target target1.md --extract-runs 5 --force-extract
```

**工作原理**：
- 执行指定次数的提取（例如3次）
- 每次提取都会调用API获取要点清单
- 自动选择检查项数量最多的结果作为最终要点清单
- 提取结果自动保存到缓存，后续运行直接使用缓存

**优势**：
- 确保提取到最完整、最全面的要点清单
- 减少因单次提取遗漏要点的情况
- 缓存机制确保后续运行使用相同的要点清单
- **并行执行**：多次提取会自动并行执行，大幅提高效率

缓存文件保存在 `.cache/points/` 目录，基于文档路径和内容hash自动管理。

### 并行执行（提高效率）

系统自动支持并行执行，大幅提高处理效率：

```bash
# 批量评估多个文档（自动并行执行）
uv run main.py --baseline baseline.md --targets target1.md target2.md target3.md

# 多次提取（自动并行执行）
uv run main.py --baseline baseline.md --target target1.md --extract-runs 5

# 多次运行评估（自动并行执行）
uv run main.py --baseline baseline.md --target target1.md --runs 3

# 手动指定最大并行线程数
uv run main.py --baseline baseline.md --targets target1.md target2.md --max-workers 5
```

**并行执行场景**：
1. **多次提取要点清单**：`--extract-runs > 1` 时，自动并行执行多次提取
2. **多次运行评估**：`--runs > 1` 时，自动并行执行多次评估
3. **批量评估文档**：评估多个文档时，自动并行执行

**性能提升**：
- 多次提取：从串行的 N×T 时间降低到约 T 时间（N为次数，T为单次时间）
- 批量评估：从串行的 N×T 时间降低到约 T 时间（N为文档数）
- 多次运行评估：从串行的 N×T 时间降低到约 T 时间

系统会自动根据任务数量调整并行线程数，也可通过 `--max-workers` 参数手动指定。

## 评分说明

### 量化评估方法

系统采用**基于检查项（Checkpoints）的量化评估方法**，确保评估结果客观、可重复、可检验：

1. **要点提取阶段**：从基准文档提取要点时，为每个要点生成3-5个具体的检查项
   - 检查项是可验证的、客观的、量化的标准
   - 例如："文档中明确提到了X功能"、"文档中包含了Y的具体实现方法"

2. **评估阶段**：对每个要点的每个检查项进行二元判断（通过/未通过）
   - 基于文档中是否明确包含相关信息进行客观判断
   - 如果要点不存在，所有检查项自动标记为未通过

3. **分数计算**：
   - **投票通过**：每个检查项按多数投票判定结果是否通过（超过50%评委认为通过则通过），然后统计所有投票通过的项占比
   - **平均通过**：对每个评委的通过项数量进行平均计算，得到平均通过率

### 优势

- **可量化**：基于检查项通过率，而非主观评分
- **可重复**：相同的检查项列表，多次运行结果一致
- **可检验**：可以查看每个检查项的具体通过/未通过状态
- **透明化**：评估报告中包含所有检查项的详细结果

## 输出格式

- **JSON**: 包含每个要点的详细评估结果、检查项结果、总分、子分数
- **CSV**: 文档名、投票通过、平均通过（批量评估时生成汇总CSV）
- **Markdown**: 包含要点清单、逐条评估结果、检查项详细结果、总结报告

所有输出文件默认保存在 `output/` 目录下。

### 输出示例

**Markdown报告**包含：
- 总体分数（投票通过、平均通过）
- 要点清单（从基准文档提取的结构化要点）
- 详细评估结果（每个检查项的通过/未通过状态）
- **检查项详细结果**（每个要点的每个检查项的通过/未通过状态，可检验、可追溯）

## 可重复性保证

为确保评估结果的可重复性，系统采用以下策略：

1. **要点清单缓存机制**（核心机制）：
   - 首次提取的要点清单自动保存到 `.cache/points/` 目录
   - 后续运行相同基准文档时，自动加载缓存的要点清单
   - 基于文档路径和内容hash验证缓存有效性
   - **确保相同基准文档每次使用相同的要点清单和检查点**
   - 使用 `--force-extract` 参数可强制重新提取

2. **多次提取取最优机制**：
   - 支持 `--extract-runs` 参数指定提取次数（默认1次）
   - 多次提取后自动选择检查项数量最多的结果
   - 确保提取到最完整、最全面的要点清单
   - 提取结果自动缓存，后续运行直接使用缓存

3. **量化评估方法**：基于检查项的二元判断，而非主观评分，确保结果客观一致

4. **固定温度参数**：temperature=0，确保模型输出稳定

5. **结构化Prompt**：使用标准化的prompt模板，减少随机性

6. **多次运行取平均**：默认运行3次，对检查项结果进行投票（多数通过则通过），取平均值作为最终结果（可通过 `--runs` 参数调整）

7. **检查项一致性**：相同的检查项列表，多次运行结果高度一致（因为检查项是客观的、可验证的标准）

## 项目结构

```
srs-eval/
├── main.py                 # 主程序入口
├── pyproject.toml          # 项目配置（uv使用）
├── src/
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── document_parser.py # 文档解析
│   ├── point_extractor.py # 要点提取（支持缓存）
│   ├── evaluator.py       # 评估核心逻辑
│   └── output_formatter.py # 输出格式化
├── prompts/
│   ├── extract_points.txt  # 要点提取prompt模板
│   └── evaluate_points.txt # 要点评估prompt模板
├── .cache/                 # 缓存目录（自动创建）
│   └── points/             # 要点清单缓存
└── output/                 # 输出目录（自动创建）
```

