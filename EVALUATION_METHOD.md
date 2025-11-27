# SRS评估方法详细说明

## 目录

1. [评估概述](#评估概述)
2. [评估原理](#评估原理)
3. [评估流程](#评估流程)
4. [提示词模板](#提示词模板)
5. [评分方法](#评分方法)
6. [检查项分类](#检查项分类)
7. [可重复性保证](#可重复性保证)
8. [输出结果说明](#输出结果说明)
9. [评估示例](#评估示例)

---

## 评估概述

本系统采用**基于检查项（Checkpoints）的量化评估方法**，以基准SRS文档为真值，评估待评估文档与基准文档的符合程度。

### 核心特点

- **客观量化**：基于检查项通过率，而非主观评分
- **可重复**：相同的检查项列表，多次运行结果一致
- **可检验**：可以查看每个检查项的具体通过/未通过状态
- **透明化**：评估报告中包含所有检查项的详细结果
- **多维评估**：从功能、业务流程、边界条件、异常处理、数据状态、一致性等多个维度进行评估

---

## 评估原理

### 基本思想

评估过程分为三个阶段：

1. **要点提取阶段**：从基准文档中提取结构化的检查项清单
2. **评估阶段**：对每个检查项进行二元判断（通过/未通过）
3. **分数计算阶段**：基于检查项通过率计算综合得分

### 评估逻辑

```
基准SRS文档
    ↓
[要点提取] → 检查项清单（Checkpoints）
    ↓
[逐项评估] → 检查项结果（通过/未通过）
    ↓
[分数计算] → 综合得分（加权得分）
```

---

## 评估流程

### 阶段1：要点提取（Point Extraction）

#### 1.1 提取过程

使用大语言模型（如 GPT-4、DeepSeek 等）分析基准SRS文档，提取原子化、可验证的检查项。

#### 1.2 检查项特征

- **原子化**：每个检查项对应一个可验证的陈述
- **可验证**：能够明确判断文档中是否包含相关信息
- **客观性**：不依赖主观判断，基于文档中是否存在明确描述

#### 1.3 提取示例

**基准文档内容**：
> 系统必须支持用户登录功能，包括用户名密码验证和双因素认证。

**提取的检查项**：
```
[FUNCTIONAL] 系统必须支持用户登录功能
[FUNCTIONAL] 登录功能必须包含用户名密码验证
[FUNCTIONAL] 登录功能必须支持双因素认证
```

#### 1.4 缓存机制

- 首次提取的检查项清单自动保存到 `.cache/points/` 目录
- 基于文档路径和内容hash验证缓存有效性
- 确保相同基准文档每次使用相同的检查项清单

### 阶段2：评估执行（Evaluation）

#### 2.1 评估过程

1. 将检查项清单和待评估文档内容发送给大语言模型
2. 模型逐项判断：文档中是否明确包含该检查项的相关内容
3. 输出二元判断结果：`yes`（通过）或 `no`（未通过）

#### 2.2 判断标准

- **通过（yes）**：文档中明确包含与检查项相关的信息
- **未通过（no）**：文档中未找到与检查项相关的明确描述

#### 2.3 多评委机制

- 默认使用3个评委（可通过 `--judges` 参数调整）
- 每个评委独立评估所有检查项
- 使用多数投票机制合并结果（超过50%评委认为通过则通过）

#### 2.4 评估示例

**检查项**：`[FUNCTIONAL] 系统必须支持用户登录功能`

**待评估文档内容**：
> 系统提供用户认证功能，用户可以通过输入用户名和密码进行登录。

**评估结果**：
- 评委1：✓ 通过（文档明确提到"用户可以通过输入用户名和密码进行登录"）
- 评委2：✓ 通过（文档提到"用户认证功能"和"登录"）
- 评委3：✗ 未通过（文档提到"认证"但未明确说明是"登录功能"）

**多数投票结果**：✓ 通过（2/3评委认为通过）

### 阶段3：分数计算（Scoring）

#### 3.1 维度得分

对每个维度的检查项计算通过率：

```
维度得分 = (该维度通过的检查项数 / 该维度总检查项数) × 100
```

#### 3.2 加权得分

采用加权计算方式，对不同维度给予不同权重：

| 维度 | 权重 |
|------|------|
| 功能覆盖 / 行为规则 (FUNCTIONAL) | 25% |
| 业务流程完整性 (BUSINESS_FLOW) | 15% |
| 边界条件完整性 (BOUNDARY) | 10% |
| 异常处理覆盖度 (EXCEPTION) | 20% |
| 数据与状态完整性 (DATA_STATE) | 20% |
| 一致性 / 冲突检测 (CONSISTENCY_RULE) | 10% |

**计算公式**：
```
加权得分 = min(100.0, Σ(维度得分 × 维度权重) × 2.0)
```

**计算步骤说明**：

1. **第一步：计算加权平均**
   ```
   加权平均 = Σ(维度得分 × 维度权重)
   ```
   - 每个维度得分范围：0-100（百分比）
   - 权重总和：1.0
   - 加权平均结果范围：0-100

2. **第二步：乘以缩放系数2.0**
   ```
   加权得分 = 加权平均 × 2.0
   ```
   - 将分数区间从15-20提升到30-40（基准文档）
   - 更好的文档可达60+分
   - 提高分数区分度

3. **第三步：应用上限**
   ```
   最终得分 = min(100.0, 加权得分)
   ```
   - 确保最高分不超过100分

**为什么要乘以2.0？**

根据实际评估数据，如果不乘以2.0：
- 基准文档的加权平均通常在15-20分左右
- 分数偏低，区分度不足
- 难以直观理解文档质量

乘以2.0后：
- 基准文档分数提升到30-40分
- 较好文档可达50-60分
- 优秀文档可达80-100分
- 分数分布更符合百分制的直观理解
- 提高不同质量文档之间的区分度

**示例计算**：

假设各维度得分：
- FUNCTIONAL: 60% × 0.25 = 15.0
- BUSINESS_FLOW: 50% × 0.15 = 7.5
- BOUNDARY: 40% × 0.10 = 4.0
- EXCEPTION: 55% × 0.20 = 11.0
- DATA_STATE: 50% × 0.20 = 10.0
- CONSISTENCY_RULE: 45% × 0.10 = 4.5

加权平均 = 15.0 + 7.5 + 4.0 + 11.0 + 10.0 + 4.5 = **52.0**

加权得分 = 52.0 × 2.0 = **104.0**

最终得分 = min(100.0, 104.0) = **100.0**

#### 3.3 其他指标

- **投票通过率**：每个检查项按多数投票判定，统计所有投票通过的项占比
- **平均通过率**：对每个评委的通过项数量进行平均计算

---

## 提示词模板

系统使用两个核心提示词模板来实现评估流程：

### 1. 检查项提取提示词（extract_points.md）

用于从基准文档中提取结构化的检查项清单。

**提示词内容**：

~~~markdown
You are a professional requirements analysis expert. Read the document and extract **atomic, verifiable checkpoints** with explicit category labels.

### Categories (Type column)
- FUNCTIONAL: single behavior/rule/system response.
- BUSINESS_FLOW: process steps, branches, pre/post conditions, state transitions, rollback/compensation.
- BOUNDARY: numeric/time/size/range/limit constraints (min/max/length/capacity/concurrency/pagination/etc).
- EXCEPTION: error/abnormal flows, failure handling, retry/rollback/degrade/alert.
- DATA_STATE: entities, fields, formats, required/default, relations, state machine transitions, illegal transitions.
- CONSISTENCY_RULE: invariants/permissions/mutual exclusion/conflict-free rules/parameter consistency (for conflict checks).

### Extraction rules
- One checkpoint = one verifiable statement; avoid compound requirements.
- Make implicit conditions explicit (actor/trigger/expected result).
- Cover main + alternate + abnormal flows; include boundaries and data/state constraints.
- Phrase CONSISTENCY_RULE as "<rule> must hold consistently" for later conflict checking.

### Output format (strict)
- Use TSV inside ```tsv code block.
- Header: `Type	Checkpoint`
- One checkpoint per line, **no numbering or extra text**.

Example:
```tsv
Type	Checkpoint
FUNCTIONAL	Support user login with credential validation
BUSINESS_FLOW	Checkout flow requires address selection before payment
BOUNDARY	Page size must not exceed 100 items
EXCEPTION	On payment gateway timeout, trigger retry then degrade to manual confirmation
DATA_STATE	Order states: created -> paid -> shipped -> completed; cancel allowed only before shipped
CONSISTENCY_RULE	Login requirement must be consistent across all access paths
```

Document content:
{document_content}

Please output only the TSV block.
~~~

**关键特点**：
- 要求提取原子化、可验证的检查项
- 明确6个维度的分类标准
- 严格规定输出格式为TSV表格
- 每个检查项必须包含类别标签

### 2. 检查项评估提示词（evaluate_points.md）

用于评估待评估文档是否包含检查项清单中的内容。

**提示词内容**：

~~~markdown
You are a professional requirements document evaluation expert. Your task is: Based on the given **checkpoint list**, verify each item against the **document content to be evaluated**, determine whether the document explicitly contains these checkpoints, and output structured results.

### Evaluation Principles

1. **Item-by-item verification**: Every checkpoint in the checkpoint list must be evaluated.
2. **Objective and verifiable**: Judgments must be strictly based on whether there are **explicit, locatable relevant statements** in the "document content to be evaluated", and must not be subjectively inferred or supplemented.
3. **Complete coverage**: All checkpoints must appear in the output results, regardless of whether relevant content is found in the document.
4. **Index numbering**: Index numbers in the output start from `0` and increment sequentially according to the order of checkpoints in the list.
5. **Judgment rationale**: Must briefly explain, **must explicitly mention the specific content of the checkpoint**, cannot use vague expressions like "this checkpoint", "xxx checkpoint":
   
   * If found: Explicitly state the specific content of the checkpoint and explain the relevant information found in the document (you can summarize the location or key points of the content).
   * If not found: Explicitly state the specific content of the checkpoint and explain "No explicit description directly related to [specific checkpoint content] was found in the document".
   
   **Examples**:
   - Wrong: `No explicit description directly related to this checkpoint was found in the document`
   - Correct: `No explicit description directly related to "support user login functionality" was found in the document`
   - Correct: `No explicit description supporting user login functionality was found in the document`
6. **Binary judgment**:
   
   * Use `yes` to indicate that explicit content corresponding to the checkpoint was found in the document;
   * Use `no` to indicate that corresponding content was not found.
7. **Information source limitation**: Only use the content provided in `document content to be evaluated` for judgment, and do not reference external common knowledge or experience.

### Output Format Requirements (Important! Must strictly comply)

**WARNING: Format requirements are mandatory, any deviation will cause parsing to fail!**

1. **Must wrap TSV output in code blocks**, format as follows:
   
   ```tsv
   Index	Reason	Result
   0	The document explicitly describes content related to "support user registration functionality" including form validation and user information storage	yes
   1	No explicit description directly related to "support user login functionality" was found in the document	no
   ```

2. **TSV format requirements**:
   - Use **tab character** as column separator (NOT comma, NOT space)
   - **The first line must be the header**, and must be exactly (must be completely consistent, no variations): `Index	Reason	Result` (tabs between columns)
   - **Starting from the second line, each line must contain three columns**, separated by tabs, corresponding to one checkpoint in the checkpoint list

3. **Column requirements**:
   * **First column - Index**: Integer starting from `0`, incrementing sequentially according to checkpoint order (0, 1, 2, 3, ...).
   * **Second column - Reason**: Brief explanation in English of findings in the document, or reason for not finding. **You can freely use commas, semicolons, and other punctuation in this column** - tabs are only used as column separators.
   * **Third column - Result**: **Must** be `yes` or `no` (lowercase, no quotes, no spaces). **This column must absolutely not be omitted, cannot be empty, and cannot be other values!**

4. **Format example** (please strictly follow this format for output, note that the reason must explicitly mention the specific content of the checkpoint):
   
   ```tsv
   Index	Reason	Result
   0	The document explicitly describes content related to "support user registration functionality", including form validation and user information storage	yes
   1	No explicit description directly related to "support user login functionality" was found in the document	no
   2	The document explicitly describes API functions like `coem_send_data` and `coem_receive_data` for data transmission	yes
   ```

5. **Key requirements**:
   - **Must wrap the entire TSV content in ```tsv code blocks**
   - Each line must have three columns, separated by **tab characters** (NOT commas)
   - The third column must be `yes` or `no`, cannot be omitted
   - Must cover all checkpoints, no omissions or merging allowed
   - Index must start from 0 and increment continuously
   - **The reason must explicitly mention the specific content of the checkpoint**, cannot use vague expressions like "this checkpoint", "xxx checkpoint"
   - **Reason column can contain commas, semicolons, and any punctuation** - only tabs are used as separators
   - Do not repeat words like "Reason" at the end of the reason
   - Do not use commas as column separators
   - Do not omit the Result column
   - Do not use vague expressions like "this checkpoint", "with xxx checkpoint" in the output
   - The output must not contain any additional text explanations, blank lines, or comments outside the code blocks

6. **Pre-output check**: Before outputting, please confirm:
   - Use ```tsv code blocks to wrap
   - The first line is the header: `Index	Reason	Result` (with tabs)
   - Each line has three columns, separated by tabs (NOT commas)
   - The third column is `yes` or `no`
   - Index starts from 0 and increments continuously
   - **The reason explicitly mentions the specific content of the checkpoint**, no vague expressions like "this checkpoint" are used
   - No blank lines, no additional explanatory text

### Task

Please evaluate the given checkpoint list and document content to be evaluated according to the above requirements, **use ```tsv code blocks to wrap the final TSV table content**, and do not output any other text explanations or comments.

[Checkpoint List]
```tsv
Index	Checkpoint
{checkpoints_table}
```

[Document Content to be Evaluated]
{target_content}
~~~

**关键特点**：
- 强调客观性和可验证性，必须基于文档中的明确描述
- 要求逐项评估，不能遗漏任何检查项
- 判断理由必须明确提及检查项的具体内容，不能使用模糊表达
- 严格规定输出格式为TSV表格，包含Index、Reason、Result三列
- 使用二元判断（yes/no），不允许中间值

### 提示词设计原则

1. **明确性**：提示词明确规定了任务、输出格式和要求
2. **客观性**：强调基于文档内容进行客观判断，不允许主观推断
3. **结构化**：要求输出结构化数据（TSV格式），便于程序解析
4. **完整性**：要求覆盖所有检查项，不能遗漏
5. **可追溯性**：要求提供判断理由，便于验证和追溯

### 提示词版本管理

系统支持多个版本的提示词，默认使用 `v1` 版本。可以通过 `--prompt-version` 参数指定使用其他版本的提示词。

提示词文件位置：`prompts/{version}/extract_points.md` 和 `prompts/{version}/evaluate_points.md`

---

## 评分方法

### 投票通过（Voting Pass）

**定义**：每个检查项按多数投票判定结果是否通过（超过50%评委认为通过则通过），然后统计所有投票通过的项占比。

**计算公式**：
```
投票通过率 = (多数投票通过的检查项数 / 总检查项数) × 100
```

**示例**：
- 总检查项数：100
- 多数投票通过的检查项数：65
- 投票通过率 = 65/100 × 100 = 65%

### 平均通过（Average Pass）

**定义**：对每个评委的通过项数量进行平均计算，得到平均通过率。

**计算公式**：
```
平均通过率 = (Σ(每个评委的通过项数) / 评委数) / 总检查项数 × 100
```

**示例**：
- 总检查项数：100
- 评委1通过项数：70
- 评委2通过项数：65
- 评委3通过项数：60
- 平均通过率 = ((70 + 65 + 60) / 3) / 100 × 100 = 65%

### 加权得分（Weighted Score）

**定义**：综合得分指标，采用加权计算方式，对业务流程完整性、异常处理覆盖度、数据与状态完整性、一致性/冲突检测等维度给予更高权重。

**特点**：
- 得分范围：0-100
- 越高表示目标文档越接近基准SRS
- 是主要的评估指标

---

## 检查项分类

系统将检查项分为6个维度：

### 1. 功能覆盖 / 行为规则 (FUNCTIONAL)

评估文档是否覆盖了基准SRS中定义的功能需求和行为规则。

**示例检查项**：
- `[FUNCTIONAL] 系统必须支持用户注册功能`
- `[FUNCTIONAL] 系统必须提供数据导出功能`
- `[FUNCTIONAL] 系统必须支持多语言界面`

### 2. 业务流程完整性 (BUSINESS_FLOW)

评估文档是否完整描述了业务流程和操作序列。

**示例检查项**：
- `[BUSINESS_FLOW] 用户注册流程必须包含邮箱验证步骤`
- `[BUSINESS_FLOW] 订单处理流程必须包含支付确认环节`
- `[BUSINESS_FLOW] 数据备份流程必须包含完整性校验`

### 3. 边界条件完整性 (BOUNDARY)

评估文档是否明确描述了系统边界、输入输出边界、性能边界等边界条件。

**示例检查项**：
- `[BOUNDARY] 单次查询返回结果数不得超过100条`
- `[BOUNDARY] 系统响应时间不得超过3秒`
- `[BOUNDARY] 文件上传大小限制为10MB`

### 4. 异常处理覆盖度 (EXCEPTION)

评估文档是否充分描述了异常情况和错误处理机制。

**示例检查项**：
- `[EXCEPTION] 网络连接失败时系统必须提供重试机制`
- `[EXCEPTION] 用户输入无效数据时系统必须显示明确的错误提示`
- `[EXCEPTION] 数据库连接超时时系统必须记录错误日志`

### 5. 数据与状态完整性 (DATA_STATE)

评估文档是否完整描述了数据模型、状态转换、数据约束等。

**示例检查项**：
- `[DATA_STATE] 用户状态必须包含：未激活、已激活、已禁用`
- `[DATA_STATE] 订单状态转换：待支付 -> 已支付 -> 已发货 -> 已完成`
- `[DATA_STATE] 用户密码必须满足至少8位字符的要求`

### 6. 一致性 / 冲突检测 (CONSISTENCY_RULE)

评估文档内部是否存在冲突、矛盾或不一致的地方。

**示例检查项**：
- `[CONSISTENCY_RULE] 用户权限规则必须一致地应用于所有访问路径`
- `[CONSISTENCY_RULE] 数据格式定义必须与接口规范保持一致`
- `[CONSISTENCY_RULE] 错误码定义必须在整个系统中保持一致`

---

## 可重复性保证

为确保评估结果的可重复性，系统采用以下策略：

### 1. 要点清单缓存机制

- 首次提取的要点清单自动保存到 `.cache/points/` 目录
- 基于文档路径和内容hash验证缓存有效性
- 确保相同基准文档每次使用相同的要点清单和检查点
- 使用 `--force-extract` 参数可强制重新提取

### 2. 固定温度参数

- 默认 `temperature=0`，确保模型输出稳定
- 减少随机性，提高结果一致性

### 3. 结构化Prompt

- 使用标准化的prompt模板
- 减少因prompt表述差异带来的影响

### 4. 多次运行取平均

- 默认运行3次（可通过 `--judges` 参数调整）
- 对检查项结果进行投票（多数通过则通过）
- 取平均值作为最终结果

### 5. 检查项一致性

- 相同的检查项列表，多次运行结果高度一致
- 因为检查项是客观的、可验证的标准

---

## 输出结果说明

### JSON格式

包含完整的评估数据：

```json
{
  "target_document": "document.md",
  "scores": {
    "voting_score": 65.00,
    "average_score": 65.50,
    "weighted_score": 72.35
  },
  "checkpoints": [
    "[FUNCTIONAL] 系统必须支持用户登录功能",
    ...
  ],
  "checkpoint_results": [
    {
      "checkpoint": "系统必须支持用户登录功能",
      "category": "FUNCTIONAL",
      "passed": true,
      "pass_rate": 1.0
    },
    ...
  ],
  "all_judge_results": [
    [
      {"checkpoint": "...", "passed": true},
      ...
    ],
    ...
  ]
}
```

### Markdown格式

包含：
- 评估信息（时间、模型、基准文档等）
- 统计信息（检查项总数）
- 加权得分
- 维度得分（各维度的通过率和得分）
- 检查项详细结果（每个检查项的通过/未通过状态）

### TSV格式

包含每个检查项的详细评估结果，格式为：
```
Index	Checkpoint	Judge1	Judge2	Judge3	Majority
0	[FUNCTIONAL] ...	✓	✓	✗	✓
1	[FUNCTIONAL] ...	✗	✗	✗	✗
...
```

### 跨阶段对比报告

包含：
- 各阶段的平均加权得分
- 各阶段的平均投票通过率
- 各阶段的平均通过率
- 各维度的平均得分对比
- 文档得分趋势分析

---

## 评估示例

### 示例1：单个文档评估

**输入**：
- 基准文档：`baseline.md`
- 待评估文档：`target.md`

**输出**：
- `target_evaluation.json`：完整评估数据
- `target_evaluation.md`：可读性报告
- `target_evaluation.tsv`：详细检查项结果

**报告内容**：
```
加权得分：72.35

维度得分：
- 功能覆盖：65.00%
- 业务流程：80.00%
- 边界条件：70.00%
- 异常处理：75.00%
- 数据状态：85.00%
- 一致性：60.00%
```

### 示例2：批量评估

**输入**：
- 基准文档目录：`baseline_dir/`
- 待评估文档目录：`target_dir/`

**输出**：
- 每个文档的评估结果（JSON、Markdown、TSV）
- `summary_report.md`：聚合统计报告
- `evaluations_summary.csv`：汇总CSV

**聚合报告内容**：
- 加权得分统计（平均值、中位数、最大值、最小值、标准差）
- 维度平均得分
- 详细评估结果表格
- 分数分布统计

### 示例3：跨阶段对比

**输入**：
- 基准文档目录：`baseline_dir/`
- SRS集合目录：`srs_collection_dir/`（包含多个阶段的文档）

**输出**：
- 每个阶段的评估结果
- `cross_stage_comparison.md`：跨阶段对比报告

**对比报告内容**：
- 各阶段总体得分对比
- 各阶段维度得分对比
- 阶段排名
- 得分趋势分析
- 文档得分对比表格

---

## 评估方法优势

### 1. 客观性

- 基于检查项的二元判断，而非主观评分
- 减少评估者的主观偏见
- 评估结果可验证、可追溯

### 2. 可重复性

- 要点清单缓存机制确保一致性
- 固定温度参数减少随机性
- 多次运行取平均提高稳定性

### 3. 全面性

- 从6个维度全面评估文档质量
- 覆盖功能、流程、边界、异常、数据、一致性等各个方面

### 4. 透明性

- 评估报告中包含所有检查项的详细结果
- 可以查看每个检查项的通过/未通过状态
- 支持多评委机制，显示每个评委的判断

### 5. 可扩展性

- 支持自定义检查项分类
- 支持调整维度权重
- 支持多种输出格式

---

## 注意事项

### 1. 评估基准

- 评估结果依赖于基准文档的质量
- 基准文档应该完整、准确、规范

### 2. 检查项质量

- 检查项应该原子化、可验证
- 避免过于宽泛或过于具体的检查项

### 3. 评委数量

- 建议使用3个或更多评委
- 评委数量越多，结果越稳定，但成本也越高

### 4. 模型选择

- 不同模型的评估结果可能有差异
- 建议使用性能较好的模型（如 GPT-4、DeepSeek 等）

### 5. 文档格式

- 系统支持 Markdown 格式的文档
- 其他格式的文档需要先转换为 Markdown

---

## 总结

本评估方法通过以下方式确保评估的客观性、可重复性和全面性：

1. **结构化提取**：从基准文档中提取原子化、可验证的检查项
2. **客观评估**：基于文档内容进行二元判断，不依赖主观评分
3. **多维分析**：从6个维度全面评估文档质量
4. **加权计算**：对不同维度给予不同权重，突出重要方面
5. **多评委机制**：使用多数投票提高评估的可靠性
6. **缓存机制**：确保相同基准文档使用相同的检查项清单

通过这些机制，系统能够提供客观、可重复、可检验的评估结果，帮助开发者了解待评估文档与基准文档的符合程度。

