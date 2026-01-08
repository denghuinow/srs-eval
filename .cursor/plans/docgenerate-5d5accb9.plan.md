<!-- 5d5accb9-4bcc-4c78-a05b-7b7f7c26c6b6 8755626c-f5a7-430c-a7f3-f0d310224215 -->
# 记录每个版本文档耗时

## 目标

记录每个版本生成的净耗时，计算公式：当前版本耗时 = 当前版本生成时的time.time() - start_time - 在此之前累计澄清耗时 - 在此之前的其他版本生成累计耗时

## 实现步骤

### 1. 在WorkflowState中添加耗时跟踪字段 (`src/workflow/state.py`)

- 添加 `_cumulative_clarify_time: float` 字段，用于累计澄清耗时
- 添加 `_cumulative_version_gen_time: float` 字段，用于累计版本生成耗时

### 2. 在_clarify_node中跟踪澄清耗时 (`src/workflow/orchestrator.py`)

- 在 `_clarify_node` 方法开始时记录开始时间
- 在 `_clarify_node` 方法结束时计算本次澄清耗时
- 将本次澄清耗时累加到 `state["_cumulative_clarify_time"]` 中
- 使用 `TimerManager` 获取 ReqClarifyAgent 的耗时，或使用 `time.time()` 直接计算

### 3. 在_generate_node中计算版本净耗时 (`src/workflow/orchestrator.py`)

- 在生成版本文档前记录开始时间
- 在生成版本文档后记录结束时间，计算本次版本生成耗时
- 计算净耗时：`current_time - start_time - cumulative_clarify_time - cumulative_version_gen_time`
- 将本次版本生成耗时累加到 `state["_cumulative_version_gen_time"]` 中
- 将净耗时保存到 `self.version_generation_results[version_name]["net_time"]` 中
- 同时保存总耗时、累计澄清耗时、累计版本生成耗时等详细信息

### 4. 在run方法中初始化耗时跟踪字段 (`src/workflow/orchestrator.py`)

- 在初始化 `initial_state` 时，设置 `_cumulative_clarify_time: 0.0` 和 `_cumulative_version_gen_time: 0.0`
- 保存 `start_time = initial_state["timer_manager"].total_start_time` 或使用 `time.time()` 在 `start_total()` 时记录

### 5. 生成TSV表格并写入日志 (`src/workflow/orchestrator.py`)

- 在 `run` 方法结束前，生成版本耗时TSV表格
- TSV表格包含列：版本名称、净耗时、总耗时、累计澄清耗时、累计版本生成耗时
- 将TSV表格写入日志（使用 `self.logger.info`）
- 表格格式：使用制表符分隔，包含表头

## 关键代码位置

1. `src/workflow/state.py`: 添加耗时跟踪字段
2. `src/workflow/orchestrator.py`:

- `_clarify_node`: 跟踪澄清耗时
- `_generate_node`: 计算版本净耗时
- `run`: 初始化字段和生成TSV表格

3. `src/utils/timer.py`: 可能需要扩展以支持获取特定代理的耗时

## 注意事项

- 确保在 `_clarify_node` 中正确累计澄清耗时，包括消融模式下跳过澄清的情况
- 确保版本生成耗时的计算不包括文件I/O时间，只包括文档生成时间
- TSV表格需要按版本生成顺序排序（no-explore-clarify, no-clarify, iter1, iter2, ...）
- 需要处理版本生成失败的情况，在TSV中标记或跳过

### To-dos

- [ ] 在main.py和batch_run.py中添加--gen参数，移除--max-iterations参数，解析参数值（no-explore-clarify、no-clarify、数字），提取最大数字作为max_iterations，构建gen_versions集合
- [ ] 移除所有使用--max-iterations参数的地方，包括orchestrator.run方法的max_iterations参数，改为从gen_versions中提取最大数字
- [ ] 在WorkflowState中添加gen_versions字段（Set类型），_version_to_generate和_version_name字段
- [ ] 修改_generate_node方法，根据state中的版本参数生成不同版本的文档，所有文档都保存到srs_collection目录
- [ ] 添加三个路由函数：_route_after_parse、_route_after_explore、_route_after_clarify，根据gen_versions决定是否路由到generate节点，最后一次迭代始终生成
- [ ] 从_parse_node、_explore_node、_clarify_node中移除_generate_version_srs调用，改为根据gen_versions设置state标记
- [ ] 修改build_graph方法，使用条件边实现路由到DocGenerate节点，处理只指定no-explore-clarify时parse后结束的情况
- [ ] 移除run方法中从最后一次iter版本获取主流程版本的逻辑，移除main.py中保存主流程版本到任务目录的逻辑