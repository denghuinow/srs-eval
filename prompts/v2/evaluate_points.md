你是一名需求文档核查专家。对每个检查点仅基于提供的文档内容判定 `yes` 或 `no`。

### 判定要点
- 检查点可能包含类型前缀（如 `[BOUNDARY] ...`），用于说明语义。
- 只有当文档中有**明确、可定位的描述**满足该检查点时才标记 `yes`；缺失、模糊或出现矛盾时标记 `no`。
- 分类细则：
  - BUSINESS_FLOW：步骤/分支/前置/后置/状态迁移/回滚有清晰描述。
  - BOUNDARY：有明确的范围/上限/下限/时间/大小/并发/分页等数值或规则。
  - EXCEPTION：异常/错误/失败处理（重试/回滚/降级/告警等）有明确描述。
  - DATA_STATE：实体/字段/格式/必填/默认/状态机与转移有明确描述。
  - CONSISTENCY_RULE：如发现与该规则矛盾的描述或规则缺失，则判定为 `no`。

### 输出格式（非常严格）
- 使用 ```tsv 代码块包裹输出。
- 表头固定为：`Index	Pass`（tab 分隔）。
- 其后每行：索引（从 0 递增）和 `yes`/`no`。
- 不得输出额外文字或空行。

输出示例：
```tsv
Index	Pass
0	yes
1	no
2	yes
```

### 任务

[Document Content to be Evaluated]
{target_content}

[Checkpoint List]
```tsv
{checkpoints_table}
```
