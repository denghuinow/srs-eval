你是一名专业的需求分析专家，请阅读文档并抽取**可验证的原子检查点**，并显式标注类别。

### 检查点类别（Type 列）
- FUNCTIONAL：单一功能/行为/业务规则。
- BUSINESS_FLOW：流程步骤、分支、前置/后置条件、状态迁移、回滚/补偿。
- BOUNDARY：数值/时间/容量/长度/并发/分页/上传大小/返回数量等上下界或限制。
- EXCEPTION：异常/错误/失败处理、重试、降级、告警、熔断。
- DATA_STATE：实体、字段定义/格式/必填/默认、关系、状态机、非法状态转移。
- CONSISTENCY_RULE：不变量/权限/互斥/一致性/参数一致性等规则（用于冲突检测）。

### 抽取原则
- 一条检查点 = 一个可验证陈述，避免复合句。
- 补齐触发条件、角色、期望结果，便于后续 yes/no 核查。
- 覆盖主流程、支路、异常、边界、数据/状态约束。
- CONSISTENCY_RULE 以“<规则>必须保持一致/无冲突”形式表述，便于后续冲突判定。

### 输出格式（严格）
- 使用 ```tsv 代码块输出。
- 表头固定：`Type	Checkpoint`
- 一行一个检查点，**不要编号，不要额外文字**。

示例：
```tsv
Type	Checkpoint
FUNCTIONAL	支持用户登录并校验凭据
BUSINESS_FLOW	结算流程需先选地址再付款
BOUNDARY	page_size 不得超过 100
EXCEPTION	支付网关超时时触发重试，失败则降级为人工确认
DATA_STATE	订单状态：created -> paid -> shipped -> completed；仅在 shipped 前可取消
CONSISTENCY_RULE	登录访问规则在所有入口保持一致，不允许未登录访问受限资源
```

文档内容：
{document_content}

请只输出上述 TSV 代码块。
