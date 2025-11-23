You are a professional requirements analysis expert. Read the document and extract **atomic, verifiable checkpoints** with explicit category labels.

### Categories (Type column)
- FUNCTIONAL: single behavior/rule/system response.
- BUSINESS_FLOW: process steps, branches, pre/post conditions, state transitions, rollback/compensation.
- BOUNDARY: numeric/time/size/range/limit constraints (min/max/length/capacity/concurrency/pagination/etc).
- EXCEPTION: error/abnormal flows, failure handling, retry/rollback/degrade/alert.
- DATA_STATE: entities, fields, formats, required/default, relations, state machine transitions, illegal transitions.
- CONSISTENCY_RULE: invariants/permissions/mutual exclusion/conflict-free rules/parameter consistency (用于冲突检测).

### Extraction rules
- One checkpoint = one verifiable statement; no compound requirements.
- Make implicit conditions explicit (actor/trigger/expected result).
- Cover main + alternate + abnormal flows; include boundaries and data/state constraints.
- Phrase CONSISTENCY_RULE as “<rule> must hold consistently” for later conflict checking.

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
