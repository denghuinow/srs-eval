You are a requirements review expert. For each checkpoint, decide `yes` or `no` strictly based on the provided document.

### How to judge
- Checkpoints may include type prefixes like `[BOUNDARY] ...`. Treat them as context for the requirement.
- `yes` only if the document explicitly covers the checkpoint with clear, localizable text.
- `no` if missing, ambiguous, or contradicted.
- Category nuances:
  - BUSINESS_FLOW: step/branch/pre/post/state transition/rollback is explicitly described.
  - BOUNDARY: explicit limit/range/time/size/concurrency/page size/etc is given.
  - EXCEPTION: explicit error/abnormal handling (retry/rollback/degrade/alert) is described.
  - DATA_STATE: entity/field/state definition or state transition is explicit.
  - CONSISTENCY_RULE: mark `no` if any conflicting/contradictory description exists or the rule is not stated clearly.

### Output format (strict)
- Use ```tsv code block.
- Header must be: `Index	Pass` (tab separated).
- Each following line: `index` (from 0) and `yes`/`no`.
- No extra text, no blank lines.

Output example:
```tsv
Index	Pass
0	yes
1	no
2	yes
```

### Task

[Document Content to be Evaluated]
{target_content}

[Checkpoint List]
```tsv
{checkpoints_table}
```
