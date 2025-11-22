# Eirene文件无分类问题分析与修复

## 问题描述

Eirene文件（`2007 - eirene fun 7.pdf_342c1113b382.json`）多次提取后仍然没有Category分类信息，所有128个检查点都是纯文本格式，没有"Category\tCheckpoint"格式。

## 根本原因分析

### 1. 代码逻辑问题

在`point_extractor.py`的`_extract_points_single`方法中：

1. **虽然解析了带Category的TSV**：
   - 代码尝试解析"Category\tCheckpoint"格式的TSV
   - 创建了`checkpoints_with_category`列表存储Category信息

2. **但Category信息丢失**：
   - `_extract_points_single`方法只返回`list[str]`（检查点文本列表）
   - `checkpoints_with_category`信息在返回时丢失

3. **缓存保存时没有Category**：
   - `extract_points`方法调用`_save_points_cache`时，没有传递`checkpoints_with_category`参数
   - 缓存文件中只保存了纯文本检查点

### 2. 问题流程

```
API返回TSV (Category\tCheckpoint)
    ↓
解析TSV，提取checkpoints_with_category
    ↓
只返回checkpoints (list[str]) ← Category信息丢失
    ↓
保存缓存时没有Category信息
    ↓
缓存文件中只有纯文本检查点
```

## 修复方案

### 1. 修改`_extract_points_single`返回值

**修改前**：
```python
def _extract_points_single(...) -> list[str]:
    # ...
    return checkpoints  # 只返回检查点文本
```

**修改后**：
```python
def _extract_points_single(...) -> tuple[list[str], list[tuple[str, str]] | None]:
    # ...
    return checkpoints, checkpoints_with_category  # 返回检查点和Category信息
```

### 2. 修改`extract_points`方法

**修改前**：
```python
checkpoints = self._extract_points_single(...)
self._save_points_cache(document_path, checkpoints, content_hash)
```

**修改后**：
```python
checkpoints, checkpoints_with_category = self._extract_points_single(...)
self._save_points_cache(document_path, checkpoints, content_hash, checkpoints_with_category)
```

### 3. 修改缓存保存格式

**修改前**：
```python
cache_data = {
    "checkpoints": checkpoints,  # 只有纯文本
}
```

**修改后**：
```python
if checkpoints_with_category and self.prompt_version == "v3":
    # 保存为"Category\tCheckpoint"格式
    checkpoints_with_cat_format = [f"{cat}\t{cp}" for cat, cp in checkpoints_with_category]
    cache_data = {
        "checkpoints": checkpoints_with_cat_format,  # 带Category格式
        "checkpoints_with_category": checkpoints_with_category,  # 结构化数据
    }
```

### 4. 修改评估器处理逻辑

在`evaluator.py`的`evaluate_single_run`方法中，添加对"Category\tCheckpoint"格式的解析：

```python
# 检查checkpoints是否是"Category\tCheckpoint"格式
if checkpoints and '\t' in checkpoints[0]:
    parts = checkpoints[0].split('\t', 1)
    if parts[0] in ['FUNCTIONAL', 'BUSINESS_FLOW', ...]:
        # 解析Category信息
        checkpoints_with_category = [...]
```

## 修复后的效果

### 1. 新提取的文件
- ✅ 使用v3提示词提取时，会保存"Category\tCheckpoint"格式
- ✅ 缓存文件中包含Category信息
- ✅ 评估时可以从checkpoints中解析Category

### 2. 已存在的缓存文件
- ⚠️ 旧缓存文件仍然是纯文本格式
- ✅ 需要强制重新提取（使用`--force-extract`参数）

## 使用建议

### 重新提取Eirene文件

```bash
# 使用v3提示词强制重新提取
python main.py extract \
    --baseline "../srs-docs/resources/req_md_sample10/2007 - eirene fun 7.pdf.md" \
    --prompt-version v3 \
    --force-extract
```

### 验证修复效果

提取后检查缓存文件：
```python
import json
with open('.cache/points/2007 - eirene fun 7.pdf_*.json', 'r') as f:
    data = json.load(f)
    checkpoints = data['checkpoints']
    # 检查第一个检查点是否包含Category
    if checkpoints and '\t' in checkpoints[0]:
        print("✅ 包含Category信息")
        print(f"示例: {checkpoints[0]}")
    else:
        print("❌ 仍然无Category信息")
```

## 总结

**问题根源**：代码虽然解析了Category信息，但在返回和保存时丢失了。

**修复方案**：
1. ✅ 修改返回值类型，包含Category信息
2. ✅ 修改缓存保存格式，保存"Category\tCheckpoint"格式
3. ✅ 修改评估器，支持解析"Category\tCheckpoint"格式

**下一步**：使用修复后的代码重新提取Eirene文件，验证Category信息是否正确保存。

