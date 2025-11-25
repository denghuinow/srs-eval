#!/usr/bin/env python3
"""
分析评委一致性情况
"""

import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics

def parse_evaluation_file(md_path: Path) -> List[Dict]:
    """
    解析单个评估文件，提取所有检查项的评委评分
    
    Returns:
        List of dicts, each containing:
        - checkpoint_index: int
        - checkpoint: str
        - judges: List[bool] - 10个评委的评分结果
        - majority_vote: bool - 多数投票结果
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 查找表格开始位置
    table_start = content.find("## 检查项详细结果")
    if table_start == -1:
        return []
    
    table_content = content[table_start:]
    lines = table_content.split('\n')
    
    results = []
    in_table = False
    num_judges = 10  # 默认10个评委
    
    for line in lines:
        line = line.strip()
        if not line or not line.startswith('|'):
            continue
        
        # 检测表头，确定评委数量
        if '多数投票' in line and '评委' in line:
            cols = [c.strip() for c in line.split('|') if c.strip()]
            num_judges = sum(1 for c in cols if '评委' in c)
            in_table = True
            continue
        
        if not in_table:
            continue
        
        # 跳过分隔行
        if line.startswith('|:') or line.startswith('|-'):
            continue
        
        # 解析数据行
        cols = [c.strip() for c in line.split('|') if c.strip()]
        if len(cols) < 3:
            continue
        
        try:
            index = int(cols[0])
            checkpoint_text = cols[1]
            
            # 提取评委评分 (从第3列到第2+num_judges列)
            judge_scores = []
            for i in range(2, 2 + num_judges):
                if i < len(cols):
                    judge_col = cols[i]
                    # ✓ 表示通过, ✗ 表示未通过
                    passed = "✓" in judge_col or "通过" in judge_col
                    judge_scores.append(passed)
                else:
                    judge_scores.append(False)
            
            # 提取多数投票结果
            majority_vote_col = cols[-1] if cols else ""
            majority_passed = "通过" in majority_vote_col or "✓ 通过" in majority_vote_col
            
            results.append({
                "checkpoint_index": index,
                "checkpoint": checkpoint_text,
                "judges": judge_scores,
                "majority_vote": majority_passed
            })
        except (ValueError, IndexError) as e:
            continue
    
    return results

def calculate_consistency_metrics(all_results: List[Dict]) -> Dict:
    """
    计算评委一致性指标
    """
    metrics = {
        "total_checkpoints": len(all_results),
        "total_judges": 10,
        "unanimous_agreement": 0,  # 完全一致
        "majority_agreement": 0,   # 多数一致(>50%)
        "split_votes": 0,          # 分歧(5-5)
        "judge_agreement_rates": {},  # 每个评委与其他评委的一致性
        "judge_pass_rates": {},       # 每个评委的通过率
        "inter_judge_agreement": {},  # 评委两两之间的一致性
        "category_consistency": defaultdict(lambda: {
            "total": 0,
            "unanimous": 0,
            "majority": 0,
            "split": 0
        })
    }
    
    # 统计每个评委的通过率
    judge_pass_counts = defaultdict(int)
    judge_total_counts = defaultdict(int)
    
    # 统计评委两两之间的一致性
    judge_pair_agreement = defaultdict(lambda: {"agree": 0, "total": 0})
    
    for result in all_results:
        judges = result["judges"]
        if len(judges) != 10:
            continue
        
        # 统计通过数
        pass_count = sum(judges)
        
        # 完全一致
        if pass_count == 0 or pass_count == 10:
            metrics["unanimous_agreement"] += 1
        # 多数一致
        elif pass_count > 5:
            metrics["majority_agreement"] += 1
        elif pass_count < 5:
            metrics["majority_agreement"] += 1
        # 分歧(5-5)
        else:
            metrics["split_votes"] += 1
        
        # 统计每个评委的评分
        for i, passed in enumerate(judges):
            judge_total_counts[i] += 1
            if passed:
                judge_pass_counts[i] += 1
        
        # 统计评委两两之间的一致性
        for i in range(10):
            for j in range(i + 1, 10):
                pair_key = f"{i+1}-{j+1}"
                judge_pair_agreement[pair_key]["total"] += 1
                if judges[i] == judges[j]:
                    judge_pair_agreement[pair_key]["agree"] += 1
        
        # 按类别统计
        checkpoint = result["checkpoint"]
        category = "UNKNOWN"
        if checkpoint.startswith('[') and ']' in checkpoint:
            match = re.match(r'\[([^\]]+)\]\s*(.+)', checkpoint)
            if match:
                category = match.group(1)
        
        cat_metrics = metrics["category_consistency"][category]
        cat_metrics["total"] += 1
        if pass_count == 0 or pass_count == 10:
            cat_metrics["unanimous"] += 1
        elif pass_count > 5 or pass_count < 5:
            cat_metrics["majority"] += 1
        else:
            cat_metrics["split"] += 1
    
    # 计算每个评委的通过率
    for i in range(10):
        total = judge_total_counts.get(i, 1)
        passed = judge_pass_counts.get(i, 0)
        metrics["judge_pass_rates"][f"评委{i+1}"] = {
            "pass_rate": round(passed / total * 100, 2) if total > 0 else 0,
            "pass_count": passed,
            "total_count": total
        }
    
    # 计算评委两两之间的一致性
    for pair_key, pair_data in judge_pair_agreement.items():
        total = pair_data["total"]
        agree = pair_data["agree"]
        metrics["inter_judge_agreement"][pair_key] = {
            "agreement_rate": round(agree / total * 100, 2) if total > 0 else 0,
            "agree_count": agree,
            "total_count": total
        }
    
    # 计算每个评委与其他评委的平均一致性
    for i in range(10):
        agreements = []
        for j in range(10):
            if i != j:
                pair_key = f"{min(i+1, j+1)}-{max(i+1, j+1)}"
                if pair_key in metrics["inter_judge_agreement"]:
                    agreements.append(metrics["inter_judge_agreement"][pair_key]["agreement_rate"])
        
        if agreements:
            metrics["judge_agreement_rates"][f"评委{i+1}"] = {
                "avg_agreement": round(statistics.mean(agreements), 2),
                "min_agreement": round(min(agreements), 2),
                "max_agreement": round(max(agreements), 2)
            }
    
    return metrics

def analyze_all_files(directory: Path) -> Dict:
    """
    分析目录下所有评估文件
    """
    all_results = []
    file_stats = {}
    
    # 查找所有评估文件
    eval_files = sorted(directory.glob("*_evaluation.md"))
    
    for eval_file in eval_files:
        print(f"正在分析: {eval_file.name}")
        results = parse_evaluation_file(eval_file)
        all_results.extend(results)
        
        # 统计单个文件的一致性
        file_metrics = calculate_consistency_metrics(results)
        file_stats[eval_file.name] = {
            "checkpoint_count": len(results),
            "unanimous_rate": round(file_metrics["unanimous_agreement"] / max(len(results), 1) * 100, 2),
            "majority_rate": round(file_metrics["majority_agreement"] / max(len(results), 1) * 100, 2),
            "split_rate": round(file_metrics["split_votes"] / max(len(results), 1) * 100, 2)
        }
    
    # 计算总体指标
    overall_metrics = calculate_consistency_metrics(all_results)
    
    return {
        "overall": overall_metrics,
        "file_stats": file_stats,
        "total_files": len(eval_files),
        "total_checkpoints": len(all_results)
    }

def format_report(analysis_result: Dict) -> str:
    """
    格式化分析报告
    """
    overall = analysis_result["overall"]
    file_stats = analysis_result["file_stats"]
    
    report = []
    report.append("# 评委一致性分析报告\n")
    report.append(f"## 总体统计\n")
    report.append(f"- **评估文件总数**: {analysis_result['total_files']}")
    report.append(f"- **检查项总数**: {analysis_result['total_checkpoints']}")
    report.append(f"- **评委数量**: {overall['total_judges']}\n")
    
    report.append(f"### 一致性分布\n")
    total = overall["total_checkpoints"]
    if total > 0:
        unanimous_rate = round(overall["unanimous_agreement"] / total * 100, 2)
        majority_rate = round(overall["majority_agreement"] / total * 100, 2)
        split_rate = round(overall["split_votes"] / total * 100, 2)
        
        report.append(f"| 一致性类型 | 数量 | 占比 |")
        report.append(f"|:----------|:----:|:----:|")
        report.append(f"| 完全一致 (10-0 或 0-10) | {overall['unanimous_agreement']} | {unanimous_rate}% |")
        report.append(f"| 多数一致 (>5 或 <5) | {overall['majority_agreement']} | {majority_rate}% |")
        report.append(f"| 分歧 (5-5) | {overall['split_votes']} | {split_rate}% |\n")
    
    report.append(f"### 评委通过率统计\n")
    report.append(f"| 评委 | 通过率(%) | 通过数 | 总数 |")
    report.append(f"|:-----|:---------:|:------:|:----:|")
    for judge_name in sorted(overall["judge_pass_rates"].keys(), key=lambda x: int(x.replace("评委", ""))):
        judge_data = overall["judge_pass_rates"][judge_name]
        report.append(f"| {judge_name} | {judge_data['pass_rate']} | {judge_data['pass_count']} | {judge_data['total_count']} |")
    report.append("")
    
    report.append(f"### 评委间一致性\n")
    report.append(f"| 评委 | 平均一致性(%) | 最低一致性(%) | 最高一致性(%) |")
    report.append(f"|:-----|:------------:|:------------:|:------------:|")
    for judge_name in sorted(overall["judge_agreement_rates"].keys(), key=lambda x: int(x.replace("评委", ""))):
        judge_data = overall["judge_agreement_rates"][judge_name]
        report.append(f"| {judge_name} | {judge_data['avg_agreement']} | {judge_data['min_agreement']} | {judge_data['max_agreement']} |")
    report.append("")
    
    # 计算平均一致性
    if overall["inter_judge_agreement"]:
        avg_agreements = [v["agreement_rate"] for v in overall["inter_judge_agreement"].values()]
        report.append(f"**评委两两之间平均一致性**: {round(statistics.mean(avg_agreements), 2)}%")
        report.append(f"**评委两两之间最低一致性**: {round(min(avg_agreements), 2)}%")
        report.append(f"**评委两两之间最高一致性**: {round(max(avg_agreements), 2)}%\n")
    
    # 按类别统计
    if overall["category_consistency"]:
        report.append(f"### 按类别统计\n")
        report.append(f"| 类别 | 总数 | 完全一致 | 多数一致 | 分歧 | 完全一致率(%) |")
        report.append(f"|:-----|:----:|:--------:|:--------:|:----:|:------------:|")
        for category in sorted(overall["category_consistency"].keys()):
            cat_data = overall["category_consistency"][category]
            total = cat_data["total"]
            if total > 0:
                unanimous_rate = round(cat_data["unanimous"] / total * 100, 2)
                report.append(f"| {category} | {total} | {cat_data['unanimous']} | {cat_data['majority']} | {cat_data['split']} | {unanimous_rate} |")
        report.append("")
    
    # 文件级别统计
    report.append(f"### 文件级别一致性统计\n")
    report.append(f"| 文件名 | 检查项数 | 完全一致率(%) | 多数一致率(%) | 分歧率(%) |")
    report.append(f"|:-------|:--------:|:------------:|:------------:|:---------:|")
    for filename, stats in sorted(file_stats.items()):
        report.append(f"| {filename} | {stats['checkpoint_count']} | {stats['unanimous_rate']} | {stats['majority_rate']} | {stats['split_rate']} |")
    report.append("")
    
    return "\n".join(report)

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python analyze_judge_consistency.py <评估结果目录>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    if not directory.exists():
        print(f"错误: 目录不存在: {directory}")
        sys.exit(1)
    
    print(f"开始分析目录: {directory}")
    analysis_result = analyze_all_files(directory)
    
    # 生成报告
    report = format_report(analysis_result)
    
    # 输出报告
    output_file = directory / "judge_consistency_report.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n分析完成! 报告已保存到: {output_file}")
    print("\n" + "="*80)
    print(report)

if __name__ == "__main__":
    main()


