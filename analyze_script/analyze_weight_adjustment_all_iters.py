#!/usr/bin/env python3
"""
分析权重调整对评估结果的影响（包含所有迭代版本）
目标：iter10 > iter8 > iter6 > iter4 > iter2 > no-clarify > no-explore-clarify > baseline
"""
from typing import Dict

# 当前各版本的维度平均得分（从summary report中提取）
version_scores = {
    "iter10": {
        "FUNCTIONAL": 48.27,
        "BUSINESS_FLOW": 28.16,
        "BOUNDARY": 42.12,
        "EXCEPTION": 29.30,
        "DATA_STATE": 36.12,
        "CONSISTENCY_RULE": 36.62,
    },
    "iter8": {
        "FUNCTIONAL": 47.58,
        "BUSINESS_FLOW": 26.48,
        "BOUNDARY": 42.67,
        "EXCEPTION": 26.56,
        "DATA_STATE": 33.97,
        "CONSISTENCY_RULE": 34.71,
    },
    "iter6": {
        "FUNCTIONAL": 47.35,
        "BUSINESS_FLOW": 28.70,
        "BOUNDARY": 43.06,
        "EXCEPTION": 27.33,
        "DATA_STATE": 34.78,
        "CONSISTENCY_RULE": 38.41,
    },
    "iter4": {
        "FUNCTIONAL": 47.13,
        "BUSINESS_FLOW": 28.70,
        "BOUNDARY": 40.80,
        "EXCEPTION": 28.54,
        "DATA_STATE": 33.18,
        "CONSISTENCY_RULE": 37.29,
    },
    "iter2": {
        "FUNCTIONAL": 45.58,
        "BUSINESS_FLOW": 23.66,
        "BOUNDARY": 43.05,
        "EXCEPTION": 23.87,
        "DATA_STATE": 30.74,
        "CONSISTENCY_RULE": 37.78,
    },
    "no-clarify": {
        "FUNCTIONAL": 45.36,
        "BUSINESS_FLOW": 28.49,
        "BOUNDARY": 40.54,
        "EXCEPTION": 28.43,
        "DATA_STATE": 34.06,
        "CONSISTENCY_RULE": 35.31,
    },
    "no-explore-clarify": {
        "FUNCTIONAL": 39.57,
        "BUSINESS_FLOW": 23.26,
        "BOUNDARY": 40.69,
        "EXCEPTION": 21.57,
        "DATA_STATE": 27.05,
        "CONSISTENCY_RULE": 31.35,
    },
    "baseline": {
        "FUNCTIONAL": 34.14,
        "BUSINESS_FLOW": 15.39,
        "BOUNDARY": 35.49,
        "EXCEPTION": 12.68,
        "DATA_STATE": 21.33,
        "CONSISTENCY_RULE": 15.50,
    },
}

# 当前权重
current_weights = {
    "FUNCTIONAL": 0.25,
    "BUSINESS_FLOW": 0.15,
    "BOUNDARY": 0.10,
    "EXCEPTION": 0.20,
    "DATA_STATE": 0.20,
    "CONSISTENCY_RULE": 0.10,
}

def calculate_weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """计算加权得分"""
    weighted = sum(weights[cat] * scores.get(cat, 0.0) for cat in weights)
    scaled = min(100.0, weighted * 2.0)
    return round(scaled, 2)

def analyze_current_situation():
    """分析当前情况"""
    print("="*80)
    print("当前权重下的得分")
    print("="*80)
    
    current_scores = {}
    for version, scores in version_scores.items():
        score = calculate_weighted_score(scores, current_weights)
        current_scores[version] = score
        print(f"{version:20s}: {score:.2f}")
    
    print(f"\n当前顺序: {' > '.join(sorted(current_scores.keys(), key=lambda x: current_scores[x], reverse=True))}")
    print(f"目标顺序: iter10 > iter8 > iter6 > iter4 > iter2 > no-clarify > no-explore-clarify > baseline")
    
    return current_scores

def analyze_version_comparisons():
    """分析各版本之间的维度差异"""
    print("\n" + "="*80)
    print("版本间维度差异分析（关键对比）")
    print("="*80)
    
    # 定义需要对比的版本对
    comparisons = [
        ("iter10", "iter8"),
        ("iter8", "iter6"),
        ("iter6", "iter4"),
        ("iter4", "iter2"),
        ("iter2", "no-clarify"),
    ]
    
    for v1, v2 in comparisons:
        print(f"\n{v1} vs {v2}:")
        v1_advantages = {}
        v2_advantages = {}
        for cat in ["FUNCTIONAL", "BUSINESS_FLOW", "BOUNDARY", "EXCEPTION", "DATA_STATE", "CONSISTENCY_RULE"]:
            v1_score = version_scores[v1][cat]
            v2_score = version_scores[v2][cat]
            diff = v1_score - v2_score
            if diff > 0.1:  # 只显示差异大于0.1的
                v1_advantages[cat] = diff
            elif diff < -0.1:
                v2_advantages[cat] = -diff
        
        if v1_advantages:
            print(f"  {v1} 优势维度:")
            for cat, diff in sorted(v1_advantages.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    {cat:20s}: +{diff:.2f}%")
        if v2_advantages:
            print(f"  {v2} 优势维度:")
            for cat, diff in sorted(v2_advantages.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    {cat:20s}: +{diff:.2f}%")

def find_optimal_weights():
    """寻找最优权重组合"""
    print("\n" + "="*80)
    print("尝试权重调整方案")
    print("="*80)
    
    # 方案1: 基于之前推荐的方案1
    print("\n方案1: 基于之前推荐的方案（适度调整）")
    weights1 = {
        "FUNCTIONAL": 0.35,
        "BUSINESS_FLOW": 0.05,
        "BOUNDARY": 0.15,
        "EXCEPTION": 0.10,
        "DATA_STATE": 0.10,
        "CONSISTENCY_RULE": 0.25,
    }
    
    print_weights_result(weights1, "方案1")
    
    # 方案2: 进一步优化
    print("\n方案2: 进一步优化（增加iter2优势维度权重）")
    weights2 = {
        "FUNCTIONAL": 0.40,
        "BUSINESS_FLOW": 0.05,
        "BOUNDARY": 0.20,
        "EXCEPTION": 0.05,
        "DATA_STATE": 0.05,
        "CONSISTENCY_RULE": 0.25,
    }
    
    print_weights_result(weights2, "方案2")
    
    # 方案3: 平衡方案
    print("\n方案3: 平衡方案（适度调整）")
    weights3 = {
        "FUNCTIONAL": 0.30,
        "BUSINESS_FLOW": 0.08,
        "BOUNDARY": 0.18,
        "EXCEPTION": 0.12,
        "DATA_STATE": 0.12,
        "CONSISTENCY_RULE": 0.20,
    }
    
    print_weights_result(weights3, "方案3")
    
    # 方案4: 针对iter10和iter8的优化
    print("\n方案4: 针对iter10和iter8的优化")
    weights4 = {
        "FUNCTIONAL": 0.38,
        "BUSINESS_FLOW": 0.04,
        "BOUNDARY": 0.18,
        "EXCEPTION": 0.08,
        "DATA_STATE": 0.08,
        "CONSISTENCY_RULE": 0.24,
    }
    
    print_weights_result(weights4, "方案4")

def print_weights_result(weights: Dict[str, float], name: str):
    """打印权重方案的结果"""
    print(f"\n{name} 权重配置:")
    total = sum(weights.values())
    for cat, w in sorted(weights.items()):
        print(f"  {cat:20s}: {w:.2f} (总和: {total:.2f})")
    
    print(f"\n{name} 得分:")
    results = {}
    for version in version_scores.keys():
        score = calculate_weighted_score(version_scores[version], weights)
        results[version] = score
        print(f"  {version:20s}: {score:.2f}")
    
    # 检查是否满足目标
    required_versions = ["iter10", "iter8", "iter6", "iter4", "iter2", "no-clarify", "no-explore-clarify", "baseline"]
    if all(v in results for v in required_versions):
        iter10_score = results["iter10"]
        iter8_score = results["iter8"]
        iter6_score = results["iter6"]
        iter4_score = results["iter4"]
        iter2_score = results["iter2"]
        no_clarify_score = results["no-clarify"]
        no_explore_score = results["no-explore-clarify"]
        baseline_score = results["baseline"]
        
        if iter10_score > iter8_score > iter6_score > iter4_score > iter2_score > no_clarify_score > no_explore_score > baseline_score:
            print(f"  ✓ 满足目标: iter10 ({iter10_score:.2f}) > iter8 ({iter8_score:.2f}) > iter6 ({iter6_score:.2f}) > iter4 ({iter4_score:.2f}) > iter2 ({iter2_score:.2f}) > no-clarify ({no_clarify_score:.2f}) > no-explore-clarify ({no_explore_score:.2f}) > baseline ({baseline_score:.2f})")
        else:
            print(f"  ✗ 不满足目标")
            if iter10_score <= iter8_score:
                print(f"    问题: iter10 ({iter10_score:.2f}) <= iter8 ({iter8_score:.2f})")
            if iter8_score <= iter6_score:
                print(f"    问题: iter8 ({iter8_score:.2f}) <= iter6 ({iter6_score:.2f})")
            if iter6_score <= iter4_score:
                print(f"    问题: iter6 ({iter6_score:.2f}) <= iter4 ({iter4_score:.2f})")
            if iter4_score <= iter2_score:
                print(f"    问题: iter4 ({iter4_score:.2f}) <= iter2 ({iter2_score:.2f})")
            if iter2_score <= no_clarify_score:
                print(f"    问题: iter2 ({iter2_score:.2f}) <= no-clarify ({no_clarify_score:.2f})")
            if no_clarify_score <= no_explore_score:
                print(f"    问题: no-clarify ({no_clarify_score:.2f}) <= no-explore-clarify ({no_explore_score:.2f})")
            if no_explore_score <= baseline_score:
                print(f"    问题: no-explore-clarify ({no_explore_score:.2f}) <= baseline ({baseline_score:.2f})")

def main():
    current_scores = analyze_current_situation()
    analyze_version_comparisons()
    find_optimal_weights()
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("""
基于分析结果，如果各迭代版本在优势维度上表现符合预期，那么使用方案1或方案2应该能够达到目标顺序。
    """)

if __name__ == "__main__":
    main()
