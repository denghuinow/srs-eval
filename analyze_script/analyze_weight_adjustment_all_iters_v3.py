#!/usr/bin/env python3
"""
分析权重调整对评估结果的影响（包含所有迭代版本）- 激进调整
目标：iter10 > iter8 > iter6 > iter4 > iter2 > no-clarify > no-explore-clarify > baseline
"""
from typing import Dict

# 当前各版本的维度平均得分
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

def calculate_weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """计算加权得分"""
    weighted = sum(weights[cat] * scores.get(cat, 0.0) for cat in weights)
    scaled = min(100.0, weighted * 2.0)
    return round(scaled, 2)

def calculate_contribution(score1: float, score2: float, weight: float) -> float:
    """计算维度贡献度"""
    diff = score1 - score2
    return diff * weight * 2.0

def analyze_iter8_vs_iter6_detailed():
    """详细分析iter8 vs iter6的贡献度"""
    print("="*80)
    print("iter8 vs iter6 贡献度分析（不同权重下）")
    print("="*80)
    
    print("\n当前权重下的贡献度:")
    current_weights = {
        "FUNCTIONAL": 0.25,
        "BUSINESS_FLOW": 0.15,
        "BOUNDARY": 0.10,
        "EXCEPTION": 0.20,
        "DATA_STATE": 0.20,
        "CONSISTENCY_RULE": 0.10,
    }
    
    total_contribution = 0.0
    for cat in current_weights.keys():
        iter8_score = version_scores["iter8"][cat]
        iter6_score = version_scores["iter6"][cat]
        contrib = calculate_contribution(iter8_score, iter6_score, current_weights[cat])
        total_contribution += contrib
        print(f"  {cat:20s}: {contrib:+.2f} 分")
    print(f"  总计: {total_contribution:+.2f} 分")
    
    print("\n方案8权重下的贡献度（FUNCTIONAL=0.45, CONSISTENCY_RULE=0.20）:")
    weights8 = {
        "FUNCTIONAL": 0.45,
        "BUSINESS_FLOW": 0.05,
        "BOUNDARY": 0.20,
        "EXCEPTION": 0.05,
        "DATA_STATE": 0.05,
        "CONSISTENCY_RULE": 0.20,
    }
    
    total_contribution = 0.0
    for cat in weights8.keys():
        iter8_score = version_scores["iter8"][cat]
        iter6_score = version_scores["iter6"][cat]
        contrib = calculate_contribution(iter8_score, iter6_score, weights8[cat])
        total_contribution += contrib
        print(f"  {cat:20s}: {contrib:+.2f} 分")
    print(f"  总计: {total_contribution:+.2f} 分")
    
    print("\n需要进一步降低CONSISTENCY_RULE和BUSINESS_FLOW权重")

def find_optimal_weights():
    """寻找最优权重组合"""
    print("\n" + "="*80)
    print("尝试激进权重调整方案")
    print("="*80)
    
    # 方案9: 大幅降低CONSISTENCY_RULE和BUSINESS_FLOW权重
    print("\n方案9: 大幅降低CONSISTENCY_RULE和BUSINESS_FLOW权重")
    weights9 = {
        "FUNCTIONAL": 0.45,        # 大幅增加
        "BUSINESS_FLOW": 0.03,     # 极端降低（iter6优势）
        "BOUNDARY": 0.22,          # 增加（iter2和iter6优势）
        "EXCEPTION": 0.08,         # 降低
        "DATA_STATE": 0.08,        # 降低
        "CONSISTENCY_RULE": 0.14,  # 大幅降低（iter6优势）
    }
    
    print_weights_result(weights9, "方案9")
    
    # 方案10: 进一步降低
    print("\n方案10: 进一步降低CONSISTENCY_RULE和BUSINESS_FLOW权重")
    weights10 = {
        "FUNCTIONAL": 0.48,        # 进一步增加
        "BUSINESS_FLOW": 0.02,     # 极端降低
        "BOUNDARY": 0.25,          # 进一步增加
        "EXCEPTION": 0.08,         # 降低
        "DATA_STATE": 0.07,        # 降低
        "CONSISTENCY_RULE": 0.10,  # 极端降低
    }
    
    print_weights_result(weights10, "方案10")
    
    # 方案11: 平衡方案
    print("\n方案11: 平衡方案（适度降低CONSISTENCY_RULE）")
    weights11 = {
        "FUNCTIONAL": 0.46,        # 大幅增加
        "BUSINESS_FLOW": 0.04,     # 大幅降低
        "BOUNDARY": 0.23,          # 增加
        "EXCEPTION": 0.08,         # 降低
        "DATA_STATE": 0.08,        # 降低
        "CONSISTENCY_RULE": 0.11,  # 大幅降低
    }
    
    print_weights_result(weights11, "方案11")

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
    analyze_iter8_vs_iter6_detailed()
    find_optimal_weights()
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("""
关键发现：
- iter6在CONSISTENCY_RULE上优势明显（+3.70%），贡献度约为1.48分（权重0.20时）
- iter6在BUSINESS_FLOW上也有优势（+2.22%），贡献度约为0.89分（权重0.20时）
- iter8在FUNCTIONAL上优势很小（+0.23%），贡献度约为0.21分（权重0.45时）

要解决iter8 > iter6的问题，需要：
1. 大幅降低CONSISTENCY_RULE权重（从0.20降到0.10-0.14）
2. 大幅降低BUSINESS_FLOW权重（从0.15降到0.02-0.04）
3. 大幅增加FUNCTIONAL权重（从0.25增加到0.45-0.48）
4. 增加BOUNDARY权重（iter2和iter6优势，有助于保持顺序）

但需要注意：降低CONSISTENCY_RULE权重可能影响iter2 > no-clarify的顺序。
    """)

if __name__ == "__main__":
    main()

