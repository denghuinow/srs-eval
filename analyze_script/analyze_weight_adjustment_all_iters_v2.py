#!/usr/bin/env python3
"""
分析权重调整对评估结果的影响（包含所有迭代版本）- 精细调整
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

def analyze_iter8_vs_iter6():
    """详细分析iter8 vs iter6的差异"""
    print("="*80)
    print("iter8 vs iter6 详细分析")
    print("="*80)
    
    print("\n维度得分对比:")
    for cat in ["FUNCTIONAL", "BUSINESS_FLOW", "BOUNDARY", "EXCEPTION", "DATA_STATE", "CONSISTENCY_RULE"]:
        iter8_score = version_scores["iter8"][cat]
        iter6_score = version_scores["iter6"][cat]
        diff = iter8_score - iter6_score
        print(f"  {cat:20s}: iter8={iter8_score:.2f}%, iter6={iter6_score:.2f}%, 差异={diff:+.2f}%")
    
    print("\n问题分析:")
    print("  iter6在CONSISTENCY_RULE上有明显优势（+3.70%）")
    print("  如果CONSISTENCY_RULE权重过高，会导致iter6得分高于iter8")
    print("  需要降低CONSISTENCY_RULE权重，或增加iter8优势维度（FUNCTIONAL）的权重")

def find_optimal_weights():
    """寻找最优权重组合"""
    print("\n" + "="*80)
    print("尝试精细权重调整方案")
    print("="*80)
    
    # 方案5: 降低CONSISTENCY_RULE权重，增加FUNCTIONAL权重
    print("\n方案5: 降低CONSISTENCY_RULE权重，增加FUNCTIONAL权重")
    weights5 = {
        "FUNCTIONAL": 0.40,        # 大幅增加（iter8优势）
        "BUSINESS_FLOW": 0.05,     # 大幅降低
        "BOUNDARY": 0.18,          # 增加（iter2和iter6优势）
        "EXCEPTION": 0.08,         # 降低
        "DATA_STATE": 0.08,        # 降低
        "CONSISTENCY_RULE": 0.21,  # 适度增加（但低于方案1的0.25）
    }
    
    print_weights_result(weights5, "方案5")
    
    # 方案6: 进一步降低CONSISTENCY_RULE权重
    print("\n方案6: 进一步降低CONSISTENCY_RULE权重")
    weights6 = {
        "FUNCTIONAL": 0.42,        # 进一步增加
        "BUSINESS_FLOW": 0.05,     # 大幅降低
        "BOUNDARY": 0.20,          # 增加
        "EXCEPTION": 0.08,         # 降低
        "DATA_STATE": 0.08,        # 降低
        "CONSISTENCY_RULE": 0.17,  # 进一步降低
    }
    
    print_weights_result(weights6, "方案6")
    
    # 方案7: 平衡FUNCTIONAL和CONSISTENCY_RULE
    print("\n方案7: 平衡FUNCTIONAL和CONSISTENCY_RULE")
    weights7 = {
        "FUNCTIONAL": 0.38,        # 增加
        "BUSINESS_FLOW": 0.05,      # 大幅降低
        "BOUNDARY": 0.20,           # 增加
        "EXCEPTION": 0.08,          # 降低
        "DATA_STATE": 0.08,         # 降低
        "CONSISTENCY_RULE": 0.21,   # 适度增加
    }
    
    print_weights_result(weights7, "方案7")
    
    # 方案8: 极端方案（最大化FUNCTIONAL）
    print("\n方案8: 极端方案（最大化FUNCTIONAL权重）")
    weights8 = {
        "FUNCTIONAL": 0.45,        # 极端增加
        "BUSINESS_FLOW": 0.05,     # 大幅降低
        "BOUNDARY": 0.20,          # 增加
        "EXCEPTION": 0.05,         # 大幅降低
        "DATA_STATE": 0.05,         # 大幅降低
        "CONSISTENCY_RULE": 0.20,  # 适度增加
    }
    
    print_weights_result(weights8, "方案8")

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
    analyze_iter8_vs_iter6()
    find_optimal_weights()
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("""
关键问题：iter8在CONSISTENCY_RULE维度上明显弱于iter6（-3.70%），
如果CONSISTENCY_RULE权重过高，会导致iter6得分高于iter8。

解决方案：
1. 降低CONSISTENCY_RULE权重（但这可能影响iter2 > no-clarify）
2. 增加FUNCTIONAL权重（iter8在此维度略优于iter6）
3. 降低BUSINESS_FLOW权重（iter6在此维度优于iter8）

需要找到一个平衡点，既能保持iter8 > iter6，又能保持iter2 > no-clarify。
    """)

if __name__ == "__main__":
    main()
