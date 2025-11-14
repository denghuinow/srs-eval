"""对比不同合并算法的效果"""

import argparse
import logging
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.config import load_config
from src.evaluator import Evaluator, MergeStrategy
from src.point_extractor import PointExtractor

# 加载环境变量
load_dotenv()

# 配置日志 - 从环境变量读取日志级别和文件路径
def get_log_level():
    """从环境变量获取日志级别，默认为INFO"""
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return log_levels.get(log_level_str, logging.INFO)

def setup_logging():
    """配置日志系统，控制台只显示异常，文件记录完整日志"""
    from datetime import datetime
    
    # 文件日志级别：从环境变量读取，默认为DEBUG以记录完整信息
    file_log_level_str = os.getenv("LOG_LEVEL", "DEBUG").upper()
    file_log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    file_log_level = file_log_levels.get(file_log_level_str, logging.DEBUG)
    
    # 控制台日志级别：固定为WARNING，只显示异常和错误
    console_log_level = logging.WARNING
    
    # 获取根logger，设置为最低级别（DEBUG）以允许所有日志通过
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 清除现有的处理器
    root_logger.handlers.clear()
    
    # 控制台格式化器：简洁格式（只显示级别和消息）
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # 文件格式化器：完整格式（包含时间戳、模块名、级别、消息）
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台处理器（WARNING级别，只显示异常）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 处理日志文件路径（支持时间戳占位符）
    log_file = os.getenv("LOG_FILE")
    if log_file:
        # 替换时间戳占位符
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_file.replace('{timestamp}', timestamp)
        log_file = log_file.replace('{datetime}', timestamp)
        log_file = log_file.replace('{date}', datetime.now().strftime('%Y%m%d'))
        log_file = log_file.replace('{time}', datetime.now().strftime('%H%M%S'))
        
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用追加模式
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        # 使用logger.warning确保这个消息会显示在控制台
        logger = logging.getLogger(__name__)
        logger.warning(f"日志文件: {log_file} (级别: {logging.getLevelName(file_log_level)})")
    else:
        logger = logging.getLogger(__name__)
        # 不输出到控制台，只在文件中记录
        logger.debug("日志仅输出到控制台（设置LOG_FILE环境变量可同时输出到文件）")
    
    return logger

logger = setup_logging()


def compare_algorithms(
    baseline_path: str | Path,
    target_path: str | Path,
    judges: int = 5,  # 评委数量（每次评估会运行多次，然后合并结果）
    extract_runs: int = 1,  # 要点提取次数
    compare_count: int = 3,  # 比较次数（重复评估多少次来计算标准差）
    force_extract: bool = False,
):
    """对比不同算法的评估结果
    
    Args:
        baseline_path: 基准文档路径
        target_path: 目标文档路径
        judges: 评委数量，每次评估会运行指定次数，然后使用合并策略（如多数投票）合并结果
        extract_runs: 要点提取次数，多次提取后选择要点数量最多的结果
        compare_count: 比较次数，重复整个评估过程多少次来计算标准差和变异系数
        force_extract: 是否强制重新提取要点（忽略缓存）
    """
    
    print("=" * 80)
    print("算法对比工具")
    print("=" * 80)
    print(f"基准文档: {baseline_path}")
    print(f"目标文档: {target_path}")
    print(f"评委数量: {judges} (由{judges}个评委评估，然后合并结果)")
    print(f"要点提取次数: {extract_runs}")
    print(f"比较次数: {compare_count} (重复评估{compare_count}次来计算稳定性)")
    if force_extract:
        print("⚠ 强制重新提取模式（忽略缓存）")
    print()
    
    # 加载配置
    try:
        config = load_config()
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 提取要点
    extractor = PointExtractor(config)
    print("正在提取要点...")
    if extract_runs > 1:
        print(f"ℹ 多次提取模式：将执行 {extract_runs} 次提取，选择要点数量最多的结果")
    points = extractor.extract_points(
        baseline_path,
        force_extract=force_extract,
        extract_runs=extract_runs,
    )
    print(f"✓ 提取到 {len(points)} 个要点\n")
    
    # 创建评估器
    evaluator = Evaluator(config)
    
    # 定义策略列表
    # 检查项合并适用的策略（前7个）
    checkpoint_strategies = [
        MergeStrategy.AVERAGE,
        MergeStrategy.MAJORITY,
        MergeStrategy.CONSENSUS_67,
        MergeStrategy.CONSENSUS_75,
        MergeStrategy.MEDIAN,
        MergeStrategy.WEIGHTED,
        MergeStrategy.CONFIDENCE,
    ]
    
    # 准确性合并适用的策略（全部10个）
    accuracy_strategies = [
        MergeStrategy.AVERAGE,
        MergeStrategy.MAJORITY,
        MergeStrategy.CONSENSUS_67,
        MergeStrategy.CONSENSUS_75,
        MergeStrategy.MEDIAN,
        MergeStrategy.WEIGHTED,
        MergeStrategy.CONFIDENCE,
        MergeStrategy.TRIMMED_MEAN,
        MergeStrategy.QUANTILE_WEIGHTED,
        MergeStrategy.BAYESIAN_AVERAGE,
    ]
    
    print("=" * 80)
    print("算法对比（同时测试检查项合并和准确性合并）")
    print("=" * 80)
    print()
    
    # 稳定性分析：重复 compare_count 次
    checkpoint_results = {}  # strategy -> list of evaluations
    accuracy_results = {}  # strategy -> list of evaluations
    
    for strategy in checkpoint_strategies:
        checkpoint_results[strategy.value] = []
    for strategy in accuracy_strategies:
        accuracy_results[strategy.value] = []
    
    for round_num in range(compare_count):
        print(f"第 {round_num + 1}/{compare_count} 轮评估...")
        logger.info(f"开始第 {round_num + 1}/{compare_count} 轮评估")
        
        # 执行一次完整的评估，获取原始结果（请求 judges 个评委并行评估）
        print(f"  请求 {judges} 个评委并行评估...")
        logger.info(f"请求 {judges} 个评委并行评估")
        raw_results = []
        
        # 并行执行多次评估
        with ThreadPoolExecutor(max_workers=judges) as executor:
            futures = {
                executor.submit(evaluator.evaluate_single_run, points, target_path): i
                for i in range(judges)
            }
            
            completed = 0
            success_count = 0
            fail_count = 0
            for future in as_completed(futures):
                completed += 1
                try:
                    result = future.result()
                    raw_results.append(result)
                    success_count += 1
                    logger.debug(f"评委 {completed}/{judges} 评估完成 - 准确性: {result.accuracy:.2f}")
                    print(f"  [{completed}/{judges}] 评委评估完成")
                except Exception as e:
                    fail_count += 1
                    logger.warning(f"评委 {completed}/{judges} 评估失败: {e}")
                    print(f"  [{completed}/{judges}] 评委评估失败: {e}")
                    continue
            
            logger.info(f"第 {round_num + 1} 轮评估完成 - 成功: {success_count}, 失败: {fail_count}, 总计: {len(raw_results)} 个结果")
        
        # 对原始结果应用不同的合并策略
        print(f"  正在应用检查项合并策略 ({len(checkpoint_strategies)} 种)...")
        logger.info(f"应用检查项合并策略 ({len(checkpoint_strategies)} 种)")
        for strategy in checkpoint_strategies:
            try:
                evaluation = evaluator.merge_evaluation_results_by_checkpoints(
                    raw_results, points, target_path, merge_strategy=strategy
                )
                checkpoint_results[strategy.value].append(evaluation)
                logger.debug(f"检查项合并策略 {strategy.value} - 准确性: {evaluation.accuracy:.2f}")
            except Exception as e:
                logger.error(f"检查项合并策略 {strategy.value} 失败: {e}")
                raise
        
        print(f"  正在应用准确性合并策略 ({len(accuracy_strategies)} 种)...")
        logger.info(f"应用准确性合并策略 ({len(accuracy_strategies)} 种)")
        for strategy in accuracy_strategies:
            try:
                evaluation = evaluator.merge_evaluation_results(
                    raw_results, points, target_path, merge_strategy=strategy
                )
                accuracy_results[strategy.value].append(evaluation)
                logger.debug(f"准确性合并策略 {strategy.value} - 准确性: {evaluation.accuracy:.2f}")
            except Exception as e:
                logger.error(f"准确性合并策略 {strategy.value} 失败: {e}")
                raise
        print()
    
    # 分析稳定性（计算标准差）
    print(f"\n{'=' * 80}")
    print(f"稳定性分析（基于准确性评分，{compare_count}次比较）")
    print(f"{'=' * 80}")
    
    # 检查项合并结果
    print(f"\n【检查项合并】结果:")
    print("-" * 80)
    checkpoint_stability = {}
    for strategy in checkpoint_strategies:
        scores = [eval.accuracy for eval in checkpoint_results[strategy.value]]
        
        if len(scores) > 1:
            std_dev = statistics.stdev(scores)
            mean_score = statistics.mean(scores)
            checkpoint_stability[strategy.value] = {
                "mean": mean_score,
                "std": std_dev,
                "cv": std_dev / mean_score if mean_score > 0 else 0,
                "min": min(scores),
                "max": max(scores),
                "range": max(scores) - min(scores),
            }
        else:
            checkpoint_stability[strategy.value] = {
                "mean": scores[0] if scores else 0,
                "std": 0.0,
                "cv": 0.0,
                "min": scores[0] if scores else 0,
                "max": scores[0] if scores else 0,
                "range": 0.0,
            }
    
    print(f"\n{'算法':<20} {'平均准确性':<15} {'标准差':<12} {'变异系数':<12} {'范围':<12}")
    print("-" * 80)
    for strategy, stats in sorted(
        checkpoint_stability.items(), key=lambda x: x[1]["std"]
    ):
        print(
            f"{strategy:<20} "
            f"{stats['mean']:>13.2f}  "
            f"{stats['std']:>10.2f}  "
            f"{stats['cv']:>10.4f}  "
            f"{stats['range']:>10.2f}"
        )
    
    # 准确性合并结果
    print(f"\n【准确性合并】结果:")
    print("-" * 80)
    accuracy_stability = {}
    for strategy in accuracy_strategies:
        scores = [eval.accuracy for eval in accuracy_results[strategy.value]]
        
        if len(scores) > 1:
            std_dev = statistics.stdev(scores)
            mean_score = statistics.mean(scores)
            accuracy_stability[strategy.value] = {
                "mean": mean_score,
                "std": std_dev,
                "cv": std_dev / mean_score if mean_score > 0 else 0,
                "min": min(scores),
                "max": max(scores),
                "range": max(scores) - min(scores),
            }
        else:
            accuracy_stability[strategy.value] = {
                "mean": scores[0] if scores else 0,
                "std": 0.0,
                "cv": 0.0,
                "min": scores[0] if scores else 0,
                "max": scores[0] if scores else 0,
                "range": 0.0,
            }
    
    print(f"\n{'算法':<20} {'平均准确性':<15} {'标准差':<12} {'变异系数':<12} {'范围':<12}")
    print("-" * 80)
    for strategy, stats in sorted(
        accuracy_stability.items(), key=lambda x: x[1]["std"]
    ):
        print(
            f"{strategy:<20} "
            f"{stats['mean']:>13.2f}  "
            f"{stats['std']:>10.2f}  "
            f"{stats['cv']:>10.4f}  "
            f"{stats['range']:>10.2f}"
        )
    
    # 推荐
    print("\n" + "=" * 80)
    print("推荐：")
    if checkpoint_stability:
        most_stable_checkpoint = min(
            checkpoint_stability.items(), key=lambda x: x[1]["std"]
        )
        print(f"  检查项合并最稳定算法: {most_stable_checkpoint[0]} (标准差: {most_stable_checkpoint[1]['std']:.2f})")
    
    if accuracy_stability:
        most_stable_accuracy = min(
            accuracy_stability.items(), key=lambda x: x[1]["std"]
        )
        print(f"  准确性合并最稳定算法: {most_stable_accuracy[0]} (标准差: {most_stable_accuracy[1]['std']:.2f})")
    
    # 显示算法说明
    print("\n算法说明：")
    print("-" * 80)
    print("  检查项合并：合并每个检查项的投票结果，然后重新计算准确性")
    print("  准确性合并：直接合并每个评委的整体准确性分数")
    print()
    print("  策略说明：")
    print("  AVERAGE:          平均值")
    print("  MAJORITY:         多数投票")
    print("  CONSENSUS_67:     2/3一致性")
    print("  CONSENSUS_75:     3/4一致性")
    print("  MEDIAN:           中位数")
    print("  WEIGHTED:         加权投票")
    print("  CONFIDENCE:       置信区间法")
    print("  TRIMMED_MEAN:     截断均值（仅准确性合并）")
    print("  QUANTILE_WEIGHTED: 分位数加权（仅准确性合并）")
    print("  BAYESIAN_AVERAGE:  贝叶斯平均（仅准确性合并）")
    print("=" * 80)
    
    return checkpoint_stability, accuracy_stability


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="对比不同合并算法的评估效果"
    )
    parser.add_argument(
        "baseline",
        type=str,
        help="基准文档路径（作为真值）",
    )
    parser.add_argument(
        "target",
        type=str,
        help="待评估文档路径",
    )
    parser.add_argument(
        "--judges",
        type=int,
        default=None,
        help="评委数量，每次评估会运行指定次数，然后使用合并策略（如多数投票）合并结果。默认使用配置中的值（通常为3），建议5次以上以便对比",
    )
    parser.add_argument(
        "--extract-runs",
        type=int,
        default=1,
        help="要点提取次数，多次提取后选择要点数量最多的结果（默认：1）",
    )
    parser.add_argument(
        "--compare-count",
        type=int,
        default=3,
        help="比较次数，重复整个评估过程多少次来计算标准差和变异系数（默认：3）",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="强制重新提取要点清单（忽略缓存）",
    )
    
    args = parser.parse_args()
    
    # 加载配置以获取默认值
    try:
        config = load_config()
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 确定评委数量
    judges = args.judges if args.judges is not None else config.eval.default_runs
    
    baseline = Path(args.baseline)
    target = Path(args.target)
    
    if not baseline.exists():
        print(f"错误: 基准文档不存在: {baseline}", file=sys.stderr)
        sys.exit(1)
    
    if not target.exists():
        print(f"错误: 目标文档不存在: {target}", file=sys.stderr)
        sys.exit(1)
    
    compare_algorithms(
        baseline, 
        target, 
        judges=judges,
        extract_runs=args.extract_runs,
        compare_count=args.compare_count,
        force_extract=args.force_extract,
    )

