"""对比不同合并算法的效果"""

import argparse
import json
import logging
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    
    # 禁用第三方库的详细日志（只显示WARNING及以上级别）
    # 这些库会产生大量DEBUG和INFO级别的HTTP请求日志
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logger

logger = setup_logging()


def find_matching_files(
    baseline_dir: Path,
    target_dir: Path,
    file_extensions: List[str] = None,
) -> List[Tuple[Path, Path]]:
    """找到两个文件夹中的同名文件对
    
    Args:
        baseline_dir: 基准文件夹路径
        target_dir: 目标文件夹路径
        file_extensions: 要处理的文件扩展名列表，None表示处理所有文件
        
    Returns:
        文件对列表，每个元素是 (baseline_file, target_file) 的元组
    """
    if file_extensions is None:
        file_extensions = ['.md', '.txt']
    
    # 将扩展名转换为小写并确保以点开头
    file_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                      for ext in file_extensions]
    
    # 扫描基准文件夹
    baseline_files = {}
    if baseline_dir.is_dir():
        for file_path in baseline_dir.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in file_extensions:
                    # 使用文件名作为键（不含路径）
                    baseline_files[file_path.name] = file_path
    else:
        raise ValueError(f"基准路径不是文件夹: {baseline_dir}")
    
    # 扫描目标文件夹
    target_files = {}
    if target_dir.is_dir():
        for file_path in target_dir.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in file_extensions:
                    target_files[file_path.name] = file_path
    else:
        raise ValueError(f"目标路径不是文件夹: {target_dir}")
    
    # 找到同名文件对
    matching_pairs = []
    for filename in baseline_files:
        if filename in target_files:
            matching_pairs.append((baseline_files[filename], target_files[filename]))
    
    return sorted(matching_pairs, key=lambda x: x[0].name)


def compare_algorithms(
    baseline_path: str | Path,
    target_path: str | Path,
    judges: int = 5,  # 评委数量（每次评估会运行多次，然后合并结果）
    extract_runs: int = 1,  # 要点提取次数
    compare_count: int = 3,  # 比较次数（重复评估多少次来计算标准差）
    force_extract: bool = False,
    quiet: bool = False,  # 是否静默模式（批量模式下使用）
):
    """对比不同算法的评估结果
    
    Args:
        baseline_path: 基准文档路径
        target_path: 目标文档路径
        judges: 评委数量，每次评估会运行指定次数，然后使用合并策略（如多数投票）合并结果
        extract_runs: 要点提取次数，多次提取后选择检查项数量最多的结果
        compare_count: 比较次数，重复整个评估过程多少次来计算标准差和变异系数
        force_extract: 是否强制重新提取要点（忽略缓存）
        quiet: 是否静默模式，减少输出（批量模式下使用）
    """
    
    if not quiet:
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
    if not quiet:
        print("正在提取要点...")
        if extract_runs > 1:
            print(f"ℹ 多次提取模式：将执行 {extract_runs} 次提取，选择检查项数量最多的结果")
    checkpoints = extractor.extract_points(
        baseline_path,
        force_extract=force_extract,
        extract_runs=extract_runs,
    )
    if not quiet:
        print(f"✓ 提取到 {len(checkpoints)} 个检查项\n")
    
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
    
    if not quiet:
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
        if not quiet:
            print(f"第 {round_num + 1}/{compare_count} 轮评估...")
        logger.info(f"开始第 {round_num + 1}/{compare_count} 轮评估")
        
        # 执行一次完整的评估，获取原始结果（请求 judges 个评委并行评估）
        if not quiet:
            print(f"  请求 {judges} 个评委并行评估...")
        logger.info(f"请求 {judges} 个评委并行评估")
        raw_results = []
        
        # 并行执行多次评估
        with ThreadPoolExecutor(max_workers=judges) as executor:
            futures = {
                executor.submit(evaluator.evaluate_single_run, checkpoints, target_path): i
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
                    if not quiet:
                        print(f"  [{completed}/{judges}] 评委评估完成")
                except Exception as e:
                    fail_count += 1
                    logger.warning(f"评委 {completed}/{judges} 评估失败: {e}")
                    print(f"  [{completed}/{judges}] 评委评估失败: {e}")
                    continue
            
            logger.info(f"第 {round_num + 1} 轮评估完成 - 成功: {success_count}, 失败: {fail_count}, 总计: {len(raw_results)} 个结果")
        
        # 对原始结果应用不同的合并策略
        if not quiet:
            print(f"  正在应用检查项合并策略 ({len(checkpoint_strategies)} 种)...")
        logger.info(f"应用检查项合并策略 ({len(checkpoint_strategies)} 种)")
        for strategy in checkpoint_strategies:
            try:
                evaluation = evaluator.merge_evaluation_results_by_checkpoints(
                    raw_results, checkpoints, target_path, merge_strategy=strategy
                )
                checkpoint_results[strategy.value].append(evaluation)
                logger.debug(f"检查项合并策略 {strategy.value} - 准确性: {evaluation.accuracy:.2f}")
            except Exception as e:
                logger.error(f"检查项合并策略 {strategy.value} 失败: {e}")
                raise
        
        if not quiet:
            print(f"  正在应用准确性合并策略 ({len(accuracy_strategies)} 种)...")
        logger.info(f"应用准确性合并策略 ({len(accuracy_strategies)} 种)")
        for strategy in accuracy_strategies:
            try:
                evaluation = evaluator.merge_evaluation_results(
                    raw_results, checkpoints, target_path, merge_strategy=strategy
                )
                accuracy_results[strategy.value].append(evaluation)
                logger.debug(f"准确性合并策略 {strategy.value} - 准确性: {evaluation.accuracy:.2f}")
            except Exception as e:
                logger.error(f"准确性合并策略 {strategy.value} 失败: {e}")
                raise
        if not quiet:
            print()
    
    # 分析稳定性（计算标准差）
    if not quiet:
        print(f"\n{'=' * 80}")
        print(f"稳定性分析（基于准确性评分，{compare_count}次比较）")
        print(f"{'=' * 80}")
    
    # 检查项合并结果
    if not quiet:
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
    
    if not quiet:
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
    if not quiet:
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
    
    if not quiet:
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


def build_cache_only(
    baseline_dir: Path,
    extract_runs: int = 1,
    force_extract: bool = False,
    workers: int = 1,
    file_extensions: List[str] = None,
) -> Dict[str, Any]:
    """仅构建检查点缓存文件（不执行评估）
    
    Args:
        baseline_dir: 基准文件夹路径
        extract_runs: 要点提取次数
        force_extract: 是否强制重新提取要点（忽略缓存）
        workers: 并行处理的工作线程数（1表示顺序处理）
        file_extensions: 要处理的文件扩展名列表
        
    Returns:
        汇总结果字典
    """
    print("=" * 80)
    print("仅构建检查点缓存文件")
    print("=" * 80)
    print(f"基准文件夹: {baseline_dir}")
    print(f"要点提取次数: {extract_runs}")
    print(f"并行工作线程数: {workers}")
    if force_extract:
        print("⚠ 强制重新提取模式（忽略缓存）")
    print()
    
    # 加载配置
    try:
        config = load_config()
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 创建提取器
    extractor = PointExtractor(config)
    
    # 扫描基准文件夹中的文件
    if file_extensions is None:
        file_extensions = ['.md', '.txt']
    
    file_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                      for ext in file_extensions]
    
    baseline_files = []
    if baseline_dir.is_dir():
        for file_path in baseline_dir.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in file_extensions:
                    baseline_files.append(file_path)
    else:
        raise ValueError(f"基准路径不是文件夹: {baseline_dir}")
    
    baseline_files = sorted(baseline_files, key=lambda x: x.name)
    
    if not baseline_files:
        print(f"❌ 未找到基准文件！")
        print(f"   文件扩展名: {file_extensions}")
        return {}
    
    print(f"✓ 找到 {len(baseline_files)} 个基准文件\n")
    
    # 存储结果
    all_results = []
    failed_files = []
    
    # 处理每个文件
    def process_baseline_file(file_idx: int, baseline_file: Path):
        """处理单个基准文件"""
        try:
            print(f"[{file_idx + 1}/{len(baseline_files)}] 正在处理: {baseline_file.name}")
            logger.info(f"处理基准文件 [{file_idx + 1}/{len(baseline_files)}]: {baseline_file.name}")
            
            # 提取要点（会自动保存到缓存）
            checkpoints = extractor.extract_points(
                baseline_file,
                force_extract=force_extract,
                extract_runs=extract_runs,
            )
            
            result = {
                "filename": baseline_file.name,
                "baseline_path": str(baseline_file),
                "checkpoints_count": len(checkpoints),
                "status": "success",
            }
            
            print(f"  ✓ 完成: {baseline_file.name} ({len(checkpoints)} 个检查项)\n")
            logger.info(f"文件 {baseline_file.name} 处理完成，提取到 {len(checkpoints)} 个检查项")
            
            return result
        except Exception as e:
            error_msg = f"处理文件 {baseline_file.name} 时出错: {e}"
            print(f"  ❌ {error_msg}\n")
            logger.error(error_msg, exc_info=True)
            return {
                "filename": baseline_file.name,
                "baseline_path": str(baseline_file),
                "status": "failed",
                "error": str(e),
            }
    
    # 并行或顺序处理
    if workers > 1:
        print(f"使用 {workers} 个工作线程并行处理...\n")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_baseline_file, idx, baseline): baseline
                for idx, baseline in enumerate(baseline_files)
            }
            
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                if result["status"] == "failed":
                    failed_files.append(result["filename"])
    else:
        print("顺序处理文件...\n")
        for idx, baseline in enumerate(baseline_files):
            result = process_baseline_file(idx, baseline)
            all_results.append(result)
            if result["status"] == "failed":
                failed_files.append(result["filename"])
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    
    successful_results = [r for r in all_results if r["status"] == "success"]
    print(f"\n处理了 {len(baseline_files)} 个文件")
    print(f"  成功: {len(successful_results)}")
    print(f"  失败: {len(failed_files)}")
    
    if successful_results:
        total_checkpoints = sum(r.get("checkpoints_count", 0) for r in successful_results)
        avg_checkpoints = total_checkpoints / len(successful_results) if successful_results else 0
        print(f"  总检查项数: {total_checkpoints}")
        print(f"  平均检查项数: {avg_checkpoints:.1f}")
    
    if failed_files:
        print(f"\n失败的文件:")
        for filename in failed_files:
            print(f"  - {filename}")
    
    print("=" * 80)
    
    return {
        "total_files": len(baseline_files),
        "successful_files": len(successful_results),
        "failed_files": len(failed_files),
        "detailed_results": all_results,
    }


def compare_algorithms_batch(
    baseline_dir: Path,
    target_dir: Path,
    judges: int = 5,
    extract_runs: int = 1,
    compare_count: int = 3,
    force_extract: bool = False,
    workers: int = 1,
    file_extensions: List[str] = None,
    output_file: str = None,
    build_cache_only: bool = False,
    command_args: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """批量对比文件夹中的文件
    
    Args:
        baseline_dir: 基准文件夹路径
        target_dir: 目标文件夹路径
        judges: 评委数量
        extract_runs: 要点提取次数
        compare_count: 比较次数
        force_extract: 是否强制重新提取要点
        workers: 并行处理的工作线程数（1表示顺序处理）
        file_extensions: 要处理的文件扩展名列表
        output_file: 结果输出文件路径（可选）
        build_cache_only: 是否仅构建缓存（不执行评估）
        command_args: 执行的命令参数字典（用于记录到报告中）
        
    Returns:
        汇总结果字典
    """
    # 如果仅构建缓存，调用专门的函数
    if build_cache_only:
        return build_cache_only(
            baseline_dir=baseline_dir,
            extract_runs=extract_runs,
            force_extract=force_extract,
            workers=workers,
            file_extensions=file_extensions,
        )
    
    # 加载配置以获取模型名称
    try:
        config = load_config()
        model_name = config.openai.model
    except ValueError as e:
        logger.warning(f"无法加载配置获取模型名称: {e}")
        model_name = None
    
    print("=" * 80)
    print("批量算法对比工具")
    print("=" * 80)
    print(f"基准文件夹: {baseline_dir}")
    print(f"目标文件夹: {target_dir}")
    print(f"评委数量: {judges}")
    print(f"要点提取次数: {extract_runs}")
    print(f"比较次数: {compare_count}")
    print(f"并行工作线程数: {workers}")
    if model_name:
        print(f"OpenAI模型: {model_name}")
    if force_extract:
        print("⚠ 强制重新提取模式（忽略缓存）")
    print()
    
    # 查找匹配的文件对
    print("正在扫描文件...")
    matching_pairs = find_matching_files(baseline_dir, target_dir, file_extensions)
    
    if not matching_pairs:
        print(f"❌ 未找到匹配的文件对！")
        print(f"   基准文件夹中的文件扩展名: {file_extensions or '所有'}")
        return {}
    
    print(f"✓ 找到 {len(matching_pairs)} 对匹配文件\n")
    
    # 存储所有文件的结果
    all_results = []
    failed_files = []
    
    # 处理每个文件对
    def process_file_pair(pair_idx: int, baseline_file: Path, target_file: Path):
        """处理单个文件对"""
        try:
            print(f"[{pair_idx + 1}/{len(matching_pairs)}] 正在处理: {baseline_file.name}")
            logger.info(f"处理文件对 [{pair_idx + 1}/{len(matching_pairs)}]: {baseline_file.name}")
            
            # 调用单文件对比函数（批量模式下使用静默模式）
            checkpoint_stability, accuracy_stability = compare_algorithms(
                baseline_file,
                target_file,
                judges=judges,
                extract_runs=extract_runs,
                compare_count=compare_count,
                force_extract=force_extract,
                quiet=True,  # 批量模式下减少输出
            )
            
            result = {
                "filename": baseline_file.name,
                "baseline_path": str(baseline_file),
                "target_path": str(target_file),
                "checkpoint_stability": checkpoint_stability,
                "accuracy_stability": accuracy_stability,
                "status": "success",
            }
            
            print(f"  ✓ 完成: {baseline_file.name}\n")
            logger.info(f"文件 {baseline_file.name} 处理完成")
            
            return result
        except Exception as e:
            error_msg = f"处理文件 {baseline_file.name} 时出错: {e}"
            print(f"  ❌ {error_msg}\n")
            logger.error(error_msg, exc_info=True)
            return {
                "filename": baseline_file.name,
                "baseline_path": str(baseline_file),
                "target_path": str(target_file),
                "status": "failed",
                "error": str(e),
            }
    
    # 并行或顺序处理
    if workers > 1:
        print(f"使用 {workers} 个工作线程并行处理...\n")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_file_pair, idx, baseline, target): (baseline, target)
                for idx, (baseline, target) in enumerate(matching_pairs)
            }
            
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                if result["status"] == "failed":
                    failed_files.append(result["filename"])
    else:
        print("顺序处理文件...\n")
        for idx, (baseline, target) in enumerate(matching_pairs):
            result = process_file_pair(idx, baseline, target)
            all_results.append(result)
            if result["status"] == "failed":
                failed_files.append(result["filename"])
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    
    summary = aggregate_results(all_results)
    
    # 输出汇总报告
    print_summary_report(summary, len(matching_pairs), len(failed_files))
    
    # 保存结果到文件
    if output_file:
        # 检查输出文件格式
        output_path = Path(output_file)
        if output_path.suffix.lower() == '.md':
            # 如果是.md文件，保存为Markdown格式，同时生成JSON
            save_results_to_markdown(
                all_results, summary, output_file, baseline_dir, target_dir, command_args, model_name
            )
            # 同时生成JSON文件
            json_path = output_path.with_suffix('.json')
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "baseline_dir": str(baseline_dir),
                "target_dir": str(target_dir),
                "model_name": model_name,
                "command_args": command_args or {},
                "summary": summary,
                "detailed_results": all_results,
            }
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"✓ JSON结果已保存到: {json_path}")
            logger.info(f"JSON结果已保存到: {json_path}")
        else:
            # 默认保存为JSON格式，同时生成Markdown
            save_results_to_file(
                all_results, summary, output_file, baseline_dir, target_dir, command_args, None, model_name
            )
            # 自动生成同名的Markdown文件
            md_path = output_path.with_suffix('.md')
            save_results_to_markdown(
                all_results, summary, str(md_path), baseline_dir, target_dir, command_args, model_name
            )
    
    return summary


def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总所有文件的对比结果
    
    Args:
        all_results: 所有文件的对比结果列表
        
    Returns:
        汇总结果字典
    """
    successful_results = [r for r in all_results if r["status"] == "success"]
    
    if not successful_results:
        return {
            "total_files": len(all_results),
            "successful_files": 0,
            "failed_files": len(all_results),
        }
    
    # 收集所有策略的稳定性数据
    checkpoint_stats = defaultdict(list)  # strategy -> list of std values
    accuracy_stats = defaultdict(list)  # strategy -> list of std values
    
    checkpoint_means = defaultdict(list)  # strategy -> list of mean values
    accuracy_means = defaultdict(list)  # strategy -> list of mean values
    
    for result in successful_results:
        if "checkpoint_stability" in result:
            for strategy, stats in result["checkpoint_stability"].items():
                checkpoint_stats[strategy].append(stats["std"])
                checkpoint_means[strategy].append(stats["mean"])
        
        if "accuracy_stability" in result:
            for strategy, stats in result["accuracy_stability"].items():
                accuracy_stats[strategy].append(stats["std"])
                accuracy_means[strategy].append(stats["mean"])
    
    # 计算平均稳定性
    checkpoint_avg_stability = {}
    for strategy in checkpoint_stats:
        std_values = checkpoint_stats[strategy]
        mean_values = checkpoint_means[strategy]
        checkpoint_avg_stability[strategy] = {
            "avg_std": statistics.mean(std_values) if std_values else 0,
            "avg_mean": statistics.mean(mean_values) if mean_values else 0,
            "std_of_std": statistics.stdev(std_values) if len(std_values) > 1 else 0,
        }
    
    accuracy_avg_stability = {}
    for strategy in accuracy_stats:
        std_values = accuracy_stats[strategy]
        mean_values = accuracy_means[strategy]
        accuracy_avg_stability[strategy] = {
            "avg_std": statistics.mean(std_values) if std_values else 0,
            "avg_mean": statistics.mean(mean_values) if mean_values else 0,
            "std_of_std": statistics.stdev(std_values) if len(std_values) > 1 else 0,
        }
    
    return {
        "total_files": len(all_results),
        "successful_files": len(successful_results),
        "failed_files": len(all_results) - len(successful_results),
        "checkpoint_avg_stability": checkpoint_avg_stability,
        "accuracy_avg_stability": accuracy_avg_stability,
        "detailed_results": all_results,
    }


def print_summary_report(
    summary: Dict[str, Any],
    total_files: int,
    failed_count: int,
):
    """打印汇总报告"""
    print(f"\n处理了 {total_files} 个文件")
    print(f"  成功: {summary['successful_files']}")
    print(f"  失败: {failed_count}")
    
    if summary["successful_files"] == 0:
        print("\n⚠ 没有成功处理的文件，无法生成汇总统计")
        return
    
    # 检查项合并结果
    print(f"\n【检查项合并】平均稳定性（基于 {summary['successful_files']} 个文件）:")
    print("-" * 80)
    print(f"{'算法':<20} {'平均标准差':<15} {'标准差的标准差':<18} {'平均准确性':<15}")
    print("-" * 80)
    
    checkpoint_stability = summary.get("checkpoint_avg_stability", {})
    for strategy, stats in sorted(
        checkpoint_stability.items(), key=lambda x: x[1]["avg_std"]
    ):
        print(
            f"{strategy:<20} "
            f"{stats['avg_std']:>13.4f}  "
            f"{stats['std_of_std']:>16.4f}  "
            f"{stats['avg_mean']:>13.4f}"
        )
    
    # 准确性合并结果
    print(f"\n【准确性合并】平均稳定性（基于 {summary['successful_files']} 个文件）:")
    print("-" * 80)
    print(f"{'算法':<20} {'平均标准差':<15} {'标准差的标准差':<18} {'平均准确性':<15}")
    print("-" * 80)
    
    accuracy_stability = summary.get("accuracy_avg_stability", {})
    for strategy, stats in sorted(
        accuracy_stability.items(), key=lambda x: x[1]["avg_std"]
    ):
        print(
            f"{strategy:<20} "
            f"{stats['avg_std']:>13.4f}  "
            f"{stats['std_of_std']:>16.4f}  "
            f"{stats['avg_mean']:>13.4f}"
        )
    
    # 推荐算法
    print("\n" + "=" * 80)
    print("推荐：")
    if checkpoint_stability:
        most_stable_checkpoint = min(
            checkpoint_stability.items(), key=lambda x: x[1]["avg_std"]
        )
        print(f"  检查项合并最稳定算法: {most_stable_checkpoint[0]} "
              f"(平均标准差: {most_stable_checkpoint[1]['avg_std']:.4f})")
    
    if accuracy_stability:
        most_stable_accuracy = min(
            accuracy_stability.items(), key=lambda x: x[1]["avg_std"]
        )
        print(f"  准确性合并最稳定算法: {most_stable_accuracy[0]} "
              f"(平均标准差: {most_stable_accuracy[1]['avg_std']:.4f})")
    
    print("=" * 80)


def save_results_to_markdown(
    all_results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_file: str,
    baseline_dir: Path,
    target_dir: Path,
    command_args: Dict[str, Any] = None,
    model_name: str = None,
):
    """保存结果为Markdown格式
    
    Args:
        all_results: 所有文件的详细结果
        summary: 汇总结果
        output_file: 输出文件路径
        baseline_dir: 基准文件夹路径
        target_dir: 目标文件夹路径
        command_args: 执行的命令参数字典（可选）
        model_name: 使用的OpenAI模型名称（可选）
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    lines = []
    lines.append("# 算法对比批量评估报告")
    lines.append("")
    lines.append(f"**生成时间**: {timestamp}")
    lines.append("")
    lines.append(f"**基准文件夹**: `{baseline_dir}`")
    lines.append(f"**目标文件夹**: `{target_dir}`")
    if model_name:
        lines.append(f"**OpenAI模型**: `{model_name}`")
    lines.append("")
    
    # 添加命令参数部分
    if command_args:
        lines.append("## 执行命令参数")
        lines.append("")
        lines.append("| 参数 | 值 |")
        lines.append("|------|-----|")
        for key, value in sorted(command_args.items()):
            if value is None:
                display_value = "（默认）"
            elif isinstance(value, bool):
                display_value = "是" if value else "否"
            elif isinstance(value, list):
                display_value = ", ".join(str(v) for v in value) if value else "（无）"
            else:
                display_value = str(value)
            lines.append(f"| {key} | `{display_value}` |")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # 处理统计
    lines.append("## 处理统计")
    lines.append("")
    lines.append(f"- **总文件数**: {summary['total_files']}")
    lines.append(f"- **成功处理**: {summary['successful_files']}")
    lines.append(f"- **处理失败**: {summary['failed_files']}")
    lines.append("")
    
    if summary["successful_files"] == 0:
        lines.append("⚠️ **警告**: 没有成功处理的文件，无法生成汇总统计")
        lines.append("")
    else:
        # 检查项合并结果
        lines.append("## 检查项合并 - 平均稳定性")
        lines.append("")
        lines.append(f"基于 {summary['successful_files']} 个文件的统计结果：")
        lines.append("")
        lines.append("| 算法 | 平均标准差 | 标准差的标准差 | 平均准确性 |")
        lines.append("|------|-----------|--------------|----------|")
        
        checkpoint_stability = summary.get("checkpoint_avg_stability", {})
        for strategy, stats in sorted(
            checkpoint_stability.items(), key=lambda x: x[1]["avg_std"]
        ):
            lines.append(
                f"| {strategy} | "
                f"{stats['avg_std']:.4f} | "
                f"{stats['std_of_std']:.4f} | "
                f"{stats['avg_mean']:.4f} |"
            )
        lines.append("")
        
        # 准确性合并结果
        lines.append("## 准确性合并 - 平均稳定性")
        lines.append("")
        lines.append(f"基于 {summary['successful_files']} 个文件的统计结果：")
        lines.append("")
        lines.append("| 算法 | 平均标准差 | 标准差的标准差 | 平均准确性 |")
        lines.append("|------|-----------|--------------|----------|")
        
        accuracy_stability = summary.get("accuracy_avg_stability", {})
        for strategy, stats in sorted(
            accuracy_stability.items(), key=lambda x: x[1]["avg_std"]
        ):
            lines.append(
                f"| {strategy} | "
                f"{stats['avg_std']:.4f} | "
                f"{stats['std_of_std']:.4f} | "
                f"{stats['avg_mean']:.4f} |"
            )
        lines.append("")
        
        # 推荐算法
        lines.append("## 推荐算法")
        lines.append("")
        if checkpoint_stability:
            most_stable_checkpoint = min(
                checkpoint_stability.items(), key=lambda x: x[1]["avg_std"]
            )
            lines.append(
                f"- **检查项合并最稳定算法**: `{most_stable_checkpoint[0]}` "
                f"(平均标准差: {most_stable_checkpoint[1]['avg_std']:.4f})"
            )
        
        if accuracy_stability:
            most_stable_accuracy = min(
                accuracy_stability.items(), key=lambda x: x[1]["avg_std"]
            )
            lines.append(
                f"- **准确性合并最稳定算法**: `{most_stable_accuracy[0]}` "
                f"(平均标准差: {most_stable_accuracy[1]['avg_std']:.4f})"
            )
        lines.append("")
        
        # 详细结果
        lines.append("## 详细结果")
        lines.append("")
        successful_results = [r for r in all_results if r["status"] == "success"]
        
        if successful_results:
            lines.append("### 成功处理的文件")
            lines.append("")
            for result in successful_results:
                lines.append(f"#### {result['filename']}")
                lines.append("")
                lines.append(f"- **基准路径**: `{result['baseline_path']}`")
                lines.append(f"- **目标路径**: `{result['target_path']}`")
                lines.append("")
                
                # 检查项合并结果
                if "checkpoint_stability" in result:
                    lines.append("**检查项合并稳定性**:")
                    lines.append("")
                    lines.append("| 算法 | 平均准确性 | 标准差 | 变异系数 | 范围 |")
                    lines.append("|------|-----------|--------|---------|------|")
                    for strategy, stats in sorted(
                        result["checkpoint_stability"].items(), 
                        key=lambda x: x[1]["std"]
                    ):
                        lines.append(
                            f"| {strategy} | "
                            f"{stats['mean']:.2f} | "
                            f"{stats['std']:.2f} | "
                            f"{stats['cv']:.4f} | "
                            f"{stats['range']:.2f} |"
                        )
                    lines.append("")
                
                # 准确性合并结果
                if "accuracy_stability" in result:
                    lines.append("**准确性合并稳定性**:")
                    lines.append("")
                    lines.append("| 算法 | 平均准确性 | 标准差 | 变异系数 | 范围 |")
                    lines.append("|------|-----------|--------|---------|------|")
                    for strategy, stats in sorted(
                        result["accuracy_stability"].items(), 
                        key=lambda x: x[1]["std"]
                    ):
                        lines.append(
                            f"| {strategy} | "
                            f"{stats['mean']:.2f} | "
                            f"{stats['std']:.2f} | "
                            f"{stats['cv']:.4f} | "
                            f"{stats['range']:.2f} |"
                        )
                    lines.append("")
        
        # 失败的文件
        failed_results = [r for r in all_results if r["status"] == "failed"]
        if failed_results:
            lines.append("### 处理失败的文件")
            lines.append("")
            for result in failed_results:
                lines.append(f"- **{result['filename']}**: {result.get('error', '未知错误')}")
            lines.append("")
        
        # 算法说明
        lines.append("## 算法说明")
        lines.append("")
        lines.append("### 合并方式")
        lines.append("")
        lines.append("- **检查项合并**: 合并每个检查项的投票结果，然后重新计算准确性")
        lines.append("- **准确性合并**: 直接合并每个评委的整体准确性分数")
        lines.append("")
        lines.append("### 策略说明")
        lines.append("")
        lines.append("| 策略 | 说明 |")
        lines.append("|------|------|")
        lines.append("| AVERAGE | 平均值 |")
        lines.append("| MAJORITY | 多数投票 |")
        lines.append("| CONSENSUS_67 | 2/3一致性 |")
        lines.append("| CONSENSUS_75 | 3/4一致性 |")
        lines.append("| MEDIAN | 中位数 |")
        lines.append("| WEIGHTED | 加权投票 |")
        lines.append("| CONFIDENCE | 置信区间法 |")
        lines.append("| TRIMMED_MEAN | 截断均值（仅准确性合并） |")
        lines.append("| QUANTILE_WEIGHTED | 分位数加权（仅准确性合并） |")
        lines.append("| BAYESIAN_AVERAGE | 贝叶斯平均（仅准确性合并） |")
        lines.append("")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n✓ Markdown报告已保存到: {output_path}")
    logger.info(f"Markdown报告已保存到: {output_path}")


def save_results_to_file(
    all_results: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_file: str,
    baseline_dir: Path,
    target_dir: Path,
    command_args: Dict[str, Any] = None,
    output_markdown: str = None,
    model_name: str = None,
):
    """保存结果到文件
    
    Args:
        all_results: 所有文件的详细结果
        summary: 汇总结果
        output_file: 输出文件路径（JSON格式）
        baseline_dir: 基准文件夹路径
        target_dir: 目标文件夹路径
        command_args: 执行的命令参数字典（可选）
        output_markdown: Markdown输出文件路径（可选）
        model_name: 使用的OpenAI模型名称（可选）
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备输出数据
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "baseline_dir": str(baseline_dir),
        "target_dir": str(target_dir),
        "model_name": model_name,
        "command_args": command_args or {},
        "summary": summary,
        "detailed_results": all_results,
    }
    
    # 保存为JSON格式
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ JSON结果已保存到: {output_path}")
    logger.info(f"JSON结果已保存到: {output_path}")
    
    # 如果指定了Markdown输出，或者输出文件是.md格式，则同时生成Markdown
    if output_markdown:
        save_results_to_markdown(
            all_results, summary, output_markdown, baseline_dir, target_dir, command_args, model_name
        )
    elif output_path.suffix.lower() == '.md':
        # 如果输出文件是.md，则直接保存为Markdown格式
        save_results_to_markdown(
            all_results, summary, output_file, baseline_dir, target_dir, command_args, model_name
        )
        # 同时生成JSON文件（同名但扩展名为.json）
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"✓ JSON结果已保存到: {json_path}")
        logger.info(f"JSON结果已保存到: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="对比不同合并算法的评估效果（支持单文件和批量文件夹模式）"
    )
    parser.add_argument(
        "baseline",
        type=str,
        help="基准文档/文件夹路径（作为真值），自动检测是文件还是文件夹",
    )
    parser.add_argument(
        "target",
        type=str,
        nargs="?",
        help="待评估文档/文件夹路径，自动检测是文件还是文件夹。使用 --build-cache-only 时此参数可选",
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
        help="要点提取次数，多次提取后选择检查项数量最多的结果（默认：1）",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="批量模式下的并行工作线程数（默认：1，顺序处理）",
    )
    parser.add_argument(
        "--file-extensions",
        type=str,
        nargs="+",
        default=None,
        help="批量模式下要处理的文件扩展名列表（默认：.md .txt）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="批量模式下结果输出文件路径。支持JSON格式（.json）和Markdown格式（.md）。如果指定.json，会同时生成同名的.md文件；如果指定.md，会同时生成同名的.json文件。单文件模式不支持",
    )
    parser.add_argument(
        "--build-cache-only",
        action="store_true",
        help="仅构建检查点缓存文件（不执行评估）。仅适用于批量文件夹模式，只需要指定基准文件夹路径，target参数可忽略或设置为任意值",
    )
    
    args = parser.parse_args()
    
    # 如果仅构建缓存，处理逻辑不同
    if args.build_cache_only:
        baseline = Path(args.baseline)
        
        if not baseline.exists():
            print(f"错误: 基准路径不存在: {baseline}", file=sys.stderr)
            sys.exit(1)
        
        if not baseline.is_dir():
            print(f"错误: --build-cache-only 模式仅支持文件夹路径", file=sys.stderr)
            sys.exit(1)
        
        # 加载配置
        try:
            config = load_config()
        except ValueError as e:
            print(f"配置错误: {e}", file=sys.stderr)
            sys.exit(1)
        
        # 调用仅构建缓存的函数
        build_cache_only(
            baseline_dir=baseline,
            extract_runs=args.extract_runs,
            force_extract=args.force_extract,
            workers=args.workers,
            file_extensions=args.file_extensions,
        )
        sys.exit(0)
    
    # 加载配置以获取默认值
    try:
        config = load_config()
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 确定评委数量
    judges = args.judges if args.judges is not None else config.eval.default_runs
    
    baseline = Path(args.baseline)
    
    if not baseline.exists():
        print(f"错误: 基准路径不存在: {baseline}", file=sys.stderr)
        sys.exit(1)
    
    # 检查 target 参数
    if args.target is None:
        print(f"错误: 未指定目标路径（target参数）。使用 --build-cache-only 时不需要此参数", file=sys.stderr)
        sys.exit(1)
    
    target = Path(args.target)
    
    if not target.exists():
        print(f"错误: 目标路径不存在: {target}", file=sys.stderr)
        sys.exit(1)
    
    # 自动检测是文件还是文件夹
    baseline_is_dir = baseline.is_dir()
    target_is_dir = target.is_dir()
    
    # 检查模式一致性
    if baseline_is_dir != target_is_dir:
        print(f"错误: 基准和目标必须同为文件或同为文件夹", file=sys.stderr)
        print(f"  基准: {'文件夹' if baseline_is_dir else '文件'}", file=sys.stderr)
        print(f"  目标: {'文件夹' if target_is_dir else '文件'}", file=sys.stderr)
        sys.exit(1)
    
    # 根据类型选择处理模式
    if baseline_is_dir:
        # 批量文件夹模式
        if args.output is None:
            # 生成默认输出文件名（JSON格式）
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output = f"compare_results_{timestamp}.json"
        
        # 构建命令参数字典
        command_args = {
            "baseline": str(baseline),
            "target": str(target),
            "judges": judges,
            "extract_runs": args.extract_runs,
            "compare_count": args.compare_count,
            "force_extract": args.force_extract,
            "workers": args.workers,
            "file_extensions": args.file_extensions,
            "output": args.output,
        }
        
        compare_algorithms_batch(
            baseline,
            target,
            judges=judges,
            extract_runs=args.extract_runs,
            compare_count=args.compare_count,
            force_extract=args.force_extract,
            workers=args.workers,
            file_extensions=args.file_extensions,
            output_file=args.output,
            build_cache_only=False,
            command_args=command_args,
        )
    else:
        # 单文件模式
        if args.workers > 1:
            print("警告: --workers 参数仅在批量文件夹模式下有效，已忽略", file=sys.stderr)
        if args.file_extensions:
            print("警告: --file-extensions 参数仅在批量文件夹模式下有效，已忽略", file=sys.stderr)
        if args.output:
            print("警告: --output 参数仅在批量文件夹模式下有效，已忽略", file=sys.stderr)
        
        compare_algorithms(
            baseline,
            target,
            judges=judges,
            extract_runs=args.extract_runs,
            compare_count=args.compare_count,
            force_extract=args.force_extract,
        )

