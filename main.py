"""主程序入口"""

import argparse
import logging
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.config import load_config
from src.evaluator import DocumentEvaluation, Evaluator
from src.output_formatter import OutputFormatter
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
    """配置日志系统，控制台显示进度和简短异常，文件记录完整日志
    
    控制台日志级别为INFO，只显示进度、结果和异常信息。
    文件日志级别为DEBUG，记录所有详细信息（包括完整的模型交互消息）。
    详细的模型交互消息使用DEBUG级别，因此不会显示在控制台。
    """
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
    
    # 控制台日志级别：INFO，只显示进度、结果和异常信息
    # DEBUG级别的详细交互消息不会显示在控制台
    console_log_level = logging.INFO
    
    # 获取根logger，设置为最低级别（DEBUG）以允许所有日志通过
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 清除现有的处理器
    root_logger.handlers.clear()
    
    # 控制台格式化器：简洁格式（只显示消息，不显示级别）
    console_formatter = logging.Formatter('%(message)s')
    
    # 文件格式化器：完整格式（包含时间戳、模块名、级别、消息）
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台处理器（INFO级别）
    # DEBUG级别的日志不会显示在控制台，因为级别低于INFO
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
        # 文件处理器记录所有详细信息（DEBUG级别）
        root_logger.addHandler(file_handler)
        # 使用logger.info确保这个消息会显示在控制台
        logger = logging.getLogger(__name__)
        logger.info(f"日志文件: {log_file} (级别: {logging.getLevelName(file_log_level)})")
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


def find_matching_baseline(target_file: Path, baseline_dir: Path) -> Path | None:
    """根据目标文档文件名查找匹配的基准文档
    
    Args:
        target_file: 目标文档路径
        baseline_dir: 基准文档目录
        
    Returns:
        匹配的基准文档路径，如果未找到则返回None
    """
    target_stem = target_file.stem  # 不含扩展名的文件名
    
    # 1. 完全匹配文件名（忽略扩展名）
    for ext in ['.md', '.txt', '.markdown']:
        baseline_path = baseline_dir / f"{target_stem}{ext}"
        if baseline_path.exists() and baseline_path.is_file():
            return baseline_path
    
    # 2. 尝试在基准目录中查找包含目标文件名的文件
    for baseline_file in baseline_dir.glob("*"):
        if baseline_file.is_file():
            baseline_stem = baseline_file.stem
            # 如果基准文件名包含目标文件名，或者目标文件名包含基准文件名
            if target_stem in baseline_stem or baseline_stem in target_stem:
                return baseline_file
    
    return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="需求文档差异评估系统 - 基于大模型评估需求文档的完整性和准确性"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="基准文档路径（作为真值，单个文件）",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        help="基准文档文件夹路径（作为真值，文件夹中的第一个 .md 文件）",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="待评估文档路径（单个文档）",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        help="待评估文档路径（多个文档）",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        help="待评估文档文件夹路径（批量评估文件夹中的所有 .md 文件）",
    )
    parser.add_argument(
        "--judges",
        type=int,
        default=None,
        help="评委数量，每次评估会运行指定次数，然后使用合并策略（如多数投票）合并结果。默认使用配置中的值",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["json", "csv", "markdown", "all"],
        default="markdown",
        help="输出格式（默认：markdown）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="输出目录（默认：output）",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="强制重新提取要点清单（忽略缓存）",
    )
    parser.add_argument(
        "--extract-runs",
        type=int,
        default=1,
        help="提取要点清单的运行次数，多次提取后选择检查项数量最多的结果（默认：1）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="并行执行的最大工作线程数（默认：自动，根据任务数量调整）",
    )

    args = parser.parse_args()

    # 验证参数
    if not args.baseline and not args.baseline_dir:
        parser.error("必须指定 --baseline 或 --baseline-dir")
    
    if not args.target and not args.targets and not args.target_dir:
        parser.error("必须指定 --target、--targets 或 --target-dir")

    # 加载配置
    try:
        config = load_config()
    except ValueError as e:
        logger.error(f"配置错误: {e}")
        sys.exit(1)

    # 记录总开始时间
    total_start_time = time.time()

    # 确定评委数量
    judges = args.judges if args.judges is not None else config.eval.default_runs

    # 确定基准文档或基准文档目录
    baseline_path = None
    baseline_dir = None
    use_matching_mode = False  # 是否使用匹配模式（为每个目标文档匹配对应的基准文档）
    
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            logger.error(f"基准文档不存在: {args.baseline}")
            sys.exit(1)
        if not baseline_path.is_file():
            logger.error(f"基准文档路径不是文件: {args.baseline}")
            sys.exit(1)
    elif args.baseline_dir:
        baseline_dir = Path(args.baseline_dir)
        if not baseline_dir.exists():
            logger.error(f"基准文档文件夹不存在: {args.baseline_dir}")
            sys.exit(1)
        if not baseline_dir.is_dir():
            logger.error(f"基准文档路径不是文件夹: {args.baseline_dir}")
            sys.exit(1)
        # 检查是否同时指定了目标文档目录，如果是，则使用匹配模式
        if args.target_dir:
            use_matching_mode = True
            logger.info(f"使用匹配模式：将为每个目标文档匹配对应的基准文档")
        else:
            # 如果没有指定目标文档目录，则使用第一个 .md 文件作为基准文档
            md_files = sorted(baseline_dir.glob("*.md")) + sorted(baseline_dir.glob("*.markdown"))
            if not md_files:
                logger.error(f"基准文档文件夹中没有找到 .md 文件: {args.baseline_dir}")
                sys.exit(1)
            baseline_path = md_files[0]
            logger.info(f"从基准文档文件夹中选择: {baseline_path.name}")

    # 确定待评估文档列表
    target_paths = []
    if args.target:
        target_paths.append(Path(args.target))
    if args.targets:
        target_paths.extend([Path(t) for t in args.targets])
    if args.target_dir:
        target_dir = Path(args.target_dir)
        if not target_dir.exists():
            logger.error(f"待评估文档文件夹不存在: {args.target_dir}")
            sys.exit(1)
        if not target_dir.is_dir():
            logger.error(f"待评估文档路径不是文件夹: {args.target_dir}")
            sys.exit(1)
        # 扫描文件夹中的所有 .md 文件（递归）
        md_files = sorted(target_dir.rglob("*.md")) + sorted(target_dir.rglob("*.markdown"))
        if not md_files:
            logger.error(f"待评估文档文件夹中没有找到 .md 文件: {args.target_dir}")
            sys.exit(1)
        target_paths.extend(md_files)
        logger.info(f"从待评估文档文件夹中找到 {len(md_files)} 个文档")

    # 验证待评估文档
    for target_path in target_paths:
        if not target_path.exists():
            logger.error(f"待评估文档不存在: {target_path}")
            sys.exit(1)
        if not target_path.is_file():
            logger.error(f"待评估文档路径不是文件: {target_path}")
            sys.exit(1)

    # 如果不是匹配模式，则从单个基准文档提取要点清单
    if not use_matching_mode:
        logger.info(f"正在从基准文档提取要点清单: {baseline_path}")
        logger.info("-" * 60)

        # 提取要点清单
        try:
            extractor = PointExtractor(config)
            logger.info(f"使用模型: {config.openai.model}")
            logger.info(f"API地址: {config.openai.base_url}")
            
            if args.force_extract:
                logger.info("⚠ 强制重新提取模式（忽略缓存）")
            else:
                logger.info("ℹ 使用缓存机制（如果存在）")
            
            if args.extract_runs > 1:
                logger.info(f"ℹ 多次提取模式：将执行 {args.extract_runs} 次提取，选择检查项数量最多的结果")
            
            checkpoints = extractor.extract_points(
                baseline_path,
                force_extract=args.force_extract,
                extract_runs=args.extract_runs,
            )
            
            logger.info(f"✓ 检查项清单：共 {len(checkpoints)} 个检查项")
            logger.info("")
        except Exception as e:
            logger.error(f"提取要点失败: {e}")
            logger.debug(f"调试信息:", exc_info=True)
            logger.debug(f"  - 基准文档: {baseline_path}")
            logger.debug(f"  - 文档是否存在: {baseline_path.exists()}")
            if baseline_path.exists():
                try:
                    from src.document_parser import DocumentParser
                    parser = DocumentParser()
                    content = parser.read_markdown(baseline_path)
                    logger.debug(f"  - 文档大小: {len(content)} 字符")
                except Exception as e2:
                    logger.debug(f"  - 读取文档失败: {e2}")
            logger.debug(f"  - 模型: {config.openai.model}")
            logger.debug(f"  - API地址: {config.openai.base_url}")
            sys.exit(1)
    else:
        # 匹配模式：不在这里提取要点清单，而是为每个目标文档单独提取
        checkpoints = None
        extractor = PointExtractor(config)
        logger.info(f"使用模型: {config.openai.model}")
        logger.info(f"API地址: {config.openai.base_url}")
        if args.force_extract:
            logger.info("⚠ 强制重新提取模式（忽略缓存）")
        else:
            logger.info("ℹ 使用缓存机制（如果存在）")
        if args.extract_runs > 1:
            logger.info(f"ℹ 多次提取模式：将执行 {args.extract_runs} 次提取，选择检查项数量最多的结果")
        logger.info("")

    # 评估文档（支持并行执行）
    evaluator = Evaluator(config)
    evaluations = []

    # 确定是否并行执行
    parallel_eval = len(target_paths) > 1 or judges > 1
    max_workers = args.max_workers
    
    if parallel_eval:
        if max_workers is None:
            # 自动计算：如果是批量评估，每个文档并行；如果是多次运行，并行运行
            if len(target_paths) > 1:
                max_workers = min(len(target_paths), 10)  # 最多10个并行
            else:
                max_workers = judges
        logger.info(f"ℹ 并行执行模式：最大工作线程数 = {max_workers}")
        logger.info("")

    def evaluate_document(target_path: Path) -> tuple[Path, DocumentEvaluation | None]:
        """评估单个文档的函数，用于并行执行"""
        try:
            # 如果是匹配模式，为每个目标文档找到对应的基准文档
            doc_baseline_path = baseline_path
            doc_checkpoints = checkpoints
            
            if use_matching_mode:
                # 查找匹配的基准文档
                matched_baseline = find_matching_baseline(target_path, baseline_dir)
                if matched_baseline is None:
                    logger.warning(f"未找到 {target_path.name} 的匹配基准文档，跳过评估")
                    return (target_path, None)
                doc_baseline_path = matched_baseline
                
                # 从对应的基准文档提取检查项清单
                try:
                    doc_checkpoints = extractor.extract_points(
                        doc_baseline_path,
                        force_extract=args.force_extract,
                        extract_runs=args.extract_runs,
                    )
                except Exception as e:
                    logger.error(f"从基准文档 {doc_baseline_path.name} 提取要点失败: {e}")
                    logger.debug(f"提取要点失败详情:", exc_info=True)
                    return (target_path, None)
            
            if judges > 1:
                evaluation = evaluator.evaluate_multiple_runs(
                    doc_checkpoints, target_path, runs=judges, baseline_document_path=doc_baseline_path
                )
            else:
                # 记录开始时间
                start_time = time.time()
                evaluation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                evaluation = evaluator.evaluate_single_run(doc_checkpoints, target_path)
                # 为单次评估也添加元信息
                evaluation.model_name = config.openai.model
                evaluation.baseline_document = str(doc_baseline_path)
                evaluation.evaluation_time = evaluation_time
                evaluation.evaluation_duration = time.time() - start_time
            return (target_path, evaluation)
        except Exception as e:
            logger.error(f"评估文档 {target_path} 失败: {e}")
            logger.debug(f"评估文档 {target_path} 失败详情:", exc_info=True)
            return (target_path, None)

    if parallel_eval and len(target_paths) > 1:
        # 批量评估多个文档，并行执行
        logger.info(f"正在并行评估 {len(target_paths)} 个文档...")
        logger.info("-" * 60)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluate_document, target_path): target_path
                for target_path in target_paths
            }
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                target_path, evaluation = future.result()
                if evaluation is not None:
                    evaluations.append(evaluation)
                    baseline_info = ""
                    if use_matching_mode and evaluation.baseline_document:
                        baseline_name = Path(evaluation.baseline_document).name
                        baseline_info = f" (基准: {baseline_name})"
                    logger.info(
                        f"[{completed}/{len(target_paths)}] ✓ {target_path.name}{baseline_info} - "
                        f"完整性: {evaluation.completeness:.2f}, "
                        f"准确性: {evaluation.accuracy:.2f}, "
                        f"综合: {evaluation.comprehensive:.2f}"
                    )
        logger.info("")
    else:
        # 串行执行（单个文档或不需要并行）
        for target_path in target_paths:
            logger.info(f"正在评估文档: {target_path}")
            if judges > 1:
                logger.info(f"评委数量: {judges}")
            
            # 如果是匹配模式，为每个目标文档找到对应的基准文档
            doc_baseline_path = baseline_path
            doc_checkpoints = checkpoints
            
            if use_matching_mode:
                # 查找匹配的基准文档
                matched_baseline = find_matching_baseline(target_path, baseline_dir)
                if matched_baseline is None:
                    logger.warning(f"未找到 {target_path.name} 的匹配基准文档，跳过评估")
                    continue
                doc_baseline_path = matched_baseline
                logger.info(f"匹配的基准文档: {doc_baseline_path.name}")
                
                # 从对应的基准文档提取检查项清单
                try:
                    logger.info(f"正在从基准文档提取要点清单: {doc_baseline_path.name}")
                    doc_checkpoints = extractor.extract_points(
                        doc_baseline_path,
                        force_extract=args.force_extract,
                        extract_runs=args.extract_runs,
                    )
                    logger.info(f"✓ 检查项清单：共 {len(doc_checkpoints)} 个检查项")
                except Exception as e:
                    logger.error(f"从基准文档 {doc_baseline_path.name} 提取要点失败: {e}")
                    logger.debug(f"提取要点失败详情:", exc_info=True)
                    continue
            
            try:
                if judges > 1:
                    evaluation = evaluator.evaluate_multiple_runs(
                        doc_checkpoints, target_path, runs=judges, baseline_document_path=doc_baseline_path
                    )
                else:
                    # 记录开始时间
                    start_time = time.time()
                    evaluation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    evaluation = evaluator.evaluate_single_run(doc_checkpoints, target_path)
                    # 为单次评估也添加元信息
                    evaluation.model_name = config.openai.model
                    evaluation.baseline_document = str(doc_baseline_path)
                    evaluation.evaluation_time = evaluation_time
                    evaluation.evaluation_duration = time.time() - start_time

                evaluations.append(evaluation)
                logger.info(
                    f"✓ 评估完成 - 完整性: {evaluation.completeness:.2f}, "
                    f"准确性: {evaluation.accuracy:.2f}, "
                    f"综合: {evaluation.comprehensive:.2f}"
                )
                logger.info("")
            except Exception as e:
                logger.error(f"评估失败: {e}")
                logger.debug(f"评估失败详情:", exc_info=True)
                continue

    if not evaluations:
        logger.error("没有成功评估任何文档")
        sys.exit(1)

    # 输出结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formatter = OutputFormatter()

    logger.info("正在保存评估结果...")
    logger.info("-" * 60)

    for i, evaluation in enumerate(evaluations):
        doc_name = Path(evaluation.target_document).stem

        if args.output in ["json", "all"]:
            json_path = output_dir / f"{doc_name}_evaluation.json"
            formatter.save_json(evaluation, json_path)
            logger.info(f"✓ JSON: {json_path}")

        if args.output in ["markdown", "all"]:
            md_path = output_dir / f"{doc_name}_evaluation.md"
            formatter.save_markdown(evaluation, md_path)
            logger.info(f"✓ Markdown: {md_path}")

    if args.output in ["csv", "all"]:
        csv_path = output_dir / "evaluations_summary.csv"
        formatter.to_csv(evaluations, csv_path)
        logger.info(f"✓ CSV: {csv_path}")

    # 如果有多个评估结果，生成聚合统计报告
    if len(evaluations) > 1:
        logger.info("")
        logger.info("正在生成聚合统计报告...")
        summary_path = output_dir / "summary_report.md"
        total_time = time.time() - total_start_time
        # 确定target_dir和baseline_dir
        target_dir_path = None
        if args.target_dir:
            target_dir_path = Path(args.target_dir)
        baseline_dir_path = None
        if args.baseline_dir:
            baseline_dir_path = Path(args.baseline_dir)
        formatter.save_summary_report(
            evaluations, 
            summary_path, 
            baseline_path,
            target_dir=target_dir_path,
            baseline_dir=baseline_dir_path,
            output_dir=output_dir,
            judges=judges,
            total_time=total_time,
        )
        logger.info(f"✓ 聚合统计报告: {summary_path}")

    logger.info("")
    logger.info("评估完成！")

    # 打印简要总结
    logger.info("\n评估总结:")
    logger.info("-" * 60)
    for evaluation in evaluations:
        doc_name = Path(evaluation.target_document).name
        logger.info(
            f"{doc_name}: "
            f"完整性={evaluation.completeness:.2f}, "
            f"准确性={evaluation.accuracy:.2f}, "
            f"综合={evaluation.comprehensive:.2f}"
        )
    
    # 如果有多个评估结果，打印聚合统计
    if len(evaluations) > 1:
        logger.info("")
        logger.info("聚合统计:")
        logger.info("-" * 60)
        completeness_scores = [e.completeness for e in evaluations]
        accuracy_scores = [e.accuracy for e in evaluations]
        comprehensive_scores = [e.comprehensive for e in evaluations]
        logger.info(
            f"完整性 - 平均: {statistics.mean(completeness_scores):.2f}, "
            f"中位数: {statistics.median(completeness_scores):.2f}, "
            f"范围: [{min(completeness_scores):.2f}, {max(completeness_scores):.2f}]"
        )
        logger.info(
            f"准确性 - 平均: {statistics.mean(accuracy_scores):.2f}, "
            f"中位数: {statistics.median(accuracy_scores):.2f}, "
            f"范围: [{min(accuracy_scores):.2f}, {max(accuracy_scores):.2f}]"
        )
        logger.info(
            f"综合 - 平均: {statistics.mean(comprehensive_scores):.2f}, "
            f"中位数: {statistics.median(comprehensive_scores):.2f}, "
            f"范围: [{min(comprehensive_scores):.2f}, {max(comprehensive_scores):.2f}]"
        )


if __name__ == "__main__":
    main()
