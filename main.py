"""主程序入口"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="需求文档差异评估系统 - 基于大模型评估需求文档的完整性和准确性"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="基准文档路径（作为真值）",
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
    if not args.target and not args.targets:
        parser.error("必须指定 --target 或 --targets")

    # 加载配置
    try:
        config = load_config()
    except ValueError as e:
        logger.error(f"配置错误: {e}")
        sys.exit(1)

    # 确定评委数量
    judges = args.judges if args.judges is not None else config.eval.default_runs

    # 验证基准文档
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        logger.error(f"基准文档不存在: {args.baseline}")
        sys.exit(1)

    # 确定待评估文档列表
    target_paths = []
    if args.target:
        target_paths.append(Path(args.target))
    if args.targets:
        target_paths.extend([Path(t) for t in args.targets])

    # 验证待评估文档
    for target_path in target_paths:
        if not target_path.exists():
            logger.error(f"待评估文档不存在: {target_path}")
            sys.exit(1)

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
            if judges > 1:
                evaluation = evaluator.evaluate_multiple_runs(
                    checkpoints, target_path, runs=judges
                )
            else:
                evaluation = evaluator.evaluate_single_run(checkpoints, target_path)
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
                    logger.info(
                        f"[{completed}/{len(target_paths)}] ✓ {target_path.name} - "
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
            try:
                if judges > 1:
                    evaluation = evaluator.evaluate_multiple_runs(
                        checkpoints, target_path, runs=judges
                    )
                else:
                    evaluation = evaluator.evaluate_single_run(checkpoints, target_path)

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


if __name__ == "__main__":
    main()
