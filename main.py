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
import re

# 加载环境变量
load_dotenv()


def detect_stages(srs_collection_dir: Path) -> dict[str, list[Path]]:
    """
    检测srs_collection目录下的各个阶段
    
    Args:
        srs_collection_dir: srs_collection目录路径
        
    Returns:
        字典，键为阶段名称，值为该阶段下的文档路径列表
    """
    stages = {}
    
    if not srs_collection_dir.exists() or not srs_collection_dir.is_dir():
        return stages
    
    # 遍历所有子目录
    for stage_dir in srs_collection_dir.iterdir():
        if not stage_dir.is_dir():
            continue
        
        stage_name = stage_dir.name
        # 检查是否是阶段目录（srs_document_开头）
        if not stage_name.startswith("srs_document_"):
            continue
        
        # 收集该阶段下的所有.md文件
        md_files = sorted(stage_dir.glob("*.md")) + sorted(stage_dir.glob("*.markdown"))
        if md_files:
            stages[stage_name] = md_files
    
    return stages


def evaluate_stage(
    stage_docs: list[Path],
    baseline_path: Path | None,
    baseline_dir: Path | None,
    stage_output_dir: Path,
    config,
    args,
    judges: int,
    logger
) -> list[DocumentEvaluation]:
    """
    评估单个阶段的所有文档
    
    Args:
        stage_docs: 该阶段的文档路径列表
        baseline_path: 基准文档路径（单个基准文档模式）
        baseline_dir: 基准文档目录（匹配模式）
        stage_output_dir: 该阶段的输出目录
        config: 配置对象
        args: 命令行参数
        judges: 评委数量
        logger: 日志记录器
        
    Returns:
        该阶段的评估结果列表
    """
    formatter = OutputFormatter()
    use_matching_mode = baseline_dir is not None
    
    # 如果不是匹配模式，从基准文档提取要点清单
    checkpoints = None
    if not use_matching_mode and baseline_path:
        logger.info(f"正在从基准文档提取要点清单: {baseline_path}")
        logger.info("-" * 60)
        try:
            extractor = PointExtractor(config, prompt_version=config.prompt_version)
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
            return []
    elif use_matching_mode:
        extractor = PointExtractor(config, prompt_version=config.prompt_version)
    
    # 检查已存在的评估结果
    evaluations = []
    new_target_paths = []
    
    if args.skip_existing:
        force_re_eval_set = set()
        if args.force_re_eval:
            for item in args.force_re_eval:
                item_path = Path(item)
                force_re_eval_set.add(item_path.stem)
        
        for target_path in stage_docs:
            doc_name = Path(target_path).stem
            
            if args.force_re_eval and doc_name in force_re_eval_set:
                logger.info(f"🔄 {doc_name} - 强制重新评估")
                new_target_paths.append(target_path)
                continue
            
            json_path = stage_output_dir / f"{doc_name}_evaluation.json"
            if json_path.exists():
                existing_eval = formatter.load_json(json_path)
                if existing_eval:
                    evaluations.append(existing_eval)
                    logger.info(f"⊘ {doc_name} - 已存在，跳过评估")
                    continue
            
            new_target_paths.append(target_path)
        
        if evaluations:
            logger.info(f"已跳过 {len(evaluations)} 个已存在的评估结果")
        if new_target_paths:
            logger.info(f"需要评估 {len(new_target_paths)} 个新文档")
        logger.info("")
    else:
        new_target_paths = stage_docs
    
    # 评估文档
    evaluator = Evaluator(config, prompt_version=config.prompt_version)
    
    parallel_eval = len(new_target_paths) > 1
    max_workers = args.max_workers
    if parallel_eval and new_target_paths:
        if max_workers is None:
            max_workers = min(len(new_target_paths), 10)
        logger.info(f"ℹ 并行执行模式：最大工作线程数 = {max_workers}")
        logger.info("")
    
    def evaluate_document(target_path: Path) -> tuple[Path, DocumentEvaluation | None]:
        """评估单个文档的函数，用于并行执行"""
        try:
            doc_baseline_path = baseline_path
            doc_checkpoints = checkpoints
            
            if use_matching_mode:
                matched_baseline = find_matching_baseline(target_path, baseline_dir)
                if matched_baseline is None:
                    logger.warning(f"未找到 {target_path.name} 的匹配基准文档，跳过评估")
                    return (target_path, None)
                doc_baseline_path = matched_baseline
                
                try:
                    doc_checkpoints = extractor.extract_points(
                        doc_baseline_path,
                        force_extract=args.force_extract,
                        extract_runs=args.extract_runs,
                    )
                except Exception as e:
                    logger.error(f"从基准文档 {doc_baseline_path.name} 提取要点失败: {e}")
                    return (target_path, None)
            
            if judges > 1:
                evaluation = evaluator.evaluate_multiple_runs(
                    doc_checkpoints, target_path, runs=judges, baseline_document_path=doc_baseline_path
                )
            else:
                start_time = time.time()
                evaluation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                evaluation = evaluator.evaluate_single_run(doc_checkpoints, target_path)
                evaluation.model_name = config.openai.model
                evaluation.baseline_document = str(doc_baseline_path)
                evaluation.evaluation_time = evaluation_time
                evaluation.evaluation_duration = time.time() - start_time
            return (target_path, evaluation)
        except Exception as e:
            logger.error(f"评估文档 {target_path} 失败: {e}")
            logger.debug(f"评估失败详情:", exc_info=True)
            return (target_path, None)
    
    # 执行评估
    if parallel_eval and new_target_paths:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(evaluate_document, path): path for path in new_target_paths}
            completed = 0
            for future in as_completed(future_to_path):
                completed += 1
                target_path, evaluation = future.result()
                if evaluation:
                    evaluations.append(evaluation)
                    doc_name = target_path.stem
                    weighted_score = OutputFormatter._calculate_weighted_score(evaluation)
                    logger.info(f"[{completed}/{len(new_target_paths)}] ✓ {doc_name}: 加权得分={weighted_score:.2f}")
                else:
                    logger.error(f"[{completed}/{len(new_target_paths)}] ✗ {target_path.stem}: 评估失败")
    else:
        for target_path in new_target_paths:
            target_path, evaluation = evaluate_document(target_path)
            if evaluation:
                evaluations.append(evaluation)
                doc_name = target_path.stem
                weighted_score = OutputFormatter._calculate_weighted_score(evaluation)
                logger.info(f"✓ {doc_name}: 加权得分={weighted_score:.2f}")
            else:
                logger.error(f"✗ {target_path.stem}: 评估失败")
    
    # 保存评估结果
    for evaluation in evaluations:
        doc_name = Path(evaluation.target_document).stem
        
        json_path = stage_output_dir / f"{doc_name}_evaluation.json"
        formatter.save_json(evaluation, json_path)
        
        if args.output in ["markdown", "all"]:
            md_path = stage_output_dir / f"{doc_name}_evaluation.md"
            formatter.save_markdown(evaluation, md_path)
        
        tsv_path = stage_output_dir / f"{doc_name}_evaluation.tsv"
        formatter.save_tsv(evaluation, tsv_path)
    
    # 生成CSV汇总
    if args.output in ["csv", "all"] and evaluations:
        csv_path = stage_output_dir / "evaluations_summary.csv"
        formatter.to_csv(evaluations, csv_path)
        logger.info(f"✓ CSV: {csv_path}")
    
    # 生成阶段聚合报告
    if len(evaluations) > 1:
        logger.info("")
        logger.info("正在生成阶段聚合统计报告...")
        summary_path = stage_output_dir / "summary_report.md"
        total_time = sum(
            e.evaluation_duration for e in evaluations 
            if e.evaluation_duration is not None
        )
        
        target_dir_path = None
        baseline_dir_path = None
        if use_matching_mode:
            baseline_dir_path = baseline_dir
        
        formatter.save_summary_report(
            evaluations,
            summary_path,
            baseline_path,
            target_dir=target_dir_path,
            baseline_dir=baseline_dir_path,
            output_dir=stage_output_dir,
            judges=judges,
            total_time=total_time,
        )
        logger.info(f"✓ 阶段聚合统计报告: {summary_path}")
    
    return evaluations


def sort_stage_names(stage_names: list[str]) -> list[str]:
    """
    对阶段名称进行排序
    
    排序规则：
    1. no-explore-clarify
    2. no-clarify
    3. iter1, iter2, iter3, ...
    
    Args:
        stage_names: 阶段名称列表
        
    Returns:
        排序后的阶段名称列表
    """
    def stage_key(name: str) -> tuple:
        """生成排序键"""
        if name == "srs_document_no-explore-clarify":
            return (0, 0)
        elif name == "srs_document_no-clarify":
            return (1, 0)
        elif name.startswith("srs_document_iter"):
            # 提取迭代次数
            match = re.search(r'iter(\d+)', name)
            if match:
                return (2, int(match.group(1)))
            return (2, 999)  # 无法解析的iter放在最后
        else:
            return (3, 0)  # 其他阶段放在最后
    
    return sorted(stage_names, key=stage_key)

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
        description="需求文档差异评估系统 - 基于大模型评估需求文档"
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
        "--srs-collection-dir",
        type=str,
        help="srs_collection目录路径（自动按阶段分组评估，每个阶段生成独立报告）",
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
    parser.add_argument(
        "--extract-cache-only",
        action="store_true",
        help="仅构建要点缓存，不进行评估",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        help="提示词版本（默认：从环境变量PROMPT_VERSION或配置中读取，默认值为v1）",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已存在的评估结果（如果输出文件已存在，则加载已有结果而不是重新评估）",
    )
    parser.add_argument(
        "--force-re-eval",
        type=str,
        nargs="+",
        help="强制重新评估指定的文档（即使已存在评估结果，也会重新评估）。可以指定文档名称（不含扩展名）或完整路径",
    )

    args = parser.parse_args()

    # 验证参数
    if not args.baseline and not args.baseline_dir:
        parser.error("必须指定 --baseline 或 --baseline-dir")
    
    # 如果使用 --extract-cache-only，可以不指定 target
    if not args.extract_cache_only:
        if not args.target and not args.targets and not args.target_dir and not args.srs_collection_dir:
            parser.error("必须指定 --target、--targets、--target-dir 或 --srs-collection-dir")

    # 加载配置
    try:
        config = load_config()
        # 如果通过参数指定了提示词版本，覆盖配置中的版本
        if args.prompt_version:
            config.prompt_version = args.prompt_version
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
        elif not args.extract_cache_only:
            # 如果没有指定目标文档目录，且不是仅构建缓存模式，则使用第一个 .md 文件作为基准文档
            md_files = sorted(baseline_dir.glob("*.md")) + sorted(baseline_dir.glob("*.markdown"))
            if not md_files:
                logger.error(f"基准文档文件夹中没有找到 .md 文件: {args.baseline_dir}")
                sys.exit(1)
            baseline_path = md_files[0]
            logger.info(f"从基准文档文件夹中选择: {baseline_path.name}")

    # 如果使用 --extract-cache-only，仅构建要点缓存，不进行评估
    if args.extract_cache_only:
        logger.info("=" * 60)
        logger.info("仅构建要点缓存模式（不进行评估）")
        logger.info("=" * 60)
        logger.info("")
        
        extractor = PointExtractor(config, prompt_version=config.prompt_version)
        logger.info(f"使用模型: {config.openai.model}")
        logger.info(f"API地址: {config.openai.base_url}")
        if args.prompt_version:
            logger.info(f"提示词版本: {config.prompt_version}")
        
        if args.force_extract:
            logger.info("⚠ 强制重新提取模式（忽略缓存）")
        else:
            logger.info("ℹ 使用缓存机制（如果存在）")
        
        if args.extract_runs > 1:
            logger.info(f"ℹ 多次提取模式：将执行 {args.extract_runs} 次提取，选择检查项数量最多的结果")
        logger.info("")
        
        # 确定要提取的基准文档列表
        baseline_docs = []
        if baseline_dir:
            # 如果指定了基准目录，提取目录中的所有 .md 文件（递归）
            md_files = sorted(baseline_dir.rglob("*.md")) + sorted(baseline_dir.rglob("*.markdown"))
            if not md_files:
                logger.error(f"基准文档文件夹中没有找到 .md 文件: {args.baseline_dir}")
                sys.exit(1)
            baseline_docs.extend(md_files)
            logger.info(f"找到 {len(baseline_docs)} 个基准文档")
        elif baseline_path:
            # 如果只指定了单个基准文档，只处理该文档
            baseline_docs.append(baseline_path)
        
        # 检查缓存并过滤需要处理的文档
        from src.document_parser import DocumentParser
        parser = DocumentParser()
        docs_to_process = []
        skipped_count = 0
        
        if not args.force_extract:
            logger.info("检查缓存状态...")
            for doc_path in baseline_docs:
                try:
                    # 读取文档内容以计算hash
                    content = parser.read_markdown(doc_path)
                    content_hash = extractor._get_content_hash(content)
                    
                    # 检查缓存是否存在
                    if extractor.has_cache(doc_path, content_hash):
                        logger.info(f"⊘ {doc_path.name} - 缓存已存在，跳过")
                        skipped_count += 1
                    else:
                        docs_to_process.append(doc_path)
                except Exception as e:
                    logger.warning(f"⚠ {doc_path.name} - 检查缓存时出错: {e}，将尝试提取")
                    docs_to_process.append(doc_path)
        else:
            docs_to_process = baseline_docs
        
        if skipped_count > 0:
            logger.info(f"已跳过 {skipped_count} 个已有缓存的文档")
        if docs_to_process:
            logger.info(f"需要处理 {len(docs_to_process)} 个文档")
        logger.info("")
        
        if not docs_to_process:
            logger.info("=" * 60)
            logger.info("所有文档的缓存已存在，无需处理！")
            logger.info("=" * 60)
            sys.exit(0)
        
        # 确定是否并行执行
        parallel_extract = len(docs_to_process) > 1
        max_workers = args.max_workers
        
        if parallel_extract:
            if max_workers is None:
                max_workers = min(len(docs_to_process), 10)  # 最多10个并行
            logger.info(f"ℹ 并行执行模式：最大工作线程数 = {max_workers}")
            logger.info("")
        
        # 提取要点缓存的函数
        def extract_document(doc_path: Path) -> tuple[Path, bool, int | None]:
            """提取单个文档的要点缓存"""
            try:
                checkpoints = extractor.extract_points(
                    doc_path,
                    force_extract=args.force_extract,
                    extract_runs=args.extract_runs,
                )
                return (doc_path, True, len(checkpoints))
            except Exception as e:
                logger.error(f"✗ {doc_path.name} - 提取要点失败: {e}")
                logger.debug(f"提取要点失败详情:", exc_info=True)
                return (doc_path, False, None)
        
        # 提取要点缓存
        success_count = 0
        if parallel_extract:
            # 并行执行
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(extract_document, doc_path): doc_path
                    for doc_path in docs_to_process
                }
                
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    doc_path, success, checkpoint_count = future.result()
                    if success:
                        success_count += 1
                        logger.info(
                            f"[{completed}/{len(docs_to_process)}] ✓ {doc_path.name} - "
                            f"检查项清单：共 {checkpoint_count} 个检查项"
                        )
                    else:
                        logger.info(f"[{completed}/{len(docs_to_process)}] ✗ {doc_path.name} - 提取失败")
        else:
            # 串行执行
            for doc_path in docs_to_process:
                logger.info(f"正在从基准文档提取要点清单: {doc_path.name}")
                logger.info("-" * 60)
                
                doc_path, success, checkpoint_count = extract_document(doc_path)
                if success:
                    success_count += 1
                    logger.info(f"✓ {doc_path.name} - 检查项清单：共 {checkpoint_count} 个检查项")
                logger.info("")
        
        logger.info("=" * 60)
        logger.info(f"要点缓存构建完成！")
        logger.info(f"  成功: {success_count}/{len(docs_to_process)}")
        if skipped_count > 0:
            logger.info(f"  跳过: {skipped_count} (已有缓存)")
        logger.info(f"  总计: {len(baseline_docs)}")
        logger.info("=" * 60)
        sys.exit(0)

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
        
        # 如果指定了 --force-re-eval，只评估指定的文件
        if args.force_re_eval:
            # 准备强制重新评估的文档名称集合（支持多种格式）
            force_re_eval_set = set()
            for item in args.force_re_eval:
                # 支持文档名称（不含扩展名）或完整路径
                item_path = Path(item)
                if item_path.is_absolute() or item_path.exists():
                    # 是完整路径
                    force_re_eval_set.add(item_path.stem)
                else:
                    # 是文档名称（去掉扩展名以匹配 doc_name 格式）
                    force_re_eval_set.add(item_path.stem)
            
            # 只保留在 force_re_eval_set 中的文件
            filtered_files = [f for f in md_files if f.stem in force_re_eval_set]
            if not filtered_files:
                logger.warning(f"在目录中未找到 --force-re-eval 指定的文件")
                logger.info(f"  指定的文件: {args.force_re_eval}")
                logger.info(f"  目录中的文件数量: {len(md_files)}")
            else:
                logger.info(f"从待评估文档文件夹中找到 {len(md_files)} 个文档，将只评估 {len(filtered_files)} 个指定文件")
                md_files = filtered_files
        else:
            logger.info(f"从待评估文档文件夹中找到 {len(md_files)} 个文档")
        
        target_paths.extend(md_files)
    
    # 处理srs_collection目录（按阶段分组评估）
    stage_evaluations = {}  # 存储每个阶段的评估结果
    if args.srs_collection_dir:
        srs_collection_dir = Path(args.srs_collection_dir)
        if not srs_collection_dir.exists():
            logger.error(f"srs_collection目录不存在: {args.srs_collection_dir}")
            sys.exit(1)
        if not srs_collection_dir.is_dir():
            logger.error(f"srs_collection路径不是文件夹: {args.srs_collection_dir}")
            sys.exit(1)
        
        # 检测各个阶段
        stages = detect_stages(srs_collection_dir)
        if not stages:
            logger.error(f"在srs_collection目录中未找到任何阶段目录: {args.srs_collection_dir}")
            sys.exit(1)
        
        # 对阶段进行排序
        sorted_stage_names = sort_stage_names(list(stages.keys()))
        logger.info(f"检测到 {len(sorted_stage_names)} 个阶段: {', '.join(sorted_stage_names)}")
        
        # 为每个阶段进行评估
        for stage_name in sorted_stage_names:
            stage_docs = stages[stage_name]
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"开始评估阶段: {stage_name} ({len(stage_docs)} 个文档)")
            logger.info("=" * 60)
            
            # 为每个阶段创建独立的输出目录
            stage_output_dir = Path(args.output_dir) / stage_name
            stage_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 评估该阶段的所有文档
            stage_eval_results = evaluate_stage(
                stage_docs,
                baseline_path,
                baseline_dir,
                stage_output_dir,
                config,
                args,
                judges,
                logger
            )
            
            stage_evaluations[stage_name] = {
                "evaluations": stage_eval_results,
                "output_dir": stage_output_dir
            }
            
            logger.info(f"阶段 {stage_name} 评估完成，共 {len(stage_eval_results)} 个文档")
        
        # 生成跨阶段对比报告
        if len(stage_evaluations) > 1:
            logger.info("")
            logger.info("=" * 60)
            logger.info("生成跨阶段对比报告...")
            logger.info("=" * 60)
            
            cross_stage_report_path = Path(args.output_dir) / "cross_stage_comparison.md"
            formatter = OutputFormatter()
            cross_stage_report = formatter.generate_cross_stage_comparison_report(
                stage_evaluations,
                baseline_dir=baseline_dir,
                output_dir=Path(args.output_dir)
            )
            
            with open(cross_stage_report_path, "w", encoding="utf-8") as f:
                f.write(cross_stage_report)
            
            logger.info(f"✓ 跨阶段对比报告: {cross_stage_report_path}")
        
        # 退出，因为已经完成了所有评估
        logger.info("")
        logger.info("所有阶段评估完成！")
        sys.exit(0)

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
            extractor = PointExtractor(config, prompt_version=config.prompt_version)
            logger.info(f"使用模型: {config.openai.model}")
            logger.info(f"API地址: {config.openai.base_url}")
            if args.prompt_version:
                logger.info(f"提示词版本: {config.prompt_version}")
            
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
        extractor = PointExtractor(config, prompt_version=config.prompt_version)
        logger.info(f"使用模型: {config.openai.model}")
        logger.info(f"API地址: {config.openai.base_url}")
        if args.prompt_version:
            logger.info(f"提示词版本: {config.prompt_version}")
        if args.force_extract:
            logger.info("⚠ 强制重新提取模式（忽略缓存）")
        else:
            logger.info("ℹ 使用缓存机制（如果存在）")
        if args.extract_runs > 1:
            logger.info(f"ℹ 多次提取模式：将执行 {args.extract_runs} 次提取，选择检查项数量最多的结果")
        logger.info("")

    # 输出目录（提前创建，用于检查已存在的文件）
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = OutputFormatter()

    # 如果使用 --skip-existing，检查并加载已存在的评估结果
    evaluations = []
    if args.skip_existing:
        logger.info("检查已存在的评估结果...")
        logger.info("-" * 60)
        
        existing_evaluations = []
        new_target_paths = []
        
        # 准备强制重新评估的文档名称集合（支持多种格式）
        force_re_eval_set = set()
        if args.force_re_eval:
            for item in args.force_re_eval:
                # 支持文档名称（不含扩展名）或完整路径
                item_path = Path(item)
                if item_path.is_absolute() or item_path.exists():
                    # 是完整路径
                    force_re_eval_set.add(item_path.stem)
                else:
                    # 是文档名称（去掉扩展名以匹配 doc_name 格式）
                    force_re_eval_set.add(item_path.stem)
        
        for target_path in target_paths:
            doc_name = Path(target_path).stem
            
            # 检查是否需要强制重新评估
            if args.force_re_eval and doc_name in force_re_eval_set:
                logger.info(f"🔄 {doc_name} - 强制重新评估（忽略已存在的结果）")
                new_target_paths.append(target_path)
                continue
            
            json_path = output_dir / f"{doc_name}_evaluation.json"
            md_path = output_dir / f"{doc_name}_evaluation.md"
            
            # 优先检查 JSON 文件（包含完整评估数据）
            if json_path.exists():
                # 尝试加载已存在的评估结果
                existing_eval = formatter.load_json(json_path)
                if existing_eval:
                    existing_evaluations.append(existing_eval)
                    logger.info(f"⊘ {doc_name} - 已存在（JSON），跳过评估")
                else:
                    # 加载失败，需要重新评估
                    new_target_paths.append(target_path)
            # 如果 JSON 不存在，检查 Markdown 文件是否存在（作为已存在的标志）
            elif md_path.exists():
                # Markdown 文件存在但 JSON 不存在，尝试从 Markdown 解析评估结果
                existing_eval = formatter.load_from_markdown(md_path)
                if existing_eval:
                    existing_evaluations.append(existing_eval)
                    # 保存为 JSON 文件，以便下次直接加载
                    formatter.save_json(existing_eval, json_path)
                    logger.info(f"⊘ {doc_name} - 已存在（Markdown），已从Markdown解析并保存为JSON，跳过评估")
                else:
                    # 解析失败，需要重新评估
                    logger.warning(f"⚠ {doc_name} - Markdown存在但解析失败，将重新评估")
                    new_target_paths.append(target_path)
            else:
                # 文件不存在，需要评估
                new_target_paths.append(target_path)
        
        if existing_evaluations:
            logger.info(f"已跳过 {len(existing_evaluations)} 个已存在的评估结果（已加载）")
        if new_target_paths:
            logger.info(f"需要评估 {len(new_target_paths)} 个新文档")
        logger.info("")
        
        # 将已存在的评估结果添加到 evaluations 列表
        evaluations.extend(existing_evaluations)
        
        # 更新 target_paths 为需要评估的文档
        target_paths = new_target_paths

    # 评估文档（支持并行执行）
    evaluator = Evaluator(config, prompt_version=config.prompt_version)

    # 确定是否并行执行（仅根据文档数量，judges 现在是串行的）
    parallel_eval = len(target_paths) > 1
    max_workers = args.max_workers
    
    if parallel_eval and target_paths:
        if max_workers is None:
            # 自动计算：如果是批量评估，每个文档并行
            max_workers = min(len(target_paths), 10)  # 最多10个并行
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
                    weighted_score = OutputFormatter._calculate_weighted_score(evaluation)
                    logger.info(
                        f"[{completed}/{len(target_paths)}] ✓ {target_path.name}{baseline_info} - "
                        f"加权得分: {weighted_score:.2f}"
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
                weighted_score = OutputFormatter._calculate_weighted_score(evaluation)
                logger.info(f"✓ 评估完成 - 加权得分: {weighted_score:.2f}")
                logger.info("")
            except Exception as e:
                logger.error(f"评估失败: {e}")
                logger.debug(f"评估失败详情:", exc_info=True)
                continue

    if not evaluations:
        logger.error("没有成功评估任何文档，也没有已存在的评估结果")
        sys.exit(1)

    # 保存新评估的结果
    if target_paths:
        logger.info("正在保存评估结果...")
        logger.info("-" * 60)
        
        # 只保存新评估的结果（target_paths 中的文档）
        new_evaluations = [e for e in evaluations if Path(e.target_document) in target_paths]
        for evaluation in new_evaluations:
            doc_name = Path(evaluation.target_document).stem

            # 总是保存 JSON 文件（用于 --skip-existing 功能）
            json_path = output_dir / f"{doc_name}_evaluation.json"
            formatter.save_json(evaluation, json_path)
            logger.info(f"✓ JSON: {json_path}")

            if args.output in ["markdown", "all"]:
                md_path = output_dir / f"{doc_name}_evaluation.md"
                formatter.save_markdown(evaluation, md_path)
                logger.info(f"✓ Markdown: {md_path}")
            
            # 始终保存 TSV 文件（包含所有评委的详细结果）
            tsv_path = output_dir / f"{doc_name}_evaluation.tsv"
            formatter.save_tsv(evaluation, tsv_path)
            logger.info(f"✓ TSV: {tsv_path}")
        logger.info("")

    if args.output in ["csv", "all"]:
        csv_path = output_dir / "evaluations_summary.csv"
        formatter.to_csv(evaluations, csv_path)
        logger.info(f"✓ CSV: {csv_path}")

    # 如果有多个评估结果，生成聚合统计报告
    # 基于输出目录中的所有最新评估结果重新生成聚合报告
    if len(evaluations) > 1 or output_dir.exists():
        logger.info("")
        logger.info("正在生成聚合统计报告...")
        summary_path = output_dir / "summary_report.md"
        total_time = time.time() - total_start_time
        
        # 从输出目录中加载所有评估结果（基于最新数据）
        all_evaluations = []
        json_files = sorted(output_dir.glob("*_evaluation.json"))
        
        if json_files:
            logger.info(f"从输出目录加载 {len(json_files)} 个评估结果...")
            for json_file in json_files:
                eval_result = formatter.load_json(json_file)
                if eval_result:
                    all_evaluations.append(eval_result)
            
            if all_evaluations:
                logger.info(f"成功加载 {len(all_evaluations)} 个评估结果")
            else:
                logger.warning("未能加载任何评估结果，使用本次评估的结果")
                all_evaluations = evaluations
        else:
            # 如果没有JSON文件，使用本次评估的结果
            all_evaluations = evaluations
        
        # 如果使用了 --force-re-eval，只包含指定的文件
        report_evaluations = all_evaluations
        if args.force_re_eval:
            # 准备强制重新评估的文档名称集合
            force_re_eval_set = set()
            for item in args.force_re_eval:
                item_path = Path(item)
                if item_path.is_absolute() or item_path.exists():
                    force_re_eval_set.add(item_path.stem)
                else:
                    force_re_eval_set.add(item_path.stem)
            
            # 只保留指定的文件（在 force_re_eval_set 中的文件）
            report_evaluations = [
                eval for eval in all_evaluations 
                if Path(eval.target_document).stem in force_re_eval_set
            ]
            
            if len(report_evaluations) != len(all_evaluations):
                logger.info(f"聚合报告将只包含指定的 {len(report_evaluations)} 个文件（输出目录中共 {len(all_evaluations)} 个评估结果）")
        
        if len(report_evaluations) > 1:
            # 确定target_dir和baseline_dir
            target_dir_path = None
            if args.target_dir:
                target_dir_path = Path(args.target_dir)
            baseline_dir_path = None
            if args.baseline_dir:
                baseline_dir_path = Path(args.baseline_dir)
            formatter.save_summary_report(
                report_evaluations, 
                summary_path, 
                baseline_path,
                target_dir=target_dir_path,
                baseline_dir=baseline_dir_path,
                output_dir=output_dir,
                judges=judges,
                total_time=total_time,
            )
            logger.info(f"✓ 聚合统计报告: {summary_path} (基于 {len(report_evaluations)} 个评估结果)")
        elif len(report_evaluations) == 1:
            logger.info("只有1个评估结果，跳过聚合报告生成")
        else:
            logger.warning("没有评估结果，跳过聚合报告生成")

    logger.info("")
    logger.info("评估完成！")

    # 打印简要总结
    logger.info("\n评估总结:")
    logger.info("-" * 60)
    for evaluation in evaluations:
        doc_name = Path(evaluation.target_document).name
        weighted_score = OutputFormatter._calculate_weighted_score(evaluation)
        logger.info(f"{doc_name}: 加权得分={weighted_score:.2f}")
    
    # 如果有多个评估结果，打印聚合统计
    if len(evaluations) > 1:
        logger.info("")
        logger.info("聚合统计:")
        logger.info("-" * 60)
        weighted_scores = [
            OutputFormatter._calculate_weighted_score(evaluation)
            for evaluation in evaluations
        ]
        logger.info(
            f"加权得分 - 平均: {statistics.mean(weighted_scores):.2f}, "
            f"中位数: {statistics.median(weighted_scores):.2f}, "
            f"范围: [{min(weighted_scores):.2f}, {max(weighted_scores):.2f}]"
        )


if __name__ == "__main__":
    main()
