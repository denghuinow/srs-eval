"""主程序入口"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.config import load_config
from src.evaluator import DocumentEvaluation, Evaluator
from src.output_formatter import OutputFormatter
from src.point_extractor import PointExtractor


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
        "--runs",
        type=int,
        default=None,
        help="运行次数（用于取平均，提高可重复性），默认使用配置中的值",
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
        help="提取要点清单的运行次数，多次提取后选择要点数量最多的结果（默认：1）",
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
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)

    # 确定运行次数
    runs = args.runs if args.runs is not None else config.eval.default_runs

    # 验证基准文档
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"错误: 基准文档不存在: {args.baseline}", file=sys.stderr)
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
            print(f"错误: 待评估文档不存在: {target_path}", file=sys.stderr)
            sys.exit(1)

    print(f"正在从基准文档提取要点清单: {baseline_path}")
    print("-" * 60)

    # 提取要点清单
    try:
        extractor = PointExtractor(config)
        print(f"使用模型: {config.openai.model}")
        print(f"API地址: {config.openai.base_url}")
        
        if args.force_extract:
            print("⚠ 强制重新提取模式（忽略缓存）")
        else:
            print("ℹ 使用缓存机制（如果存在）")
        
        if args.extract_runs > 1:
            print(f"ℹ 多次提取模式：将执行 {args.extract_runs} 次提取，选择要点数量最多的结果")
        
        points = extractor.extract_points(
            baseline_path,
            force_extract=args.force_extract,
            extract_runs=args.extract_runs,
        )
        
        # 统计检查项数量
        total_checkpoints = sum(
            len(point.get("checkpoints", [])) for point in points
        )
        
        print(f"✓ 要点清单：{len(points)} 个要点，共 {total_checkpoints} 个检查项")
        print()
    except Exception as e:
        print(f"错误: 提取要点失败: {e}", file=sys.stderr)
        print(f"\n调试信息:", file=sys.stderr)
        print(f"  - 基准文档: {baseline_path}", file=sys.stderr)
        print(f"  - 文档是否存在: {baseline_path.exists()}", file=sys.stderr)
        if baseline_path.exists():
            try:
                from src.document_parser import DocumentParser
                parser = DocumentParser()
                content = parser.read_markdown(baseline_path)
                print(f"  - 文档大小: {len(content)} 字符", file=sys.stderr)
            except Exception as e2:
                print(f"  - 读取文档失败: {e2}", file=sys.stderr)
        print(f"  - 模型: {config.openai.model}", file=sys.stderr)
        print(f"  - API地址: {config.openai.base_url}", file=sys.stderr)
        sys.exit(1)

    # 评估文档（支持并行执行）
    evaluator = Evaluator(config)
    evaluations = []

    # 确定是否并行执行
    parallel_eval = len(target_paths) > 1 or runs > 1
    max_workers = args.max_workers
    
    if parallel_eval:
        if max_workers is None:
            # 自动计算：如果是批量评估，每个文档并行；如果是多次运行，并行运行
            if len(target_paths) > 1:
                max_workers = min(len(target_paths), 10)  # 最多10个并行
            else:
                max_workers = runs
        print(f"ℹ 并行执行模式：最大工作线程数 = {max_workers}")
        print()

    def evaluate_document(target_path: Path) -> tuple[Path, DocumentEvaluation | None]:
        """评估单个文档的函数，用于并行执行"""
        try:
            if runs > 1:
                evaluation = evaluator.evaluate_multiple_runs(
                    points, target_path, runs=runs
                )
            else:
                evaluation = evaluator.evaluate_single_run(points, target_path)
            return (target_path, evaluation)
        except Exception as e:
            print(f"错误: 评估文档 {target_path} 失败: {e}", file=sys.stderr)
            return (target_path, None)

    if parallel_eval and len(target_paths) > 1:
        # 批量评估多个文档，并行执行
        print(f"正在并行评估 {len(target_paths)} 个文档...")
        print("-" * 60)
        
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
                    print(
                        f"[{completed}/{len(target_paths)}] ✓ {target_path.name} - "
                        f"完整性: {evaluation.completeness:.2f}, "
                        f"准确性: {evaluation.accuracy:.2f}, "
                        f"综合: {evaluation.comprehensive:.2f}"
                    )
        print()
    else:
        # 串行执行（单个文档或不需要并行）
        for target_path in target_paths:
            print(f"正在评估文档: {target_path}")
            if runs > 1:
                print(f"运行次数: {runs}")
            try:
                if runs > 1:
                    evaluation = evaluator.evaluate_multiple_runs(
                        points, target_path, runs=runs
                    )
                else:
                    evaluation = evaluator.evaluate_single_run(points, target_path)

                evaluations.append(evaluation)
                print(
                    f"✓ 评估完成 - 完整性: {evaluation.completeness:.2f}, "
                    f"准确性: {evaluation.accuracy:.2f}, "
                    f"综合: {evaluation.comprehensive:.2f}"
                )
                print()
            except Exception as e:
                print(f"错误: 评估失败: {e}", file=sys.stderr)
                continue

    if not evaluations:
        print("错误: 没有成功评估任何文档", file=sys.stderr)
        sys.exit(1)

    # 输出结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formatter = OutputFormatter()

    print("正在保存评估结果...")
    print("-" * 60)

    for i, evaluation in enumerate(evaluations):
        doc_name = Path(evaluation.target_document).stem

        if args.output in ["json", "all"]:
            json_path = output_dir / f"{doc_name}_evaluation.json"
            formatter.save_json(evaluation, json_path)
            print(f"✓ JSON: {json_path}")

        if args.output in ["markdown", "all"]:
            md_path = output_dir / f"{doc_name}_evaluation.md"
            formatter.save_markdown(evaluation, md_path)
            print(f"✓ Markdown: {md_path}")

    if args.output in ["csv", "all"]:
        csv_path = output_dir / "evaluations_summary.csv"
        formatter.to_csv(evaluations, csv_path)
        print(f"✓ CSV: {csv_path}")

    print()
    print("评估完成！")

    # 打印简要总结
    print("\n评估总结:")
    print("-" * 60)
    for evaluation in evaluations:
        doc_name = Path(evaluation.target_document).name
        print(
            f"{doc_name}: "
            f"完整性={evaluation.completeness:.2f}, "
            f"准确性={evaluation.accuracy:.2f}, "
            f"综合={evaluation.comprehensive:.2f}"
        )


if __name__ == "__main__":
    main()
