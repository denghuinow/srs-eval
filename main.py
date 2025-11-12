"""主程序入口"""

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.evaluator import Evaluator
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
        points = extractor.extract_points(baseline_path)
        print(f"✓ 成功提取 {len(points)} 个要点")
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

    # 评估文档
    evaluator = Evaluator(config)
    evaluations = []

    for target_path in target_paths:
        print(f"正在评估文档: {target_path}")
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
