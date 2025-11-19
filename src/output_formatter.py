"""输出格式化模块"""

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from src.evaluator import DocumentEvaluation


class OutputFormatter:
    """输出格式化器"""

    @staticmethod
    def to_json(evaluation: DocumentEvaluation) -> dict[str, Any]:
        """
        转换为JSON格式

        Args:
            evaluation: 评估结果

        Returns:
            JSON格式的字典
        """
        checkpoint_results_data = [
            {
                "checkpoint": cp.checkpoint,
                "passed": cp.passed,
            }
            for cp in evaluation.checkpoint_results
        ]

        return {
            "target_document": evaluation.target_document,
            "scores": {
                "completeness": round(evaluation.completeness, 2),
                "accuracy": round(evaluation.accuracy, 2),
                "comprehensive": round(evaluation.comprehensive, 2),
            },
            "checkpoints": evaluation.checkpoints,
            "checkpoint_results": checkpoint_results_data,
        }

    @staticmethod
    def to_csv(
        evaluations: list[DocumentEvaluation], output_path: str | Path | None = None
    ) -> str:
        """
        转换为CSV格式

        Args:
            evaluations: 评估结果列表
            output_path: 输出路径，如果为None则返回字符串

        Returns:
            CSV格式的字符串（如果output_path为None）
        """
        rows = []
        for eval_result in evaluations:
            doc_name = Path(eval_result.target_document).name
            rows.append(
                {
                    "文档名": doc_name,
                    "完整性分数": round(eval_result.completeness, 2),
                    "准确性分数": round(eval_result.accuracy, 2),
                    "综合分数": round(eval_result.comprehensive, 2),
                }
            )

        if output_path:
            with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["文档名", "完整性分数", "准确性分数", "综合分数"]
                )
                writer.writeheader()
                writer.writerows(rows)
            return ""
        else:
            # 返回字符串
            output = []
            output.append(",".join(["文档名", "完整性分数", "准确性分数", "综合分数"]))
            for row in rows:
                output.append(
                    ",".join(
                        [
                            row["文档名"],
                            str(row["完整性分数"]),
                            str(row["准确性分数"]),
                            str(row["综合分数"]),
                        ]
                    )
                )
            return "\n".join(output)

    @staticmethod
    def to_markdown(evaluation: DocumentEvaluation) -> str:
        """
        转换为Markdown格式报告

        Args:
            evaluation: 评估结果

        Returns:
            Markdown格式的字符串
        """
        lines = []
        doc_name = Path(evaluation.target_document).name

        # 标题
        lines.append(f"# 需求文档完整性评估报告: {doc_name}\n")

        # 评估信息
        lines.append("## 评估信息\n")
        lines.append("| 项目 | 内容 |")
        lines.append("|------|------|")
        if evaluation.evaluation_time:
            lines.append(f"| 评估时间 | {evaluation.evaluation_time} |")
        if evaluation.model_name:
            lines.append(f"| 评估模型 | {evaluation.model_name} |")
        if evaluation.baseline_document:
            baseline_name = Path(evaluation.baseline_document).name
            lines.append(f"| 基准文档 | {baseline_name} |")
        lines.append(f"| 评估文档 | {doc_name} |")
        if evaluation.evaluation_duration is not None:
            lines.append(f"| 评估耗时 | {evaluation.evaluation_duration:.1f}秒 |")
        lines.append("")

        # 合并的统计信息部分
        total_checkpoints = len(evaluation.checkpoints)
        lines.append("## 统计信息\n")
        
        # 如果有多个评委的结果，显示两种策略的结果
        if evaluation.all_judge_results and len(evaluation.all_judge_results) > 1:
            num_judges = len(evaluation.all_judge_results)
            
            # 获取两种策略的结果
            checkpoint_result = evaluation.checkpoint_merge_result if evaluation.checkpoint_merge_result else evaluation
            accuracy_result = evaluation.accuracy_merge_result if evaluation.accuracy_merge_result else evaluation
            
            # 计算多数投票统计（用于检查项合并策略）
            majority_passed_count = 0
            for index in range(total_checkpoints):
                passed_votes = 0
                for judge_idx in range(num_judges):
                    if index < len(evaluation.all_judge_results[judge_idx]):
                        if evaluation.all_judge_results[judge_idx][index].passed:
                            passed_votes += 1
                if passed_votes > (num_judges / 2):
                    majority_passed_count += 1
            
            majority_completeness = (majority_passed_count / total_checkpoints) * 100 if total_checkpoints > 0 else 0.0
            
            # 计算平均通过数量
            avg_passed_count = statistics.mean([
                sum(1 for cp in judge_results if cp.passed)
                for judge_results in evaluation.all_judge_results
            ])
            avg_completeness = (avg_passed_count / total_checkpoints) * 100 if total_checkpoints > 0 else 0.0
            
            # 显示说明
            lines.append("**投票通过**：每个检查项按多数投票判定结果是否通过（超过50%评委认为通过则通过），然后统计所有投票通过的项占比。\n")
            lines.append("**平均通过**：对每个评委的通过项数量进行平均计算，得到平均通过率。\n")
            lines.append("")
            
            # 显示统计结果表格（包含检查项总数，优化对齐）
            lines.append("| 项目 | 数量 | 得分 |")
            lines.append("|:-----|:----:|:----:|")
            lines.append(f"| 检查项总数 | {total_checkpoints} | - |")
            lines.append(f"| 投票通过 | {majority_passed_count} | {majority_completeness:.2f} |")
            lines.append(f"| 平均通过 | {avg_passed_count:.1f} | {avg_completeness:.2f} |")
            lines.append("")
            
            # 每个评委的统计信息
            lines.append("### 评委统计信息\n")
            # 根据评委数量调整表格列宽
            lines.append("| 评委 | 通过数量 | 分数 |")
            lines.append("|:-----|:--------:|:----:|")
            
            for judge_idx in range(num_judges):
                judge_results = evaluation.all_judge_results[judge_idx]
                passed_count = sum(1 for cp in judge_results if cp.passed)
                completeness = (passed_count / total_checkpoints) * 100 if total_checkpoints > 0 else 0.0
                lines.append(f"| 评委{judge_idx+1} | {passed_count} | {completeness:.2f} |")
            
            lines.append("")
        else:
            # 只有一个评委或没有保存多个评委结果，只显示基本统计
            lines.append("| 项目 | 数值 |")
            lines.append("|------|------|")
            lines.append(f"| 检查项总数 | {total_checkpoints} |")
            
            # 如果完整性、准确性和综合分数都相同，合并显示
            if (abs(evaluation.completeness - evaluation.accuracy) < 0.01 and 
                abs(evaluation.accuracy - evaluation.comprehensive) < 0.01):
                lines.append(f"| 综合分数 | {evaluation.completeness:.2f} |")
            else:
                lines.append(f"| 完整性分数 | {evaluation.completeness:.2f} |")
                lines.append(f"| 准确性分数 | {evaluation.accuracy:.2f} |")
                lines.append(f"| 综合分数 | {evaluation.comprehensive:.2f} |")
            
            lines.append("")

        # 检查项详细结果
        lines.append("## 检查项详细结果\n")
        
        # 如果有多个评委的结果，显示每个评委的结果和多数投票
        if evaluation.all_judge_results and len(evaluation.all_judge_results) > 1:
            num_judges = len(evaluation.all_judge_results)
            # 构建表头（优化对齐）
            header = "| 索引 | 检查项 |"
            separator = "|:-----|:-------|"
            for i in range(num_judges):
                header += f" 评委{i+1} |"
                separator += ":-----:|"
            header += " 多数投票 |"
            separator += ":------:|"
            
            lines.append(header)
            lines.append(separator)
            
            # 对每个检查项，显示每个评委的结果和多数投票
            for index, checkpoint in enumerate(evaluation.checkpoints):
                row = f"| {index} | {checkpoint} |"
                
                # 收集每个评委的判断
                judge_votes = []
                for judge_idx in range(num_judges):
                    if index < len(evaluation.all_judge_results[judge_idx]):
                        judge_result = evaluation.all_judge_results[judge_idx][index]
                        passed = judge_result.passed
                        judge_votes.append(passed)
                        status_str = "✓" if passed else "✗"
                        row += f" {status_str} |"
                    else:
                        judge_votes.append(False)
                        row += " ? |"
                
                # 计算多数投票（超过50%认为通过则通过）
                passed_count = sum(judge_votes)
                majority_passed = passed_count > (num_judges / 2)
                majority_str = "✓ 通过" if majority_passed else "✗ 未通过"
                row += f" {majority_str} |"
                
                lines.append(row)
        else:
            # 只有一个评委或没有保存多个评委结果，使用原来的格式
            lines.append("| 索引 | 检查项 | 状态 |")
            lines.append("|------|--------|------|")
            for index, cp_result in enumerate(evaluation.checkpoint_results):
                status_str = "✓ 通过" if cp_result.passed else "✗ 未通过"
                lines.append(f"| {index} | {cp_result.checkpoint} | {status_str} |")
        
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def save_json(
        evaluation: DocumentEvaluation, output_path: str | Path
    ) -> None:
        """
        保存为JSON文件

        Args:
            evaluation: 评估结果
            output_path: 输出路径
        """
        data = OutputFormatter.to_json(evaluation)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def save_markdown(
        evaluation: DocumentEvaluation, output_path: str | Path
    ) -> None:
        """
        保存为Markdown文件

        Args:
            evaluation: 评估结果
            output_path: 输出路径
        """
        content = OutputFormatter.to_markdown(evaluation)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def generate_summary_report(
        evaluations: list[DocumentEvaluation],
        baseline_document: str | Path | None = None,
    ) -> str:
        """
        生成聚合统计报告

        Args:
            evaluations: 评估结果列表
            baseline_document: 基准文档路径（可选）

        Returns:
            Markdown格式的聚合统计报告
        """
        if not evaluations:
            return "# 聚合统计报告\n\n没有评估结果。\n"

        lines = []
        lines.append("# 批量评估聚合统计报告\n")
        lines.append("")

        # 评估信息
        lines.append("## 评估信息\n")
        lines.append("| 项目 | 内容 |")
        lines.append("|------|------|")
        if baseline_document:
            baseline_name = Path(baseline_document).name
            lines.append(f"| 基准文档 | {baseline_name} |")
        lines.append(f"| 评估文档总数 | {len(evaluations)} |")
        if evaluations and evaluations[0].model_name:
            lines.append(f"| 评估模型 | {evaluations[0].model_name} |")
        if evaluations and evaluations[0].evaluation_time:
            lines.append(f"| 评估时间 | {evaluations[0].evaluation_time} |")
        lines.append("")
        lines.append("")

        # 统计信息
        completeness_scores = [e.completeness for e in evaluations]
        accuracy_scores = [e.accuracy for e in evaluations]
        comprehensive_scores = [e.comprehensive for e in evaluations]

        lines.append("## 统计摘要\n")
        lines.append("")

        # 完整性统计
        lines.append("### 完整性分数统计\n")
        lines.append("| 统计项 | 数值 |")
        lines.append("|:-------|:----:|")
        lines.append(f"| 平均值 | {statistics.mean(completeness_scores):.2f} |")
        lines.append(f"| 中位数 | {statistics.median(completeness_scores):.2f} |")
        lines.append(f"| 最大值 | {max(completeness_scores):.2f} |")
        lines.append(f"| 最小值 | {min(completeness_scores):.2f} |")
        if len(completeness_scores) > 1:
            lines.append(f"| 标准差 | {statistics.stdev(completeness_scores):.2f} |")
        lines.append("")
        lines.append("")

        # 准确性统计
        lines.append("### 准确性分数统计\n")
        lines.append("| 统计项 | 数值 |")
        lines.append("|:-------|:----:|")
        lines.append(f"| 平均值 | {statistics.mean(accuracy_scores):.2f} |")
        lines.append(f"| 中位数 | {statistics.median(accuracy_scores):.2f} |")
        lines.append(f"| 最大值 | {max(accuracy_scores):.2f} |")
        lines.append(f"| 最小值 | {min(accuracy_scores):.2f} |")
        if len(accuracy_scores) > 1:
            lines.append(f"| 标准差 | {statistics.stdev(accuracy_scores):.2f} |")
        lines.append("")
        lines.append("")

        # 综合分数统计
        lines.append("### 综合分数统计\n")
        lines.append("| 统计项 | 数值 |")
        lines.append("|:-------|:----:|")
        lines.append(f"| 平均值 | {statistics.mean(comprehensive_scores):.2f} |")
        lines.append(f"| 中位数 | {statistics.median(comprehensive_scores):.2f} |")
        lines.append(f"| 最大值 | {max(comprehensive_scores):.2f} |")
        lines.append(f"| 最小值 | {min(comprehensive_scores):.2f} |")
        if len(comprehensive_scores) > 1:
            lines.append(f"| 标准差 | {statistics.stdev(comprehensive_scores):.2f} |")
        lines.append("")
        lines.append("")

        # 详细列表
        lines.append("## 详细评估结果\n")
        lines.append("")
        lines.append("| 文档名 | 完整性 | 准确性 | 综合分数 |")
        lines.append("|:-------|:------:|:------:|:--------:|")
        
        # 按综合分数排序（降序）
        sorted_evaluations = sorted(evaluations, key=lambda e: e.comprehensive, reverse=True)
        for evaluation in sorted_evaluations:
            doc_name = Path(evaluation.target_document).name
            lines.append(
                f"| {doc_name} | {evaluation.completeness:.2f} | "
                f"{evaluation.accuracy:.2f} | {evaluation.comprehensive:.2f} |"
            )
        lines.append("")
        lines.append("")

        # 分数分布
        lines.append("## 分数分布\n")
        lines.append("")

        # 完整性分布
        lines.append("### 完整性分数分布\n")
        score_ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
        lines.append("| 分数区间 | 文档数量 | 占比 |")
        lines.append("|:---------|:--------:|:----:|")
        for low, high in score_ranges:
            count = sum(1 for s in completeness_scores if low <= s < high)
            if high == 100:
                count = sum(1 for s in completeness_scores if low <= s <= high)
            percentage = (count / len(completeness_scores)) * 100 if completeness_scores else 0
            lines.append(f"| {low}-{high} | {count} | {percentage:.1f}% |")
        lines.append("")
        lines.append("")

        # 准确性分布
        lines.append("### 准确性分数分布\n")
        lines.append("| 分数区间 | 文档数量 | 占比 |")
        lines.append("|:---------|:--------:|:----:|")
        for low, high in score_ranges:
            count = sum(1 for s in accuracy_scores if low <= s < high)
            if high == 100:
                count = sum(1 for s in accuracy_scores if low <= s <= high)
            percentage = (count / len(accuracy_scores)) * 100 if accuracy_scores else 0
            lines.append(f"| {low}-{high} | {count} | {percentage:.1f}% |")
        lines.append("")
        lines.append("")

        # 综合分数分布
        lines.append("### 综合分数分布\n")
        lines.append("| 分数区间 | 文档数量 | 占比 |")
        lines.append("|:---------|:--------:|:----:|")
        for low, high in score_ranges:
            count = sum(1 for s in comprehensive_scores if low <= s < high)
            if high == 100:
                count = sum(1 for s in comprehensive_scores if low <= s <= high)
            percentage = (count / len(comprehensive_scores)) * 100 if comprehensive_scores else 0
            lines.append(f"| {low}-{high} | {count} | {percentage:.1f}% |")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def save_summary_report(
        evaluations: list[DocumentEvaluation],
        output_path: str | Path,
        baseline_document: str | Path | None = None,
    ) -> None:
        """
        保存聚合统计报告

        Args:
            evaluations: 评估结果列表
            output_path: 输出路径
            baseline_document: 基准文档路径（可选）
        """
        content = OutputFormatter.generate_summary_report(evaluations, baseline_document)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

