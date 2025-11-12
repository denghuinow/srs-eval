"""输出格式化模块"""

import csv
import json
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
        return {
            "target_document": evaluation.target_document,
            "scores": {
                "completeness": round(evaluation.completeness, 2),
                "accuracy": round(evaluation.accuracy, 2),
                "comprehensive": round(evaluation.comprehensive, 2),
            },
            "points": evaluation.points,
            "evaluations": [
                {
                    "point_id": e.point_id,
                    "exists": e.exists,
                    "accuracy": round(e.accuracy, 2),
                    "explanation": e.explanation,
                }
                for e in evaluation.evaluations
            ],
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
        lines.append(f"# 需求文档评估报告: {doc_name}\n")

        # 总体分数
        lines.append("## 总体分数\n")
        lines.append("| 维度 | 分数 |")
        lines.append("|------|------|")
        lines.append(f"| 完整性 | {evaluation.completeness:.2f} |")
        lines.append(f"| 准确性 | {evaluation.accuracy:.2f} |")
        lines.append(f"| 综合分数 | {evaluation.comprehensive:.2f} |")
        lines.append("")

        # 要点清单
        lines.append("## 要点清单\n")
        for point in evaluation.points:
            level = point.get("level", 1)
            indent = "  " * (level - 1)
            title = point.get("title", "")
            description = point.get("description", "")
            point_id = point.get("id", "")
            lines.append(f"{indent}- **{point_id} {title}**")
            if description:
                lines.append(f"{indent}  {description}")
        lines.append("")

        # 详细评估结果
        lines.append("## 详细评估结果\n")
        lines.append("| 要点ID | 存在 | 准确性 | 说明 |")
        lines.append("|--------|------|--------|------|")
        for eval_result in evaluation.evaluations:
            exists_str = "✓" if eval_result.exists else "✗"
            lines.append(
                f"| {eval_result.point_id} | {exists_str} | "
                f"{eval_result.accuracy:.2f} | {eval_result.explanation} |"
            )
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

