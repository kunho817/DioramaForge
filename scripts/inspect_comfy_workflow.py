from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.comfy_workflow import inspect_comfy_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a ComfyUI workflow JSON and suggest DioramaForge placeholder mapping points."
    )
    parser.add_argument("workflow", help="Path to a ComfyUI workflow JSON file.")
    parser.add_argument("--stage", default="stage3", choices=["stage3", "stage35", "refine"])
    parser.add_argument("--output-node-id", default="")
    parser.add_argument("--format", default="markdown", choices=["markdown", "json"])
    args = parser.parse_args()

    path = Path(args.workflow)
    if not path.is_absolute():
        path = ROOT / path
    report = inspect_comfy_workflow(path, args.stage, output_node_id=args.output_node_id)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_markdown(report))
    if report.get("format") in {"invalid_json", "missing", "unknown"}:
        raise SystemExit(1)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ComfyUI Workflow Inspection",
        "",
        f"- Path: `{report.get('path')}`",
        f"- Format: `{report.get('format')}`",
        f"- Stage: `{report.get('stage')}`",
        f"- Nodes: {report.get('node_count', 0)}",
        "",
    ]
    errors = report.get("errors") or []
    if errors:
        lines.extend(["## Errors", ""])
        lines.extend(f"- {item}" for item in errors)
        lines.append("")

    validation = report.get("validation") or {}
    if validation:
        lines.extend(
            [
                "## Contract Validation",
                "",
                f"- Status: {'pass' if validation.get('ok') else 'needs edits'}",
            ]
        )
        for item in validation.get("errors", []):
            lines.append(f"- Error: {item}")
        for item in validation.get("warnings", []):
            lines.append(f"- Warning: {item}")
        lines.append("")

    _candidate_table(lines, "Load Image Candidates", report.get("load_image_candidates", []), ("node_id", "class_type", "field", "value_preview"))
    _candidate_table(
        lines,
        "Text Candidates",
        report.get("text_candidates", []),
        ("node_id", "class_type", "field", "suggested_placeholder", "value_preview"),
    )
    _candidate_table(lines, "Sampler Candidates", report.get("sampler_candidates", []), ("node_id", "class_type", "fields"))
    _candidate_table(lines, "Size Candidates", report.get("size_candidates", []), ("node_id", "class_type", "fields"))
    _candidate_table(lines, "Output Candidates", report.get("output_node_candidates", []), ("node_id", "class_type"))

    suggestions = report.get("suggested_stage3_mapping") or []
    if suggestions:
        lines.extend(["## Suggested Stage 3 Placeholder Mapping", ""])
        for item in suggestions:
            lines.append(f"- `{item.get('placeholder')}`: {item.get('use')}")
        lines.append("")

    if report.get("format") == "ui":
        lines.extend(
            [
                "## Next Step",
                "",
                "Export the workflow again from ComfyUI with **Save (API Format)**, then run this inspector on that file.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Next Step",
                "",
                "Replace the relevant workflow input values with the suggested placeholders, then validate with:",
                "",
                "```powershell",
                ".\\.venv\\Scripts\\python.exe scripts\\check_comfy_workflows.py",
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _candidate_table(lines: list[str], title: str, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    lines.extend([f"## {title}", ""])
    if not rows:
        lines.extend(["No candidates found.", ""])
        return
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows[:20]:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, dict):
                value = ", ".join(f"{key}={item}" for key, item in value.items())
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")


if __name__ == "__main__":
    main()
