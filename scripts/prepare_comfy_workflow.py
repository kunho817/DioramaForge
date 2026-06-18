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

from diorama_forge.comfy_workflow import prepare_comfy_workflow_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-patch a ComfyUI API workflow with DioramaForge placeholders."
    )
    parser.add_argument("input", help="Path to a ComfyUI Save (API Format) workflow JSON.")
    parser.add_argument("output", help="Path to write the prepared workflow JSON.")
    parser.add_argument("--stage", default="stage3", choices=["stage3", "stage35", "refine"])
    parser.add_argument("--output-node-id", default="")
    parser.add_argument("--report", default="", help="Optional path to write a JSON preparation report.")
    args = parser.parse_args()

    input_path = _resolve(args.input)
    output_path = _resolve(args.output)
    result = prepare_comfy_workflow_bytes(
        data=input_path.read_bytes(),
        stage=args.stage,
        output_node_id=args.output_node_id,
    )
    report = {key: value for key, value in result.items() if key != "prepared_json"}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        report_path = _resolve(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result.get("ok"):
        raise SystemExit(1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(result["prepared_json"]), encoding="utf-8")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
