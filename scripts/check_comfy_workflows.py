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

from diorama_forge.comfy import ComfyUIClient
from diorama_forge.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statically validate DioramaForge ComfyUI API workflow contracts."
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="Exit with failure when the required ComfyUI workflows are missing or invalid.",
    )
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "default.json")
    workflows = ComfyUIClient(config.comfy).workflow_status()
    stage35_required = str(config.product_pipeline.stage35_backend_mode).strip().lower() == "comfyui"
    refine_required = config.comfy.refine_workflow.exists()

    stage3_ok = bool(workflows["stage3"]["validation"].get("ok"))
    stage35_ok = bool(workflows["stage35"]["validation"].get("ok")) or not stage35_required
    refine_ok = bool(workflows["refine"]["validation"].get("ok")) or not refine_required
    failures = _failures(workflows, stage35_required, refine_required)

    payload: dict[str, Any] = {
        "ok": stage3_ok and stage35_ok and refine_ok,
        "required": {
            "stage3": True,
            "stage35": stage35_required,
            "refine": refine_required,
        },
        "failures": failures,
        "stage3": workflows["stage3"],
        "stage35": workflows["stage35"],
        "refine": workflows["refine"],
        "placeholders": workflows["placeholders"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.require and failures:
        raise SystemExit(1)


def _failures(workflows: dict[str, Any], stage35_required: bool, refine_required: bool) -> list[str]:
    failures: list[str] = []
    for name, required in (
        ("stage3", True),
        ("stage35", stage35_required),
        ("refine", refine_required),
    ):
        if not required:
            continue
        validation = workflows[name]["validation"]
        if validation.get("ok"):
            continue
        errors = validation.get("errors") or ["unknown workflow contract error"]
        failures.append(f"{name}: {errors[0]}")
    return failures


if __name__ == "__main__":
    main()
