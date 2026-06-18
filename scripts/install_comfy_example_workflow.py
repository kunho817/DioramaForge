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
from diorama_forge.comfy_workflow import inspect_comfy_workflow, stage_key, validate_comfy_workflow
from diorama_forge.config import load_config


EXAMPLE_BY_STAGE = {
    "stage3": ROOT / "workflows" / "comfy" / "examples" / "stage3_sdxl_depth_img2img_api.example.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or install a bundled DioramaForge ComfyUI example workflow."
    )
    parser.add_argument("--stage", default="stage3", help="Workflow stage to prepare. Currently supports stage3.")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the bundled example into the configured workflow path. Omit for dry-run validation.",
    )
    parser.add_argument("--format", choices=("text", "json"), default="json")
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "default.json")
    client = ComfyUIClient(config.comfy)
    normalized_stage = stage_key(args.stage)
    example_path = EXAMPLE_BY_STAGE.get(normalized_stage)
    if example_path is None:
        _finish(
            {
                "ok": False,
                "stage": normalized_stage,
                "errors": [f"No bundled ComfyUI example workflow is available for stage: {args.stage}"],
            },
            args.format,
            exit_code=1,
        )
    if not example_path.exists():
        _finish(
            {
                "ok": False,
                "stage": normalized_stage,
                "example_path": str(example_path),
                "errors": [f"Bundled ComfyUI example workflow is missing: {example_path}"],
            },
            args.format,
            exit_code=1,
        )

    target_path = client.workflow_path_for_stage(normalized_stage)
    validation = validate_comfy_workflow(example_path, normalized_stage, output_node_id=config.comfy.output_node_id)
    inspection = inspect_comfy_workflow(example_path, normalized_stage, output_node_id=config.comfy.output_node_id)
    payload: dict[str, Any] = {
        "ok": bool(validation.get("ok")),
        "installed": False,
        "dry_run": not args.install,
        "stage": normalized_stage,
        "example_path": str(example_path),
        "target_path": str(target_path),
        "validation": validation,
        "inspection": _inspection_summary(inspection),
        "next_action": (
            f"Run this command with --install to write {target_path}."
            if validation.get("ok") and not args.install
            else "Fix the bundled example contract errors before installing."
        ),
    }
    if not validation.get("ok"):
        _finish(payload, args.format, exit_code=1)

    if args.install:
        install_result = client.install_workflow(normalized_stage, example_path.read_bytes())
        payload["install"] = install_result
        payload["installed"] = bool(install_result.get("ok"))
        payload["ok"] = bool(install_result.get("ok"))
        payload["next_action"] = (
            "Start ComfyUI and run scripts\\check_comfy_node_compatibility.py --require."
            if install_result.get("ok")
            else "Install failed; inspect install.errors."
        )
        if not install_result.get("ok"):
            _finish(payload, args.format, exit_code=1)

    _finish(payload, args.format, exit_code=0)


def _inspection_summary(inspection: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": inspection.get("format"),
        "node_count": inspection.get("node_count"),
        "class_types": inspection.get("class_types", []),
        "output_node_candidates": inspection.get("output_node_candidates", []),
        "suggested_stage3_mapping": inspection.get("suggested_stage3_mapping", []),
    }


def _finish(payload: dict[str, Any], output_format: str, exit_code: int) -> None:
    if output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        status = "ok" if payload.get("ok") else "failed"
        print(f"Stage: {payload.get('stage', '-')}")
        print(f"Status: {status}")
        print(f"Example: {payload.get('example_path', '-')}")
        print(f"Target: {payload.get('target_path', '-')}")
        print(f"Installed: {payload.get('installed', False)}")
        print(f"Next: {payload.get('next_action', '-')}")
        errors = payload.get("errors") or payload.get("validation", {}).get("errors", [])
        for error in errors:
            print(f"Error: {error}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
