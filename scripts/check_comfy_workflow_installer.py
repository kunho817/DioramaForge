from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.comfy_workflow import install_comfy_workflow_bytes, validate_comfy_workflow


def main() -> None:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="diorama_comfy_installer_") as tmp:
        target = Path(tmp) / "stage3_style_api.json"
        invalid_result = install_comfy_workflow_bytes(
            data=json.dumps({"nodes": []}).encode("utf-8"),
            target_path=target,
            stage="stage3",
        )
        if invalid_result.get("ok"):
            failures.append("Invalid UI-format workflow was accepted.")
        if target.exists():
            failures.append("Invalid workflow wrote the target file.")

        first_result = install_comfy_workflow_bytes(
            data=json.dumps(_valid_stage3_workflow("__PROMPT__")).encode("utf-8"),
            target_path=target,
            stage="stage3",
        )
        if not first_result.get("ok"):
            failures.append(f"Valid Stage 3 workflow was rejected: {first_result.get('errors')}")
        first_validation = validate_comfy_workflow(target, "stage3")
        if not first_validation.get("ok"):
            failures.append(f"Installed Stage 3 workflow did not validate: {first_validation.get('errors')}")

        second_result = install_comfy_workflow_bytes(
            data=json.dumps(_valid_stage3_workflow("__CLIP_PROMPT__")).encode("utf-8"),
            target_path=target,
            stage="stage3",
        )
        if not second_result.get("ok"):
            failures.append(f"Replacing Stage 3 workflow was rejected: {second_result.get('errors')}")
        if not second_result.get("backup_path"):
            failures.append("Replacing an existing workflow did not report a backup path.")
        elif not Path(str(second_result["backup_path"])).exists():
            failures.append(f"Reported backup path does not exist: {second_result['backup_path']}")

        example_target = Path(tmp) / "stage3_example_api.json"
        example_path = ROOT / "workflows" / "comfy" / "examples" / "stage3_sdxl_depth_img2img_api.example.json"
        if not example_path.exists():
            failures.append(f"Bundled Stage 3 example is missing: {example_path}")
        else:
            example_result = install_comfy_workflow_bytes(
                data=example_path.read_bytes(),
                target_path=example_target,
                stage="stage3",
            )
            if not example_result.get("ok"):
                failures.append(f"Bundled Stage 3 example did not install: {example_result.get('errors')}")

    summary = {
        "ok": not failures,
        "failures": failures,
        "checks": [
            "rejects_ui_format_export",
            "installs_valid_api_workflow",
            "backs_up_existing_workflow",
            "installs_bundled_stage3_example",
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _valid_stage3_workflow(prompt_placeholder: str) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": "__SOURCE_IMAGE__",
            },
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {
                "image": "__CONTROL_IMAGE__",
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt_placeholder,
            },
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "seed": "__SEED__",
                "steps": "__STEPS__",
                "cfg": "__GUIDANCE__",
                "denoise": "__DENOISE__",
            },
        },
        "5": {
            "class_type": "SaveImage",
            "inputs": {
                "images": [
                    "4",
                    0,
                ],
            },
        },
    }


if __name__ == "__main__":
    main()
