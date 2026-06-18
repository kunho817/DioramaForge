from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.comfy_workflow import prepare_comfy_workflow_bytes, validate_comfy_workflow


def main() -> None:
    failures: list[str] = []
    valid_result = prepare_comfy_workflow_bytes(
        data=json.dumps(_raw_stage3_api_workflow()).encode("utf-8"),
        stage="stage3",
    )
    if not valid_result.get("ok"):
        failures.append(f"Raw Stage 3 API workflow was not prepared: {valid_result.get('errors')}")
    if len(valid_result.get("changes", [])) < 6:
        failures.append("Stage 3 preparation made too few placeholder changes.")
    if "__SOURCE_IMAGE__" not in str(valid_result.get("prepared_json", "")):
        failures.append("Prepared workflow is missing __SOURCE_IMAGE__.")
    if "__CONTROL_IMAGE__" not in str(valid_result.get("prepared_json", "")):
        failures.append("Prepared workflow is missing __CONTROL_IMAGE__.")

    with tempfile.TemporaryDirectory(prefix="diorama_comfy_prepare_check_") as tmp:
        prepared_path = Path(tmp) / "prepared.json"
        prepared_path.write_text(str(valid_result.get("prepared_json", "")), encoding="utf-8")
        validation = validate_comfy_workflow(prepared_path, "stage3")
        if not validation.get("ok"):
            failures.append(f"Prepared workflow did not validate: {validation.get('errors')}")

    invalid_result = prepare_comfy_workflow_bytes(
        data=json.dumps({"nodes": [{"id": 1, "type": "LoadImage"}]}).encode("utf-8"),
        stage="stage3",
    )
    if invalid_result.get("ok"):
        failures.append("UI-format workflow was prepared even though it should be rejected.")

    summary = {
        "ok": not failures,
        "failures": failures,
        "checks": [
            "auto_patches_stage3_api_workflow",
            "prepared_workflow_validates",
            "rejects_ui_format_export",
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _raw_stage3_api_workflow() -> dict[str, object]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "control.png"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "fantasy diorama"}},
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 123,
                "steps": 4,
                "cfg": 3.5,
                "denoise": 0.45,
            },
        },
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}},
    }


if __name__ == "__main__":
    main()
