from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.comfy_workflow import inspect_comfy_workflow


def main() -> None:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="diorama_comfy_inspector_") as tmp:
        tmp_path = Path(tmp)
        api_path = tmp_path / "api.json"
        api_path.write_text(json.dumps(_api_workflow()), encoding="utf-8")
        api_report = inspect_comfy_workflow(api_path, "stage3")
        if api_report.get("format") != "api":
            failures.append("API workflow format was not detected.")
        if not api_report.get("load_image_candidates"):
            failures.append("LoadImage candidates were not detected.")
        if not api_report.get("text_candidates"):
            failures.append("Text candidates were not detected.")
        if not api_report.get("sampler_candidates"):
            failures.append("Sampler candidates were not detected.")
        if not api_report.get("output_node_candidates"):
            failures.append("Output candidates were not detected.")

        ui_path = tmp_path / "ui.json"
        ui_path.write_text(json.dumps({"nodes": [{"id": 1, "type": "LoadImage"}]}), encoding="utf-8")
        ui_report = inspect_comfy_workflow(ui_path, "stage3")
        if ui_report.get("format") != "ui":
            failures.append("UI workflow format was not detected.")

    summary = {
        "ok": not failures,
        "failures": failures,
        "checks": [
            "detects_api_format",
            "detects_ui_format",
            "finds_placeholder_mapping_candidates",
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _api_workflow() -> dict[str, object]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "control.png"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "fantasy diorama"}},
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 1,
                "steps": 4,
                "cfg": 3.5,
                "denoise": 0.45,
            },
        },
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}},
    }


if __name__ == "__main__":
    main()
