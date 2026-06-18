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

from diorama_forge.comfy import _workflow_node_compatibility_with_object_info


def main() -> None:
    checks = [
        _check_accepts_available_model_choice(),
        _check_rejects_missing_model_choice(),
        _check_skips_runtime_placeholder_and_links(),
        _check_rejects_missing_node_class(),
    ]
    failures = [check for check in checks if not check["ok"]]
    print(json.dumps({"ok": not failures, "failures": failures, "checks": checks}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _check_accepts_available_model_choice() -> dict[str, Any]:
    workflow = _workflow(ckpt_name="fast_depth_model.safetensors")
    result = _compatibility(workflow)
    return {
        "name": "accepts_available_model_choice",
        "ok": bool(result.get("ok")) and not result.get("invalid_input_choices"),
        "result": _summary(result),
    }


def _check_rejects_missing_model_choice() -> dict[str, Any]:
    workflow = _workflow(ckpt_name="missing_model.safetensors")
    result = _compatibility(workflow)
    invalid = result.get("invalid_input_choices") or []
    return {
        "name": "rejects_missing_model_choice",
        "ok": not result.get("ok") and bool(invalid) and invalid[0].get("field") == "ckpt_name",
        "result": _summary(result),
    }


def _check_skips_runtime_placeholder_and_links() -> dict[str, Any]:
    workflow = _workflow(ckpt_name="__MODEL_NAME__")
    workflow["4"] = {
        "class_type": "ControlNetLoader",
        "inputs": {
            "control_net_name": "depth.safetensors",
            "image": ["2", 0],
        },
    }
    object_info = _object_info()
    object_info["ControlNetLoader"] = {
        "input": {
            "required": {
                "control_net_name": [["depth.safetensors"], {}],
                "image": ["IMAGE", {}],
            }
        }
    }
    result = _compatibility(workflow, object_info)
    return {
        "name": "skips_runtime_placeholder_and_links",
        "ok": bool(result.get("ok")) and not result.get("invalid_input_choices"),
        "result": _summary(result),
    }


def _check_rejects_missing_node_class() -> dict[str, Any]:
    workflow = _workflow(ckpt_name="fast_depth_model.safetensors")
    object_info = _object_info()
    object_info.pop("KSampler")
    result = _compatibility(workflow, object_info)
    return {
        "name": "rejects_missing_node_class",
        "ok": not result.get("ok") and "KSampler" in result.get("missing_class_types", []),
        "result": _summary(result),
    }


def _compatibility(workflow: dict[str, Any], object_info: dict[str, Any] | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="diorama_comfy_node_contract_") as tmp:
        path = Path(tmp) / "workflow.json"
        path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
        return _workflow_node_compatibility_with_object_info(path, object_info or _object_info())


def _workflow(ckpt_name: str) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": ckpt_name,
            },
        },
        "2": {
            "class_type": "KSampler",
            "inputs": {
                "seed": "__SEED__",
                "steps": "__STEPS__",
                "cfg": "__GUIDANCE__",
                "denoise": "__DENOISE__",
            },
        },
        "3": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "diorama_forge",
                "images": ["2", 0],
            },
        },
    }


def _object_info() -> dict[str, Any]:
    return {
        "CheckpointLoaderSimple": {
            "input": {
                "required": {
                    "ckpt_name": [["fast_depth_model.safetensors"], {}],
                }
            }
        },
        "KSampler": {
            "input": {
                "required": {
                    "seed": ["INT", {}],
                    "steps": ["INT", {}],
                    "cfg": ["FLOAT", {}],
                    "denoise": ["FLOAT", {}],
                }
            }
        },
        "SaveImage": {
            "input": {
                "required": {
                    "images": ["IMAGE", {}],
                    "filename_prefix": ["STRING", {}],
                }
            }
        },
    }


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": result.get("ok"),
        "missing_class_types": result.get("missing_class_types", []),
        "invalid_input_choices": result.get("invalid_input_choices", []),
        "error": result.get("error", ""),
    }


if __name__ == "__main__":
    main()
