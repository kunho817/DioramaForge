from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.comfy_workflow_models import patch_workflow_model_fields, workflow_model_fields


def main() -> None:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="diorama-comfy-models-") as tmp:
        workflow_path = Path(tmp) / "stage3_style_api.json"
        workflow_path.write_text(json.dumps(_workflow(), ensure_ascii=False, indent=2), encoding="utf-8")

        fields = [field.as_dict() for field in workflow_model_fields(workflow_path)]
        if not _has_field(fields, "checkpoint", "old_checkpoint.safetensors"):
            failures.append("initial checkpoint field was not detected")
        if not _has_field(fields, "controlnet", "old_controlnet.safetensors"):
            failures.append("initial ControlNet field was not detected")

        dry = patch_workflow_model_fields(
            workflow_path,
            {"checkpoint": "new_checkpoint.safetensors", "controlnet": "new_controlnet.safetensors"},
            dry_run=True,
        )
        persisted_after_dry = json.loads(workflow_path.read_text(encoding="utf-8"))
        if persisted_after_dry["3"]["inputs"]["ckpt_name"] != "old_checkpoint.safetensors":
            failures.append("dry-run unexpectedly modified the workflow file")
        if not dry["validation"].get("ok"):
            failures.append("dry-run patched workflow did not validate")

        patched = patch_workflow_model_fields(
            workflow_path,
            {
                "checkpoint": "new_checkpoint.safetensors",
                "controlnet": "new_controlnet.safetensors",
                "sampler": "dpmpp_2m",
                "scheduler": "karras",
            },
        )
        persisted = json.loads(workflow_path.read_text(encoding="utf-8"))
        if persisted["3"]["inputs"]["ckpt_name"] != "new_checkpoint.safetensors":
            failures.append("checkpoint patch was not persisted")
        if persisted["4"]["inputs"]["control_net_name"] != "new_controlnet.safetensors":
            failures.append("controlnet patch was not persisted")
        if persisted["9"]["inputs"]["sampler_name"] != "dpmpp_2m":
            failures.append("sampler patch was not persisted")
        if persisted["9"]["inputs"]["scheduler"] != "karras":
            failures.append("scheduler patch was not persisted")
        if not patched.get("backup_path") or not Path(patched["backup_path"]).exists():
            failures.append("patch did not create a backup file")
        if not patched["validation"].get("ok"):
            failures.append("patched workflow did not validate")

        missing = patch_workflow_model_fields(workflow_path, {"lora": "unused_lora.safetensors"}, dry_run=True)
        if "lora" not in missing.get("missing_update_targets", []):
            failures.append("missing update target was not reported")

    payload = {
        "ok": not failures,
        "failures": failures,
        "checks": [
            "detects_model_fields",
            "dry_run_does_not_write",
            "patches_model_fields",
            "creates_backup",
            "reports_missing_targets",
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _has_field(fields: list[dict[str, str]], key: str, value: str) -> bool:
    return any(field.get("key") == key and field.get("value") == value for field in fields)


def _workflow() -> dict[str, object]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "__SOURCE_IMAGE__"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "__CONTROL_IMAGE__"}},
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "old_checkpoint.safetensors"}},
        "4": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "old_controlnet.safetensors"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 1], "text": "__PROMPT__"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 1], "text": "__NEGATIVE_PROMPT__"}},
        "7": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "control_net": ["4", 0],
                "image": ["2", 0],
                "strength": "__STRENGTH__",
                "start_percent": 0.0,
                "end_percent": 0.85,
            },
        },
        "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["3", 2]}},
        "9": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["8", 0],
                "seed": "__SEED__",
                "steps": "__STEPS__",
                "cfg": "__GUIDANCE__",
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": "__DENOISE__",
            },
        },
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 2]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "diorama_forge_stage3"}},
    }


if __name__ == "__main__":
    main()
