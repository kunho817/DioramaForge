from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.comfy_workflow_models import patch_workflow_model_fields, workflow_model_fields
from diorama_forge.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List or patch static model choices in the active DioramaForge Stage 3 ComfyUI API workflow."
    )
    parser.add_argument(
        "--workflow",
        default="",
        help="Workflow JSON path. Defaults to configs/default.json comfyui.stage3_workflow.",
    )
    parser.add_argument("--checkpoint", default="", help="Checkpoint filename for CheckpointLoaderSimple.ckpt_name.")
    parser.add_argument("--controlnet", default="", help="ControlNet filename for ControlNetLoader.control_net_name.")
    parser.add_argument("--lora", default="", help="LoRA filename for LoraLoader.lora_name when present.")
    parser.add_argument("--vae", default="", help="VAE filename for VAELoader.vae_name when present.")
    parser.add_argument("--upscale-model", default="", help="Upscale model filename for UpscaleModelLoader.model_name when present.")
    parser.add_argument("--sampler", default="", help="Sampler name for KSampler.sampler_name.")
    parser.add_argument("--scheduler", default="", help="Scheduler name for KSampler.scheduler.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print changes without writing the workflow.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create a timestamped backup before writing.")
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "default.json")
    workflow_path = Path(args.workflow) if args.workflow else config.comfy.stage3_workflow
    if not workflow_path.is_absolute():
        workflow_path = config.root / workflow_path

    updates = {
        "checkpoint": args.checkpoint,
        "controlnet": args.controlnet,
        "lora": args.lora,
        "vae": args.vae,
        "upscale_model": args.upscale_model,
        "sampler": args.sampler,
        "scheduler": args.scheduler,
    }
    updates = {key: value for key, value in updates.items() if value}

    if not updates:
        payload = {
            "ok": workflow_path.exists(),
            "path": str(workflow_path),
            "fields": [field.as_dict() for field in workflow_model_fields(workflow_path)] if workflow_path.exists() else [],
            "next_action": (
                "Pass --checkpoint and/or --controlnet to patch filenames for your portable ComfyUI models."
                if workflow_path.exists()
                else "Install the Stage 3 workflow first."
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not payload["ok"]:
            raise SystemExit(1)
        return

    result = patch_workflow_model_fields(
        workflow_path,
        updates,
        stage="stage3",
        output_node_id=config.comfy.output_node_id,
        dry_run=args.dry_run,
        backup=not args.no_backup,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
