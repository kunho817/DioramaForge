from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.config import load_config
from diorama_forge.model_status import hf_token_available, project_hf_cache, project_hf_home


MODEL_KEYS = (
    "depth",
    "sam",
    "flux",
    "sdxl_base",
    "sdxl_controlnet",
    "sdxl_lora",
    "trellis",
    "ultrashape",
)
MODEL_GROUPS = {
    "style_sdxl": ("sdxl_base", "sdxl_controlnet", "sdxl_lora"),
}


def download_repo(
    repo_id: str,
    cache_dir: Path,
    token: str | None,
    allow_patterns: list[str] | None = None,
) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        token=token,
        local_files_only=False,
        resume_download=True,
        allow_patterns=allow_patterns,
    )
    return {"repo_id": repo_id, "path": path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DioramaForge model snapshots into the project cache.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*MODEL_KEYS, *MODEL_GROUPS.keys(), "all"],
        default=["depth", "sam"],
        help="Models to download. Use style_sdxl for SDXL base + depth ControlNet + Lightning LoRA.",
    )
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Prefer env HF_TOKEN for persistence.")
    parser.add_argument(
        "--hf-token-file",
        default="",
        help="Optional file containing a Hugging Face token. The token value is not printed.",
    )
    parser.add_argument(
        "--include-flux-without-token",
        action="store_true",
        help="Attempt FLUX download even when no Hugging Face token is present.",
    )
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "default.json")
    hf_home = project_hf_home(config.root)
    cache_dir = project_hf_cache(config.root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))

    selected = set(MODEL_KEYS if "all" in args.models else args.models)
    for group_name, members in MODEL_GROUPS.items():
        if group_name in selected:
            selected.remove(group_name)
            selected.update(members)
    token = (
        args.hf_token
        or _read_token_file(args.hf_token_file)
        or _read_token_file(str(ROOT / "key" / "hf.txt"))
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    results: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    repo_by_key = {
        "depth": config.depth.model_id,
        "sam": config.sam.model_id,
        "flux": config.flux.model_id,
        "sdxl_base": config.sdxl_depth_lightning.base_model_id,
        "sdxl_controlnet": config.sdxl_depth_lightning.controlnet_model_id,
        "sdxl_lora": config.sdxl_depth_lightning.lora_model_id,
        "trellis": config.trellis.model_id,
        "ultrashape": config.ultrashape.model_id,
    }
    allow_patterns_by_key = {
        "sdxl_lora": [
            f"*{config.sdxl_depth_lightning.lora_weight_name}",
            "*.json",
            "*.md",
            "*.txt",
        ],
    }

    for key in MODEL_KEYS:
        if key not in selected:
            continue
        if key == "flux" and not token and not args.include_flux_without_token and not hf_token_available():
            failures.append(
                {
                    "model": key,
                    "repo_id": repo_by_key[key],
                    "error": "HF token missing; skipped to avoid gated-model failure.",
                }
            )
            continue
        try:
            print(f"Downloading {key}: {repo_by_key[key]}", flush=True)
            results.append(
                {
                    "model": key,
                    **download_repo(
                        repo_by_key[key],
                        cache_dir,
                        token,
                        allow_patterns=allow_patterns_by_key.get(key),
                    ),
                }
            )
        except Exception as exc:
            failures.append({"model": key, "repo_id": repo_by_key[key], "error": str(exc)})

    status = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cache_dir": str(cache_dir),
        "downloads": results,
        "failures": failures,
    }
    status_path = config.root / "models" / "model_download_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8") as fh:
        json.dump(status, fh, ensure_ascii=False, indent=2)

    print(json.dumps(status, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _read_token_file(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


if __name__ == "__main__":
    main()
