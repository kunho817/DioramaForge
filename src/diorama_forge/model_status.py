from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from .config import DioramaConfig


@dataclass(frozen=True)
class ModelStatus:
    name: str
    package: str
    package_ready: bool
    model_id: str
    cache_hint: str
    note: str


def package_ready(import_name: str) -> bool:
    if importlib.util.find_spec(import_name) is None:
        return False
    if import_name != "requests":
        return True
    try:
        module = importlib.import_module(import_name)
    except Exception:
        return False
    return hasattr(module, "get") and hasattr(module, "post")


def hf_token_available() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))


def meshy_key_available(config: DioramaConfig) -> bool:
    return bool(os.environ.get(config.meshy.api_key_env))


def project_hf_home(root: Path) -> Path:
    return root / "models" / "huggingface"


def project_hf_cache(root: Path) -> Path:
    return project_hf_home(root) / "hub"


def cache_hint(root: Path, model_id: str, expected_file: str | None = None) -> str:
    model_folder = "models--" + model_id.replace("/", "--")
    for local in (project_hf_cache(root) / model_folder, project_hf_home(root) / model_folder):
        if local.exists():
            if expected_file:
                snapshots_dir = local / "snapshots"
                snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()] if snapshots_dir.exists() else []
                if not snapshots:
                    return f"incomplete cache: no snapshots in {local}"
                latest = max(snapshots, key=lambda path: path.stat().st_mtime)
                if not _contains_expected_file(latest, expected_file):
                    return f"incomplete cache: missing {expected_file} in {latest}"
            return str(local)
    return "not found in project cache"


def _contains_expected_file(path: Path, expected_file: str) -> bool:
    if expected_file.startswith("*."):
        return next(path.rglob(expected_file), None) is not None
    return (path / expected_file).exists() or next(path.rglob(expected_file), None) is not None


def collect_model_status(config: DioramaConfig) -> list[ModelStatus]:
    token_note = "HF token detected" if hf_token_available() else "HF token not detected"
    comfy_workflow_note = (
        "workflow ready" if config.comfy.stage3_workflow.exists() else "stage3 workflow missing"
    )
    meshy_note = (
        f"{config.meshy.api_key_env} detected"
        if meshy_key_available(config)
        else f"{config.meshy.api_key_env} not detected"
    )
    remote_note = "enabled" if config.remote.enabled else "disabled"
    items = [
        ModelStatus(
            name="Depth Anything 3",
            package="depth_anything_3",
            package_ready=package_ready("depth_anything_3"),
            model_id=config.depth.model_id,
            cache_hint=cache_hint(config.root, config.depth.model_id),
            note="Uses Transformers depth fallback when DA3 package is unavailable.",
        ),
        ModelStatus(
            name="SAM 2",
            package="sam2",
            package_ready=package_ready("sam2"),
            model_id=config.sam.model_id,
            cache_hint=cache_hint(config.root, config.sam.model_id),
            note="Uses Transformers mask-generation fallback when sam2 package is unavailable.",
        ),
        ModelStatus(
            name="FLUX.1 Depth Compatibility",
            package="diffusers",
            package_ready=package_ready("diffusers"),
            model_id=config.flux.model_id,
            cache_hint=(
                config.flux.local_path
                if config.flux.local_path
                else cache_hint(config.root, config.flux.model_id, expected_file="model_index.json")
            ),
            note=f"{token_note}. Kept as a research baseline and compatibility fallback; not the product GUI selector.",
        ),
        ModelStatus(
            name="Configured Stage 3 Target",
            package="requests" if config.style_engine.backend_mode.lower() == "comfyui" else "diffusers",
            package_ready=package_ready("requests" if config.style_engine.backend_mode.lower() == "comfyui" else "diffusers"),
            model_id=config.style_engine.target,
            cache_hint=str(config.comfy.stage3_workflow) if config.style_engine.backend_mode.lower() == "comfyui" else "internal config",
            note="Product Generate uses this hidden Stage 3 target; the GUI does not expose engine choices.",
        ),
        ModelStatus(
            name="SDXL Depth Lightning Base",
            package="diffusers",
            package_ready=package_ready("diffusers"),
            model_id=config.sdxl_depth_lightning.base_model_id,
            cache_hint=(
                config.sdxl_depth_lightning.base_local_path
                if config.sdxl_depth_lightning.base_local_path
                else cache_hint(config.root, config.sdxl_depth_lightning.base_model_id, expected_file="model_index.json")
            ),
            note="Candidate base for a faster depth-conditioned ComfyUI or Diffusers workflow.",
        ),
        ModelStatus(
            name="SDXL Depth ControlNet",
            package="diffusers",
            package_ready=package_ready("diffusers"),
            model_id=config.sdxl_depth_lightning.controlnet_model_id,
            cache_hint=(
                config.sdxl_depth_lightning.controlnet_local_path
                if config.sdxl_depth_lightning.controlnet_local_path
                else cache_hint(config.root, config.sdxl_depth_lightning.controlnet_model_id, expected_file="config.json")
            ),
            note="Depth conditioning model for preserving the source layout.",
        ),
        ModelStatus(
            name="SDXL Lightning LoRA",
            package="diffusers",
            package_ready=package_ready("diffusers"),
            model_id=config.sdxl_depth_lightning.lora_model_id,
            cache_hint=(
                config.sdxl_depth_lightning.lora_local_path
                if config.sdxl_depth_lightning.lora_local_path
                else cache_hint(
                    config.root,
                    config.sdxl_depth_lightning.lora_model_id,
                    expected_file=config.sdxl_depth_lightning.lora_weight_name,
                )
            ),
            note=f"Fast-step candidate: {config.sdxl_depth_lightning.lora_weight_name}.",
        ),
        ModelStatus(
            name="TRELLIS",
            package="trellis",
            package_ready=package_ready("trellis"),
            model_id=config.trellis.model_id,
            cache_hint=(
                config.trellis.local_path
                if config.trellis.local_path
                else cache_hint(config.root, config.trellis.model_id)
            ),
            note="Stage 4 currently creates a reconstruction package and proxy OBJ unless a real adapter is added.",
        ),
        ModelStatus(
            name="ComfyUI Stage 3",
            package="requests",
            package_ready=package_ready("requests"),
            model_id=config.comfy.base_url,
            cache_hint=str(config.comfy.stage3_workflow),
            note=f"Configured internal product backend for Stage 3 image generation; {comfy_workflow_note}.",
        ),
        ModelStatus(
            name="Meshy AI Image to 3D",
            package="requests",
            package_ready=package_ready("requests"),
            model_id=config.meshy.base_url,
            cache_hint=", ".join(config.meshy.target_formats),
            note=f"Configured Stage 4 shortcut backend; {meshy_note}.",
        ),
        ModelStatus(
            name="UltraShape",
            package="ultrashape",
            package_ready=package_ready("ultrashape"),
            model_id=config.ultrashape.model_id,
            cache_hint=(
                config.ultrashape.local_path
                if config.ultrashape.local_path
                else cache_hint(config.root, config.ultrashape.model_id)
            ),
            note="Stage 5 currently writes a print proxy STL and records refinement targets.",
        ),
    ]
    if config.style_engine.legacy_remote_visible:
        items.append(
            ModelStatus(
                name="Remote A100 Backend",
                package="requests",
                package_ready=package_ready("requests"),
                model_id=config.remote.base_url,
                cache_hint=config.remote.execution_backend,
                note=f"Legacy cloud backend is {remote_note}; hidden from the default GUI.",
            )
        )
    return items


def model_status_markdown(config: DioramaConfig) -> str:
    rows = [
        "| Model | Package | Model ID | Project Cache | Note |",
        "|---|---:|---|---|---|",
    ]
    for item in collect_model_status(config):
        package = "ready" if item.package_ready else f"missing `{item.package}`"
        rows.append(
            f"| {item.name} | {package} | `{item.model_id}` | {item.cache_hint} | {item.note} |"
        )
    return "\n".join(rows)
