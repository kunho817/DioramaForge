from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DioramaConfig
from .model_status import project_hf_cache, project_hf_home


SDXL_ENGINE_NAMES = {"sdxl_depth_lightning", "sdxl_lightning", "sdxl"}
FLUX_ENGINE_NAMES = {"flux_depth", "flux", "flux_depth_compat"}
AUTO_ENGINE_NAMES = {"", "auto", "best", "auto_fast"}


def resolve_style_engine(config: DioramaConfig) -> str:
    active = str(config.style_engine.active or "auto").strip().lower()
    if active in SDXL_ENGINE_NAMES:
        return "sdxl_depth_lightning"
    if active in FLUX_ENGINE_NAMES:
        return "flux_depth"
    if active in AUTO_ENGINE_NAMES:
        return "sdxl_depth_lightning" if style_engine_readiness(config)["sdxl_depth_lightning"]["ready"] else "flux_depth"
    return "flux_depth"


def style_engine_readiness(config: DioramaConfig) -> dict[str, Any]:
    sdxl = config.sdxl_depth_lightning
    flux_model = _component_status(
        config.root,
        config.flux.model_id,
        config.flux.local_path,
        ("model_index.json",),
    )
    sdxl_base = _component_status(
        config.root,
        sdxl.base_model_id,
        sdxl.base_local_path,
        ("model_index.json",),
    )
    sdxl_controlnet = _component_status(
        config.root,
        sdxl.controlnet_model_id,
        sdxl.controlnet_local_path,
        ("config.json",),
    )
    sdxl_lora = _component_status(
        config.root,
        sdxl.lora_model_id,
        sdxl.lora_local_path,
        (sdxl.lora_weight_name,),
    )
    sdxl_ready = sdxl_base["ready"] and sdxl_controlnet["ready"] and sdxl_lora["ready"]
    return {
        "flux_depth": {
            "ready": flux_model["ready"],
            "components": {"model": flux_model},
        },
        "sdxl_depth_lightning": {
            "ready": sdxl_ready,
            "components": {
                "base": sdxl_base,
                "controlnet": sdxl_controlnet,
                "lora": sdxl_lora,
            },
        },
    }


def _component_status(
    root: Path,
    model_id: str,
    local_path: str,
    expected_files: tuple[str, ...],
) -> dict[str, Any]:
    local = _local_component_path(local_path, expected_files)
    if local is not None:
        return {"ready": True, "source": "local_path", "path": str(local), "model_id": model_id}
    cache = _cache_component_path(root, model_id, expected_files)
    if cache is not None:
        return {"ready": True, "source": "project_cache", "path": str(cache), "model_id": model_id}
    return {"ready": False, "source": "missing", "path": "", "model_id": model_id}


def _local_component_path(local_path: str, expected_files: tuple[str, ...]) -> Path | None:
    if not local_path:
        return None
    path = Path(local_path)
    if path.is_file():
        return path if _file_matches(path, expected_files) else None
    if not path.is_dir():
        return None
    return path if _directory_contains(path, expected_files) else None


def _cache_component_path(root: Path, model_id: str, expected_files: tuple[str, ...]) -> Path | None:
    model_folder = "models--" + model_id.replace("/", "--")
    for cache_root in (project_hf_cache(root), project_hf_home(root)):
        model_root = cache_root / model_folder
        snapshots_root = model_root / "snapshots"
        if snapshots_root.exists():
            snapshots = [path for path in snapshots_root.iterdir() if path.is_dir()]
            for snapshot in sorted(snapshots, key=lambda path: path.stat().st_mtime, reverse=True):
                if _directory_contains(snapshot, expected_files):
                    return snapshot
        if model_root.exists() and _directory_contains(model_root, expected_files):
            return model_root
    return None


def _directory_contains(path: Path, expected_files: tuple[str, ...]) -> bool:
    for expected in expected_files:
        if expected.startswith("*."):
            if next(path.rglob(expected), None) is not None:
                return True
            continue
        if (path / expected).exists() or next(path.rglob(expected), None) is not None:
            return True
    return False


def _file_matches(path: Path, expected_files: tuple[str, ...]) -> bool:
    for expected in expected_files:
        if expected.startswith("*.") and path.match(expected):
            return True
        if path.name == expected:
            return True
    return False
