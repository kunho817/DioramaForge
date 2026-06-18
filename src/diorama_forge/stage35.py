from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .config import DioramaConfig
from .image_utils import save_json


Stage35Status = Callable[[str], None]


@dataclass(frozen=True)
class Stage35Options:
    mode: str
    backend_mode: str = "auto"
    upscale_scale: float = 2.0
    refinement_strength: float = 0.22
    max_side: int = 1536


@dataclass(frozen=True)
class Stage35Artifacts:
    stage35_dir: Path
    visual_path: Path
    reconstruction_path: Path
    refined_path: Path
    metadata_path: Path
    log: list[str]


def build_stage35_refinement(
    config: DioramaConfig,
    run_dir_value: str | Path,
    options: Stage35Options,
    status: Stage35Status | None = None,
    comfy_client: Any | None = None,
) -> Stage35Artifacts:
    logs: list[str] = []

    def emit(message: str) -> None:
        logs.append(message)
        if status:
            status(message)

    run_dir = _resolve_run_dir(config.root, run_dir_value)
    metadata_path = run_dir / "run_metadata.json"
    metadata = _read_json(metadata_path)
    artifacts = metadata.get("artifacts", {})

    styled_path = _artifact_path(run_dir, artifacts, "final_image", "flux_result.png")
    depth_path = _artifact_path(run_dir, artifacts, "depth_png", "depth.png")
    region_overlay_path = _optional_artifact_path(run_dir, artifacts, "region_overlay", "regions/region_overlay.png")
    styled = Image.open(styled_path).convert("RGB")
    depth = Image.open(depth_path).convert("RGB")
    if depth.size != styled.size:
        depth = depth.resize(styled.size, Image.Resampling.BICUBIC)

    stage35_dir = run_dir / "stage35_refinement"
    stage35_dir.mkdir(parents=True, exist_ok=True)
    emit("Stage 3.5 folder prepared")

    backend_mode = str(options.backend_mode or "auto").strip().lower()
    if backend_mode == "remote":
        raise RuntimeError("Stage 3.5 remote execution is only supported through the API RemoteModelClient path.")
    if backend_mode not in {"demo", "auto", "comfyui"} and not config.app.allow_local_heavy_models:
        raise RuntimeError(
            "Stage 3.5 local heavy/refinement execution is disabled. "
            "Enable local heavy execution explicitly or use the proxy handoff path."
        )
    if backend_mode == "real":
        raise RuntimeError(
            "Stage 3.5 real backend is not connected to a dedicated upscale/refinement adapter yet. "
            "Use the ComfyUI bridge or the proxy handoff path."
        )

    visual: Image.Image | None = None
    reconstruction: Image.Image | None = None
    refined: Image.Image | None = None
    backend_label = "pillow_structure_preserving_proxy"
    extra_metadata: dict[str, Any] = {}
    use_comfy = backend_mode == "comfyui" or (
        backend_mode == "auto" and config.comfy.enabled and config.comfy.stage35_workflow.exists()
    )
    if use_comfy and comfy_client is not None:
        try:
            emit("Running ComfyUI Stage 3.5 workflow")
            comfy_result = comfy_client.generate_stage35(
                run_dir=run_dir,
                source_image=styled,
                depth_image=depth,
                mode=options.mode,
                upscale_scale=options.upscale_scale,
                refinement_strength=options.refinement_strength,
                max_side=options.max_side,
            )
            visual = comfy_result.visual_image.convert("RGB")
            reconstruction = comfy_result.reconstruction_image.convert("RGB")
            refined = comfy_result.refined_image.convert("RGB") if comfy_result.refined_image else None
            backend_label = comfy_result.backend
            extra_metadata["comfyui"] = comfy_result.metadata
            emit("ComfyUI Stage 3.5 workflow complete")
        except Exception as exc:
            if backend_mode == "comfyui":
                raise
            emit(f"ComfyUI Stage 3.5 unavailable; using proxy handoff: {exc}")

    if visual is None or reconstruction is None:
        target_size = _target_size(styled.size, options.upscale_scale, options.max_side)
        visual = _visual_upscale(styled, target_size)
        reconstruction = _reconstruction_upscale(styled, depth, target_size)
        emit(f"Proxy upscale generated: {target_size[0]}x{target_size[1]}")

    visual_path = stage35_dir / "stage35_upscaled_visual.png"
    visual.save(visual_path)

    reconstruction_path = stage35_dir / "stage35_reconstruction_input.png"
    reconstruction.save(reconstruction_path)
    emit("Stage 4 reconstruction input generated")

    if refined is None:
        refined = _refine_for_structure(
            reconstruction,
            depth.resize(reconstruction.size, Image.Resampling.BICUBIC),
            options,
        )
    refined_path = stage35_dir / "stage35_refined.png"
    refined.save(refined_path)
    emit("Structure refinement pass complete")

    stage35_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "stage35_refinement",
        "backend": backend_label,
        "requested_backend": backend_mode,
        "source_run_dir": str(run_dir),
        "mode": options.mode,
        "upscale_scale": options.upscale_scale,
        "refinement_strength": options.refinement_strength,
        "max_side": options.max_side,
        "output_size": {
            "visual": list(visual.size),
            "reconstruction_input": list(reconstruction.size),
            "refined": list(refined.size),
        },
        "inputs": {
            "styled_image": str(styled_path),
            "depth_image": str(depth_path),
            "region_overlay": str(region_overlay_path) if region_overlay_path else None,
        },
        "outputs": {
            "visual": str(visual_path),
            "reconstruction_input": str(reconstruction_path),
            "refined": str(refined_path),
        },
        "note": (
            "Stage 3.5 validates the high-resolution handoff contract before Stage 4. "
            "When requested_backend is comfyui, the reconstruction image comes from a ComfyUI workflow."
        ),
    }
    stage35_metadata.update(extra_metadata)
    stage35_metadata_path = stage35_dir / "stage35_metadata.json"
    save_json(stage35_metadata_path, stage35_metadata)

    metadata.setdefault("artifacts", {})
    metadata["artifacts"]["stage35_visual"] = str(visual_path)
    metadata["artifacts"]["stage35_reconstruction_input"] = str(reconstruction_path)
    metadata["artifacts"]["stage35_refined"] = str(refined_path)
    metadata["artifacts"]["stage35_metadata"] = str(stage35_metadata_path)
    metadata["stage35"] = stage35_metadata
    save_json(metadata_path, metadata)
    emit("Stage 3.5 artifacts linked in run_metadata.json")

    return Stage35Artifacts(
        stage35_dir=stage35_dir,
        visual_path=visual_path,
        reconstruction_path=reconstruction_path,
        refined_path=refined_path,
        metadata_path=stage35_metadata_path,
        log=logs,
    )


def _resolve_run_dir(root: Path, run_dir_value: str | Path) -> Path:
    text = str(run_dir_value or "").strip().strip('"')
    if not text:
        raise RuntimeError("Stage 3 run folder is empty.")
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise RuntimeError(f"Run folder was not found: {path}")
    if not (path / "run_metadata.json").exists():
        raise RuntimeError(f"run_metadata.json was not found: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _artifact_path(run_dir: Path, artifacts: dict[str, Any], key: str, fallback: str) -> Path:
    value = artifacts.get(key)
    path = Path(value) if value else run_dir / fallback
    if not path.is_absolute():
        path = run_dir / path
    if not path.exists():
        raise RuntimeError(f"Required artifact was not found: {key} -> {path}")
    return path


def _optional_artifact_path(run_dir: Path, artifacts: dict[str, Any], key: str, fallback: str) -> Path | None:
    value = artifacts.get(key)
    path = Path(value) if value else run_dir / fallback
    if not path.is_absolute():
        path = run_dir / path
    return path if path.exists() else None


def _target_size(size: tuple[int, int], scale: float, max_side: int) -> tuple[int, int]:
    width, height = size
    scale = max(1.0, float(scale))
    target_width = int(round(width * scale))
    target_height = int(round(height * scale))
    limit = max(256, int(max_side))
    if max(target_width, target_height) > limit:
        shrink = limit / max(target_width, target_height)
        target_width = int(round(target_width * shrink))
        target_height = int(round(target_height * shrink))
    target_width = max(64, (target_width // 8) * 8)
    target_height = max(64, (target_height // 8) * 8)
    return target_width, target_height


def _visual_upscale(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    upscaled = image.resize(target_size, Image.Resampling.LANCZOS)
    upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1.4, percent=120, threshold=3))
    upscaled = ImageEnhance.Color(upscaled).enhance(1.08)
    upscaled = ImageEnhance.Contrast(upscaled).enhance(1.05)
    return upscaled


def _reconstruction_upscale(
    image: Image.Image,
    depth: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    upscaled = image.resize(target_size, Image.Resampling.LANCZOS)
    depth_up = ImageOps.grayscale(depth.resize(target_size, Image.Resampling.BICUBIC))
    edges = depth_up.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(0.45))
    edge_rgb = ImageOps.colorize(edges, black=(0, 0, 0), white=(255, 255, 255)).convert("RGB")
    sharpened = upscaled.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=6))
    return Image.blend(sharpened, edge_rgb, 0.08)


def _refine_for_structure(
    image: Image.Image,
    depth: Image.Image,
    options: Stage35Options,
) -> Image.Image:
    strength = max(0.0, min(0.5, float(options.refinement_strength)))
    depth_gray = ImageOps.grayscale(depth)
    depth_edges = depth_gray.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(0.35))
    edge_rgb = ImageOps.colorize(depth_edges, black=(10, 12, 14), white=(230, 235, 238)).convert("RGB")
    blended = Image.blend(image.convert("RGB"), edge_rgb, strength * 0.22)
    blended = blended.filter(ImageFilter.UnsharpMask(radius=0.8, percent=int(60 + strength * 120), threshold=4))
    return blended
