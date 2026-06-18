from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import DioramaConfig
from .image_utils import mask_bbox, save_json
from .meshy import MeshyClient
from .prompting import build_meshy_texture_prompt, compact_prompt


StageStatus = Callable[[str], None]


@dataclass(frozen=True)
class Stage4Options:
    backend_mode: str
    mesh_resolution: int = 96
    max_parts: int = 12


@dataclass(frozen=True)
class Stage4Artifacts:
    stage4_dir: Path
    manifest_path: Path
    contact_sheet_path: Path
    obj_path: Path
    log: list[str]


@dataclass(frozen=True)
class Stage5Options:
    backend_mode: str
    width_mm: float
    relief_height_mm: float
    base_thickness_mm: float
    mesh_resolution: int = 96


@dataclass(frozen=True)
class Stage5Artifacts:
    stage5_dir: Path
    manifest_path: Path
    stl_path: Path
    preview_path: Path
    checklist_path: Path
    log: list[str]


def build_stage4_package(
    config: DioramaConfig,
    run_dir_value: str | Path,
    options: Stage4Options,
    status: StageStatus | None = None,
) -> Stage4Artifacts:
    logs: list[str] = []

    def emit(message: str) -> None:
        logs.append(message)
        if status:
            status(message)

    run_dir = _resolve_run_dir(config.root, run_dir_value)
    metadata = _load_metadata(run_dir)
    requested_backend = str(options.backend_mode or "auto").strip().lower()
    if requested_backend == "remote":
        raise RuntimeError("Stage 4 remote execution is only supported through the API RemoteModelClient path.")
    backend = _selected_backend(options.backend_mode, config.trellis.backend)
    if backend == "real" and not config.app.allow_local_heavy_models:
        raise RuntimeError(
            "Stage 4 local heavy 3D model execution is disabled. "
            "Enable local heavy execution explicitly or use the proxy reconstruction package path."
        )
    if backend == "real":
        raise RuntimeError(
            "Stage 4 real backend is not connected to the TRELLIS adapter yet. "
            "Use the proxy reconstruction package path."
        )
    if backend == "meshy":
        _ensure_meshy_backend_ready(config, "Stage 4")

    stage4_dir = run_dir / "stage4_reconstruction"
    parts_dir = stage4_dir / "parts"
    stage4_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    emit("Stage 4 폴더 준비 완료")

    source_path = _artifact_path(run_dir, metadata, "input", "input.png")
    styled_path, styled_source = _preferred_styled_input_path(run_dir, metadata)
    depth_path = _artifact_path(run_dir, metadata, "depth_png", "depth.png")
    region_manifest_path = _optional_artifact_path(run_dir, metadata, "region_manifest", "regions/region_plan.json")

    source_image = Image.open(source_path).convert("RGB")
    styled_image = Image.open(styled_path).convert("RGB")
    depth_image = Image.open(depth_path).convert("RGB")
    if source_image.size != styled_image.size:
        source_image = source_image.resize(styled_image.size, Image.Resampling.LANCZOS)
    if depth_image.size != styled_image.size:
        depth_image = depth_image.resize(styled_image.size, Image.Resampling.BICUBIC)
    depth_image.save(stage4_dir / "depth_input.png")

    region_plan = _read_json(region_manifest_path) if region_manifest_path and region_manifest_path.exists() else {}
    region_masks = _region_mask_paths(run_dir, metadata)
    stage4_inputs = _prepare_stage4_input_images(
        stage4_dir=stage4_dir,
        styled_image=styled_image,
        region_masks=region_masks,
    )
    trellis_image = Image.open(stage4_inputs["trellis_input"]).convert("RGB")
    parts = _export_region_parts(
        parts_dir=parts_dir,
        source_image=source_image,
        styled_image=styled_image,
        depth_image=depth_image,
        region_masks=region_masks,
        max_parts=options.max_parts,
    )
    contact_sheet_path = stage4_dir / "part_contact_sheet.png"
    _save_part_contact_sheet(contact_sheet_path, parts)
    emit(f"SAM/region 기반 이미지 분할 완료: {len(parts)}개 part")

    proxy_mesh = _write_heightfield_obj(
        stage4_dir=stage4_dir,
        styled_image=trellis_image,
        depth_image=depth_image,
        mesh_resolution=options.mesh_resolution,
    )
    emit("검증용 depth-relief OBJ 생성 완료")

    meshy_result: dict[str, Any] | None = None
    if backend == "meshy":
        meshy_dir = stage4_dir / "meshy"
        meshy = MeshyClient(config.meshy).run_image_to_3d(
            image_path=Path(stage4_inputs["meshy_input"]),
            output_dir=meshy_dir,
            texture_prompt=_stage4_texture_prompt(metadata, region_plan),
            status=emit,
        )
        meshy_result = {
            "task_id": meshy.task_id,
            "status": meshy.task.get("status", ""),
            "progress": meshy.task.get("progress"),
            "consumed_credits": meshy.task.get("consumed_credits"),
            "request": str(meshy.request_path),
            "task": str(meshy.task_path),
            "downloads_manifest": str(meshy.downloads_path),
            "downloads": meshy.downloads,
            "model_urls": meshy.task.get("model_urls", {}),
            "thumbnail_url": meshy.task.get("thumbnail_url", ""),
        }
        if not _model_download_keys(meshy.downloads):
            raise RuntimeError(
                "Meshy task completed but no GLB, OBJ, or STL model files were downloaded. "
                "Check meshy_ai.target_formats and meshy_ai.download_outputs in configs/default.json."
            )
        emit(f"Meshy image-to-3D complete: {meshy.task_id}")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "stage4_reconstruction_package",
        "backend": "meshy_image_to_3d" if meshy_result else "demo_depth_relief_proxy",
        "source_run_dir": str(run_dir),
        "note": (
            "This package prepares Stage 4 inputs from Stage 1-3 outputs. "
            "The live-demo path can use Meshy Image to 3D for the actual model output; "
            "proxy mesh outputs remain for validation and fallback inspection."
        ),
        "model_targets": {
            "trellis": {
                "model_id": config.trellis.model_id,
                "local_path": config.trellis.local_path,
            },
            "ultrashape": {
                "model_id": config.ultrashape.model_id,
                "local_path": config.ultrashape.local_path,
            },
        },
        "inputs": {
            "source_image": str(source_path),
            "styled_image": str(styled_path),
            "styled_image_source": styled_source,
            "depth_image": str(depth_path),
            "region_manifest": str(region_manifest_path) if region_manifest_path else None,
            "full_styled_input": stage4_inputs["full_styled_input"],
            "trellis_input": stage4_inputs["trellis_input"],
            "meshy_input": stage4_inputs["meshy_input"],
            "sky_exclusion_mask": stage4_inputs["sky_exclusion_mask"],
            "stage4_input_strategy": stage4_inputs["strategy"],
            "depth_input": str(stage4_dir / "depth_input.png"),
        },
        "semantic_region_plan": region_plan,
        "parts": parts,
        "proxy_mesh": proxy_mesh,
        "meshy": meshy_result,
        "next_stage": {
            "stage5_expected_input": str(stage4_dir / "reconstruction_package.json"),
            "print_proxy_supported": True,
            "meshy_model_supported": meshy_result is not None,
        },
    }
    manifest_path = stage4_dir / "reconstruction_package.json"
    save_json(manifest_path, manifest)
    metadata.setdefault("artifacts", {})
    metadata["artifacts"]["stage4_manifest"] = str(manifest_path)
    metadata["artifacts"]["stage4_contact_sheet"] = str(contact_sheet_path)
    metadata["artifacts"]["stage4_proxy_obj"] = str(proxy_mesh["obj"])
    if meshy_result:
        metadata["artifacts"]["stage4_meshy_task"] = str(meshy_result["task"])
        metadata["artifacts"]["stage4_meshy_downloads"] = str(meshy_result["downloads_manifest"])
        for key, path_value in meshy_result.get("downloads", {}).items():
            metadata["artifacts"][f"stage4_meshy_{key}"] = str(path_value)
    metadata["stage4"] = {
        "created_at": manifest["created_at"],
        "backend": manifest["backend"],
        "requested_backend": options.backend_mode,
        "part_count": len(parts),
        "mesh_resolution": options.mesh_resolution,
        "max_parts": options.max_parts,
        "manifest": str(manifest_path),
        "contact_sheet": str(contact_sheet_path),
        "proxy_obj": str(proxy_mesh["obj"]),
        "meshy": meshy_result,
        "styled_image_source": styled_source,
        "stage4_input_strategy": stage4_inputs["strategy"],
        "meshy_input": stage4_inputs["meshy_input"],
    }
    save_json(run_dir / "run_metadata.json", metadata)
    emit("Stage 4 manifest 저장 완료")

    return Stage4Artifacts(
        stage4_dir=stage4_dir,
        manifest_path=manifest_path,
        contact_sheet_path=contact_sheet_path,
        obj_path=_stage4_primary_obj_path(meshy_result, proxy_mesh),
        log=logs,
    )


def build_stage5_print_package(
    config: DioramaConfig,
    run_dir_value: str | Path,
    options: Stage5Options,
    status: StageStatus | None = None,
) -> Stage5Artifacts:
    logs: list[str] = []

    def emit(message: str) -> None:
        logs.append(message)
        if status:
            status(message)

    run_dir = _resolve_run_dir(config.root, run_dir_value)
    metadata = _load_metadata(run_dir)
    requested_backend = str(options.backend_mode or "auto").strip().lower()
    if requested_backend == "remote":
        raise RuntimeError("Stage 5 remote execution is only supported through the API RemoteModelClient path.")
    backend = _selected_backend(options.backend_mode, config.ultrashape.backend)
    if backend == "real" and not config.app.allow_local_heavy_models:
        raise RuntimeError(
            "Stage 5 local heavy 3D/mesh refinement execution is disabled. "
            "Enable local heavy execution explicitly or use the proxy print package path."
        )
    if backend == "real":
        raise RuntimeError(
            "Stage 5 real backend is not connected to UltraShape/Open3D/Trimesh post-processing yet. "
            "Use the proxy print package path."
        )
    stage4_manifest = run_dir / "stage4_reconstruction" / "reconstruction_package.json"
    stage4_data = _read_json(stage4_manifest) if stage4_manifest.exists() else {}
    meshy_downloads = (
        _require_meshy_downloads_for_stage5(stage4_manifest, stage4_data)
        if backend == "meshy"
        else _meshy_downloads_from_stage4(stage4_data)
    )

    stage5_dir = run_dir / "stage5_print"
    stage5_dir.mkdir(parents=True, exist_ok=True)
    emit("Stage 5 폴더 준비 완료")

    styled_path, styled_source = _preferred_styled_input_path(run_dir, metadata)
    depth_path = _artifact_path(run_dir, metadata, "depth_png", "depth.png")
    depth_image = Image.open(depth_path).convert("RGB")
    styled_image = Image.open(styled_path).convert("RGB")
    if depth_image.size != styled_image.size:
        depth_image = depth_image.resize(styled_image.size, Image.Resampling.BICUBIC)

    stl_path = stage5_dir / "print_ready_relief_proxy.stl"
    mesh_stats = _write_relief_stl(
        path=stl_path,
        depth_image=depth_image,
        width_mm=options.width_mm,
        relief_height_mm=options.relief_height_mm,
        base_thickness_mm=options.base_thickness_mm,
        mesh_resolution=options.mesh_resolution,
    )
    emit("base plate 포함 STL proxy 생성 완료")

    preview_path = stage5_dir / "print_preview.png"
    _save_print_preview(preview_path, styled_image, depth_image)
    model_files: dict[str, str] = {}
    if backend == "meshy" and meshy_downloads:
        model_files = _copy_meshy_model_files(stage5_dir, meshy_downloads)
        thumbnail = Path(model_files.get("thumbnail", ""))
        if thumbnail.exists():
            shutil.copy2(thumbnail, preview_path)
        mesh_stats["meshy_model_files"] = model_files
        mesh_stats["meshy_packaged"] = True
        emit("Packaged Meshy model outputs for Stage 5")
    checklist_path = stage5_dir / "print_checklist.md"
    _write_print_checklist(checklist_path, options, mesh_stats)

    stage4_manifest = run_dir / "stage4_reconstruction" / "reconstruction_package.json"
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "stage5_print_package",
        "backend": "meshy_model_package" if model_files else "demo_depth_relief_print_proxy",
        "source_run_dir": str(run_dir),
        "note": (
            "This package prefers Meshy Image-to-3D outputs when Stage 4 produced them. "
            "The depth-relief proxy remains as a validation and slicer-inspection fallback."
        ),
        "inputs": {
            "styled_image": str(styled_path),
            "styled_image_source": styled_source,
            "depth_image": str(depth_path),
            "stage4_manifest": str(stage4_manifest) if stage4_manifest.exists() else None,
        },
        "print_settings": {
            "width_mm": options.width_mm,
            "relief_height_mm": options.relief_height_mm,
            "base_thickness_mm": options.base_thickness_mm,
            "mesh_resolution": options.mesh_resolution,
        },
        "outputs": {
            "stl": str(stl_path),
            "preview": str(preview_path),
            "checklist": str(checklist_path),
            "model_files": model_files,
        },
        "mesh_stats": mesh_stats,
        "warnings": [
            "Meshy outputs still need manual printability inspection before printing.",
            "Proxy STL is useful for pipeline validation and slicer inspection when Meshy output is unavailable.",
            "Final Stage 5 should run mesh repair, watertight validation, wall-thickness checks, and scale review.",
        ],
    }
    manifest_path = stage5_dir / "print_package.json"
    save_json(manifest_path, manifest)
    metadata.setdefault("artifacts", {})
    metadata["artifacts"]["stage5_manifest"] = str(manifest_path)
    metadata["artifacts"]["stage5_preview"] = str(preview_path)
    metadata["artifacts"]["stage5_proxy_stl"] = str(stl_path)
    metadata["artifacts"]["stage5_checklist"] = str(checklist_path)
    for key, path_value in model_files.items():
        metadata["artifacts"][f"stage5_model_{key}"] = str(path_value)
    metadata["stage5"] = {
        "created_at": manifest["created_at"],
        "backend": manifest["backend"],
        "requested_backend": options.backend_mode,
        "print_settings": manifest["print_settings"],
        "mesh_stats": mesh_stats,
        "manifest": str(manifest_path),
        "preview": str(preview_path),
        "proxy_stl": str(stl_path),
        "model_files": model_files,
        "checklist": str(checklist_path),
        "styled_image_source": styled_source,
    }
    save_json(run_dir / "run_metadata.json", metadata)
    emit("Stage 5 manifest 저장 완료")

    return Stage5Artifacts(
        stage5_dir=stage5_dir,
        manifest_path=manifest_path,
        stl_path=stl_path,
        preview_path=preview_path,
        checklist_path=checklist_path,
        log=logs,
    )


def _stage4_primary_obj_path(meshy_result: dict[str, Any] | None, proxy_mesh: dict[str, Any]) -> Path:
    if meshy_result:
        obj_path = meshy_result.get("downloads", {}).get("obj")
        if obj_path and Path(obj_path).exists():
            return Path(obj_path)
    return Path(proxy_mesh["obj"])


def _prepare_stage4_input_images(
    stage4_dir: Path,
    styled_image: Image.Image,
    region_masks: list[tuple[str, Path]],
) -> dict[str, str | None]:
    full_path = stage4_dir / "stage4_full_styled_input.png"
    trellis_path = stage4_dir / "trellis_input.png"
    meshy_path = stage4_dir / "meshy_input.png"
    sky_mask_path = stage4_dir / "sky_exclusion_mask.png"

    styled_rgb = styled_image.convert("RGB")
    styled_rgb.save(full_path)
    sky_mask = _combined_region_mask(region_masks, {"sky"}, styled_rgb.size)
    if sky_mask is None:
        sky_mask = _estimate_sky_backdrop_mask(styled_rgb)

    total_area = styled_rgb.size[0] * styled_rgb.size[1]
    sky_area = int(sky_mask.sum()) if sky_mask is not None else 0
    if sky_mask is None or sky_area < total_area * 0.04 or sky_area > total_area * 0.95:
        styled_rgb.save(trellis_path)
        styled_rgb.save(meshy_path)
        return {
            "full_styled_input": str(full_path),
            "trellis_input": str(trellis_path),
            "meshy_input": str(meshy_path),
            "sky_exclusion_mask": None,
            "strategy": "full_styled_image",
        }

    subject_mask = ~sky_mask
    if int(subject_mask.sum()) < total_area * 0.03:
        styled_rgb.save(trellis_path)
        styled_rgb.save(meshy_path)
        return {
            "full_styled_input": str(full_path),
            "trellis_input": str(trellis_path),
            "meshy_input": str(meshy_path),
            "sky_exclusion_mask": None,
            "strategy": "full_styled_image_subject_too_small",
        }

    alpha = Image.fromarray((subject_mask * 255).astype(np.uint8), mode="L")
    Image.fromarray((sky_mask * 255).astype(np.uint8), mode="L").save(sky_mask_path)

    neutral = Image.new("RGB", styled_rgb.size, (245, 245, 242))
    neutral.paste(styled_rgb, mask=alpha)
    neutral.save(trellis_path)

    subject = styled_rgb.convert("RGBA")
    subject.putalpha(alpha)
    x, y, width, height = _expand_bbox(mask_bbox(subject_mask), styled_rgb.size[0], styled_rgb.size[1], padding=16)
    subject.crop((x, y, x + width, y + height)).save(meshy_path)
    return {
        "full_styled_input": str(full_path),
        "trellis_input": str(trellis_path),
        "meshy_input": str(meshy_path),
        "sky_exclusion_mask": str(sky_mask_path),
        "strategy": "sky_backdrop_removed_for_3d",
    }


def _stage4_texture_prompt(metadata: dict[str, Any], region_plan: dict[str, Any]) -> str:
    options = metadata.get("options", {}) if isinstance(metadata, dict) else {}
    stored_prompt = str(options.get("meshy_texture_prompt", "")).strip()
    if stored_prompt:
        return compact_prompt(stored_prompt, 600)

    preset_name = str(options.get("preset_name", "") or "Fantasy Diorama")
    generated = build_meshy_texture_prompt(preset_name=preset_name, region_plan=region_plan)
    if generated:
        return generated
    return "diorama style miniature terrain, clean readable details, coherent materials"


def _meshy_downloads_from_stage4(stage4_manifest: dict[str, Any]) -> dict[str, str]:
    meshy = stage4_manifest.get("meshy") if isinstance(stage4_manifest, dict) else None
    downloads = meshy.get("downloads", {}) if isinstance(meshy, dict) else {}
    return {str(key): str(value) for key, value in downloads.items() if value}


def _require_meshy_downloads_for_stage5(stage4_manifest_path: Path, stage4_manifest: dict[str, Any]) -> dict[str, str]:
    if not stage4_manifest_path.exists():
        raise RuntimeError(
            "Stage 5 was requested with Meshy packaging, but Stage 4 Meshy manifest is missing: "
            f"{stage4_manifest_path}"
        )
    meshy = stage4_manifest.get("meshy") if isinstance(stage4_manifest, dict) else None
    if not isinstance(meshy, dict) or not meshy:
        raise RuntimeError("Stage 5 was requested with Meshy packaging, but Stage 4 contains no Meshy task metadata.")
    status = str(meshy.get("status", "")).strip().upper()
    if status != "SUCCEEDED":
        raise RuntimeError(f"Stage 5 requires a successful Meshy Stage 4 task. Current status: {status or '-'}")
    downloads = _meshy_downloads_from_stage4(stage4_manifest)
    model_keys = _model_download_keys(downloads)
    if not model_keys:
        raise RuntimeError("Stage 5 was requested with Meshy packaging, but Stage 4 has no GLB, OBJ, or STL downloads.")
    resolved: dict[str, str] = {}
    for key, value in downloads.items():
        path = _resolve_stage4_manifest_path(stage4_manifest_path, value)
        if str(key).lower() in {"glb", "obj", "stl"} and (not path.exists() or not path.is_file() or path.stat().st_size <= 0):
            raise RuntimeError(f"Stage 5 Meshy packaging requires a non-empty model file for {key}: {path}")
        resolved[str(key)] = str(path)
    return resolved


def _model_download_keys(downloads: dict[str, str]) -> list[str]:
    return sorted(str(key) for key in downloads if str(key).lower() in {"glb", "obj", "stl"} and downloads.get(key))


def _resolve_stage4_manifest_path(stage4_manifest_path: Path, value: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = stage4_manifest_path.parent / path
    return path


def _copy_meshy_model_files(stage5_dir: Path, downloads: dict[str, str]) -> dict[str, str]:
    model_dir = stage5_dir / "meshy_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key, value in downloads.items():
        source = Path(value)
        if not source.exists() or not source.is_file():
            continue
        suffix = source.suffix or f".{key}"
        target = model_dir / f"{_safe_name(key)}{suffix}"
        shutil.copy2(source, target)
        copied[str(key)] = str(target)
    return copied


def _resolve_run_dir(root: Path, run_dir_value: str | Path) -> Path:
    text = str(run_dir_value or "").strip().strip('"')
    if not text:
        raise RuntimeError("Stage 3 실행 폴더가 비어 있습니다.")
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise RuntimeError(f"실행 폴더를 찾을 수 없습니다: {path}")
    if not (path / "run_metadata.json").exists():
        raise RuntimeError(f"run_metadata.json을 찾을 수 없습니다: {path}")
    return path


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "run_metadata.json")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _selected_backend(backend_mode: str, configured_backend: str) -> str:
    user_backend = str(backend_mode or "auto").lower()
    if user_backend in {"real", "demo", "meshy"}:
        return user_backend
    configured = str(configured_backend or "auto").lower()
    if configured in {"real", "meshy"}:
        return configured
    return "demo"


def _ensure_meshy_backend_ready(config: DioramaConfig, stage_label: str) -> None:
    status = MeshyClient(config.meshy).status()
    if status.get("ok"):
        return
    key_env = status.get("api_key_env") or "MESHY_API_KEY"
    if not status.get("enabled"):
        raise RuntimeError(f"{stage_label}: Meshy AI backend is disabled in configs/default.json.")
    if not status.get("api_key_present"):
        raise RuntimeError(f"{stage_label}: Meshy AI API key is missing. Set {key_env} before running Meshy output.")
    if not status.get("requests_ready"):
        raise RuntimeError(f"{stage_label}: Python requests package is not ready for Meshy AI: {status.get('requests_detail')}")
    if not status.get("download_outputs_ready"):
        raise RuntimeError(f"{stage_label}: meshy_ai.download_outputs must be true for Meshy model packaging.")
    if not status.get("model_output_formats_ready"):
        raise RuntimeError(f"{stage_label}: meshy_ai.target_formats must include glb, obj, or stl for Meshy model packaging.")
    raise RuntimeError(f"{stage_label}: Meshy AI backend is not ready.")


def _artifact_path(run_dir: Path, metadata: dict[str, Any], key: str, fallback: str) -> Path:
    value = metadata.get("artifacts", {}).get(key)
    path = Path(value) if value else run_dir / fallback
    if not path.is_absolute():
        path = run_dir / path
    if not path.exists():
        raise RuntimeError(f"필수 산출물을 찾을 수 없습니다: {key} -> {path}")
    return path


def _optional_artifact_path(run_dir: Path, metadata: dict[str, Any], key: str, fallback: str) -> Path | None:
    value = metadata.get("artifacts", {}).get(key)
    path = Path(value) if value else run_dir / fallback
    if not path.is_absolute():
        path = run_dir / path
    return path if path.exists() else None


def _preferred_styled_input_path(run_dir: Path, metadata: dict[str, Any]) -> tuple[Path, str]:
    candidates = [
        ("stage35_reconstruction_input", "stage35_refinement/stage35_reconstruction_input.png"),
        ("stage35_refined", "stage35_refinement/stage35_refined.png"),
    ]
    for key, fallback in candidates:
        path = _optional_artifact_path(run_dir, metadata, key, fallback)
        if path:
            return path, key
    return _artifact_path(run_dir, metadata, "final_image", "flux_result.png"), "final_image"


def _region_mask_paths(run_dir: Path, metadata: dict[str, Any]) -> list[tuple[str, Path]]:
    masks = metadata.get("artifacts", {}).get("region_masks") or {}
    items: list[tuple[str, Path]] = []
    for label, path_value in masks.items():
        path = Path(path_value)
        if not path.is_absolute():
            path = run_dir / path
        if path.exists():
            items.append((str(label), path))
    if items:
        return items

    region_dir = run_dir / "regions"
    if region_dir.exists():
        for path in sorted(region_dir.glob("*_mask.png")):
            items.append((path.stem.replace("_mask", ""), path))
    if items:
        return items

    masks_dir = run_dir / "masks"
    return [(path.stem, path) for path in sorted(masks_dir.glob("mask_*.png"))]


def _combined_region_mask(
    region_masks: list[tuple[str, Path]],
    labels: set[str],
    image_size: tuple[int, int],
) -> np.ndarray | None:
    combined: np.ndarray | None = None
    normalized_labels = {label.lower() for label in labels}
    for label, mask_path in region_masks:
        if str(label).strip().lower() not in normalized_labels:
            continue
        mask = _open_mask(mask_path, image_size)
        combined = mask if combined is None else (combined | mask)
    return combined


def _estimate_sky_backdrop_mask(image: Image.Image) -> np.ndarray | None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    if height <= 0 or width <= 0:
        return None
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    y_ratio = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    blue_sky = (blue > green * 1.03) & (blue > red * 1.12) & (green > red * 0.92)
    bright_cloud = (red > 145) & (green > 150) & (blue > 160) & (blue >= red * 0.9)
    top_backdrop = y_ratio < 0.62
    mask = (blue_sky | bright_cloud) & top_backdrop
    return mask if int(mask.sum()) > width * height * 0.04 else None


def _export_region_parts(
    parts_dir: Path,
    source_image: Image.Image,
    styled_image: Image.Image,
    depth_image: Image.Image,
    region_masks: list[tuple[str, Path]],
    max_parts: int,
) -> list[dict[str, Any]]:
    width, height = styled_image.size
    total_area = float(width * height)
    parts: list[dict[str, Any]] = []

    for index, (label, mask_path) in enumerate(region_masks[:max_parts], start=1):
        mask = _open_mask(mask_path, (width, height))
        area = int(mask.sum())
        if area < total_area * 0.005:
            continue
        bbox = _expand_bbox(mask_bbox(mask), width, height, padding=8)
        x, y, w, h = bbox
        part_dir = parts_dir / f"{index:02d}_{_safe_name(label)}"
        part_dir.mkdir(parents=True, exist_ok=True)

        mask_crop = Image.fromarray((mask[y : y + h, x : x + w] * 255).astype(np.uint8), mode="L")
        source_crop = source_image.crop((x, y, x + w, y + h))
        styled_crop = styled_image.crop((x, y, x + w, y + h))
        depth_crop = depth_image.crop((x, y, x + w, y + h))
        cutout = styled_crop.convert("RGBA")
        cutout.putalpha(mask_crop)

        source_file = part_dir / "source_crop.png"
        styled_file = part_dir / "styled_crop.png"
        depth_file = part_dir / "depth_crop.png"
        mask_file = part_dir / "mask.png"
        cutout_file = part_dir / "styled_cutout.png"
        source_crop.save(source_file)
        styled_crop.save(styled_file)
        depth_crop.save(depth_file)
        mask_crop.save(mask_file)
        cutout.save(cutout_file)

        parts.append(
            {
                "index": index,
                "semantic_label": label,
                "bbox": bbox,
                "area": area,
                "area_ratio": round(area / total_area, 4),
                "role": _part_role(label),
                "mesh_usage": _part_mesh_usage(label),
                "files": {
                    "source_crop": str(source_file),
                    "styled_crop": str(styled_file),
                    "depth_crop": str(depth_file),
                    "mask": str(mask_file),
                    "styled_cutout": str(cutout_file),
                },
            }
        )
    return parts


def _open_mask(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    mask_image = ImageOps.grayscale(Image.open(path))
    if mask_image.size != image_size:
        mask_image = mask_image.resize(image_size, Image.Resampling.NEAREST)
    return np.asarray(mask_image, dtype=np.uint8) > 0


def _expand_bbox(bbox: list[int], image_width: int, image_height: int, padding: int) -> list[int]:
    x, y, w, h = bbox
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(image_width, x + w + padding)
    y1 = min(image_height, y + h + padding)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _safe_name(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(chunk for chunk in text.split("_") if chunk) or "part"


def _part_role(label: str) -> str:
    roles = {
        "sky": "backdrop/reference layer, usually excluded from solid mesh",
        "ground": "base terrain and relief foundation",
        "foliage": "detail vegetation layer",
        "structure": "primary reconstruction object",
        "water": "flat/resin surface insert candidate",
    }
    return roles.get(label, "candidate reconstruction part")


def _part_mesh_usage(label: str) -> str:
    if str(label).strip().lower() == "sky":
        return "reference_backdrop_not_solid_mesh"
    return "solid_reconstruction_candidate"


def _save_part_contact_sheet(path: Path, parts: list[dict[str, Any]]) -> None:
    if not parts:
        Image.new("RGB", (640, 360), "white").save(path)
        return
    tile_w = 240
    tile_h = 190
    label_h = 42
    columns = min(3, max(1, math.ceil(math.sqrt(len(parts)))))
    rows = math.ceil(len(parts) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, part in enumerate(parts):
        x = (idx % columns) * tile_w
        y = (idx // columns) * (tile_h + label_h)
        image = Image.open(part["files"]["styled_cutout"]).convert("RGBA")
        image.thumbnail((tile_w - 20, tile_h - 20), Image.Resampling.LANCZOS)
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        px = x + (tile_w - background.width) // 2
        py = y + (tile_h - background.height) // 2
        sheet.paste(background, (px, py))
        draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + label_h), fill=(235, 235, 235))
        label = f"{part['index']:02d} {part['semantic_label']} {part['area_ratio']:.2f}"
        draw.text((x + 10, y + tile_h + 12), label, fill=(20, 20, 20), font=font)
    sheet.save(path)


def _write_heightfield_obj(
    stage4_dir: Path,
    styled_image: Image.Image,
    depth_image: Image.Image,
    mesh_resolution: int,
) -> dict[str, Any]:
    obj_path = stage4_dir / "heightfield_proxy.obj"
    mtl_path = stage4_dir / "heightfield_proxy.mtl"
    texture_path = stage4_dir / "heightfield_texture.png"
    preview_path = stage4_dir / "heightfield_preview.png"

    depth = _resized_depth(depth_image, mesh_resolution)
    rows, cols = depth.shape
    texture = styled_image.resize((cols, rows), Image.Resampling.LANCZOS)
    texture.save(texture_path)
    _save_depth_shading(preview_path, depth)

    aspect = cols / max(1, rows)
    x_values = np.linspace(-aspect / 2.0, aspect / 2.0, cols)
    y_values = np.linspace(0.5, -0.5, rows)
    z_values = depth * 0.18

    with mtl_path.open("w", encoding="utf-8") as fh:
        fh.write("newmtl styled_texture\n")
        fh.write("Kd 1.000 1.000 1.000\n")
        fh.write(f"map_Kd {texture_path.name}\n")

    with obj_path.open("w", encoding="utf-8") as fh:
        fh.write("# DioramaForge Stage 4 depth-relief proxy\n")
        fh.write(f"mtllib {mtl_path.name}\n")
        fh.write("usemtl styled_texture\n")
        for row in range(rows):
            for col in range(cols):
                fh.write(f"v {x_values[col]:.6f} {y_values[row]:.6f} {z_values[row, col]:.6f}\n")
        for row in range(rows):
            v = 1.0 - row / max(1, rows - 1)
            for col in range(cols):
                u = col / max(1, cols - 1)
                fh.write(f"vt {u:.6f} {v:.6f}\n")
        for row in range(rows - 1):
            for col in range(cols - 1):
                a = row * cols + col + 1
                b = a + 1
                c = a + cols
                d = c + 1
                fh.write(f"f {a}/{a} {b}/{b} {d}/{d}\n")
                fh.write(f"f {a}/{a} {d}/{d} {c}/{c}\n")

    return {
        "obj": str(obj_path),
        "mtl": str(mtl_path),
        "texture": str(texture_path),
        "preview": str(preview_path),
        "vertices": rows * cols,
        "faces": (rows - 1) * (cols - 1) * 2,
        "mesh_resolution": mesh_resolution,
        "type": "depth_heightfield_proxy",
    }


def _resized_depth(depth_image: Image.Image, mesh_resolution: int) -> np.ndarray:
    gray = ImageOps.grayscale(depth_image)
    width, height = gray.size
    max_side = max(8, int(mesh_resolution))
    scale = max_side / max(width, height)
    new_size = (max(3, int(round(width * scale))), max(3, int(round(height * scale))))
    gray = gray.resize(new_size, Image.Resampling.BICUBIC)
    depth = np.asarray(gray, dtype=np.float32) / 255.0
    low, high = np.percentile(depth, [1, 99])
    if high > low:
        depth = np.clip((depth - low) / (high - low), 0.0, 1.0)
    return depth.astype(np.float32)


def _save_depth_shading(path: Path, depth: np.ndarray) -> None:
    _depth_shading_image(depth).save(path)


def _depth_shading_image(depth: np.ndarray) -> Image.Image:
    gy, gx = np.gradient(depth)
    shade = 0.72 - gx * 0.55 - gy * 0.35 + depth * 0.28
    shade = np.clip(shade, 0.0, 1.0)
    return Image.fromarray((shade * 255).astype(np.uint8), mode="L").convert("RGB")


def _write_relief_stl(
    path: Path,
    depth_image: Image.Image,
    width_mm: float,
    relief_height_mm: float,
    base_thickness_mm: float,
    mesh_resolution: int,
) -> dict[str, Any]:
    depth = _resized_depth(depth_image, mesh_resolution)
    rows, cols = depth.shape
    height_mm = width_mm * rows / max(1, cols)
    xs = np.linspace(0.0, width_mm, cols)
    ys = np.linspace(0.0, height_mm, rows)
    top_z = base_thickness_mm + depth * relief_height_mm

    triangle_count = 0
    with path.open("w", encoding="utf-8") as fh:
        fh.write("solid dioramaforge_relief_proxy\n")

        def top(row: int, col: int) -> tuple[float, float, float]:
            return (float(xs[col]), float(ys[row]), float(top_z[row, col]))

        def bottom(row: int, col: int) -> tuple[float, float, float]:
            return (float(xs[col]), float(ys[row]), 0.0)

        def face(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> None:
            nonlocal triangle_count
            nx, ny, nz = _normal(a, b, c)
            fh.write(f"  facet normal {nx:.6f} {ny:.6f} {nz:.6f}\n")
            fh.write("    outer loop\n")
            fh.write(f"      vertex {a[0]:.6f} {a[1]:.6f} {a[2]:.6f}\n")
            fh.write(f"      vertex {b[0]:.6f} {b[1]:.6f} {b[2]:.6f}\n")
            fh.write(f"      vertex {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
            fh.write("    endloop\n")
            fh.write("  endfacet\n")
            triangle_count += 1

        for row in range(rows - 1):
            for col in range(cols - 1):
                face(top(row, col), top(row, col + 1), top(row + 1, col + 1))
                face(top(row, col), top(row + 1, col + 1), top(row + 1, col))
                face(bottom(row, col), bottom(row + 1, col + 1), bottom(row, col + 1))
                face(bottom(row, col), bottom(row + 1, col), bottom(row + 1, col + 1))

        for col in range(cols - 1):
            face(top(0, col), bottom(0, col), bottom(0, col + 1))
            face(top(0, col), bottom(0, col + 1), top(0, col + 1))
            face(top(rows - 1, col), top(rows - 1, col + 1), bottom(rows - 1, col + 1))
            face(top(rows - 1, col), bottom(rows - 1, col + 1), bottom(rows - 1, col))

        for row in range(rows - 1):
            face(top(row, 0), top(row + 1, 0), bottom(row + 1, 0))
            face(top(row, 0), bottom(row + 1, 0), bottom(row, 0))
            face(top(row, cols - 1), bottom(row + 1, cols - 1), top(row + 1, cols - 1))
            face(top(row, cols - 1), bottom(row, cols - 1), bottom(row + 1, cols - 1))

        fh.write("endsolid dioramaforge_relief_proxy\n")

    return {
        "vertices_grid": [int(cols), int(rows)],
        "triangles": triangle_count,
        "width_mm": round(float(width_mm), 3),
        "height_mm": round(float(height_mm), 3),
        "relief_height_mm": round(float(relief_height_mm), 3),
        "base_thickness_mm": round(float(base_thickness_mm), 3),
        "watertight_proxy": True,
    }


def _normal(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[float, float, float]:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    vc = np.asarray(c, dtype=np.float64)
    normal = np.cross(vb - va, vc - va)
    length = float(np.linalg.norm(normal))
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    normal = normal / length
    return (float(normal[0]), float(normal[1]), float(normal[2]))


def _save_print_preview(path: Path, styled_image: Image.Image, depth_image: Image.Image) -> None:
    depth = _resized_depth(depth_image, 512)
    shaded = _depth_shading_image(depth).resize(styled_image.size, Image.Resampling.BICUBIC)
    blended = Image.blend(styled_image.convert("RGB"), shaded, 0.42)
    blended.save(path)


def _write_print_checklist(path: Path, options: Stage5Options, mesh_stats: dict[str, Any]) -> None:
    lines = [
        "# DioramaForge Stage 5 Print Checklist",
        "",
        "This file is generated for the current depth-relief proxy.",
        "",
        "## Proxy Mesh",
        "",
        f"- STL: `print_ready_relief_proxy.stl`",
        f"- Width: {options.width_mm:.2f} mm",
        f"- Relief height: {options.relief_height_mm:.2f} mm",
        f"- Base thickness: {options.base_thickness_mm:.2f} mm",
        f"- Grid: {mesh_stats['vertices_grid'][0]} x {mesh_stats['vertices_grid'][1]}",
        f"- Triangles: {mesh_stats['triangles']}",
        "",
        "## Required Manual Checks",
        "",
        "- Open the STL in a slicer and inspect the relief orientation.",
        "- Confirm the base plate is thick enough for the selected printer and material.",
        "- Check that thin peaks or cliffs are printable at the intended scale.",
        "- Treat this as a proxy until TRELLIS and UltraShape reconstruction are connected.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
