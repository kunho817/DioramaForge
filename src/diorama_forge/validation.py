from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .config import DioramaConfig


def validate_run(config: DioramaConfig, run_dir_value: str | Path) -> dict[str, Any]:
    run_dir = _resolve_run_dir(config, run_dir_value)
    checks: list[dict[str, Any]] = []
    metadata_path = run_dir / "run_metadata.json"
    metadata = _load_json(metadata_path, checks, "metadata", required=True)
    artifacts = metadata.get("artifacts", {}) if isinstance(metadata, dict) else {}

    input_path = _check_artifact(run_dir, artifacts, "input", "input.png", checks, "stage3", required=True)
    depth_path = _check_artifact(run_dir, artifacts, "depth_png", "depth.png", checks, "stage1_depth", required=True)
    _check_artifact(run_dir, artifacts, "depth_npy", "depth.npy", checks, "stage1_depth", required=True)
    mask_overlay_path = _check_artifact(
        run_dir,
        artifacts,
        "mask_overlay",
        "mask_overlay.png",
        checks,
        "stage2_segmentation",
        required=True,
    )
    region_manifest_path = _check_artifact(
        run_dir,
        artifacts,
        "region_manifest",
        "regions/region_plan.json",
        checks,
        "stage2_segmentation",
        required=True,
    )
    flux_control_path = _check_artifact(
        run_dir,
        artifacts,
        "flux_control",
        "flux_control.png",
        checks,
        "stage3_flux",
        required=True,
    )
    final_path = _check_artifact(run_dir, artifacts, "final_image", "flux_result.png", checks, "stage3_flux", required=True)

    expected_size = _image_size(input_path, checks, "stage3")
    for label, path in [
        ("depth", depth_path),
        ("mask_overlay", mask_overlay_path),
        ("flux_control", flux_control_path),
        ("flux_result", final_path),
    ]:
        _check_image_size(path, expected_size, checks, "stage3", label)

    _check_region_manifest(region_manifest_path, checks)
    stage35_status = _check_stage35(run_dir, artifacts, checks)
    stage4_status = _check_stage4(run_dir, checks)
    stage5_status = _check_stage5(run_dir, checks)
    _check_pipeline_status(metadata, {"stage35": stage35_status, "stage4": stage4_status, "stage5": stage5_status}, checks)

    errors = [check for check in checks if check["level"] == "error"]
    warnings = [check for check in checks if check["level"] == "warning"]
    stage_status = {
        "stage3": not any(check["level"] == "error" and check["stage"].startswith("stage3") for check in checks),
        "stage35": stage35_status,
        "stage4": stage4_status,
        "stage5": stage5_status,
    }
    return {
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "run_dir": str(run_dir),
        "stage_status": stage_status,
        "checks": checks,
    }


def _resolve_run_dir(config: DioramaConfig, run_dir_value: str | Path) -> Path:
    path = Path(str(run_dir_value))
    if not path.is_absolute():
        path = config.root / path
    path = path.resolve()
    outputs = (config.root / "outputs").resolve()
    if outputs != path and outputs not in path.parents:
        raise RuntimeError(f"Run path must be inside outputs: {path}")
    if not path.exists():
        raise RuntimeError(f"Run directory not found: {path}")
    return path


def _load_json(path: Path, checks: list[dict[str, Any]], stage: str, required: bool) -> dict[str, Any]:
    if not path.exists():
        _add_check(checks, stage, "error" if required else "warning", "missing_json", f"Missing JSON file: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _add_check(checks, stage, "error", "invalid_json", f"Invalid JSON file: {path} ({exc})")
        return {}


def _check_artifact(
    run_dir: Path,
    artifacts: dict[str, Any],
    key: str,
    fallback: str,
    checks: list[dict[str, Any]],
    stage: str,
    required: bool,
) -> Path | None:
    path = _artifact_path(run_dir, artifacts.get(key), fallback)
    if path.exists():
        _add_check(checks, stage, "pass", f"{key}_exists", str(path))
        return path
    level = "error" if required else "warning"
    _add_check(checks, stage, level, f"{key}_missing", f"Missing artifact `{key}`: {path}")
    return None


def _artifact_path(run_dir: Path, value: Any, fallback: str) -> Path:
    path = Path(str(value)) if value else run_dir / fallback
    if not path.is_absolute():
        path = run_dir / path
    return path


def _image_size(path: Path | None, checks: list[dict[str, Any]], stage: str) -> tuple[int, int] | None:
    if path is None or not path.exists():
        return None
    try:
        with Image.open(path) as image:
            return image.size
    except Exception as exc:
        _add_check(checks, stage, "error", "invalid_image", f"Cannot open image: {path} ({exc})")
        return None


def _check_image_size(
    path: Path | None,
    expected_size: tuple[int, int] | None,
    checks: list[dict[str, Any]],
    stage: str,
    label: str,
) -> None:
    size = _image_size(path, checks, stage)
    if path is None or size is None or expected_size is None:
        return
    if size != expected_size:
        _add_check(checks, stage, "warning", f"{label}_size_mismatch", f"{label} size {size} differs from input {expected_size}")
    else:
        _add_check(checks, stage, "pass", f"{label}_size_match", f"{label} size {size}")


def _check_region_manifest(path: Path | None, checks: list[dict[str, Any]]) -> None:
    if path is None:
        return
    manifest = _load_json(path, checks, "stage2_segmentation", required=True)
    groups = manifest.get("groups", []) if isinstance(manifest, dict) else []
    if groups:
        _add_check(checks, "stage2_segmentation", "pass", "region_groups", f"{len(groups)} semantic groups")
    else:
        _add_check(checks, "stage2_segmentation", "warning", "region_groups_empty", "No semantic region groups found")


def _check_stage35(run_dir: Path, artifacts: dict[str, Any], checks: list[dict[str, Any]]) -> bool:
    stage_dir = run_dir / "stage35_refinement"
    metadata_path = _artifact_path(run_dir, artifacts.get("stage35_metadata"), "stage35_refinement/stage35_metadata.json")
    if not stage_dir.exists() and not metadata_path.exists():
        _add_check(checks, "stage35", "warning", "stage35_missing", "Stage 3.5 outputs are not present")
        return False
    _check_artifact(run_dir, artifacts, "stage35_reconstruction_input", "stage35_refinement/stage35_reconstruction_input.png", checks, "stage35", True)
    _check_artifact(run_dir, artifacts, "stage35_refined", "stage35_refinement/stage35_refined.png", checks, "stage35", True)
    _check_artifact(run_dir, artifacts, "stage35_metadata", "stage35_refinement/stage35_metadata.json", checks, "stage35", True)
    return not any(check["stage"] == "stage35" and check["level"] == "error" for check in checks)


def _check_stage4(run_dir: Path, checks: list[dict[str, Any]]) -> bool:
    stage_dir = run_dir / "stage4_reconstruction"
    manifest_path = stage_dir / "reconstruction_package.json"
    if not manifest_path.exists():
        _add_check(checks, "stage4", "warning", "stage4_missing", "Stage 4 reconstruction package is not present")
        return False
    manifest = _load_json(manifest_path, checks, "stage4", required=True)
    _check_file(stage_dir / "part_contact_sheet.png", checks, "stage4", True, "contact_sheet")
    _check_file(stage_dir / "heightfield_proxy.obj", checks, "stage4", True, "proxy_obj")
    parts = manifest.get("parts", []) if isinstance(manifest, dict) else []
    if parts:
        _add_check(checks, "stage4", "pass", "parts", f"{len(parts)} exported parts")
    else:
        _add_check(checks, "stage4", "warning", "parts_empty", "Stage 4 has no exported parts")
    if manifest.get("backend") == "meshy_image_to_3d":
        _check_stage4_meshy(run_dir, manifest, checks)
    return not any(check["stage"] == "stage4" and check["level"] == "error" for check in checks)


def _check_stage5(run_dir: Path, checks: list[dict[str, Any]]) -> bool:
    stage_dir = run_dir / "stage5_print"
    manifest_path = stage_dir / "print_package.json"
    if not manifest_path.exists():
        _add_check(checks, "stage5", "warning", "stage5_missing", "Stage 5 print package is not present")
        return False
    manifest = _load_json(manifest_path, checks, "stage5", required=True)
    _check_file(stage_dir / "print_ready_relief_proxy.stl", checks, "stage5", True, "proxy_stl")
    _check_file(stage_dir / "print_preview.png", checks, "stage5", True, "preview")
    _check_file(stage_dir / "print_checklist.md", checks, "stage5", True, "checklist")
    if manifest.get("backend") == "meshy_model_package":
        _check_stage5_meshy_model_files(run_dir, manifest, checks)
    return not any(check["stage"] == "stage5" and check["level"] == "error" for check in checks)


def _check_stage4_meshy(run_dir: Path, manifest: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    meshy = manifest.get("meshy")
    if not isinstance(meshy, dict) or not meshy:
        _add_check(checks, "stage4", "error", "meshy_missing", "Stage 4 backend is Meshy, but meshy metadata is missing")
        return

    task_id = str(meshy.get("task_id", "")).strip()
    if task_id:
        _add_check(checks, "stage4", "pass", "meshy_task_id", f"Meshy task id: {task_id}")
    else:
        _add_check(checks, "stage4", "error", "meshy_task_id_missing", "Meshy task id is missing")

    status = str(meshy.get("status", "")).strip().upper()
    if status == "SUCCEEDED":
        _add_check(checks, "stage4", "pass", "meshy_status", "Meshy task succeeded")
    else:
        _add_check(checks, "stage4", "error", "meshy_status_not_succeeded", f"Meshy task status is not SUCCEEDED: {status or '-'}")

    for key, label in (
        ("request", "meshy_request"),
        ("task", "meshy_task"),
        ("downloads_manifest", "meshy_downloads_manifest"),
    ):
        value = meshy.get(key)
        if value:
            _check_nonempty_file(_manifest_path(run_dir, value), checks, "stage4", True, label)
        else:
            _add_check(checks, "stage4", "error", f"{label}_missing", f"Missing Meshy {key} path in Stage 4 manifest")

    downloads = meshy.get("downloads", {})
    if not isinstance(downloads, dict) or not downloads:
        _add_check(checks, "stage4", "error", "meshy_downloads_empty", "Meshy Stage 4 manifest contains no downloaded model files")
        return
    model_keys = [key for key in downloads if str(key).lower() in {"glb", "obj", "stl"}]
    if model_keys:
        _add_check(checks, "stage4", "pass", "meshy_model_downloads", f"Meshy model downloads: {', '.join(sorted(str(key) for key in model_keys))}")
    else:
        _add_check(checks, "stage4", "error", "meshy_model_downloads_missing", "Meshy downloads do not contain GLB, OBJ, or STL model files")
    for key, value in downloads.items():
        required = str(key).lower() in {"glb", "obj", "stl"}
        _check_nonempty_file(_manifest_path(run_dir, value), checks, "stage4", required, f"meshy_download_{key}")


def _check_stage5_meshy_model_files(run_dir: Path, manifest: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}
    model_files = outputs.get("model_files", {}) if isinstance(outputs, dict) else {}
    if not isinstance(model_files, dict) or not model_files:
        _add_check(checks, "stage5", "error", "meshy_model_files_empty", "Stage 5 backend is Meshy, but no model files were packaged")
        return
    model_keys = [key for key in model_files if str(key).lower() in {"glb", "obj", "stl"}]
    if model_keys:
        _add_check(checks, "stage5", "pass", "meshy_model_files", f"Stage 5 model files: {', '.join(sorted(str(key) for key in model_keys))}")
    else:
        _add_check(checks, "stage5", "error", "meshy_model_files_missing", "Stage 5 model package does not contain GLB, OBJ, or STL files")
    for key, value in model_files.items():
        required = str(key).lower() in {"glb", "obj", "stl"}
        _check_nonempty_file(_manifest_path(run_dir, value), checks, "stage5", required, f"meshy_model_{key}")


def _check_file(path: Path, checks: list[dict[str, Any]], stage: str, required: bool, code: str) -> None:
    if path.exists():
        _add_check(checks, stage, "pass", f"{code}_exists", str(path))
        return
    _add_check(checks, stage, "error" if required else "warning", f"{code}_missing", f"Missing file: {path}")


def _check_nonempty_file(path: Path, checks: list[dict[str, Any]], stage: str, required: bool, code: str) -> None:
    if not path.exists():
        _add_check(checks, stage, "error" if required else "warning", f"{code}_missing", f"Missing file: {path}")
        return
    if path.is_file() and path.stat().st_size > 0:
        _add_check(checks, stage, "pass", f"{code}_exists", str(path))
        return
    _add_check(checks, stage, "error" if required else "warning", f"{code}_empty", f"File is empty or not a file: {path}")


def _manifest_path(run_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = run_dir / path
    return path


def _check_pipeline_status(metadata: dict[str, Any], actual: dict[str, bool], checks: list[dict[str, Any]]) -> None:
    status = metadata.get("pipeline", {}).get("stage_status", {}) if isinstance(metadata, dict) else {}
    if not status:
        _add_check(checks, "pipeline", "warning", "pipeline_status_missing", "pipeline.stage_status is not recorded")
        return
    for stage, actual_value in actual.items():
        recorded = bool(status.get(stage))
        if recorded != actual_value:
            _add_check(checks, "pipeline", "warning", f"{stage}_status_mismatch", f"pipeline.stage_status.{stage}={recorded}, actual={actual_value}")
        else:
            _add_check(checks, "pipeline", "pass", f"{stage}_status_match", f"{stage}={actual_value}")


def _add_check(checks: list[dict[str, Any]], stage: str, level: str, code: str, message: str) -> None:
    checks.append({"stage": stage, "level": level, "code": code, "message": message})
