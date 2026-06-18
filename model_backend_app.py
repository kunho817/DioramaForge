from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import time
import traceback
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from src.diorama_forge.config import DioramaConfig, load_config
from src.diorama_forge.pipeline import DioramaPipeline, PipelineOptions
from src.diorama_forge.runtime import runtime_status_markdown
from src.diorama_forge.stage35 import Stage35Options, build_stage35_refinement
from src.diorama_forge.stage45 import Stage4Options, build_stage4_package


def create_model_backend_app(config_path: str | Path | None = None) -> FastAPI:
    config = load_config(config_path)
    pipeline = DioramaPipeline(config)
    app = FastAPI(
        title="DioramaForge Remote Model Backend",
        version="0.1.0",
        description="GPU model backend for Stage 3, Stage 3.5, and Stage 4 remote execution.",
    )

    @app.get("/api/remote/health")
    def health(x_dioramaforge_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_key(config, x_dioramaforge_key)
        return {
            "ok": True,
            "role": "remote_model_backend",
            "runtime": runtime_status_markdown(),
            "execution_backend": config.remote.execution_backend,
            "stage35_backend": config.remote.stage35_backend,
            "stage4_backend": config.remote.stage4_backend,
            "work_dir": str(config.remote.work_dir),
            "cleanup_after_response": config.remote.cleanup_after_response,
            "hf": _hf_status(),
            "work": _work_status(config),
            "loaded": _loaded_status(pipeline),
            "models": _remote_model_status(config),
        }

    @app.post("/api/remote/stage3/run")
    def run_stage3(
        image: UploadFile = File(...),
        preset_name: str = Form("판타지 디오라마"),
        custom_prompt: str = Form(""),
        backend_mode: str = Form("remote"),
        seed: int = Form(-1),
        steps: int = Form(24),
        guidance: float = Form(10.0),
        strength: float = Form(0.45),
        max_resolution: int = Form(512),
        x_dioramaforge_key: str | None = Header(default=None),
    ) -> StreamingResponse:
        _require_key(config, x_dioramaforge_key)
        run_dir = _new_remote_run_dir(config, "stage3")
        try:
            input_image = Image.open(image.file).convert("RGB")
            cloud_backend = _cloud_stage3_backend(config, backend_mode)
            result = pipeline.run(
                input_image,
                PipelineOptions(
                    preset_name=preset_name,
                    custom_prompt=custom_prompt,
                    seed=seed,
                    steps=steps,
                    guidance=guidance,
                    strength=strength,
                    max_resolution=max_resolution,
                    backend_mode=cloud_backend,
                ),
                status=_stage_logger("stage3"),
                run_dir=run_dir,
            )
            return _zip_response(config, result.run_dir, stage="stage3")
        except Exception as exc:
            _log_exception("stage3", exc)
            _cleanup_run_dir(config, run_dir)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/remote/stage35/run")
    def run_stage35(
        run_zip: UploadFile = File(...),
        mode: str = Form("structure_preserving"),
        backend_mode: str = Form("remote"),
        upscale_scale: float = Form(2.0),
        refinement_strength: float = Form(0.22),
        max_side: int = Form(1536),
        x_dioramaforge_key: str | None = Header(default=None),
    ) -> StreamingResponse:
        _require_key(config, x_dioramaforge_key)
        run_dir: Path | None = None
        try:
            run_dir = _extract_run_zip(config, run_zip)
            build_stage35_refinement(
                config=config,
                run_dir_value=run_dir,
                options=Stage35Options(
                    mode=mode,
                    backend_mode=_cloud_stage35_backend(config, backend_mode),
                    upscale_scale=upscale_scale,
                    refinement_strength=refinement_strength,
                    max_side=max_side,
                ),
            )
            return _zip_response(config, run_dir, stage="stage35")
        except Exception as exc:
            _log_exception("stage35", exc)
            if run_dir is not None:
                _cleanup_run_dir(config, run_dir)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/remote/stage4/run")
    def run_stage4(
        run_zip: UploadFile = File(...),
        backend_mode: str = Form("remote"),
        mesh_resolution: int = Form(96),
        max_parts: int = Form(12),
        x_dioramaforge_key: str | None = Header(default=None),
    ) -> StreamingResponse:
        _require_key(config, x_dioramaforge_key)
        run_dir: Path | None = None
        try:
            run_dir = _extract_run_zip(config, run_zip)
            build_stage4_package(
                config=config,
                run_dir_value=run_dir,
                options=Stage4Options(
                    backend_mode=_cloud_stage4_backend(config, backend_mode),
                    mesh_resolution=mesh_resolution,
                    max_parts=max_parts,
                ),
            )
            return _zip_response(config, run_dir, stage="stage4")
        except Exception as exc:
            _log_exception("stage4", exc)
            if run_dir is not None:
                _cleanup_run_dir(config, run_dir)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def _require_key(config: DioramaConfig, supplied: str | None) -> None:
    expected = os.environ.get(config.remote.api_key_env, "")
    if expected and supplied != expected:
        raise HTTPException(status_code=401, detail="Invalid DioramaForge remote API key.")


def _cloud_stage3_backend(config: DioramaConfig, requested: str) -> str:
    if str(requested).lower() in {"remote", "cloud", "a100"}:
        return config.remote.execution_backend
    return requested


def _cloud_stage35_backend(config: DioramaConfig, requested: str) -> str:
    if str(requested).lower() in {"remote", "cloud", "a100"}:
        return config.remote.stage35_backend
    return requested


def _cloud_stage4_backend(config: DioramaConfig, requested: str) -> str:
    if str(requested).lower() in {"remote", "cloud", "a100"}:
        return config.remote.stage4_backend
    return requested


def _stage_logger(stage: str):
    def emit(message: str) -> None:
        print(f"[remote][{stage}] {message}", flush=True)

    return emit


def _log_exception(stage: str, exc: Exception) -> None:
    print(f"[remote][{stage}] ERROR: {exc}", flush=True)
    traceback.print_exc()


def _package_ready(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def _hf_status() -> dict[str, Any]:
    cache_dir = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", "")) / "hub")
    return {
        "token_detected": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
        "cache_dir": str(cache_dir),
        "cache_gb": round(_path_size_bytes(cache_dir) / (1024**3), 2) if cache_dir.exists() else 0.0,
    }


def _work_status(config: DioramaConfig) -> dict[str, Any]:
    work_dir = config.remote.work_dir
    return {
        "dir": str(work_dir),
        "exists": work_dir.exists(),
        "file_count": _file_count(work_dir),
        "size_mb": round(_path_size_bytes(work_dir) / (1024**2), 2) if work_dir.exists() else 0.0,
        "cleanup_after_response": config.remote.cleanup_after_response,
    }


def _loaded_status(pipeline: DioramaPipeline) -> dict[str, bool]:
    return {
        "depth": getattr(pipeline.depth, "_da3_model", None) is not None
        or getattr(pipeline.depth, "_hf_pipe", None) is not None,
        "segmentation": getattr(pipeline.segmenter, "_sam_generator", None) is not None
        or getattr(pipeline.segmenter, "_hf_pipe", None) is not None,
        "flux": getattr(pipeline.flux, "_pipe", None) is not None,
    }


def _remote_model_status(config: DioramaConfig) -> dict[str, Any]:
    return {
        "depth": _model_cache_status(config.depth.model_id, package="depth_anything_3"),
        "depth_fallback": _model_cache_status(config.depth.fallback_model_id, package="transformers"),
        "sam": _model_cache_status(config.sam.model_id, package="sam2"),
        "flux": _model_cache_status(config.flux.model_id, package="diffusers", expect_diffusers_weights=True),
    }


def _model_cache_status(model_id: str, package: str, expect_diffusers_weights: bool = False) -> dict[str, Any]:
    model_dir = _hf_model_dir(model_id)
    snapshots = []
    if model_dir.exists():
        snapshots_dir = model_dir / "snapshots"
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()] if snapshots_dir.exists() else []
    latest = max(snapshots, key=lambda path: path.stat().st_mtime) if snapshots else None
    has_weights = _snapshot_has_weights(latest) if latest else False
    ready = latest is not None and (has_weights if expect_diffusers_weights else True)
    return {
        "model_id": model_id,
        "package": package,
        "package_ready": _package_ready(package),
        "cache_dir": str(model_dir),
        "cached": model_dir.exists(),
        "latest_snapshot": str(latest) if latest else "",
        "has_weights": has_weights,
        "ready": ready,
        "size_gb": round(_path_size_bytes(model_dir) / (1024**3), 2) if model_dir.exists() else 0.0,
    }


def _hf_model_dir(model_id: str) -> Path:
    cache_dir = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", "")) / "hub")
    return cache_dir / ("models--" + model_id.replace("/", "--"))


def _snapshot_has_weights(path: Path | None) -> bool:
    if path is None:
        return False
    for pattern in ("*.safetensors", "*.bin"):
        if next(path.rglob(pattern), None) is not None:
            return True
    return False


def _file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _extract_run_zip(config: DioramaConfig, upload: UploadFile) -> Path:
    incoming_root = config.remote.work_dir / "incoming"
    incoming_root.mkdir(parents=True, exist_ok=True)
    run_dir = incoming_root / time.strftime("%Y%m%d_%H%M%S")
    suffix = 1
    while run_dir.exists():
        run_dir = incoming_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    data = upload.file.read()
    with zipfile.ZipFile(BytesIO(data), "r") as archive:
        manifest = _read_package_manifest(archive)
        _safe_extract(archive, run_dir)
    source_run_dir = str(manifest.get("remote_run_dir") or "")
    if source_run_dir:
        _rewrite_json_paths(run_dir, source_run_dir, str(run_dir))
    return run_dir


def _new_remote_run_dir(config: DioramaConfig, stage: str) -> Path:
    stage_root = config.remote.work_dir / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    run_dir = stage_root / time.strftime("%Y%m%d_%H%M%S")
    suffix = 1
    while run_dir.exists():
        run_dir = stage_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{suffix:02d}"
        suffix += 1
    return run_dir


def _zip_response(config: DioramaConfig, run_dir: Path, stage: str) -> StreamingResponse:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file():
                if path.name == "remote_package.json":
                    continue
                archive.write(path, path.relative_to(run_dir).as_posix())
        archive.writestr(
            "remote_package.json",
            json.dumps(
                {
                    "stage": stage,
                    "run_id": run_dir.name,
                    "remote_run_dir": str(run_dir),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    buffer.seek(0)
    _cleanup_run_dir(config, run_dir)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run_dir.name}_{stage}.zip"'},
    )


def _cleanup_run_dir(config: DioramaConfig, run_dir: Path) -> None:
    if not config.remote.cleanup_after_response:
        return
    work_root = config.remote.work_dir.resolve()
    target = run_dir.resolve()
    if target == work_root or work_root not in target.parents:
        return
    shutil.rmtree(target, ignore_errors=True)
    parent = target.parent
    while parent != work_root and (parent == work_root or work_root in parent.parents):
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def _read_package_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    if "remote_package.json" not in archive.namelist():
        return {}
    with archive.open("remote_package.json") as fh:
        return json.loads(fh.read().decode("utf-8"))


def _safe_extract(archive: zipfile.ZipFile, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    for member in archive.infolist():
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError(f"Unsafe path in zip: {member.filename}")
        destination = (target_root / member.filename).resolve()
        if target_root != destination and target_root not in destination.parents:
            raise RuntimeError(f"Unsafe path in zip: {member.filename}")
        if member.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, destination.open("wb") as dst:
                dst.write(src.read())


def _rewrite_json_paths(run_dir: Path, old_prefix: str, new_prefix: str) -> None:
    for path in run_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        rewritten = _replace_prefix(data, old_prefix, new_prefix)
        if rewritten != data:
            path.write_text(json.dumps(rewritten, ensure_ascii=False, indent=2), encoding="utf-8")


def _replace_prefix(value: Any, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_prefix(item, old_prefix, new_prefix) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_prefix(item, old_prefix, new_prefix) for item in value]
    if isinstance(value, str):
        return value.replace(old_prefix, new_prefix).replace(old_prefix.replace("\\", "/"), new_prefix)
    return value


app = create_model_backend_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DioramaForge remote model backend.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9008)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_model_backend_app(args.config),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
