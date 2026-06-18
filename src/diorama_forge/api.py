from __future__ import annotations

import json
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .comfy import ComfyUIClient
from .comfy_workflow import inspect_comfy_workflow, prepare_comfy_workflow_bytes
from .config import DioramaConfig, load_config
from .demo_benchmark import demo_benchmark_status
from .demo_runtime import demo_runtime_status
from .jobs import JobManager
from .meshy import MeshyClient
from .model_status import model_status_markdown
from .pipeline import DioramaPipeline, PipelineArtifacts, PipelineOptions
from .presets import DEFAULT_PRESET, preset_names
from .remote import RemoteModelClient
from .runtime import runtime_status_markdown
from .stage35 import Stage35Options, build_stage35_refinement
from .stage45 import Stage4Options, Stage5Options, build_stage4_package, build_stage5_print_package
from .style_engine import resolve_style_engine, style_engine_readiness
from .validation import validate_run


class Stage35Request(BaseModel):
    run_dir: str
    mode: str = "structure_preserving"
    backend_mode: str = ""
    upscale_scale: float = Field(default=2.0, ge=1.0, le=4.0)
    refinement_strength: float = Field(default=0.22, ge=0.0, le=0.5)
    max_side: int = Field(default=1536, ge=512, le=4096)


class Stage4Request(BaseModel):
    run_dir: str
    backend_mode: str = ""
    mesh_resolution: int = Field(default=96, ge=32, le=256)
    max_parts: int = Field(default=12, ge=1, le=64)


class Stage5Request(BaseModel):
    run_dir: str
    backend_mode: str = "demo"
    width_mm: float = Field(default=120.0, gt=10.0, le=500.0)
    relief_height_mm: float = Field(default=18.0, gt=0.5, le=120.0)
    base_thickness_mm: float = Field(default=3.0, gt=0.2, le=50.0)
    mesh_resolution: int = Field(default=96, ge=32, le=256)


class FullPipelineOptions(BaseModel):
    stage35_mode: str = "structure_preserving"
    stage35_backend_mode: str = "demo"
    stage35_upscale_scale: float = Field(default=2.0, ge=1.0, le=4.0)
    stage35_refinement_strength: float = Field(default=0.22, ge=0.0, le=0.5)
    stage35_max_side: int = Field(default=1536, ge=512, le=4096)
    stage4_backend_mode: str = "demo"
    stage4_mesh_resolution: int = Field(default=96, ge=32, le=256)
    stage4_max_parts: int = Field(default=12, ge=1, le=64)
    stage5_backend_mode: str = "demo"
    stage5_width_mm: float = Field(default=120.0, gt=10.0, le=500.0)
    stage5_relief_height_mm: float = Field(default=18.0, gt=0.5, le=120.0)
    stage5_base_thickness_mm: float = Field(default=3.0, gt=0.2, le=50.0)
    stage5_mesh_resolution: int = Field(default=96, ge=32, le=256)


def create_api_app(config_path: str | Path | None = None) -> FastAPI:
    config = load_config(config_path)
    pipeline = DioramaPipeline(config)
    comfy = ComfyUIClient(config.comfy)
    meshy = MeshyClient(config.meshy)
    remote = RemoteModelClient(config.remote, config.root, config.app.output_dir)
    jobs = JobManager(max_workers=1)
    app = FastAPI(
        title="DioramaForge API",
        version="0.2.0",
        description="Local API for Stage 1-5 DioramaForge pipeline execution.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "tauri://localhost",
            "http://tauri.localhost",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    outputs_dir = config.root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

    def execute_stage3(input_image: Image.Image, options: PipelineOptions, status=None) -> dict[str, Any]:
        _enforce_local_execution_policy(config, options.backend_mode, "Stage 1-3")
        if options.backend_mode == "remote":
            remote_result = remote.run_stage3(
                image=input_image,
                fields={
                    "preset_name": options.preset_name,
                    "custom_prompt": options.custom_prompt,
                    "backend_mode": "remote",
                    "seed": options.seed,
                    "steps": options.steps,
                    "guidance": options.guidance,
                    "strength": options.strength,
                    "max_resolution": options.max_resolution,
                },
                status=status,
            )
            return _run_detail_response(config, remote_result.run_dir)["stage3"]
        result = pipeline.run(input_image, options, status=status)
        return _stage3_response(config, result)

    def execute_stage35(request: Stage35Request, status=None) -> dict[str, Any]:
        backend_mode = _style_backend_or_default(config, request.backend_mode)
        _enforce_local_execution_policy(config, backend_mode, "Stage 3.5")
        if backend_mode == "remote":
            run_dir = _resolve_local_run_dir(config, request.run_dir)
            remote_result = remote.run_stage35(
                run_dir=run_dir,
                fields={
                    "mode": request.mode,
                    "backend_mode": "remote",
                    "upscale_scale": request.upscale_scale,
                    "refinement_strength": request.refinement_strength,
                    "max_side": request.max_side,
                },
                status=status,
            )
            stage35 = _run_detail_response(config, remote_result.run_dir).get("stage35")
            if not stage35:
                raise RuntimeError("Remote Stage 3.5 response did not contain stage35 artifacts.")
            return stage35
        artifacts = build_stage35_refinement(
            config=config,
            run_dir_value=request.run_dir,
            options=Stage35Options(
                mode=request.mode,
                backend_mode=backend_mode,
                upscale_scale=request.upscale_scale,
                refinement_strength=request.refinement_strength,
                max_side=request.max_side,
            ),
            status=status,
            comfy_client=comfy,
        )
        return {
            "stage": "stage35",
            "stage35_dir": str(artifacts.stage35_dir),
            "visual": _artifact(config, artifacts.visual_path),
            "reconstruction": _artifact(config, artifacts.reconstruction_path),
            "refined": _artifact(config, artifacts.refined_path),
            "metadata": _artifact(config, artifacts.metadata_path),
            "log": artifacts.log,
        }

    def execute_stage4(request: Stage4Request, status=None) -> dict[str, Any]:
        backend_mode = _style_backend_or_default(config, request.backend_mode)
        _enforce_local_execution_policy(config, backend_mode, "Stage 4")
        if backend_mode == "remote":
            run_dir = _resolve_local_run_dir(config, request.run_dir)
            remote_result = remote.run_stage4(
                run_dir=run_dir,
                fields={
                    "backend_mode": "remote",
                    "mesh_resolution": request.mesh_resolution,
                    "max_parts": request.max_parts,
                },
                status=status,
            )
            stage4 = _run_detail_response(config, remote_result.run_dir).get("stage4")
            if not stage4:
                raise RuntimeError("Remote Stage 4 response did not contain stage4 artifacts.")
            return stage4
        artifacts = build_stage4_package(
            config=config,
            run_dir_value=request.run_dir,
            options=Stage4Options(
                backend_mode=backend_mode,
                mesh_resolution=request.mesh_resolution,
                max_parts=request.max_parts,
            ),
            status=status,
        )
        return {
            "stage": "stage4",
            "stage4_dir": str(artifacts.stage4_dir),
            "manifest": _artifact(config, artifacts.manifest_path),
            "contact_sheet": _artifact(config, artifacts.contact_sheet_path),
            "obj": _artifact(config, artifacts.obj_path),
            "log": artifacts.log,
        }

    def execute_stage5(request: Stage5Request, status=None) -> dict[str, Any]:
        backend_mode = _normalize_backend(request.backend_mode)
        _enforce_local_execution_policy(config, backend_mode, "Stage 5")
        artifacts = build_stage5_print_package(
            config=config,
            run_dir_value=request.run_dir,
            options=Stage5Options(
                backend_mode=backend_mode,
                width_mm=request.width_mm,
                relief_height_mm=request.relief_height_mm,
                base_thickness_mm=request.base_thickness_mm,
                mesh_resolution=request.mesh_resolution,
            ),
            status=status,
        )
        return {
            "stage": "stage5",
            "stage5_dir": str(artifacts.stage5_dir),
            "manifest": _artifact(config, artifacts.manifest_path),
            "preview": _artifact(config, artifacts.preview_path),
            "stl": _artifact(config, artifacts.stl_path),
            "checklist": _artifact(config, artifacts.checklist_path),
            "log": artifacts.log,
        }

    def execute_full_pipeline(
        input_image: Image.Image,
        stage3_options: PipelineOptions,
        full_options: FullPipelineOptions,
        status=None,
    ) -> dict[str, Any]:
        emit = status or (lambda _message: None)
        emit("Full pipeline ?쒖옉")
        stage3 = execute_stage3(input_image, stage3_options, status=emit)
        run_dir = stage3["run_dir"]
        emit(f"Stage 1-3 ?꾨즺: {Path(run_dir).name}")

        emit("Stage 3.5 ?쒖옉")
        execute_stage35(
            Stage35Request(
                run_dir=run_dir,
                mode=full_options.stage35_mode,
                backend_mode=_same_or_backend(full_options.stage35_backend_mode, stage3_options.backend_mode),
                upscale_scale=full_options.stage35_upscale_scale,
                refinement_strength=full_options.stage35_refinement_strength,
                max_side=full_options.stage35_max_side,
            ),
            status=emit,
        )
        emit("Stage 3.5 ?꾨즺")

        emit("Stage 4 ?쒖옉")
        execute_stage4(
            Stage4Request(
                run_dir=run_dir,
                backend_mode=_same_or_backend(full_options.stage4_backend_mode, stage3_options.backend_mode),
                mesh_resolution=full_options.stage4_mesh_resolution,
                max_parts=full_options.stage4_max_parts,
            ),
            status=emit,
        )
        emit("Stage 4 ?꾨즺")

        emit("Stage 5 ?쒖옉")
        execute_stage5(
            Stage5Request(
                run_dir=run_dir,
                backend_mode=full_options.stage5_backend_mode,
                width_mm=full_options.stage5_width_mm,
                relief_height_mm=full_options.stage5_relief_height_mm,
                base_thickness_mm=full_options.stage5_base_thickness_mm,
                mesh_resolution=full_options.stage5_mesh_resolution,
            ),
            status=emit,
        )
        emit("Stage 5 ?꾨즺")

        run_path = _resolve_local_run_dir(config, run_dir)
        pipeline_record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "stage": "full_pipeline",
            "requested": {
                "profile": "fixed_product_generate",
                "stage_contract": ["stage3", "stage35", "stage4", "stage5"],
            },
            "backends": {
                "stage3": stage3_options.backend_mode,
                "stage35": _same_or_backend(full_options.stage35_backend_mode, stage3_options.backend_mode),
                "stage4": _same_or_backend(full_options.stage4_backend_mode, stage3_options.backend_mode),
                "stage5": full_options.stage5_backend_mode,
            },
            "stage_status": {
                "stage3": True,
                "stage35": (run_path / "stage35_refinement" / "stage35_metadata.json").exists(),
                "stage4": (run_path / "stage4_reconstruction" / "reconstruction_package.json").exists(),
                "stage5": (run_path / "stage5_print" / "print_package.json").exists(),
            },
        }
        metadata_path = run_path / "run_metadata.json"
        metadata = _read_json(metadata_path)
        metadata["pipeline"] = pipeline_record
        _write_json(metadata_path, metadata)
        detail = _run_detail_response(config, run_path)
        detail["stage"] = "full_pipeline"
        emit("Full pipeline ?꾨즺")
        return detail

    def execute_full_pipeline(
        input_image: Image.Image,
        stage3_options: PipelineOptions,
        full_options: FullPipelineOptions,
        status=None,
    ) -> dict[str, Any]:
        emit = status or (lambda _message: None)
        stage_status = {
            "stage3": False,
            "stage35": False,
            "stage4": False,
            "stage5": False,
        }
        active = {"stage": "stage3", "label": "이미지 분석"}
        run_dir = ""

        def publish(
            message: str,
            *,
            current_stage: str | None = None,
            current_label: str | None = None,
            partial_result: dict[str, Any] | None = None,
        ) -> None:
            payload: dict[str, Any] = {
                "message": message,
                "current_stage": current_stage or active["stage"],
                "current_label": current_label or active["label"],
                "stage_status": dict(stage_status),
            }
            if run_dir:
                payload["run_dir"] = run_dir
                payload["run_id"] = Path(run_dir).name
            if partial_result:
                payload["partial_result"] = partial_result
            emit(payload)

        def publish_partial(message: str, next_stage: str, next_label: str) -> None:
            active["stage"] = next_stage
            active["label"] = next_label
            partial = _partial_pipeline_detail(config, run_dir, stage_status)
            publish(message, current_stage=next_stage, current_label=next_label, partial_result=partial)

        publish("입력 이미지를 분석할 준비를 하고 있습니다.")
        stage3 = execute_stage3(input_image, stage3_options, status=emit)
        run_dir = stage3["run_dir"]
        stage_status["stage3"] = True
        publish_partial("이미지 분석과 스타일 변환이 완료되었습니다. 첫 결과를 화면에 표시합니다.", "stage35", "구조 보정")

        execute_stage35(
            Stage35Request(
                run_dir=run_dir,
                mode=full_options.stage35_mode,
                backend_mode=_same_or_backend(full_options.stage35_backend_mode, stage3_options.backend_mode),
                upscale_scale=full_options.stage35_upscale_scale,
                refinement_strength=full_options.stage35_refinement_strength,
                max_side=full_options.stage35_max_side,
            ),
            status=emit,
        )
        stage_status["stage35"] = True
        publish_partial("구조 보정 결과가 준비되었습니다. 다음 단계로 3D 변환 패키지를 만듭니다.", "stage4", "3D 변환 준비")

        execute_stage4(
            Stage4Request(
                run_dir=run_dir,
                backend_mode=_same_or_backend(full_options.stage4_backend_mode, stage3_options.backend_mode),
                mesh_resolution=full_options.stage4_mesh_resolution,
                max_parts=full_options.stage4_max_parts,
            ),
            status=emit,
        )
        stage_status["stage4"] = True
        publish_partial("이미지 분할과 3D 변환 패키지가 준비되었습니다. 최종 출력 묶음을 정리합니다.", "stage5", "출력 정리")

        execute_stage5(
            Stage5Request(
                run_dir=run_dir,
                backend_mode=full_options.stage5_backend_mode,
                width_mm=full_options.stage5_width_mm,
                relief_height_mm=full_options.stage5_relief_height_mm,
                base_thickness_mm=full_options.stage5_base_thickness_mm,
                mesh_resolution=full_options.stage5_mesh_resolution,
            ),
            status=emit,
        )
        stage_status["stage5"] = True

        run_path = _resolve_local_run_dir(config, run_dir)
        pipeline_record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "stage": "full_pipeline",
            "requested": {
                "profile": "fixed_product_generate",
                "stage_contract": ["stage3", "stage35", "stage4", "stage5"],
            },
            "backends": {
                "stage3": stage3_options.backend_mode,
                "stage35": _same_or_backend(full_options.stage35_backend_mode, stage3_options.backend_mode),
                "stage4": _same_or_backend(full_options.stage4_backend_mode, stage3_options.backend_mode),
                "stage5": full_options.stage5_backend_mode,
            },
            "stage_status": dict(stage_status),
        }
        metadata_path = run_path / "run_metadata.json"
        metadata = _read_json(metadata_path)
        metadata["pipeline"] = pipeline_record
        _write_json(metadata_path, metadata)
        detail = _run_detail_response(config, run_path)
        detail["stage"] = "full_pipeline"
        active["stage"] = "done"
        active["label"] = "완료"
        publish("모든 결과가 준비되었습니다.", current_stage="done", current_label="완료", partial_result=detail)
        return detail

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "root": str(config.root)}

    @app.get("/api/jobs")
    def list_jobs(limit: int = 20) -> dict[str, Any]:
        return {"jobs": jobs.list_recent(limit)}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.snapshot(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc

    @app.get("/api/runs")
    def list_runs(limit: int = 20) -> dict[str, Any]:
        return {"runs": _list_runs(config, limit)}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            run_dir = _resolve_run_id(config, run_id)
            return _run_detail_response(config, run_dir)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/validate")
    def validate_run_endpoint(run_id: str) -> dict[str, Any]:
        try:
            run_dir = _resolve_run_id(config, run_id)
            return validate_run(config, run_dir)
        except Exception as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runtime")
    def runtime() -> dict[str, str]:
        return {"markdown": runtime_status_markdown()}

    @app.get("/api/models")
    def models() -> dict[str, str]:
        return {"markdown": model_status_markdown(config)}

    @app.get("/api/style-engine")
    def style_engine() -> dict[str, Any]:
        return _style_engine_response(config)

    @app.get("/api/demo/readiness")
    def demo_readiness() -> dict[str, Any]:
        return _demo_readiness_response(config)

    @app.get("/api/comfy/status")
    def comfy_status() -> dict[str, Any]:
        return comfy.status()

    @app.get("/api/comfy/workflows")
    def comfy_workflows() -> dict[str, Any]:
        return comfy.workflow_status()

    @app.get("/api/comfy/model-choices")
    def comfy_model_choices() -> dict[str, Any]:
        return comfy.model_choices()

    @app.get("/api/meshy/status")
    def meshy_status() -> dict[str, Any]:
        return meshy.status()

    @app.post("/api/comfy/workflows/{stage_name}/inspect")
    def inspect_comfy_workflow_upload(stage_name: str, workflow: UploadFile = File(...)) -> dict[str, Any]:
        try:
            data = workflow.file.read()
            if not data:
                raise RuntimeError("Uploaded workflow file is empty.")
            suffix = Path(workflow.filename or "workflow.json").suffix or ".json"
            with tempfile.TemporaryDirectory(prefix="diorama_comfy_inspect_") as tmp:
                path = Path(tmp) / f"upload{suffix}"
                path.write_bytes(data)
                return inspect_comfy_workflow(path, stage_name, output_node_id=config.comfy.output_node_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/comfy/workflows/{stage_name}/install")
    def install_comfy_workflow(stage_name: str, workflow: UploadFile = File(...)) -> dict[str, Any]:
        try:
            data = workflow.file.read()
            result = comfy.install_workflow(stage_name, data)
            if not result.get("ok"):
                errors = result.get("errors") or ["ComfyUI workflow contract is invalid."]
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": str(errors[0]),
                        "result": result,
                    },
                )
            return {
                **result,
                "workflows": comfy.workflow_status(),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/comfy/workflows/{stage_name}/install-example")
    def install_example_comfy_workflow(stage_name: str) -> dict[str, Any]:
        try:
            example_path = _comfy_example_workflow_path(config, stage_name)
            data = example_path.read_bytes()
            result = comfy.install_workflow(stage_name, data)
            if not result.get("ok"):
                errors = result.get("errors") or ["Bundled ComfyUI example workflow contract is invalid."]
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": str(errors[0]),
                        "result": result,
                    },
                )
            return {
                **result,
                "example_path": str(example_path),
                "workflows": comfy.workflow_status(),
                "model_choices": comfy.model_choices(),
                "notes": [
                    "The bundled example is now the active workflow.",
                    "If ComfyUI node/model compatibility is blocked, edit static model filenames to match /api/comfy/model-choices.",
                ],
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/comfy/workflows/{stage_name}/prepare-install")
    def prepare_install_comfy_workflow(stage_name: str, workflow: UploadFile = File(...)) -> dict[str, Any]:
        try:
            data = workflow.file.read()
            prepared = prepare_comfy_workflow_bytes(
                data=data,
                stage=stage_name,
                output_node_id=config.comfy.output_node_id,
            )
            if not prepared.get("ok"):
                errors = prepared.get("errors") or ["ComfyUI workflow could not be prepared."]
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": str(errors[0]),
                        "result": prepared,
                    },
                )
            install_result = comfy.install_workflow(stage_name, str(prepared["prepared_json"]).encode("utf-8"))
            if not install_result.get("ok"):
                errors = install_result.get("errors") or ["Prepared ComfyUI workflow contract is invalid."]
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": str(errors[0]),
                        "prepare": prepared,
                        "install": install_result,
                    },
                )
            return {
                "ok": True,
                "prepare": {
                    key: value
                    for key, value in prepared.items()
                    if key != "prepared_json"
                },
                "install": install_result,
                "workflows": comfy.workflow_status(),
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/remote/status")
    def remote_status() -> dict[str, Any]:
        return remote.status()

    @app.get("/api/execution/policy")
    def execution_policy() -> dict[str, Any]:
        return _execution_policy_response(config)

    @app.get("/api/pipeline/defaults")
    def pipeline_defaults() -> dict[str, Any]:
        return {
            "user_facing_mode": "single_generate",
            "defaults": _product_pipeline_defaults(config),
            "notes": [
                "These defaults drive the desktop Generate flow.",
                "Developer/debug endpoints may expose individual stages, but the product GUI should keep a single Generate action.",
            ],
        }

    @app.get("/api/pipeline/preflight")
    def pipeline_preflight() -> dict[str, Any]:
        return _product_generate_preflight(config)

    @app.get("/api/presets")
    def presets() -> dict[str, list[str]]:
        return {"presets": preset_names()}

    @app.post("/api/stage3/run")
    def run_stage3(
        image: UploadFile = File(...),
        preset_name: str = Form(DEFAULT_PRESET),
        custom_prompt: str = Form(""),
        backend_mode: str = Form(""),
        seed: int | None = Form(None),
        steps: int | None = Form(None),
        guidance: float | None = Form(None),
        strength: float | None = Form(None),
        max_resolution: int | None = Form(None),
    ) -> dict[str, Any]:
        try:
            image_bytes = image.file.read()
            input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            options = _stage3_options_from_values(
                config,
                preset_name,
                custom_prompt,
                backend_mode,
                seed,
                steps,
                guidance,
                strength,
                max_resolution,
            )
            return execute_stage3(input_image, options)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/stage3/jobs")
    def start_stage3_job(
        image: UploadFile = File(...),
        preset_name: str = Form(DEFAULT_PRESET),
        custom_prompt: str = Form(""),
        backend_mode: str = Form(""),
        seed: int | None = Form(None),
        steps: int | None = Form(None),
        guidance: float | None = Form(None),
        strength: float | None = Form(None),
        max_resolution: int | None = Form(None),
    ) -> dict[str, Any]:
        try:
            image_bytes = image.file.read()
            input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            options = _stage3_options_from_values(
                config,
                preset_name,
                custom_prompt,
                backend_mode,
                seed,
                steps,
                guidance,
                strength,
                max_resolution,
            )
            return jobs.submit("stage3", lambda status: execute_stage3(input_image.copy(), options, status=status))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/stage35/refine")
    def run_stage35(request: Stage35Request) -> dict[str, Any]:
        try:
            return execute_stage35(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/stage35/jobs")
    def start_stage35_job(request: Stage35Request) -> dict[str, Any]:
        try:
            return jobs.submit("stage35", lambda status: execute_stage35(request, status=status))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/stage4/package")
    def run_stage4(request: Stage4Request) -> dict[str, Any]:
        try:
            return execute_stage4(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/stage4/jobs")
    def start_stage4_job(request: Stage4Request) -> dict[str, Any]:
        try:
            return jobs.submit("stage4", lambda status: execute_stage4(request, status=status))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/stage5/print")
    def run_stage5(request: Stage5Request) -> dict[str, Any]:
        try:
            return execute_stage5(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/stage5/jobs")
    def start_stage5_job(request: Stage5Request) -> dict[str, Any]:
        try:
            return jobs.submit("stage5", lambda status: execute_stage5(request, status=status))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/pipeline/jobs")
    def start_full_pipeline_job(
        image: UploadFile = File(...),
        preset_name: str = Form(DEFAULT_PRESET),
        custom_prompt: str = Form(""),
        seed: int | None = Form(None),
        steps: int | None = Form(None),
        guidance: float | None = Form(None),
        strength: float | None = Form(None),
        max_resolution: int | None = Form(None),
    ) -> dict[str, Any]:
        try:
            stage3_options = _stage3_options_from_values(
                config,
                preset_name,
                custom_prompt,
                "",
                seed,
                steps,
                guidance,
                strength,
                max_resolution,
            )
            full_options = _full_pipeline_options_from_config(config)
            preflight = _product_generate_preflight(config, stage3_options, full_options)
            if not preflight["ok"]:
                errors = preflight.get("errors") or ["생성 전 확인을 통과하지 못했습니다."]
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": str(errors[0]),
                        "preflight": preflight,
                    },
                )
            image_bytes = image.file.read()
            input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            return jobs.submit(
                "full_pipeline",
                lambda status: execute_full_pipeline(input_image.copy(), stage3_options, full_options, status=status),
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def _stage3_response(config: DioramaConfig, result: PipelineArtifacts) -> dict[str, Any]:
    metadata = _read_json(result.metadata_path)
    payload = _stage3_from_metadata(config, result.run_dir, metadata)
    payload["log"] = result.log
    return payload


def _list_runs(config: DioramaConfig, limit: int) -> list[dict[str, Any]]:
    runs_dir = config.root / "outputs" / "runs"
    if not runs_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    for run_dir in sorted(runs_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = _read_json(metadata_path)
        artifacts = metadata.get("artifacts", {})
        remote_import = _read_json(run_dir / "remote_import.json") if (run_dir / "remote_import.json").exists() else None
        items.append(
            {
                "id": run_dir.name,
                "run_dir": str(run_dir),
                "created_at": metadata.get("created_at"),
                "modified_at": metadata_path.stat().st_mtime,
                "preset_name": metadata.get("options", {}).get("preset_name", ""),
                "backend_mode": metadata.get("options", {}).get("backend_mode", ""),
                "stage3_backend": metadata.get("options", {}).get("stage3_backend", ""),
                "seed": metadata.get("options", {}).get("seed"),
                "steps": metadata.get("options", {}).get("steps"),
                "guidance": metadata.get("options", {}).get("guidance"),
                "strength": metadata.get("options", {}).get("strength"),
                "max_resolution": metadata.get("options", {}).get("max_resolution"),
                "style_engine": metadata.get("style_engine", {}),
                "is_remote": remote_import is not None,
                "remote_base_url": remote_import.get("remote_base_url", "") if remote_import else "",
                "models": {
                    key: value.get("backend", "")
                    for key, value in metadata.get("models", {}).items()
                    if isinstance(value, dict)
                },
                "pipeline": _pipeline_summary_from_metadata(run_dir, metadata),
                "has_stage35": bool(artifacts.get("stage35_reconstruction_input") or (run_dir / "stage35_refinement").exists()),
                "has_stage4": (run_dir / "stage4_reconstruction" / "reconstruction_package.json").exists(),
                "has_stage5": (run_dir / "stage5_print" / "print_package.json").exists(),
                "thumbnail": _optional_artifact(config, _artifact_path_from_value(run_dir, artifacts.get("final_image"), "flux_result.png")),
            }
        )
        if len(items) >= max(1, int(limit)):
            break
    return items


def _run_detail_response(config: DioramaConfig, run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "run_metadata.json"
    metadata = _read_json(metadata_path)
    remote_import = _read_json(run_dir / "remote_import.json") if (run_dir / "remote_import.json").exists() else None
    return {
        "id": run_dir.name,
        "run_dir": str(run_dir),
        "summary": {
            "created_at": metadata.get("created_at"),
            "preset_name": metadata.get("options", {}).get("preset_name", ""),
            "backend_mode": metadata.get("options", {}).get("backend_mode", ""),
            "stage3_backend": metadata.get("options", {}).get("stage3_backend", ""),
            "style_engine": metadata.get("style_engine", {}),
            "is_remote": remote_import is not None,
            "remote_base_url": remote_import.get("remote_base_url", "") if remote_import else "",
            "models": {
                key: value.get("backend", "")
                for key, value in metadata.get("models", {}).items()
                if isinstance(value, dict)
            },
        },
        "pipeline": _pipeline_summary_from_metadata(run_dir, metadata),
        "stage3": _stage3_from_metadata(config, run_dir, metadata),
        "stage35": _stage35_from_metadata(config, run_dir, metadata),
        "stage4": _stage4_from_run(config, run_dir),
        "stage5": _stage5_from_run(config, run_dir),
        "metadata": _artifact(config, metadata_path),
        "validation": validate_run(config, run_dir),
        "log": metadata.get("log", []),
    }


def _partial_pipeline_detail(
    config: DioramaConfig,
    run_dir_value: str | Path,
    stage_status: dict[str, bool],
) -> dict[str, Any]:
    run_dir = _resolve_local_run_dir(config, run_dir_value)
    detail = _run_detail_response(config, run_dir)
    pipeline = dict(detail.get("pipeline") or {})
    pipeline["stage"] = "full_pipeline"
    pipeline["stage_status"] = dict(stage_status)
    detail["pipeline"] = pipeline
    detail["stage"] = "full_pipeline"
    if not all(stage_status.values()):
        detail["validation"] = None
    return detail


def _stage3_from_metadata(config: DioramaConfig, run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    artifacts = metadata.get("artifacts", {})
    remote_import_path = run_dir / "remote_import.json"
    return {
        "stage": "stage3",
        "run_dir": str(run_dir),
        "images": {
            "original": _optional_artifact(config, _artifact_path_from_value(run_dir, artifacts.get("input"), "input.png")),
            "depth": _optional_artifact(config, _artifact_path_from_value(run_dir, artifacts.get("depth_png"), "depth.png")),
            "mask_overlay": _optional_artifact(config, _artifact_path_from_value(run_dir, artifacts.get("mask_overlay"), "mask_overlay.png")),
            "region_overlay": _optional_artifact(config, _artifact_path_from_value(run_dir, artifacts.get("region_overlay"), "regions/region_overlay.png")),
            "style_control": _optional_artifact(config, _artifact_path_from_value(run_dir, artifacts.get("flux_control"), "flux_control.png")),
            "style_result": _optional_artifact(config, _artifact_path_from_value(run_dir, artifacts.get("final_image"), "flux_result.png")),
            "flux_control": _optional_artifact(config, _artifact_path_from_value(run_dir, artifacts.get("flux_control"), "flux_control.png")),
            "flux_result": _optional_artifact(config, _artifact_path_from_value(run_dir, artifacts.get("final_image"), "flux_result.png")),
        },
        "files": {
            "metadata": _artifact(config, run_dir / "run_metadata.json"),
            "final_image": _optional_artifact(config, _artifact_path_from_value(run_dir, artifacts.get("final_image"), "flux_result.png")),
            "remote_import": _optional_artifact(config, remote_import_path),
        },
        "options": metadata.get("options", {}),
        "style_engine": metadata.get("style_engine", {}),
        "models": metadata.get("models", {}),
        "structure_control": metadata.get("structure_control", {}),
        "remote_import": _read_json(remote_import_path) if remote_import_path.exists() else None,
        "log": metadata.get("log", []),
    }


def _stage35_from_metadata(config: DioramaConfig, run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any] | None:
    artifacts = metadata.get("artifacts", {})
    reconstruction = _optional_artifact(
        config,
        _artifact_path_from_value(
            run_dir,
            artifacts.get("stage35_reconstruction_input"),
            "stage35_refinement/stage35_reconstruction_input.png",
        ),
    )
    if not reconstruction:
        return None
    return {
        "stage": "stage35",
        "stage35_dir": str(run_dir / "stage35_refinement"),
        "visual": _optional_artifact(
            config,
            _artifact_path_from_value(run_dir, artifacts.get("stage35_visual"), "stage35_refinement/stage35_upscaled_visual.png"),
        ),
        "reconstruction": reconstruction,
        "refined": _optional_artifact(
            config,
            _artifact_path_from_value(run_dir, artifacts.get("stage35_refined"), "stage35_refinement/stage35_refined.png"),
        ),
        "metadata": _optional_artifact(
            config,
            _artifact_path_from_value(run_dir, artifacts.get("stage35_metadata"), "stage35_refinement/stage35_metadata.json"),
        ),
        "log": [],
    }


def _stage4_from_run(config: DioramaConfig, run_dir: Path) -> dict[str, Any] | None:
    stage4_dir = run_dir / "stage4_reconstruction"
    manifest_path = stage4_dir / "reconstruction_package.json"
    if not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    meshy = manifest.get("meshy") or {}
    meshy_downloads = meshy.get("downloads", {}) if isinstance(meshy, dict) else {}
    return {
        "stage": "stage4",
        "stage4_dir": str(stage4_dir),
        "manifest": _artifact(config, manifest_path),
        "contact_sheet": _optional_artifact(config, stage4_dir / "part_contact_sheet.png"),
        "obj": _optional_artifact(config, stage4_dir / "heightfield_proxy.obj"),
        "meshy": {
            "task_id": meshy.get("task_id", "") if isinstance(meshy, dict) else "",
            "status": meshy.get("status", "") if isinstance(meshy, dict) else "",
            "downloads": {
                key: _optional_artifact(config, Path(value))
                for key, value in meshy_downloads.items()
            },
            "task": _optional_artifact(config, Path(meshy.get("task", ""))) if isinstance(meshy, dict) and meshy.get("task") else None,
            "downloads_manifest": _optional_artifact(config, Path(meshy.get("downloads_manifest", ""))) if isinstance(meshy, dict) and meshy.get("downloads_manifest") else None,
        },
        "summary": {
            "backend": manifest.get("backend", ""),
            "part_count": len(manifest.get("parts", [])),
            "styled_image_source": manifest.get("inputs", {}).get("styled_image_source", ""),
            "mesh_resolution": manifest.get("proxy_mesh", {}).get("mesh_resolution"),
        },
        "log": [],
    }


def _stage5_from_run(config: DioramaConfig, run_dir: Path) -> dict[str, Any] | None:
    stage5_dir = run_dir / "stage5_print"
    manifest_path = stage5_dir / "print_package.json"
    if not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    model_files = manifest.get("outputs", {}).get("model_files", {})
    return {
        "stage": "stage5",
        "stage5_dir": str(stage5_dir),
        "manifest": _artifact(config, manifest_path),
        "preview": _optional_artifact(config, stage5_dir / "print_preview.png"),
        "stl": _optional_artifact(config, stage5_dir / "print_ready_relief_proxy.stl"),
        "checklist": _optional_artifact(config, stage5_dir / "print_checklist.md"),
        "model_files": {
            key: _optional_artifact(config, Path(value))
            for key, value in model_files.items()
        } if isinstance(model_files, dict) else {},
        "summary": {
            "backend": manifest.get("backend", ""),
            "print_settings": manifest.get("print_settings", {}),
            "mesh_stats": manifest.get("mesh_stats", {}),
        },
        "log": [],
    }


def _artifact(config: DioramaConfig, path: Path) -> dict[str, str]:
    path = path.resolve()
    try:
        relative = path.relative_to((config.root / "outputs").resolve())
        url = "/outputs/" + "/".join(relative.parts)
    except ValueError:
        url = ""
    return {"path": str(path), "url": url}


def _optional_artifact(config: DioramaConfig, path: Path) -> dict[str, str] | None:
    return _artifact(config, path) if path.exists() else None


def _artifact_path_from_value(run_dir: Path, value: Any, fallback: str) -> Path:
    path = Path(str(value)) if value else run_dir / fallback
    if not path.is_absolute():
        path = run_dir / path
    return path


def _resolve_run_id(config: DioramaConfig, run_id: str) -> Path:
    safe_id = str(run_id or "").strip().strip('"')
    if not safe_id or "/" in safe_id or "\\" in safe_id or safe_id in {".", ".."}:
        raise RuntimeError(f"Invalid run id: {run_id}")
    runs_dir = (config.root / "outputs" / "runs").resolve()
    path = (runs_dir / safe_id).resolve()
    if runs_dir not in path.parents:
        raise RuntimeError(f"Invalid run id: {run_id}")
    if not (path / "run_metadata.json").exists():
        raise RuntimeError(f"Run metadata not found: {safe_id}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _pipeline_summary_from_metadata(run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    pipeline = metadata.get("pipeline")
    if isinstance(pipeline, dict):
        return pipeline
    return {
        "stage": "partial_or_manual",
        "stage_status": {
            "stage3": True,
            "stage35": bool(
                metadata.get("artifacts", {}).get("stage35_reconstruction_input")
                or (run_dir / "stage35_refinement" / "stage35_metadata.json").exists()
            ),
            "stage4": (run_dir / "stage4_reconstruction" / "reconstruction_package.json").exists(),
            "stage5": (run_dir / "stage5_print" / "print_package.json").exists(),
        },
    }


def _normalize_backend(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized in {"real", "real models only"}:
        return "real"
    if normalized in {"remote", "cloud", "a100", "remote a100", "elice", "elice cloud"}:
        return "remote"
    if normalized in {"comfy", "comfyui", "comfyui backend"}:
        return "comfyui"
    if normalized in {"meshy", "meshy ai", "meshy_ai", "image-to-3d"}:
        return "meshy"
    if normalized == "demo":
        return "demo"
    return "auto"


def _style_backend_or_default(config: DioramaConfig, value: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = config.style_engine.backend_mode
    return _normalize_backend(text)


def _same_or_backend(value: str, fallback: str) -> str:
    normalized = str(value or "same").strip().lower()
    if normalized in {"same", "stage3", "inherit"}:
        return fallback
    return _normalize_backend(normalized)


def _enforce_local_execution_policy(config: DioramaConfig, backend_mode: str, stage_label: str) -> None:
    normalized = _normalize_backend(backend_mode)
    if normalized in {"remote", "demo", "comfyui", "meshy"}:
        return
    if config.app.allow_local_heavy_models:
        return
    raise RuntimeError(
        f"{stage_label}: local heavy model execution is disabled. "
        "Enable allow_local_heavy_models in configs/default.json or set "
        "DIORAMA_ALLOW_LOCAL_HEAVY_MODELS=1 before restarting the API server."
    )


def _execution_policy_response(config: DioramaConfig) -> dict[str, Any]:
    return {
        "allow_local_heavy_models": config.app.allow_local_heavy_models,
        "allowed_backends": ["remote", "meshy", "demo"] if not config.app.allow_local_heavy_models else ["auto", "remote", "comfyui", "meshy", "real", "demo"],
        "blocked_local_backends": [] if config.app.allow_local_heavy_models else ["auto", "comfyui", "real"],
        "user_facing_mode": "single_generate",
        "backend_selector_visible": config.style_engine.show_backend_selector,
        "product_pipeline_defaults": _product_pipeline_defaults(config),
        "reason": (
            "Local heavy model execution is disabled to prevent slow local data acquisition."
            if not config.app.allow_local_heavy_models
            else "Local heavy model execution is explicitly enabled by configuration or environment."
        ),
        "override_env": "DIORAMA_ALLOW_LOCAL_HEAVY_MODELS=1",
    }


def _value_or_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def _stage3_options_from_values(
    config: DioramaConfig,
    preset_name: str,
    custom_prompt: str,
    backend_mode: str,
    seed: int | None,
    steps: int | None,
    guidance: float | None,
    strength: float | None,
    max_resolution: int | None,
) -> PipelineOptions:
    defaults = _product_pipeline_defaults(config)
    return PipelineOptions(
        preset_name=preset_name,
        custom_prompt=custom_prompt,
        seed=int(_value_or_default(seed, defaults["seed"])),
        steps=int(_value_or_default(steps, defaults["steps"])),
        guidance=float(_value_or_default(guidance, defaults["guidance"])),
        strength=float(_value_or_default(strength, defaults["strength"])),
        max_resolution=int(_value_or_default(max_resolution, defaults["max_resolution"])),
        backend_mode=_style_backend_or_default(config, backend_mode),
    )


def _product_pipeline_defaults(config: DioramaConfig) -> dict[str, Any]:
    defaults = config.product_pipeline
    return {
        "profile": "fixed_product_generate",
        "stage_contract": ["stage3", "stage35", "stage4", "stage5"],
        "seed": defaults.seed,
        "steps": defaults.steps,
        "guidance": defaults.guidance,
        "strength": defaults.strength,
        "max_resolution": defaults.max_resolution,
        "demo_time_budget_seconds": defaults.demo_time_budget_seconds,
        "demo_min_free_vram_gb": defaults.demo_min_free_vram_gb,
        "stage35_upscale_scale": defaults.stage35_upscale_scale,
        "stage35_refinement_strength": defaults.stage35_refinement_strength,
        "stage35_max_side": defaults.stage35_max_side,
        "stage4_mesh_resolution": defaults.stage4_mesh_resolution,
        "stage4_max_parts": defaults.stage4_max_parts,
        "stage5_width_mm": defaults.stage5_width_mm,
        "stage5_relief_height_mm": defaults.stage5_relief_height_mm,
        "stage5_base_thickness_mm": defaults.stage5_base_thickness_mm,
        "stage5_mesh_resolution": defaults.stage5_mesh_resolution,
    }


def _full_pipeline_options_from_config(config: DioramaConfig) -> FullPipelineOptions:
    defaults = config.product_pipeline
    return FullPipelineOptions(
        stage35_mode="structure_preserving",
        stage35_backend_mode=defaults.stage35_backend_mode,
        stage35_upscale_scale=defaults.stage35_upscale_scale,
        stage35_refinement_strength=defaults.stage35_refinement_strength,
        stage35_max_side=defaults.stage35_max_side,
        stage4_backend_mode=defaults.stage4_backend_mode,
        stage4_mesh_resolution=defaults.stage4_mesh_resolution,
        stage4_max_parts=defaults.stage4_max_parts,
        stage5_backend_mode=defaults.stage5_backend_mode,
        stage5_width_mm=defaults.stage5_width_mm,
        stage5_relief_height_mm=defaults.stage5_relief_height_mm,
        stage5_base_thickness_mm=defaults.stage5_base_thickness_mm,
        stage5_mesh_resolution=defaults.stage5_mesh_resolution,
    )


def _product_generate_preflight(
    config: DioramaConfig,
    stage3_options: PipelineOptions | None = None,
    full_options: FullPipelineOptions | None = None,
) -> dict[str, Any]:
    if stage3_options is None:
        stage3_options = _stage3_options_from_values(
            config,
            DEFAULT_PRESET,
            "",
            "",
            None,
            None,
            None,
            None,
            None,
        )
    if full_options is None:
        full_options = _full_pipeline_options_from_config(config)

    backend_mode = _style_backend_or_default(config, stage3_options.backend_mode)
    resolved_model = resolve_style_engine(config)
    resolved_engine = "comfyui" if backend_mode == "comfyui" else resolved_model
    runtime = demo_runtime_status(config)
    readiness = style_engine_readiness(config)
    timed_smoke = demo_benchmark_status(config)
    product_3d_backend = _product_3d_backend_status(config)
    can_generate = config.app.allow_local_heavy_models or backend_mode in {"demo", "comfyui", "remote"}

    stage_backends = {
        "stage3": backend_mode,
        "stage35": _same_or_backend(full_options.stage35_backend_mode, backend_mode),
        "stage4": _same_or_backend(full_options.stage4_backend_mode, backend_mode),
        "stage5": _normalize_backend(full_options.stage5_backend_mode),
    }

    image_backend_ready, image_backend_detail = _product_image_backend_preflight_detail(
        backend_mode=backend_mode,
        resolved_engine=resolved_engine,
        runtime=runtime,
        readiness=readiness,
    )
    product_3d_ready = bool(product_3d_backend.get("ready", True))

    checks = [
        {
            "id": "single_generate_contract",
            "label": "단일 생성 흐름",
            "ok": True,
            "blocking": True,
            "detail": "생성은 Stage 3 -> Stage 3.5 -> Stage 4 -> Stage 5 순서로 고정되어 있습니다.",
        },
        {
            "id": "local_execution_policy",
            "label": "로컬 실행 정책",
            "ok": can_generate,
            "blocking": True,
            "detail": (
                "현재 Stage 3 출력 엔진은 실행 정책에서 허용됩니다."
                if can_generate
                else "현재 Stage 3 출력 엔진은 로컬 고부하 모델 실행 정책에 막혀 있습니다."
            ),
        },
        {
            "id": "image_backend",
            "label": "Stage 3 이미지 엔진",
            "ok": image_backend_ready,
            "blocking": True,
            "detail": image_backend_detail,
        },
        {
            "id": "product_3d_backend",
            "label": "Stage 4/5 3D 백엔드",
            "ok": product_3d_ready,
            "blocking": True,
            "detail": str(product_3d_backend.get("detail") or "Stage 4/5 백엔드가 준비되어 있습니다."),
        },
        {
            "id": "timed_smoke",
            "label": "실행 시간 검증",
            "ok": bool(timed_smoke.get("verified")),
            "blocking": False,
            "detail": (
                f"최근 시간 검증이 {timed_smoke.get('elapsed_seconds')}초에 완료되었습니다."
                if timed_smoke.get("verified")
                else "현재 생성 프로필로 완료된 시간 검증 기록이 없습니다."
            ),
        },
    ]

    blocking_failures = [check for check in checks if check.get("blocking") and not check.get("ok")]
    warnings: list[str] = []
    if not timed_smoke.get("verified"):
        warnings.append(
            "시연 시간 안에 끝나는지는 아직 검증되지 않았습니다. 생성은 가능하지만 실제 소요 시간은 달라질 수 있습니다."
        )

    if blocking_failures:
        first_failure = blocking_failures[0]
        next_action = str(first_failure.get("detail") or "생성 전 확인에서 실패한 항목을 먼저 해결해 주세요.")
        if first_failure["id"] == "image_backend" and backend_mode == "comfyui":
            next_action = _comfyui_readiness_next_action(config, runtime)
        elif first_failure["id"] == "product_3d_backend":
            next_action = str(product_3d_backend.get("next_action") or next_action)
    elif not timed_smoke.get("verified"):
        next_action = "시연 시간 안에 끝나는지 확인하려면 시간 측정 벤치마크를 한 번 실행해 주세요."
    else:
        next_action = "현재 고정 생성 프로필로 시작할 수 있습니다."

    return {
        "ok": not blocking_failures,
        "user_facing_mode": "single_generate",
        "stage_contract": ["stage3", "stage35", "stage4", "stage5"],
        "backend_mode": backend_mode,
        "resolved_engine": resolved_engine,
        "can_generate": can_generate,
        "backends": stage_backends,
        "defaults": _product_pipeline_defaults(config),
        "checks": checks,
        "errors": [str(check.get("detail") or check["label"]) for check in blocking_failures],
        "warnings": warnings,
        "next_action": next_action,
        "runtime": runtime,
        "product_3d_backend": product_3d_backend,
        "timed_smoke": timed_smoke,
    }


def _product_image_backend_preflight_detail(
    backend_mode: str,
    resolved_engine: str,
    runtime: dict[str, Any],
    readiness: dict[str, Any],
) -> tuple[bool, str]:
    if backend_mode == "demo":
        return True, "데모 이미지 엔진이 설정되어 있어 Stage 3 고부하 모델 확인이 필요하지 않습니다."
    if backend_mode == "remote":
        return True, "레거시 원격 이미지 엔진이 내부 설정으로 연결되어 있습니다."
    if backend_mode == "comfyui":
        if runtime.get("ready"):
            return True, "ComfyUI 서버, Stage 3 워크플로, 필요한 노드가 준비되어 있습니다."
        failed = ", ".join(str(item) for item in runtime.get("failed_checks") or [])
        return False, f"ComfyUI 이미지 엔진이 아직 준비되지 않았습니다: {failed or '실행 환경 확인 실패'}."

    if resolved_engine == "sdxl_depth_lightning":
        sdxl = readiness.get("sdxl_depth_lightning", {})
        if runtime.get("ready") and sdxl.get("ready"):
            return True, "SDXL depth-lightning 구성과 로컬 실행 환경이 준비되어 있습니다."
        missing = _missing_style_components(sdxl)
        return False, f"SDXL depth-lightning 백엔드가 아직 준비되지 않았습니다: {', '.join(missing) or '실행 환경 확인 실패'}."

    flux = readiness.get("flux_depth", {})
    if runtime.get("ready") and flux.get("ready"):
        return True, "FLUX Depth 구성과 로컬 실행 환경이 준비되어 있습니다."
    missing = _missing_style_components(flux)
    return False, f"FLUX Depth 백엔드가 아직 준비되지 않았습니다: {', '.join(missing) or '실행 환경 확인 실패'}."


def _comfyui_readiness_next_action(config: DioramaConfig, runtime: dict[str, Any]) -> str:
    failed = {str(item) for item in runtime.get("failed_checks") or []}
    base_url = str(runtime.get("comfyui", {}).get("base_url") or config.comfy.base_url)
    if "Stage 3 workflow" in failed:
        return "스타일 변환용 ComfyUI 워크플로를 먼저 설치해 주세요."
    if "ComfyUI server" in failed:
        return f"ComfyUI를 {base_url}에서 실행한 뒤 상태를 새로고침해 주세요."
    if "ComfyUI node/model choices" in failed:
        return "워크플로의 checkpoint와 ControlNet 파일명이 ComfyUI에 설치된 모델명과 맞는지 확인해 주세요."
    if "Requests" in failed:
        return "ComfyUI API 확인에 필요한 Python requests 패키지를 복구해 주세요."
    return "생성 전 확인에서 실패한 ComfyUI 항목을 먼저 해결해 주세요."


def _missing_style_components(payload: dict[str, Any]) -> list[str]:
    components = payload.get("components", {}) if isinstance(payload, dict) else {}
    return [
        str(name)
        for name, component in components.items()
        if isinstance(component, dict) and not component.get("ready")
    ]


def _comfy_example_workflow_path(config: DioramaConfig, stage_name: str) -> Path:
    normalized = str(stage_name or "").strip().lower()
    if normalized not in {"stage3", "3", "style"}:
        raise RuntimeError(f"No bundled ComfyUI example workflow is available for stage: {stage_name}")
    path = config.root / "workflows" / "comfy" / "examples" / "stage3_sdxl_depth_img2img_api.example.json"
    if not path.exists():
        raise RuntimeError(f"Bundled ComfyUI Stage 3 example workflow is missing: {path}")
    return path


def _style_engine_response(config: DioramaConfig) -> dict[str, Any]:
    sdxl_settings = config.sdxl_depth_lightning
    configured_active = config.style_engine.active.strip().lower()
    backend_mode = _style_backend_or_default(config, "")
    resolved_model = resolve_style_engine(config)
    resolved_active = "comfyui" if backend_mode == "comfyui" else resolved_model
    sdxl_active = resolved_model == "sdxl_depth_lightning"
    readiness = style_engine_readiness(config)
    timed_smoke = demo_benchmark_status(config)
    runtime = demo_runtime_status(config)
    sdxl_status = readiness["sdxl_depth_lightning"]
    missing = [
        name
        for name, component in sdxl_status.get("components", {}).items()
        if not component.get("ready")
    ]
    if backend_mode == "comfyui":
        missing = [str(item) for item in runtime.get("failed_checks", [])]
    can_generate = config.app.allow_local_heavy_models or backend_mode in {"demo", "comfyui"}
    fast_path_ready = bool(runtime["ready"]) if backend_mode == "comfyui" else (sdxl_active and sdxl_status["ready"])
    product_3d_backend = _product_3d_backend_status(config)
    product_3d_ready = bool(product_3d_backend.get("ready", True))
    demo_checks, next_action = _demo_readiness_checks(
        config=config,
        resolved=resolved_active,
        backend_mode=backend_mode,
        missing_fast_path_components=missing,
        fast_path_ready=fast_path_ready,
        can_generate=can_generate,
        timed_smoke=timed_smoke,
        runtime=runtime,
        product_3d_backend=product_3d_backend,
    )
    return {
        "active": resolved_active,
        "configured_active": configured_active,
        "resolved_active": resolved_active,
        "target": config.style_engine.target,
        "backend_mode": backend_mode,
        "result_label": config.style_engine.result_label,
        "control_label": config.style_engine.control_label,
        "show_backend_selector": config.style_engine.show_backend_selector,
        "legacy_remote_visible": config.style_engine.legacy_remote_visible,
        "current_model_id": (
            config.comfy.base_url
            if backend_mode == "comfyui"
            else sdxl_settings.base_model_id if sdxl_active else config.flux.model_id
        ),
        "current_adapter": (
            "ComfyUI Stage 3 workflow"
            if backend_mode == "comfyui"
            else "SDXL ControlNet Depth + Lightning LoRA"
            if sdxl_active
            else "FLUX.1 Depth via Diffusers compatibility path"
        ),
        "replacement_candidate": (
            "ComfyUI Stage 3 workflow"
            if backend_mode == "comfyui"
            else "SDXL ControlNet Depth + SDXL-Lightning"
        ),
        "replacement_ready": fast_path_ready,
        "fast_path_ready": fast_path_ready,
        "demo_ready": fast_path_ready and can_generate and runtime["ready"] and timed_smoke["verified"] and product_3d_ready,
        "product_3d_backend": product_3d_backend,
        "timed_smoke": timed_smoke,
        "runtime": runtime,
        "demo_checks": demo_checks,
        "next_action": next_action,
        "readiness": readiness,
        "prepare_command": "powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\prepare_style_sdxl.ps1 -Download",
        "benchmark_command": ".\\.venv\\Scripts\\python.exe scripts\\benchmark_style_engine.py --run --force-engine auto",
        "benchmark_check_command": ".\\.venv\\Scripts\\python.exe scripts\\check_demo_benchmark.py --require",
        "sdxl_depth_lightning": {
            "base_model_id": sdxl_settings.base_model_id,
            "controlnet_model_id": sdxl_settings.controlnet_model_id,
            "lora_model_id": sdxl_settings.lora_model_id,
            "lora_weight_name": sdxl_settings.lora_weight_name,
            "scheduler": sdxl_settings.scheduler,
            "controlnet_conditioning_scale": sdxl_settings.controlnet_conditioning_scale,
            "lora_scale": sdxl_settings.lora_scale,
        },
        "notes": [
            "The GUI exposes a single Generate flow.",
            "With style_engine.active=auto, SDXL depth-lightning is selected only after the SDXL base, depth ControlNet, and Lightning LoRA are cached locally.",
        ],
    }


def _demo_readiness_response(config: DioramaConfig) -> dict[str, Any]:
    backend_mode = _style_backend_or_default(config, "")
    resolved_model = resolve_style_engine(config)
    resolved = "comfyui" if backend_mode == "comfyui" else resolved_model
    readiness = style_engine_readiness(config)
    sdxl = readiness["sdxl_depth_lightning"]
    timed_smoke = demo_benchmark_status(config)
    runtime = demo_runtime_status(config)
    missing = [
        name
        for name, component in sdxl.get("components", {}).items()
        if not component.get("ready")
    ]
    if backend_mode == "comfyui":
        missing = [str(item) for item in runtime.get("failed_checks", [])]
    fast_path_ready = bool(runtime["ready"]) if backend_mode == "comfyui" else (
        resolved == "sdxl_depth_lightning" and sdxl["ready"]
    )
    can_generate = config.app.allow_local_heavy_models or backend_mode in {"demo", "comfyui"}
    product_3d_backend = _product_3d_backend_status(config)
    product_3d_ready = bool(product_3d_backend.get("ready", True))
    warnings: list[str] = []
    if not fast_path_ready:
        warnings.append(
            "ComfyUI 이미지 엔진이 아직 준비되지 않아 계획된 Stage 3 경로를 실행할 수 없습니다."
            if backend_mode == "comfyui"
            else "설정된 빠른 로컬 이미지 엔진이 아직 준비되지 않았습니다."
        )
    elif not timed_smoke["verified"]:
        warnings.append("현재 생성 프로필이 시연 시간 안에 끝나는지는 아직 검증되지 않았습니다.")
    if not can_generate:
        warnings.append("로컬 고부하 모델 실행이 설정 정책에 의해 잠겨 있습니다.")
    if not product_3d_ready:
        warnings.append(str(product_3d_backend.get("detail") or "설정된 3D 백엔드가 아직 준비되지 않았습니다."))
    checks, next_action = _demo_readiness_checks(
        config=config,
        resolved=resolved,
        backend_mode=backend_mode,
        missing_fast_path_components=missing,
        fast_path_ready=fast_path_ready,
        can_generate=can_generate,
        timed_smoke=timed_smoke,
        runtime=runtime,
        product_3d_backend=product_3d_backend,
    )
    return {
        "ok": fast_path_ready and can_generate and runtime["ready"] and timed_smoke["verified"] and product_3d_ready,
        "fast_path_ready": fast_path_ready,
        "runtime_ready": runtime["ready"],
        "timed_smoke_ready": timed_smoke["verified"],
        "product_3d_ready": product_3d_ready,
        "demo_time_budget_seconds": config.product_pipeline.demo_time_budget_seconds,
        "can_generate": can_generate,
        "configured_engine": config.style_engine.active,
        "resolved_engine": resolved,
        "fallback_engine": "flux_depth" if resolved == "flux_depth" else "",
        "missing_fast_path_components": missing,
        "runtime": runtime,
        "timed_smoke": timed_smoke,
        "product_3d_backend": product_3d_backend,
        "product_pipeline_defaults": _product_pipeline_defaults(config),
        "prepare_command": "powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\prepare_style_sdxl.ps1 -Download",
        "benchmark_command": ".\\.venv\\Scripts\\python.exe scripts\\benchmark_style_engine.py --run --force-engine auto",
        "benchmark_check_command": ".\\.venv\\Scripts\\python.exe scripts\\check_demo_benchmark.py --require",
        "checks": checks,
        "next_action": next_action,
        "warnings": warnings,
    }


def _demo_readiness_checks(
    config: DioramaConfig,
    resolved: str,
    backend_mode: str,
    missing_fast_path_components: list[str],
    fast_path_ready: bool,
    can_generate: bool,
    timed_smoke: dict[str, Any],
    runtime: dict[str, Any],
    product_3d_backend: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    budget = config.product_pipeline.demo_time_budget_seconds
    product_3d_backend = product_3d_backend or _product_3d_backend_status(config)
    product_3d_ready = bool(product_3d_backend.get("ready", True))
    image_backend_label = "ComfyUI 이미지 엔진" if backend_mode == "comfyui" else "빠른 로컬 이미지 엔진"
    image_backend_id = "comfyui_image_backend" if backend_mode == "comfyui" else "sdxl_fast_path"
    image_backend_detail = (
        f"{resolved}로 연결되었고, ComfyUI 서버와 필요한 워크플로가 준비되어 있습니다."
        if backend_mode == "comfyui" and fast_path_ready
        else f"{resolved}로 연결되었지만 확인이 필요한 항목이 있습니다: {', '.join(missing_fast_path_components) or '알 수 없음'}."
        if backend_mode == "comfyui"
        else f"{resolved}로 연결되었고 필요한 SDXL 구성요소가 준비되어 있습니다."
        if fast_path_ready
        else f"{resolved}로 연결되었지만 누락된 구성요소가 있습니다: {', '.join(missing_fast_path_components) or '알 수 없음'}."
    )
    checks = [
        {
            "id": "single_generate_gui",
            "label": "단일 생성 GUI",
            "ok": True,
            "detail": "데스크톱 GUI는 하나의 생성 버튼만 제공하고 백엔드/모드 전환을 노출하지 않습니다.",
        },
        {
            "id": "local_execution_policy",
            "label": "로컬 생성 정책",
            "ok": can_generate,
            "detail": (
                "로컬 고부하 모델 실행이 허용되어 있거나, 현재 백엔드가 데모/ComfyUI 경로입니다."
                if can_generate
                else "로컬 고부하 모델 실행이 설정에 의해 잠겨 있습니다."
            ),
        },
        {
            "id": "local_runtime",
            "label": "로컬 실행 환경",
            "ok": bool(runtime.get("ready")),
            "detail": (
                "필요 패키지, CUDA, 여유 VRAM이 로컬 시연 프로필 기준을 통과했습니다."
                if runtime.get("ready")
                else ", ".join(runtime.get("failed_checks") or ["실행 환경 확인에 실패했습니다."])
            ),
        },
        {
            "id": image_backend_id,
            "label": image_backend_label,
            "ok": fast_path_ready,
            "detail": image_backend_detail,
        },
        {
            "id": "product_3d_backend",
            "label": "3D 백엔드",
            "ok": product_3d_ready,
            "detail": str(product_3d_backend.get("detail") or "설정된 3D 백엔드가 준비되어 있습니다."),
        },
        {
            "id": "timed_smoke",
            "label": "실행 시간 검증",
            "ok": bool(timed_smoke.get("verified")),
            "detail": (
                f"최근 시간 검증이 {budget}초 예산 안에서 {timed_smoke.get('elapsed_seconds')}초에 완료되었습니다."
                if timed_smoke.get("verified")
                else (timed_smoke.get("failures") or ["현재 생성 프로필로 완료된 시간 검증 기록이 없습니다."])[0]
            ),
        },
    ]
    if not can_generate:
        next_action = "로컬 고부하 모델 실행을 허용하거나 내부 백엔드를 데모로 바꿔 proxy 확인만 진행해 주세요."
    elif backend_mode == "comfyui" and not fast_path_ready:
        next_action = _comfyui_readiness_next_action(config, runtime)
    elif not product_3d_ready:
        next_action = str(product_3d_backend.get("next_action") or "설정된 Stage 4/5 백엔드를 먼저 준비해 주세요.")
    elif not runtime.get("ready"):
        next_action = "준비 상태 확인에서 표시된 CUDA/PyTorch/Diffusers 문제를 먼저 해결해 주세요."
    elif not fast_path_ready:
        next_action = "빠른 로컬 이미지 백엔드에 필요한 모델 파일을 먼저 준비해 주세요."
    elif not timed_smoke.get("verified"):
        next_action = "실제 시연 전에 현재 생성 프로필로 시간 측정 벤치마크를 한 번 실행해 주세요."
    else:
        next_action = "현재 설정으로 로컬 시연을 진행할 수 있습니다."
    return checks, next_action


def _product_3d_backend_status(config: DioramaConfig) -> dict[str, Any]:
    stage4_backend = _normalize_backend(config.product_pipeline.stage4_backend_mode)
    stage5_backend = _normalize_backend(config.product_pipeline.stage5_backend_mode)
    if "meshy" not in {stage4_backend, stage5_backend}:
        return {
            "required": False,
            "ready": True,
            "stage4_backend": stage4_backend,
            "stage5_backend": stage5_backend,
            "detail": "현재 생성 프로필에서는 Stage 4/5에 Meshy가 필요하지 않습니다.",
        }
    status = MeshyClient(config.meshy).status()
    key_env = status.get("api_key_env") or "MESHY_API_KEY"
    if status.get("ok"):
        detail = f"Meshy AI가 Stage 4/5 출력 형식으로 설정되어 있습니다: {', '.join(status.get('target_formats') or [])}."
        next_action = "Stage 3 ComfyUI 워크플로가 준비된 뒤 실제 생성 시간을 확인해 주세요."
    elif not status.get("api_key_present"):
        detail = f"Meshy AI가 Stage 4/5로 설정되어 있지만 {key_env}가 설정되어 있지 않습니다."
        next_action = f"실제 Meshy 실행 전에 {key_env}를 설정하거나 내부 Stage 4/5 백엔드를 데모로 바꿔 주세요."
    elif not status.get("requests_ready"):
        detail = "Meshy AI가 Stage 4/5로 설정되어 있지만 Python requests 패키지가 준비되지 않았습니다."
        next_action = "Stage 4/5 Meshy 패키징을 실행하기 전에 Python 환경을 복구해 주세요."
    elif not status.get("download_outputs_ready"):
        detail = "Meshy AI가 Stage 4/5로 설정되어 있지만 meshy_ai.download_outputs가 false입니다."
        next_action = "Stage 5가 반환된 모델 파일을 묶을 수 있도록 meshy_ai.download_outputs=true로 설정해 주세요."
    elif not status.get("model_output_formats_ready"):
        detail = "Meshy AI가 Stage 4/5로 설정되어 있지만 target_formats에 GLB, OBJ, STL 출력이 없습니다."
        next_action = "제품 파이프라인 실행 전에 meshy_ai.target_formats에 glb, obj, stl 중 하나 이상을 추가해 주세요."
    else:
        detail = "Meshy AI가 Stage 4/5로 설정되어 있지만 아직 준비되지 않았습니다."
        next_action = "정확한 Meshy 백엔드 문제는 진단 정보에서 확인해 주세요."
    return {
        "required": True,
        "ready": bool(status.get("ok")),
        "stage4_backend": stage4_backend,
        "stage5_backend": stage5_backend,
        "detail": detail,
        "next_action": next_action,
        "meshy": status,
    }


def _resolve_local_run_dir(config: DioramaConfig, value: str | Path) -> Path:
    text = str(value or "").strip().strip('"')
    if not text:
        raise RuntimeError("?ㅽ뻾 ?대뜑媛 鍮꾩뼱 ?덉뒿?덈떎.")
    path = Path(text)
    if not path.is_absolute():
        path = config.root / path
    if not (path / "run_metadata.json").exists():
        raise RuntimeError(f"run_metadata.json??李얠쓣 ???놁뒿?덈떎: {path}")
    return path
