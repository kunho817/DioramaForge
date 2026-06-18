from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppSettings:
    output_dir: Path
    allow_demo_fallback: bool = True
    allow_local_heavy_models: bool = False


@dataclass(frozen=True)
class ComfySettings:
    base_url: str
    enabled: bool
    input_subfolder: str
    stage3_workflow: Path
    stage35_workflow: Path
    refine_workflow: Path
    output_node_id: str
    timeout_seconds: int
    poll_interval_seconds: float


@dataclass(frozen=True)
class RemoteBackendSettings:
    base_url: str
    enabled: bool
    api_key_env: str
    work_dir: Path
    cleanup_after_response: bool
    timeout_seconds: int
    execution_backend: str
    stage35_backend: str
    stage4_backend: str


@dataclass(frozen=True)
class MeshySettings:
    base_url: str
    enabled: bool
    api_key_env: str
    timeout_seconds: int
    poll_interval_seconds: float
    model_type: str
    ai_model: str
    should_texture: bool
    enable_pbr: bool
    should_remesh: bool
    target_polycount: int
    target_formats: tuple[str, ...]
    image_enhancement: bool
    remove_lighting: bool
    moderation: bool
    download_outputs: bool


@dataclass(frozen=True)
class ProductPipelineSettings:
    seed: int
    steps: int
    guidance: float
    strength: float
    max_resolution: int
    demo_time_budget_seconds: int
    demo_min_free_vram_gb: float
    stage35_backend_mode: str
    stage4_backend_mode: str
    stage5_backend_mode: str
    stage35_upscale_scale: float
    stage35_refinement_strength: float
    stage35_max_side: int
    stage4_mesh_resolution: int
    stage4_max_parts: int
    stage5_width_mm: float
    stage5_relief_height_mm: float
    stage5_base_thickness_mm: float
    stage5_mesh_resolution: int


@dataclass(frozen=True)
class StyleEngineSettings:
    active: str
    target: str
    backend_mode: str
    result_label: str
    control_label: str
    show_backend_selector: bool
    legacy_remote_visible: bool


@dataclass(frozen=True)
class DepthSettings:
    backend: str
    model_id: str
    fallback_model_id: str


@dataclass(frozen=True)
class SamSettings:
    backend: str
    model_id: str
    points_per_batch: int
    max_masks: int


@dataclass(frozen=True)
class FluxSettings:
    backend: str
    model_id: str
    local_path: str
    torch_dtype: str
    quantization: str
    offload_strategy: str
    post_load_dtype: str
    cpu_offload: bool
    enable_vae_tiling: bool


@dataclass(frozen=True)
class SdxlDepthLightningSettings:
    backend: str
    base_model_id: str
    controlnet_model_id: str
    lora_model_id: str
    lora_weight_name: str
    base_local_path: str
    controlnet_local_path: str
    lora_local_path: str
    torch_dtype: str
    variant: str
    scheduler: str
    controlnet_conditioning_scale: float
    lora_scale: float
    cpu_offload: bool
    enable_vae_tiling: bool


@dataclass(frozen=True)
class MeshModelSettings:
    backend: str
    model_id: str
    local_path: str


@dataclass(frozen=True)
class PrintSettings:
    default_width_mm: float
    default_relief_height_mm: float
    default_base_thickness_mm: float


@dataclass(frozen=True)
class DioramaConfig:
    root: Path
    app: AppSettings
    comfy: ComfySettings
    remote: RemoteBackendSettings
    meshy: MeshySettings
    product_pipeline: ProductPipelineSettings
    style_engine: StyleEngineSettings
    depth: DepthSettings
    sam: SamSettings
    flux: FluxSettings
    sdxl_depth_lightning: SdxlDepthLightningSettings
    trellis: MeshModelSettings
    ultrashape: MeshModelSettings
    print: PrintSettings


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_config(path: str | Path | None = None) -> DioramaConfig:
    root = Path(__file__).resolve().parents[2]
    _load_env_local(root / ".env.local")
    os.environ.setdefault("HF_HOME", str(root / "models" / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(root / "models" / "huggingface" / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    config_path = Path(path) if path else root / "configs" / "default.json"
    if not config_path.is_absolute():
        config_path = root / config_path

    data = _read_json(config_path)
    app_data = data.get("app", {})
    comfy_data = data.get("comfyui", {})
    remote_data = data.get("remote_backend", {})
    meshy_data = data.get("meshy_ai", {})
    product_pipeline_data = data.get("product_pipeline", {})
    style_engine_data = data.get("style_engine", {})
    model_data = data.get("models", {})
    depth_data = model_data.get("depth", {})
    sam_data = model_data.get("sam", {})
    flux_data = model_data.get("flux", {})
    sdxl_data = model_data.get("sdxl_depth_lightning", {})
    trellis_data = model_data.get("trellis", {})
    ultrashape_data = model_data.get("ultrashape", {})
    print_data = model_data.get("print", {})

    output_dir = Path(app_data.get("output_dir", "outputs/runs"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    stage3_workflow = Path(comfy_data.get("stage3_workflow", "workflows/comfy/stage3_style_api.json"))
    stage35_workflow = Path(
        comfy_data.get("stage35_workflow", "workflows/comfy/stage35_upscale_reconstruction_api.json")
    )
    refine_workflow = Path(comfy_data.get("refine_workflow", "workflows/comfy/stage35_refine_api.json"))
    if not stage3_workflow.is_absolute():
        stage3_workflow = root / stage3_workflow
    if not stage35_workflow.is_absolute():
        stage35_workflow = root / stage35_workflow
    if not refine_workflow.is_absolute():
        refine_workflow = root / refine_workflow
    remote_work_dir = Path(
        os.environ.get(
            "DIORAMA_REMOTE_WORKDIR",
            str(remote_data.get("work_dir", "outputs/remote_backend/work")),
        )
    )
    if not remote_work_dir.is_absolute():
        remote_work_dir = root / remote_work_dir

    return DioramaConfig(
        root=root,
        app=AppSettings(
            output_dir=output_dir,
            allow_demo_fallback=bool(app_data.get("allow_demo_fallback", True)),
            allow_local_heavy_models=_env_bool(
                "DIORAMA_ALLOW_LOCAL_HEAVY_MODELS",
                bool(app_data.get("allow_local_heavy_models", False)),
            ),
        ),
        comfy=ComfySettings(
            base_url=str(comfy_data.get("base_url", "http://127.0.0.1:8188")).rstrip("/"),
            enabled=bool(comfy_data.get("enabled", True)),
            input_subfolder=str(comfy_data.get("input_subfolder", "diorama_forge")),
            stage3_workflow=stage3_workflow,
            stage35_workflow=stage35_workflow,
            refine_workflow=refine_workflow,
            output_node_id=str(comfy_data.get("output_node_id", "")),
            timeout_seconds=int(comfy_data.get("timeout_seconds", 1800)),
            poll_interval_seconds=float(comfy_data.get("poll_interval_seconds", 1.0)),
        ),
        remote=RemoteBackendSettings(
            base_url=str(remote_data.get("base_url", "http://127.0.0.1:9008")).rstrip("/"),
            enabled=bool(remote_data.get("enabled", False)),
            api_key_env=str(remote_data.get("api_key_env", "DIORAMA_REMOTE_API_KEY")),
            work_dir=remote_work_dir,
            cleanup_after_response=bool(remote_data.get("cleanup_after_response", True)),
            timeout_seconds=int(remote_data.get("timeout_seconds", 3600)),
            execution_backend=str(remote_data.get("execution_backend", "real")),
            stage35_backend=str(remote_data.get("stage35_backend", "auto")),
            stage4_backend=str(remote_data.get("stage4_backend", "demo")),
        ),
        meshy=MeshySettings(
            base_url=str(meshy_data.get("base_url", "https://api.meshy.ai")).rstrip("/"),
            enabled=bool(meshy_data.get("enabled", True)),
            api_key_env=str(meshy_data.get("api_key_env", "MESHY_API_KEY")),
            timeout_seconds=int(meshy_data.get("timeout_seconds", 900)),
            poll_interval_seconds=float(meshy_data.get("poll_interval_seconds", 5.0)),
            model_type=str(meshy_data.get("model_type", "standard")),
            ai_model=str(meshy_data.get("ai_model", "latest")),
            should_texture=bool(meshy_data.get("should_texture", True)),
            enable_pbr=bool(meshy_data.get("enable_pbr", False)),
            should_remesh=bool(meshy_data.get("should_remesh", True)),
            target_polycount=int(meshy_data.get("target_polycount", 30000)),
            target_formats=tuple(str(item).lower() for item in meshy_data.get("target_formats", ["glb", "obj", "stl"])),
            image_enhancement=bool(meshy_data.get("image_enhancement", True)),
            remove_lighting=bool(meshy_data.get("remove_lighting", True)),
            moderation=bool(meshy_data.get("moderation", False)),
            download_outputs=bool(meshy_data.get("download_outputs", True)),
        ),
        product_pipeline=ProductPipelineSettings(
            seed=int(product_pipeline_data.get("seed", -1)),
            steps=int(product_pipeline_data.get("steps", 4)),
            guidance=float(product_pipeline_data.get("guidance", 3.5)),
            strength=float(product_pipeline_data.get("strength", 0.55)),
            max_resolution=int(product_pipeline_data.get("max_resolution", 512)),
            demo_time_budget_seconds=int(product_pipeline_data.get("demo_time_budget_seconds", 240)),
            demo_min_free_vram_gb=float(product_pipeline_data.get("demo_min_free_vram_gb", 6.0)),
            stage35_backend_mode=str(product_pipeline_data.get("stage35_backend_mode", "demo")),
            stage4_backend_mode=str(product_pipeline_data.get("stage4_backend_mode", "demo")),
            stage5_backend_mode=str(product_pipeline_data.get("stage5_backend_mode", "demo")),
            stage35_upscale_scale=float(product_pipeline_data.get("stage35_upscale_scale", 2.0)),
            stage35_refinement_strength=float(product_pipeline_data.get("stage35_refinement_strength", 0.22)),
            stage35_max_side=int(product_pipeline_data.get("stage35_max_side", 1536)),
            stage4_mesh_resolution=int(product_pipeline_data.get("stage4_mesh_resolution", 96)),
            stage4_max_parts=int(product_pipeline_data.get("stage4_max_parts", 12)),
            stage5_width_mm=float(product_pipeline_data.get("stage5_width_mm", print_data.get("default_width_mm", 120.0))),
            stage5_relief_height_mm=float(
                product_pipeline_data.get("stage5_relief_height_mm", print_data.get("default_relief_height_mm", 18.0))
            ),
            stage5_base_thickness_mm=float(
                product_pipeline_data.get("stage5_base_thickness_mm", print_data.get("default_base_thickness_mm", 3.0))
            ),
            stage5_mesh_resolution=int(product_pipeline_data.get("stage5_mesh_resolution", 96)),
        ),
        style_engine=StyleEngineSettings(
            active=str(style_engine_data.get("active", "auto")),
            target=str(style_engine_data.get("target", "comfyui_stage3_style")),
            backend_mode=str(style_engine_data.get("backend_mode", "auto")),
            result_label=str(style_engine_data.get("result_label", "Style Result")),
            control_label=str(style_engine_data.get("control_label", "Style Control")),
            show_backend_selector=bool(style_engine_data.get("show_backend_selector", False)),
            legacy_remote_visible=bool(style_engine_data.get("legacy_remote_visible", False)),
        ),
        depth=DepthSettings(
            backend=str(depth_data.get("backend", "auto")),
            model_id=str(depth_data.get("model_id", "depth-anything/DA3NESTED-GIANT-LARGE")),
            fallback_model_id=str(
                depth_data.get("fallback_model_id", "depth-anything/Depth-Anything-V2-Small-hf")
            ),
        ),
        sam=SamSettings(
            backend=str(sam_data.get("backend", "auto")),
            model_id=str(sam_data.get("model_id", "facebook/sam2-hiera-base-plus")),
            points_per_batch=int(sam_data.get("points_per_batch", 64)),
            max_masks=int(sam_data.get("max_masks", 12)),
        ),
        flux=FluxSettings(
            backend=str(flux_data.get("backend", "auto")),
            model_id=str(flux_data.get("model_id", "black-forest-labs/FLUX.1-Depth-dev")),
            local_path=str(flux_data.get("local_path", "")),
            torch_dtype=str(flux_data.get("torch_dtype", "bfloat16")),
            quantization=str(flux_data.get("quantization", "none")),
            offload_strategy=str(flux_data.get("offload_strategy", "sequential")),
            post_load_dtype=str(flux_data.get("post_load_dtype", "float16")),
            cpu_offload=bool(flux_data.get("cpu_offload", True)),
            enable_vae_tiling=bool(flux_data.get("enable_vae_tiling", True)),
        ),
        sdxl_depth_lightning=SdxlDepthLightningSettings(
            backend=str(sdxl_data.get("backend", "auto")),
            base_model_id=str(sdxl_data.get("base_model_id", "stabilityai/stable-diffusion-xl-base-1.0")),
            controlnet_model_id=str(sdxl_data.get("controlnet_model_id", "diffusers/controlnet-depth-sdxl-1.0")),
            lora_model_id=str(sdxl_data.get("lora_model_id", "ByteDance/SDXL-Lightning")),
            lora_weight_name=str(sdxl_data.get("lora_weight_name", "sdxl_lightning_4step_lora.safetensors")),
            base_local_path=str(sdxl_data.get("base_local_path", "")),
            controlnet_local_path=str(sdxl_data.get("controlnet_local_path", "")),
            lora_local_path=str(sdxl_data.get("lora_local_path", "")),
            torch_dtype=str(sdxl_data.get("torch_dtype", "float16")),
            variant=str(sdxl_data.get("variant", "fp16")),
            scheduler=str(sdxl_data.get("scheduler", "euler")),
            controlnet_conditioning_scale=float(sdxl_data.get("controlnet_conditioning_scale", 0.85)),
            lora_scale=float(sdxl_data.get("lora_scale", 1.0)),
            cpu_offload=bool(sdxl_data.get("cpu_offload", False)),
            enable_vae_tiling=bool(sdxl_data.get("enable_vae_tiling", True)),
        ),
        trellis=MeshModelSettings(
            backend=str(trellis_data.get("backend", "auto")),
            model_id=str(trellis_data.get("model_id", "microsoft/TRELLIS-image-large")),
            local_path=str(trellis_data.get("local_path", "")),
        ),
        ultrashape=MeshModelSettings(
            backend=str(ultrashape_data.get("backend", "auto")),
            model_id=str(ultrashape_data.get("model_id", "PKU-YuanGroup/UltraShape-1.0")),
            local_path=str(ultrashape_data.get("local_path", "")),
        ),
        print=PrintSettings(
            default_width_mm=float(print_data.get("default_width_mm", 120.0)),
            default_relief_height_mm=float(print_data.get("default_relief_height_mm", 18.0)),
            default_base_thickness_mm=float(print_data.get("default_base_thickness_mm", 3.0)),
        ),
    )


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_env_local(path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not _valid_env_name(name) or name in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[name] = value
        loaded[name] = value
    return loaded


def _valid_env_name(name: str) -> bool:
    if not name or name[0].isdigit():
        return False
    return all(char == "_" or char.isalnum() for char in name)
