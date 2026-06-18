from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .adapters import DepthEstimator, FluxStylizer, SamSegmenter, SdxlDepthLightningStylizer
from .comfy import ComfyUIClient
from .config import DioramaConfig
from .image_utils import (
    ensure_rgb,
    flux_control_from_depth_and_masks,
    overlay_masks,
    resize_for_generation,
    save_json,
    save_masks,
    segmentation_layout_summary,
)
from .prompting import build_stage3_prompt_bundle
from .regions import build_region_plan, save_region_artifacts
from .style_engine import resolve_style_engine


StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class PipelineOptions:
    preset_name: str
    custom_prompt: str
    seed: int
    steps: int
    guidance: float
    strength: float
    max_resolution: int
    backend_mode: str


@dataclass(frozen=True)
class PipelineArtifacts:
    run_dir: Path
    input_image: Image.Image
    depth_image: Image.Image
    mask_overlay: Image.Image
    region_overlay: Image.Image
    control_image: Image.Image
    flux_image: Image.Image
    final_image_path: Path
    metadata_path: Path
    log: list[str]


class DioramaPipeline:
    def __init__(self, config: DioramaConfig) -> None:
        self.config = config
        self.depth = DepthEstimator(config.depth, config.app.allow_demo_fallback)
        self.segmenter = SamSegmenter(config.sam, config.app.allow_demo_fallback)
        self.flux = FluxStylizer(config.flux, config.app.allow_demo_fallback)
        self.sdxl_style = SdxlDepthLightningStylizer(
            config.sdxl_depth_lightning,
            config.app.allow_demo_fallback,
        )
        self.comfy = ComfyUIClient(config.comfy)

    def run(
        self,
        image: Image.Image,
        options: PipelineOptions,
        status: StatusCallback | None = None,
        run_dir: Path | None = None,
    ) -> PipelineArtifacts:
        _enforce_pipeline_execution_policy(self.config, options.backend_mode)
        logs: list[str] = []

        def emit(message: str) -> None:
            logs.append(message)
            if status:
                status(message)

        run_dir = run_dir or self._new_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = run_dir / "masks"
        region_dir = run_dir / "regions"
        emit("Run folder prepared")

        working_image = resize_for_generation(ensure_rgb(image), options.max_resolution)
        input_path = run_dir / "input.png"
        working_image.save(input_path)
        emit("Input image prepared")

        emit("Estimating depth")
        depth_result = self.depth.estimate(working_image, input_path, options.backend_mode)
        depth_image = depth_result.depth_image
        if depth_image.size != working_image.size:
            depth_image = depth_image.resize(working_image.size, Image.Resampling.BICUBIC)
        depth_path = run_dir / "depth.png"
        depth_npy_path = run_dir / "depth.npy"
        depth_image.save(depth_path)
        np.save(depth_npy_path, depth_result.depth)
        ray_path: str | None = None
        if depth_result.ray_map is not None:
            ray_npy_path = run_dir / "ray_map.npy"
            np.save(ray_npy_path, depth_result.ray_map)
            ray_path = str(ray_npy_path)
        emit(f"Depth estimation complete: {depth_result.backend}")

        emit("Generating segmentation masks")
        segmentation = self.segmenter.segment(working_image, options.backend_mode)
        mask_metadata = save_masks(mask_dir, segmentation.masks)
        mask_overlay = overlay_masks(working_image, segmentation.masks)
        mask_overlay_path = run_dir / "mask_overlay.png"
        mask_overlay.save(mask_overlay_path)
        emit(f"Segmentation complete: {segmentation.backend}")

        flux_control_image = depth_image.copy()
        flux_control_path = run_dir / "flux_control.png"
        flux_control_image.save(flux_control_path)
        sam_structure_hint = flux_control_from_depth_and_masks(depth_image, segmentation.masks)
        sam_structure_hint_path = run_dir / "sam_structure_hint.png"
        sam_structure_hint.save(sam_structure_hint_path)
        segmentation_prompt, segmentation_regions = segmentation_layout_summary(
            segmentation.masks,
            working_image.size,
        )
        region_plan = build_region_plan(
            image=working_image,
            depth_image=depth_image,
            masks=segmentation.masks,
            preset_name=options.preset_name,
        )
        region_artifacts = save_region_artifacts(region_dir, working_image, region_plan)
        region_overlay = Image.open(region_artifacts["overlay"]).convert("RGB")
        region_prompt = str(region_plan.get("region_prompt", ""))

        prompt_bundle = build_stage3_prompt_bundle(
            preset_name=options.preset_name,
            custom_prompt=options.custom_prompt,
            segmentation_prompt=segmentation_prompt,
            region_plan=region_plan,
            text_encoder_profile=_stage3_text_encoder_profile(self.config, options.backend_mode),
        )
        base_prompt = prompt_bundle.base_prompt
        clip_prompt = prompt_bundle.clip_prompt
        prompt = prompt_bundle.positive_prompt
        negative_prompt = prompt_bundle.negative_prompt
        seed = options.seed if options.seed >= 0 else int(datetime.now().timestamp()) % 2_147_483_647
        if options.backend_mode.lower() == "comfyui":
            emit("Running ComfyUI Stage 3 style workflow")
            style_adapter = "comfyui_stage3"
            style_model_id = self.config.comfy.base_url
            flux_result = self.comfy.generate_stage3(
                run_dir=run_dir,
                source_image=working_image,
                depth_image=depth_image,
                control_image=flux_control_image,
                prompt=prompt,
                clip_prompt=clip_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                steps=options.steps,
                guidance=options.guidance,
                strength=options.strength,
            )
        else:
            emit("Running local Stage 3 style engine")
            stylizer, style_adapter, style_model_id = self._style_stylizer()
            flux_result = stylizer.generate(
                image=working_image,
                depth_image=depth_image,
                control_image=flux_control_image,
                prompt=prompt,
                clip_prompt=clip_prompt,
                negative_prompt=negative_prompt,
                preset_name=options.preset_name,
                seed=seed,
                steps=options.steps,
                guidance=options.guidance,
                strength=options.strength,
                backend_mode=options.backend_mode,
            )
        final_path = run_dir / "flux_result.png"
        flux_result.image.save(final_path)
        emit(f"Stage 3 style generation complete: {flux_result.backend}")

        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "options": {
                "preset_name": options.preset_name,
                "custom_prompt": options.custom_prompt,
                "base_prompt": base_prompt,
                "clip_prompt": clip_prompt,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "meshy_texture_prompt": prompt_bundle.meshy_texture_prompt,
                "prompt_strategy": prompt_bundle.strategy,
                "prompt_notes": list(prompt_bundle.notes),
                "text_encoder_profile": _stage3_text_encoder_profile(self.config, options.backend_mode),
                "structure_prompt": segmentation_prompt,
                "region_prompt": region_prompt,
                "seed": seed,
                "steps": options.steps,
                "guidance": options.guidance,
                "strength": options.strength,
                "transform_strength": options.strength,
                "max_resolution": options.max_resolution,
                "backend_mode": options.backend_mode,
                "control_strategy": "source_img2img_depth_control_region_plan_prompt",
                "stage3_backend": style_adapter,
            },
            "style_engine": {
                "active": self.config.style_engine.active,
                "resolved": style_adapter,
                "target": self.config.style_engine.target,
                "backend_mode": options.backend_mode,
                "current_model_id": style_model_id,
                "adapter": style_adapter,
            },
            "artifacts": {
                "input": str(input_path),
                "depth_png": str(depth_path),
                "depth_npy": str(depth_npy_path),
                "ray_map_npy": ray_path,
                "mask_overlay": str(mask_overlay_path),
                "region_overlay": region_artifacts["overlay"],
                "region_manifest": region_artifacts["manifest"],
                "region_masks": region_artifacts["masks"],
                "flux_control": str(flux_control_path),
                "sam_structure_hint": str(sam_structure_hint_path),
                "final_image": str(final_path),
            },
            "structure_control": {
                "strategy": "source_img2img_depth_control_region_plan_prompt",
                "style_control_note": (
                    "The style engine receives the original image plus pure depth control. SAM masks are "
                    "recorded and summarized in prompt text; they are not a direct per-region generation control yet."
                ),
                "flux_control_note": "Legacy compatibility key. See style_control_note.",
                "regions": segmentation_regions,
                "semantic_region_plan": {
                    key: value for key, value in region_plan.items() if not key.startswith("_")
                },
            },
            "models": {
                "depth": {"backend": depth_result.backend, **depth_result.metadata},
                "segmentation": {"backend": segmentation.backend, **segmentation.metadata},
                "style": {"backend": flux_result.backend, **flux_result.metadata},
                "flux": {"backend": flux_result.backend, **flux_result.metadata},
            },
            "masks": mask_metadata,
            "log": logs,
        }
        metadata_path = run_dir / "run_metadata.json"
        save_json(metadata_path, metadata)
        emit("Complete")

        return PipelineArtifacts(
            run_dir=run_dir,
            input_image=working_image,
            depth_image=depth_image,
            mask_overlay=mask_overlay,
            region_overlay=region_overlay,
            control_image=flux_control_image,
            flux_image=flux_result.image,
            final_image_path=final_path,
            metadata_path=metadata_path,
            log=logs,
        )

    def _new_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.config.app.output_dir / timestamp
        candidate = base
        suffix = 1
        while candidate.exists():
            candidate = Path(f"{base}_{suffix:02d}")
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def _style_stylizer(self):
        active = resolve_style_engine(self.config)
        if active == "sdxl_depth_lightning":
            return (
                self.sdxl_style,
                "sdxl_depth_lightning",
                self.config.sdxl_depth_lightning.base_model_id,
            )
        return self.flux, "flux_depth_compat", self.config.flux.model_id


def _enforce_pipeline_execution_policy(config: DioramaConfig, backend_mode: str) -> None:
    normalized = str(backend_mode or "auto").strip().lower()
    if normalized == "remote":
        raise RuntimeError(
            "Remote backend execution is only supported through the API RemoteModelClient path. "
            "Do not pass remote backend directly to DioramaPipeline.run()."
        )
    if normalized in {"demo", "comfyui"} or config.app.allow_local_heavy_models:
        return
    raise RuntimeError(
        "Local heavy model execution is disabled. "
        "Enable allow_local_heavy_models explicitly, use ComfyUI, or use the proxy backend for structure checks."
    )


def _stage3_text_encoder_profile(config: DioramaConfig, backend_mode: str) -> str:
    if str(backend_mode or "").strip().lower() != "comfyui":
        return "flux_natural"
    workflow = _read_workflow_json(config.comfy.stage3_workflow)
    text = json.dumps(workflow, ensure_ascii=False).lower()
    if any(marker in text for marker in ("illustrious", "wai", "noobai", "pony", "anime\\", "janku")):
        return "illustrious_sdxl"
    if any(marker in text for marker in ("sd_xl_base", "stable-diffusion-xl", "sdxl", "checkpointloadersimple")):
        return "sdxl_base"
    if any(marker in text for marker in ("flux", "t5", "dualcliploader")):
        return "flux_natural"
    return "comfyui_sdxl"


def _read_workflow_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
