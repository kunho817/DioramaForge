from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .config import DepthSettings, FluxSettings, SamSettings, SdxlDepthLightningSettings
from .image_utils import demo_depth, demo_flux_style, demo_masks, depth_to_image, mask_bbox, normalize_depth
from .presets import get_preset
from .runtime import get_device, torch_dtype_from_name


def _local_snapshot_for_model(model_id: str) -> Path | None:
    if os.environ.get("HF_HUB_CACHE"):
        cache_root = Path(os.environ["HF_HUB_CACHE"])
    elif os.environ.get("HF_HOME"):
        cache_root = Path(os.environ["HF_HOME"]) / "hub"
    else:
        cache_root = Path("models") / "huggingface" / "hub"
    model_dir = cache_root / ("models--" + model_id.replace("/", "--"))
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return None
    return max(snapshots, key=lambda path: path.stat().st_mtime)


def _valid_diffusers_path(path_value: str) -> Path | None:
    return _valid_component_path(path_value, ("model_index.json",))


def _valid_component_path(path_value: str, markers: tuple[str, ...]) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_dir():
        return None
    if any((path / marker).exists() for marker in markers):
        return path
    return None


def _snapshot_has_weights(path: Path) -> bool:
    for pattern in ("*.safetensors", "*.bin"):
        if next(path.rglob(pattern), None) is not None:
            return True
    return False


def _resolve_diffusers_component(
    model_id: str,
    local_path: str,
    markers: tuple[str, ...],
) -> tuple[str, bool]:
    explicit_path = _valid_component_path(local_path, markers)
    local_snapshot = explicit_path or _local_snapshot_for_model(model_id)
    if local_snapshot is not None:
        has_marker = any((local_snapshot / marker).exists() for marker in markers)
        if not has_marker or not _snapshot_has_weights(local_snapshot):
            local_snapshot = None
    if local_snapshot is not None:
        return str(local_snapshot), True
    return model_id, False


def _resolve_lora_source(model_id: str, local_path: str, weight_name: str) -> tuple[str, str | None]:
    if local_path:
        path = Path(local_path)
        if path.is_file():
            return str(path), None
        if path.is_dir():
            if weight_name and (path / weight_name).exists():
                return str(path), weight_name
            safetensor = next(path.rglob("*.safetensors"), None)
            if safetensor is not None:
                return str(safetensor), None
            return str(path), weight_name or None

    local_snapshot = _local_snapshot_for_model(model_id)
    if local_snapshot is not None:
        if weight_name and (local_snapshot / weight_name).exists():
            return str(local_snapshot), weight_name
        safetensor = next(local_snapshot.rglob("*.safetensors"), None)
        if safetensor is not None:
            return str(safetensor), None

    return model_id, weight_name or None


def _preflight_flux_memory() -> None:
    try:
        import psutil
    except Exception:
        return

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    available_commit_gb = (memory.available + swap.free) / (1024**3)
    available_ram_gb = memory.available / (1024**3)
    current_pid = os.getpid()
    app_processes: list[str] = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            info = process.info
            if info["pid"] == current_pid:
                continue
            cmdline = " ".join(info.get("cmdline") or [])
            if "app.py" not in cmdline:
                continue
            working_set_gb = info["memory_info"].rss / (1024**3)
            if working_set_gb < 0.2:
                continue
            app_processes.append(f"pid={info['pid']} RAM={working_set_gb:.1f}GB")
        except Exception:
            continue

    # Full FLUX.1 Depth snapshots can briefly need very high CPU commit while Diffusers
    # reconstructs sharded weights. A low threshold lets Windows kill python.exe before
    # Python can raise a catchable exception, so fail early with a visible GUI message.
    if available_commit_gb < 50.0 or available_ram_gb < 16.0:
        process_note = ""
        if app_processes:
            process_note = " Other DioramaForge servers: " + ", ".join(app_processes) + "."
        raise RuntimeError(
            "FLUX 모델 로딩 전 메모리가 부족합니다. "
            f"available RAM={available_ram_gb:.1f}GB, available RAM+pagefile={available_commit_gb:.1f}GB."
            f"{process_note} 다른 DioramaForge/Gradio 서버를 종료하거나 Windows 페이징 파일을 늘린 뒤 다시 실행하세요."
        )


@dataclass
class DepthResult:
    depth: np.ndarray
    depth_image: Image.Image
    ray_map: np.ndarray | None
    backend: str
    metadata: dict[str, Any]


@dataclass
class SegmentationResult:
    masks: list[dict[str, Any]]
    backend: str
    metadata: dict[str, Any]


@dataclass
class FluxResult:
    image: Image.Image
    backend: str
    metadata: dict[str, Any]


class DepthEstimator:
    def __init__(self, settings: DepthSettings, allow_demo_fallback: bool) -> None:
        self.settings = settings
        self.allow_demo_fallback = allow_demo_fallback
        self._da3_model = None
        self._hf_pipe = None

    def estimate(self, image: Image.Image, image_path: Path, backend_mode: str) -> DepthResult:
        selected_backend = self._selected_backend(backend_mode)
        allow_fallback = self.allow_demo_fallback and selected_backend != "real"
        if selected_backend != "demo":
            try:
                return self._estimate_da3(image_path)
            except Exception as da3_exc:
                try:
                    return self._estimate_transformers(image)
                except Exception as hf_exc:
                    if not allow_fallback:
                        raise RuntimeError(f"Depth 추론 실패: DA3={da3_exc}; Transformers={hf_exc}") from hf_exc
                    return self._estimate_demo(image, f"DA3={da3_exc}; Transformers={hf_exc}")
        return self._estimate_demo(image, "demo backend selected")

    def _selected_backend(self, backend_mode: str) -> str:
        user_backend = backend_mode.lower()
        if user_backend != "auto":
            return user_backend
        return self.settings.backend.lower()

    def _estimate_da3(self, image_path: Path) -> DepthResult:
        from depth_anything_3.api import DepthAnything3

        if self._da3_model is None:
            self._da3_model = DepthAnything3.from_pretrained(self.settings.model_id).to(get_device())

        prediction = self._da3_model.inference([str(image_path)])
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        ray_map = None
        for attr in ("ray", "rays", "ray_map", "rays_map"):
            if hasattr(prediction, attr):
                candidate = getattr(prediction, attr)
                try:
                    ray_map = np.asarray(candidate[0], dtype=np.float32)
                except Exception:
                    ray_map = np.asarray(candidate, dtype=np.float32)
                break
        return DepthResult(
            depth=depth,
            depth_image=depth_to_image(depth),
            ray_map=ray_map,
            backend="Depth Anything 3",
            metadata={"model_id": self.settings.model_id, "ray_map_available": ray_map is not None},
        )

    def _estimate_transformers(self, image: Image.Image) -> DepthResult:
        from transformers import pipeline

        if self._hf_pipe is None:
            device = 0 if get_device() == "cuda" else -1
            self._hf_pipe = pipeline("depth-estimation", model=self.settings.fallback_model_id, device=device)

        output = self._hf_pipe(image)
        if "predicted_depth" in output:
            predicted = output["predicted_depth"]
            if hasattr(predicted, "detach"):
                depth = predicted.detach().cpu().float().numpy()
            else:
                depth = np.asarray(predicted, dtype=np.float32)
        elif "depth" in output:
            depth = np.asarray(ImageOps.grayscale(output["depth"]), dtype=np.float32)
        else:
            raise RuntimeError("Transformers depth pipeline returned no depth output.")

        return DepthResult(
            depth=depth,
            depth_image=depth_to_image(depth),
            ray_map=None,
            backend="Transformers depth-estimation",
            metadata={"model_id": self.settings.fallback_model_id, "ray_map_available": False},
        )

    def _estimate_demo(self, image: Image.Image, reason: str) -> DepthResult:
        depth = demo_depth(image)
        return DepthResult(
            depth=depth,
            depth_image=depth_to_image(depth),
            ray_map=None,
            backend="Demo depth",
            metadata={"reason": reason, "ray_map_available": False},
        )


class SamSegmenter:
    def __init__(self, settings: SamSettings, allow_demo_fallback: bool) -> None:
        self.settings = settings
        self.allow_demo_fallback = allow_demo_fallback
        self._sam_generator = None
        self._hf_pipe = None

    def segment(self, image: Image.Image, backend_mode: str) -> SegmentationResult:
        selected_backend = self._selected_backend(backend_mode)
        allow_fallback = self.allow_demo_fallback and selected_backend != "real"
        if selected_backend != "demo":
            try:
                return self._segment_sam2(image)
            except Exception as sam_exc:
                try:
                    return self._segment_transformers(image)
                except Exception as hf_exc:
                    if not allow_fallback:
                        raise RuntimeError(f"SAM 2 추론 실패: sam2={sam_exc}; Transformers={hf_exc}") from hf_exc
                    return self._segment_demo(image, f"sam2={sam_exc}; Transformers={hf_exc}")
        return self._segment_demo(image, "demo backend selected")

    def _selected_backend(self, backend_mode: str) -> str:
        user_backend = backend_mode.lower()
        if user_backend != "auto":
            return user_backend
        return self.settings.backend.lower()

    def _segment_sam2(self, image: Image.Image) -> SegmentationResult:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        if self._sam_generator is None:
            if hasattr(SAM2AutomaticMaskGenerator, "from_pretrained"):
                self._sam_generator = SAM2AutomaticMaskGenerator.from_pretrained(
                    self.settings.model_id,
                    points_per_batch=self.settings.points_per_batch,
                    output_mode="binary_mask",
                )
            else:
                raise RuntimeError("Installed sam2 package does not expose from_pretrained().")

        raw_masks = self._sam_generator.generate(np.asarray(image.convert("RGB")))
        masks = self._normalize_masks(raw_masks, "SAM 2")
        return SegmentationResult(
            masks=masks[: self.settings.max_masks],
            backend="SAM 2",
            metadata={"model_id": self.settings.model_id, "mask_count": len(masks)},
        )

    def _segment_transformers(self, image: Image.Image) -> SegmentationResult:
        from transformers import pipeline

        if self._hf_pipe is None:
            device = 0 if get_device() == "cuda" else -1
            self._hf_pipe = pipeline("mask-generation", model=self.settings.model_id, device=device)

        output = self._hf_pipe(image, points_per_batch=self.settings.points_per_batch)
        raw_masks = [{"segmentation": mask} for mask in output.get("masks", [])]
        masks = self._normalize_masks(raw_masks, "Transformers mask-generation")
        return SegmentationResult(
            masks=masks[: self.settings.max_masks],
            backend="Transformers mask-generation",
            metadata={"model_id": self.settings.model_id, "mask_count": len(masks)},
        )

    def _segment_demo(self, image: Image.Image, reason: str) -> SegmentationResult:
        masks = demo_masks(image, self.settings.max_masks)
        return SegmentationResult(
            masks=masks,
            backend="Demo masks",
            metadata={"reason": reason, "mask_count": len(masks)},
        )

    @staticmethod
    def _normalize_masks(raw_masks: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for raw in raw_masks:
            segmentation = raw.get("segmentation")
            if segmentation is None:
                continue
            if isinstance(segmentation, Image.Image):
                mask = np.asarray(ImageOps.grayscale(segmentation), dtype=np.uint8) > 0
            else:
                mask = np.asarray(segmentation).astype(bool)
            if mask.ndim != 2 or mask.sum() == 0:
                continue
            normalized.append(
                {
                    "segmentation": mask,
                    "area": int(raw.get("area", int(mask.sum()))),
                    "bbox": raw.get("bbox", mask_bbox(mask)),
                    "predicted_iou": raw.get("predicted_iou"),
                    "stability_score": raw.get("stability_score"),
                    "source": source,
                }
            )
        normalized.sort(key=lambda item: int(item["area"]), reverse=True)
        return normalized


class FluxStylizer:
    def __init__(self, settings: FluxSettings, allow_demo_fallback: bool) -> None:
        self.settings = settings
        self.allow_demo_fallback = allow_demo_fallback
        self._pipe = None

    def generate(
        self,
        image: Image.Image,
        depth_image: Image.Image,
        control_image: Image.Image | None,
        prompt: str,
        clip_prompt: str | None,
        negative_prompt: str | None,
        preset_name: str,
        seed: int,
        steps: int,
        guidance: float,
        strength: float,
        backend_mode: str,
    ) -> FluxResult:
        selected_backend = self._selected_backend(backend_mode)
        allow_fallback = self.allow_demo_fallback and selected_backend != "real"
        if selected_backend != "demo":
            try:
                return self._generate_flux(
                    image,
                    control_image or depth_image,
                    prompt,
                    clip_prompt,
                    negative_prompt,
                    seed,
                    steps,
                    guidance,
                    strength,
                )
            except Exception as exc:
                if not allow_fallback:
                    raise RuntimeError(f"FLUX 변환 실패: {exc}") from exc
                return self._generate_demo(image, control_image or depth_image, preset_name, strength, f"FLUX={exc}")
        return self._generate_demo(image, control_image or depth_image, preset_name, strength, "demo backend selected")

    def _selected_backend(self, backend_mode: str) -> str:
        user_backend = backend_mode.lower()
        if user_backend != "auto":
            return user_backend
        return self.settings.backend.lower()

    def _generate_flux(
        self,
        source_image: Image.Image,
        control_image: Image.Image,
        prompt: str,
        clip_prompt: str | None,
        negative_prompt: str | None,
        seed: int,
        steps: int,
        guidance: float,
        strength: float,
    ) -> FluxResult:
        import torch
        from diffusers import FluxControlImg2ImgPipeline

        if self._pipe is None:
            _preflight_flux_memory()
            started = time.perf_counter()
            dtype = torch_dtype_from_name(self.settings.torch_dtype)
            explicit_path = _valid_diffusers_path(self.settings.local_path)
            local_snapshot = explicit_path or _local_snapshot_for_model(self.settings.model_id)
            if local_snapshot is not None and (
                not (local_snapshot / "model_index.json").exists() or not _snapshot_has_weights(local_snapshot)
            ):
                local_snapshot = None
            model_source = str(local_snapshot) if local_snapshot else self.settings.model_id
            self._model_source = model_source
            pipeline_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "local_files_only": local_snapshot is not None,
            }
            if self.settings.quantization.lower() in {"fp8", "qfloat8"}:
                from diffusers.quantizers import PipelineQuantizationConfig
                from diffusers.quantizers.quantization_config import QuantoConfig

                pipeline_kwargs["quantization_config"] = PipelineQuantizationConfig(
                    quant_mapping={"transformer": QuantoConfig(weights_dtype="float8")}
                )

            self._pipe = FluxControlImg2ImgPipeline.from_pretrained(
                model_source,
                **pipeline_kwargs,
            )
            self._load_seconds = round(time.perf_counter() - started, 2)
            offload_strategy = self.settings.offload_strategy.lower()
            if self.settings.cpu_offload and offload_strategy == "sequential" and hasattr(
                self._pipe, "enable_sequential_cpu_offload"
            ):
                self._pipe.enable_sequential_cpu_offload()
            elif self.settings.cpu_offload and hasattr(self._pipe, "enable_model_cpu_offload"):
                self._pipe.enable_model_cpu_offload()
            else:
                self._pipe.to(get_device())
            if self.settings.enable_vae_tiling and hasattr(self._pipe, "vae"):
                if hasattr(self._pipe.vae, "enable_tiling"):
                    self._pipe.vae.enable_tiling()
                if hasattr(self._pipe.vae, "enable_slicing"):
                    self._pipe.vae.enable_slicing()
            post_dtype = self.settings.post_load_dtype.strip().lower()
            if post_dtype and post_dtype != "none":
                self._pipe.to(torch_dtype_from_name(post_dtype))

        width, height = control_image.size
        generator = torch.Generator(device="cpu").manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": clip_prompt or prompt,
            "image": source_image.convert("RGB"),
            "control_image": control_image.convert("RGB"),
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
        }
        signature = inspect.signature(self._pipe.__call__)
        if "prompt_2" in signature.parameters:
            kwargs["prompt_2"] = prompt
        if "strength" in signature.parameters:
            kwargs["strength"] = strength
        if "controlnet_conditioning_scale" in signature.parameters:
            kwargs["controlnet_conditioning_scale"] = strength
        negative_prompt_used = bool(negative_prompt and "negative_prompt" in signature.parameters)
        if negative_prompt_used:
            kwargs["negative_prompt"] = negative_prompt

        with torch.inference_mode():
            output = self._pipe(**kwargs)

        return FluxResult(
            image=output.images[0].convert("RGB"),
            backend="FLUX.1 Depth",
            metadata={
                "model_id": self.settings.model_id,
                "source": getattr(self, "_model_source", self.settings.model_id),
                "steps": steps,
                "guidance": guidance,
                "strength": strength,
                "pipeline_class": "FluxControlImg2ImgPipeline",
                "source_image_used": True,
                "control_image": "depth",
                "negative_prompt_used": negative_prompt_used,
                "prompt_2_used": "prompt_2" in signature.parameters,
                "quantization": self.settings.quantization,
                "offload_strategy": self.settings.offload_strategy,
                "post_load_dtype": self.settings.post_load_dtype,
                "load_seconds": getattr(self, "_load_seconds", None),
            },
        )

    def _generate_demo(
        self,
        image: Image.Image,
        depth_image: Image.Image,
        preset_name: str,
        strength: float,
        reason: str,
    ) -> FluxResult:
        preset = get_preset(preset_name)
        styled = demo_flux_style(image, depth_image, preset.palette, strength=strength)
        return FluxResult(
            image=styled,
            backend="Demo FLUX style",
            metadata={"reason": reason, "preset": preset_name, "strength": strength},
        )


class SdxlDepthLightningStylizer:
    def __init__(self, settings: SdxlDepthLightningSettings, allow_demo_fallback: bool) -> None:
        self.settings = settings
        self.allow_demo_fallback = allow_demo_fallback
        self._pipe = None

    def generate(
        self,
        image: Image.Image,
        depth_image: Image.Image,
        control_image: Image.Image | None,
        prompt: str,
        clip_prompt: str | None,
        negative_prompt: str | None,
        preset_name: str,
        seed: int,
        steps: int,
        guidance: float,
        strength: float,
        backend_mode: str,
    ) -> FluxResult:
        selected_backend = self._selected_backend(backend_mode)
        allow_fallback = self.allow_demo_fallback and selected_backend != "real"
        if selected_backend != "demo":
            try:
                return self._generate_sdxl(
                    image,
                    control_image or depth_image,
                    prompt,
                    negative_prompt,
                    seed,
                    steps,
                    guidance,
                    strength,
                )
            except Exception as exc:
                if not allow_fallback:
                    raise RuntimeError(f"SDXL depth-lightning 변환 실패: {exc}") from exc
                return self._generate_demo(image, control_image or depth_image, preset_name, strength, f"SDXL={exc}")
        return self._generate_demo(image, control_image or depth_image, preset_name, strength, "demo backend selected")

    def _selected_backend(self, backend_mode: str) -> str:
        user_backend = backend_mode.lower()
        if user_backend != "auto":
            return user_backend
        return self.settings.backend.lower()

    def _generate_sdxl(
        self,
        source_image: Image.Image,
        control_image: Image.Image,
        prompt: str,
        negative_prompt: str | None,
        seed: int,
        steps: int,
        guidance: float,
        strength: float,
    ) -> FluxResult:
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

        if self._pipe is None:
            started = time.perf_counter()
            dtype = torch_dtype_from_name(self.settings.torch_dtype)
            controlnet_source, controlnet_is_local = _resolve_diffusers_component(
                self.settings.controlnet_model_id,
                self.settings.controlnet_local_path,
                ("config.json",),
            )
            controlnet_kwargs: dict[str, Any] = {"torch_dtype": dtype}
            if controlnet_is_local:
                controlnet_kwargs["local_files_only"] = True
            if self.settings.variant:
                controlnet_kwargs["variant"] = self.settings.variant
            try:
                controlnet = ControlNetModel.from_pretrained(controlnet_source, **controlnet_kwargs)
            except TypeError:
                controlnet_kwargs.pop("variant", None)
                controlnet = ControlNetModel.from_pretrained(controlnet_source, **controlnet_kwargs)

            base_source, base_is_local = _resolve_diffusers_component(
                self.settings.base_model_id,
                self.settings.base_local_path,
                ("model_index.json",),
            )
            pipeline_kwargs: dict[str, Any] = {
                "controlnet": controlnet,
                "torch_dtype": dtype,
            }
            if base_is_local:
                pipeline_kwargs["local_files_only"] = True
            if self.settings.variant:
                pipeline_kwargs["variant"] = self.settings.variant
            try:
                self._pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    base_source,
                    **pipeline_kwargs,
                )
            except TypeError:
                pipeline_kwargs.pop("variant", None)
                self._pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    base_source,
                    **pipeline_kwargs,
                )

            self._configure_scheduler()
            self._load_lightning_lora()
            self._model_source = base_source
            self._controlnet_source = controlnet_source
            self._load_seconds = round(time.perf_counter() - started, 2)

            if self.settings.cpu_offload and hasattr(self._pipe, "enable_model_cpu_offload"):
                self._pipe.enable_model_cpu_offload()
            else:
                self._pipe.to(get_device())
            if self.settings.enable_vae_tiling and hasattr(self._pipe, "vae"):
                if hasattr(self._pipe.vae, "enable_tiling"):
                    self._pipe.vae.enable_tiling()
                if hasattr(self._pipe.vae, "enable_slicing"):
                    self._pipe.vae.enable_slicing()

        width, height = control_image.size
        generator = torch.Generator(device="cpu").manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": source_image.convert("RGB"),
            "control_image": control_image.convert("RGB"),
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
            "strength": strength,
            "controlnet_conditioning_scale": self.settings.controlnet_conditioning_scale,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if self.settings.lora_scale != 1.0 and not getattr(self, "_lora_fused", False):
            kwargs["cross_attention_kwargs"] = {"scale": self.settings.lora_scale}

        with torch.inference_mode():
            output = self._pipe(**kwargs)

        return FluxResult(
            image=output.images[0].convert("RGB"),
            backend="SDXL Depth Lightning",
            metadata={
                "base_model_id": self.settings.base_model_id,
                "controlnet_model_id": self.settings.controlnet_model_id,
                "lora_model_id": self.settings.lora_model_id,
                "lora_weight_name": self.settings.lora_weight_name,
                "lora_source": getattr(self, "_lora_source", self.settings.lora_model_id),
                "source": getattr(self, "_model_source", self.settings.base_model_id),
                "controlnet_source": getattr(self, "_controlnet_source", self.settings.controlnet_model_id),
                "steps": steps,
                "guidance": guidance,
                "strength": strength,
                "controlnet_conditioning_scale": self.settings.controlnet_conditioning_scale,
                "lora_scale": self.settings.lora_scale,
                "pipeline_class": "StableDiffusionXLControlNetImg2ImgPipeline",
                "source_image_used": True,
                "control_image": "depth",
                "scheduler": self.settings.scheduler,
                "variant": self.settings.variant,
                "load_seconds": getattr(self, "_load_seconds", None),
            },
        )

    def _configure_scheduler(self) -> None:
        scheduler_name = self.settings.scheduler.strip().lower()
        if scheduler_name in {"", "default"}:
            return
        if scheduler_name == "euler":
            from diffusers import EulerDiscreteScheduler

            self._pipe.scheduler = EulerDiscreteScheduler.from_config(
                self._pipe.scheduler.config,
                timestep_spacing="trailing",
            )
            return
        raise RuntimeError(f"Unsupported SDXL scheduler: {self.settings.scheduler}")

    def _load_lightning_lora(self) -> None:
        lora_source, weight_name = _resolve_lora_source(
            self.settings.lora_model_id,
            self.settings.lora_local_path,
            self.settings.lora_weight_name,
        )
        if weight_name:
            self._pipe.load_lora_weights(lora_source, weight_name=weight_name)
        else:
            self._pipe.load_lora_weights(lora_source)
        self._lora_source = lora_source
        self._lora_fused = False
        if hasattr(self._pipe, "fuse_lora"):
            signature = inspect.signature(self._pipe.fuse_lora)
            if "lora_scale" in signature.parameters:
                self._pipe.fuse_lora(lora_scale=self.settings.lora_scale)
            else:
                self._pipe.fuse_lora()
            self._lora_fused = True

    def _generate_demo(
        self,
        image: Image.Image,
        depth_image: Image.Image,
        preset_name: str,
        strength: float,
        reason: str,
    ) -> FluxResult:
        preset = get_preset(preset_name)
        styled = demo_flux_style(image, depth_image, preset.palette, strength=strength)
        return FluxResult(
            image=styled,
            backend="Demo SDXL depth-lightning style",
            metadata={"reason": reason, "preset": preset_name, "strength": strength},
        )
