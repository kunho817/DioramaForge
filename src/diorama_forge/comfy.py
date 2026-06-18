from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from PIL import Image

from .comfy_workflow import install_comfy_workflow_bytes, placeholder_names, stage_key, validate_comfy_workflow
from .comfy_workflow_models import workflow_model_fields
from .config import ComfySettings
from .prompting import build_stage35_prompt_bundle


@dataclass(frozen=True)
class ComfyGenerationResult:
    image: Image.Image
    backend: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ComfyStage35Result:
    visual_image: Image.Image
    reconstruction_image: Image.Image
    refined_image: Image.Image | None
    backend: str
    metadata: dict[str, Any]


class ComfyUIClient:
    def __init__(self, settings: ComfySettings) -> None:
        self.settings = settings

    def status(self) -> dict[str, Any]:
        if not self.settings.enabled:
            return {
                "ok": False,
                "base_url": self.settings.base_url,
                "error": "ComfyUI backend is disabled in configs/default.json.",
            }
        try:
            system_stats = self._get_json("/system_stats", timeout=3)
            queue = self._get_json("/queue", timeout=3)
            node_compatibility = self.node_compatibility()
            return {
                "ok": True,
                "base_url": self.settings.base_url,
                "system_stats": system_stats,
                "queue": queue,
                "workflows": self.workflow_status(),
                "node_compatibility": node_compatibility,
            }
        except Exception as exc:
            return {
                "ok": False,
                "base_url": self.settings.base_url,
                "error": str(exc),
                "workflows": self.workflow_status(),
            }

    def workflow_status(self) -> dict[str, Any]:
        return {
            "stage3": {
                "path": str(self.settings.stage3_workflow),
                "exists": self.settings.stage3_workflow.exists(),
                "model_fields": _workflow_model_field_dicts(self.settings.stage3_workflow),
                "validation": validate_comfy_workflow(
                    self.settings.stage3_workflow,
                    "stage3",
                    output_node_id=self.settings.output_node_id,
                ),
            },
            "stage35": {
                "path": str(self.settings.stage35_workflow),
                "exists": self.settings.stage35_workflow.exists(),
                "model_fields": _workflow_model_field_dicts(self.settings.stage35_workflow),
                "validation": validate_comfy_workflow(
                    self.settings.stage35_workflow,
                    "stage35",
                    output_node_id=self.settings.output_node_id,
                ),
            },
            "refine": {
                "path": str(self.settings.refine_workflow),
                "exists": self.settings.refine_workflow.exists(),
                "model_fields": _workflow_model_field_dicts(self.settings.refine_workflow),
                "validation": validate_comfy_workflow(
                    self.settings.refine_workflow,
                    "refine",
                    output_node_id=self.settings.output_node_id,
                ),
            },
            "placeholders": placeholder_names(),
        }

    def model_choices(self) -> dict[str, Any]:
        if not self.settings.enabled:
            return {
                "ok": False,
                "base_url": self.settings.base_url,
                "error": "ComfyUI backend is disabled in configs/default.json.",
                "choice_groups": [],
            }
        try:
            object_info = self._get_json("/object_info", timeout=5)
        except Exception as exc:
            return {
                "ok": False,
                "base_url": self.settings.base_url,
                "error": str(exc),
                "choice_groups": [],
            }
        if not isinstance(object_info, dict):
            return {
                "ok": False,
                "base_url": self.settings.base_url,
                "error": "ComfyUI object_info response was not a JSON object.",
                "choice_groups": [],
            }
        groups: list[dict[str, Any]] = []
        for class_type, class_info in sorted(object_info.items()):
            if not isinstance(class_info, dict):
                continue
            for field, choices in sorted(_choice_inputs_for_class(class_info).items()):
                if not _is_model_choice_field(str(class_type), field):
                    continue
                groups.append(
                    {
                        "class_type": str(class_type),
                        "field": field,
                        "count": len(choices),
                        "preview": choices[:20],
                    }
                )
        return {
            "ok": True,
            "base_url": self.settings.base_url,
            "choice_group_count": len(groups),
            "choice_groups": groups,
        }

    def node_compatibility(self) -> dict[str, Any]:
        try:
            object_info = self._get_json("/object_info", timeout=5)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"Could not read ComfyUI object_info: {exc}",
                "stage3": {},
                "stage35": {},
                "refine": {},
            }
        if not isinstance(object_info, dict):
            return {
                "ok": False,
                "error": "ComfyUI object_info response was not a JSON object.",
                "stage3": {},
                "stage35": {},
                "refine": {},
            }
        stage3 = _workflow_node_compatibility_with_object_info(self.settings.stage3_workflow, object_info)
        stage35 = _workflow_node_compatibility_with_object_info(self.settings.stage35_workflow, object_info)
        refine = _workflow_node_compatibility_with_object_info(self.settings.refine_workflow, object_info)
        required = [stage3]
        for item in (stage35, refine):
            if item.get("exists"):
                required.append(item)
        return {
            "ok": all(bool(item.get("ok")) for item in required),
            "available_node_class_count": len(object_info),
            "stage3": stage3,
            "stage35": stage35,
            "refine": refine,
        }

    def install_workflow(self, stage: str, data: bytes) -> dict[str, Any]:
        target_path = self.workflow_path_for_stage(stage)
        return install_comfy_workflow_bytes(
            data=data,
            target_path=target_path,
            stage=stage,
            output_node_id=self.settings.output_node_id,
        )

    def workflow_path_for_stage(self, stage: str) -> Path:
        stage_key_value = stage_key(stage)
        if stage_key_value == "stage3":
            return self.settings.stage3_workflow
        if stage_key_value == "stage35":
            return self.settings.stage35_workflow
        if stage_key_value == "refine":
            return self.settings.refine_workflow
        raise ValueError(f"Unsupported ComfyUI workflow stage: {stage}")

    def generate_stage3(
        self,
        run_dir: Path,
        source_image: Image.Image,
        depth_image: Image.Image,
        control_image: Image.Image,
        prompt: str,
        clip_prompt: str | None,
        negative_prompt: str | None,
        seed: int,
        steps: int,
        guidance: float,
        strength: float,
    ) -> ComfyGenerationResult:
        self._ensure_ready(self.settings.stage3_workflow, stage="stage3")
        comfy_dir = run_dir / "comfyui"
        comfy_dir.mkdir(parents=True, exist_ok=True)

        source_path = comfy_dir / "source.png"
        depth_path = comfy_dir / "depth.png"
        control_path = comfy_dir / "control_depth.png"
        source_image.convert("RGB").save(source_path)
        depth_image.convert("RGB").save(depth_path)
        control_image.convert("RGB").save(control_path)

        source_name = self.upload_image(source_path)
        depth_name = self.upload_image(depth_path)
        control_name = self.upload_image(control_path)

        replacements = _placeholder_values(
            source_image=source_name,
            depth_image=depth_name,
            control_image=control_name,
            prompt=prompt,
            clip_prompt=clip_prompt or prompt,
            negative_prompt=negative_prompt or "",
            seed=seed,
            steps=steps,
            guidance=guidance,
            strength=strength,
            width=source_image.width,
            height=source_image.height,
        )
        workflow = self.load_workflow(self.settings.stage3_workflow, replacements)
        started = time.perf_counter()
        prompt_id = self.queue_prompt(workflow)
        history = self.wait_for_history(prompt_id)
        image_info = self._find_output_image(history, prompt_id)
        output_image = self.download_image(image_info).convert("RGB")
        output_path = comfy_dir / "stage3_comfy_result.png"
        output_image.save(output_path)

        return ComfyGenerationResult(
            image=output_image,
            backend="ComfyUI Stage 3 workflow",
            metadata={
                "base_url": self.settings.base_url,
                "workflow": str(self.settings.stage3_workflow),
                "prompt_id": prompt_id,
                "output_node_id": image_info.get("node_id"),
                "output_filename": image_info.get("filename"),
                "output_subfolder": image_info.get("subfolder", ""),
                "output_type": image_info.get("type", "output"),
                "local_result": str(output_path),
                "source_image_upload": source_name,
                "depth_image_upload": depth_name,
                "control_image_upload": control_name,
                "seconds": round(time.perf_counter() - started, 2),
                "steps": steps,
                "guidance": guidance,
                "strength": strength,
                "seed": seed,
            },
        )

    def generate_stage35(
        self,
        run_dir: Path,
        source_image: Image.Image,
        depth_image: Image.Image,
        mode: str,
        upscale_scale: float,
        refinement_strength: float,
        max_side: int,
    ) -> ComfyStage35Result:
        self._ensure_ready(self.settings.stage35_workflow, "Stage 3.5 workflow", stage="stage35")
        stage35_dir = run_dir / "stage35_refinement"
        comfy_dir = stage35_dir / "comfyui"
        comfy_dir.mkdir(parents=True, exist_ok=True)

        target_width, target_height = _target_size(source_image.size, upscale_scale, max_side)
        source_path = comfy_dir / "stage35_source.png"
        depth_path = comfy_dir / "stage35_depth.png"
        source_image.convert("RGB").save(source_path)
        depth_image.convert("RGB").save(depth_path)

        source_name = self.upload_image(source_path)
        depth_name = self.upload_image(depth_path)
        prompt_bundle = build_stage35_prompt_bundle()
        common_replacements = _placeholder_values(
            source_image=source_name,
            depth_image=depth_name,
            control_image=depth_name,
            prompt=prompt_bundle.positive_prompt,
            clip_prompt=prompt_bundle.clip_prompt,
            negative_prompt=prompt_bundle.negative_prompt,
            seed=0,
            steps=8,
            guidance=3.5,
            strength=refinement_strength,
            denoise=refinement_strength,
            width=source_image.width,
            height=source_image.height,
            target_width=target_width,
            target_height=target_height,
            upscale_scale=upscale_scale,
            refinement_strength=refinement_strength,
            max_side=max_side,
            stage35_mode=mode,
        )
        reconstruction_path = comfy_dir / "stage35_comfy_reconstruction.png"
        reconstruction, reconstruction_meta = self._run_image_workflow(
            workflow_path=self.settings.stage35_workflow,
            replacements=common_replacements,
            output_path=reconstruction_path,
        )

        refined: Image.Image | None = None
        refine_meta: dict[str, Any] | None = None
        if self.settings.refine_workflow.exists():
            refine_source_path = comfy_dir / "stage35_refine_source.png"
            reconstruction.convert("RGB").save(refine_source_path)
            refine_source_name = self.upload_image(refine_source_path)
            refine_replacements = {
                **common_replacements,
                "__SOURCE_IMAGE__": refine_source_name,
                "__WIDTH__": reconstruction.width,
                "__HEIGHT__": reconstruction.height,
                "__TARGET_WIDTH__": reconstruction.width,
                "__TARGET_HEIGHT__": reconstruction.height,
            }
            refined_path = comfy_dir / "stage35_comfy_refined.png"
            refined, refine_meta = self._run_image_workflow(
                workflow_path=self.settings.refine_workflow,
                replacements=refine_replacements,
                output_path=refined_path,
            )

        return ComfyStage35Result(
            visual_image=reconstruction,
            reconstruction_image=reconstruction,
            refined_image=refined,
            backend="ComfyUI Stage 3.5 workflow",
            metadata={
                "base_url": self.settings.base_url,
                "stage35_workflow": str(self.settings.stage35_workflow),
                "refine_workflow": str(self.settings.refine_workflow) if self.settings.refine_workflow.exists() else None,
                "source_image_upload": source_name,
                "depth_image_upload": depth_name,
                "target_width": target_width,
                "target_height": target_height,
                "upscale_scale": upscale_scale,
                "refinement_strength": refinement_strength,
                "max_side": max_side,
                "mode": mode,
                "reconstruction": reconstruction_meta,
                "refine": refine_meta,
            },
        )

    def load_workflow(self, path: Path, replacements: dict[str, Any]) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            workflow = json.load(fh)
        return _replace_placeholders(workflow, replacements)

    def _run_image_workflow(
        self,
        workflow_path: Path,
        replacements: dict[str, Any],
        output_path: Path,
    ) -> tuple[Image.Image, dict[str, Any]]:
        workflow = self.load_workflow(workflow_path, replacements)
        started = time.perf_counter()
        prompt_id = self.queue_prompt(workflow)
        history = self.wait_for_history(prompt_id)
        image_info = self._find_output_image(history, prompt_id)
        image = self.download_image(image_info).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return image, {
            "workflow": str(workflow_path),
            "prompt_id": prompt_id,
            "output_node_id": image_info.get("node_id"),
            "output_filename": image_info.get("filename"),
            "output_subfolder": image_info.get("subfolder", ""),
            "output_type": image_info.get("type", "output"),
            "local_result": str(output_path),
            "seconds": round(time.perf_counter() - started, 2),
        }

    def upload_image(self, path: Path) -> str:
        requests = _requests_module(required=("post",))

        with path.open("rb") as fh:
            files = {"image": (path.name, fh, "image/png")}
            data = {
                "type": "input",
                "subfolder": self.settings.input_subfolder,
                "overwrite": "true",
            }
            response = requests.post(
                f"{self.settings.base_url}/upload/image",
                data=data,
                files=files,
                timeout=60,
            )
        response.raise_for_status()
        payload = response.json()
        name = str(payload.get("name") or path.name)
        subfolder = str(payload.get("subfolder") or self.settings.input_subfolder).strip("/")
        return f"{subfolder}/{name}" if subfolder else name

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        requests = _requests_module(required=("post",))

        client_id = str(uuid.uuid4())
        response = requests.post(
            f"{self.settings.base_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"ComfyUI workflow validation failed: {payload}")
        prompt_id = payload.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {payload}")
        return str(prompt_id)

    def wait_for_history(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.time() + self.settings.timeout_seconds
        while time.time() < deadline:
            history = self._get_json(f"/history/{prompt_id}", timeout=20)
            if prompt_id in history:
                item = history[prompt_id]
                if item.get("outputs"):
                    return history
                if item.get("status", {}).get("status_str") in {"error", "failed"}:
                    raise RuntimeError(f"ComfyUI execution failed: {item.get('status')}")
            time.sleep(max(0.1, self.settings.poll_interval_seconds))
        raise RuntimeError(f"ComfyUI prompt timed out after {self.settings.timeout_seconds}s: {prompt_id}")

    def download_image(self, image_info: dict[str, Any]) -> Image.Image:
        requests = _requests_module(required=("get",))

        params = {
            "filename": image_info["filename"],
            "subfolder": image_info.get("subfolder", ""),
            "type": image_info.get("type", "output"),
        }
        response = requests.get(
            f"{self.settings.base_url}/view?{urlencode(params)}",
            timeout=120,
        )
        response.raise_for_status()
        from io import BytesIO

        return Image.open(BytesIO(response.content)).convert("RGB")

    def _find_output_image(self, history: dict[str, Any], prompt_id: str) -> dict[str, Any]:
        outputs = history[prompt_id].get("outputs", {})
        preferred_node = self.settings.output_node_id.strip()
        if preferred_node:
            images = outputs.get(preferred_node, {}).get("images", [])
            if images:
                return {**images[0], "node_id": preferred_node}

        for node_id, output in outputs.items():
            images = output.get("images", [])
            if images:
                return {**images[0], "node_id": str(node_id)}
        raise RuntimeError(f"ComfyUI history contains no image outputs for prompt {prompt_id}.")

    def _ensure_ready(self, workflow_path: Path, label: str = "Stage 3 workflow", stage: str = "stage3") -> None:
        if not self.settings.enabled:
            raise RuntimeError("ComfyUI backend is disabled in configs/default.json.")
        validation = validate_comfy_workflow(
            workflow_path,
            stage,
            output_node_id=self.settings.output_node_id,
        )
        if not workflow_path.exists():
            raise RuntimeError(
                f"ComfyUI {label} file is missing. "
                f"Expected: {workflow_path}. Export a ComfyUI workflow with Save (API Format), "
                "then add DioramaForge placeholders described in workflows/comfy/README.md."
            )
        if not validation.get("ok"):
            errors = "; ".join(str(item) for item in validation.get("errors", []))
            raise RuntimeError(f"ComfyUI {label} contract is invalid: {errors}")
        status = self.status()
        if not status.get("ok"):
            raise RuntimeError(
                f"ComfyUI server is not reachable at {self.settings.base_url}: {status.get('error')}"
            )

    def _get_json(self, path: str, timeout: int) -> dict[str, Any]:
        requests = _requests_module(required=("get",))

        response = requests.get(f"{self.settings.base_url}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()


def _placeholder_values(**values: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "__SOURCE_IMAGE__": "",
        "__DEPTH_IMAGE__": "",
        "__CONTROL_IMAGE__": "",
        "__PROMPT__": "",
        "__CLIP_PROMPT__": "",
        "__NEGATIVE_PROMPT__": "",
        "__SEED__": 0,
        "__STEPS__": 8,
        "__GUIDANCE__": 3.5,
        "__STRENGTH__": 0.55,
        "__DENOISE__": 0.55,
        "__WIDTH__": 512,
        "__HEIGHT__": 512,
        "__TARGET_WIDTH__": 512,
        "__TARGET_HEIGHT__": 512,
        "__UPSCALE_SCALE__": 1.0,
        "__REFINEMENT_STRENGTH__": 0.22,
        "__MAX_SIDE__": 1536,
        "__STAGE35_MODE__": "structure_preserving",
    }
    aliases = {
        "source_image": "__SOURCE_IMAGE__",
        "depth_image": "__DEPTH_IMAGE__",
        "control_image": "__CONTROL_IMAGE__",
        "prompt": "__PROMPT__",
        "clip_prompt": "__CLIP_PROMPT__",
        "negative_prompt": "__NEGATIVE_PROMPT__",
        "seed": "__SEED__",
        "steps": "__STEPS__",
        "guidance": "__GUIDANCE__",
        "strength": "__STRENGTH__",
        "denoise": "__DENOISE__",
        "width": "__WIDTH__",
        "height": "__HEIGHT__",
        "target_width": "__TARGET_WIDTH__",
        "target_height": "__TARGET_HEIGHT__",
        "upscale_scale": "__UPSCALE_SCALE__",
        "refinement_strength": "__REFINEMENT_STRENGTH__",
        "max_side": "__MAX_SIDE__",
        "stage35_mode": "__STAGE35_MODE__",
    }
    for key, value in values.items():
        placeholder = aliases.get(key)
        if placeholder:
            defaults[placeholder] = value
    return defaults


def _replace_placeholders(value: Any, replacements: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_placeholders(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_placeholders(item, replacements) for item in value]
    if isinstance(value, str):
        if value in replacements:
            return replacements[value]
        output = value
        for placeholder, replacement in replacements.items():
            if placeholder in output:
                output = output.replace(placeholder, str(replacement))
        return output
    return value


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


def _workflow_node_compatibility(path: Path, available: set[str]) -> dict[str, Any]:
    return _workflow_node_compatibility_with_object_info(path, {class_type: {} for class_type in available})


def _workflow_node_compatibility_with_object_info(path: Path, object_info: dict[str, Any]) -> dict[str, Any]:
    available = set(str(key) for key in object_info)
    result: dict[str, Any] = {
        "ok": False,
        "path": str(path),
        "exists": path.exists(),
        "class_types": [],
        "missing_class_types": [],
        "invalid_input_choices": [],
        "error": "",
    }
    if not path.exists():
        result["error"] = f"Workflow file does not exist: {path}"
        return result
    try:
        with path.open("r", encoding="utf-8") as fh:
            workflow = json.load(fh)
    except Exception as exc:
        result["error"] = f"Workflow could not be read: {exc}"
        return result
    if not isinstance(workflow, dict):
        result["error"] = "Workflow is not a JSON object."
        return result
    class_types = sorted(
        {
            str(value.get("class_type"))
            for value in workflow.values()
            if isinstance(value, dict) and value.get("class_type")
        }
    )
    result["class_types"] = class_types
    result["missing_class_types"] = [item for item in class_types if item not in available]
    result["invalid_input_choices"] = _invalid_workflow_input_choices(workflow, object_info)
    result["ok"] = not result["missing_class_types"] and not result["invalid_input_choices"]
    if result["missing_class_types"]:
        result["error"] = "ComfyUI server is missing node classes used by this workflow."
    elif result["invalid_input_choices"]:
        result["error"] = "ComfyUI server is missing model/input choices used by this workflow."
    return result


def _invalid_workflow_input_choices(workflow: dict[str, Any], object_info: dict[str, Any]) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        class_info = object_info.get(class_type)
        if not isinstance(class_info, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        choices_by_field = _choice_inputs_for_class(class_info)
        for field, value in inputs.items():
            field_name = str(field)
            choices = choices_by_field.get(field_name)
            if not choices or _is_runtime_or_link_value(value):
                continue
            text_value = str(value)
            if text_value in choices:
                continue
            invalid.append(
                {
                    "node_id": str(node_id),
                    "class_type": class_type,
                    "field": field_name,
                    "value": text_value,
                    "available_preview": choices[:8],
                    "available_count": len(choices),
                }
            )
    return invalid


def _choice_inputs_for_class(class_info: dict[str, Any]) -> dict[str, list[str]]:
    input_info = class_info.get("input", {})
    if not isinstance(input_info, dict):
        return {}
    choices: dict[str, list[str]] = {}
    for section_name in ("required", "optional"):
        section = input_info.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for field, spec in section.items():
            values = _choices_from_input_spec(spec)
            if values:
                choices[str(field)] = values
    return choices


def _choices_from_input_spec(spec: Any) -> list[str]:
    if not isinstance(spec, (list, tuple)) or not spec:
        return []
    first = spec[0]
    if not isinstance(first, (list, tuple)):
        return []
    values = [str(item) for item in first if isinstance(item, (str, int, float, bool))]
    return sorted({item for item in values if item})


def _is_runtime_or_link_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("__") and value.endswith("__")
    if isinstance(value, list):
        return True
    return value is None


def _is_model_choice_field(class_type: str, field: str) -> bool:
    text = f"{class_type} {field}".lower()
    markers = (
        "ckpt",
        "checkpoint",
        "control_net",
        "controlnet",
        "lora",
        "vae",
        "unet",
        "diffusion_model",
        "clip_name",
        "upscale_model",
        "sampler_name",
        "scheduler",
    )
    return any(marker in text for marker in markers)


def _workflow_model_field_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        return [field.as_dict() for field in workflow_model_fields(path)]
    except Exception:
        return []


def _requests_module(required: tuple[str, ...]):
    import requests

    missing = [name for name in required if not hasattr(requests, name)]
    if missing:
        module_file = getattr(requests, "__file__", None) or "namespace/no __file__"
        raise RuntimeError(f"Python package requests is incomplete at {module_file}; missing {', '.join(missing)}.")
    return requests
