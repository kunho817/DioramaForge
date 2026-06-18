from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from .comfy import ComfyUIClient
from .comfy_workflow import validate_comfy_workflow
from .config import DioramaConfig


REQUIRED_STYLE_PACKAGES = (
    ("torch", "PyTorch"),
    ("diffusers", "Diffusers"),
    ("transformers", "Transformers"),
    ("accelerate", "Accelerate"),
    ("safetensors", "Safetensors"),
)


def demo_runtime_status(config: DioramaConfig) -> dict[str, Any]:
    backend_mode = str(config.style_engine.backend_mode or "auto").strip().lower()
    if backend_mode == "comfyui":
        return _comfyui_runtime_status(config)

    checks: list[dict[str, Any]] = []
    for import_name, label in REQUIRED_STYLE_PACKAGES:
        ready, detail = _package_import_ready(import_name)
        checks.append(
            {
                "id": f"package_{import_name}",
                "label": label,
                "ok": ready,
                "detail": detail,
            }
        )

    torch_status = _torch_status(config.product_pipeline.demo_min_free_vram_gb)
    checks.extend(torch_status["checks"])
    failed = [check for check in checks if not check["ok"]]
    return {
        "backend_mode": backend_mode,
        "ready": not failed,
        "failed_check_count": len(failed),
        "failed_checks": [check["label"] for check in failed],
        "min_free_vram_gb": config.product_pipeline.demo_min_free_vram_gb,
        "torch": torch_status["torch"],
        "checks": checks,
    }


def _comfyui_runtime_status(config: DioramaConfig) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    stage3_validation = validate_comfy_workflow(
        config.comfy.stage3_workflow,
        "stage3",
        output_node_id=config.comfy.output_node_id,
    )
    stage35_validation = validate_comfy_workflow(
        config.comfy.stage35_workflow,
        "stage35",
        output_node_id=config.comfy.output_node_id,
    )
    requests_ready, requests_detail = _package_import_ready("requests", required_attrs=("get", "post"))
    checks.append(
        {
            "id": "package_requests",
            "label": "Requests",
            "ok": requests_ready,
            "detail": requests_detail,
        }
    )
    checks.append(
        {
            "id": "comfy_enabled",
            "label": "ComfyUI enabled",
            "ok": bool(config.comfy.enabled),
            "detail": (
                f"ComfyUI backend enabled at {config.comfy.base_url}."
                if config.comfy.enabled
                else "ComfyUI backend is disabled in configs/default.json."
            ),
        }
    )
    checks.append(
        {
            "id": "stage3_workflow",
            "label": "Stage 3 workflow",
            "ok": bool(stage3_validation.get("ok")),
            "detail": _workflow_detail(stage3_validation, "Stage 3 workflow"),
            "validation": stage3_validation,
        }
    )
    stage35_mode = str(config.product_pipeline.stage35_backend_mode or "auto").strip().lower()
    stage35_required = stage35_mode == "comfyui"
    checks.append(
        {
            "id": "stage35_workflow",
            "label": "Stage 3.5 workflow",
            "ok": bool(stage35_validation.get("ok")) or not stage35_required,
            "detail": _optional_workflow_detail(
                stage35_validation,
                "Stage 3.5 workflow",
                "Stage 3.5 ComfyUI workflow is optional because product stage35 backend is auto.",
            ),
            "validation": stage35_validation,
            "required": stage35_required,
        }
    )

    status: dict[str, Any]
    if config.comfy.enabled and requests_ready:
        status = ComfyUIClient(config.comfy).status()
    else:
        status = {"ok": False, "base_url": config.comfy.base_url, "error": "ComfyUI preflight prerequisites failed."}
    checks.append(
        {
            "id": "comfy_server",
            "label": "ComfyUI server",
            "ok": bool(status.get("ok")),
            "detail": (
                f"Reachable at {config.comfy.base_url}."
                if status.get("ok")
                else f"Not reachable at {config.comfy.base_url}: {status.get('error', 'unknown error')}"
            ),
        }
    )
    if status.get("ok"):
        node_compatibility = status.get("node_compatibility", {})
        checks.append(
            {
                "id": "comfy_node_classes",
                "label": "ComfyUI node/model choices",
                "ok": bool(node_compatibility.get("ok")),
                "detail": _node_compatibility_detail(node_compatibility),
                "compatibility": node_compatibility,
            }
        )

    failed = [check for check in checks if not check["ok"]]
    return {
        "backend_mode": "comfyui",
        "ready": not failed,
        "failed_check_count": len(failed),
        "failed_checks": [check["label"] for check in failed],
        "min_free_vram_gb": config.product_pipeline.demo_min_free_vram_gb,
        "torch": {
            "available": None,
            "version": "",
            "cuda_available": None,
            "device_name": "ComfyUI",
            "free_vram_gb": None,
            "total_vram_gb": None,
        },
        "comfyui": status,
        "checks": checks,
    }


def _workflow_detail(validation: dict[str, Any], label: str) -> str:
    if validation.get("ok"):
        return (
            f"{label} valid: {validation.get('path')} "
            f"({validation.get('node_count', 0)} nodes)."
        )
    errors = validation.get("errors") or []
    if errors:
        return f"{label} invalid: {errors[0]}"
    return f"{label} invalid: unknown workflow contract error."


def _optional_workflow_detail(validation: dict[str, Any], label: str, optional_message: str) -> str:
    if validation.get("ok"):
        return (
            f"{label} valid: {validation.get('path')} "
            f"({validation.get('node_count', 0)} nodes)."
        )
    if not validation.get("exists"):
        return optional_message
    errors = validation.get("errors") or []
    if errors:
        return f"{label} invalid: {errors[0]}"
    return f"{label} invalid: unknown workflow contract error."


def _node_compatibility_detail(compatibility: dict[str, Any]) -> str:
    if compatibility.get("ok"):
        count = compatibility.get("available_node_class_count", 0)
        return f"ComfyUI server exposes required workflow node classes ({count} available classes)."
    if compatibility.get("error"):
        return str(compatibility["error"])
    stage3 = compatibility.get("stage3", {})
    missing = stage3.get("missing_class_types") or []
    if missing:
        return "Stage 3 workflow uses missing ComfyUI node classes: " + ", ".join(str(item) for item in missing)
    invalid_choices = stage3.get("invalid_input_choices") or []
    if invalid_choices:
        first = invalid_choices[0]
        return (
            "Stage 3 workflow references a ComfyUI input/model that is not available: "
            f"{first.get('class_type')}.{first.get('field')}={first.get('value')}"
        )
    return "ComfyUI workflow node class compatibility failed."


def _torch_status(min_free_vram_gb: float) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "torch": {
                "available": False,
                "version": "not installed",
                "cuda_available": False,
                "device_name": "CPU",
                "free_vram_gb": None,
                "total_vram_gb": None,
            },
            "checks": [
                {
                    "id": "torch_import",
                    "label": "PyTorch import",
                    "ok": False,
                    "detail": f"PyTorch import failed: {exc}",
                },
                {
                    "id": "cuda_available",
                    "label": "CUDA available",
                    "ok": False,
                    "detail": "CUDA cannot be checked because PyTorch import failed.",
                },
                {
                    "id": "free_vram",
                    "label": "Free VRAM",
                    "ok": False,
                    "detail": f"Need at least {min_free_vram_gb} GB free VRAM for the local demo profile.",
                },
            ],
        }

    version = getattr(torch, "__version__", "unknown")
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return {
            "torch": {
                "available": True,
                "version": version,
                "cuda_available": False,
                "device_name": "CPU",
                "free_vram_gb": None,
                "total_vram_gb": None,
            },
            "checks": [
                {
                    "id": "cuda_available",
                    "label": "CUDA available",
                    "ok": False,
                    "detail": "CUDA is not available; live local generation would fall back to CPU or fail.",
                },
                {
                    "id": "free_vram",
                    "label": "Free VRAM",
                    "ok": False,
                    "detail": f"Need at least {min_free_vram_gb} GB free VRAM for the local demo profile.",
                },
            ],
        }

    try:
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        free_gb = round(free_bytes / (1024**3), 2)
        total_gb = round(total_bytes / (1024**3), 2)
        enough_vram = free_gb >= min_free_vram_gb
        return {
            "torch": {
                "available": True,
                "version": version,
                "cuda_available": True,
                "device_name": props.name,
                "free_vram_gb": free_gb,
                "total_vram_gb": total_gb,
            },
            "checks": [
                {
                    "id": "cuda_available",
                    "label": "CUDA available",
                    "ok": True,
                    "detail": f"CUDA device detected: {props.name}.",
                },
                {
                    "id": "free_vram",
                    "label": "Free VRAM",
                    "ok": enough_vram,
                    "detail": (
                        f"{free_gb} GB free / {total_gb} GB total; minimum is {min_free_vram_gb} GB."
                    ),
                },
            ],
        }
    except Exception as exc:
        return {
            "torch": {
                "available": True,
                "version": version,
                "cuda_available": True,
                "device_name": "CUDA",
                "free_vram_gb": None,
                "total_vram_gb": None,
            },
            "checks": [
                {
                    "id": "cuda_available",
                    "label": "CUDA available",
                    "ok": True,
                    "detail": "CUDA is available through PyTorch.",
                },
                {
                    "id": "free_vram",
                    "label": "Free VRAM",
                    "ok": False,
                    "detail": f"Could not read CUDA memory info: {exc}",
                },
            ],
        }


def _package_import_ready(import_name: str, required_attrs: tuple[str, ...] = ()) -> tuple[bool, str]:
    if importlib.util.find_spec(import_name) is None:
        return False, f"Missing Python package: {import_name}."
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:
        return False, f"Package import failed for {import_name}: {exc}"
    missing_attrs = [attr for attr in required_attrs if not hasattr(module, attr)]
    if missing_attrs:
        module_file = getattr(module, "__file__", None) or "namespace/no __file__"
        return False, f"Package {import_name} is incomplete at {module_file}; missing {', '.join(missing_attrs)}."
    return True, "Package import is available."
