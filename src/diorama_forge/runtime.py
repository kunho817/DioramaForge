from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeStatus:
    torch_available: bool
    torch_version: str
    cuda_available: bool
    device_name: str
    total_vram_gb: float | None
    free_vram_gb: float | None
    message: str


def get_runtime_status() -> RuntimeStatus:
    try:
        import torch
    except Exception as exc:
        return RuntimeStatus(False, "not installed", False, "CPU", None, None, f"PyTorch 로드 실패: {exc}")

    torch_version = getattr(torch, "__version__", "unknown")
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return RuntimeStatus(True, torch_version, False, "CPU", None, None, "CUDA 사용 불가: CPU 모드")

    try:
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        return RuntimeStatus(
            True,
            torch_version,
            True,
            props.name,
            round(total_bytes / (1024**3), 2),
            round(free_bytes / (1024**3), 2),
            "CUDA 사용 가능",
        )
    except Exception as exc:
        return RuntimeStatus(True, torch_version, True, "CUDA", None, None, f"CUDA 상태 확인 실패: {exc}")


def get_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def torch_dtype_from_name(name: str):
    import torch

    lowered = name.lower()
    if lowered == "float16":
        return torch.float16
    if lowered == "float32":
        return torch.float32
    return torch.bfloat16


def runtime_status_markdown() -> str:
    status = get_runtime_status()
    if status.cuda_available:
        return (
            f"**Runtime**: {status.message} | **GPU**: {status.device_name} | "
            f"**VRAM**: {status.free_vram_gb} GB free / {status.total_vram_gb} GB total | "
            f"**PyTorch**: {status.torch_version}"
        )
    return f"**Runtime**: {status.message} | **Device**: {status.device_name} | **PyTorch**: {status.torch_version}"
