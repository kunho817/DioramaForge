from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import MeshySettings


SUCCEEDED = "SUCCEEDED"
FAILED_STATUSES = {"FAILED", "EXPIRED", "CANCELED", "CANCELLED"}


@dataclass(frozen=True)
class MeshyRunResult:
    task_id: str
    task: dict[str, Any]
    request_payload: dict[str, Any]
    downloads: dict[str, str]
    request_path: Path
    task_path: Path
    downloads_path: Path


class MeshyClient:
    def __init__(self, settings: MeshySettings) -> None:
        self.settings = settings

    def status(self) -> dict[str, Any]:
        api_key = self._api_key()
        requests_ready, requests_detail = _requests_ready()
        model_output_formats = _model_output_formats(self.settings.target_formats)
        model_output_formats_ready = bool(model_output_formats)
        download_outputs_ready = bool(self.settings.download_outputs)
        return {
            "ok": bool(
                self.settings.enabled
                and api_key
                and requests_ready
                and download_outputs_ready
                and model_output_formats_ready
            ),
            "enabled": self.settings.enabled,
            "base_url": self.settings.base_url,
            "api_key_env": self.settings.api_key_env,
            "api_key_present": bool(api_key),
            "requests_ready": requests_ready,
            "requests_detail": requests_detail,
            "target_formats": list(self.settings.target_formats),
            "download_outputs": self.settings.download_outputs,
            "download_outputs_ready": download_outputs_ready,
            "model_output_formats": model_output_formats,
            "model_output_formats_ready": model_output_formats_ready,
            "model_type": self.settings.model_type,
            "ai_model": self.settings.ai_model,
        }

    def run_image_to_3d(
        self,
        image_path: Path,
        output_dir: Path,
        texture_prompt: str = "",
        status=None,
    ) -> MeshyRunResult:
        if not self.settings.enabled:
            raise RuntimeError("Meshy AI backend is disabled in configs/default.json.")
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(
                f"Meshy AI API key is missing. Set {self.settings.api_key_env} before running Stage 4 with Meshy."
            )
        requests = _requests_module(required=("get", "post"))
        output_dir.mkdir(parents=True, exist_ok=True)
        emit = status or (lambda _message: None)

        payload = self._request_payload(image_path, texture_prompt)
        request_path = output_dir / "meshy_request.json"
        request_path.write_text(json.dumps(_redact_payload(payload), ensure_ascii=False, indent=2), encoding="utf-8")

        emit("Submitting Meshy image-to-3D task")
        response = requests.post(
            f"{self.settings.base_url}/openapi/v1/image-to-3d",
            headers=self._headers(api_key),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        created = response.json()
        task_id = created.get("result")
        if not task_id:
            raise RuntimeError(f"Meshy did not return a task id: {created}")

        emit(f"Polling Meshy task: {task_id}")
        task = self.wait_for_task(str(task_id), api_key)
        task_path = output_dir / "meshy_task.json"
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

        downloads: dict[str, str] = {}
        if self.settings.download_outputs:
            downloads = self.download_outputs(task, output_dir, api_key)
        downloads_path = output_dir / "meshy_downloads.json"
        downloads_path.write_text(json.dumps(downloads, ensure_ascii=False, indent=2), encoding="utf-8")

        return MeshyRunResult(
            task_id=str(task_id),
            task=task,
            request_payload=_redact_payload(payload),
            downloads=downloads,
            request_path=request_path,
            task_path=task_path,
            downloads_path=downloads_path,
        )

    def wait_for_task(self, task_id: str, api_key: str) -> dict[str, Any]:
        requests = _requests_module(required=("get",))
        deadline = time.time() + max(1, self.settings.timeout_seconds)
        last_task: dict[str, Any] = {}
        while time.time() < deadline:
            response = requests.get(
                f"{self.settings.base_url}/openapi/v1/image-to-3d/{task_id}",
                headers=self._headers(api_key),
                timeout=60,
            )
            response.raise_for_status()
            last_task = response.json()
            task_status = str(last_task.get("status", "")).upper()
            if task_status == SUCCEEDED:
                return last_task
            if task_status in FAILED_STATUSES:
                error = last_task.get("task_error") or {}
                message = error.get("message") if isinstance(error, dict) else ""
                raise RuntimeError(f"Meshy task failed with status {task_status}: {message or last_task}")
            time.sleep(max(0.5, self.settings.poll_interval_seconds))
        raise RuntimeError(f"Meshy task timed out after {self.settings.timeout_seconds}s: {task_id}")

    def download_outputs(self, task: dict[str, Any], output_dir: Path, api_key: str) -> dict[str, str]:
        requests = _requests_module(required=("get",))
        downloads: dict[str, str] = {}
        model_urls = task.get("model_urls") or {}
        if isinstance(model_urls, dict):
            for key, url in model_urls.items():
                if not url:
                    continue
                suffix = _suffix_from_url(str(url), fallback=f".{key}")
                path = output_dir / f"model_{key}{suffix}"
                self._download_url(requests, str(url), path, api_key)
                downloads[str(key)] = str(path)

        thumbnail_url = task.get("thumbnail_url")
        if thumbnail_url:
            path = output_dir / f"thumbnail{_suffix_from_url(str(thumbnail_url), '.png')}"
            self._download_url(requests, str(thumbnail_url), path, api_key)
            downloads["thumbnail"] = str(path)
        return downloads

    def _request_payload(self, image_path: Path, texture_prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "image_url": _image_data_uri(image_path),
            "model_type": self.settings.model_type,
            "ai_model": self.settings.ai_model,
            "should_texture": self.settings.should_texture,
            "enable_pbr": self.settings.enable_pbr,
            "should_remesh": self.settings.should_remesh,
            "target_polycount": self.settings.target_polycount,
            "target_formats": list(self.settings.target_formats),
            "image_enhancement": self.settings.image_enhancement,
            "remove_lighting": self.settings.remove_lighting,
            "moderation": self.settings.moderation,
        }
        prompt = texture_prompt.strip()
        if prompt:
            payload["texture_prompt"] = prompt[:600]
        return payload

    def _download_url(self, requests, url: str, path: Path, api_key: str) -> None:
        response = requests.get(url, headers=self._headers(api_key), timeout=180)
        response.raise_for_status()
        path.write_bytes(response.content)

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _api_key(self) -> str:
        return os.environ.get(self.settings.api_key_env, "").strip()


def _image_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(payload)
    image_url = str(redacted.get("image_url", ""))
    if image_url.startswith("data:"):
        redacted["image_url"] = image_url[:48] + "...<redacted base64>"
    return redacted


def _model_output_formats(target_formats: tuple[str, ...]) -> list[str]:
    return sorted({str(item).lower() for item in target_formats if str(item).lower() in {"glb", "obj", "stl"}})


def _suffix_from_url(url: str, fallback: str) -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix
    return suffix or fallback


def _requests_ready() -> tuple[bool, str]:
    try:
        requests = _requests_module(required=("get", "post"))
    except Exception as exc:
        return False, str(exc)
    module_file = getattr(requests, "__file__", None) or "namespace/no __file__"
    return True, f"requests available at {module_file}"


def _requests_module(required: tuple[str, ...]):
    import requests

    missing = [name for name in required if not hasattr(requests, name)]
    if missing:
        module_file = getattr(requests, "__file__", None) or "namespace/no __file__"
        raise RuntimeError(f"Python package requests is incomplete at {module_file}; missing {', '.join(missing)}.")
    return requests
