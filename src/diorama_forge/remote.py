from __future__ import annotations

import json
import os
import time
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .config import RemoteBackendSettings


StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class RemoteImportResult:
    run_dir: Path
    metadata: dict[str, Any]


class RemoteModelClient:
    def __init__(self, settings: RemoteBackendSettings, root: Path, output_dir: Path) -> None:
        self.settings = settings
        self.root = root
        self.output_dir = output_dir

    def status(self) -> dict[str, Any]:
        if not self.settings.enabled:
            return {
                "ok": False,
                "base_url": self.settings.base_url,
                "error": "Remote backend is disabled in configs/default.json.",
            }
        try:
            import requests

            response = requests.get(
                f"{self.settings.base_url}/api/remote/health",
                headers=self._headers(),
                timeout=5,
            )
            response.raise_for_status()
            payload = response.json()
            return {"ok": True, "base_url": self.settings.base_url, **payload}
        except Exception as exc:
            return {"ok": False, "base_url": self.settings.base_url, "error": str(exc)}

    def run_stage3(
        self,
        image: Image.Image,
        fields: dict[str, Any],
        status: StatusCallback | None = None,
    ) -> RemoteImportResult:
        self._ensure_enabled()
        emit = status or (lambda _message: None)
        emit("Remote A100 Stage 3 요청 전송 중")
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        buffer.seek(0)
        response_bytes = self._post_zip(
            "/api/remote/stage3/run",
            data={key: str(value) for key, value in fields.items()},
            files={"image": ("input.png", buffer.getvalue(), "image/png")},
        )
        result = self._import_zip(response_bytes, target_run_dir=None)
        emit(f"Remote A100 Stage 3 결과 수신 완료: {result.run_dir}")
        return result

    def run_stage35(
        self,
        run_dir: Path,
        fields: dict[str, Any],
        status: StatusCallback | None = None,
    ) -> RemoteImportResult:
        self._ensure_enabled()
        emit = status or (lambda _message: None)
        emit("Remote A100 Stage 3.5 요청 패키징 중")
        package = _zip_directory(run_dir, package_name="local_stage35_input")
        response_bytes = self._post_zip(
            "/api/remote/stage35/run",
            data={key: str(value) for key, value in fields.items()},
            files={"run_zip": ("run.zip", package, "application/zip")},
        )
        result = self._import_zip(response_bytes, target_run_dir=run_dir)
        emit("Remote A100 Stage 3.5 결과 수신 완료")
        return result

    def run_stage4(
        self,
        run_dir: Path,
        fields: dict[str, Any],
        status: StatusCallback | None = None,
    ) -> RemoteImportResult:
        self._ensure_enabled()
        emit = status or (lambda _message: None)
        emit("Remote A100 Stage 4 요청 패키징 중")
        package = _zip_directory(run_dir, package_name="local_stage4_input")
        response_bytes = self._post_zip(
            "/api/remote/stage4/run",
            data={key: str(value) for key, value in fields.items()},
            files={"run_zip": ("run.zip", package, "application/zip")},
        )
        result = self._import_zip(response_bytes, target_run_dir=run_dir)
        emit("Remote A100 Stage 4 결과 수신 완료")
        return result

    def _post_zip(self, path: str, data: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> bytes:
        import requests

        response = requests.post(
            f"{self.settings.base_url}{path}",
            headers=self._headers(),
            data=data,
            files=files,
            timeout=self.settings.timeout_seconds,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:2000]
            raise RuntimeError(
                f"Remote backend request failed ({response.status_code}) for {path}: {detail}"
            ) from exc
        content_type = response.headers.get("content-type", "")
        if "application/zip" not in content_type and not response.content.startswith(b"PK"):
            raise RuntimeError(f"Remote backend returned non-zip response: {content_type} {response.text[:500]}")
        return response.content

    def _import_zip(self, content: bytes, target_run_dir: Path | None) -> RemoteImportResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BytesIO(content), "r") as archive:
            manifest = _read_package_manifest(archive)
            if target_run_dir is None:
                run_id = str(manifest.get("run_id") or time.strftime("%Y%m%d_%H%M%S_remote"))
                target_run_dir = _unique_run_dir(self.output_dir, run_id)
            target_run_dir.mkdir(parents=True, exist_ok=True)
            _safe_extract(archive, target_run_dir)

        remote_run_dir = str(manifest.get("remote_run_dir") or "")
        if remote_run_dir:
            _rewrite_json_paths(target_run_dir, remote_run_dir, str(target_run_dir))
        import_manifest = {
            "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "remote_base_url": self.settings.base_url,
            "remote_run_dir": remote_run_dir,
            "local_run_dir": str(target_run_dir),
            "stage": manifest.get("stage"),
        }
        (target_run_dir / "remote_import.json").write_text(
            json.dumps(import_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return RemoteImportResult(run_dir=target_run_dir, metadata=import_manifest)

    def _headers(self) -> dict[str, str]:
        token = os.environ.get(self.settings.api_key_env, "")
        return {"X-DioramaForge-Key": token} if token else {}

    def _ensure_enabled(self) -> None:
        if not self.settings.enabled:
            raise RuntimeError("Remote backend is disabled. Set remote_backend.enabled=true in configs/default.json.")


def _zip_directory(run_dir: Path, package_name: str) -> bytes:
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
                    "package": package_name,
                    "run_id": run_dir.name,
                    "remote_run_dir": str(run_dir),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
    return buffer.getvalue()


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
            raise RuntimeError(f"Unsafe path in remote zip: {member.filename}")
        destination = (target_root / member.filename).resolve()
        if target_root != destination and target_root not in destination.parents:
            raise RuntimeError(f"Unsafe path in remote zip: {member.filename}")
        if member.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, destination.open("wb") as dst:
                dst.write(src.read())


def _unique_run_dir(output_dir: Path, run_id: str) -> Path:
    candidate = output_dir / run_id
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        next_candidate = output_dir / f"{run_id}_remote_{suffix:02d}"
        if not next_candidate.exists():
            return next_candidate
        suffix += 1


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
