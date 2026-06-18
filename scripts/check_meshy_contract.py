from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image

from diorama_forge.config import load_config
from diorama_forge.meshy import MeshyClient, _redact_payload


def main() -> None:
    config = load_config(ROOT / "configs" / "default.json")
    client = MeshyClient(config.meshy)
    failures: list[str] = []

    original_key = os.environ.get(config.meshy.api_key_env)
    try:
        os.environ.pop(config.meshy.api_key_env, None)
        status_without_key = client.status()
        if status_without_key.get("api_key_present"):
            failures.append("status reports api_key_present=true without environment key")

        os.environ[config.meshy.api_key_env] = "msy_contract_test"
        status_with_key = client.status()
        if not status_with_key.get("api_key_present"):
            failures.append("status does not detect configured API key environment variable")
        if not status_with_key.get("requests_ready"):
            failures.append("requests package is not ready for Meshy client")
        if not status_with_key.get("download_outputs_ready"):
            failures.append("Meshy status does not require download_outputs=true")
        if not status_with_key.get("model_output_formats_ready"):
            failures.append("Meshy status does not detect model output target formats")
        if sorted(status_with_key.get("model_output_formats", [])) != sorted(
            {item for item in config.meshy.target_formats if item in {"glb", "obj", "stl"}}
        ):
            failures.append("Meshy status model_output_formats does not match target_formats")
        if MeshyClient(replace(config.meshy, download_outputs=False)).status().get("ok"):
            failures.append("Meshy status accepted download_outputs=false")
        if MeshyClient(replace(config.meshy, target_formats=("fbx",))).status().get("ok"):
            failures.append("Meshy status accepted target_formats without GLB/OBJ/STL")

        with TemporaryDirectory(prefix="diorama_meshy_contract_") as temp_value:
            temp_dir = Path(temp_value)
            image_path = temp_dir / "input.png"
            Image.new("RGB", (32, 24), (120, 160, 200)).save(image_path)
            payload = client._request_payload(image_path, "small diorama model")
            if not str(payload.get("image_url", "")).startswith("data:image/png;base64,"):
                failures.append("Meshy payload does not use a PNG data URI")
            if payload.get("target_formats") != list(config.meshy.target_formats):
                failures.append("Meshy payload target_formats does not match config")
            redacted = _redact_payload(payload)
            if "base64" not in str(redacted.get("image_url", "")) or len(str(redacted.get("image_url", ""))) > 90:
                failures.append("Meshy payload redaction did not shorten the data URI")
    finally:
        if original_key is None:
            os.environ.pop(config.meshy.api_key_env, None)
        else:
            os.environ[config.meshy.api_key_env] = original_key

    summary = {
        "ok": not failures,
        "failures": failures,
        "base_url": config.meshy.base_url,
        "api_key_env": config.meshy.api_key_env,
        "target_formats": list(config.meshy.target_formats),
        "model_output_formats": status_with_key.get("model_output_formats", []),
        "network_used": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
