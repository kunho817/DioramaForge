from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.comfy import ComfyUIClient
from diorama_forge.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether a running ComfyUI server exposes node classes and model choices used by configured workflows."
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="Exit with failure if ComfyUI is unreachable or workflow node/model compatibility fails.",
    )
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "default.json")
    client = ComfyUIClient(config.comfy)
    status = client.status()
    compatibility = status.get("node_compatibility") or {}
    payload: dict[str, Any] = {
        "ok": bool(status.get("ok")) and bool(compatibility.get("ok")),
        "server_ok": bool(status.get("ok")),
        "base_url": config.comfy.base_url,
        "error": status.get("error", ""),
        "node_compatibility": compatibility,
        "failures": _failures(status, compatibility),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.require and not payload["ok"]:
        raise SystemExit(1)


def _failures(status: dict[str, Any], compatibility: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not status.get("ok"):
        failures.append(f"ComfyUI server is not reachable: {status.get('error', 'unknown error')}")
        return failures
    if not compatibility.get("ok"):
        if compatibility.get("error"):
            failures.append(str(compatibility["error"]))
        for name in ("stage3", "stage35", "refine"):
            item = compatibility.get(name, {})
            if item.get("exists") and item.get("missing_class_types"):
                missing = ", ".join(str(value) for value in item["missing_class_types"])
                failures.append(f"{name} workflow missing node classes: {missing}")
            invalid_choices = item.get("invalid_input_choices") or []
            for invalid in invalid_choices[:3]:
                failures.append(
                    f"{name} workflow missing input choice: "
                    f"{invalid.get('class_type')}.{invalid.get('field')}={invalid.get('value')}"
                )
    return failures


if __name__ == "__main__":
    main()
