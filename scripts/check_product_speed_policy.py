from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.api import create_api_app


CONFIG_PATH = ROOT / "configs" / "default.json"
DESKTOP_MAIN = ROOT / "desktop" / "src" / "main.jsx"
DEMO_BUDGET_SECONDS = 240
MAX_PRODUCT_STEPS = 12


def main() -> None:
    failures: list[str] = []
    config = _read_json(CONFIG_PATH)
    product = config.get("product_pipeline", {})
    style = config.get("style_engine", {})
    desktop_source = DESKTOP_MAIN.read_text(encoding="utf-8") if DESKTOP_MAIN.exists() else ""

    if int(product.get("demo_time_budget_seconds", 0)) > DEMO_BUDGET_SECONDS:
        failures.append(f"product demo budget must be <= {DEMO_BUDGET_SECONDS}s")
    if int(product.get("steps", 999)) > MAX_PRODUCT_STEPS:
        failures.append(f"product Generate should stay at {MAX_PRODUCT_STEPS} steps or fewer for the live profile")
    if int(product.get("max_resolution", 9999)) > 512:
        failures.append("product Generate should stay at 512 px or lower for the live profile")
    if str(product.get("stage35_backend_mode", "")).lower() == "comfyui":
        failures.append("Stage 3.5 live default must not trigger a second heavy ComfyUI pass")
    if str(style.get("backend_mode", "")).lower() != "comfyui":
        failures.append("Stage 3 product backend must stay routed through ComfyUI workflow")
    if style.get("show_backend_selector") is not False:
        failures.append("style_engine.show_backend_selector must remain false")
    if style.get("legacy_remote_visible") is not False:
        failures.append("style_engine.legacy_remote_visible must remain false")

    forbidden_config_keys = {
        "quality_modes",
        "quality_profiles",
        "test_modes",
        "paper_modes",
        "operation_modes",
        "user_modes",
        "backend_selector",
    }
    present_forbidden = sorted(key for key in forbidden_config_keys if _contains_key(config, key))
    if present_forbidden:
        failures.append(f"forbidden product mode keys found in config: {', '.join(present_forbidden)}")

    forbidden_desktop_terms = (
        "Quality Mode",
        "Test Mode",
        "Paper Mode",
        "Model Selector",
        "Backend Selector",
        "Real Models Only",
    )
    for term in forbidden_desktop_terms:
        if term in desktop_source:
            failures.append(f"forbidden user-facing mode text found in desktop: {term}")

    app = create_api_app(CONFIG_PATH)
    routes = {getattr(route, "path", ""): route for route in app.routes}
    product_route = routes.get("/api/pipeline/jobs")
    if product_route is None:
        failures.append("missing /api/pipeline/jobs route")
    else:
        signature = inspect.signature(product_route.endpoint)
        forbidden_params = (
            "backend_mode",
            "stage35_backend_mode",
            "stage4_backend_mode",
            "stage5_backend_mode",
            "quality_mode",
            "test_mode",
            "paper_mode",
        )
        for name in forbidden_params:
            if name in signature.parameters:
                failures.append(f"/api/pipeline/jobs exposes forbidden mode parameter: {name}")

    summary = {
        "ok": not failures,
        "failures": failures,
        "demo_budget_seconds": product.get("demo_time_budget_seconds"),
        "steps": product.get("steps"),
        "max_resolution": product.get("max_resolution"),
        "style_backend": style.get("backend_mode"),
        "stage35_backend": product.get("stage35_backend_mode"),
        "user_facing_mode": "single_generate",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_key(value: Any, needle: str) -> bool:
    if isinstance(value, dict):
        return any(key == needle or _contains_key(item, needle) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_key(item, needle) for item in value)
    return False


if __name__ == "__main__":
    main()
