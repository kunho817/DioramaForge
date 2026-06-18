from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.api import create_api_app


REQUIRED_ROUTES = (
    "/api/style-engine",
    "/api/demo/readiness",
    "/api/pipeline/defaults",
    "/api/pipeline/preflight",
    "/api/execution/policy",
    "/api/comfy/workflows",
    "/api/comfy/model-choices",
    "/api/meshy/status",
    "/api/comfy/workflows/{stage_name}/inspect",
    "/api/comfy/workflows/{stage_name}/install",
    "/api/comfy/workflows/{stage_name}/install-example",
    "/api/comfy/workflows/{stage_name}/prepare-install",
    "/api/stage3/run",
    "/api/stage3/jobs",
    "/api/pipeline/jobs",
)

CALLABLE_ROUTES = (
    "/api/style-engine",
    "/api/demo/readiness",
    "/api/pipeline/defaults",
    "/api/pipeline/preflight",
    "/api/execution/policy",
    "/api/comfy/workflows",
    "/api/comfy/model-choices",
    "/api/meshy/status",
)


def main() -> None:
    app = create_api_app(ROOT / "configs" / "default.json")
    routes = {getattr(route, "path", ""): route for route in app.routes}
    failures: list[str] = []

    for path in REQUIRED_ROUTES:
        if path not in routes:
            failures.append(f"missing route: {path}")

    payloads: dict[str, Any] = {}
    for path in CALLABLE_ROUTES:
        if path not in routes:
            continue
        payloads[path] = routes[path].endpoint()

    if "/api/style-engine" in payloads:
        _expect_keys(
            failures,
            "/api/style-engine",
            payloads["/api/style-engine"],
            (
                "resolved_active",
                "configured_active",
                "target",
                "backend_mode",
                "fast_path_ready",
                "demo_ready",
                "timed_smoke",
                "runtime",
                "demo_checks",
                "next_action",
                "readiness",
                "prepare_command",
                "benchmark_command",
                "benchmark_check_command",
            ),
        )
        readiness = payloads["/api/style-engine"].get("readiness", {})
        _expect_keys(failures, "/api/style-engine.readiness", readiness, ("flux_depth", "sdxl_depth_lightning"))
        if payloads["/api/style-engine"].get("demo_ready") and not payloads["/api/style-engine"].get("fast_path_ready"):
            failures.append("/api/style-engine: demo_ready cannot be true when fast_path_ready is false")

    if "/api/demo/readiness" in payloads:
        _expect_keys(
            failures,
            "/api/demo/readiness",
            payloads["/api/demo/readiness"],
            (
                "ok",
                "fast_path_ready",
                "runtime_ready",
                "timed_smoke_ready",
                "product_3d_ready",
                "demo_time_budget_seconds",
                "can_generate",
                "resolved_engine",
                "missing_fast_path_components",
                "runtime",
                "timed_smoke",
                "product_3d_backend",
                "product_pipeline_defaults",
                "prepare_command",
                "benchmark_command",
                "benchmark_check_command",
                "checks",
                "next_action",
                "warnings",
            ),
        )

    if "/api/pipeline/defaults" in payloads:
        defaults_payload = payloads["/api/pipeline/defaults"]
        _expect_keys(failures, "/api/pipeline/defaults", defaults_payload, ("user_facing_mode", "defaults"))
        _expect_keys(
            failures,
            "/api/pipeline/defaults.defaults",
            defaults_payload.get("defaults", {}),
            (
                "steps",
                "guidance",
                "strength",
                "max_resolution",
                "profile",
                "stage_contract",
                "demo_time_budget_seconds",
                "demo_min_free_vram_gb",
            ),
        )
        stage_contract = defaults_payload.get("defaults", {}).get("stage_contract")
        if stage_contract != ["stage3", "stage35", "stage4", "stage5"]:
            failures.append("/api/pipeline/defaults.defaults: stage_contract must be stage3 -> stage35 -> stage4 -> stage5")

    if "/api/pipeline/preflight" in payloads:
        preflight_payload = payloads["/api/pipeline/preflight"]
        _expect_keys(
            failures,
            "/api/pipeline/preflight",
            preflight_payload,
            (
                "ok",
                "user_facing_mode",
                "stage_contract",
                "backend_mode",
                "resolved_engine",
                "can_generate",
                "backends",
                "defaults",
                "checks",
                "errors",
                "warnings",
                "next_action",
                "runtime",
                "product_3d_backend",
                "timed_smoke",
            ),
        )
        if preflight_payload.get("user_facing_mode") != "single_generate":
            failures.append("/api/pipeline/preflight: user_facing_mode must be single_generate")
        if preflight_payload.get("stage_contract") != ["stage3", "stage35", "stage4", "stage5"]:
            failures.append("/api/pipeline/preflight: stage_contract must be stage3 -> stage35 -> stage4 -> stage5")
        checks = preflight_payload.get("checks", [])
        if not isinstance(checks, list) or not checks:
            failures.append("/api/pipeline/preflight: checks must be a non-empty list")
        else:
            required_check_ids = {"single_generate_contract", "local_execution_policy", "image_backend", "product_3d_backend", "timed_smoke"}
            found_check_ids = {item.get("id") for item in checks if isinstance(item, dict)}
            missing_check_ids = sorted(required_check_ids - found_check_ids)
            if missing_check_ids:
                failures.append(f"/api/pipeline/preflight: missing checks {', '.join(missing_check_ids)}")
            for item in checks:
                if not isinstance(item, dict):
                    failures.append("/api/pipeline/preflight: each check must be an object")
                    continue
                _expect_keys(failures, f"/api/pipeline/preflight.check.{item.get('id', '?')}", item, ("id", "label", "ok", "blocking", "detail"))

    if "/api/execution/policy" in payloads:
        _expect_keys(
            failures,
            "/api/execution/policy",
            payloads["/api/execution/policy"],
            (
                "allow_local_heavy_models",
                "user_facing_mode",
                "backend_selector_visible",
                "product_pipeline_defaults",
            ),
        )

    if "/api/comfy/workflows" in payloads:
        workflows_payload = payloads["/api/comfy/workflows"]
        _expect_keys(failures, "/api/comfy/workflows", workflows_payload, ("stage3", "stage35", "refine", "placeholders"))
        for name in ("stage3", "stage35", "refine"):
            workflow_item = workflows_payload.get(name, {})
            _expect_keys(
                failures,
                f"/api/comfy/workflows.{name}",
                workflow_item,
                ("path", "exists", "validation"),
            )
            _expect_keys(
                failures,
                f"/api/comfy/workflows.{name}.validation",
                workflow_item.get("validation", {}),
                (
                    "ok",
                    "stage",
                    "path",
                    "exists",
                    "errors",
                    "warnings",
                    "node_count",
                    "placeholders_found",
                    "missing_requirements",
                    "output_node_candidates",
                ),
            )

    if "/api/comfy/model-choices" in payloads:
        choices_payload = payloads["/api/comfy/model-choices"]
        _expect_keys(
            failures,
            "/api/comfy/model-choices",
            choices_payload,
            ("ok", "base_url", "choice_groups"),
        )
        choice_groups = choices_payload.get("choice_groups")
        if not isinstance(choice_groups, list):
            failures.append("/api/comfy/model-choices: choice_groups must be a list")
        elif choices_payload.get("ok"):
            for item in choice_groups[:10]:
                _expect_keys(
                    failures,
                    f"/api/comfy/model-choices.choice.{item.get('class_type', '?') if isinstance(item, dict) else '?'}",
                    item,
                    ("class_type", "field", "count", "preview"),
                )

    if "/api/meshy/status" in payloads:
        _expect_keys(
            failures,
            "/api/meshy/status",
            payloads["/api/meshy/status"],
            (
                "ok",
                "enabled",
                "base_url",
                "api_key_env",
                "api_key_present",
                "requests_ready",
                "target_formats",
                "download_outputs",
                "download_outputs_ready",
                "model_output_formats",
                "model_output_formats_ready",
                "model_type",
                "ai_model",
            ),
        )

    for path in ("/api/stage3/run", "/api/stage3/jobs", "/api/pipeline/jobs"):
        if path in routes:
            _expect_product_default_form_params(failures, path, routes[path].endpoint)
    if "/api/pipeline/jobs" in routes:
        _expect_no_product_stage_mode_params(failures, "/api/pipeline/jobs", routes["/api/pipeline/jobs"].endpoint)

    summary = {
        "ok": not failures,
        "failures": failures,
        "resolved_engine": payloads.get("/api/style-engine", {}).get("resolved_active"),
        "fast_path_ready": payloads.get("/api/demo/readiness", {}).get("fast_path_ready"),
        "runtime_ready": payloads.get("/api/demo/readiness", {}).get("runtime_ready"),
        "timed_smoke_ready": payloads.get("/api/demo/readiness", {}).get("timed_smoke_ready"),
        "product_3d_ready": payloads.get("/api/demo/readiness", {}).get("product_3d_ready"),
        "pipeline_preflight_ok": payloads.get("/api/pipeline/preflight", {}).get("ok"),
        "product_defaults": payloads.get("/api/pipeline/defaults", {}).get("defaults", {}),
        "comfy_stage3_workflow_valid": payloads.get("/api/comfy/workflows", {})
        .get("stage3", {})
        .get("validation", {})
        .get("ok"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _expect_keys(failures: list[str], label: str, payload: Any, keys: tuple[str, ...]) -> None:
    if not isinstance(payload, dict):
        failures.append(f"{label}: expected object")
        return
    for key in keys:
        if key not in payload:
            failures.append(f"{label}: missing key {key}")


def _expect_product_default_form_params(failures: list[str], label: str, endpoint: Any) -> None:
    signature = inspect.signature(endpoint)
    for name in ("seed", "steps", "guidance", "strength", "max_resolution"):
        param = signature.parameters.get(name)
        if param is None:
            failures.append(f"{label}: missing form parameter {name}")
            continue
        default = param.default
        default_value = getattr(default, "default", None)
        if default_value is not None:
            failures.append(f"{label}: {name} should default to None and merge product_pipeline")


def _expect_no_product_stage_mode_params(failures: list[str], label: str, endpoint: Any) -> None:
    signature = inspect.signature(endpoint)
    forbidden = (
        "backend_mode",
        "include_stage35",
        "include_stage4",
        "include_stage5",
        "stage35_mode",
        "stage35_backend_mode",
        "stage4_backend_mode",
        "stage5_backend_mode",
    )
    for name in forbidden:
        if name in signature.parameters:
            failures.append(f"{label}: product Generate must not expose stage/mode form parameter {name}")


if __name__ == "__main__":
    main()
