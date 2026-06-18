from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.api import create_api_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a serverless DioramaForge live-demo readiness report."
    )
    parser.add_argument("--out-dir", default="outputs/readiness")
    parser.add_argument("--json-only", action="store_true", help="Write only the JSON report.")
    args = parser.parse_args()

    app = create_api_app(ROOT / "configs" / "default.json")
    routes = {getattr(route, "path", ""): route for route in app.routes}
    payloads = {
        "style_engine": _call_route(routes, "/api/style-engine"),
        "demo_readiness": _call_route(routes, "/api/demo/readiness"),
        "pipeline_defaults": _call_route(routes, "/api/pipeline/defaults"),
        "pipeline_preflight": _call_route(routes, "/api/pipeline/preflight"),
        "execution_policy": _call_route(routes, "/api/execution/policy"),
        "meshy_status": _call_route(routes, "/api/meshy/status"),
    }
    created_at = datetime.now().isoformat(timespec="seconds")
    report = {
        "created_at": created_at,
        "repo": str(ROOT),
        "summary": _summary(payloads),
        **payloads,
    }

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime("demo_readiness_%Y%m%d_%H%M%S")
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = None
    if not args.json_only:
        markdown_path = out_dir / f"{stem}.md"
        markdown_path.write_text(_markdown_report(report), encoding="utf-8")

    result = {
        "ok": True,
        "json": str(json_path),
        "markdown": str(markdown_path) if markdown_path else "",
        "summary": report["summary"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _call_route(routes: dict[str, Any], path: str) -> dict[str, Any]:
    if path not in routes:
        raise RuntimeError(f"Missing API route: {path}")
    payload = routes[path].endpoint()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Route did not return an object: {path}")
    return payload


def _summary(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    readiness = payloads["demo_readiness"]
    preflight = payloads["pipeline_preflight"]
    checks = readiness.get("checks", [])
    preflight_checks = preflight.get("checks", [])
    failed = [check for check in checks if not check.get("ok")]
    blocking = [check for check in preflight_checks if check.get("blocking") and not check.get("ok")]
    return {
        "ok": bool(readiness.get("ok")),
        "pipeline_preflight_ok": bool(preflight.get("ok")),
        "resolved_engine": readiness.get("resolved_engine", ""),
        "fast_path_ready": bool(readiness.get("fast_path_ready")),
        "runtime_ready": bool(readiness.get("runtime_ready")),
        "timed_smoke_ready": bool(readiness.get("timed_smoke_ready")),
        "product_3d_ready": bool(readiness.get("product_3d_ready", True)),
        "can_generate": bool(readiness.get("can_generate")),
        "failed_check_count": len(failed),
        "failed_checks": [check.get("label", check.get("id", "unknown")) for check in failed],
        "preflight_blocking_count": len(blocking),
        "preflight_blocking_checks": [check.get("label", check.get("id", "unknown")) for check in blocking],
        "next_action": readiness.get("next_action", ""),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    readiness = report["demo_readiness"]
    defaults = report["pipeline_defaults"].get("defaults", {})
    preflight = report["pipeline_preflight"]
    style_engine = report["style_engine"]
    timed = readiness.get("timed_smoke", {})
    runtime = readiness.get("runtime", {})
    product_3d = readiness.get("product_3d_backend", {})
    meshy = report.get("meshy_status", {})
    torch_status = runtime.get("torch", {})
    prepare_command = readiness.get("prepare_command", "")
    benchmark_command = readiness.get("benchmark_command", "")
    benchmark_check_command = readiness.get("benchmark_check_command", "")
    lines = [
        "# DioramaForge Demo Readiness",
        "",
        f"- Created: {report['created_at']}",
        f"- Repo: `{report['repo']}`",
        f"- Overall ready: {_yes_no(summary['ok'])}",
        f"- Resolved engine: `{summary['resolved_engine']}`",
        f"- Fast path ready: {_yes_no(summary['fast_path_ready'])}",
        f"- Generate preflight ready: {_yes_no(summary['pipeline_preflight_ok'])}",
        f"- Runtime ready: {_yes_no(readiness.get('runtime_ready'))}",
        f"- Timed smoke ready: {_yes_no(summary['timed_smoke_ready'])}",
        f"- Product 3D ready: {_yes_no(summary['product_3d_ready'])}",
        f"- Can generate: {_yes_no(summary['can_generate'])}",
        f"- Next action: {summary['next_action'] or '-'}",
        "",
        "## Product Generate Defaults",
        "",
        f"- Resolution: {defaults.get('max_resolution')} px",
        f"- Steps: {defaults.get('steps')}",
        f"- Guidance: {defaults.get('guidance')}",
        f"- Strength: {defaults.get('strength')}",
        f"- Demo budget: {defaults.get('demo_time_budget_seconds')} s",
        f"- Minimum free VRAM: {defaults.get('demo_min_free_vram_gb')} GB",
        "",
        "## Checks",
        "",
    ]
    for check in readiness.get("checks", []):
        lines.append(f"- [{_mark(check.get('ok'))}] {check.get('label')}: {check.get('detail')}")
    lines.extend(
        [
            "",
            "## Generate Preflight",
            "",
            f"- Ready to start: {_yes_no(preflight.get('ok'))}",
            f"- Next action: {_display(preflight.get('next_action'))}",
        ]
    )
    for check in preflight.get("checks", []):
        blocking = "blocking" if check.get("blocking") else "warning"
        lines.append(f"- [{_mark(check.get('ok'))}] {check.get('label')} ({blocking}): {check.get('detail')}")
    lines.extend(
        [
            "",
            "## Style Engine",
            "",
            f"- Configured: `{style_engine.get('configured_active')}`",
            f"- Active: `{style_engine.get('resolved_active')}`",
            f"- Adapter: {style_engine.get('current_adapter')}",
            f"- Current model: `{style_engine.get('current_model_id')}`",
            "",
            "## Product 3D Backend",
            "",
            f"- Required: {_yes_no(product_3d.get('required'))}",
            f"- Ready: {_yes_no(product_3d.get('ready', True))}",
            f"- Stage 4 backend: `{_display(product_3d.get('stage4_backend'))}`",
            f"- Stage 5 backend: `{_display(product_3d.get('stage5_backend'))}`",
            f"- Meshy API key env: `{_display(meshy.get('api_key_env'))}`",
            f"- Meshy API key present: {_yes_no(meshy.get('api_key_present'))}",
            f"- Detail: {_display(product_3d.get('detail'))}",
            "",
            "## Runtime",
            "",
            f"- Ready: {_yes_no(runtime.get('ready'))}",
            f"- PyTorch: `{_display(torch_status.get('version'))}`",
            f"- CUDA: {_yes_no(torch_status.get('cuda_available'))}",
            f"- Device: {_display(torch_status.get('device_name'))}",
            f"- Free VRAM: {_display(torch_status.get('free_vram_gb'))} GB",
            f"- Total VRAM: {_display(torch_status.get('total_vram_gb'))} GB",
            "",
            "## Timed Smoke",
            "",
            f"- Verified: {_yes_no(timed.get('verified'))}",
            f"- Latest report: `{_display(timed.get('latest_report'))}`",
            f"- Elapsed: {_display(timed.get('elapsed_seconds'))}",
        ]
    )
    failures = timed.get("failures") or []
    if failures:
        lines.append("- Failures:")
        for failure in failures:
            lines.append(f"  - {failure}")
    lines.extend(
        [
            "",
            "## Commands",
            "",
            f"- Prepare fast path: `{_display(prepare_command)}`",
            f"- Run timed smoke: `{_display(benchmark_command)}`",
            f"- Verify timed smoke: `{_display(benchmark_check_command)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _mark(value: Any) -> str:
    return "x" if bool(value) else " "


def _display(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


if __name__ == "__main__":
    main()
