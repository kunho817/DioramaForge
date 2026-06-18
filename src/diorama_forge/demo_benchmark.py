from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DioramaConfig


DEFAULT_DEMO_ENGINE = "sdxl_depth_lightning"


def expected_demo_engine(config: DioramaConfig) -> str:
    backend_mode = str(config.style_engine.backend_mode or "auto").strip().lower()
    if backend_mode == "comfyui":
        return "comfyui"
    return DEFAULT_DEMO_ENGINE


def demo_benchmark_status(config: DioramaConfig) -> dict[str, Any]:
    budget = int(config.product_pipeline.demo_time_budget_seconds)
    expected_engine = expected_demo_engine(config)
    report_path = _latest_benchmark_report(config.root)
    base: dict[str, Any] = {
        "verified": False,
        "max_seconds": budget,
        "expected_engine": expected_engine,
        "latest_report": str(report_path) if report_path else "",
        "created_at": "",
        "elapsed_seconds": None,
        "resolved_engine": "",
        "run_executed": False,
        "settings_match_product": False,
        "failures": [],
    }
    if report_path is None:
        base["failures"].append("No timed style-engine benchmark report was found.")
        return base

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        base["failures"].append(f"Could not read benchmark report: {exc}")
        return base

    settings = report.get("settings", {})
    elapsed = report.get("elapsed_seconds")
    failures: list[str] = []
    run_executed = bool(settings.get("run"))
    settings_failures = _product_setting_failures(config, settings)
    artifact_failures = _artifact_failures(config.root, report)

    if not run_executed:
        failures.append("Benchmark report is a dry run; run a timed benchmark with --run.")
    if report.get("resolved_engine") != expected_engine:
        failures.append(
            f"Benchmark resolved to {report.get('resolved_engine') or 'unknown'}, not {expected_engine}."
        )
    if not isinstance(elapsed, (int, float)):
        failures.append("Benchmark report has no numeric elapsed_seconds.")
    elif float(elapsed) > budget:
        failures.append(f"Benchmark took {elapsed}s, above the {budget}s demo budget.")
    failures.extend(settings_failures)
    failures.extend(artifact_failures)

    return {
        **base,
        "verified": not failures,
        "created_at": str(report.get("created_at", "")),
        "elapsed_seconds": elapsed,
        "resolved_engine": str(report.get("resolved_engine", "")),
        "run_executed": run_executed,
        "settings_match_product": not settings_failures,
        "artifacts_exist": not artifact_failures,
        "settings": settings,
        "failures": failures,
    }


def _latest_benchmark_report(root: Path) -> Path | None:
    benchmark_root = root / "outputs" / "benchmarks"
    if not benchmark_root.exists():
        return None
    reports = [
        path
        for path in benchmark_root.rglob("style_engine_benchmark.json")
        if path.is_file()
    ]
    if not reports:
        return None
    return max(reports, key=lambda path: path.stat().st_mtime)


def _product_setting_failures(config: DioramaConfig, settings: dict[str, Any]) -> list[str]:
    product = config.product_pipeline
    expected = {
        "size": product.max_resolution,
        "steps": product.steps,
        "guidance": product.guidance,
        "strength": product.strength,
        "seed": product.seed,
        "backend_mode": config.style_engine.backend_mode,
    }
    failures: list[str] = []
    for key, expected_value in expected.items():
        actual = settings.get(key)
        if isinstance(expected_value, float):
            try:
                matches = abs(float(actual) - expected_value) < 1e-6
            except (TypeError, ValueError):
                matches = False
        else:
            matches = str(actual) == str(expected_value)
        if not matches:
            failures.append(f"Benchmark setting {key}={actual!r} does not match product default {expected_value!r}.")
    return failures


def _artifact_failures(root: Path, report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key in ("run_dir", "final_image", "metadata"):
        value = report.get(key)
        if not value:
            failures.append(f"Benchmark report is missing {key}.")
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            failures.append(f"Benchmark {key} does not exist: {path}")
            continue
        if key != "run_dir" and not path.is_file():
            failures.append(f"Benchmark {key} is not a file: {path}")
        if key == "run_dir" and not path.is_dir():
            failures.append(f"Benchmark run_dir is not a directory: {path}")
    return failures
