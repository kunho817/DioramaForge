from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.config import DioramaConfig, load_config
from diorama_forge.demo_benchmark import demo_benchmark_status, expected_demo_engine


def main() -> None:
    config = load_config(ROOT / "configs" / "default.json")
    expected_engine = expected_demo_engine(config)
    checks = [
        _case("missing_report", config, None, expected_verified=False, expected_failure="No timed"),
        _case("valid_report", config, _report(config), expected_verified=True),
        _case(
            "dry_run_report",
            config,
            _report(config, settings={"run": False}),
            expected_verified=False,
            expected_failure="dry run",
        ),
        _case(
            "wrong_engine_report",
            config,
            _report(config, resolved_engine=_wrong_engine(expected_engine)),
            expected_verified=False,
            expected_failure=expected_engine,
        ),
        _case(
            "slow_report",
            config,
            _report(config, elapsed_seconds=config.product_pipeline.demo_time_budget_seconds + 1),
            expected_verified=False,
            expected_failure="above",
        ),
        _case(
            "setting_mismatch_report",
            config,
            _report(config, settings={"size": config.product_pipeline.max_resolution + 128}),
            expected_verified=False,
            expected_failure="product default",
        ),
        _case(
            "missing_elapsed_report",
            config,
            _report(config, elapsed_seconds=None),
            expected_verified=False,
            expected_failure="elapsed_seconds",
        ),
        _case(
            "missing_artifacts_report",
            config,
            _report(config, create_artifacts=False),
            expected_verified=False,
            expected_failure="does not exist",
        ),
    ]
    failures = [check for check in checks if not check["ok"]]
    summary = {"ok": not failures, "failures": failures, "checks": checks}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _case(
    name: str,
    config: DioramaConfig,
    report: dict[str, Any] | None,
    expected_verified: bool,
    expected_failure: str = "",
) -> dict[str, Any]:
    with TemporaryDirectory(prefix="diorama_benchmark_contract_") as temp_value:
        temp_root = Path(temp_value)
        temp_config = replace(config, root=temp_root)
        if report is not None:
            if report.pop("_create_artifacts", False):
                _create_artifacts(temp_root, report)
            report_path = temp_root / "outputs" / "benchmarks" / name / "style_engine_benchmark.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        status = demo_benchmark_status(temp_config)
        failures = status.get("failures", [])
        failure_text = "\n".join(str(item) for item in failures)
        ok = status["verified"] is expected_verified
        if expected_failure:
            ok = ok and expected_failure.lower() in failure_text.lower()
        return {
            "name": name,
            "ok": ok,
            "expected_verified": expected_verified,
            "actual_verified": status["verified"],
            "failures": failures,
        }


def _report(
    config: DioramaConfig,
    resolved_engine: str | None = None,
    elapsed_seconds: float | None = 60.0,
    settings: dict[str, Any] | None = None,
    create_artifacts: bool = True,
) -> dict[str, Any]:
    merged_settings = {
        "run": True,
        "image": "",
        "preset": "Fantasy Diorama",
        "size": config.product_pipeline.max_resolution,
        "steps": config.product_pipeline.steps,
        "guidance": config.product_pipeline.guidance,
        "strength": config.product_pipeline.strength,
        "seed": config.product_pipeline.seed,
        "backend_mode": config.style_engine.backend_mode,
    }
    if settings:
        merged_settings.update(settings)
    resolved_engine = resolved_engine or expected_demo_engine(config)
    report: dict[str, Any] = {
        "created_at": "2026-06-14T00:00:00",
        "configured_engine": config.style_engine.active,
        "resolved_engine": resolved_engine,
        "run_dir": "outputs/benchmarks/contract-run/run",
        "final_image": "outputs/benchmarks/contract-run/run/flux_result.png",
        "metadata": "outputs/benchmarks/contract-run/run/run_metadata.json",
        "settings": merged_settings,
        "_create_artifacts": create_artifacts,
    }
    if elapsed_seconds is not None:
        report["elapsed_seconds"] = elapsed_seconds
    return report


def _wrong_engine(expected_engine: str) -> str:
    return "flux_depth" if expected_engine != "flux_depth" else "sdxl_depth_lightning"


def _create_artifacts(root: Path, report: dict[str, Any]) -> None:
    run_dir = root / str(report["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (root / str(report["final_image"])).write_bytes(b"not-a-real-image-contract-placeholder")
    (root / str(report["metadata"])).write_text("{}", encoding="utf-8")


if __name__ == "__main__":
    main()
