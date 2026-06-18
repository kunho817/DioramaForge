from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "desktop" / "src" / "main.jsx"
STYLES = ROOT / "desktop" / "src" / "styles.css"

FORBIDDEN_SNIPPETS = (
    "Advanced Generation Settings",
    "Full Pipeline Stages",
    "Developer Stage Controls",
    "Real Models Only",
    "Remote A100",
    "backend selector",
    "Backend Selector",
    "<details",
    'type="checkbox"',
    "runStage3",
    "runStage35",
    "runStage4",
    "runStage5",
    'form.append("backend_mode"',
    "include_stage35",
    "include_stage4",
    "include_stage5",
    "stage35_mode",
    "stage35_backend_mode",
    "stage4_backend_mode",
    "stage5_backend_mode",
)

REQUIRED_SNIPPETS = (
    'fetchJsonQuiet("/api/demo/readiness")',
    'fetchJsonQuiet("/api/pipeline/defaults")',
    'fetchJsonQuiet("/api/pipeline/preflight")',
    'fetchJsonQuiet("/api/comfy/model-choices")',
    "LiveReadinessBanner",
    "ReadinessChecks",
    "생성 전 확인",
    "ComfyUI 모델",
    "formatComfyModelChoices",
    "formatPipelinePreflight",
    "예제 설치",
    "installExampleStage3Workflow",
    "runFullPipeline",
    "생성 시작",
)


def main() -> None:
    failures: list[str] = []
    if not MAIN.exists():
        failures.append(f"missing desktop source: {MAIN}")
        _finish(failures)
    source = MAIN.read_text(encoding="utf-8")
    styles = STYLES.read_text(encoding="utf-8") if STYLES.exists() else ""

    for snippet in REQUIRED_SNIPPETS:
        if snippet not in source:
            failures.append(f"desktop main missing required snippet: {snippet}")

    for snippet in FORBIDDEN_SNIPPETS:
        if snippet in source:
            failures.append(f"desktop main contains forbidden user-facing mode snippet: {snippet}")

    if "readiness-banner" not in styles:
        failures.append("desktop styles missing readiness-banner styles")
    if "readiness-checks" not in styles:
        failures.append("desktop styles missing readiness-checks styles")

    generate_buttons = re.findall(r"<button[^>]*>\s*(?:Generate|생성 시작)\s*</button>", source)
    if len(generate_buttons) != 1:
        failures.append(f"expected exactly one Generate button, found {len(generate_buttons)}")

    select_count = source.count("<select ")
    if select_count != 2:
        failures.append(f"expected exactly two user-facing selects (preset and recent run), found {select_count}")

    summary = {
        "ok": not failures,
        "failures": failures,
        "generate_buttons": len(generate_buttons),
        "select_count": select_count,
        "readiness_banner": "readiness-banner" in styles,
        "readiness_checks": "readiness-checks" in styles,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _finish(failures: list[str]) -> None:
    print(json.dumps({"ok": False, "failures": failures}, ensure_ascii=False, indent=2))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
