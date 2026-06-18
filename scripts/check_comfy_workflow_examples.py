from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.comfy_workflow import validate_comfy_workflow


EXAMPLES = (
    ("stage3", ROOT / "workflows" / "comfy" / "examples" / "stage3_sdxl_depth_img2img_api.example.json"),
)


def main() -> None:
    items = [_validate(stage, path) for stage, path in EXAMPLES]
    failures = [item for item in items if not item["ok"]]
    print(json.dumps({"ok": not failures, "failures": failures, "examples": items}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _validate(stage: str, path: Path) -> dict[str, Any]:
    validation = validate_comfy_workflow(path, stage)
    return {
        "stage": stage,
        "path": str(path),
        "ok": bool(validation.get("ok")),
        "errors": validation.get("errors", []),
        "warnings": validation.get("warnings", []),
        "placeholders_found": validation.get("placeholders_found", []),
        "output_node_candidates": validation.get("output_node_candidates", []),
    }


if __name__ == "__main__":
    main()
