from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.config import load_config
from diorama_forge.demo_runtime import demo_runtime_status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check local runtime prerequisites for the DioramaForge live demo path."
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="Exit with failure when packages/CUDA/free VRAM are not ready.",
    )
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "default.json")
    status = demo_runtime_status(config)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if args.require and not status["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
