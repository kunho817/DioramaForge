from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.config import load_config
from diorama_forge.model_status import model_status_markdown
from diorama_forge.runtime import runtime_status_markdown


def main() -> None:
    config = load_config(ROOT / "configs" / "default.json")
    print(runtime_status_markdown())
    print()
    print(model_status_markdown(config))


if __name__ == "__main__":
    main()
