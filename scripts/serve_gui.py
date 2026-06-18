from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.gui import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch DioramaForge GUI and keep the process alive.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("DioramaForge GUI creating app", flush=True)
    app = create_app(config_path=ROOT / args.config)
    print("DioramaForge GUI launching server", flush=True)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        prevent_thread_lock=True,
    )
    print(f"DioramaForge GUI running on http://{args.server_name}:{args.server_port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
