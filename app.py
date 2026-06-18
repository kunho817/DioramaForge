from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.gui import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the DioramaForge Gradio GUI.")
    parser.add_argument("--config", default="configs/default.json", help="Path to the app config JSON.")
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio server host.")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--inbrowser", action="store_true", help="Open the app in a browser on launch.")
    args = parser.parse_args()

    app = create_app(config_path=ROOT / args.config)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
    )


if __name__ == "__main__":
    main()
