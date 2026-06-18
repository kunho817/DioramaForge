from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diorama_forge.config import load_config
from src.diorama_forge.presets import DEFAULT_PRESET
from src.diorama_forge.remote import RemoteModelClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small remote Real Stage 3 smoke test.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--custom-prompt", default="remote A100 real model smoke test")
    parser.add_argument("--seed", type=int, default=123456789)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--max-resolution", type=int, default=256)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    config = load_config()
    client = RemoteModelClient(config.remote, config.root, config.app.output_dir)
    image = Image.open(image_path).convert("RGB")
    result = client.run_stage3(
        image=image,
        fields={
            "preset_name": args.preset,
            "custom_prompt": args.custom_prompt,
            "backend_mode": "remote",
            "seed": args.seed,
            "steps": args.steps,
            "guidance": args.guidance,
            "strength": args.strength,
            "max_resolution": args.max_resolution,
        },
        status=lambda message: print(message, flush=True),
    )
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "metadata": result.metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
