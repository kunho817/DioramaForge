from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image, ImageDraw

from diorama_forge.adapters import FluxStylizer
from diorama_forge.config import load_config
from diorama_forge.presets import DEFAULT_PRESET


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal real FLUX.1 Depth smoke test.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default="outputs/flux_smoke_real.png")
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "default.json")
    image = Image.new("RGB", (args.size, args.size), (120, 160, 190))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, int(args.size * 0.58), args.size, args.size), fill=(80, 130, 90))
    draw.rectangle(
        (int(args.size * 0.22), int(args.size * 0.28), int(args.size * 0.52), int(args.size * 0.65)),
        fill=(145, 110, 80),
    )
    draw.ellipse(
        (int(args.size * 0.56), int(args.size * 0.18), int(args.size * 0.86), int(args.size * 0.51)),
        fill=(50, 120, 75),
    )
    depth = Image.linear_gradient("L").resize((args.size, args.size)).convert("RGB")

    started = time.perf_counter()
    print("FLUX smoke: start", flush=True)
    print(
        f"settings: size={args.size}, steps={args.steps}, dtype={config.flux.torch_dtype}, "
        f"quantization={config.flux.quantization}, offload={config.flux.offload_strategy}, "
        f"post_dtype={config.flux.post_load_dtype}, cpu_offload={config.flux.cpu_offload}",
        flush=True,
    )
    result = FluxStylizer(config.flux, allow_demo_fallback=False).generate(
        image=image,
        depth_image=depth,
        control_image=depth,
        prompt="a handcrafted fantasy miniature diorama, preserved layout, warm lights, detailed terrain",
        clip_prompt="fantasy miniature diorama, preserved layout",
        negative_prompt="macro texture only, missing source layout",
        preset_name=DEFAULT_PRESET,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        strength=args.strength,
        backend_mode="real",
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(out)
    print(f"backend: {result.backend}", flush=True)
    print(f"metadata: {result.metadata}", flush=True)
    print(f"saved: {out}", flush=True)
    print(f"elapsed_seconds: {round(time.perf_counter() - started, 2)}", flush=True)


if __name__ == "__main__":
    main()
