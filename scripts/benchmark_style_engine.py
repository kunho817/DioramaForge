from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image, ImageDraw

from diorama_forge.config import load_config
from diorama_forge.pipeline import DioramaPipeline, PipelineOptions
from diorama_forge.presets import DEFAULT_PRESET
from diorama_forge.style_engine import resolve_style_engine, style_engine_readiness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or benchmark the active DioramaForge Stage 3 style engine."
    )
    parser.add_argument("--run", action="store_true", help="Actually run Stage 1-3 generation.")
    parser.add_argument("--image", default="", help="Optional source image path. Uses a synthetic scene if omitted.")
    parser.add_argument(
        "--force-engine",
        choices=["auto", "flux_depth", "sdxl_depth_lightning"],
        default="auto",
        help="Override style_engine.active for this benchmark only.",
    )
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--size", type=int, default=None, help="Generation size. Defaults to product_pipeline.max_resolution.")
    parser.add_argument("--steps", type=int, default=None, help="Diffusion steps. Defaults to product_pipeline.steps.")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance. Defaults to product_pipeline.guidance.")
    parser.add_argument("--strength", type=float, default=None, help="Img2img strength. Defaults to product_pipeline.strength.")
    parser.add_argument("--seed", type=int, default=None, help="Seed. Defaults to product_pipeline.seed.")
    parser.add_argument(
        "--backend-mode",
        choices=["auto", "real", "demo", "comfyui"],
        default=None,
        help="Pipeline backend mode. Defaults to configs/default.json style_engine.backend_mode.",
    )
    parser.add_argument("--out-dir", default="outputs/benchmarks")
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "default.json")
    if args.force_engine != "auto":
        config = replace(config, style_engine=replace(config.style_engine, active=args.force_engine))
    product = config.product_pipeline
    size = int(args.size if args.size is not None else product.max_resolution)
    steps = int(args.steps if args.steps is not None else product.steps)
    guidance = float(args.guidance if args.guidance is not None else product.guidance)
    strength = float(args.strength if args.strength is not None else product.strength)
    seed = int(args.seed if args.seed is not None else product.seed)
    backend_mode = args.backend_mode or config.style_engine.backend_mode or "auto"

    resolved = "comfyui" if backend_mode.strip().lower() == "comfyui" else resolve_style_engine(config)
    readiness = style_engine_readiness(config)
    status: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "configured_engine": config.style_engine.active,
        "resolved_engine": resolved,
        "readiness": readiness,
        "settings": {
            "run": args.run,
            "image": args.image,
            "preset": args.preset,
            "size": size,
            "steps": steps,
            "guidance": guidance,
            "strength": strength,
            "seed": seed,
            "backend_mode": backend_mode,
        },
    }

    if not args.run:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        print("Dry run only. Add --run to execute real generation.", flush=True)
        return

    benchmark_dir = ROOT / args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    image = _load_or_create_image(args.image, size)
    logs: list[str] = []

    started = time.perf_counter()
    result = DioramaPipeline(config).run(
        image,
        PipelineOptions(
            preset_name=args.preset,
            custom_prompt=args.prompt,
            seed=seed,
            steps=steps,
            guidance=guidance,
            strength=strength,
            max_resolution=size,
            backend_mode=backend_mode,
        ),
        status=lambda message: logs.append(f"{round(time.perf_counter() - started, 2):>8}s {message}"),
        run_dir=benchmark_dir / "run",
    )
    elapsed = round(time.perf_counter() - started, 2)
    status.update(
        {
            "elapsed_seconds": elapsed,
            "run_dir": str(result.run_dir),
            "final_image": str(result.final_image_path),
            "metadata": str(result.metadata_path),
            "log": logs,
        }
    )
    report_path = benchmark_dir / "style_engine_benchmark.json"
    report_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"Benchmark report: {report_path}", flush=True)


def _load_or_create_image(path_value: str, size: int) -> Image.Image:
    if path_value:
        return Image.open(path_value).convert("RGB")
    image = Image.new("RGB", (size, size), (135, 175, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, int(size * 0.58), size, size), fill=(80, 135, 92))
    draw.polygon(
        [
            (0, int(size * 0.62)),
            (int(size * 0.32), int(size * 0.32)),
            (int(size * 0.62), int(size * 0.62)),
        ],
        fill=(95, 105, 110),
    )
    draw.polygon(
        [
            (int(size * 0.38), int(size * 0.62)),
            (int(size * 0.68), int(size * 0.28)),
            (size, int(size * 0.62)),
        ],
        fill=(105, 112, 118),
    )
    draw.rectangle(
        (int(size * 0.18), int(size * 0.47), int(size * 0.38), int(size * 0.72)),
        fill=(145, 105, 78),
    )
    draw.polygon(
        [
            (int(size * 0.14), int(size * 0.48)),
            (int(size * 0.28), int(size * 0.34)),
            (int(size * 0.42), int(size * 0.48)),
        ],
        fill=(95, 55, 48),
    )
    draw.ellipse(
        (int(size * 0.68), int(size * 0.1), int(size * 0.84), int(size * 0.26)),
        fill=(245, 190, 82),
    )
    return image


if __name__ == "__main__":
    main()
