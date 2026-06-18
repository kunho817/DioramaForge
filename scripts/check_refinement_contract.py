from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.config import load_config
from diorama_forge.image_utils import save_json
from diorama_forge.regions import build_region_plan, save_region_artifacts
from diorama_forge.stage45 import Stage4Options, build_stage4_package


def main() -> None:
    failures: list[str] = []
    image = _synthetic_landscape()
    depth = _synthetic_depth(image.size)
    masks = _synthetic_masks(image.size)
    region_plan = build_region_plan(image=image, depth_image=depth, masks=masks, preset_name="Fantasy Diorama")

    groups = {str(group.get("semantic_label")): group for group in region_plan.get("groups", [])}
    if "sky" not in groups:
        failures.append("Synthetic broad top sky mask was not classified as sky.")
    if "ground" not in groups:
        failures.append("Synthetic lower terrain mask was not classified as ground.")

    with tempfile.TemporaryDirectory(prefix="diorama_refine_contract_", dir=ROOT / "outputs") as temp_value:
        run_dir = Path(temp_value) / "run"
        run_dir.mkdir(parents=True)
        image_path = run_dir / "input.png"
        depth_path = run_dir / "depth.png"
        final_path = run_dir / "flux_result.png"
        image.save(image_path)
        depth.save(depth_path)
        image.save(final_path)

        region_artifacts = save_region_artifacts(run_dir / "regions", image, region_plan)
        save_json(
            run_dir / "run_metadata.json",
            {
                "created_at": "contract",
                "run_dir": str(run_dir),
                "options": {"preset_name": "Fantasy Diorama", "prompt": "contract diorama prompt"},
                "artifacts": {
                    "input": str(image_path),
                    "depth_png": str(depth_path),
                    "final_image": str(final_path),
                    "region_manifest": region_artifacts["manifest"],
                    "region_masks": region_artifacts["masks"],
                },
            },
        )

        config = load_config(ROOT / "configs" / "default.json")
        stage4 = build_stage4_package(
            config,
            run_dir,
            Stage4Options(backend_mode="demo", mesh_resolution=32, max_parts=8),
        )
        manifest = json.loads(stage4.manifest_path.read_text(encoding="utf-8"))
        inputs = manifest.get("inputs", {})
        if inputs.get("stage4_input_strategy") != "sky_backdrop_removed_for_3d":
            failures.append("Stage 4 did not record sky_backdrop_removed_for_3d input strategy.")

        meshy_input = Path(str(inputs.get("meshy_input", "")))
        sky_mask = Path(str(inputs.get("sky_exclusion_mask", "")))
        if not meshy_input.exists():
            failures.append("Stage 4 did not write meshy_input.png.")
        else:
            meshy_image = Image.open(meshy_input)
            if meshy_image.mode != "RGBA":
                failures.append(f"meshy_input.png should preserve alpha after sky removal, got {meshy_image.mode}.")
            elif min(meshy_image.getchannel("A").getextrema()) >= 255:
                failures.append("meshy_input.png alpha channel has no transparent sky pixels.")
        if not sky_mask.exists():
            failures.append("Stage 4 did not write sky_exclusion_mask.png.")

        sky_parts = [
            part
            for part in manifest.get("parts", [])
            if str(part.get("semantic_label", "")).lower() == "sky"
        ]
        if sky_parts and sky_parts[0].get("mesh_usage") != "reference_backdrop_not_solid_mesh":
            failures.append("Sky part mesh_usage should mark it as a non-solid reference backdrop.")

    summary = {
        "ok": not failures,
        "failures": failures,
        "checks": [
            "classifies_sky_backdrop",
            "writes_sky_removed_meshy_input",
            "marks_sky_part_as_reference",
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _synthetic_landscape(size: tuple[int, int] = (512, 288)) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, (84, 164, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, int(height * 0.58), width, height), fill=(104, 176, 55))
    draw.ellipse((70, 42, 190, 94), fill=(236, 242, 248))
    draw.ellipse((315, 30, 420, 82), fill=(241, 245, 249))
    draw.rectangle((0, int(height * 0.55), width, int(height * 0.63)), fill=(58, 92, 54))
    draw.ellipse((330, 135, 382, 205), fill=(43, 111, 44))
    draw.rectangle((353, 194, 361, 232), fill=(92, 72, 48))
    return image


def _synthetic_depth(size: tuple[int, int]) -> Image.Image:
    width, height = size
    gradient = np.tile(np.linspace(225, 35, height, dtype=np.uint8)[:, None], (1, width))
    return Image.fromarray(gradient, mode="L").convert("RGB")


def _synthetic_masks(size: tuple[int, int]) -> list[dict[str, np.ndarray]]:
    width, height = size
    sky = np.zeros((height, width), dtype=bool)
    sky[: int(height * 0.62), :] = True
    ground = np.zeros((height, width), dtype=bool)
    ground[int(height * 0.55) :, :] = True
    tree = np.zeros((height, width), dtype=bool)
    tree[int(height * 0.45) : int(height * 0.82), int(width * 0.63) : int(width * 0.76)] = True
    return [
        {"segmentation": sky},
        {"segmentation": ground},
        {"segmentation": tree},
    ]


if __name__ == "__main__":
    main()
