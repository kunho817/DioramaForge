from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diorama_forge.presets import preset_names
from diorama_forge.prompting import (
    build_meshy_texture_prompt,
    build_stage3_prompt_bundle,
    build_stage35_prompt_bundle,
)


def main() -> None:
    failures: list[str] = []
    region_plan = _sample_region_plan()
    segmentation_prompt = (
        "Preserve the original camera viewpoint, horizon line, foreground/background layer positions, "
        "and the segmentation layout: upper center wide horizontal region; lower center broad terrain region."
    )

    for preset_name in preset_names():
        bundle = build_stage3_prompt_bundle(
            preset_name=preset_name,
            custom_prompt="extra handmade miniature detail without changing the source layout",
            segmentation_prompt=segmentation_prompt,
            region_plan=region_plan,
        )
        _require(bundle.base_prompt, f"{preset_name}: base prompt is empty", failures)
        _require(bundle.positive_prompt, f"{preset_name}: positive prompt is empty", failures)
        _require(bundle.clip_prompt, f"{preset_name}: CLIP prompt is empty", failures)
        _require(bundle.negative_prompt, f"{preset_name}: negative prompt is empty", failures)
        _require(bundle.meshy_texture_prompt, f"{preset_name}: Meshy texture prompt is empty", failures)
        _contains(bundle.positive_prompt, "camera angle", f"{preset_name}: Stage 3 prompt must preserve camera", failures)
        _contains(bundle.positive_prompt, "horizon line", f"{preset_name}: Stage 3 prompt must preserve horizon", failures)
        _contains(bundle.positive_prompt, "Restyle existing regions only", f"{preset_name}: Stage 3 prompt must be region preserving", failures)
        _contains(bundle.negative_prompt, "texture-only close-up", f"{preset_name}: negative prompt must reject texture-only outputs", failures)
        _contains(bundle.negative_prompt, "new water body", f"{preset_name}: negative prompt must reject false water", failures)
        _contains(bundle.negative_prompt, "changed camera angle", f"{preset_name}: negative prompt must reject camera drift", failures)
        if len(bundle.positive_prompt) > 2200:
            failures.append(f"{preset_name}: Stage 3 prompt is too long for stable workflow use")
        if len(bundle.clip_prompt) > 260:
            failures.append(f"{preset_name}: CLIP prompt should remain concise")
        _check_meshy_prompt(preset_name, bundle.meshy_texture_prompt, failures)

        sdxl_base_bundle = build_stage3_prompt_bundle(
            preset_name=preset_name,
            custom_prompt="",
            segmentation_prompt=segmentation_prompt,
            region_plan=region_plan,
            text_encoder_profile="sdxl_base",
        )
        _contains(
            sdxl_base_bundle.positive_prompt,
            "professional miniature photography",
            f"{preset_name}: SDXL base prompt must use base-model friendly natural quality phrase",
            failures,
        )
        _contains(
            sdxl_base_bundle.positive_prompt,
            "same composition",
            f"{preset_name}: SDXL base prompt must preserve composition",
            failures,
        )
        _contains(
            sdxl_base_bundle.positive_prompt,
            "miniature",
            f"{preset_name}: SDXL base prompt must strongly encode miniature style",
            failures,
        )
        if "score_9" in sdxl_base_bundle.positive_prompt:
            failures.append(f"{preset_name}: SDXL base prompt should not use Illustrious score tags")

        illustrious_bundle = build_stage3_prompt_bundle(
            preset_name=preset_name,
            custom_prompt="",
            segmentation_prompt=segmentation_prompt,
            region_plan=region_plan,
            text_encoder_profile="illustrious_sdxl",
        )
        _contains(illustrious_bundle.positive_prompt, "score_9", f"{preset_name}: Illustrious prompt must include score tags", failures)
        _contains(
            illustrious_bundle.positive_prompt,
            "same composition",
            f"{preset_name}: Illustrious prompt must preserve composition",
            failures,
        )
        _contains(
            illustrious_bundle.positive_prompt,
            "miniature",
            f"{preset_name}: Illustrious prompt must strongly encode miniature style",
            failures,
        )
        _contains(
            illustrious_bundle.negative_prompt,
            "vertical line artifacts",
            f"{preset_name}: Illustrious negative prompt must reject observed line artifacts",
            failures,
        )

    meshy_prompt = build_meshy_texture_prompt("Fantasy Diorama", region_plan)
    _contains(meshy_prompt, "visible solid regions: ground, foliage", "Meshy region summary should exclude sky", failures)
    _check_meshy_prompt("direct Meshy builder", meshy_prompt, failures)

    stage35 = build_stage35_prompt_bundle()
    _contains(stage35.positive_prompt, "Structure-preserving", "Stage 3.5 prompt must preserve structure", failures)
    _contains(stage35.negative_prompt, "changed viewpoint", "Stage 3.5 negative prompt must reject viewpoint drift", failures)

    summary = {
        "ok": not failures,
        "failures": failures,
        "preset_count": len(preset_names()),
        "strategy": "model_specific_prompt_bundle_v1",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _sample_region_plan() -> dict[str, Any]:
    return {
        "groups": [
            {
                "semantic_label": "sky",
                "area_ratio": 0.32,
                "style_prompt": "painted fantasy backdrop sky, preserved clouds and horizon",
            },
            {
                "semantic_label": "ground",
                "area_ratio": 0.44,
                "style_prompt": "preserve the ground plane, sculpted terrain, moss, grass, no new water",
            },
            {
                "semantic_label": "foliage",
                "area_ratio": 0.16,
                "style_prompt": "miniature crafted foliage, grasses, tiny leaves",
            },
        ]
    }


def _check_meshy_prompt(label: str, prompt: str, failures: list[str]) -> None:
    if len(prompt) > 600:
        failures.append(f"{label}: Meshy texture prompt exceeds 600 characters")
    _contains(prompt, "exclude sky", f"{label}: Meshy prompt must exclude sky/backdrop geometry", failures)
    _contains(prompt, "no people", f"{label}: Meshy prompt must reject people", failures)
    forbidden = ("camera angle", "horizon line", "A composition-preserving")
    for term in forbidden:
        if term in prompt:
            failures.append(f"{label}: Meshy prompt includes 2D generation instruction: {term}")


def _contains(value: str, needle: str, message: str, failures: list[str]) -> None:
    if needle not in value:
        failures.append(message)


def _require(value: str, message: str, failures: list[str]) -> None:
    if not value.strip():
        failures.append(message)


if __name__ == "__main__":
    main()
