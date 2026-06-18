from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .presets import build_prompt, get_preset


@dataclass(frozen=True)
class PromptBundle:
    base_prompt: str
    clip_prompt: str
    positive_prompt: str
    negative_prompt: str
    meshy_texture_prompt: str
    strategy: str
    notes: tuple[str, ...]


STRUCTURE_PRESERVATION_PROMPT = (
    "Keep the same camera angle, horizon line, scene boundaries, foreground-to-background order, "
    "terrain slope, major silhouettes, and object count from the source image."
)

REGION_PRESERVATION_PROMPT = (
    "Restyle existing regions only. Preserve sky as sky, ground as ground, foliage as foliage, "
    "water only where water already exists, and structures only where structures already exist."
)

NEGATIVE_IMAGE_PROMPT = (
    "texture-only close-up, abstract material surface, cropped macro surface, missing horizon, "
    "missing sky, missing foreground, changed camera angle, shifted horizon, rearranged foreground and background, "
    "unrecognizable source scene, new water body, flooded grass, extra people, characters, animals, vehicles, "
    "added buildings, new focal props, duplicate focal objects, changed object count, warped perspective, "
    "melted terrain, collapsed geometry, floating geometry, noisy artifacts, text, watermark"
)

NEGATIVE_SDXL_PROMPT = (
    "low quality, worst quality, blurry, noisy, jpeg artifacts, text, watermark, signature, "
    "texture-only close-up, abstract material surface, cropped macro surface, missing horizon, missing sky, "
    "missing foreground, changed camera angle, shifted horizon, rearranged foreground and background, "
    "new water body, flooded grass, extra people, character, animal, vehicle, added building, new focal prop, "
    "duplicate object, warped perspective, melted terrain, collapsed geometry, vertical line artifacts"
)

NEGATIVE_REFINEMENT_PROMPT = (
    "changed viewpoint, shifted horizon, changed layout, new objects, missing foreground, missing sky, "
    "warped perspective, collapsed geometry, over-smoothed detail, noisy artifacts"
)

MESHY_BASE_PROMPT = (
    "physical miniature terrain and existing solid props only; clean hand-painted texture boundaries; "
    "exclude sky and painted backdrop from solid geometry; no people, vehicles, new water, or extra focal props"
)

SDXL_STYLE_TAGS = {
    "Fantasy Diorama": (
        "(fantasy miniature diorama:1.35), (tabletop scale model:1.30), (handcrafted terrain model:1.25), "
        "model kit, miniature landscape, sculpted terrain foam, hand-painted moss, grass tufts, carved stone, "
        "painted backdrop, tilt-shift macro, scenic background, no humans"
    ),
    "Animated Miniature": (
        "(cozy animated miniature diorama:1.35), (hand-painted model set:1.25), miniature landscape, "
        "soft clay-like terrain, paper foliage, rounded handmade edges, warm storybook background, no humans"
    ),
    "Medieval Village": (
        "(medieval village miniature diorama:1.35), (handcrafted scale model:1.25), balsa wood, carved stone, "
        "clay roof tiles, cobblestone texture, moss, tiny lantern accents, tabletop model, no humans"
    ),
    "Enchanted Forest": (
        "(enchanted forest miniature diorama:1.35), (tabletop fantasy terrain:1.25), mossy ground, tiny leaves, "
        "twisted roots, glowing mushroom accents, magical miniature set, scenic background, no humans"
    ),
    "Ruined City": (
        "(ruined city miniature diorama:1.35), (post-apocalyptic scale model:1.25), cracked concrete, "
        "broken masonry, rusted metal, overgrown vines, weathered handmade props, no humans"
    ),
}

SDXL_STRUCTURE_TAGS = (
    "(same composition as source image:1.25), (same camera angle:1.20), (same horizon line:1.20), "
    "(same foreground and background arrangement:1.20), preserve major silhouettes, preserve scene boundaries, "
    "preserve object count, restyle existing regions only"
)

SDXL_BASE_STYLE_PHRASES = {
    "Fantasy Diorama": (
        "a handcrafted fantasy tabletop diorama, miniature landscape scale model, sculpted foam terrain, "
        "hand-painted moss and grass tufts, carved stone accents, painted scenic backdrop, tilt-shift macro photography"
    ),
    "Animated Miniature": (
        "a cozy hand-painted animated miniature environment, tactile model set, soft clay-like terrain, "
        "paper foliage, rounded handmade edges, warm storybook miniature background"
    ),
    "Medieval Village": (
        "a handcrafted medieval village tabletop diorama, balsa wood model architecture, carved stone, "
        "clay roof tiles, cobblestone texture, mossy miniature terrain"
    ),
    "Enchanted Forest": (
        "an enchanted forest tabletop diorama, mossy miniature terrain, tiny leaves, twisted roots, "
        "small glowing mushroom accents, magical handmade model set"
    ),
    "Ruined City": (
        "a post-apocalyptic ruined city tabletop diorama, cracked concrete miniature model, broken masonry, "
        "rusted metal, overgrown vines, weathered handmade props"
    ),
}


def build_stage3_prompt_bundle(
    preset_name: str,
    custom_prompt: str | None,
    segmentation_prompt: str,
    region_plan: dict[str, Any] | None,
    text_encoder_profile: str = "flux_natural",
) -> PromptBundle:
    preset = get_preset(preset_name)
    base_prompt = build_prompt(preset_name, custom_prompt)
    region_prompt = _regional_prompt(region_plan or {})
    user_prompt = _user_prompt(custom_prompt)

    if _profile_is_illustrious(text_encoder_profile):
        positive_prompt = _build_sdxl_positive_prompt(
            preset_name=preset_name,
            custom_prompt=custom_prompt,
            segmentation_prompt=segmentation_prompt,
            region_plan=region_plan or {},
        )
        negative_prompt = NEGATIVE_SDXL_PROMPT
    elif _profile_is_sdxl_base(text_encoder_profile):
        positive_prompt = _build_sdxl_base_positive_prompt(
            preset_name=preset_name,
            custom_prompt=custom_prompt,
            segmentation_prompt=segmentation_prompt,
            region_plan=region_plan or {},
        )
        negative_prompt = NEGATIVE_SDXL_PROMPT
    else:
        positive_parts = [
            f"A composition-preserving image-to-image transformation of the source image into {preset.prompt}.",
            STRUCTURE_PRESERVATION_PROMPT,
            REGION_PRESERVATION_PROMPT,
            f"Use {preset.material_prompt}.",
            f"Lighting: {preset.lighting_prompt}.",
            f"Color palette: {preset.color_prompt}.",
            f"Camera and framing: {preset.camera_prompt}.",
            _compact_text(segmentation_prompt, 380),
            region_prompt,
            user_prompt,
            "Apply the style through materials, lighting, and surface treatment instead of replacing the scene.",
        ]
        positive_prompt = _join_sentences(positive_parts)
        negative_prompt = NEGATIVE_IMAGE_PROMPT

    clip_prompt = _join_phrases(
        [
            preset.clip_hint,
            "same composition",
            "same camera angle",
            "same horizon",
            "preserve foreground and background",
            "restyle existing regions only",
            _compact_text(custom_prompt or "", 90),
        ]
    )

    meshy_texture_prompt = build_meshy_texture_prompt(
        preset_name=preset_name,
        region_plan=region_plan or {},
    )

    return PromptBundle(
        base_prompt=base_prompt,
        clip_prompt=clip_prompt,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        meshy_texture_prompt=meshy_texture_prompt,
        strategy=f"model_specific_prompt_bundle_v1:{_normalize_profile(text_encoder_profile)}",
        notes=(
            "FLUX/T5 prompt uses natural-language scene instructions; SDXL/Illustrious prompt uses weighted tag-style instructions.",
            "CLIP prompt stays short and structure-focused.",
            "Meshy prompt is capped for texture guidance and excludes 2D camera/backdrop instructions.",
        ),
    )


def build_stage35_prompt_bundle() -> PromptBundle:
    positive_prompt = _join_sentences(
        [
            "Structure-preserving diorama refinement of the current styled image.",
            STRUCTURE_PRESERVATION_PROMPT,
            "Sharpen material boundaries and reconstruction-readable detail without changing the layout.",
            "Keep the output suitable as a Stage 4 image-to-3D handoff.",
        ]
    )
    return PromptBundle(
        base_prompt="structure-preserving diorama refinement",
        clip_prompt="structure preserving diorama refinement, same composition, clean reconstruction handoff",
        positive_prompt=positive_prompt,
        negative_prompt=NEGATIVE_REFINEMENT_PROMPT,
        meshy_texture_prompt="",
        strategy="stage35_structure_refinement_prompt_v1",
        notes=("Used by fixed Stage 3.5 ComfyUI refinement workflows.",),
    )


def build_meshy_texture_prompt(
    preset_name: str,
    region_plan: dict[str, Any] | None = None,
) -> str:
    preset = get_preset(preset_name)
    region_text = _meshy_region_summary(region_plan or {})
    prompt = _join_phrases([preset.meshy_prompt, region_text, MESHY_BASE_PROMPT])
    return _compact_text(prompt, 600)


def compact_prompt(value: str | None, max_chars: int) -> str:
    return _compact_text(value or "", max_chars)


def _build_sdxl_positive_prompt(
    preset_name: str,
    custom_prompt: str | None,
    segmentation_prompt: str,
    region_plan: dict[str, Any],
) -> str:
    style_tags = SDXL_STYLE_TAGS.get(preset_name, SDXL_STYLE_TAGS["Fantasy Diorama"])
    region_tags = _sdxl_region_tags(region_plan)
    segmentation_tags = _compact_text(segmentation_prompt, 240)
    custom = _compact_text(custom_prompt or "", 180)
    parts = [
        "score_9, score_8_up, masterpiece, best quality",
        style_tags,
        SDXL_STRUCTURE_TAGS,
        region_tags,
        segmentation_tags,
        custom,
        "style change through miniature materials, lighting, painted surfaces, readable handcrafted details",
    ]
    return _join_phrases(parts)


def _build_sdxl_base_positive_prompt(
    preset_name: str,
    custom_prompt: str | None,
    segmentation_prompt: str,
    region_plan: dict[str, Any],
) -> str:
    style = SDXL_BASE_STYLE_PHRASES.get(preset_name, SDXL_BASE_STYLE_PHRASES["Fantasy Diorama"])
    region_tags = _sdxl_region_tags(region_plan)
    custom = _compact_text(custom_prompt or "", 180)
    parts = [
        "high quality, detailed, professional miniature photography",
        f"({style}:1.25)",
        "(same composition as the source image:1.20)",
        "(same camera angle and horizon line:1.15)",
        "(same foreground, midground, and background layout:1.15)",
        "preserve major silhouettes and object count",
        "transform the scene through miniature materials, lighting, and painted surface details",
        region_tags,
        _compact_text(segmentation_prompt, 220),
        custom,
    ]
    return _join_phrases(parts)


def _sdxl_region_tags(region_plan: dict[str, Any]) -> str:
    groups = region_plan.get("groups") if isinstance(region_plan, dict) else None
    if not isinstance(groups, list) or not groups:
        return "preserve semantic regions, sky remains sky, terrain remains terrain, no new water"
    tags: list[str] = []
    for group in groups[:5]:
        if not isinstance(group, dict):
            continue
        label = str(group.get("semantic_label") or "").strip().lower()
        if label == "sky":
            tags.append("sky remains painted backdrop")
        elif label == "ground":
            tags.append("ground remains sculpted terrain, no water")
        elif label == "foliage":
            tags.append("foliage becomes miniature trees and grass")
        elif label == "water":
            tags.append("existing water only as resin water")
        elif label == "structure":
            tags.append("existing structures become miniature architecture")
    return ", ".join(tags)


def _profile_is_illustrious(profile: str) -> bool:
    normalized = _normalize_profile(profile)
    return normalized in {"sdxl_tag", "illustrious_sdxl"}


def _profile_is_sdxl_base(profile: str) -> bool:
    normalized = _normalize_profile(profile)
    return normalized in {"sdxl_base", "sdxl", "comfyui_sdxl"}


def _normalize_profile(profile: str) -> str:
    return str(profile or "flux_natural").strip().lower()


def _regional_prompt(region_plan: dict[str, Any]) -> str:
    groups = region_plan.get("groups") if isinstance(region_plan, dict) else None
    if not isinstance(groups, list) or not groups:
        return ""
    clauses: list[str] = []
    for group in groups[:5]:
        if not isinstance(group, dict):
            continue
        label = str(group.get("semantic_label") or "").strip()
        style_prompt = str(group.get("style_prompt") or "").strip()
        if not label or not style_prompt:
            continue
        area = group.get("area_ratio")
        area_text = ""
        if isinstance(area, (int, float)):
            area_text = f" ({float(area) * 100:.0f}% of image)"
        clauses.append(f"{label}{area_text}: {style_prompt}")
    if not clauses:
        return ""
    return _compact_text(
        "Semantic region guidance. " + "; ".join(clauses) + ".",
        520,
    )


def _meshy_region_summary(region_plan: dict[str, Any]) -> str:
    groups = region_plan.get("groups") if isinstance(region_plan, dict) else None
    if not isinstance(groups, list):
        return ""
    labels: list[str] = []
    for group in groups[:6]:
        if not isinstance(group, dict):
            continue
        label = str(group.get("semantic_label") or "").strip().lower()
        if label and label != "sky" and label not in labels:
            labels.append(label)
    if not labels:
        return ""
    return "visible solid regions: " + ", ".join(labels)


def _user_prompt(custom_prompt: str | None) -> str:
    custom = _compact_text(custom_prompt or "", 280)
    if not custom:
        return ""
    return f"User style note: {custom}."


def _join_sentences(parts: list[str]) -> str:
    cleaned: list[str] = []
    for part in parts:
        text = _normalize_spaces(part)
        if not text:
            continue
        if text[-1] not in ".!?":
            text += "."
        cleaned.append(text)
    return " ".join(cleaned)


def _join_phrases(parts: list[str]) -> str:
    return ", ".join(_normalize_spaces(part).strip(" ,.") for part in parts if _normalize_spaces(part))


def _compact_text(value: str, max_chars: int) -> str:
    text = _normalize_spaces(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    shortened = text[: max_chars - 1].rstrip()
    boundary = max(shortened.rfind("."), shortened.rfind(";"), shortened.rfind(","))
    if boundary >= max_chars * 0.55:
        shortened = shortened[:boundary].rstrip()
    return shortened.rstrip(" ,.;") + "."


def _normalize_spaces(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
