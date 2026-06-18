from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .image_utils import MASK_COLORS, mask_bbox, save_json


REGION_LABELS = ("sky", "water", "structure", "foliage", "ground", "other")

REGION_DISPLAY_NAMES = {
    "sky": "Sky / backdrop",
    "water": "Water / reflection",
    "structure": "Building / focal structure",
    "foliage": "Trees / foliage",
    "ground": "Ground / terrain",
    "other": "Other",
}

REGION_PROMPTS = {
    "sky": "painted fantasy backdrop sky, preserved clouds and horizon, reference layer rather than solid mesh",
    "water": "preserve existing water only, glossy miniature resin water where water is already visible",
    "structure": "preserve the original building silhouette, handcrafted miniature architecture, tactile roof and wood details",
    "foliage": "miniature crafted foliage, grasses, tiny leaves, fantasy diorama vegetation",
    "ground": "preserve the ground plane, sculpted terrain, moss, grass, stone edging, miniature ground materials, do not turn terrain into water",
    "other": "subtle handcrafted miniature material treatment",
}


def build_region_plan(
    image: Image.Image,
    depth_image: Image.Image,
    masks: list[dict[str, Any]],
    preset_name: str,
    max_regions: int = 12,
) -> dict[str, Any]:
    width, height = image.size
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    depth = np.asarray(depth_image.convert("L").resize((width, height)), dtype=np.float32) / 255.0
    total_area = float(width * height)

    regions: list[dict[str, Any]] = []
    combined_masks = {label: np.zeros((height, width), dtype=bool) for label in REGION_LABELS}

    for index, mask_data in enumerate(masks[:max_regions]):
        mask = _mask_for_size(mask_data["segmentation"], (width, height))
        area = int(mask.sum())
        area_ratio = area / total_area
        if area_ratio < 0.01 or area_ratio > 0.98:
            continue

        bbox = mask_bbox(mask)
        features = _region_features(rgb, depth, mask, bbox, width, height)
        label = _classify_region(features)
        combined_masks[label] |= mask
        regions.append(
            {
                "source_mask_index": index,
                "semantic_label": label,
                "semantic_name": REGION_DISPLAY_NAMES[label],
                "area": area,
                "area_ratio": round(area_ratio, 4),
                "bbox": bbox,
                "center": [round(features["cx"], 4), round(features["cy"], 4)],
                "mean_rgb": [round(value, 2) for value in features["mean_rgb"]],
                "mean_depth": round(features["mean_depth"], 4),
                "style_prompt": REGION_PROMPTS[label],
            }
        )

    grouped_regions = []
    for label in REGION_LABELS:
        mask = combined_masks[label]
        if not mask.any():
            continue
        grouped_regions.append(
            {
                "semantic_label": label,
                "semantic_name": REGION_DISPLAY_NAMES[label],
                "area": int(mask.sum()),
                "area_ratio": round(float(mask.sum()) / total_area, 4),
                "bbox": mask_bbox(mask),
                "style_prompt": REGION_PROMPTS[label],
            }
        )

    grouped_regions.sort(key=lambda item: item["area"], reverse=True)
    prompt = build_region_style_prompt(grouped_regions)
    return {
        "strategy": "sam_semantic_region_plan",
        "preset_name": preset_name,
        "note": (
            "Stage 3 uses this as a semantic region plan and prompt guide. "
            "Stage 4 will use SAM units for actual image/part splitting."
        ),
        "region_prompt": prompt,
        "regions": regions,
        "groups": grouped_regions,
        "_combined_masks": combined_masks,
    }


def build_region_style_prompt(grouped_regions: list[dict[str, Any]]) -> str:
    active = [region for region in grouped_regions if region["semantic_label"] != "other"]
    if not active:
        return "Apply the diorama style as material and lighting changes while preserving each original region."
    clauses = [
        f"{region['semantic_label']}: {region['style_prompt']}"
        for region in active[:5]
    ]
    return (
        "Apply style regionally without replacing scene objects. "
        + "; ".join(clauses)
        + "."
    )


def save_region_artifacts(
    region_dir: Path,
    image: Image.Image,
    region_plan: dict[str, Any],
) -> dict[str, Any]:
    region_dir.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in region_plan.items() if not key.startswith("_")}
    manifest_path = region_dir / "region_plan.json"
    save_json(manifest_path, serializable)

    mask_paths: dict[str, str] = {}
    combined_masks = region_plan.get("_combined_masks", {})
    for label, mask in combined_masks.items():
        if not np.asarray(mask).any():
            continue
        path = region_dir / f"{label}_mask.png"
        Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8), mode="L").save(path)
        mask_paths[label] = str(path)

    overlay_path = region_dir / "region_overlay.png"
    region_overlay(image, combined_masks).save(overlay_path)
    return {
        "manifest": str(manifest_path),
        "overlay": str(overlay_path),
        "masks": mask_paths,
    }


def region_overlay(image: Image.Image, combined_masks: dict[str, np.ndarray], alpha: float = 0.38) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    for index, label in enumerate(REGION_LABELS):
        mask = np.asarray(combined_masks.get(label, np.zeros(base.shape[:2], dtype=bool)), dtype=bool)
        if not mask.any():
            continue
        color = np.asarray(MASK_COLORS[index % len(MASK_COLORS)], dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def _region_features(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    bbox: list[int],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    x, y, width, height = bbox
    mean_rgb = rgb[mask].mean(axis=0)
    cx = (x + width / 2.0) / image_width
    cy = (y + height / 2.0) / image_height
    width_ratio = width / image_width
    height_ratio = height / image_height
    saturation = float((rgb[mask].max(axis=1) - rgb[mask].min(axis=1)).mean() / 255.0)
    return {
        "mean_rgb": mean_rgb.tolist(),
        "mean_depth": float(depth[mask].mean()),
        "cx": float(cx),
        "cy": float(cy),
        "width_ratio": float(width_ratio),
        "height_ratio": float(height_ratio),
        "saturation": saturation,
    }


def _classify_region(features: dict[str, Any]) -> str:
    red, green, blue = features["mean_rgb"]
    cy = features["cy"]
    width_ratio = features["width_ratio"]
    height_ratio = features["height_ratio"]
    saturation = features["saturation"]

    if _is_sky_region(red, green, blue, saturation, cy, width_ratio, height_ratio):
        return "sky"
    if cy > 0.50 and width_ratio > 0.45 and height_ratio < 0.58:
        return "ground"
    if _is_foliage_color(red, green, blue, saturation) and cy < 0.82:
        return "foliage"
    if (
        cy > 0.55
        and width_ratio > 0.35
        and height_ratio < 0.5
        and _is_water_color(red, green, blue, saturation)
    ):
        return "water"
    if 0.25 < cy < 0.72 and width_ratio < 0.55 and height_ratio < 0.55:
        return "structure"
    if cy > 0.45:
        return "ground"
    return "other"


def _is_sky_region(
    red: float,
    green: float,
    blue: float,
    saturation: float,
    cy: float,
    width_ratio: float,
    height_ratio: float,
) -> bool:
    top_band = cy < 0.48
    broad_backdrop = cy < 0.56 and width_ratio > 0.35 and height_ratio > 0.08
    if not (top_band or broad_backdrop):
        return False
    blue_sky = blue > green * 1.04 and blue > red * 1.16 and green > red * 1.02 and saturation > 0.08
    bright_cloud = (
        red > 145
        and green > 150
        and blue > 160
        and blue >= red * 0.92
        and abs(red - green) < 80
        and saturation < 0.42
    )
    return blue_sky or bright_cloud


def _is_foliage_color(red: float, green: float, blue: float, saturation: float) -> bool:
    warm_tree = red > green * 0.9 and red > blue * 1.15 and saturation > 0.12
    green_tree = green > red * 0.85 and green > blue * 0.9 and saturation > 0.10
    return warm_tree or green_tree


def _is_water_color(red: float, green: float, blue: float, saturation: float) -> bool:
    # Be conservative: false water labels strongly push FLUX toward rivers/ponds.
    blue_water = blue > red * 1.12 and blue > green * 0.98 and saturation > 0.08
    cyan_water = green > red * 1.08 and blue > red * 1.08 and saturation > 0.10
    return blue_water or cyan_water


def _mask_for_size(mask_value: Any, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    mask = np.asarray(mask_value).astype(bool)
    if mask.shape == (height, width):
        return mask
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(mask_image, dtype=np.uint8) > 0
