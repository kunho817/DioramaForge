from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


MASK_COLORS: tuple[tuple[int, int, int], ...] = (
    (239, 68, 68),
    (245, 158, 11),
    (34, 197, 94),
    (14, 165, 233),
    (99, 102, 241),
    (217, 70, 239),
    (20, 184, 166),
    (234, 179, 8),
)


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def resize_for_generation(image: Image.Image, max_side: int) -> Image.Image:
    image = ensure_rgb(image)
    width, height = image.size
    scale = min(max_side / max(width, height), 1.0)
    new_width = max(64, int(math.floor(width * scale / 16) * 16))
    new_height = max(64, int(math.floor(height * scale / 16) * 16))
    if (new_width, new_height) == image.size:
        return image
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros(depth.shape, dtype=np.float32)
    valid = depth[finite]
    low, high = np.percentile(valid, [1, 99])
    if high <= low:
        high = float(valid.max())
        low = float(valid.min())
    if high <= low:
        return np.zeros(depth.shape, dtype=np.float32)
    normalized = (depth - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def depth_to_image(depth: np.ndarray) -> Image.Image:
    normalized = normalize_depth(depth)
    gray = (normalized * 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def demo_depth(image: Image.Image) -> np.ndarray:
    rgb = ensure_rgb(image)
    gray = np.asarray(ImageOps.grayscale(rgb), dtype=np.float32) / 255.0
    height, width = gray.shape
    vertical = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    center_x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    perspective = 1.0 - np.clip(np.abs(center_x), 0.0, 1.0) * 0.15
    depth = 0.55 * vertical + 0.25 * (1.0 - gray) + 0.20 * perspective
    depth_image = depth_to_image(depth).filter(ImageFilter.GaussianBlur(radius=3))
    return np.asarray(ImageOps.grayscale(depth_image), dtype=np.float32) / 255.0


def mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def demo_masks(image: Image.Image, max_masks: int) -> list[dict[str, Any]]:
    gray = np.asarray(ImageOps.grayscale(ensure_rgb(image)), dtype=np.float32)
    height, width = gray.shape
    thresholds = np.percentile(gray, [20, 40, 60, 80])
    classes = np.digitize(gray, thresholds)
    masks: list[dict[str, Any]] = []

    for class_id in range(5):
        mask = classes == class_id
        if mask.sum() < width * height * 0.015:
            continue
        masks.append(
            {
                "segmentation": mask,
                "area": int(mask.sum()),
                "bbox": mask_bbox(mask),
                "predicted_iou": None,
                "stability_score": None,
                "source": "demo",
            }
        )

    # Add broad top/bottom regions so landscape photos expose sky/ground-like layers.
    top = np.zeros((height, width), dtype=bool)
    top[: max(1, height // 3), :] = True
    bottom = np.zeros((height, width), dtype=bool)
    bottom[(height * 2) // 3 :, :] = True
    for label_mask in (top, bottom):
        masks.append(
            {
                "segmentation": label_mask,
                "area": int(label_mask.sum()),
                "bbox": mask_bbox(label_mask),
                "predicted_iou": None,
                "stability_score": None,
                "source": "demo",
            }
        )

    masks.sort(key=lambda item: int(item["area"]), reverse=True)
    return masks[:max_masks]


def overlay_masks(image: Image.Image, masks: list[dict[str, Any]], alpha: float = 0.42) -> Image.Image:
    base = np.asarray(ensure_rgb(image), dtype=np.float32)
    overlay = base.copy()
    for idx, mask_data in enumerate(masks):
        mask = np.asarray(mask_data["segmentation"], dtype=bool)
        color = np.asarray(MASK_COLORS[idx % len(MASK_COLORS)], dtype=np.float32)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def flux_control_from_depth_and_masks(
    depth_image: Image.Image,
    masks: list[dict[str, Any]],
    edge_weight: float = 0.38,
    max_masks: int = 10,
) -> Image.Image:
    depth_gray = ImageOps.grayscale(depth_image)
    width, height = depth_gray.size
    depth = np.asarray(depth_gray, dtype=np.float32) / 255.0
    edge_map = np.zeros((height, width), dtype=np.float32)

    for mask_data in masks[:max_masks]:
        mask = _mask_for_size(mask_data["segmentation"], (width, height))
        area_ratio = float(mask.sum()) / float(width * height)
        if area_ratio < 0.01 or area_ratio > 0.97:
            continue
        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        edges = mask_image.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
        edge_arr = np.asarray(edges, dtype=np.float32) / 255.0
        edge_map = np.maximum(edge_map, edge_arr)

    if edge_map.max() > 0:
        edge_image = Image.fromarray((edge_map * 255).astype(np.uint8), mode="L")
        edge_map = np.asarray(edge_image.filter(ImageFilter.GaussianBlur(0.45)), dtype=np.float32) / 255.0

    weight = np.clip(edge_map * edge_weight, 0.0, 0.85)
    control = depth * (1.0 - weight) + 0.98 * weight
    return Image.fromarray(np.clip(control * 255, 0, 255).astype(np.uint8), mode="L").convert("RGB")


def segmentation_layout_summary(
    masks: list[dict[str, Any]],
    image_size: tuple[int, int],
    max_regions: int = 6,
) -> tuple[str, list[dict[str, Any]]]:
    width, height = image_size
    regions: list[dict[str, Any]] = []
    total_area = float(width * height)

    for index, mask_data in enumerate(masks):
        mask = _mask_for_size(mask_data["segmentation"], image_size)
        area_ratio = float(mask.sum()) / total_area
        if area_ratio < 0.02 or area_ratio > 0.97:
            continue
        x, y, w, h = mask_bbox(mask)
        cx = (x + w / 2.0) / width
        cy = (y + h / 2.0) / height
        regions.append(
            {
                "index": index,
                "area_ratio": round(area_ratio, 4),
                "bbox": [x, y, w, h],
                "horizontal_position": _horizontal_position(cx),
                "vertical_position": _vertical_position(cy),
                "shape": _region_shape(w, h, width, height),
            }
        )

    regions.sort(key=lambda item: item["area_ratio"], reverse=True)
    regions = regions[:max_regions]
    if not regions:
        return (
            "Preserve the original camera viewpoint, horizon line, foreground/background layer positions, and major silhouettes.",
            [],
        )

    phrases = [
        f"{region['vertical_position']} {region['horizontal_position']} {region['shape']} region"
        for region in regions
    ]
    summary = (
        "Preserve the original camera viewpoint, horizon line, foreground/background layer positions, "
        "and the segmentation layout: " + "; ".join(phrases) + "."
    )
    return summary, regions


def _mask_for_size(mask_value: Any, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    mask = np.asarray(mask_value).astype(bool)
    if mask.shape == (height, width):
        return mask
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(mask_image, dtype=np.uint8) > 0


def _horizontal_position(value: float) -> str:
    if value < 0.33:
        return "left"
    if value > 0.67:
        return "right"
    return "center"


def _vertical_position(value: float) -> str:
    if value < 0.33:
        return "upper"
    if value > 0.67:
        return "lower"
    return "middle"


def _region_shape(width: int, height: int, image_width: int, image_height: int) -> str:
    width_ratio = width / image_width
    height_ratio = height / image_height
    if width_ratio > 0.72 and height_ratio < 0.45:
        return "wide horizontal"
    if height_ratio > 0.62 and width_ratio < 0.45:
        return "tall vertical"
    if width_ratio > 0.55 and height_ratio > 0.55:
        return "broad"
    return "localized"


def save_masks(mask_dir: Path, masks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mask_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    for index, mask_data in enumerate(masks):
        mask = np.asarray(mask_data["segmentation"], dtype=bool)
        path = mask_dir / f"mask_{index:02d}.png"
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(path)
        metadata.append(
            {
                "file": str(path),
                "area": int(mask_data.get("area", int(mask.sum()))),
                "bbox": list(mask_data.get("bbox", mask_bbox(mask))),
                "predicted_iou": mask_data.get("predicted_iou"),
                "stability_score": mask_data.get("stability_score"),
                "source": mask_data.get("source", "unknown"),
            }
        )
    return metadata


def demo_flux_style(
    image: Image.Image,
    depth_image: Image.Image,
    palette: tuple[tuple[int, int, int], ...],
    strength: float,
) -> Image.Image:
    rgb = ensure_rgb(image)
    depth = np.asarray(ImageOps.grayscale(depth_image), dtype=np.float32) / 255.0
    source = np.asarray(rgb, dtype=np.float32) / 255.0
    palette_arr = np.asarray(palette, dtype=np.float32) / 255.0
    scaled = np.clip(depth * (len(palette_arr) - 1), 0, len(palette_arr) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(palette_arr) - 1)
    t = scaled[..., None] - lower[..., None]
    colorized = palette_arr[lower] * (1.0 - t) + palette_arr[upper] * t

    luminance = np.asarray(ImageOps.grayscale(rgb), dtype=np.float32)[..., None] / 255.0
    colorized = colorized * (0.72 + luminance * 0.45)
    blended = source * (1.0 - strength) + colorized * strength

    edges = ImageOps.grayscale(rgb).filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(0.6))
    edge_arr = np.asarray(edges, dtype=np.float32)[..., None] / 255.0
    blended = blended * (1.0 - edge_arr * 0.25)

    output = Image.fromarray(np.clip(blended * 255, 0, 255).astype(np.uint8), mode="RGB")
    output = ImageEnhance.Color(output).enhance(1.18)
    output = ImageEnhance.Contrast(output).enhance(1.08)
    return output
