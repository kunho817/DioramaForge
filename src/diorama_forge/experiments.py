from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from .config import DioramaConfig
from .pipeline import DioramaPipeline, PipelineOptions


ExperimentStatus = Callable[[str], None]


@dataclass(frozen=True)
class ExperimentGrid:
    seeds: list[int]
    steps: list[int]
    guidances: list[float]
    strengths: list[float]
    max_resolution: int
    max_runs: int


@dataclass(frozen=True)
class ExperimentArtifacts:
    experiment_dir: Path
    contact_sheet_path: Path
    summary_csv_path: Path
    report_md_path: Path
    log: list[str]


def parse_int_values(raw: str, default: list[int]) -> list[int]:
    values = _parse_values(raw, cast=int)
    return values or default


def parse_float_values(raw: str, default: list[float]) -> list[float]:
    values = _parse_values(raw, cast=float)
    return values or default


def _parse_values(raw: str, cast):
    text = (raw or "").strip()
    if not text:
        return []
    values = []
    for chunk in text.replace("\n", ",").split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" in item:
            parts = [part.strip() for part in item.split(":")]
            if len(parts) not in {2, 3}:
                raise ValueError(f"범위 형식이 올바르지 않습니다: {item}")
            start = float(parts[0])
            stop = float(parts[1])
            step = float(parts[2]) if len(parts) == 3 else 1.0
            if step == 0:
                raise ValueError(f"범위 step은 0일 수 없습니다: {item}")
            current = start
            if step > 0:
                while current <= stop + 1e-9:
                    values.append(cast(current))
                    current += step
            else:
                while current >= stop - 1e-9:
                    values.append(cast(current))
                    current += step
        else:
            values.append(cast(item))
    deduped = []
    seen = set()
    for value in values:
        key = str(value)
        if key not in seen:
            deduped.append(value)
            seen.add(key)
    return deduped


def build_experiment_grid(
    seeds_raw: str,
    steps_raw: str,
    guidances_raw: str,
    strengths_raw: str,
    max_resolution: int,
    max_runs: int,
) -> ExperimentGrid:
    return ExperimentGrid(
        seeds=parse_int_values(seeds_raw, [-1]),
        steps=parse_int_values(steps_raw, [4]),
        guidances=parse_float_values(guidances_raw, [3.5]),
        strengths=parse_float_values(strengths_raw, [0.72]),
        max_resolution=int(max_resolution),
        max_runs=max(1, int(max_runs)),
    )


def run_experiment(
    config: DioramaConfig,
    pipeline: DioramaPipeline,
    image: Image.Image,
    preset_name: str,
    custom_prompt: str,
    backend_mode: str,
    grid: ExperimentGrid,
    status: ExperimentStatus | None = None,
) -> ExperimentArtifacts:
    logs: list[str] = []

    def emit(message: str) -> None:
        logs.append(message)
        if status:
            status(message)

    combos = [
        (seed, steps, guidance, strength)
        for seed in grid.seeds
        for steps in grid.steps
        for guidance in grid.guidances
        for strength in grid.strengths
    ]
    if len(combos) > grid.max_runs:
        emit(f"조합 {len(combos)}개 중 최대 실행 수 {grid.max_runs}개만 실행합니다.")
        combos = combos[: grid.max_runs]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = config.root / "outputs" / "experiments" / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    final_images: list[tuple[Image.Image, str]] = []

    for index, (seed, steps, guidance, strength) in enumerate(combos, start=1):
        run_name = f"run_{index:03d}_seed{seed}_s{steps}_g{guidance:g}_str{strength:g}"
        run_dir = experiment_dir / run_name
        emit(f"[{index}/{len(combos)}] 실행 시작: seed={seed}, steps={steps}, guidance={guidance}, strength={strength}")
        options = PipelineOptions(
            preset_name=preset_name,
            custom_prompt=custom_prompt,
            seed=seed,
            steps=steps,
            guidance=guidance,
            strength=strength,
            max_resolution=grid.max_resolution,
            backend_mode=backend_mode,
        )
        result = pipeline.run(image, options, run_dir=run_dir)
        metadata = _read_json(result.metadata_path)
        options_meta = metadata.get("options", {})
        artifacts_meta = metadata.get("artifacts", {})
        semantic_plan = metadata.get("structure_control", {}).get("semantic_region_plan", {})
        actual_seed = options_meta.get("seed", seed)
        flux_backend = metadata.get("models", {}).get("flux", {}).get("backend", "")
        control_strategy = options_meta.get("control_strategy", "")
        region_prompt = options_meta.get("region_prompt", "")
        semantic_groups = ", ".join(
            group.get("semantic_label", "")
            for group in semantic_plan.get("groups", [])
            if group.get("semantic_label")
        )
        label = f"#{index} seed={actual_seed}\nsteps={steps} g={guidance:g} str={strength:g}"
        final_images.append((result.flux_image.copy(), label))
        rows.append(
            {
                "run_index": index,
                "run_dir": str(result.run_dir),
                "seed": actual_seed,
                "steps": steps,
                "guidance": guidance,
                "transform_strength": strength,
                "max_resolution": grid.max_resolution,
                "backend_mode": backend_mode,
                "flux_backend": flux_backend,
                "control_strategy": control_strategy,
                "region_prompt": region_prompt,
                "semantic_groups": semantic_groups,
                "region_manifest": artifacts_meta.get("region_manifest", ""),
                "region_overlay": artifacts_meta.get("region_overlay", ""),
                "final_image": str(result.final_image_path),
                "metadata": str(result.metadata_path),
                "original_preservation_score": "",
                "style_satisfaction_score": "",
                "diorama_suitability_score": "",
                "failure_type": "",
                "notes": "",
            }
        )
        emit(f"[{index}/{len(combos)}] 실행 완료")

    contact_sheet_path = experiment_dir / "contact_sheet.png"
    summary_csv_path = experiment_dir / "experiment_summary.csv"
    report_md_path = experiment_dir / "experiment_report.md"

    _save_contact_sheet(contact_sheet_path, final_images)
    _save_summary_csv(summary_csv_path, rows)
    _save_report(report_md_path, experiment_dir, preset_name, custom_prompt, rows, logs)
    emit("실험 산출물 저장 완료")

    return ExperimentArtifacts(
        experiment_dir=experiment_dir,
        contact_sheet_path=contact_sheet_path,
        summary_csv_path=summary_csv_path,
        report_md_path=report_md_path,
        log=logs,
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_contact_sheet(path: Path, images: list[tuple[Image.Image, str]]) -> None:
    if not images:
        Image.new("RGB", (640, 360), "white").save(path)
        return
    tile_w = 320
    tile_h = 320
    label_h = 72
    columns = min(3, max(1, math.ceil(math.sqrt(len(images)))))
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, (image, label) in enumerate(images):
        x = (idx % columns) * tile_w
        y = (idx // columns) * (tile_h + label_h)
        image = image.convert("RGB")
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        px = x + (tile_w - image.width) // 2
        py = y + (tile_h - image.height) // 2
        sheet.paste(image, (px, py))
        draw.rectangle((x, y + tile_h, x + tile_w, y + tile_h + label_h), fill=(238, 238, 238))
        draw.multiline_text((x + 10, y + tile_h + 10), label, fill=(20, 20, 20), font=font, spacing=4)

    sheet.save(path)


def _save_report(
    path: Path,
    experiment_dir: Path,
    preset_name: str,
    custom_prompt: str,
    rows: list[dict[str, Any]],
    logs: list[str],
) -> None:
    lines = [
        "# DioramaForge FLUX Experiment Report",
        "",
        "## Summary",
        "",
        f"- Experiment directory: `{experiment_dir}`",
        f"- Preset: `{preset_name}`",
        f"- Custom prompt: `{custom_prompt or '(none)'}`",
        f"- Run count: {len(rows)}",
        "",
        "## Runs",
        "",
        "| # | Seed | Steps | Guidance | Strength | Region Groups | Control | Final Image | Failure Type | Notes |",
        "|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {run_index} | {seed} | {steps} | {guidance} | {transform_strength} | {semantic_groups} | {control_strategy} | `{final_image}` |  |  |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Qualitative Evaluation Guide",
            "",
            "- Original preservation: whether horizon, major landforms, and semantic regions remain recognizable.",
            "- Style satisfaction: whether the fantasy diorama style is clear without overpowering the source scene.",
            "- Diorama suitability: whether the output suggests layered miniature terrain useful for later 3D work.",
            "- Semantic region validity: whether the region plan avoids false labels such as grassland being treated as water.",
            "- Failure type examples: composition collapse, semantic drift, false water, texture dominance, depth mismatch, over-stylization.",
            "",
            "## Execution Log",
            "",
        ]
    )
    lines.extend(f"- {message}" for message in logs)
    path.write_text("\n".join(lines), encoding="utf-8")
