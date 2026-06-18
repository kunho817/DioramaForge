from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image

from diorama_forge.config import load_config
from diorama_forge.stage45 import Stage4Options, Stage5Options, build_stage4_package, build_stage5_print_package
from diorama_forge.validation import validate_run


def main() -> None:
    config = load_config(ROOT / "configs" / "default.json")
    outputs_dir = config.root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="validation_meshy_", dir=outputs_dir) as temp_value:
        run_dir = Path(temp_value) / "run"
        _write_run(run_dir)
        valid = validate_run(config, run_dir)
        if not valid.get("ok"):
            failures.append(f"valid Meshy run did not validate: {_error_codes(valid)}")

        missing_file = run_dir / "stage5_print" / "meshy_model" / "glb.glb"
        missing_file.unlink()
        invalid = validate_run(config, run_dir)
        if invalid.get("ok"):
            failures.append("validation accepted a Meshy Stage 5 package with a missing GLB file")
        if "meshy_model_glb_missing" not in _error_codes(invalid):
            failures.append(f"missing GLB error code not reported: {_error_codes(invalid)}")

    with tempfile.TemporaryDirectory(prefix="validation_meshy_failfast_", dir=outputs_dir) as temp_value:
        run_dir = Path(temp_value) / "run"
        _write_minimal_run(run_dir)
        original_key = os.environ.get(config.meshy.api_key_env)
        try:
            os.environ.pop(config.meshy.api_key_env, None)
            try:
                build_stage4_package(config, run_dir, Stage4Options(backend_mode="meshy", mesh_resolution=32, max_parts=1))
                failures.append("Stage 4 Meshy package did not fail when MESHY_API_KEY was missing")
            except RuntimeError as exc:
                if "API key is missing" not in str(exc):
                    failures.append(f"Stage 4 Meshy missing-key failure was not specific: {exc}")
            if (run_dir / "stage4_reconstruction").exists():
                failures.append("Stage 4 created output files before Meshy missing-key preflight failed")
        finally:
            if original_key is None:
                os.environ.pop(config.meshy.api_key_env, None)
            else:
                os.environ[config.meshy.api_key_env] = original_key

    with tempfile.TemporaryDirectory(prefix="validation_meshy_stage5_", dir=outputs_dir) as temp_value:
        run_dir = Path(temp_value) / "run"
        _write_run(run_dir)
        stage4_manifest = run_dir / "stage4_reconstruction" / "reconstruction_package.json"
        manifest = json.loads(stage4_manifest.read_text(encoding="utf-8"))
        manifest["meshy"]["downloads"] = {}
        _json(stage4_manifest, manifest)
        shutil.rmtree(run_dir / "stage5_print")
        try:
            build_stage5_print_package(
                config,
                run_dir,
                Stage5Options(
                    backend_mode="meshy",
                    width_mm=120.0,
                    relief_height_mm=18.0,
                    base_thickness_mm=3.0,
                    mesh_resolution=32,
                ),
            )
            failures.append("Stage 5 Meshy package did not fail when Stage 4 downloads were missing")
        except RuntimeError as exc:
            if "no GLB, OBJ, or STL downloads" not in str(exc):
                failures.append(f"Stage 5 Meshy missing-download failure was not specific: {exc}")
        if (run_dir / "stage5_print").exists():
            failures.append("Stage 5 created output files before missing Meshy downloads preflight failed")

    summary = {
        "ok": not failures,
        "failures": failures,
        "network_used": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _write_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    regions_dir = run_dir / "regions"
    stage35_dir = run_dir / "stage35_refinement"
    stage4_dir = run_dir / "stage4_reconstruction"
    stage4_meshy_dir = stage4_dir / "meshy"
    stage5_dir = run_dir / "stage5_print"
    stage5_model_dir = stage5_dir / "meshy_model"
    for directory in (regions_dir, stage35_dir, stage4_dir, stage4_meshy_dir, stage5_dir, stage5_model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for name in ("input.png", "depth.png", "mask_overlay.png", "flux_control.png", "flux_result.png"):
        _image(run_dir / name)
    _image(stage35_dir / "stage35_reconstruction_input.png")
    _image(stage35_dir / "stage35_refined.png")
    _image(stage4_dir / "part_contact_sheet.png")
    _image(stage5_dir / "print_preview.png")
    (run_dir / "depth.npy").write_bytes(b"synthetic-depth")
    _json(regions_dir / "region_plan.json", {"groups": [{"label": "ground", "area_ratio": 0.5}], "region_prompt": "ground"})
    _json(stage35_dir / "stage35_metadata.json", {"stage": "stage35"})
    _text(stage4_dir / "heightfield_proxy.obj", "o proxy\n")
    _text(stage5_dir / "print_ready_relief_proxy.stl", "solid proxy\nendsolid proxy\n")
    _text(stage5_dir / "print_checklist.md", "# Checklist\n")

    downloads = {
        "glb": str(stage4_meshy_dir / "model_glb.glb"),
        "obj": str(stage4_meshy_dir / "model_obj.obj"),
        "stl": str(stage4_meshy_dir / "model_stl.stl"),
    }
    for path in downloads.values():
        _text(Path(path), "model-data\n")
    _json(stage4_meshy_dir / "meshy_request.json", {"image_url": "data:image/png;base64,...<redacted base64>"})
    _json(stage4_meshy_dir / "meshy_task.json", {"status": "SUCCEEDED", "model_urls": {"glb": "https://example/glb"}})
    _json(stage4_meshy_dir / "meshy_downloads.json", downloads)

    stage4_manifest = {
        "stage": "stage4_reconstruction_package",
        "backend": "meshy_image_to_3d",
        "inputs": {
            "styled_image_source": "stage35_reconstruction_input",
        },
        "parts": [{"index": 1, "semantic_label": "ground"}],
        "proxy_mesh": {"obj": str(stage4_dir / "heightfield_proxy.obj"), "mesh_resolution": 32},
        "meshy": {
            "task_id": "contract-task",
            "status": "SUCCEEDED",
            "request": str(stage4_meshy_dir / "meshy_request.json"),
            "task": str(stage4_meshy_dir / "meshy_task.json"),
            "downloads_manifest": str(stage4_meshy_dir / "meshy_downloads.json"),
            "downloads": downloads,
        },
    }
    _json(stage4_dir / "reconstruction_package.json", stage4_manifest)

    model_files = {
        "glb": str(stage5_model_dir / "glb.glb"),
        "obj": str(stage5_model_dir / "obj.obj"),
        "stl": str(stage5_model_dir / "stl.stl"),
    }
    for key, target in model_files.items():
        shutil.copy2(downloads[key], target)
    _json(
        stage5_dir / "print_package.json",
        {
            "stage": "stage5_print_package",
            "backend": "meshy_model_package",
            "outputs": {
                "stl": str(stage5_dir / "print_ready_relief_proxy.stl"),
                "preview": str(stage5_dir / "print_preview.png"),
                "checklist": str(stage5_dir / "print_checklist.md"),
                "model_files": model_files,
            },
            "mesh_stats": {"vertices_grid": [4, 4], "triangles": 12},
        },
    )

    _json(
        run_dir / "run_metadata.json",
        {
            "artifacts": {
                "input": str(run_dir / "input.png"),
                "depth_png": str(run_dir / "depth.png"),
                "depth_npy": str(run_dir / "depth.npy"),
                "mask_overlay": str(run_dir / "mask_overlay.png"),
                "region_manifest": str(regions_dir / "region_plan.json"),
                "flux_control": str(run_dir / "flux_control.png"),
                "final_image": str(run_dir / "flux_result.png"),
                "stage35_reconstruction_input": str(stage35_dir / "stage35_reconstruction_input.png"),
                "stage35_refined": str(stage35_dir / "stage35_refined.png"),
                "stage35_metadata": str(stage35_dir / "stage35_metadata.json"),
            },
            "pipeline": {
                "stage_status": {
                    "stage35": True,
                    "stage4": True,
                    "stage5": True,
                }
            },
        },
    )


def _write_minimal_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _json(run_dir / "run_metadata.json", {"artifacts": {}})


def _image(path: Path) -> None:
    Image.new("RGB", (32, 24), (140, 170, 210)).save(path)


def _text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _error_codes(payload: dict[str, Any]) -> list[str]:
    return [str(check.get("code")) for check in payload.get("checks", []) if check.get("level") == "error"]


if __name__ == "__main__":
    main()
