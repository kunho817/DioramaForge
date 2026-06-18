from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from .config import load_config
from .experiments import build_experiment_grid, run_experiment
from .model_status import model_status_markdown
from .pipeline import DioramaPipeline, PipelineOptions
from .presets import DEFAULT_PRESET, preset_names
from .runtime import runtime_status_markdown
from .stage45 import Stage4Options, Stage5Options, build_stage4_package, build_stage5_print_package


CSS = """
.df-header { margin-bottom: 8px; }
.df-header h1 { font-size: 28px; line-height: 1.2; margin-bottom: 0; }
.df-status { font-size: 13px; }
"""


def create_app(config_path: str | Path | None = None) -> gr.Blocks:
    config = load_config(config_path)
    pipeline = DioramaPipeline(config)

    with gr.Blocks(title="DioramaForge", css=CSS) as app:
        gr.Markdown("# DioramaForge", elem_classes=["df-header"])
        runtime_box = gr.Markdown(runtime_status_markdown(), elem_classes=["df-status"])

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                input_image = gr.Image(
                    label="입력 사진",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=430,
                )
            with gr.Column(scale=4):
                style_preset = gr.Dropdown(
                    label="스타일 프리셋",
                    choices=preset_names(),
                    value=DEFAULT_PRESET,
                )
                custom_prompt = gr.Textbox(
                    label="추가 프롬프트",
                    lines=4,
                    placeholder="예: rainy evening, tiny lanterns, moss-covered stone stairs",
                )
                backend_mode = gr.State(value=config.style_engine.backend_mode)
                with gr.Row():
                    refresh_runtime = gr.Button("Runtime 새로고침")
                    refresh_models = gr.Button("모델 상태 새로고침")
                    run_button = gr.Button("변환 실행", variant="primary")

        with gr.Accordion("고급 옵션", open=False):
            with gr.Row():
                seed = gr.Number(label="Seed (-1은 자동)", value=-1, precision=0)
                max_resolution = gr.Dropdown(
                    label="최대 해상도",
                    choices=[256, 512, 768, 1024],
                    value=512,
                )
            with gr.Row():
                steps = gr.Slider(label="Steps", minimum=4, maximum=50, value=4, step=1)
                guidance = gr.Slider(label="Guidance", minimum=0.0, maximum=30.0, value=3.5, step=0.5)
                strength = gr.Slider(
                    label="변환 강도 (낮을수록 원본 보존)",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.55,
                    step=0.02,
                )

        with gr.Accordion("로컬 모델 상태", open=False):
            model_status_box = gr.Markdown(model_status_markdown(config))
            gr.Markdown(
                "FLUX.1 Depth는 Hugging Face 라이선스 동의와 토큰이 필요할 수 있습니다. "
                "`scripts/setup_windows.ps1` 또는 `scripts/download_models.py`로 프로젝트 캐시에 받을 수 있습니다."
            )

        with gr.Row():
            original_output = gr.Image(label="원본", type="pil", height=320)
            depth_output = gr.Image(label="Depth", type="pil", height=320)
        with gr.Row():
            mask_output = gr.Image(label="Mask Overlay", type="pil", height=320)
            region_output = gr.Image(label="Region Overlay", type="pil", height=320)
        with gr.Row():
            control_output = gr.Image(label=config.style_engine.control_label, type="pil", height=320)
            flux_output = gr.Image(label=config.style_engine.result_label, type="pil", height=320)

        run_log = gr.Textbox(label="실행 로그", lines=8, interactive=False)
        with gr.Row():
            final_file = gr.File(label="최종 이미지")
            metadata_file = gr.File(label="메타데이터")
            output_dir = gr.Textbox(label="출력 폴더", interactive=False)
        stage_run_dir = gr.Textbox(
            label="Stage 4/5 대상 실행 폴더",
            interactive=True,
            placeholder="변환 실행 후 자동 입력됩니다. 기존 outputs/runs/... 폴더도 붙여넣을 수 있습니다.",
        )

        with gr.Accordion("Legacy Experiment Utilities", open=False):
            gr.Markdown(
                "같은 입력 이미지로 여러 파라미터 조합을 실행하고, 논문 정리용 contact sheet, CSV, Markdown 보고서를 생성합니다. "
                "실제 스타일 엔진은 매우 느리므로 먼저 낮은 해상도와 낮은 steps 조합으로 확인하세요."
            )
            with gr.Row():
                experiment_seeds = gr.Textbox(label="Seeds", value="-1", placeholder="-1, 42, 123")
                experiment_steps = gr.Textbox(label="Steps", value="4", placeholder="1, 4, 8 또는 4:12:4")
            with gr.Row():
                experiment_guidances = gr.Textbox(label="Guidance 값", value="3.5", placeholder="2.5, 3.5, 5.0")
                experiment_strengths = gr.Textbox(
                    label="변환 강도",
                    value="0.45, 0.55, 0.65",
                    placeholder="0.45, 0.55, 0.65",
                )
            with gr.Row():
                experiment_max_resolution = gr.Dropdown(
                    label="실험 최대 해상도",
                    choices=[256, 512, 768, 1024],
                    value=256,
                )
                experiment_max_runs = gr.Number(label="최대 실행 수", value=6, precision=0)
                experiment_button = gr.Button("실험 실행", variant="primary")
            experiment_contact_sheet = gr.Image(label="Contact Sheet", type="pil", height=420)
            experiment_log = gr.Textbox(label="실험 로그", lines=10, interactive=False)
            with gr.Row():
                experiment_summary_file = gr.File(label="실험 CSV")
                experiment_report_file = gr.File(label="실험 보고서")
                experiment_dir = gr.Textbox(label="실험 폴더", interactive=False)

        with gr.Accordion("Stage 4/5 3D 패키지", open=False):
            gr.Markdown(
                "Stage 4는 SAM/region 단위 이미지 분할과 TRELLIS 입력 패키지를 생성합니다. "
                "Stage 5는 현재 depth-relief proxy STL과 프린트 체크리스트를 생성합니다."
            )
            stage_backend = gr.State(value="demo")
            with gr.Row():
                stage4_mesh_resolution = gr.Slider(
                    label="Stage 4 proxy mesh resolution",
                    minimum=32,
                    maximum=192,
                    value=96,
                    step=8,
                )
                stage4_max_parts = gr.Number(label="분할 최대 part 수", value=12, precision=0)
            stage4_button = gr.Button("Stage 4 패키지 생성", variant="primary")
            stage4_contact_sheet = gr.Image(label="Stage 4 Part Contact Sheet", type="pil", height=360)
            stage4_log = gr.Textbox(label="Stage 4 로그", lines=7, interactive=False)
            with gr.Row():
                stage4_manifest_file = gr.File(label="Stage 4 Manifest")
                stage4_obj_file = gr.File(label="Proxy OBJ")
                stage4_dir = gr.Textbox(label="Stage 4 폴더", interactive=False)

            with gr.Row():
                stage5_width_mm = gr.Number(
                    label="출력 폭 mm",
                    value=config.print.default_width_mm,
                    precision=1,
                )
                stage5_relief_height_mm = gr.Number(
                    label="부조 높이 mm",
                    value=config.print.default_relief_height_mm,
                    precision=1,
                )
                stage5_base_thickness_mm = gr.Number(
                    label="베이스 두께 mm",
                    value=config.print.default_base_thickness_mm,
                    precision=1,
                )
                stage5_mesh_resolution = gr.Slider(
                    label="Stage 5 STL resolution",
                    minimum=32,
                    maximum=192,
                    value=96,
                    step=8,
                )
            stage5_button = gr.Button("Stage 5 프린트 패키지 생성", variant="primary")
            stage5_preview = gr.Image(label="Stage 5 Print Preview", type="pil", height=360)
            stage5_log = gr.Textbox(label="Stage 5 로그", lines=7, interactive=False)
            with gr.Row():
                stage5_stl_file = gr.File(label="Proxy STL")
                stage5_manifest_file = gr.File(label="Stage 5 Manifest")
                stage5_checklist_file = gr.File(label="Print Checklist")
                stage5_dir = gr.Textbox(label="Stage 5 폴더", interactive=False)

        def refresh_runtime_status() -> str:
            return runtime_status_markdown()

        def refresh_model_status() -> str:
            return model_status_markdown(config)

        def normalize_backend(value: str) -> str:
            normalized = str(value or "auto").strip().lower()
            if normalized in {"real", "real models only"}:
                return "real"
            if normalized in {"remote", "remote a100"}:
                return "remote"
            if normalized in {"comfy", "comfyui"}:
                return "comfyui"
            if normalized == "demo":
                return "demo"
            return "auto"

        def run_clicked(
            image: Image.Image | None,
            preset: str,
            prompt: str,
            backend: str,
            seed_value: Any,
            max_resolution_value: int,
            steps_value: int,
            guidance_value: float,
            strength_value: float,
        ):
            if image is None:
                return (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "입력 사진을 업로드하세요.",
                    None,
                    None,
                    "",
                    "",
                )

            messages: list[str] = []

            def status(message: str) -> None:
                messages.append(message)

            try:
                options = PipelineOptions(
                    preset_name=preset,
                    custom_prompt=prompt or "",
                    seed=int(seed_value),
                    steps=int(steps_value),
                    guidance=float(guidance_value),
                    strength=float(strength_value),
                    max_resolution=int(max_resolution_value),
                    backend_mode=normalize_backend(str(backend)),
                )
                result = pipeline.run(image, options, status=status)
                log_text = "\n".join(result.log)
                return (
                    result.input_image,
                    result.depth_image,
                    result.mask_overlay,
                    result.region_overlay,
                    result.control_image,
                    result.flux_image,
                    log_text,
                    str(result.final_image_path),
                    str(result.metadata_path),
                    str(result.run_dir),
                    str(result.run_dir),
                )
            except RuntimeError as exc:
                messages.append(str(exc))
            except Exception as exc:
                messages.append(f"예상하지 못한 오류: {exc}")

            log_text = "\n".join(messages)
            return (None, None, None, None, None, None, log_text, None, None, "", "")

        def experiment_clicked(
            image: Image.Image | None,
            preset: str,
            prompt: str,
            backend: str,
            seeds_raw: str,
            steps_raw: str,
            guidances_raw: str,
            strengths_raw: str,
            max_resolution_value: int,
            max_runs_value: Any,
        ):
            if image is None:
                return None, "입력 사진을 업로드하세요.", None, None, ""

            messages: list[str] = []

            def status(message: str) -> None:
                messages.append(message)

            try:
                grid = build_experiment_grid(
                    seeds_raw=seeds_raw,
                    steps_raw=steps_raw,
                    guidances_raw=guidances_raw,
                    strengths_raw=strengths_raw,
                    max_resolution=int(max_resolution_value),
                    max_runs=int(max_runs_value),
                )
                artifacts = run_experiment(
                    config=config,
                    pipeline=pipeline,
                    image=image,
                    preset_name=preset,
                    custom_prompt=prompt or "",
                    backend_mode=normalize_backend(str(backend)),
                    grid=grid,
                    status=status,
                )
                contact = Image.open(artifacts.contact_sheet_path).convert("RGB")
                return (
                    contact,
                    "\n".join(artifacts.log),
                    str(artifacts.summary_csv_path),
                    str(artifacts.report_md_path),
                    str(artifacts.experiment_dir),
                )
            except Exception as exc:
                messages.append(f"실험 실행 실패: {exc}")
                return None, "\n".join(messages), None, None, ""

        def stage4_clicked(
            run_dir_text: str,
            backend: str,
            mesh_resolution_value: int,
            max_parts_value: Any,
        ):
            messages: list[str] = []

            def status(message: str) -> None:
                messages.append(message)

            try:
                artifacts = build_stage4_package(
                    config=config,
                    run_dir_value=run_dir_text,
                    options=Stage4Options(
                        backend_mode=normalize_backend(str(backend)),
                        mesh_resolution=int(mesh_resolution_value),
                        max_parts=int(max_parts_value),
                    ),
                    status=status,
                )
                contact = Image.open(artifacts.contact_sheet_path).convert("RGB")
                return (
                    contact,
                    "\n".join(artifacts.log),
                    str(artifacts.manifest_path),
                    str(artifacts.obj_path),
                    str(artifacts.stage4_dir),
                )
            except Exception as exc:
                messages.append(f"Stage 4 실행 실패: {exc}")
                return None, "\n".join(messages), None, None, ""

        def stage5_clicked(
            run_dir_text: str,
            backend: str,
            width_mm_value: Any,
            relief_height_value: Any,
            base_thickness_value: Any,
            mesh_resolution_value: int,
        ):
            messages: list[str] = []

            def status(message: str) -> None:
                messages.append(message)

            try:
                artifacts = build_stage5_print_package(
                    config=config,
                    run_dir_value=run_dir_text,
                    options=Stage5Options(
                        backend_mode=normalize_backend(str(backend)),
                        width_mm=float(width_mm_value),
                        relief_height_mm=float(relief_height_value),
                        base_thickness_mm=float(base_thickness_value),
                        mesh_resolution=int(mesh_resolution_value),
                    ),
                    status=status,
                )
                preview = Image.open(artifacts.preview_path).convert("RGB")
                return (
                    preview,
                    "\n".join(artifacts.log),
                    str(artifacts.stl_path),
                    str(artifacts.manifest_path),
                    str(artifacts.checklist_path),
                    str(artifacts.stage5_dir),
                )
            except Exception as exc:
                messages.append(f"Stage 5 실행 실패: {exc}")
                return None, "\n".join(messages), None, None, None, ""

        refresh_runtime.click(refresh_runtime_status, outputs=[runtime_box])
        refresh_models.click(refresh_model_status, outputs=[model_status_box])
        run_button.click(
            run_clicked,
            inputs=[
                input_image,
                style_preset,
                custom_prompt,
                backend_mode,
                seed,
                max_resolution,
                steps,
                guidance,
                strength,
            ],
            outputs=[
                original_output,
                depth_output,
                mask_output,
                region_output,
                control_output,
                flux_output,
                run_log,
                final_file,
                metadata_file,
                output_dir,
                stage_run_dir,
            ],
        )
        experiment_button.click(
            experiment_clicked,
            inputs=[
                input_image,
                style_preset,
                custom_prompt,
                backend_mode,
                experiment_seeds,
                experiment_steps,
                experiment_guidances,
                experiment_strengths,
                experiment_max_resolution,
                experiment_max_runs,
            ],
            outputs=[
                experiment_contact_sheet,
                experiment_log,
                experiment_summary_file,
                experiment_report_file,
                experiment_dir,
            ],
        )
        stage4_button.click(
            stage4_clicked,
            inputs=[
                stage_run_dir,
                stage_backend,
                stage4_mesh_resolution,
                stage4_max_parts,
            ],
            outputs=[
                stage4_contact_sheet,
                stage4_log,
                stage4_manifest_file,
                stage4_obj_file,
                stage4_dir,
            ],
        )
        stage5_button.click(
            stage5_clicked,
            inputs=[
                stage_run_dir,
                stage_backend,
                stage5_width_mm,
                stage5_relief_height_mm,
                stage5_base_thickness_mm,
                stage5_mesh_resolution,
            ],
            outputs=[
                stage5_preview,
                stage5_log,
                stage5_stl_file,
                stage5_manifest_file,
                stage5_checklist_file,
                stage5_dir,
            ],
        )

    return app
