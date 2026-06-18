---
name: diorama-forge-local
description: Use when working inside the DioramaForge repository, especially for local-first development of the image-to-depth-to-segmentation-to-style-generation-to-3D-handoff pipeline. Applies when the user asks to continue this project, adjust its GUI/API pipeline, validate outputs, prepare local model execution, or update project docs/scripts.
---

# DioramaForge Local Project Workflow

## Current Direction

- Treat DioramaForge as a local-first project.
- The external Elice/cloud FLUX backend plan is canceled unless the user explicitly reintroduces it.
- Do not make cloud backend work the default path.
- Keep remote scripts only as optional legacy utilities.
- Local FLUX may be slow; do not start long real-model generation unless the user clearly asks to run it.
- The current product Stage 3 route is the internal ComfyUI API workflow configured by `style_engine.backend_mode=comfyui`.
- Treat FLUX as a compatibility/research baseline unless the user explicitly asks to make it the active Stage 3 workflow.
- The short-term 3D path uses Meshy AI Image to 3D for Stage 4 and Stage 5 packaging. Local TRELLIS/UltraShape implementation is deferred until after the demo.
- The desktop GUI should remain a single product flow. Do not reintroduce paper/test/quality mode switches, backend selectors, or per-stage checkboxes into the user-facing GUI.
- Product `/api/pipeline/jobs` must stay mode-free too. It should not accept Stage on/off toggles or backend override form fields; it should use the fixed local config profile.
- Product `Generate` must run `/api/pipeline/preflight` before reading the uploaded image or starting Depth/SAM work. Use this to fail quickly on missing ComfyUI workflow/server/node classes or missing Stage 4/5 Meshy requirements.

## Product Goal

DioramaForge converts a source image through this staged pipeline:

1. Stage 1: Depth Anything depth/ray estimation.
2. Stage 2: SAM/SAM2 segmentation and region planning.
3. Stage 3: depth-conditioned style transformation while preserving source layout. The product route is a model-family-neutral ComfyUI workflow at `workflows/comfy/stage3_style_api.json`.
4. Stage 3.5: structure-preserving refinement/upscale handoff.
5. Stage 4: segmentation-unit image split plus Meshy AI Image-to-3D task/package.
6. Stage 5: Meshy model package and print handoff, with proxy STL fallback for inspection.

The user cares about preserving original spatial structure while applying the selected style.

## Execution Policy

- Prefer code, GUI, API, validation, metadata, and documentation work when the user says "continue work."
- Avoid assuming "continue work" means "run experiments."
- Do not run local real FLUX/Depth/SAM jobs casually. They may take a very long time.
- For quick verification, use compile/build checks and artifact validation.
- Demo fallback runs are acceptable only when a lightweight structural smoke test is genuinely needed.
- If a real local model run is necessary, state the expected cost/risk before starting and keep parameters conservative unless the user specifies otherwise.

## Important Local Commands

Use these for non-generating verification:

```powershell
python -m compileall -q src scripts app.py api_app.py model_backend_app.py
npm run build
cargo check
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_readiness.ps1
.\.venv\Scripts\python.exe scripts\benchmark_style_engine.py
.\.venv\Scripts\python.exe scripts\check_demo_runtime.py
.\.venv\Scripts\python.exe scripts\check_demo_benchmark.py
.\.venv\Scripts\python.exe scripts\check_demo_benchmark_contract.py
.\.venv\Scripts\python.exe scripts\check_product_speed_policy.py
.\.venv\Scripts\python.exe scripts\check_env_local_contract.py
.\.venv\Scripts\python.exe scripts\write_demo_readiness_report.py
.\.venv\Scripts\python.exe scripts\check_comfy_workflows.py
.\.venv\Scripts\python.exe scripts\check_comfy_workflow_installer.py
.\.venv\Scripts\python.exe scripts\check_comfy_workflow_inspector.py
.\.venv\Scripts\python.exe scripts\check_comfy_workflow_preparer.py
.\.venv\Scripts\python.exe scripts\check_comfy_workflow_examples.py
.\.venv\Scripts\python.exe scripts\install_comfy_example_workflow.py
.\.venv\Scripts\python.exe scripts\check_comfy_node_compatibility_contract.py
.\.venv\Scripts\python.exe scripts\check_comfy_node_compatibility.py
.\.venv\Scripts\python.exe scripts\patch_comfy_stage3_models.py
.\.venv\Scripts\python.exe scripts\check_comfy_model_patch_contract.py
.\.venv\Scripts\python.exe scripts\check_meshy_contract.py
.\.venv\Scripts\python.exe scripts\check_validation_meshy_contract.py
.\.venv\Scripts\python.exe scripts\check_api_contract.py
.\.venv\Scripts\python.exe scripts\check_desktop_contract.py
```

`scripts\benchmark_style_engine.py` without `--run` is a dry readiness check. It reports the configured/resolved style engine and missing ComfyUI/FLUX/SDXL components without loading generation models.

`scripts\check_demo_runtime.py` verifies Python packages, CUDA availability, and minimum free VRAM without loading generation models.

`scripts\check_demo_benchmark.py` reads the latest timed benchmark report without running generation. It must pass with `--require` before claiming the local desktop path is ready for the configured live-demo budget, currently 240 seconds.

`scripts\check_demo_benchmark_contract.py` uses synthetic JSON reports to prove the gate rejects dry-run, wrong-engine, slow, settings-mismatched, and missing-artifact benchmark reports.

`scripts\write_demo_readiness_report.py` writes timestamped JSON/Markdown readiness reports from serverless API payloads without starting the server or loading models.

`scripts\check_comfy_workflows.py` statically validates the configured ComfyUI API workflow files. It checks API-format JSON, required placeholders, and image output nodes without starting ComfyUI or loading models.

`scripts\check_comfy_workflow_installer.py` self-tests the upload/install path with synthetic workflow JSON and never touches configured project workflow files.

`scripts\inspect_comfy_workflow.py path\to\workflow_api.json --stage stage3` helps map an exported ComfyUI API workflow to DioramaForge placeholders by listing likely LoadImage, prompt, sampler, size, and output nodes. `scripts\check_comfy_workflow_inspector.py` self-tests that inspection logic.

`scripts\prepare_comfy_workflow.py path\to\workflow_api.json workflows\comfy\stage3_style_api.json --stage stage3` auto-patches obvious Stage 3 placeholders for simple API-format exports. `scripts\check_comfy_workflow_preparer.py` self-tests that preparation logic.

`workflows\comfy\examples\stage3_sdxl_depth_img2img_api.example.json` is an editable Stage 3 starting point. It is not active until copied or installed as `workflows\comfy\stage3_style_api.json`; static model filenames must match the user's ComfyUI model folders. `scripts\check_comfy_workflow_examples.py` validates example contracts without running ComfyUI.

`scripts\install_comfy_example_workflow.py` validates the bundled Stage 3 example by default and installs it only with `--install`, using the same validation/backup path as the API.

`scripts\check_comfy_node_compatibility_contract.py` verifies the local node/model choice compatibility checker without a running ComfyUI server.

`scripts\check_comfy_node_compatibility.py` checks a running ComfyUI `/object_info` response against the node classes and static model/input choices used by the configured workflow files. Use it after installing a workflow and starting ComfyUI; readiness also reports this when the server is reachable.

`scripts\patch_comfy_stage3_models.py` lists or patches the active Stage 3 workflow's static checkpoint, ControlNet, sampler, and scheduler values so the bundled example can be matched to the user's portable ComfyUI filenames without editing JSON by hand.

`scripts\check_comfy_model_patch_contract.py` verifies that Stage 3 workflow model-field patching, dry-run behavior, backup creation, and validation work without contacting ComfyUI.

`scripts\check_meshy_contract.py` verifies Meshy payload/status contract locally without network calls or credits. Real Meshy generation requires `MESHY_API_KEY`, `meshy_ai.download_outputs=true`, and at least one of `glb`, `obj`, or `stl` in `meshy_ai.target_formats`.

`scripts\check_validation_meshy_contract.py` verifies, without network calls, that Stage 4 fails fast when Meshy is requested without a ready API key, Stage 5 fails fast when Meshy is requested without Stage 4 model downloads, and run validation rejects missing Meshy model files in Stage 4/5 packages.

`scripts\check_product_speed_policy.py` verifies that the product profile remains a single Generate flow, stays on the ComfyUI Stage 3 workflow route, keeps the live profile at or below 512 px / 4 steps / 240 seconds, and does not reintroduce user-facing quality/test/paper/backend modes.

`scripts\check_env_local_contract.py` verifies `.env.local` loading without printing secret values.

For local demo environment variables, `.env.example` can be copied to `.env.local`. The launcher, readiness script, API config loader, and serverless checks read `.env.local`; process environment variables take precedence. Do not print or commit secret values.

The desktop shell includes a Stage 3 workflow setup card. It can inspect a ComfyUI **Save (API Format)** JSON through `/api/comfy/workflows/stage3/inspect`, install a contract-ready file through `/api/comfy/workflows/stage3/install`, install the bundled starting example through `/api/comfy/workflows/stage3/install-example`, or auto-patch and install straightforward API exports through `/api/comfy/workflows/stage3/prepare-install`; the API validates the placeholder contract before replacing `workflows/comfy/stage3_style_api.json`.

`/api/comfy/model-choices` reads running ComfyUI `/object_info` and lists model-like choices such as checkpoint, ControlNet, LoRA, VAE, upscaler, sampler, and scheduler fields. Use it to edit static model filenames in workflow examples before installing them.

`configs/default.json` `product_pipeline` is the hidden desktop Generate profile. Product Generate always runs Stage 3 -> Stage 3.5 -> Stage 4 -> Stage 5. Stage 4/5 default to `meshy` for the live-demo shortcut. Change config values or the ComfyUI workflow for demo tuning; do not add GUI mode switches or request-level stage toggles.

`/api/pipeline/preflight` exposes the blocking checks used by product Generate. Timed smoke failure should remain a readiness warning, not a new GUI mode or a reason to add quality/test profile controls.

Use this when the user explicitly asks to prepare the Diffusers SDXL style engine candidate models:

```powershell
.\scripts\prepare_style_sdxl.ps1
.\scripts\prepare_style_sdxl.ps1 -Download
.\scripts\setup_windows.ps1 -DownloadStyleSdxl -HfToken "hf_..."
```

Useful local servers:

```text
API: http://127.0.0.1:8008
GUI: http://127.0.0.1:5173
```

Start both through:

```powershell
.\scripts\start_diorama_forge.bat
```

Use this only when the user explicitly wants a real timed local generation benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_style_engine.py --run --force-engine auto
```

## Key Files

- `configs/default.json`: model and execution policy defaults.
- `src/diorama_forge/api.py`: FastAPI endpoints and job wiring.
- `src/diorama_forge/pipeline.py`: Stage 1-3 local pipeline.
- `src/diorama_forge/stage35.py`: Stage 3.5 handoff/refinement.
- `src/diorama_forge/stage45.py`: Stage 4/5 package/proxy outputs.
- `src/diorama_forge/meshy.py`: Meshy AI Image-to-3D client.
- `src/diorama_forge/validation.py`: run artifact contract validation.
- `desktop/src/main.jsx`: React GUI.
- `desktop/src/styles.css`: GUI styling.
- `scripts/start_diorama_forge.bat`: local startup.
- `.env.example`: template for local demo variables; copy to ignored `.env.local`.
- `scripts/check_readiness.ps1`: readiness and policy check.
- `scripts/check_api_contract.py`: serverless API payload contract check for desktop-readiness endpoints.
- `scripts/check_desktop_contract.py`: source-level check that the GUI remains single-Generate and mode-free.
- `scripts/check_comfy_workflows.py`: static ComfyUI workflow contract check.
- `scripts/check_comfy_workflow_installer.py`: static self-test for workflow install/backup behavior.
- `scripts/inspect_comfy_workflow.py`: static workflow inspection and placeholder mapping report.
- `scripts/check_comfy_workflow_inspector.py`: static self-test for workflow inspection behavior.
- `scripts/prepare_comfy_workflow.py`: static placeholder auto-patcher for simple ComfyUI API exports.
- `scripts/check_comfy_workflow_preparer.py`: static self-test for workflow auto-patching behavior.
- `scripts/check_comfy_workflow_examples.py`: static contract check for bundled ComfyUI example workflows.
- `scripts/install_comfy_example_workflow.py`: dry-run/default and optional installer for the bundled Stage 3 ComfyUI example.
- `scripts/check_comfy_node_compatibility_contract.py`: static self-test for ComfyUI node/model choice compatibility checks.
- `scripts/check_comfy_node_compatibility.py`: checks running ComfyUI node classes and model/input choices against configured workflow files.
- `scripts/patch_comfy_stage3_models.py`: lists or patches static model filenames in the active Stage 3 workflow.
- `scripts/check_comfy_model_patch_contract.py`: static self-test for Stage 3 workflow model filename patching.
- `scripts/check_meshy_contract.py`: static Meshy API integration contract check without network calls.
- `scripts/check_validation_meshy_contract.py`: static run-validation contract for Meshy Stage 4/5 model file packaging.
- `scripts/prepare_style_sdxl.ps1`: dry-first SDXL candidate preparation; downloads only with `-Download`.
- `scripts/benchmark_style_engine.py`: dry style-engine readiness report by default; real timed generation only with `--run`.
- `scripts/check_demo_runtime.py`: verifies package/CUDA/free-VRAM readiness without loading generation models.
- `scripts/check_demo_benchmark.py`: verifies that the latest real timed benchmark used product defaults, resolved to the expected product engine, completed within the configured demo budget, and still has its output artifacts.
- `scripts/check_demo_benchmark_contract.py`: validates the timed-smoke gate logic without loading models.
- `scripts/check_product_speed_policy.py`: validates the 240-second single-Generate live profile and rejects user-facing product modes.
- `scripts/check_env_local_contract.py`: validates `.env.local` loading behavior without exposing secret values.
- `scripts/write_demo_readiness_report.py`: writes JSON/Markdown demo readiness reports without starting the API server.
- `docs/local_style_engine_speed_plan_20260614.md`: current local style-engine speed and GUI simplification plan.
- `docs/product_generate_speed_and_mode_cleanup_20260615.md`: product Generate speed strategy and mode cleanup decision record.

## GUI Direction

- GUI is the primary user interface.
- Prefer clear controls over CLI-first features.
- Keep the product GUI mode-free: one primary `Generate` action, with backend/model choices handled by internal `style_engine` configuration.
- `Generate` should execute the same full product path for every user: Stage 3, Stage 3.5, Stage 4, and Stage 5 proxy packaging.
- The hidden Generate defaults come from `/api/pipeline/defaults`, backed by `configs/default.json` `product_pipeline`.
- The Generate preflight comes from `/api/pipeline/preflight`; show it as status and block the single Generate action only when its blocking checks fail.
- The live readiness banner comes from `/api/demo/readiness`; keep it status-only, not a backend selector.
- The Stage 3 workflow setup card may install the required ComfyUI API workflow, but it must not become a backend or mode selector.
- Distinguish `fast_path_ready`, `runtime_ready`, `product_3d_ready`, and `timed_smoke_ready`. With the ComfyUI product route, the first means the ComfyUI server and required Stage 3 workflow are ready; the second means the configured runtime preflight passed; the third means Stage 4/5 requirements such as `MESHY_API_KEY` are ready when Meshy is configured; the fourth means a real timed run passed `product_pipeline.demo_time_budget_seconds` and its run/image/metadata artifacts exist.
- Readiness must distinguish static workflow contract validity, ComfyUI server reachability, and running-server node-class compatibility.
- Show intermediate artifacts: original, depth, masks, region overlay, style control, style result, Stage 3.5, Stage 4 parts, Stage 5 preview.
- Show Meshy model artifacts when Stage 4 downloads GLB/OBJ/STL outputs.
- Show validation results for loaded runs.
- Do not hide failures behind generic errors; distinguish missing file, model path, memory, engine, legacy remote, and policy failures.
- Treat `style_engine.backend_mode=comfyui` as the default product policy. Stage 3 uses `workflows/comfy/stage3_style_api.json`; the workflow can internally use FLUX, SDXL-Lightning, SDXL-Turbo/img2img, or another depth-conditioned graph. Do not claim live-demo readiness until a real timed benchmark passes the configured budget.

## Validation Direction

When improving the project without running models, prioritize:

- run folder contract checks
- metadata consistency
- artifact link integrity
- GUI run loading
- local execution policy clarity
- model path configuration
- Stage 4 segmentation-unit export quality
- Meshy task/download metadata integrity
- Stage 5 package manifest/checklist completeness

## Remote/Cloud Note

The repo still contains Elice/remote scripts from an earlier plan. Treat them as optional legacy tooling. Do not route new FLUX work to external cloud by default.
