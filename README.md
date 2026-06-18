# DioramaForge

Local pipeline for converting a single everyday scene photo into a fantasy diorama-style image and downstream 3D handoff package through:

`input image -> depth estimation -> segmentation -> source image + depth control -> style-engine transform`

The current local MVP runs through Stage 5 packaging scaffolding: Stage 3 style transformation, Stage 3.5 structure-preserving upscale handoff images, Stage 4 segmentation-unit packaging, a depth-relief proxy OBJ, and a print proxy STL for validating the 3D handoff before TRELLIS/UltraShape adapters are connected.

## Quick Start: API + Desktop Shell

The preferred GUI direction is now:

`Python FastAPI server -> React UI -> Tauri desktop shell`

Use the batch launcher from the project root:

```bat
scripts\start_diorama_forge.bat
```

It starts the local API server at `http://127.0.0.1:8008`, installs desktop dependencies when `desktop/node_modules` is missing, then starts the Tauri desktop shell. The API also exposes OpenAPI docs at `http://127.0.0.1:8008/docs` when the server is running.

The desktop shell exposes one primary `Generate` flow. It always runs the same product path: Stage 3 style generation, Stage 3.5 handoff/refinement, Stage 4 segmentation-unit packaging, and Stage 5 print proxy packaging. Backend choices such as FLUX, ComfyUI, demo fallback, or SDXL-depth engines are internal configuration details, not user-facing operation modes. Long style-generation jobs can be monitored through `/api/jobs/{job_id}` instead of blocking the UI until completion.

When the ComfyUI Stage 3 workflow is missing, the desktop shell shows a Stage 3 workflow setup card. Export a ComfyUI graph with **Save (API Format)**, add the placeholders described in `workflows/comfy/README.md`, then install that JSON through the setup card. The file is validated before it replaces `workflows/comfy/stage3_style_api.json`.

The setup card also includes `Install Example`, which installs the bundled SDXL-depth img2img API example as the active Stage 3 workflow. This is a starting point, not a guarantee that local ComfyUI has the same checkpoint or ControlNet filenames. Use the `ComfyUI Models` status card and node/model compatibility check to align those static filenames.

Before product `Generate` starts, the API runs a fast preflight through `/api/pipeline/preflight`. It checks the fixed product contract, the configured Stage 3 image backend, and the Stage 4/5 3D backend requirements such as `MESHY_API_KEY`. This prevents a live demo from spending minutes on Depth/SAM before discovering that ComfyUI or Meshy is not ready. Timed smoke verification remains a readiness warning, not a user-facing mode.

For local demonstration variables, copy `.env.example` to `.env.local`. The launcher, readiness script, API, and serverless contract checks read `.env.local` without printing secret values; process environment variables still take precedence. Use it for values such as:

```text
COMFYUI_DIR=D:\ComfyUI
MESHY_API_KEY=msy_your_key_here
```

The GUI also includes a recent-run loader. Existing folders under `outputs/runs/` can be restored from the sidebar for comparison, validation, and artifact path review.

If you want the launcher to start the internal ComfyUI Stage 3 image backend too, set `COMFYUI_DIR` before running the batch file:

```bat
set COMFYUI_DIR=D:\ComfyUI
scripts\start_diorama_forge.bat
```

If `COMFYUI_DIR` is not set, DioramaForge reuses an already running ComfyUI server at `http://127.0.0.1:8188`.

The legacy Elice/remote tunnel is not started by default. Set `DIORAMA_ENABLE_REMOTE_TUNNEL=1` only if you are intentionally testing the old remote path.

Run the API alone:

```powershell
.\.venv\Scripts\python.exe api_app.py --host 127.0.0.1 --port 8008
```

Run the React/Tauri desktop shell alone:

```powershell
cd desktop
npm run tauri:dev
```

## Legacy Gradio GUI

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py --server-port 7860 --inbrowser
```

Open `http://127.0.0.1:7860` if the browser does not open automatically.

## Style Engine And Model Behavior

The GUI does not expose execution modes. `configs/default.json` defines the internal style engine:

```json
"style_engine": {
  "active": "auto",
  "target": "comfyui_stage3_style",
  "backend_mode": "comfyui"
}
```

`backend_mode=comfyui` keeps the GUI mode-free while routing Stage 3 image generation through a reusable ComfyUI API workflow. The workflow file is configured as `workflows/comfy/stage3_style_api.json`. That graph can use FLUX Depth, SDXL-Lightning depth, SDXL-Turbo/img2img, or another depth-conditioned model without changing the desktop GUI or API contract.

The product GUI does not expose paper/test/quality modes. Those settings are fixed in the desktop flow so demonstrations and normal operation use the same functional path. Research comparisons and speed measurements should use scripts or config changes instead of adding mode switches back to the GUI.

For speed, replace the internal ComfyUI graph rather than adding GUI model modes. The live-demo target is a warm, 512 px, low-step depth-conditioned workflow such as SDXL-Lightning depth or an equivalent fast img2img/depth graph. The current readiness budget is 240 seconds, leaving margin inside a 5-minute demonstration. FLUX.1 Depth remains a research-quality baseline unless a real timed benchmark proves it fits the configured demo budget.

The fixed `Generate` profile lives in `configs/default.json` under `product_pipeline`:

```json
"product_pipeline": {
  "steps": 4,
  "guidance": 3.5,
  "strength": 0.55,
  "max_resolution": 512,
  "demo_time_budget_seconds": 240,
  "demo_min_free_vram_gb": 6.0,
  "stage35_backend_mode": "demo",
  "stage4_backend_mode": "meshy",
  "stage5_backend_mode": "meshy"
}
```

The desktop app reads the user-safe part of this profile through `/api/pipeline/defaults`. Product `Generate` always runs the fixed Stage 3 -> Stage 3.5 -> Stage 4 -> Stage 5 contract; it does not accept stage toggles, backend overrides, or quality/test/paper modes from the GUI. Change the internal config or the ComfyUI workflow for local demonstration tuning without adding user-facing modes back to the GUI.

`GET /api/pipeline/preflight` exposes the blocking checks used by product `Generate`. `POST /api/pipeline/jobs` returns HTTP 409 with the same preflight payload if the fixed product profile cannot start safely.

See `docs/product_generate_speed_and_mode_cleanup_20260615.md` for the current live-demo speed and mode-cleanup decision record.

To force-test the Diffusers SDXL-Lightning candidate instead of the ComfyUI product path, change:

```json
"style_engine": {
  "active": "sdxl_depth_lightning",
  "target": "sdxl_depth_lightning",
  "backend_mode": "auto"
}
```

The SDXL engine uses:

```json
"sdxl_depth_lightning": {
  "base_model_id": "stabilityai/stable-diffusion-xl-base-1.0",
  "controlnet_model_id": "diffusers/controlnet-depth-sdxl-1.0",
  "lora_model_id": "ByteDance/SDXL-Lightning",
  "lora_weight_name": "sdxl_lightning_4step_lora.safetensors"
}
```

- Depth: tries Depth Anything 3 through `depth_anything_3`, then a Transformers depth pipeline, then demo depth.
- Segmentation: tries SAM 2 through `sam2`, then a Transformers mask-generation pipeline, then demo masks.
- Style engine: the configured product path uses the original image plus depth control through the ComfyUI Stage 3 workflow. FLUX.1 Depth remains available as a Diffusers compatibility path and research baseline. The optional SDXL ControlNet Depth plus SDXL-Lightning path is a faster local candidate. SAM masks are saved, previewed, and converted into a conservative semantic region plan for prompt guidance, but they are not yet direct per-region generation controls.
- ComfyUI: internal Stage 3 image backend. Export ComfyUI workflows with "Save (API Format)" and place them under `workflows/comfy/`. The required Stage 3 filename is `stage3_style_api.json`. Replacing FLUX with SDXL-Lightning, SDXL-Turbo/img2img, or another depth-conditioned graph should happen inside this workflow contract, not through GUI modes.
- Stage 3.5: provides a structure-preserving upscale/refinement handoff from the Stage 3 style result and depth map. The live product default uses the deterministic handoff/proxy path to avoid a second heavy generation pass during short demonstrations. It writes a visual upscale, a reconstruction handoff image, and metadata. Stage 4/5 use `stage35_reconstruction_input.png` first when it exists.
- Stage 4/5 3D: the live-demo shortcut uses Meshy AI Image to 3D. Set `MESHY_API_KEY`, keep `meshy_ai.download_outputs=true`, and include at least one of `glb`, `obj`, or `stl` in `meshy_ai.target_formats` before running product Generate with `stage4_backend_mode=meshy`. Stage 4 still writes segmentation-unit crops and a proxy OBJ for validation; for Meshy it also writes a sky/backdrop-removed `meshy_input.png` so the 3D task focuses on physical terrain, trees, structures, and props. Meshy GLB/OBJ/STL downloads are recorded when the API task succeeds. Stage 5 packages those Meshy model files and keeps a proxy STL fallback for print inspection. When Meshy is explicitly requested, missing API keys, disabled downloads, missing model output formats, or missing Stage 4 Meshy downloads fail before writing misleading package outputs. `scripts\check_validation_meshy_contract.py` verifies this behavior without network calls.

## Legacy Remote Backend

The Elice/A100 backend path is kept only as legacy tooling. It is not the default product route and is not shown in the desktop GUI.

Run this on the GPU cloud machine:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional-models.txt
./scripts/start_model_backend.sh
```

Or use the setup helper:

```bash
export DIORAMA_PERSIST_DIR=/mnt/diorama
export DIORAMA_REMOTE_WORKDIR=/tmp/diorama-runs
./scripts/setup_elice_cloud.sh
source .venv/bin/activate
source .env.remote
export HF_TOKEN=<set-this-on-server>
./scripts/start_model_backend.sh
```

The remote backend writes temporary Stage 3/3.5/4 runs under `DIORAMA_REMOTE_WORKDIR` and deletes each run directory immediately after the result zip has been sent back to the local API. Persistent storage should hold only model/cache data unless you disable `remote_backend.cleanup_after_response`.

For a private tunnel from your PC:

```powershell
ssh -L 9008:127.0.0.1:9008 user@ELICE_HOST
```

For the current Elice tunnel format, the local helper can be used after placing the pem file under `key/`:

```powershell
.\scripts\connect_elice_tunnel.ps1
```

Then set local `configs/default.json`:

```json
"remote_backend": {
  "base_url": "http://127.0.0.1:9008",
  "enabled": true
}
```

If `DIORAMA_REMOTE_API_KEY` is set on the cloud server, set the same environment variable on the local PC before launching DioramaForge. Remote execution can still be tested through API fields, but it is not part of the primary GUI flow.

For full-quality local inference, install optional model packages after your CUDA/PyTorch environment is stable:

```powershell
pip install -r requirements-optional-models.txt
```

Some FLUX checkpoints may require accepting the model license on Hugging Face and authenticating with a token.

Full FLUX.1 Depth snapshots are memory-heavy during Diffusers loading. Run only one DioramaForge app server with real FLUX loaded at a time. If model loading reports low RAM/pagefile, close other app servers or increase the Windows paging file before retrying.

With the img2img FLUX path, `strength` is the transformation strength. Lower values preserve the input scene more strongly. For research comparison runs, start around `0.35-0.55` before trying higher values.

## Local Model Setup

Use the PowerShell helper to create an isolated project environment:

```powershell
.\scripts\setup_windows.ps1
```

On this workstation you can reuse the already installed CUDA PyTorch wheel instead of downloading another large Torch build:

```powershell
.\scripts\setup_windows.ps1 -UseSystemSitePackages
```

Install optional DA3/SAM2 integrations:

```powershell
.\scripts\setup_windows.ps1 -UseSystemSitePackages -InstallOptionalModels
```

Download Depth Anything 3 and SAM 2 model snapshots into `models/huggingface`:

```powershell
.\scripts\setup_windows.ps1 -DownloadDepthAndSam
```

Download FLUX after accepting the Hugging Face license and providing a token:

```powershell
.\scripts\setup_windows.ps1 -DownloadFlux -HfToken "hf_..."
```

If you downloaded FLUX somewhere else, set `models.flux.local_path` in `configs/default.json` to the folder that contains `model_index.json`.

Download the faster SDXL style-engine candidate after accepting any required Hugging Face licenses:

```powershell
.\scripts\setup_windows.ps1 -DownloadStyleSdxl -HfToken "hf_..."
```

The dry-first preparation helper checks the current state before downloading:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_style_sdxl.ps1
```

After confirming that you want to download the fast local style-engine models:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_style_sdxl.ps1 -Download
```

If `key\hf.txt` exists, both setup helpers use it without printing the token value.

The direct downloader equivalent is:

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py --models style_sdxl --hf-token "hf_..."
```

This downloads the SDXL base model, SDXL depth ControlNet, and the configured SDXL-Lightning LoRA into `models/huggingface`.

## Style Engine Readiness And Benchmark

Check local model readiness without loading heavy models:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_style_engine.py
```

This prints the configured engine, resolved engine, and missing ComfyUI/FLUX/SDXL components. It does not run generation unless `--run` is provided.

Check API/GUI contract payloads without starting the server:

```powershell
.\.venv\Scripts\python.exe scripts\check_api_contract.py
.\.venv\Scripts\python.exe scripts\check_desktop_contract.py
.\.venv\Scripts\python.exe scripts\check_product_speed_policy.py
.\.venv\Scripts\python.exe scripts\check_env_local_contract.py
.\.venv\Scripts\python.exe scripts\check_comfy_workflows.py
.\.venv\Scripts\python.exe scripts\check_comfy_workflow_installer.py
.\.venv\Scripts\python.exe scripts\check_comfy_workflow_inspector.py
.\.venv\Scripts\python.exe scripts\check_comfy_workflow_preparer.py
.\.venv\Scripts\python.exe scripts\check_comfy_workflow_examples.py
.\.venv\Scripts\python.exe scripts\install_comfy_example_workflow.py
.\.venv\Scripts\python.exe scripts\check_comfy_node_compatibility_contract.py
.\.venv\Scripts\python.exe scripts\check_comfy_model_patch_contract.py
.\.venv\Scripts\python.exe scripts\check_comfy_node_compatibility.py
.\.venv\Scripts\python.exe scripts\check_meshy_contract.py
.\.venv\Scripts\python.exe scripts\check_validation_meshy_contract.py
.\.venv\Scripts\python.exe scripts\check_demo_runtime.py
.\.venv\Scripts\python.exe scripts\check_demo_benchmark_contract.py
```

`scripts\check_comfy_workflows.py` validates the configured ComfyUI API workflow files without starting ComfyUI or loading models. The required Stage 3 workflow must pass this contract before the desktop path can be considered ready for a live demonstration.

Use the inspector when converting a ComfyUI export into a DioramaForge workflow:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_comfy_workflow.py path\to\workflow_api.json --stage stage3
```

For simple API-format exports, the preparer can patch obvious input fields automatically:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_comfy_workflow.py path\to\workflow_api.json workflows\comfy\stage3_style_api.json --stage stage3
```

An editable Stage 3 example lives at `workflows/comfy/examples/stage3_sdxl_depth_img2img_api.example.json`. It is not active until copied or installed as `workflows/comfy/stage3_style_api.json`, and its static model filenames must be edited to match your ComfyUI model folders.

You can validate the bundled example without starting the API server:

```powershell
.\.venv\Scripts\python.exe scripts\install_comfy_example_workflow.py
```

To install it as the active Stage 3 workflow through the same validation and backup path used by the GUI:

```powershell
.\.venv\Scripts\python.exe scripts\install_comfy_example_workflow.py --install
```

The same validation is available through the API for the desktop setup flow:

```text
GET  /api/comfy/workflows
GET  /api/comfy/model-choices
POST /api/comfy/workflows/stage3/inspect
POST /api/comfy/workflows/stage3/install
POST /api/comfy/workflows/stage3/install-example
POST /api/comfy/workflows/stage3/prepare-install
```

`/api/comfy/model-choices` reads ComfyUI `/object_info` and lists relevant checkpoint, ControlNet, LoRA, VAE, upscaler, sampler, and scheduler choices when the server is running.

After installing the bundled example, inspect the active workflow's static model names:

```powershell
.\.venv\Scripts\python.exe scripts\patch_comfy_stage3_models.py
```

Patch them to match a portable ComfyUI install without opening the JSON by hand:

```powershell
.\.venv\Scripts\python.exe scripts\patch_comfy_stage3_models.py --checkpoint "your_checkpoint.safetensors" --controlnet "your_depth_controlnet.safetensors"
```

The patcher creates a timestamped backup by default and keeps the Stage 3 workflow contract valid. It can also patch `--sampler` and `--scheduler`.

When ComfyUI is running, check that the server has every node class used by the installed workflow:

```powershell
.\.venv\Scripts\python.exe scripts\check_comfy_node_compatibility.py
.\.venv\Scripts\python.exe scripts\check_comfy_node_compatibility.py --require
```

The live node compatibility check also compares static model/control choices from the workflow against `/object_info` choices when ComfyUI exposes them, so a missing checkpoint or ControlNet filename can be reported before queueing a prompt. The contract test for that behavior does not require ComfyUI:

```powershell
.\.venv\Scripts\python.exe scripts\check_comfy_node_compatibility_contract.py
.\.venv\Scripts\python.exe scripts\check_comfy_model_patch_contract.py
```

When the API is running, `/api/demo/readiness` exposes the same live-run readiness status for the desktop banner without loading generation models. It separates `fast_path_ready`, `runtime_ready`, `product_3d_ready`, and `timed_smoke_ready`: with the ComfyUI product path, the first means the ComfyUI server and required Stage 3 workflow are ready; the second means the configured runtime preflight passed; the third means Stage 4/5 requirements such as `MESHY_API_KEY` are ready when Meshy is configured; the fourth means a real timed benchmark has already completed within `product_pipeline.demo_time_budget_seconds` and its `run_dir`, `final_image`, and `metadata` artifacts still exist.

Check local runtime prerequisites without loading generation models:

```powershell
.\.venv\Scripts\python.exe scripts\check_demo_runtime.py
.\.venv\Scripts\python.exe scripts\check_demo_runtime.py --require
```

After the required model caches are ready and you intentionally want a timed smoke run:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_style_engine.py --run --force-engine auto --size 512 --steps 4
```

You can omit `--size` and `--steps` to use the current `product_pipeline` defaults:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_style_engine.py --run --force-engine auto
```

Use this benchmark to verify whether the configured product Stage 3 backend is fast enough for the 240-second readiness budget before changing expectations. By default it uses `product_pipeline` settings. Real local FLUX Diffusers runs can still take much longer.

Check whether the latest timed smoke report proves the live-demo budget:

```powershell
.\.venv\Scripts\python.exe scripts\check_demo_benchmark.py
.\.venv\Scripts\python.exe scripts\check_demo_benchmark.py --require
```

The benchmark contract check uses synthetic JSON reports and does not run models. It verifies that dry-run, wrong-engine, slow, settings-mismatched, and missing-artifact benchmark reports cannot be mistaken for demo readiness.

Write a timestamped serverless readiness report for presentation preparation:

```powershell
.\.venv\Scripts\python.exe scripts\write_demo_readiness_report.py
```

This writes JSON and Markdown under `outputs/readiness/` using the same `/api/demo/readiness`, `/api/style-engine`, `/api/pipeline/defaults`, `/api/pipeline/preflight`, `/api/execution/policy`, and `/api/meshy/status` payloads, without starting the API server or loading generation models.

See `docs/local_style_engine_speed_plan_20260614.md` for the current local speed optimization plan.

Run the legacy Gradio GUI from the project venv only for debugging:

```powershell
.\.venv\Scripts\python.exe app.py --server-port 7860 --inbrowser
```

## Outputs

Each run creates a timestamped folder under `outputs/runs/`:

- `input.png`
- `depth.png`
- `depth.npy`
- `masks/mask_*.png`
- `mask_overlay.png`
- `regions/region_plan.json`
- `regions/region_overlay.png`
- `regions/{semantic_label}_mask.png`
- `flux_control.png`
- `sam_structure_hint.png`
- `flux_result.png`
- `run_metadata.json`
- `stage35_refinement/stage35_upscaled_visual.png`
- `stage35_refinement/stage35_reconstruction_input.png`
- `stage35_refinement/stage35_refined.png`
- `stage35_refinement/stage35_metadata.json`
- `stage4_reconstruction/reconstruction_package.json`
- `stage4_reconstruction/parts/{part}/source_crop.png`
- `stage4_reconstruction/parts/{part}/styled_cutout.png`
- `stage4_reconstruction/heightfield_proxy.obj`
- `stage4_reconstruction/meshy/model_glb.glb`
- `stage4_reconstruction/meshy/model_obj.obj`
- `stage4_reconstruction/meshy/model_stl.stl`
- `stage5_print/print_package.json`
- `stage5_print/meshy_model/glb.glb`
- `stage5_print/meshy_model/obj.obj`
- `stage5_print/meshy_model/stl.stl`
- `stage5_print/print_ready_relief_proxy.stl`
- `stage5_print/print_checklist.md`

`regions/region_plan.json` is a Stage 3 planning artifact. It groups SAM masks into broad semantic regions such as sky, foliage, terrain, structure, and water only when there is enough visual evidence. Actual segmentation-unit image splitting and per-part extraction are reserved for Stage 4, where physical regions become 3D reconstruction/printing units while sky/backdrop regions are kept as visual reference rather than solid mesh.

The Stage 4/5 proxy mesh outputs are retained for validation and slicer handoff. The current short-term live-demo 3D output path is Meshy AI, not local TRELLIS/UltraShape inference.

Model checkpoints are not committed to the project. Change model IDs and runtime behavior in `configs/default.json`.
The app defaults `HF_HOME` to `models/huggingface` so downloads stay inside this project.

## Legacy Experiment Utilities

Experiment-grid utilities remain for research records, but they are no longer part of the primary desktop GUI. They run a small parameter grid against one input image and write:

- `outputs/experiments/{timestamp}/contact_sheet.png`
- `outputs/experiments/{timestamp}/experiment_summary.csv`
- `outputs/experiments/{timestamp}/experiment_report.md`
- one run folder per parameter combination

Start with low resolution and low steps before launching larger style-generation experiments.
