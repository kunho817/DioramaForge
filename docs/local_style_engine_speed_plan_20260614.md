# Local Stage 3 Speed Plan

Date: 2026-06-14
Updated: 2026-06-15

## Decision

DioramaForge remains local-first. The external cloud backend is not part of the current execution plan.

The user-facing desktop GUI must provide one consistent workflow:

```text
Input image
-> Stage 1 depth
-> Stage 2 segmentation/regions
-> Stage 3 style generation
-> Stage 3.5 structure handoff/refinement
-> Stage 4 segmentation-unit package
-> Stage 5 print proxy package
```

The GUI must not expose backend selectors, paper/test/quality modes, or per-stage checkboxes. Those choices belong in internal configuration, API/debug endpoints, benchmark scripts, and ComfyUI workflow files.

The product API path must also stay mode-free. `/api/pipeline/jobs` should accept only the source image, style preset, optional prompt, and Stage 3 tuning values from the fixed product profile. It must not accept Stage on/off toggles or backend override fields from the desktop GUI.

Product `Generate` must run `/api/pipeline/preflight` before starting work. The preflight blocks obvious setup failures such as a missing ComfyUI workflow/server/node class or missing Meshy API key, so the app does not spend minutes on early stages before failing. Timed smoke remains a demo-readiness gate, not a user-facing mode.

For the short-term live demo, the 3D portion is simplified through Meshy AI Image to 3D. Stage 4 submits the styled reconstruction image to Meshy and Stage 5 packages the returned model files. Local TRELLIS/UltraShape work is deferred.

## Speed Problem

The Diffusers FLUX.1 Depth path can take longer than a practical live demonstration window on the local workstation. The target is a live run under `product_pipeline.demo_time_budget_seconds` seconds, currently 240 seconds so the run leaves margin inside a 5-minute presentation slot.

The main speed strategy is to move Stage 3 image generation behind an internal ComfyUI API workflow:

```json
"style_engine": {
  "active": "auto",
  "target": "comfyui_stage3_style",
  "backend_mode": "comfyui"
}
```

The required workflow file is:

```text
workflows/comfy/stage3_style_api.json
```

That workflow name is intentionally model-family neutral. It can be replaced internally without changing the desktop GUI.

## Engine Candidates

### 1. ComfyUI SDXL-Lightning Depth Workflow

Recommended live-demo candidate.

Use an SDXL base/checkpoint with a depth conditioning graph and SDXL-Lightning acceleration. The ByteDance SDXL-Lightning model card provides 1, 2, 4, and 8 step checkpoints and explicitly documents ComfyUI usage. It also warns to match the checkpoint to the inference step count.

Current DioramaForge product defaults use the 12-step Illustrious SDXL direction:

```text
resolution: 512
steps: 12
guidance: 3.5
strength: 0.55
Stage 3.5 backend: deterministic handoff/proxy
demo budget: 240 seconds
```

This path is expected to be much more demo-friendly than the full FLUX Diffusers path after ComfyUI is warm-loaded.

### 2. ComfyUI FLUX Depth Workflow

Research-quality or comparison candidate.

FLUX Depth can produce strong style fidelity, but the local Diffusers snapshot is large and slow in the current environment. Keep it available for paper comparison or selected high-quality runs, but do not make it the live-demo expectation until a real timed ComfyUI run proves the 240 second budget.

### 3. SDXL-Turbo Img2Img Workflow

Fallback speed candidate.

SDXL-Turbo is designed for 1 to 4 step generation and image-to-image use. It is promising for speed, but it does not automatically solve depth/segmentation structure preservation. Use it only if a ComfyUI workflow can keep the source image and depth control stable enough for DioramaForge's goal.

## Product Policy

The product GUI exposes:

- image upload
- style preset
- optional prompt
- one `Generate` button
- recent run loader
- artifacts and readiness status

It does not expose:

- quality mode
- paper mode
- test mode
- model picker
- backend picker
- stage checkbox controls

The fixed product profile lives in `configs/default.json` under `product_pipeline`. Change that config when tuning the live-demo profile. Do not add UI switches.

## ComfyUI Workflow Contract

Export the graph with ComfyUI `Save (API Format)` and replace input values with the placeholders in `workflows/comfy/README.md`.

Minimum Stage 3 requirements:

- Load `__SOURCE_IMAGE__` as the original source image.
- Load `__CONTROL_IMAGE__` or `__DEPTH_IMAGE__` as the depth control image.
- Use `__PROMPT__` or `__CLIP_PROMPT__`.
- Use `__SEED__`, `__STEPS__`, `__GUIDANCE__`, and `__DENOISE__`/`__STRENGTH__`.
- Return one final image output in ComfyUI history.

If the final image is not the first image output in history, set `comfyui.output_node_id` in `configs/default.json`.

## Demo Procedure

Before the live demo:

1. Start ComfyUI and load the Stage 3 workflow once.
2. Start the DioramaForge API and desktop shell through `scripts/start_diorama_forge.bat`.
3. Run readiness checks without generating:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_style_engine.py
.\.venv\Scripts\python.exe scripts\check_demo_runtime.py
.\.venv\Scripts\python.exe scripts\check_api_contract.py
.\.venv\Scripts\python.exe scripts\check_desktop_contract.py
```

4. Confirm `/api/pipeline/preflight` is ready from the desktop status area or API docs. This proves the fixed Generate path can start; it does not prove the time budget.

5. Run a real timed smoke only when intentionally preparing the demo:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_style_engine.py --run --force-engine auto
.\.venv\Scripts\python.exe scripts\check_demo_benchmark.py --require
```

The smoke run must use the product profile and must produce a valid `run_dir`, `final_image`, and `metadata`.

## Acceptance Criteria

- Desktop GUI exposes exactly one `Generate` action.
- `Generate` always runs the same product path.
- Product `Generate` always executes Stage 3 -> Stage 3.5 -> Stage 4 -> Stage 5; Stage inclusion is not a user-facing or request-level option.
- Product `Generate` runs a blocking preflight before reading the uploaded image or starting Depth/SAM work.
- Stage 3 uses the configured ComfyUI workflow when `style_engine.backend_mode=comfyui`.
- FLUX replacement happens by swapping the internal ComfyUI Stage 3 workflow graph, not by exposing a model picker or quality mode.
- Missing ComfyUI server/workflow is reported as readiness failure, not hidden as a user-facing mode.
- `/api/demo/readiness` separates setup readiness, runtime readiness, node/model-choice compatibility, and timed benchmark readiness.
- `scripts/check_demo_benchmark.py --require` must pass before claiming the 240-second live-demo readiness budget.
- FLUX remains a baseline/comparison path, not the live-demo default.

## Sources Checked

- ByteDance SDXL-Lightning model card: https://huggingface.co/ByteDance/SDXL-Lightning
- Diffusers SDXL depth ControlNet model card: https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0
- Stability AI SDXL-Turbo model card: https://huggingface.co/stabilityai/sdxl-turbo
