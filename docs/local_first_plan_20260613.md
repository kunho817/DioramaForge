# DioramaForge Local-First Plan

Date: 2026-06-13

## Current Assumption

External cloud execution for FLUX is canceled. DioramaForge should now be treated as a local-first desktop/API application.

## What This Changes

- `Remote A100` is no longer the default user path.
- The desktop GUI exposes one primary `Generate` flow instead of user-selectable execution modes.
- `Generate` always runs the same product path: Stage 3, Stage 3.5, Stage 4, and Stage 5 proxy packaging.
- The hidden Generate profile is configured by `product_pipeline` in `configs/default.json` and surfaced through `/api/pipeline/defaults`.
- Local backend/model choices remain only as internal style-engine settings.
- Local heavy model execution is allowed by configuration, but should not be started casually during development because real generation jobs can take a long time.
- Cloud/Elice scripts remain in the repo as optional legacy utilities.
- The current product style route is `backend_mode=comfyui`; Stage 3 expects `workflows/comfy/stage3_style_api.json`.

## Near-Term Development Priorities

1. Stabilize local model configuration.
   - Confirm model paths/caches are visible from the GUI.
   - Make memory warnings explicit before FLUX loading.
   - Keep local fallback/error messages specific.

2. Improve and replace Stage 3 controllability.
   - Preserve original layout.
   - Make strength/guidance/steps effects easier to compare.
   - Separate source image influence from depth/control influence where possible.
   - Validate the configured ComfyUI Stage 3 workflow after the workflow JSON and model files are available.

3. Strengthen artifact validation.
   - Use `src/diorama_forge/validation.py` for run contract checks.
   - Surface validation errors in the GUI.
   - Treat missing manifests or stage mismatch as actionable issues.

4. Expand Stage 4 segmentation-unit packaging.
   - Ensure each semantic region exports source crop, styled crop, depth crop, mask, and cutout.
   - Keep manifest fields stable for later 3D adapters.

5. Prepare Stage 5 adapter boundary.
   - Current STL/OBJ outputs are proxy validation artifacts.
   - Keep print settings and mesh stats in manifests.
   - Add real mesh repair/refinement only after the local Stage 3/4 handoff is stable.

## Safe Development Checks

These checks do not run real model generation:

```powershell
python -m compileall -q src scripts app.py api_app.py model_backend_app.py
npm run build
cargo check
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_readiness.ps1
.\.venv\Scripts\python.exe scripts\benchmark_style_engine.py
.\.venv\Scripts\python.exe scripts\check_api_contract.py
.\.venv\Scripts\python.exe scripts\check_desktop_contract.py
```

`scripts\benchmark_style_engine.py` without `--run` is a dry readiness check. It does not load real generation models.

## Local Real Run Caution

Before starting a real local generation run:

- Use a small resolution first.
- Keep steps low for smoke tests.
- Confirm there is enough RAM/VRAM.
- Avoid batch experiments unless the user explicitly asks for them.
- Record settings in `run_metadata.json` and validate the run afterward.

For a timed smoke benchmark after the configured Stage 3 backend is ready:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_style_engine.py --run --force-engine auto --size 512 --steps 4
```

You can omit `--size` and `--steps` to use `product_pipeline` defaults. Do not use this command during routine development unless the user explicitly wants a real model run.

## Current Default

`configs/default.json` sets:

```json
"allow_local_heavy_models": true
```

This allows local execution, but it does not mean every development task should run a model.

The user-facing desktop flow is intentionally mode-free:

```json
"style_engine": {
  "active": "auto",
  "target": "comfyui_stage3_style",
  "backend_mode": "comfyui",
  "show_backend_selector": false
}
```

The fixed Generate defaults are intentionally not exposed as GUI controls:

```json
"product_pipeline": {
  "steps": 4,
  "guidance": 3.5,
  "strength": 0.55,
  "max_resolution": 512
}
```

With `backend_mode=comfyui`, DioramaForge routes Stage 3 through the configured ComfyUI API workflow. That workflow can internally use FLUX, SDXL-Lightning, SDXL-Turbo/img2img, or another depth-conditioned graph without changing the GUI.

If the ComfyUI workflow uses the SDXL-Lightning candidate, prepare those caches with:

```powershell
.\scripts\setup_windows.ps1 -DownloadStyleSdxl -HfToken "hf_..."
```

Or use the dry-first helper:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_style_sdxl.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_style_sdxl.ps1 -Download
```
