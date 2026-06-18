# Product Generate Speed And Mode Cleanup

Date: 2026-06-15

## Decision

The product GUI and product API expose one functional path:

```text
Input image
-> Stage 1 depth
-> Stage 2 segmentation/region plan
-> Stage 3 style generation
-> Stage 3.5 structure handoff
-> Stage 4 Meshy Image-to-3D package
-> Stage 5 Meshy model package with print proxy fallback
```

There are no paper, test, quality, backend, or per-stage operation modes in the user-facing flow.

## Speed Strategy

The bottleneck is Stage 3 image generation. The product route keeps Stage 3 behind a model-family-neutral ComfyUI API workflow:

```text
workflows/comfy/stage3_style_api.json
```

This lets the project replace the current FLUX-heavy graph with a faster depth-conditioned graph without changing the GUI or API contract.

Recommended live-demo direction:

- Use a warm ComfyUI server.
- Use a 512 px product profile.
- Use a 12-step depth-conditioned graph for the current Illustrious SDXL route, then re-benchmark before live demonstration.
- Use the current 240-second readiness budget rather than the full 5-minute presentation window, so there is room for upload, explanation, and artifact review.
- Keep Stage 3.5 on the deterministic handoff/proxy backend for the live product default, so the demo does not pay for a second heavy generation pass.
- Replace models by swapping the internal ComfyUI workflow graph, not by adding FLUX/SDXL/quality/test mode switches to the product GUI.

FLUX remains useful as a research-quality comparison baseline, but it should not be the expected live-demo path unless a real timed benchmark proves it under the configured budget.

## Implementation Plan

1. Keep the desktop and `/api/pipeline/jobs` product path fixed to one `Generate` action.
2. Keep `product_pipeline` at 512 px, 12 steps, and a 240-second readiness budget for live demonstration.
3. Replace slow FLUX output by installing a faster depth-conditioned ComfyUI API workflow at `workflows/comfy/stage3_style_api.json`.
4. Keep Stage 3.5 on the deterministic handoff backend for live runs so the app does not spend time on a second generation pass.
5. Validate readiness with static contracts first, then run one real timed smoke benchmark only after ComfyUI, workflow node/model choices, and Meshy requirements are ready.
6. Keep paper-quality FLUX or higher-resolution workflows outside the product GUI. Use config changes, saved workflows, and benchmark reports for research comparison instead of user-facing modes.

## 3D Shortcut Strategy

The local TRELLIS/UltraShape-style 3D path is deferred for the live demonstration because implementing and validating the full local 3D reconstruction and print-repair stack is too large for the remaining schedule.

The short-term product backend uses Meshy AI Image to 3D:

```json
{
  "stage4_backend_mode": "meshy",
  "stage5_backend_mode": "meshy"
}
```

Stage 4 derives a Meshy-specific input from the Stage 3.5 reconstruction image, removes sky/backdrop regions when they are detected, submits that focused image to Meshy as a data URI, polls the Image-to-3D task, and downloads the configured formats. The default requested outputs are:

```json
["glb", "obj", "stl"]
```

Stage 4 still exports segmentation-unit crops and a depth proxy OBJ. These artifacts remain useful for explaining the planned pipeline and validating the source image decomposition, but the actual short-term 3D model output comes from Meshy.

Stage 5 packages Meshy model files when available and keeps the generated depth-relief STL as a fallback/inspection proxy. Meshy outputs still need printability checks before physical printing.

Run validation now treats Meshy model files as first-class artifacts. When Stage 4 records `backend=meshy_image_to_3d`, validation checks the Meshy request, task, downloads manifest, task status, and downloaded GLB/OBJ/STL files. When Stage 5 records `backend=meshy_model_package`, validation checks the packaged model files as well as the proxy STL. Stage 4 fails before writing package outputs if Meshy is requested without a ready API key, with `meshy_ai.download_outputs=false`, or without a GLB/OBJ/STL target format. Stage 5 fails before writing package outputs if Meshy is requested but Stage 4 has no downloaded GLB/OBJ/STL model file. `scripts\check_validation_meshy_contract.py` verifies this behavior without network calls.

## Product API Contract

`POST /api/pipeline/jobs` should accept only:

- source image
- style preset
- optional prompt
- Stage 3 numeric tuning values from the fixed product profile

It should not accept:

- Stage on/off toggles
- backend override fields
- quality/test/paper mode fields

The internal product profile remains in `configs/default.json`. `/api/pipeline/defaults` exposes the user-safe defaults and the fixed stage contract:

```json
{
  "profile": "fixed_product_generate",
  "stage_contract": ["stage3", "stage35", "stage4", "stage5"]
}
```

`GET /api/pipeline/preflight` exposes the same blocking checks used by product `Generate`. `POST /api/pipeline/jobs` must run this preflight before reading the image or starting Depth/SAM work. If ComfyUI, the Stage 3 workflow, node-class compatibility, local execution policy, or the configured Stage 4/5 backend is not ready, the endpoint returns HTTP 409 with the preflight payload. Timed smoke failure is a readiness warning, not a request-level mode.

## Readiness Gate

Do not claim live demonstration readiness until all of these are true:

- Stage 3 ComfyUI workflow contract is valid.
- ComfyUI server is reachable.
- Running ComfyUI exposes all node classes and static model/input choices used by the workflow.
- `product_3d_ready` is true. When Stage 4/5 use Meshy, Meshy AI must be configured and `MESHY_API_KEY` must be set.
- A real timed smoke benchmark passed `product_pipeline.demo_time_budget_seconds` (currently 240 seconds).
- The benchmark artifacts still exist.

Non-generating checks are enough for development progress, but they do not prove live demo speed.
