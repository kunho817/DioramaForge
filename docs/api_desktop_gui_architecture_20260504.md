# DioramaForge API + Desktop GUI Architecture

Updated: 2026-06-15

## Direction

DioramaForge uses this product architecture:

```text
Python FastAPI server
-> React/Vite UI
-> Tauri desktop shell
-> one-click batch launcher
```

The desktop shell is the product GUI. Gradio is retained only as a legacy/debug path.

The user-facing GUI exposes one workflow:

```text
Upload image -> choose style preset -> optional prompt -> Generate -> inspect artifacts
```

It does not expose paper/test/quality modes, model pickers, backend selectors, or per-stage checkboxes.

## API Responsibilities

The API server owns model execution, artifact writing, job state, and run loading. React/Tauri never loads models directly.

Entry point:

```text
api_app.py
```

Primary endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | API health |
| GET | `/api/jobs` | Recent background jobs |
| GET | `/api/jobs/{job_id}` | Job status, logs, result |
| GET | `/api/runs` | Recent `outputs/runs` folders |
| GET | `/api/runs/{run_id}` | Restore Stage 3/3.5/4/5 artifacts |
| GET | `/api/runtime` | CUDA/PyTorch/runtime status |
| GET | `/api/models` | DA3/SAM2/style/3D package status |
| GET | `/api/style-engine` | Internal Stage 3 engine readiness |
| GET | `/api/demo/readiness` | Live-run readiness gate |
| GET | `/api/pipeline/defaults` | Hidden product `Generate` profile |
| GET | `/api/comfy/status` | ComfyUI server/queue/workflow status |
| GET | `/api/comfy/workflows` | Configured ComfyUI workflow file status |
| GET | `/api/meshy/status` | Meshy AI Image-to-3D configuration and API key readiness |
| POST | `/api/comfy/workflows/stage3/inspect` | Inspect an uploaded ComfyUI API workflow without installing it |
| POST | `/api/comfy/workflows/stage3/install` | Validate and install a Stage 3 ComfyUI API workflow JSON |
| POST | `/api/comfy/workflows/stage3/prepare-install` | Auto-patch obvious placeholders, validate, and install a Stage 3 workflow |
| GET | `/api/remote/status` | Legacy remote backend status |
| GET | `/api/presets` | Style preset names |
| POST | `/api/pipeline/jobs` | Product `Generate` background job |
| POST | `/api/stage3/run` | Developer/debug Stage 1-3 run |
| POST | `/api/stage3/jobs` | Developer/debug Stage 1-3 background job |
| POST | `/api/stage35/refine` | Developer/debug Stage 3.5 handoff |
| POST | `/api/stage35/jobs` | Developer/debug Stage 3.5 background job |
| POST | `/api/stage4/package` | Developer/debug Stage 4 package |
| POST | `/api/stage4/jobs` | Developer/debug Stage 4 background job |
| POST | `/api/stage5/print` | Developer/debug Stage 5 print package |
| POST | `/api/stage5/jobs` | Developer/debug Stage 5 background job |
| GET | `/outputs/...` | Static artifact serving |

## Style Engine Boundary

The product GUI calls `/api/pipeline/jobs`. It does not decide which image generation backend to use.

`configs/default.json` controls the hidden product path:

```json
"style_engine": {
  "active": "auto",
  "target": "comfyui_stage3_style",
  "backend_mode": "comfyui"
}
```

With `backend_mode=comfyui`, Stage 3 uses the ComfyUI API workflow configured at:

```text
workflows/comfy/stage3_style_api.json
```

That workflow can internally use FLUX Depth, SDXL-Lightning, SDXL-Turbo/img2img, or another depth-conditioned graph. The GUI and API contract do not change when the graph changes.

## Product Generate Profile

The fixed product profile lives in `configs/default.json` under `product_pipeline`.

React/Tauri reads it from `/api/pipeline/defaults` and submits it silently with `Generate`.

Example defaults:

```json
{
  "steps": 4,
  "guidance": 3.5,
  "strength": 0.55,
  "max_resolution": 512,
  "demo_time_budget_seconds": 240,
  "stage_contract": ["stage3", "stage35", "stage4", "stage5"]
}
```

The product `/api/pipeline/jobs` endpoint does not accept Stage on/off toggles, backend overrides, or quality/test/paper mode fields. Change internal config values or the ComfyUI workflow for demo tuning. Do not add visible GUI modes.

## Live Readiness

`/api/demo/readiness` is a status endpoint, not a mode selector.

It separates:

- `fast_path_ready`: the configured Stage 3 image backend is ready. With ComfyUI, this means the server is reachable, the required Stage 3 workflow exists, and the server exposes the node classes used by that workflow.
- `runtime_ready`: the configured runtime preflight passed.
- `product_3d_ready`: the configured Stage 4/5 backend is ready. When Stage 4/5 use Meshy, this includes `MESHY_API_KEY` and the local `requests` dependency.
- `timed_smoke_ready`: the latest real benchmark used the product profile, completed within `product_pipeline.demo_time_budget_seconds`, and still has `run_dir`, `final_image`, and `metadata` artifacts.

The desktop banner can show readiness and next action. It must not expose backend switching controls.

## ComfyUI Backend

Default server:

```text
http://127.0.0.1:8188
```

Required workflow:

```text
workflows/comfy/stage3_style_api.json
```

Optional workflows:

```text
workflows/comfy/stage35_upscale_reconstruction_api.json
workflows/comfy/stage35_refine_api.json
```

Communication uses:

- `/upload/image`
- `/prompt`
- `/history/{prompt_id}`
- `/view`

The expected live-demo setup is a warm ComfyUI server with the Stage 3 graph already available.

## Stage 3.5 Handoff

Stage 3.5 prepares a structure-preserving handoff image before Stage 4.

Outputs:

- `stage35_refinement/stage35_upscaled_visual.png`
- `stage35_refinement/stage35_reconstruction_input.png`
- `stage35_refinement/stage35_refined.png`
- `stage35_refinement/stage35_metadata.json`

The live product default keeps Stage 3.5 in the fixed pipeline but uses the deterministic proxy/handoff backend so short demonstrations do not trigger a second heavy generation pass. A Stage 3.5 ComfyUI workflow can still be used through internal config/debug paths when a real refinement pass is needed.

## Stage 4/5 Meshy Shortcut

The short-term live-demo 3D route uses Meshy AI Image to 3D instead of local TRELLIS/UltraShape inference.

Default product config:

```json
{
  "stage4_backend_mode": "meshy",
  "stage5_backend_mode": "meshy"
}
```

Stage 4 derives a Meshy-focused image from `stage35_refinement/stage35_reconstruction_input.png` or the final Stage 3 style result, removes sky/backdrop regions when available, submits that image as a Meshy Image-to-3D data URI, then downloads the configured model formats under:

```text
stage4_reconstruction/meshy/
```

Stage 5 copies those model files into:

```text
stage5_print/meshy_model/
```

The proxy OBJ/STL outputs remain in the artifact contract for validation and fallback inspection.

## Background Jobs

Long work runs through a single-worker job manager:

1. GUI posts to `/api/pipeline/jobs`.
2. API immediately returns `job_id`, `status`, and initial log.
3. GUI polls `/api/jobs/{job_id}`.
4. When the job succeeds, GUI applies `result`.
5. When the job fails, GUI shows `error` and the accumulated log.

The single worker reduces local GPU/model-memory conflicts.

## Run Browser

The desktop GUI uses `/api/runs` and `/api/runs/{run_id}` to restore existing run folders.

Use cases:

- review previous research outputs
- compare Stage 3/3.5/4/5 artifacts
- validate package manifests
- inspect proxy OBJ/STL handoff files

## Legacy Remote Backend

Elice/A100 remote execution remains optional legacy tooling. It is not the default product route and is not exposed in the desktop GUI.

Keep remote scripts only for manual debugging unless the project explicitly reintroduces cloud execution.

## Desktop UI Contract

Current product UI:

- exactly one `Generate` action
- one style preset select
- one recent-run select
- no backend selector
- no mode selector
- no per-stage checkbox controls
- displays original/depth/masks/region/style/Stage 3.5/Stage 4/Stage 5 artifacts
- displays readiness and validation status

Contract checks:

```powershell
.\.venv\Scripts\python.exe scripts\check_api_contract.py
.\.venv\Scripts\python.exe scripts\check_desktop_contract.py
```
