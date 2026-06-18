# DioramaForge ComfyUI Workflows

This directory stores ComfyUI workflows exported with **Save (API Format)**.

The default config expects:

```text
workflows/comfy/stage3_style_api.json
workflows/comfy/stage35_upscale_reconstruction_api.json
workflows/comfy/stage35_refine_api.json
```

Only `stage3_style_api.json` is required for the first ComfyUI backend milestone. `stage35_upscale_reconstruction_api.json` enables the Stage 3.5 ComfyUI handoff path, and `stage35_refine_api.json` is optional for a second refinement workflow.

`stage3_style_api.json` should stay model-family neutral. It may be a FLUX Depth graph, an SDXL-Lightning depth graph, an SDXL-Turbo/img2img graph, or another depth-conditioned graph as long as it accepts the placeholder contract below and returns one final image. The DioramaForge desktop GUI does not change when the internal graph changes.

Example workflows are under `workflows/comfy/examples/`. They are not active product workflows. Copy one to `stage3_style_api.json`, then edit static model names such as checkpoint and ControlNet filenames to match your ComfyUI installation.

## Placeholder Contract

After exporting a workflow in API format, replace the relevant input values with these placeholders:

| Placeholder | Meaning |
|---|---|
| `__SOURCE_IMAGE__` | Uploaded source image filename for LoadImage |
| `__DEPTH_IMAGE__` | Uploaded DA3 depth image filename |
| `__CONTROL_IMAGE__` | Uploaded Stage 3 control image filename, currently pure depth |
| `__PROMPT__` | Full DioramaForge prompt |
| `__CLIP_PROMPT__` | Short CLIP/T5 prompt when the workflow separates prompt encoders |
| `__NEGATIVE_PROMPT__` | Negative prompt |
| `__SEED__` | Seed integer |
| `__STEPS__` | Sampling steps |
| `__GUIDANCE__` | Guidance value |
| `__STRENGTH__` | Img2img strength |
| `__DENOISE__` | Alias for denoise/strength in many ComfyUI workflows |
| `__WIDTH__` | Source width |
| `__HEIGHT__` | Source height |
| `__TARGET_WIDTH__` | Stage 3.5 target width after scale/max-side constraints |
| `__TARGET_HEIGHT__` | Stage 3.5 target height after scale/max-side constraints |
| `__UPSCALE_SCALE__` | Stage 3.5 requested upscale scale |
| `__REFINEMENT_STRENGTH__` | Stage 3.5 refinement strength |
| `__MAX_SIDE__` | Stage 3.5 maximum output side |
| `__STAGE35_MODE__` | Stage 3.5 mode string, currently `structure_preserving` |

Example snippets inside an API workflow:

```json
{
  "12": {
    "class_type": "LoadImage",
    "inputs": {
      "image": "__SOURCE_IMAGE__"
    }
  },
  "18": {
    "class_type": "LoadImage",
    "inputs": {
      "image": "__CONTROL_IMAGE__"
    }
  },
  "31": {
    "class_type": "KSampler",
    "inputs": {
      "seed": "__SEED__",
      "steps": "__STEPS__",
      "cfg": "__GUIDANCE__",
      "denoise": "__DENOISE__"
    }
  }
}
```

If the final image is not the first image output in the ComfyUI history, set `comfyui.output_node_id` in `configs/default.json`.

Validate workflow files without starting ComfyUI or loading models:

```powershell
.\.venv\Scripts\python.exe scripts\check_comfy_workflows.py
.\.venv\Scripts\python.exe scripts\check_comfy_workflows.py --require
```

The static validator checks that the file was exported with **Save (API Format)**, contains the required placeholders for the configured stage, and has an obvious image output node or a configured `comfyui.output_node_id`.

Inspect an exported workflow before editing placeholders:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_comfy_workflow.py path\to\workflow_api.json --stage stage3
.\.venv\Scripts\python.exe scripts\inspect_comfy_workflow.py path\to\workflow_api.json --stage stage3 --format json
```

The inspector reports whether the file is API format or normal UI format, then lists likely `LoadImage`, text prompt, sampler, size, and output nodes. Use that report to decide where to replace values with `__SOURCE_IMAGE__`, `__CONTROL_IMAGE__`, `__PROMPT__`, `__SEED__`, `__STEPS__`, `__GUIDANCE__`, and `__DENOISE__`.

For straightforward API-format exports, prepare a placeholder-patched copy automatically:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_comfy_workflow.py path\to\workflow_api.json workflows\comfy\stage3_style_api.json --stage stage3
```

The preparer only handles obvious mappings: the first image input becomes `__SOURCE_IMAGE__`, the second image input becomes `__CONTROL_IMAGE__`, the first positive text input becomes `__PROMPT__`, and sampler fields become `__SEED__`, `__STEPS__`, `__GUIDANCE__`, and `__DENOISE__`. Inspect the result before relying on it for complex workflows.

When the API/desktop shell is running, the same Stage 3 validation is available from the setup card in the left panel. The API endpoint is:

```text
GET /api/comfy/model-choices
POST /api/comfy/workflows/stage3/inspect
POST /api/comfy/workflows/stage3/install
POST /api/comfy/workflows/stage3/prepare-install
```

`GET /api/comfy/model-choices` is useful before installation. It reads ComfyUI `/object_info` and lists model-like choices so the example workflow's static checkpoint, ControlNet, LoRA, VAE, or scheduler values can be edited to match the local ComfyUI setup.

The active Stage 3 workflow can also be inspected and patched without editing JSON by hand:

```powershell
.\.venv\Scripts\python.exe scripts\patch_comfy_stage3_models.py
.\.venv\Scripts\python.exe scripts\patch_comfy_stage3_models.py --checkpoint "your_checkpoint.safetensors" --controlnet "your_depth_controlnet.safetensors"
```

The patcher supports `--checkpoint`, `--controlnet`, `--lora`, `--vae`, `--upscale-model`, `--sampler`, and `--scheduler` when those node fields exist in the workflow. It creates a timestamped backup by default and validates the workflow contract after patching. Use `--dry-run` to preview changes without writing.

Use `Inspect Workflow` first to check the uploaded JSON without replacing any project file. Use `Prepare & Install` for straightforward raw API exports, or `Install Workflow` after the contract is already correct. During installation, the uploaded JSON is first written as an incoming temporary file, validated, and only then moved into `workflows/comfy/stage3_style_api.json`. Existing workflow files are backed up with a timestamped `.bak.json` suffix before replacement.

Use `Install Example` to copy the bundled Stage 3 example into `stage3_style_api.json` through the same validation and backup path. After that, the workflow still needs running-server compatibility: if ComfyUI reports missing checkpoint, ControlNet, or other model choices, edit the static model filenames and reinstall the corrected workflow.

The same action is available without the API/desktop shell:

```powershell
.\.venv\Scripts\python.exe scripts\install_comfy_example_workflow.py
.\.venv\Scripts\python.exe scripts\install_comfy_example_workflow.py --install
```

After installing a workflow and starting ComfyUI, check that the running server exposes the workflow's node classes:

```powershell
.\.venv\Scripts\python.exe scripts\check_comfy_node_compatibility.py
```

This catches missing ComfyUI custom nodes before a long Generate job is queued. When ComfyUI exposes input choices through `/object_info`, the check also validates static workflow values such as checkpoint, ControlNet, LoRA, VAE, or upscaler names against the available choices. That helps catch missing model files before DioramaForge uploads images or queues a prompt.

The non-network contract test for that behavior is:

```powershell
.\.venv\Scripts\python.exe scripts\check_comfy_node_compatibility_contract.py
.\.venv\Scripts\python.exe scripts\check_comfy_model_patch_contract.py
```

## Stage 3.5 Workflow Notes

`stage35_upscale_reconstruction_api.json` should load `__SOURCE_IMAGE__` as the Stage 3 style result and `__DEPTH_IMAGE__` or `__CONTROL_IMAGE__` as the depth condition. Its final image becomes:

```text
stage35_refinement/stage35_reconstruction_input.png
```

If `stage35_refine_api.json` exists, DioramaForge runs it after the reconstruction workflow. In that second workflow, `__SOURCE_IMAGE__` is the reconstruction output from the first workflow. If the refine workflow is absent, DioramaForge creates `stage35_refined.png` with the deterministic local structure-preserving pass.
