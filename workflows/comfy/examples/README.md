# DioramaForge ComfyUI Examples

These files are examples, not active product workflows.

Copy an example to `workflows/comfy/stage3_style_api.json`, then edit the static ComfyUI model filenames to match the models installed in your ComfyUI folders. The DioramaForge validator replaces only runtime placeholders such as images, prompt, seed, steps, guidance, and denoise.

Recommended live-demo direction:

```text
source image + depth/control image -> fast SDXL/Lightning img2img-depth graph -> final styled image
```

Use the desktop Stage 3 workflow setup card or this command after editing:

```powershell
.\.venv\Scripts\python.exe scripts\check_comfy_workflows.py --require
```

After starting ComfyUI, run:

```powershell
.\.venv\Scripts\python.exe scripts\check_comfy_node_compatibility.py --require
```

That second check verifies node classes and, when ComfyUI exposes choices through `/object_info`, static model names such as checkpoint and ControlNet filenames.
