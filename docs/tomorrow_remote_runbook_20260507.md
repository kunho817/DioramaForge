# DioramaForge Remote Experiment Runbook

Date: 2026-05-07

> Status: deprecated as of 2026-06-13. The external cloud FLUX execution plan was canceled.
> Keep this document only as historical reference. Current development should assume local-first execution.

## Goal

Prepare the project so experiment data is collected only through the remote A100 backend. Local execution is limited to GUI/API development, metadata validation, and Demo-mode structure checks.

## Current Execution Policy

- Local heavy model execution is disabled by default.
- Allowed backends under the default policy: `remote`, `demo`.
- Blocked local backends: `auto`, `comfyui`, `real`.
- Override exists only for exceptional debugging: `DIORAMA_ALLOW_LOCAL_HEAVY_MODELS=1`.
- Do not enable the override while collecting paper data.

## Start Of Day Checklist

1. Start the Elice cloud instance.
2. Confirm the SSH target still matches the current tunnel assignment.
   - Default host: `central-01.tcp.tunnel.elice.io`
   - Default port: `21042`
   - Default user: `elicer`
   - Default key: `key\elice-cloud-ondemand-846e0032-d5dc-4fdd-88fd-1390c9304a5a.pem`
3. From the project root, run:

```powershell
.\scripts\resume_elice_session.ps1 -RequireRemote
```

4. If backend code changed and the instance is reachable, run:

```powershell
.\scripts\resume_elice_session.ps1 -SyncBackend -RequireRemote
```

5. Open the GUI:

```text
http://127.0.0.1:5173
```

6. Confirm the GUI status cards show:
   - Remote A100 reachable
   - HF token detected on remote
   - FLUX cache present
   - Work file count near zero
   - Local heavy execution locked

## Data Collection Rule

Only use `Remote A100` mode for paper-quality outputs. Do not use local `Auto`, `ComfyUI`, or `Real Models Only`; these are intentionally blocked because local generation takes too long and can create inconsistent data.

## First Smoke Run

Use one representative image and conservative settings:

- Backend: `Remote A100`
- Resolution: `256` or `512`
- Steps: `4`
- Guidance: `3.5`
- Strength: `0.35` to `0.45`
- Stage 3.5: enabled
- Stage 4: enabled
- Stage 5: enabled

Confirm the run folder contains:

- `input.png`
- `depth.png`
- `mask_overlay.png`
- `regions/region_plan.json`
- `flux_control.png`
- `flux_result.png`
- `stage35_refinement/stage35_metadata.json`
- `stage4_reconstruction/reconstruction_package.json`
- `stage5_print/print_package.json`
- `run_metadata.json` with `pipeline.stage_status` all true

## Experiment Notes To Record

For each accepted run, record:

- Source image category
- Preset and custom prompt
- Seed
- Resolution
- Steps
- Guidance
- Strength
- Whether foreground/midground/background structure was preserved
- Whether the style transformation is visible without destroying the source layout
- Failure type if applicable

Recommended failure labels:

- composition collapse
- semantic drift
- false water
- texture dominance
- depth mismatch
- over-stylization
- insufficient style change

## If Remote Is Down

Do not collect local replacement data. Continue only with:

- GUI/API improvements
- documentation
- metadata/schema checks
- Demo-mode dry validation when necessary

Run:

```powershell
.\scripts\check_readiness.ps1
```

Remote-down warnings are acceptable while the instance is closed. Local-heavy-model failures are not acceptable.
