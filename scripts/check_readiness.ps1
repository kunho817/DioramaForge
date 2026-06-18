param(
  [string]$ApiBase = "http://127.0.0.1:8008",
  [switch]$RequireRemote
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DefaultKey = Join-Path $RepoRoot "key\elice-cloud-ondemand-846e0032-d5dc-4fdd-88fd-1390c9304a5a.pem"
$HfTokenFile = Join-Path $RepoRoot "key\hf.txt"
$EnvLocalFile = Join-Path $RepoRoot ".env.local"

$script:Failures = 0
$script:Warnings = 0

function Write-Check {
  param(
    [string]$Status,
    [string]$Message
  )
  $color = switch ($Status) {
    "PASS" { "Green" }
    "WARN" { "Yellow" }
    "FAIL" { "Red" }
    default { "White" }
  }
  Write-Host ("[{0}] {1}" -f $Status, $Message) -ForegroundColor $color
  if ($Status -eq "FAIL") { $script:Failures += 1 }
  if ($Status -eq "WARN") { $script:Warnings += 1 }
}

function Invoke-Json {
  param(
    [string]$Uri,
    [int]$TimeoutSec = 5
  )
  try {
    return Invoke-RestMethod -Uri $Uri -TimeoutSec $TimeoutSec
  } catch {
    return $null
  }
}

function Import-DotEnvLocal {
  param([string]$Path)
  if (!(Test-Path -LiteralPath $Path)) {
    return $false
  }
  $lines = Get-Content -LiteralPath $Path -ErrorAction Stop
  foreach ($line in $lines) {
    $trimmed = $line.Trim()
    if (!$trimmed -or $trimmed.StartsWith("#")) {
      continue
    }
    $equals = $trimmed.IndexOf("=")
    if ($equals -le 0) {
      continue
    }
    $name = $trimmed.Substring(0, $equals).Trim()
    $value = $trimmed.Substring($equals + 1).Trim()
    if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
      $value = $value.Substring(1, $value.Length - 2)
    }
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
  }
  return $true
}

function Get-HfSnapshotFile {
  param(
    [string]$ModelId,
    [string]$ExpectedFile
  )
  $modelFolder = "models--" + $ModelId.Replace("/", "--")
  $snapshotsRoot = Join-Path $RepoRoot ("models\huggingface\hub\" + $modelFolder + "\snapshots")
  if (!(Test-Path -LiteralPath $snapshotsRoot)) {
    return $null
  }
  $snapshots = Get-ChildItem -LiteralPath $snapshotsRoot -Directory -ErrorAction SilentlyContinue
  foreach ($snapshot in $snapshots) {
    $candidate = Join-Path $snapshot.FullName $ExpectedFile
    if (Test-Path -LiteralPath $candidate) {
      return $candidate
    }
    $recursive = Get-ChildItem -LiteralPath $snapshot.FullName -Recurse -File -Filter $ExpectedFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($recursive) {
      return $recursive.FullName
    }
  }
  return $null
}

function Write-CacheCheck {
  param(
    [string]$Label,
    [string]$ModelId,
    [string]$ExpectedFile,
    [bool]$Required
  )
  $found = Get-HfSnapshotFile -ModelId $ModelId -ExpectedFile $ExpectedFile
  if ($found) {
    Write-Check "PASS" "$Label cache found: $ModelId"
    return $true
  }
  if ($Required) {
    Write-Check "FAIL" "$Label cache missing: $ModelId ($ExpectedFile)"
  } else {
    Write-Check "PASS" "$Label cache optional or not active: $ModelId"
  }
  return $false
}

Write-Host "DioramaForge readiness check"
Write-Host ("Repo: {0}" -f $RepoRoot)
Write-Host ""

if (Import-DotEnvLocal -Path $EnvLocalFile) {
  Write-Check "PASS" ".env.local loaded for this readiness check. Secret values were not printed."
} else {
  Write-Check "WARN" ".env.local not found. Copy .env.example to .env.local before the live demo if Meshy or ComfyUI paths need local variables."
}

if (Test-Path -LiteralPath (Join-Path $RepoRoot ".venv\Scripts\python.exe")) {
  Write-Check "PASS" "Python virtualenv found."
  $PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
} else {
  Write-Check "FAIL" "Python virtualenv missing. Run scripts\setup_windows.ps1."
  $PythonExe = $null
}

if (Test-Path -LiteralPath (Join-Path $RepoRoot "desktop\node_modules")) {
  Write-Check "PASS" "Desktop node_modules found."
} else {
  Write-Check "WARN" "Desktop node_modules missing. start_diorama_forge.bat can install it."
}

if (Test-Path -LiteralPath $DefaultKey) {
  Write-Check "PASS" "Optional legacy Elice SSH key found."
} else {
  if ($RequireRemote) {
    Write-Check "WARN" "Elice SSH key not found. Remote A100 cannot be opened from this PC."
  } else {
    Write-Check "PASS" "Optional legacy Elice SSH key not configured."
  }
}

if (Test-Path -LiteralPath $HfTokenFile) {
  Write-Check "PASS" "Optional HF token file exists. Token value was not printed."
} else {
  if ($RequireRemote) {
    Write-Check "WARN" "key\hf.txt not found. Remote setup may need a HuggingFace token."
  } else {
    Write-Check "PASS" "Optional HF token file not configured."
  }
}

if ($env:DIORAMA_ALLOW_LOCAL_HEAVY_MODELS -and $env:DIORAMA_ALLOW_LOCAL_HEAVY_MODELS -match "^(1|true|yes|on)$") {
  Write-Check "PASS" "DIORAMA_ALLOW_LOCAL_HEAVY_MODELS override is enabled."
} else {
  Write-Check "PASS" "No local-heavy override is set; config/default.json controls local execution."
}

$configPath = Join-Path $RepoRoot "configs\default.json"
if (Test-Path -LiteralPath $configPath) {
  $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
  $activeStyle = [string]$config.style_engine.active
  $styleBackend = [string]$config.style_engine.backend_mode
  Write-Check "PASS" "Configured style engine: $activeStyle"
  Write-Check "PASS" "Configured image backend: $styleBackend"
  if ($config.product_pipeline) {
    Write-Check "PASS" ("Generate defaults: {0}px, {1} steps, guidance {2}, strength {3}." -f `
      $config.product_pipeline.max_resolution, `
      $config.product_pipeline.steps, `
      $config.product_pipeline.guidance, `
      $config.product_pipeline.strength)
    if ([string]$config.product_pipeline.stage4_backend_mode -eq "meshy") {
      Write-Check "PASS" "Stage 4 is configured for Meshy AI image-to-3D."
    }
  } else {
    Write-Check "WARN" "product_pipeline defaults are missing from configs\default.json."
  }
  if ($config.meshy_ai -and $config.meshy_ai.enabled) {
    $meshyEnv = [string]$config.meshy_ai.api_key_env
    if ([Environment]::GetEnvironmentVariable($meshyEnv)) {
      Write-Check "PASS" "Meshy API key environment variable is set: $meshyEnv"
    } else {
      Write-Check "WARN" "Meshy API key environment variable is not set: $meshyEnv"
    }
  } else {
    Write-Check "WARN" "Meshy AI backend is disabled or missing from configs\default.json."
  }
  if ($styleBackend -match "^(comfy|comfyui|comfyui backend)$") {
    $stage3Workflow = [string]$config.comfyui.stage3_workflow
    if (![System.IO.Path]::IsPathRooted($stage3Workflow)) {
      $stage3Workflow = Join-Path $RepoRoot $stage3Workflow
    }
    if (Test-Path -LiteralPath $stage3Workflow) {
      Write-Check "PASS" "ComfyUI Stage 3 workflow found: $stage3Workflow"
    } else {
      Write-Check "WARN" "ComfyUI Stage 3 workflow missing: $stage3Workflow"
    }
    $resolvedStyle = "comfyui"
  } elseif ($activeStyle -match "^(sdxl_depth_lightning|sdxl_lightning|sdxl)$") {
    $sdxlBaseReady = Write-CacheCheck -Label "SDXL base" -ModelId $config.models.sdxl_depth_lightning.base_model_id -ExpectedFile "model_index.json" -Required $true
    $sdxlControlReady = Write-CacheCheck -Label "SDXL depth ControlNet" -ModelId $config.models.sdxl_depth_lightning.controlnet_model_id -ExpectedFile "config.json" -Required $true
    $sdxlLoraReady = Write-CacheCheck -Label "SDXL Lightning LoRA" -ModelId $config.models.sdxl_depth_lightning.lora_model_id -ExpectedFile $config.models.sdxl_depth_lightning.lora_weight_name -Required $true
    $resolvedStyle = "sdxl_depth_lightning"
  } else {
    $fluxReady = Write-CacheCheck -Label "Current FLUX style engine" -ModelId $config.models.flux.model_id -ExpectedFile "model_index.json" -Required $false
    $sdxlBaseReady = Write-CacheCheck -Label "SDXL base candidate" -ModelId $config.models.sdxl_depth_lightning.base_model_id -ExpectedFile "model_index.json" -Required $false
    $sdxlControlReady = Write-CacheCheck -Label "SDXL depth ControlNet candidate" -ModelId $config.models.sdxl_depth_lightning.controlnet_model_id -ExpectedFile "config.json" -Required $false
    $sdxlLoraReady = Write-CacheCheck -Label "SDXL Lightning LoRA candidate" -ModelId $config.models.sdxl_depth_lightning.lora_model_id -ExpectedFile $config.models.sdxl_depth_lightning.lora_weight_name -Required $false
    $resolvedStyle = if ($activeStyle -match "^(auto|best|auto_fast)?$" -and $sdxlBaseReady -and $sdxlControlReady -and $sdxlLoraReady) { "sdxl_depth_lightning" } else { "flux_depth" }
  }
  Write-Check "PASS" "Resolved style engine: $resolvedStyle"
  if ($styleBackend -match "^(comfy|comfyui|comfyui backend)$") {
    Write-Check "PASS" "Generate is configured to use the internal ComfyUI Stage 3 image backend."
  } elseif ($activeStyle -match "^(auto|best|auto_fast)?$" -and $resolvedStyle -ne "sdxl_depth_lightning") {
    Write-Check "WARN" "Fast local image backend is not ready; Generate will use the configured fallback style engine."
  }
} else {
  Write-Check "FAIL" "configs\default.json not found."
}

if ($PythonExe) {
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_env_local_contract.py") | Out-Host
    Write-Check "PASS" ".env.local loading contract passed."
  } catch {
    Write-Check "FAIL" ".env.local loading contract failed."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_api_contract.py") | Out-Host
    Write-Check "PASS" "API contract check passed without starting the server."
  } catch {
    Write-Check "FAIL" "API contract check failed. Run scripts\check_api_contract.py for details."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_product_speed_policy.py") | Out-Host
    Write-Check "PASS" "Product speed/mode policy contract passed."
  } catch {
    Write-Check "FAIL" "Product speed/mode policy contract failed."
  }
  try {
    $workflowRaw = & $PythonExe (Join-Path $RepoRoot "scripts\check_comfy_workflows.py") | Out-String
    $workflowStatus = $workflowRaw | ConvertFrom-Json
    if ($workflowStatus.stage3.validation.ok) {
      Write-Check "PASS" ("ComfyUI Stage 3 workflow contract is valid: {0}" -f $workflowStatus.stage3.path)
    } else {
      $workflowErrors = @($workflowStatus.stage3.validation.errors)
      $workflowReason = if ($workflowErrors.Count -gt 0) { $workflowErrors[0] } else { "workflow contract invalid" }
      Write-Check "WARN" ("ComfyUI Stage 3 workflow contract is not ready: {0}" -f $workflowReason)
    }
    if ($workflowStatus.required.stage35 -and -not $workflowStatus.stage35.validation.ok) {
      $stage35Errors = @($workflowStatus.stage35.validation.errors)
      $stage35Reason = if ($stage35Errors.Count -gt 0) { $stage35Errors[0] } else { "workflow contract invalid" }
      Write-Check "WARN" ("ComfyUI Stage 3.5 workflow contract is not ready: {0}" -f $stage35Reason)
    }
  } catch {
    Write-Check "WARN" "ComfyUI workflow contract check could not run."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_comfy_workflow_installer.py") | Out-Host
    Write-Check "PASS" "ComfyUI workflow installer contract passed."
  } catch {
    Write-Check "FAIL" "ComfyUI workflow installer contract failed."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_comfy_workflow_inspector.py") | Out-Host
    Write-Check "PASS" "ComfyUI workflow inspector contract passed."
  } catch {
    Write-Check "FAIL" "ComfyUI workflow inspector contract failed."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_comfy_workflow_preparer.py") | Out-Host
    Write-Check "PASS" "ComfyUI workflow preparer contract passed."
  } catch {
    Write-Check "FAIL" "ComfyUI workflow preparer contract failed."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_comfy_workflow_examples.py") | Out-Host
    Write-Check "PASS" "ComfyUI workflow examples contract passed."
  } catch {
    Write-Check "FAIL" "ComfyUI workflow examples contract failed."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\install_comfy_example_workflow.py") | Out-Host
    Write-Check "PASS" "ComfyUI example install dry-run passed."
  } catch {
    Write-Check "FAIL" "ComfyUI example install dry-run failed."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_comfy_node_compatibility_contract.py") | Out-Host
    Write-Check "PASS" "ComfyUI node/model compatibility contract passed."
  } catch {
    Write-Check "FAIL" "ComfyUI node/model compatibility contract failed."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_comfy_model_patch_contract.py") | Out-Host
    Write-Check "PASS" "ComfyUI workflow model patch contract passed."
  } catch {
    Write-Check "FAIL" "ComfyUI workflow model patch contract failed."
  }
  try {
    $nodeCompatRaw = & $PythonExe (Join-Path $RepoRoot "scripts\check_comfy_node_compatibility.py") | Out-String
    $nodeCompat = $nodeCompatRaw | ConvertFrom-Json
    if ($nodeCompat.ok) {
      Write-Check "PASS" "ComfyUI node/model compatibility passed."
    } elseif ($nodeCompat.server_ok) {
      $nodeReason = if ($nodeCompat.failures -and $nodeCompat.failures.Count -gt 0) { $nodeCompat.failures[0] } else { "workflow nodes or model choices are not compatible" }
      Write-Check "WARN" ("ComfyUI node/model compatibility is not ready: {0}" -f $nodeReason)
    } else {
      Write-Check "WARN" "ComfyUI node/model compatibility could not be checked because the server is not reachable."
    }
  } catch {
    Write-Check "WARN" "ComfyUI node/model compatibility check could not run."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_desktop_contract.py") | Out-Host
    Write-Check "PASS" "Desktop UI contract check passed."
  } catch {
    Write-Check "FAIL" "Desktop UI contract check failed. Run scripts\check_desktop_contract.py for details."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_job_progress_contract.py") | Out-Host
    Write-Check "PASS" "Job progress contract passed."
  } catch {
    Write-Check "FAIL" "Job progress contract failed. Run scripts\check_job_progress_contract.py for details."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_prompt_contract.py") | Out-Host
    Write-Check "PASS" "Model-specific prompt contract passed."
  } catch {
    Write-Check "FAIL" "Model-specific prompt contract failed. Run scripts\check_prompt_contract.py for details."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_meshy_contract.py") | Out-Host
    Write-Check "PASS" "Meshy API contract check passed without network calls."
  } catch {
    Write-Check "FAIL" "Meshy API contract check failed. Run scripts\check_meshy_contract.py for details."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_validation_meshy_contract.py") | Out-Host
    Write-Check "PASS" "Meshy Stage 4/5 validation contract passed."
  } catch {
    Write-Check "FAIL" "Meshy Stage 4/5 validation contract failed. Run scripts\check_validation_meshy_contract.py for details."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_refinement_contract.py") | Out-Host
    Write-Check "PASS" "Result refinement contract passed."
  } catch {
    Write-Check "FAIL" "Result refinement contract failed. Run scripts\check_refinement_contract.py for details."
  }
  try {
    & $PythonExe (Join-Path $RepoRoot "scripts\check_demo_benchmark_contract.py") | Out-Host
    Write-Check "PASS" "Timed smoke benchmark gate contract passed."
  } catch {
    Write-Check "FAIL" "Timed smoke benchmark gate contract failed. Run scripts\check_demo_benchmark_contract.py for details."
  }
  try {
    $runtimeRaw = & $PythonExe (Join-Path $RepoRoot "scripts\check_demo_runtime.py") | Out-String
    $runtimeStatus = $runtimeRaw | ConvertFrom-Json
    if ($runtimeStatus.ready) {
      Write-Check "PASS" ("Demo runtime ready: {0}, {1} GB free VRAM." -f `
        $runtimeStatus.torch.device_name, `
        $runtimeStatus.torch.free_vram_gb)
    } else {
      $reason = if ($runtimeStatus.failed_checks -and $runtimeStatus.failed_checks.Count -gt 0) { $runtimeStatus.failed_checks -join ", " } else { "runtime preflight failed" }
      Write-Check "WARN" ("Demo runtime not ready: {0}" -f $reason)
    }
  } catch {
    Write-Check "WARN" "Demo runtime preflight could not run."
  }
  try {
    $benchmarkRaw = & $PythonExe (Join-Path $RepoRoot "scripts\check_demo_benchmark.py") | Out-String
    $benchmarkStatus = $benchmarkRaw | ConvertFrom-Json
    if ($benchmarkStatus.verified) {
      Write-Check "PASS" ("Timed smoke benchmark verified: {0}s / {1}s using {2}." -f `
        $benchmarkStatus.elapsed_seconds, `
        $benchmarkStatus.max_seconds, `
        $benchmarkStatus.resolved_engine)
    } else {
      $reason = if ($benchmarkStatus.failures -and $benchmarkStatus.failures.Count -gt 0) { $benchmarkStatus.failures[0] } else { "no verified timed benchmark" }
      Write-Check "WARN" ("Timed smoke benchmark not verified: {0}" -f $reason)
    }
  } catch {
    Write-Check "WARN" "Timed smoke benchmark check could not run."
  }
}

$apiHealth = Invoke-Json -Uri "$ApiBase/api/health" -TimeoutSec 3
if ($apiHealth -and $apiHealth.ok) {
  Write-Check "PASS" "Local API is reachable at $ApiBase."
} else {
  Write-Check "WARN" "Local API is not reachable at $ApiBase. Start it with scripts\start_diorama_forge.bat or api_app.py."
}

$policy = Invoke-Json -Uri "$ApiBase/api/execution/policy" -TimeoutSec 3
if ($policy) {
  if ($policy.allow_local_heavy_models) {
    Write-Check "PASS" "API policy allows local model execution. Allowed backends: $($policy.allowed_backends -join ', ')."
  } else {
    Write-Check "WARN" "API policy blocks local heavy models. Local FLUX will not run until the policy is changed."
  }
} elseif ($apiHealth) {
  Write-Check "WARN" "API is up, but execution policy endpoint did not respond."
}

$remote = Invoke-Json -Uri "$ApiBase/api/remote/status" -TimeoutSec 5
if ($remote -and $remote.ok) {
  Write-Check "PASS" "Remote backend reachable at $($remote.base_url)."
  if ($remote.runtime) {
    Write-Host ("      Runtime: {0}" -f $remote.runtime)
  }
  if ($remote.hf) {
    Write-Host ("      HF cache: {0} GB" -f $remote.hf.cache_gb)
  }
  if ($remote.work) {
    Write-Host ("      Remote work files: {0}, cleanup={1}" -f $remote.work.file_count, $remote.cleanup_after_response)
  }
  if ($remote.models) {
    foreach ($name in @("depth", "sam", "flux")) {
      $item = $remote.models.$name
      if ($item) {
        $status = if ($item.ready) { "ready" } else { "not ready" }
        Write-Host ("      {0}: {1}, cached={2}, package={3}" -f $name, $status, $item.cached, $item.package_ready)
      }
    }
  }
} else {
  $message = "Remote backend is not reachable. Remote is optional in the current local-first plan."
  if ($RequireRemote) {
    Write-Check "FAIL" $message
  } else {
    Write-Check "PASS" $message
  }
}

$apiPort = Get-NetTCPConnection -LocalPort 8008 -State Listen -ErrorAction SilentlyContinue
if ($apiPort) {
  Write-Check "PASS" "Port 8008 is listening."
} else {
  Write-Check "WARN" "Port 8008 is not listening."
}

$guiPort = Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue
if ($guiPort) {
  Write-Check "PASS" "Port 5173 is listening."
} else {
  Write-Check "WARN" "Port 5173 is not listening."
}

Write-Host ""
Write-Host ("Summary: {0} failure(s), {1} warning(s)" -f $script:Failures, $script:Warnings)
if ($script:Failures -gt 0) {
  exit 1
}
