param(
  [switch]$Download,
  [string]$HfToken = "",
  [string]$TokenFile = "",
  [switch]$SkipReadiness
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Downloader = Join-Path $RepoRoot "scripts\download_models.py"
$Benchmark = Join-Path $RepoRoot "scripts\benchmark_style_engine.py"
$BenchmarkCheck = Join-Path $RepoRoot "scripts\check_demo_benchmark.py"
$Readiness = Join-Path $RepoRoot "scripts\check_readiness.ps1"
$DefaultTokenFile = Join-Path $RepoRoot "key\hf.txt"

function Write-Step {
  param([string]$Message)
  Write-Host ("[DioramaForge] {0}" -f $Message) -ForegroundColor Cyan
}

if (!(Test-Path -LiteralPath $Python)) {
  throw "Python virtualenv not found: $Python. Run scripts\setup_windows.ps1 first."
}

$env:HF_HOME = Join-Path $RepoRoot "models\huggingface"
$env:HF_HUB_CACHE = Join-Path $RepoRoot "models\huggingface\hub"

$tokenArgs = @()
if ($HfToken.Length -gt 0) {
  $tokenArgs = @("--hf-token", $HfToken)
} else {
  $resolvedTokenFile = if ($TokenFile.Length -gt 0) { $TokenFile } else { $DefaultTokenFile }
  if (Test-Path -LiteralPath $resolvedTokenFile) {
    $tokenArgs = @("--hf-token-file", $resolvedTokenFile)
    Write-Step "Using Hugging Face token file. Token value will not be printed."
  } else {
    Write-Step "No Hugging Face token file found. Public SDXL components may still download; gated components may fail."
  }
}

if (!$SkipReadiness) {
  Write-Step "Current readiness before SDXL preparation."
  powershell -NoProfile -ExecutionPolicy Bypass -File $Readiness
}

Write-Step "Current style-engine dry-run status."
& $Python $Benchmark

if (!$Download) {
  Write-Step "Dry preparation only. Re-run with -Download to fetch SDXL base, depth ControlNet, and Lightning LoRA."
  Write-Host "powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_style_sdxl.ps1 -Download"
  exit 0
}

Write-Step "Downloading SDXL depth-lightning style-engine components into the project Hugging Face cache."
& $Python $Downloader --models style_sdxl @tokenArgs

Write-Step "Readiness after SDXL download."
powershell -NoProfile -ExecutionPolicy Bypass -File $Readiness

Write-Step "Style-engine dry-run after SDXL download."
& $Python $Benchmark

Write-Step "Timed smoke status. Run the benchmark command below before a live demo."
& $Python $BenchmarkCheck
Write-Host ".\.venv\Scripts\python.exe scripts\benchmark_style_engine.py --run --force-engine auto"
Write-Host ".\.venv\Scripts\python.exe scripts\check_demo_benchmark.py --require"
