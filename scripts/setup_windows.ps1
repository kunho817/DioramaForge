param(
    [switch]$UseSystemSitePackages,
    [switch]$InstallOptionalModels,
    [switch]$DownloadDepthAndSam,
    [switch]$DownloadFlux,
    [switch]$DownloadStyleSdxl,
    [string]$HfToken = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$DefaultTokenFile = Join-Path $Root "key\hf.txt"

if (!(Test-Path $Python)) {
    if ($UseSystemSitePackages) {
        python -m venv --system-site-packages $Venv
    } else {
        python -m venv $Venv
    }
}

& $Python -m pip install --upgrade pip setuptools wheel
& $Python -m pip install -r (Join-Path $Root "requirements.txt")

if ($InstallOptionalModels) {
    & $Python -m pip install -r (Join-Path $Root "requirements-optional-models.txt")
}

if ($DownloadDepthAndSam -or $DownloadFlux -or $DownloadStyleSdxl) {
    $env:HF_HOME = Join-Path $Root "models\huggingface"
    if ($HfToken.Length -gt 0) {
        $env:HF_TOKEN = $HfToken
    } elseif (Test-Path -LiteralPath $DefaultTokenFile) {
        $env:HF_TOKEN = (Get-Content -LiteralPath $DefaultTokenFile -Raw).Trim()
    }

    if ($DownloadDepthAndSam) {
        & $Python (Join-Path $Root "scripts\download_models.py") --models depth sam
    }

    if ($DownloadFlux) {
        if ($HfToken.Length -gt 0) {
            & $Python (Join-Path $Root "scripts\download_models.py") --models flux --hf-token $HfToken
        } else {
            & $Python (Join-Path $Root "scripts\download_models.py") --models flux
        }
    }

    if ($DownloadStyleSdxl) {
        if ($HfToken.Length -gt 0) {
            & $Python (Join-Path $Root "scripts\download_models.py") --models style_sdxl --hf-token $HfToken
        } else {
            & $Python (Join-Path $Root "scripts\download_models.py") --models style_sdxl
        }
    }
}

& $Python (Join-Path $Root "scripts\check_models.py")
