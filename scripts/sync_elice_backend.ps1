$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$DefaultKey = Join-Path $RepoRoot "key\elice-cloud-ondemand-846e0032-d5dc-4fdd-88fd-1390c9304a5a.pem"

$KeyPath = if ($env:ELICE_SSH_KEY) { $env:ELICE_SSH_KEY } else { $DefaultKey }
$HostName = if ($env:ELICE_SSH_HOST) { $env:ELICE_SSH_HOST } else { "central-01.tcp.tunnel.elice.io" }
$Port = if ($env:ELICE_SSH_PORT) { $env:ELICE_SSH_PORT } else { "21042" }
$User = if ($env:ELICE_SSH_USER) { $env:ELICE_SSH_USER } else { "elicer" }
$RemoteRoot = if ($env:DIORAMA_REMOTE_ROOT) { $env:DIORAMA_REMOTE_ROOT } else { "~/DioramaForge" }

if (-not (Test-Path -LiteralPath $KeyPath)) {
  throw "SSH key not found: $KeyPath"
}

Push-Location $RepoRoot
try {
  Write-Host "[DioramaForge] Syncing backend files to ${User}@${HostName}:${RemoteRoot}"
  scp -i $KeyPath -P $Port -r `
    src `
    configs `
    scripts `
    workflows `
    requirements.txt `
    requirements-optional-models.txt `
    model_backend_app.py `
    api_app.py `
    app.py `
    README.md `
    "${User}@${HostName}:${RemoteRoot}/"

  Write-Host "[DioramaForge] Restarting remote model backend"
  ssh -i $KeyPath -p $Port -o BatchMode=yes "${User}@${HostName}" "cd ${RemoteRoot} && chmod +x scripts/*.sh && bash scripts/restart_model_backend.sh"
} finally {
  Pop-Location
}
