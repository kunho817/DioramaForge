$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$DefaultKey = Join-Path $RepoRoot "key\elice-cloud-ondemand-846e0032-d5dc-4fdd-88fd-1390c9304a5a.pem"

$KeyPath = if ($env:ELICE_SSH_KEY) { $env:ELICE_SSH_KEY } else { $DefaultKey }
$HostName = if ($env:ELICE_SSH_HOST) { $env:ELICE_SSH_HOST } else { "central-01.tcp.tunnel.elice.io" }
$Port = if ($env:ELICE_SSH_PORT) { $env:ELICE_SSH_PORT } else { "21042" }
$User = if ($env:ELICE_SSH_USER) { $env:ELICE_SSH_USER } else { "elicer" }
$LocalPort = if ($env:DIORAMA_LOCAL_REMOTE_PORT) { $env:DIORAMA_LOCAL_REMOTE_PORT } else { "9008" }
$RemotePort = if ($env:DIORAMA_MODEL_BACKEND_PORT) { $env:DIORAMA_MODEL_BACKEND_PORT } else { "9008" }

if (-not (Test-Path -LiteralPath $KeyPath)) {
  throw "SSH key not found: $KeyPath"
}

$ExistingTunnel = Get-NetTCPConnection -LocalPort ([int]$LocalPort) -State Listen -ErrorAction SilentlyContinue
if ($ExistingTunnel) {
  Write-Host "[DioramaForge] Local remote backend tunnel already appears to be listening on port $LocalPort."
  exit 0
}

ssh `
  -i $KeyPath `
  -N `
  -L "${LocalPort}:127.0.0.1:${RemotePort}" `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 `
  -o ServerAliveCountMax=3 `
  "${User}@${HostName}" `
  -p $Port
