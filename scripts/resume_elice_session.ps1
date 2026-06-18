param(
  [switch]$SyncBackend,
  [switch]$RequireRemote,
  [int]$WaitSeconds = 45
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$TunnelPort = if ($env:DIORAMA_LOCAL_REMOTE_PORT) { [int]$env:DIORAMA_LOCAL_REMOTE_PORT } else { 9008 }

function Test-RemoteHealth {
  try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$TunnelPort/api/remote/health" -TimeoutSec 3
    return [bool]$health
  } catch {
    return $false
  }
}

Write-Host "[DioramaForge] Resuming Elice remote session."
Write-Host "[DioramaForge] Local remote tunnel port: $TunnelPort"

$tunnel = Get-NetTCPConnection -LocalPort $TunnelPort -State Listen -ErrorAction SilentlyContinue
if (-not $tunnel) {
  Write-Host "[DioramaForge] Opening SSH tunnel in a background PowerShell window."
  Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $RepoRoot "scripts\connect_elice_tunnel.ps1")) `
    -WindowStyle Hidden | Out-Null
} else {
  Write-Host "[DioramaForge] SSH tunnel already appears to be listening."
}

$deadline = (Get-Date).AddSeconds($WaitSeconds)
while ((Get-Date) -lt $deadline) {
  if (Test-RemoteHealth) {
    Write-Host "[DioramaForge] Remote backend health endpoint is reachable."
    break
  }
  Start-Sleep -Seconds 2
}

if (-not (Test-RemoteHealth)) {
  $message = "[DioramaForge] Remote backend is not reachable yet. Start the Elice instance and backend, then retry."
  if ($RequireRemote) {
    throw $message
  }
  Write-Warning $message
} elseif ($SyncBackend) {
  Write-Host "[DioramaForge] Syncing local backend files and restarting remote backend."
  & (Join-Path $RepoRoot "scripts\sync_elice_backend.ps1")
}

Write-Host "[DioramaForge] Running readiness check."
$checkArgs = @()
if ($RequireRemote) {
  $checkArgs += "-RequireRemote"
}
& (Join-Path $RepoRoot "scripts\check_readiness.ps1") @checkArgs
