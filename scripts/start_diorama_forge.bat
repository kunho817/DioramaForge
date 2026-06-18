@echo off
setlocal

set "ROOT=%~dp0.."
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
set "DESKTOP=%ROOT%\desktop"
set "ENV_FILE=%ROOT%\.env.local"

if exist "%ENV_FILE%" (
  echo [DioramaForge] Loading local environment from %ENV_FILE%
  for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
    if not "%%A"=="" set "%%A=%%B"
  )
) else (
  echo [DioramaForge] .env.local not found. Using the current shell environment.
)

if not exist "%PYTHON%" (
  echo [DioramaForge] Python venv not found: %PYTHON%
  echo Run scripts\setup_windows.ps1 first.
  pause
  exit /b 1
)

if defined COMFYUI_DIR (
  if exist "%COMFYUI_DIR%\main.py" (
    echo [DioramaForge] Starting ComfyUI on http://127.0.0.1:8188
    start "ComfyUI" /D "%COMFYUI_DIR%" cmd /k "python main.py --listen 127.0.0.1 --port 8188"
    timeout /t 5 /nobreak > nul
  ) else (
    echo [DioramaForge] COMFYUI_DIR is set but main.py was not found: %COMFYUI_DIR%
  )
) else (
  echo [DioramaForge] COMFYUI_DIR is not set. Reusing an already running ComfyUI server if available.
)

if "%DIORAMA_ENABLE_REMOTE_TUNNEL%"=="1" (
  if exist "%ROOT%\key\elice-cloud-ondemand-846e0032-d5dc-4fdd-88fd-1390c9304a5a.pem" (
    echo [DioramaForge] Starting legacy Remote A100 tunnel on http://127.0.0.1:9008
    start "DioramaForge Remote Tunnel" powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\connect_elice_tunnel.ps1"
    timeout /t 3 /nobreak > nul
  ) else (
    echo [DioramaForge] DIORAMA_ENABLE_REMOTE_TUNNEL=1 but Elice SSH key was not found.
  )
) else (
  echo [DioramaForge] Legacy remote tunnel is disabled. Set DIORAMA_ENABLE_REMOTE_TUNNEL=1 to open it explicitly.
)

echo [DioramaForge] Starting local API server on http://127.0.0.1:8008
start "DioramaForge API" /D "%ROOT%" cmd /k ""%PYTHON%" api_app.py --host 127.0.0.1 --port 8008"

timeout /t 3 /nobreak > nul

if not exist "%DESKTOP%\node_modules" (
  echo [DioramaForge] Installing desktop dependencies. This runs only when node_modules is missing.
  pushd "%DESKTOP%"
  call npm install
  if errorlevel 1 (
    echo [DioramaForge] npm install failed.
    popd
    pause
    exit /b 1
  )
  popd
)

echo [DioramaForge] Starting Tauri desktop shell.
start "DioramaForge Desktop" /D "%DESKTOP%" cmd /k "npm run tauri:dev"

echo [DioramaForge] API and desktop launch commands were started.
endlocal
