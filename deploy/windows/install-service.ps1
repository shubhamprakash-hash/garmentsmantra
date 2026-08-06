<#
install-service.ps1
====================
Installs the Garments Mantra forecasting service as a Windows Service using
NSSM (Non-Sucking Service Manager), so it starts automatically on server
boot and restarts itself if it crashes — the Windows equivalent of what
Render's platform did for you automatically.

PREREQUISITES (do these once, manually, before running this script):
  1. Install Python 3.11+ on the server, and make sure `python` is on PATH.
  2. Copy this whole project folder to the server, e.g. C:\apps\garments-forecast
  3. Open PowerShell AS ADMINISTRATOR in that folder and run:
         python -m venv venv
         .\venv\Scripts\pip install -r requirements.txt
  4. Download NSSM from https://nssm.cc/download, extract it, and put
     nssm.exe somewhere on PATH (e.g. C:\tools\nssm\nssm.exe) — or pass
     -NssmPath pointing at it when you run this script.
  5. Set the live-data env vars for the service account that will run it
     (see set-env-vars.ps1 in this folder) BEFORE installing the service,
     since NSSM captures env vars at install time.

USAGE (run as Administrator):
    .\install-service.ps1
    .\install-service.ps1 -NssmPath "C:\tools\nssm\nssm.exe" -Port 8000

WHAT THIS DOES:
  - Registers a service named "GarmentsMantraForecast"
  - Runs:  <project>\venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port <Port>
  - Sets it to auto-start on boot and restart on failure
  - Logs stdout/stderr to <project>\logs\service.log / service-error.log

AFTER INSTALLING:
    Start it with:   nssm start GarmentsMantraForecast
    Stop it with:    nssm stop GarmentsMantraForecast
    Remove it with:  nssm remove GarmentsMantraForecast confirm
    Check status:    Get-Service GarmentsMantraForecast
#>

param(
    [string]$ServiceName = "GarmentsMantraForecast",
    [string]$ProjectDir  = (Get-Location).Path,
    [string]$NssmPath    = "nssm.exe",
    [int]$Port           = 8000
)

$ErrorActionPreference = "Stop"

$PythonExe = Join-Path $ProjectDir "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Could not find $PythonExe — create the venv and install requirements first (see the header of this script)."
}

$LogDir = Join-Path $ProjectDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "Installing service '$ServiceName' ..."
& $NssmPath install $ServiceName $PythonExe "-m uvicorn app:app --host 0.0.0.0 --port $Port"
& $NssmPath set $ServiceName AppDirectory $ProjectDir
& $NssmPath set $ServiceName AppStdout (Join-Path $LogDir "service.log")
& $NssmPath set $ServiceName AppStderr (Join-Path $LogDir "service-error.log")
& $NssmPath set $ServiceName Start SERVICE_AUTO_START
& $NssmPath set $ServiceName AppRestartDelay 5000   # ms — restart 5s after a crash

Write-Host ""
Write-Host "Service '$ServiceName' installed. Before starting it, make sure the" -ForegroundColor Yellow
Write-Host "live-data env vars are set (see set-env-vars.ps1), then run:" -ForegroundColor Yellow
Write-Host "    nssm start $ServiceName" -ForegroundColor Yellow
Write-Host "It will listen on http://localhost:$Port  (and on the server's" -ForegroundColor Yellow
Write-Host "hostname/IP on port $Port for anyone else on the network)." -ForegroundColor Yellow
