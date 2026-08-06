<#
set-env-vars.ps1
=================
Sets the live-data env vars as SYSTEM (machine-level) environment variables,
so the Windows Service picks them up regardless of which account runs it.

Run this ONCE, as Administrator, BEFORE installing/starting the service
(install-service.ps1). If you change a value later, re-run this script and
then restart the service (nssm restart GarmentsMantraForecast) — env var
changes don't apply to an already-running service.

Fill in the real values below once the .NET team confirms them, then run:
    .\set-env-vars.ps1
#>

[System.Environment]::SetEnvironmentVariable("GM_API_BASE_URL", "https://REPLACE-ME.internal", "Machine")
[System.Environment]::SetEnvironmentVariable("GM_API_KEY", "REPLACE-ME", "Machine")

# Only set these if they differ from the defaults:
# [System.Environment]::SetEnvironmentVariable("GM_API_ENDPOINT", "/api/v1/GetSalesHistory", "Machine")
# [System.Environment]::SetEnvironmentVariable("GM_API_AUTH_MODE", "api_key", "Machine")  # api_key | bearer | none

Write-Host "Env vars set at machine level. A server reboot, or re-login, may be" -ForegroundColor Yellow
Write-Host "needed for other already-open sessions to see them. If the service" -ForegroundColor Yellow
Write-Host "is already running, restart it: nssm restart GarmentsMantraForecast" -ForegroundColor Yellow
