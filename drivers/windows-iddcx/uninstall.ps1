# uninstall.ps1 - Windows IddCx Driver Uninstaller
# Run as Administrator in PowerShell

$ErrorActionPreference = "Stop"

Write-Host "=== Uninstalling vkremote Virtual GPU Adapter ===" -ForegroundColor Yellow

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "This script must be executed in an elevated PowerShell session (Run as Administrator)."
    exit 1
}

$devcon = (Get-ChildItem -Path "C:\Program Files (x86)\Windows Kits" -Filter "devcon.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if ($devcon) {
    & $devcon remove "Root\vkremote_IddCx"
}

# Find published INF name and delete
$publishedInfs = pnputil.exe /enum-drivers
$oemInf = $null
$lines = $publishedInfs -split "`r?`n"
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match "vkremote-iddcx\.inf") {
        # The preceding or subsequent line has the Published Name: oemXX.inf
        for ($j = [Math]::Max(0, $i - 5); $j -le [Math]::Min($lines.Count - 1, $i + 5); $j++) {
            if ($lines[$j] -match "Published Name:\s+(oem\d+\.inf)") {
                $oemInf = $matches[1]
                break
            }
        }
        break
    }
}

if ($oemInf) {
    Write-Host "Removing driver package $oemInf..." -ForegroundColor Green
    & pnputil.exe /delete-driver $oemInf /uninstall /force
} else {
    Write-Host "Driver package already removed or not found."
}

Write-Host "Uninstallation complete." -ForegroundColor Green
