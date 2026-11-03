# install.ps1 - Automated Windows IddCx Driver Installer
# Run as Administrator in PowerShell

param (
    [switch]$ForceCertificateCreation = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=== vkremote Windows Virtual GPU (IddCx) Installer ===" -ForegroundColor Cyan

# 1. Require Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "This script must be executed in an elevated PowerShell session (Run as Administrator)."
    exit 1
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# 2. Check TestSigning
$bcdOutput = bcdedit /enum "{current}" | Out-String
if ($bcdOutput -notmatch "testsigning\s+Yes") {
    Write-Warning "Windows Test Signing is not currently active."
    Write-Host "Enabling Windows Test Signing: bcdedit /set testsigning on" -ForegroundColor Yellow
    bcdedit /set testsigning on
    Write-Warning "NOTE: A reboot is required after enabling testsigning if unsigned drivers are rejected."
}

# 3. Create or reuse self-signed certificate
$CertSubject = "CN=vkremote Virtual GPU Driver"
$Cert = Get-ChildItem Cert:\LocalMachine\Root | Where-Object { $_.Subject -eq $CertSubject } | Select-Object -First 1

if (-not $Cert -or $ForceCertificateCreation) {
    Write-Host "Creating self-signed code-signing certificate..." -ForegroundColor Green
    $Cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject $CertSubject -CertStoreLocation Cert:\LocalMachine\My
    
    # Export and add to Trusted Root and Trusted Publisher
    $certPath = Join-Path $ScriptDir "vkremote-driver.cer"
    Export-Certificate -Cert $Cert -FilePath $certPath | Out-Null

    Import-Certificate -FilePath $certPath -CertStoreLocation Cert:\LocalMachine\Root | Out-Null
    Import-Certificate -FilePath $certPath -CertStoreLocation Cert:\LocalMachine\TrustedPublisher | Out-Null
    Write-Host "Certificate installed into LocalMachine\Root and LocalMachine\TrustedPublisher." -ForegroundColor Green
}

# 4. Sign Driver Files if signtool is present
$signtool = (Get-ChildItem -Path "C:\Program Files (x86)\Windows Kits" -Filter "signtool.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1).FullName

if ($signtool) {
    Write-Host "Signing driver files with signtool..." -ForegroundColor Green
    $dllPath = Join-Path $ScriptDir "vkremote-iddcx.dll"
    if (Test-Path $dllPath) {
        & $signtool sign /v /s "My" /n "vkremote Virtual GPU Driver" /t "http://timestamp.digicert.com" $dllPath
    }
} else {
    Write-Warning "signtool.exe not found in Windows Kits. If pre-signed, skipping signing step."
}

# 5. Install INF using pnputil
$infPath = Join-Path $ScriptDir "vkremote-iddcx.inf"
Write-Host "Registering driver package via pnputil..." -ForegroundColor Green
& pnputil.exe /add-driver $infPath /install

# 6. Install root device node if not present
Write-Host "Ensuring Root\vkremote_IddCx device node is created..." -ForegroundColor Green
# Using devcon or PnP device creation
$devcon = (Get-ChildItem -Path "C:\Program Files (x86)\Windows Kits" -Filter "devcon.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
if ($devcon) {
    & $devcon install $infPath "Root\vkremote_IddCx"
} else {
    Write-Host "Device package installed. Windows PnP will instantiate Root\vkremote_IddCx."
}

Write-Host "\n=== SUCCESS ===" -ForegroundColor Green
Write-Host "vkremote Virtual GPU Adapter driver registered!"
Write-Host "Open Windows Device Manager -> Display Adapters to verify."
Write-Host "Open Windows Task Manager -> Performance tab to view the GPU graph."
