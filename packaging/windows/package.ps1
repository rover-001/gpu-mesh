param(
    [string]$ProjectRoot = "C:\rgpu",
    [string]$OutputRoot = "$PSScriptRoot\..\..\dist\gpu-mesh-windows"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$Binary = Join-Path $ProjectRoot "target\release\gpu-mesh.exe"
$RpcRuntime = Join-Path $ProjectRoot "rpc"

Write-Host "Building gpu-mesh release binary..."
Push-Location $ProjectRoot
try {
    cargo build --release -p gpu-mesh
}
finally {
    Pop-Location
}

if (!(Test-Path $Binary)) {
    throw "Release binary was not produced: $Binary"
}
if (!(Test-Path (Join-Path $RpcRuntime "ggml-rpc-server.exe"))) {
    throw "RPC runtime is missing: $RpcRuntime\ggml-rpc-server.exe"
}

if (Test-Path $OutputRoot) {
    Remove-Item $OutputRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $OutputRoot | Out-Null

Copy-Item $Binary (Join-Path $OutputRoot "gpu-mesh.exe")
Copy-Item $RpcRuntime (Join-Path $OutputRoot "rpc") -Recurse

@"
GPU Mesh portable bundle

Run gpu-mesh.exe. Rust and Cargo are not required on the target machine.
The RPC runtime is included under .\rpc.
Place GGUF models under .\models or select a model path in the app.
"@ | Set-Content (Join-Path $OutputRoot "README.txt") -Encoding UTF8

$Archive = "$OutputRoot.zip"
if (Test-Path $Archive) {
    Remove-Item $Archive -Force
}
Compress-Archive -Path "$OutputRoot\*" -DestinationPath $Archive
Write-Host "Created portable bundle: $Archive"
