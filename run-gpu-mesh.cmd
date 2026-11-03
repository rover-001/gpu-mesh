@echo off
rem rgpu: start GPU Mesh in the desktop session
cd /d C:\rgpu
if exist C:\rgpu\target\release\gpu-mesh.exe (
    C:\rgpu\target\release\gpu-mesh.exe > C:\rgpu\gpu-mesh.log 2>&1
) else (
    C:\rgpu\target\debug\gpu-mesh.exe > C:\rgpu\gpu-mesh.log 2>&1
)
