@echo off
rem Launch the vkremote replay server on this GPU host.
rem
rem The shared auth token is deliberately NOT stored in this file. Set it in
rem the environment first (see README) and export the same value on every
rem client:
rem
rem   setx VKREMOTE_TOKEN "<new-random-token>"
rem
rem To rotate: generate a new value, update the host and all clients, then
rem restart this script. A token that has ever been committed must be treated
rem as burned - rotating is what revokes it.

if "%VKREMOTE_TOKEN%"=="" goto :no_token

set RUST_LOG=info
C:\rgpu\target\release\vkremote-server.exe --bind 0.0.0.0 --port 50051 --token "%VKREMOTE_TOKEN%" > C:\rgpu\server_live.log 2>&1
exit /b %ERRORLEVEL%

:no_token
echo [run-server] VKREMOTE_TOKEN is not set. Refusing to start with a known default token.
echo [run-server] Generate one: powershell -NoProfile -Command "[guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')"
echo [run-server] Store it:     setx VKREMOTE_TOKEN ^<token^>
echo [run-server] Clients must export the same VKREMOTE_TOKEN value.
exit /b 1
