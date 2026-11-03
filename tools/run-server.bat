@echo off
set RUST_LOG=info
C:\rgpu\target\release\vkremote-server.exe --bind 0.0.0.0 --port 50051 --token ***REMOVED*** > C:\rgpu\server_live.log 2>&1
