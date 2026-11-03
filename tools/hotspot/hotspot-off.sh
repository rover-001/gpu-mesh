#!/usr/bin/env bash
# Stop the offline rgpu hotspot and let NetworkManager reconnect to a normal Wi-Fi network.
set -euo pipefail
nmcli connection down omarchy-hotspot || true
nmcli device connect "${IFACE:-wlo1}" >/dev/null 2>&1 || true
echo "Hotspot off. $(nmcli -t -f DEVICE,STATE,CONNECTION dev | grep "^${IFACE:-wlo1}:")"
