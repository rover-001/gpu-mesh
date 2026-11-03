#!/usr/bin/env bash
# Turn this laptop's Wi-Fi into the offline rgpu hotspot.
#   SSID: omarchy   password: 12345678   this laptop: 10.42.0.1
# Other devices join "omarchy"; vkremote / gpu-mesh then use 10.42.0.1:50051.
# Note: wlo1 leaves any current Wi-Fi network (no internet while the hotspot is on).
set -euo pipefail
IFACE="${IFACE:-wlo1}"
CON="omarchy-hotspot"

if ! nmcli -t -f NAME connection show | grep -qx "$CON"; then
  nmcli connection add type wifi ifname "$IFACE" con-name "$CON" autoconnect no \
    ssid omarchy 802-11-wireless.mode ap 802-11-wireless.band a 802-11-wireless.channel 149 \
    ipv4.method shared wifi-sec.key-mgmt wpa-psk wifi-sec.psk 12345678 >/dev/null
fi

# Let hotspot clients get an address and reach the internet through this laptop's
# uplink (USB tether / Ethernet). ufw's default policy drops DHCP/DNS on wlo1 and all
# forwarded traffic, which shows up on clients as "connected, no internet".
if command -v ufw >/dev/null && sudo ufw status | grep -q "Status: active"; then
  sudo ufw allow in on "$IFACE" to any port 67 proto udp >/dev/null
  sudo ufw allow in on "$IFACE" to any port 53 >/dev/null
  sudo ufw route allow in on "$IFACE" >/dev/null
fi
echo "net.ipv4.ip_forward=1" | sudo tee /etc/sysctl.d/90-omarchy-hotspot.conf >/dev/null
sudo sysctl -q -w net.ipv4.ip_forward=1

nmcli connection up "$CON"
echo "Hotspot 'omarchy' is up. This laptop: $(ip -4 -br addr show "$IFACE" | awk '{print $3}')"
echo "GPU server endpoint for other devices: 10.42.0.1:50051"
