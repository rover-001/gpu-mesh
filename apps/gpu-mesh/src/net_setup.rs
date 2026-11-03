//! Per-device network setup run at app start: make sure this machine's firewall
//! lets peers reach the ports gpu-mesh listens on.
//!
//!   50051  vkremote Giver (GPU sharing)
//!   50052  llama.cpp RPC server (layer split across GPUs)
//!   50053  remote LLM worker control
//!   18080  remote LLM worker's llama-server (web UI / API proxied by the requester)
//!
//! Nothing is stored and no passwords are embedded: when a rule is missing the OS's
//! own elevation prompt is used (pkexec / a terminal running sudo on Linux, UAC on
//! Windows). If the ports are already open, this does nothing and asks nothing.

use std::process::Command;

pub const PEER_TCP_PORTS: [u16; 4] = [50051, 50052, 50053, 18080];

/// Check and, if needed, open the peer ports. Returns human-readable log lines.
pub fn ensure_peer_ports_open() -> Vec<String> {
    #[cfg(target_os = "windows")]
    {
        windows::ensure()
    }
    #[cfg(target_os = "linux")]
    {
        linux::ensure()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        Vec::new()
    }
}

fn run_ok(cmd: &mut Command) -> bool {
    cmd.output().map(|o| o.status.success()).unwrap_or(false)
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{run_ok, PEER_TCP_PORTS};
    use std::process::Command;

    /// Ports allowed for all sources by `ufw allow <port>[/tcp]`-style rules.
    /// /etc/ufw/user.rules is world-readable, so no root is needed to check.
    pub(super) fn ufw_open_ports(rules: &str) -> Vec<(u16, u16)> {
        let mut open = Vec::new();
        for line in rules.lines() {
            let line = line.trim();
            if !line.starts_with("-A ufw-user-input") || !line.ends_with("-j ACCEPT") {
                continue;
            }
            // Source-restricted rules (e.g. -s 10.42.0.0/24) only cover some peers.
            if line.contains(" -s ") || line.contains(" -i ") || line.contains("-p udp") {
                continue;
            }
            let mut toks = line.split_whitespace();
            while let Some(t) = toks.next() {
                if t == "--dport" || t == "--dports" {
                    if let Some(spec) = toks.next() {
                        for part in spec.split(',') {
                            let mut it = part.split(':');
                            let a = it.next().and_then(|x| x.parse::<u16>().ok());
                            let b = it.next().and_then(|x| x.parse::<u16>().ok());
                            if let Some(a) = a {
                                open.push((a, b.unwrap_or(a)));
                            }
                        }
                    }
                }
            }
        }
        open
    }

    fn ufw_active() -> bool {
        std::fs::read_to_string("/etc/ufw/ufw.conf")
            .map(|c| c.lines().any(|l| l.trim() == "ENABLED=yes"))
            .unwrap_or(false)
            && run_ok(Command::new("systemctl").args(["is-active", "--quiet", "ufw"]))
    }

    fn firewalld_active() -> bool {
        run_ok(Command::new("systemctl").args(["is-active", "--quiet", "firewalld"]))
    }

    fn polkit_agent_running() -> bool {
        Command::new("pgrep")
            .args(["-f", "polkit.*agent|polkitagent|polkit-gnome|polkit-kde|lxpolkit"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Run `script` as root: pkexec when a graphical polkit agent can prompt,
    /// otherwise open a terminal where the user types their sudo password once.
    fn run_elevated(script: &str) -> Result<String, String> {
        if polkit_agent_running() {
            let out = Command::new("pkexec")
                .args(["sh", "-c", script])
                .output()
                .map_err(|e| format!("pkexec failed to start: {e}"))?;
            return if out.status.success() {
                Ok("opened via pkexec".to_string())
            } else {
                Err(format!("pkexec exited with {}", out.status))
            };
        }
        let banner = "gpu-mesh: opening firewall ports so other devices can use this GPU";
        let term_script = format!(
            "echo '{banner}'; echo; sudo sh -c '{script}' && echo 'Done.' || echo 'Not changed.'; sleep 3"
        );
        for term in ["xdg-terminal-exec", "x-terminal-emulator"] {
            if let Ok(mut child) = Command::new(term).args(["sh", "-c", &term_script]).spawn() {
                let _ = child.wait();
                return Ok(format!("asked for sudo password in a terminal ({term})"));
            }
        }
        Err(format!("no elevation method available; run as root: {script}"))
    }

    pub(super) fn ensure() -> Vec<String> {
        let mut log = Vec::new();
        if ufw_active() {
            let rules = std::fs::read_to_string("/etc/ufw/user.rules").unwrap_or_default();
            let open = ufw_open_ports(&rules);
            let missing: Vec<u16> = PEER_TCP_PORTS
                .iter()
                .copied()
                .filter(|p| !open.iter().any(|(a, b)| a <= p && p <= b))
                .collect();
            if missing.is_empty() {
                log.push("Firewall (ufw): peer ports already open".to_string());
                return log;
            }
            let script = missing
                .iter()
                .map(|p| format!("ufw allow {p}/tcp"))
                .collect::<Vec<_>>()
                .join(" && ");
            log.push(format!("Firewall (ufw): opening TCP {:?} for peers", missing));
            match run_elevated(&script) {
                Ok(how) => {
                    let rules = std::fs::read_to_string("/etc/ufw/user.rules").unwrap_or_default();
                    let open = ufw_open_ports(&rules);
                    let still: Vec<u16> = missing
                        .into_iter()
                        .filter(|p| !open.iter().any(|(a, b)| a <= p && p <= b))
                        .collect();
                    if still.is_empty() {
                        log.push(format!("Firewall (ufw): ports opened ({how})"));
                    } else {
                        log.push(format!("Firewall (ufw): still closed {:?} ({how})", still));
                    }
                }
                Err(e) => log.push(format!("Firewall (ufw): {e}")),
            }
        } else if firewalld_active() {
            let missing: Vec<u16> = PEER_TCP_PORTS
                .iter()
                .copied()
                .filter(|p| {
                    !run_ok(Command::new("firewall-cmd").arg(format!("--query-port={p}/tcp")))
                })
                .collect();
            if missing.is_empty() {
                log.push("Firewall (firewalld): peer ports already open".to_string());
                return log;
            }
            let mut script = missing
                .iter()
                .map(|p| format!("firewall-cmd --permanent --add-port={p}/tcp"))
                .collect::<Vec<_>>()
                .join(" && ");
            script.push_str(" && firewall-cmd --reload");
            log.push(format!("Firewall (firewalld): opening TCP {:?} for peers", missing));
            match run_elevated(&script) {
                Ok(how) => log.push(format!("Firewall (firewalld): {how}")),
                Err(e) => log.push(format!("Firewall (firewalld): {e}")),
            }
        }
        log
    }
}

#[cfg(target_os = "windows")]
mod windows {
    use super::PEER_TCP_PORTS;
    use std::process::Command;

    const RULE: &str = "gpu-mesh peer ports";

    fn rule_exists() -> bool {
        Command::new("netsh")
            .args(["advfirewall", "firewall", "show", "rule", &format!("name={RULE}")])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    pub(super) fn ensure() -> Vec<String> {
        let mut log = Vec::new();
        if rule_exists() {
            log.push("Firewall (Windows): peer ports already open".to_string());
            return log;
        }
        let ports = PEER_TCP_PORTS
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",");
        // netsh needs admin: elevate just this one command via UAC.
        let args = format!(
            "advfirewall firewall add rule name=\"{RULE}\" dir=in action=allow protocol=TCP localport={ports} profile=any"
        );
        let ps = format!(
            "Start-Process netsh -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList '{}'",
            args.replace('\'', "''")
        );
        log.push(format!("Firewall (Windows): requesting UAC to open TCP {ports}"));
        let _ = Command::new("powershell")
            .args(["-NoProfile", "-Command", &ps])
            .output();
        if rule_exists() {
            log.push("Firewall (Windows): peer ports opened".to_string());
        } else {
            log.push("Firewall (Windows): not changed (UAC declined?)".to_string());
        }
        log
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::linux::ufw_open_ports;

    #[test]
    fn parses_single_range_and_ignores_restricted() {
        let rules = "\
-A ufw-user-input -p tcp --dport 50051 -j ACCEPT
-A ufw-user-input -p tcp --dport 50052:50053 -j ACCEPT
-A ufw-user-input -p tcp --dport 18080 -s 10.42.0.0/24 -j ACCEPT
-A ufw-user-input -p udp --dport 67 -j ACCEPT
-A ufw-user-input -p tcp -m multiport --dports 8000,9000:9001 -j ACCEPT";
        let open = ufw_open_ports(rules);
        assert!(open.contains(&(50051, 50051)));
        assert!(open.contains(&(50052, 50053)));
        assert!(!open.iter().any(|(a, _)| *a == 18080));
        assert!(!open.iter().any(|(a, _)| *a == 67));
        assert!(open.contains(&(9000, 9001)));
    }
}
