use std::process::{Command, Stdio};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

/// Minimal HTTP GET to a local port without spawning `curl` (the health poll runs
/// every 1.5 s on every device; a process per poll is wasteful and flashes console
/// windows on Windows). Returns the response body for any status code, because
/// llama-server reports "loading" with HTTP 503.
pub fn http_get_local(
    port: u16,
    path: &str,
    timeout: std::time::Duration,
) -> Result<String, String> {
    use std::io::{Read, Write};
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = std::net::TcpStream::connect_timeout(&addr, timeout)
        .map_err(|e| format!("connect 127.0.0.1:{port}: {e}"))?;
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));
    let req = format!("GET {path} HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("write 127.0.0.1:{port}: {e}"))?;
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|e| format!("read 127.0.0.1:{port}: {e}"))?;
    let text = String::from_utf8_lossy(&raw);
    Ok(match text.find("\r\n\r\n") {
        Some(i) => text[i + 4..].to_string(),
        None => String::new(),
    })
}

/// This machine's rgpu `tools` directory (Linux/macOS). Checked in order:
/// `$RGPU_ROOT/tools`, the checkout this binary was built from (dev builds run
/// from each machine's own repo), `~/Projects/rgpu/tools`, then the original
/// /home/rover path. Lets the same code find llama.cpp on every peer.
pub fn rgpu_tools_dir() -> std::path::PathBuf {
    use std::path::PathBuf;
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(root) = std::env::var_os("RGPU_ROOT") {
        candidates.push(PathBuf::from(root).join("tools"));
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools"));
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join("Projects/rgpu/tools"));
    }
    candidates.push(PathBuf::from("/home/rover/Projects/rgpu/tools"));
    candidates
        .iter()
        .find(|p| p.join("llama-bin").is_dir())
        .or_else(|| candidates.iter().find(|p| p.is_dir()))
        .cloned()
        .unwrap_or_else(|| candidates[candidates.len() - 1].clone())
}

/// Directory holding the bundled llama.cpp build (llama-server, ggml-rpc-server, .so files).
pub fn llama_bin_dir() -> std::path::PathBuf {
    rgpu_tools_dir().join("llama-bin").join("llama-b12020")
}

/// Bundled llama.cpp web UI directory.
pub fn llama_webui_dir() -> std::path::PathBuf {
    rgpu_tools_dir().join("llama-webui")
}

fn engine_log_path() -> std::path::PathBuf {
    std::env::temp_dir().join("llama-server.log")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineStatus {
    Stopped,
    Starting(String),
    Running,
    Failed(String),
}

pub struct ClusterEngineManager;

impl ClusterEngineManager {
    pub fn get_default_bin_path() -> String {
        if cfg!(target_os = "windows") {
            let candidates = [
                "C:\\rgpu\\rpc\\llama-server.exe",
                "C:\\rgpu\\tools\\llama-server.exe",
            ];
            for p in candidates {
                if std::path::Path::new(p).exists() {
                    return p.to_string();
                }
            }
            if let Ok(appdata) = std::env::var("LOCALAPPDATA") {
                let p = format!(
                    "{}\\Programs\\Ollama\\lib\\ollama\\llama-server.exe",
                    appdata
                );
                if std::path::Path::new(&p).exists() {
                    return p;
                }
            }
            "C:\\rgpu\\rpc\\llama-server.exe".to_string()
        } else {
            // Priority order for Linux:
            // 1. Project-local llama-bin directory (most reliable for development)
            let home_ollama = format!(
                "{}/.ollama/bin/llama-server",
                std::env::var("HOME").unwrap_or_default()
            );
            let project_bin = llama_bin_dir().join("llama-server");
            let project_bin = project_bin.to_string_lossy();
            let candidates = vec![
                project_bin.as_ref(),
                // 2. User's home Ollama installation
                home_ollama.as_str(),
                // 3. System-wide Ollama installation
                "/usr/local/bin/llama-server",
                "/usr/bin/llama-server",
            ];

            for path in candidates {
                if std::path::Path::new(path).exists() {
                    return path.to_string();
                }
            }

            // Fallback: check PATH
            "llama-server".to_string()
        }
    }

    pub fn get_model_path_for_manifest(manifest: &crate::remote_llm::ModelManifest) -> String {
        let candidate_dirs = if cfg!(target_os = "windows") {
            vec![
                std::path::PathBuf::from("C:\\rgpu\\models"),
                crate::remote_llm::get_cache_dir(),
            ]
        } else {
            // Current user's ~/models and ~ first so every peer (e.g. /home/rahul)
            // resolves its own copy; then the original /home/rover locations.
            let mut dirs = Vec::new();
            if let Some(home) = std::env::var_os("HOME") {
                let home = std::path::PathBuf::from(home);
                dirs.push(home.join("models"));
                dirs.push(home);
            }
            dirs.push(std::path::PathBuf::from("/home/rover/models"));
            dirs.push(std::path::PathBuf::from("/home/rover"));
            dirs.push(crate::remote_llm::get_cache_dir());
            dirs
        };

        // Skip truncated/stub files (e.g. a 49-byte partial download in ~/models).
        for dir in candidate_dirs {
            let p = dir.join(&manifest.filename);
            if crate::remote_llm::is_usable_gguf(&p, manifest.size) {
                return p.to_string_lossy().to_string();
            }
        }

        if cfg!(target_os = "windows") {
            format!("C:\\rgpu\\models\\{}", manifest.filename)
        } else {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/home/rover".to_string());
            format!("{home}/models/{}", manifest.filename)
        }
    }

    pub fn get_default_model_path() -> String {
        if cfg!(target_os = "windows") {
            let userprofile = std::env::var("USERPROFILE").unwrap_or_default();
            let search_dirs = vec!["C:\\rgpu\\models", userprofile.as_str()];

            for dir_path in search_dirs {
                let dir = std::path::Path::new(dir_path);
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        if p.is_file() && p.extension().map_or(false, |ext| ext == "gguf") {
                            return p.to_string_lossy().to_string();
                        }
                    }
                }
            }
            "C:\\rgpu\\models\\model.gguf".to_string()
        } else {
            // Priority order for Linux: specific known models first, then search
            // Current user's home first (works on every peer), then the original
            // /home/rover locations.
            let home = std::env::var("HOME").unwrap_or_else(|_| "/home/rover".to_string());
            let mut known_models = vec![
                format!("{home}/qwen2.5-coder-7b-instruct-q4_k_m.gguf"),
                format!("{home}/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"),
                format!("{home}/.ollama/models/blobs/model.gguf"),
            ];
            known_models.push("/home/rover/qwen2.5-coder-7b-instruct-q4_k_m.gguf".to_string());
            known_models.push("/home/rover/.ollama/models/blobs/model.gguf".to_string());

            for model_path in &known_models {
                if std::path::Path::new(model_path).exists() {
                    return model_path.to_string();
                }
            }

            // Search home directory for any .gguf file
            let mut search_dirs = vec![
                home.clone(),
                format!("{home}/models"),
                format!("{home}/Downloads"),
            ];
            for d in ["/home/rover", "/home/rover/models", "/home/rover/Downloads"] {
                if !search_dirs.iter().any(|x| x == d) {
                    search_dirs.push(d.to_string());
                }
            }

            for dir_path in &search_dirs {
                let dir = std::path::Path::new(dir_path);
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        if p.is_file() && p.extension().map_or(false, |ext| ext == "gguf") {
                            return p.to_string_lossy().to_string();
                        }
                    }
                }
            }

            // Fallback
            format!("{home}/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
        }
    }

    /// Spawn llama-server with VRAM-proportional layer splitting across all devices.
    ///
    /// `local_vram_mb`       – local GPU VRAM in MiB (0 = unknown, defaults to 6144 MB)
    /// `remote_vram_mb_list` – VRAM in MiB for each remote peer in the same order as the
    ///                         comma-separated `rpc_endpoint` string (empty = unknown peer VRAM,
    ///                         defaults to 4096 MB per peer so the split is still proportional)
    ///
    /// Layer split formula: layers_i = round(total_layers * vram_i / total_vram)
    /// Adding more peers in the future is fully automatic — no code changes needed.
    pub fn spawn_server(
        rpc_endpoint: &str,
        model_path: &str,
        local_vram_mb: u64,
        remote_vram_mb_list: &[u64],
    ) -> Result<u32, String> {
        let bin_path = Self::get_default_bin_path();

        // Kill any existing llama-server processes to avoid port conflicts
        eprintln!("[cluster_engine] Checking for existing llama-server processes...");
        let _ = Command::new("pkill").arg("-9").arg("llama-server").status();

        // Wait for kill to complete and reap any zombies
        std::thread::sleep(std::time::Duration::from_millis(500));
        let _ = Command::new("sh")
            .arg("-c")
            .arg("wait 2>/dev/null || true")
            .status();

        // Check if RPC endpoint is reachable before trying to use it
        let rpc_endpoints: Vec<&str> = if !rpc_endpoint.is_empty() {
            rpc_endpoint.split(',').collect()
        } else {
            vec![]
        };

        let mut reachable_endpoints = Vec::new();
        // Track which remote_vram_mb_list entries correspond to reachable endpoints
        let mut reachable_remote_vram: Vec<u64> = Vec::new();
        for (i, endpoint) in rpc_endpoints.iter().enumerate() {
            if endpoint.is_empty() {
                continue;
            }
            eprintln!(
                "[cluster_engine] Checking RPC endpoint connectivity: {}",
                endpoint
            );
            let reachable = std::net::TcpStream::connect_timeout(
                &endpoint
                    .parse()
                    .unwrap_or_else(|_| "127.0.0.1:1".parse().unwrap()),
                std::time::Duration::from_secs(2),
            )
            .is_ok();

            if reachable {
                eprintln!("[cluster_engine] ✓ RPC endpoint {} is reachable", endpoint);
                reachable_endpoints.push(*endpoint);
                // Use provided VRAM or fall back to 4096 MB if unknown
                let vram = remote_vram_mb_list.get(i).copied().unwrap_or(4096);
                reachable_remote_vram.push(if vram == 0 { 4096 } else { vram });
            } else {
                eprintln!(
                    "[cluster_engine] ✗ RPC endpoint {} is unreachable - skipping",
                    endpoint
                );
            }
        }

        let final_rpc_endpoint = reachable_endpoints.join(",");
        // Always use all reachable remote GPUs — more devices = more VRAM = faster inference.
        // The layer split is VRAM-proportional so every peer contributes proportionally.
        let effective_local_vram_mb = if local_vram_mb == 0 {
            6144
        } else {
            local_vram_mb
        };
        let use_rpc = !final_rpc_endpoint.is_empty();
        // Verify binary exists before attempting spawn
        if !std::path::Path::new(&bin_path).exists() {
            return Err(format!(
                "llama-server binary not found at: {}\n\
                 Searched locations:\n\
                 - {}\n\
                 - ~/.ollama/bin/llama-server\n\
                 - /usr/local/bin/llama-server\n\
                 - /usr/bin/llama-server\n\
                 Please install llama.cpp or place llama-server in tools/llama-bin/",
                bin_path,
                llama_bin_dir().join("llama-server").display()
            ));
        }

        // Verify model exists before attempting spawn
        if !std::path::Path::new(model_path).exists() {
            return Err(format!(
                "Model file not found at: {}\n\
                 Please download a GGUF model file and place it in:\n\
                 - ~/ (home directory)\n\
                 - ~/models/\n\
                 Example: wget https://huggingface.co/.../model.gguf",
                model_path
            ));
        }

        let current_bytes = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
        if model_path.contains("14b") && current_bytes < 8_000_000_000 {
            return Err(format!(
                "Qwen 14B is downloading in background ({:.2} GB / 8.37 GB). Select Qwen 7B to run immediately, or wait for 14B to finish.",
                current_bytes as f64 / 1_073_741_824.0
            ));
        }

        let lib_path = format!(
            "/usr/local/lib/ollama/cuda_v12:{}",
            llama_bin_dir().display()
        );

        let mut cmd = Command::new(&bin_path);
        if !cfg!(target_os = "windows") {
            cmd.env("LD_LIBRARY_PATH", &lib_path);
        }

        let mut args = vec![
            "-m".to_string(),
            model_path.to_string(),
            "-sm".to_string(),
            "layer".to_string(),
            "--port".to_string(),
            "18081".to_string(),
            "--host".to_string(),
            "0.0.0.0".to_string(),
            "-fa".to_string(),
            "on".to_string(),
            "-c".to_string(),
            "4096".to_string(),
        ];

        if let Some(adapter) = crate::remote_llm::trained_adapter_for_model(std::path::Path::new(model_path))
            .map_err(|error| format!("Trained adapter validation failed: {error:#}"))?
        {
            args.extend(["--lora".to_string(), adapter.to_string_lossy().to_string()]);
        }

        let webui_path = if cfg!(target_os = "windows") {
            "C:\\rgpu\\tools\\llama-webui".to_string()
        } else {
            llama_webui_dir().to_string_lossy().to_string()
        };
        if std::path::Path::new(&webui_path).exists() {
            args.push("--path".to_string());
            args.push(webui_path);
        }

        // Local-only: let llama.cpp fit layers to free VRAM (a 7B Q4 model needs
        // ~4.2 GB; forcing -ngl 999 on a 4 GB card is an instant CUDA OOM, and an
        // explicit -ngl disables llama.cpp's own fitting). The multi-GPU RPC split
        // keeps all layers on GPUs, sized by --tensor-split.
        if use_rpc {
            args.push("-ngl".to_string());
            args.push("999".to_string());
        }

        if use_rpc {
            args.push("--rpc".to_string());
            args.push(final_rpc_endpoint.to_string());

            // Devices sorted by VRAM descending (largest GPU gets first layers)
            let mut device_specs: Vec<(String, u64)> =
                vec![("CUDA0".to_string(), effective_local_vram_mb)];
            for (i, &vram) in reachable_remote_vram.iter().enumerate() {
                device_specs.push((format!("RPC{}", i), vram));
            }
            device_specs.sort_by_key(|(_, vram)| std::cmp::Reverse(*vram));

            let devices = device_specs
                .iter()
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>();
            let device_str = devices.join(",");
            let split_str = device_specs
                .iter()
                .map(|(_, vram)| vram.to_string())
                .collect::<Vec<_>>()
                .join(",");

            args.push("-dev".to_string());
            args.push(device_str.clone());

            args.push("-ts".to_string());
            args.push(split_str.clone());

            eprintln!(
                "[cluster_engine] VRAM proportional split: local={}MB, remote={:?}MB",
                effective_local_vram_mb, reachable_remote_vram
            );
            eprintln!(
                "[cluster_engine] Using {} GPU(s): {} with split: {}",
                devices.len(),
                device_str,
                split_str
            );
        }

        eprintln!("[cluster_engine] Spawning llama-server:");
        eprintln!("  Binary: {}", bin_path);
        eprintln!("  Model: {}", model_path);
        eprintln!(
            "  RPC Endpoint: {}",
            if use_rpc {
                &final_rpc_endpoint
            } else {
                "None (local GPU only)"
            }
        );
        eprintln!("  Args: {:?}", args);
        eprintln!("  LD_LIBRARY_PATH: {}", lib_path);

        // Create log file for debugging (truncate old content)
        let log_path = engine_log_path();
        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true) // Clear old logs on each start
            .open(&log_path)
            .map_err(|e| format!("Failed to create log file {}: {}", log_path.display(), e))?;

        cmd.args(&args)
            .stdout(Stdio::from(log_file.try_clone().map_err(|e| {
                format!("Failed to clone log file handle: {}", e)
            })?))
            .stderr(Stdio::from(log_file));
        #[cfg(unix)]
        cmd.process_group(0);

        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "Failed to spawn llama-server process: {}\n\
                 Binary: {}\n\
                 Model: {}\n\
                 Check logs at: {}\n\
                 Error: {}",
                e, bin_path, model_path, log_path.display(), e
            )
        })?;

        let pid = child.id();

        // Spawn a background thread to wait for the child process
        // This prevents zombie processes while still allowing proper cleanup
        std::thread::spawn(move || {
            let _ = child.wait();
        });

        eprintln!(
            "[cluster_engine] llama-server spawned successfully with PID: {}",
            pid
        );
        eprintln!("[cluster_engine] Logs available at: {}", log_path.display());
        eprintln!("[cluster_engine] Waiting for server to become ready (health check on :8080)...");

        Ok(pid)
    }

    /// Stop a llama-server left running by an earlier gpu-mesh instance (the app
    /// was killed, so its child kept the model loaded and port 8080 busy, making
    /// the new instance look like it auto-started a server). Only processes with
    /// gpu-mesh's own signature (`--port 8080` and the bundled `llama-webui` path)
    /// are touched. Returns how many were stopped.
    pub fn kill_orphaned_engines() -> usize {
        if cfg!(target_os = "windows") {
            let ps = "$p = Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | \
                Where-Object { $_.CommandLine -match '--port (8080|18081)' -and $_.CommandLine -match 'llama-webui' }; \
                $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; ($p | Measure-Object).Count";
            return Command::new("powershell")
                .args(["-NoProfile", "-Command", ps])
                .output()
                .ok()
                .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
                .unwrap_or(0);
        }
        let mut killed = 0;
        if let Ok(entries) = std::fs::read_dir("/proc") {
            for e in entries.flatten() {
                let Ok(pid) = e.file_name().to_string_lossy().parse::<u32>() else { continue };
                let Ok(raw) = std::fs::read(format!("/proc/{pid}/cmdline")) else { continue };
                let args: Vec<String> = raw
                    .split(|b| *b == 0)
                    .map(|a| String::from_utf8_lossy(a).to_string())
                    .collect();
                let is_llama = args.first().map_or(false, |a| a.ends_with("/llama-server"));
                let port_8080 = args.windows(2).any(|w| w[0] == "--port" && (w[1] == "8080" || w[1] == "18081"));
                let ours = args.iter().any(|a| a.ends_with("llama-webui"));
                if is_llama && port_8080 && ours {
                    let _ = Command::new("kill").arg("-9").arg(pid.to_string()).status();
                    killed += 1;
                }
            }
        }
        killed
    }

    /// `Some(reason)` if the llama-server process has exited (e.g. CUDA OOM while
    /// loading), with the last error lines from its log; `None` while it runs.
    pub fn engine_exit_reason(pid: u32) -> Option<String> {
        let alive = if cfg!(target_os = "windows") {
            Command::new("tasklist")
                .args(["/FI", &format!("PID eq {pid}"), "/NH"])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
                .unwrap_or(true)
        } else {
            // An exited-but-unreaped child is a zombie ('Z'): treat it as exited.
            match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
                Ok(stat) => stat
                    .rsplit(')')
                    .next()
                    .and_then(|rest| rest.split_whitespace().next())
                    .map_or(true, |state| state != "Z"),
                Err(_) => false,
            }
        };
        if alive {
            return None;
        }
        let log_path = engine_log_path();
        let log = std::fs::read_to_string(&log_path).unwrap_or_default();
        let errors: Vec<&str> = log
            .lines()
            .filter(|l| l.contains(" E ") || l.contains("error") || l.contains("failed"))
            .collect();
        let tail = errors[errors.len().saturating_sub(3)..].join("\n");
        Some(if tail.is_empty() {
            format!("llama-server exited during startup. Check {}", log_path.display())
        } else {
            format!("llama-server exited during startup:\n{tail}")
        })
    }

    pub fn stop_server(pid: u32) {
        eprintln!("[cluster_engine] Stopping llama-server PID: {}", pid);

        if cfg!(target_os = "windows") {
            let _ = Command::new("taskkill")
                .args(&["/F", "/PID", &pid.to_string()])
                .status();
        } else {
            // First try graceful SIGTERM
            let _ = Command::new("kill")
                .arg("-15")
                .arg(pid.to_string())
                .status();

            // Wait a bit
            std::thread::sleep(std::time::Duration::from_millis(500));

            // Then force kill if still running
            let _ = Command::new("kill").arg("-9").arg(pid.to_string()).status();
        }

        // Clean up any zombie processes
        std::thread::sleep(std::time::Duration::from_millis(200));
        let _ = Command::new("sh")
            .arg("-c")
            .arg("wait 2>/dev/null || true")
            .status();
    }

    pub fn check_health() -> Result<String, String> {
        let body = http_get_local(8080, "/health", std::time::Duration::from_secs(3))
            .map_err(|e| {
                format!("Server not responding on :8080 ({e})\nCheck {}", engine_log_path().display())
            })?;

        // Check for various "ok" response formats
        if body.contains("\"status\":\"ok\"") || body.contains("\"ok\"") {
            return Ok("ok".to_string());
        }

        // Check for loading state (might be in error response)
        if body.contains("Loading model")
            || body.contains("loading")
            || body.contains("\"type\":\"unavailable_error\"")
        {
            return Ok("loading".to_string());
        }

        // Empty response means server not started yet
        if body.trim().is_empty() {
            return Err("Empty response (server starting)".to_string());
        }

        // Return the response body as-is for debugging
        Ok(body.trim().to_string())
    }

    pub fn parse_completion_response(resp_str: &str) -> (String, String) {
        let content = if let Some(idx) = resp_str.find("\"content\":") {
            let start = idx + 10;
            let slice = &resp_str[start..];
            let trimmed = slice.trim_start();
            if trimmed.starts_with('"') {
                let inner = &trimmed[1..];
                if let Some(end) = inner.find('"') {
                    inner[..end].replace("\\n", "\n").replace("\\\"", "\"")
                } else {
                    trimmed.to_string()
                }
            } else {
                trimmed.to_string()
            }
        } else {
            resp_str.to_string()
        };

        let speed = if let Some(idx) = resp_str.find("\"predicted_per_second\":") {
            let start = idx + 23;
            let slice = &resp_str[start..];
            let end = slice
                .find(',')
                .or_else(|| slice.find('}'))
                .unwrap_or(slice.len());
            let num_str = slice[..end].trim();
            format!("{} tokens/sec", num_str)
        } else {
            "Cluster GPU Active".to_string()
        };

        (content, speed)
    }

    pub fn execute_prompt(prompt: &str) -> Result<(String, String), String> {
        let json_body = format!(
            r#"{{"prompt": "{}", "n_predict": 48, "temperature": 0.7, "stream": false}}"#,
            prompt.replace('"', "\\\"").replace('\n', " ")
        );

        let output = Command::new("curl")
            .args(&[
                "-s",
                "-m",
                "45",
                "-X",
                "POST",
                "http://127.0.0.1:8080/completion",
                "-H",
                "Content-Type: application/json",
                "-d",
                &json_body,
            ])
            .output()
            .map_err(|e| format!("Failed to execute test request: {}", e))?;

        if !output.status.success() {
            return Err("HTTP request failed".to_string());
        }

        let resp_str = String::from_utf8_lossy(&output.stdout);
        Ok(Self::parse_completion_response(&resp_str))
    }

    pub fn open_browser() {
        if cfg!(target_os = "windows") {
            let _ = Command::new("cmd")
                .args(&["/c", "start", "http://127.0.0.1:8080"])
                .spawn();
        } else {
            let _ = Command::new("xdg-open")
                .arg("http://127.0.0.1:8080")
                .spawn();
        }
    }
}

#[cfg(test)]
mod http_get_local_tests {
    use super::http_get_local;
    use std::io::{Read, Write};

    fn serve_once(resp: &'static str) -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 512];
            let _ = s.read(&mut buf);
            s.write_all(resp.as_bytes()).unwrap();
        });
        port
    }

    #[test]
    fn returns_body_for_ok_and_loading() {
        let ok = serve_once("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"ok\"}");
        let body = http_get_local(ok, "/health", std::time::Duration::from_secs(2)).unwrap();
        assert_eq!(body, "{\"status\":\"ok\"}");

        let loading = serve_once("HTTP/1.1 503 Service Unavailable\r\n\r\n{\"error\":{\"message\":\"Loading model\",\"type\":\"unavailable_error\"}}");
        let body = http_get_local(loading, "/health", std::time::Duration::from_secs(2)).unwrap();
        assert!(body.contains("Loading model"));
    }

    #[test]
    fn closed_port_is_an_error_not_a_hang() {
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        assert!(http_get_local(port, "/health", std::time::Duration::from_millis(500)).is_err());
    }
}
