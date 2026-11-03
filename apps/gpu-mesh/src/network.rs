use crate::mesh::{ConnectionStatus, GpuDevice, RemoteGiver};
use anyhow::{Context, Result};
use std::net::UdpSocket;
use std::time::Instant;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tracing::{error, info};
use vkremote_server::{run_server_with_config, ServerConfig};
use vkremote_wire::framing::{client_handshake, read_msg, write_msg, DEFAULT_MAX_FRAME_SIZE};
use vkremote_wire::types::{Request, Response};

pub fn get_local_ip() -> String {
    let socket = UdpSocket::bind("0.0.0.0:0").ok();
    if let Some(s) = socket {
        if s.connect("8.8.8.8:80").is_ok() {
            if let Ok(addr) = s.local_addr() {
                return addr.ip().to_string();
            }
        }
    }
    "127.0.0.1".to_string()
}

pub fn get_host_name() -> String {
    if let Ok(name) = std::env::var("COMPUTERNAME") {
        let trimmed = name.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Ok(name) = std::env::var("HOSTNAME") {
        let trimmed = name.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Ok(name) = std::fs::read_to_string("/etc/hostname") {
        let trimmed = name.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if cfg!(target_os = "windows") {
        "Windows PC".to_string()
    } else {
        "Linux Host".to_string()
    }
}

pub fn start_giver_server(
    port: u16,
    token: String,
    gpu_index: Option<usize>,
) -> Result<oneshot::Sender<()>> {
    let addr = format!("0.0.0.0:{}", port);
    info!("Binding vkremote Giver server on {}", addr);

    let std_listener = std::net::TcpListener::bind(&addr)
        .with_context(|| format!("Failed to bind to {}", addr))?;
    std_listener.set_nonblocking(true)?;

    let (stop_tx, mut stop_rx) = oneshot::channel();

    crate::tokio_rt().spawn(async move {
        let listener = match TcpListener::from_std(std_listener) {
            Ok(l) => l,
            Err(e) => {
                error!("Failed to wrap listener into Tokio: {}", e);
                return;
            }
        };

        let config = ServerConfig {
            token,
            gpu_index,
            force_mock: false,
        };

        tokio::select! {
            res = run_server_with_config(listener, config) => {
                if let Err(e) = res {
                    error!("Giver server stopped with error: {}", e);
                }
            }
            _ = &mut stop_rx => {
                info!("Giver server received shutdown signal");
            }
        }
    });

    Ok(stop_tx)
}

pub async fn probe_remote_giver(endpoint: &str, token: &str) -> Result<RemoteGiver> {
    let start = Instant::now();
    let mut stream = TcpStream::connect(endpoint)
        .await
        .with_context(|| format!("Could not connect to {}", endpoint))?;

    client_handshake(&mut stream, token.to_string())
        .await
        .context("Handshake with remote host failed")?;

    write_msg(&mut stream, &Request::GetDeviceSnapshot)
        .await
        .context("Failed to send GetDeviceSnapshot request")?;

    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE)
        .await
        .context("Failed to receive response from host")?;

    let latency = start.elapsed().as_secs_f32() * 1000.0;

    let snapshot = match resp {
        Response::DeviceSnapshot(s) => s,
        _ => return Err(anyhow::anyhow!("Unexpected response from host")),
    };

    let mut vram_bytes: u64 = 0;
    for heap in snapshot.memory_properties.memory_heaps.iter() {
        if heap.flags & 1 != 0 {
            // DEVICE_LOCAL
            if heap.size > vram_bytes {
                vram_bytes = heap.size;
            }
        }
    }
    if vram_bytes == 0 && !snapshot.memory_properties.memory_heaps.is_empty() {
        vram_bytes = snapshot.memory_properties.memory_heaps[0].size;
    }

    let vram_mb = vram_bytes / (1024 * 1024);
    let gpu_name = snapshot.properties.device_name.clone();

    let api_version = format!(
        "{}.{}.{}",
        ash::vk::api_version_major(snapshot.properties.api_version),
        ash::vk::api_version_minor(snapshot.properties.api_version),
        ash::vk::api_version_patch(snapshot.properties.api_version)
    );

    let driver_version = format!(
        "{}.{}.{}",
        (snapshot.properties.driver_version >> 22) & 0x3ff,
        (snapshot.properties.driver_version >> 12) & 0x3ff,
        snapshot.properties.driver_version & 0xfff
    );

    let gpu = GpuDevice {
        name: gpu_name,
        vram_mb,
        is_discrete: true,
        is_shared: true,
        driver_version,
        api_version,
    };

    Ok(RemoteGiver {
        id: endpoint.to_string(),
        name: format!("Host @ {}", endpoint),
        endpoint: endpoint.to_string(),
        status: ConnectionStatus::Connected,
        latency_ms: latency,
        gpus: vec![gpu],
        last_ping: Some(Instant::now()),
        total_vram_mb: vram_mb,
        allocated_vram_mb: None,
    })
}

pub async fn ping_remote_giver(endpoint: &str, token: &str) -> Result<f32> {
    let start = Instant::now();
    let mut stream = TcpStream::connect(endpoint)
        .await
        .with_context(|| format!("Could not connect to {}", endpoint))?;

    client_handshake(&mut stream, token.to_string()).await?;
    write_msg(&mut stream, &Request::GetDeviceSnapshot).await?;
    let _resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await?;

    let latency = start.elapsed().as_secs_f32() * 1000.0;
    Ok(latency)
}

pub async fn scan_lan_for_givers(local_ip: &str, port: u16) -> Vec<String> {
    let parts: Vec<&str> = local_ip.split('.').collect();
    if parts.len() != 4 {
        return Vec::new();
    }
    let prefix = format!("{}.{}.{}.", parts[0], parts[1], parts[2]);
    let local_octet: u8 = parts[3].parse().unwrap_or(0);

    let mut tasks = Vec::new();
    for i in 1..=254 {
        // Skip the local machine's IP - don't connect to ourselves
        if i == local_octet {
            continue;
        }
        let target = format!("{}{}:{}", prefix, i, port);
        tasks.push(tokio::spawn(async move {
            let timeout = std::time::Duration::from_millis(600);
            match tokio::time::timeout(timeout, TcpStream::connect(&target)).await {
                Ok(Ok(_)) => Some(target),
                _ => None,
            }
        }));
    }

    let mut found = Vec::new();
    let local_endpoint = format!("{}", local_ip);
    for task in tasks {
        if let Ok(Some(addr)) = task.await {
            // Double-check: exclude any address that matches our local IP
            // (in case local_octet parsing failed earlier)
            if !addr.starts_with(&local_endpoint) {
                found.push(addr);
            }
        }
    }

    // An open port alone is not a giver: 50051 is also the stock gRPC port, so on
    // shared Wi-Fi unrelated devices answered and flooded the candidate list.
    // Keep only hosts that complete the vkremote handshake (checked in parallel).
    let checks: Vec<_> = found
        .into_iter()
        .map(|addr| {
            tokio::spawn(async move {
                let ok = matches!(
                    tokio::time::timeout(
                        std::time::Duration::from_secs(2),
                        probe_remote_giver(&addr, "vkremote-secret"),
                    )
                    .await,
                    Ok(Ok(_))
                );
                ok.then_some(addr)
            })
        })
        .collect();
    let mut givers = Vec::new();
    for c in checks {
        if let Ok(Some(addr)) = c.await {
            givers.push(addr);
        }
    }
    givers
}

pub fn start_rpc_server(port: u16) -> Result<std::process::Child> {
    use std::process::{Command, Stdio};

    // Prefer the standalone ggml RPC worker. It keeps remote model tensors
    // cached on the giver instead of restarting a full llama server.
    let (bin_path, rpc_args) = if cfg!(target_os = "windows") {
        let candidates = [
            "C:\\rgpu\\rpc\\ggml-rpc-server.exe",
            "C:\\rgpu\\tools\\ggml-rpc-server.exe",
            "C:\\rgpu\\tools\\llama-server.exe",
        ];
        let path = candidates
            .iter()
            .find(|p| std::path::Path::new(p).exists())
            .copied()
            .unwrap_or(candidates[0]);
        if path.ends_with("ggml-rpc-server.exe") {
            (
                path.to_string(),
                vec![
                    "-H".to_string(),
                    "0.0.0.0".to_string(),
                    "-p".to_string(),
                    port.to_string(),
                    "-d".to_string(),
                    "CUDA0".to_string(),
                ],
            )
        } else {
            (
                path.to_string(),
                vec![
                    "--rpc-server-host".to_string(),
                    "0.0.0.0".to_string(),
                    "--rpc-server-port".to_string(),
                    port.to_string(),
                ],
            )
        }
    } else {
        let bin_dir = crate::cluster_engine::llama_bin_dir();
        let candidates: Vec<String> = vec![
            bin_dir.join("ggml-rpc-server").to_string_lossy().to_string(),
            bin_dir.join("llama-server").to_string_lossy().to_string(),
            "/usr/local/bin/llama-server".to_string(),
            "/usr/bin/llama-server".to_string(),
        ];
        let path = candidates
            .iter()
            .find(|p| std::path::Path::new(p.as_str()).exists())
            .cloned()
            .unwrap_or_else(|| candidates[0].clone());
        if path.ends_with("ggml-rpc-server") {
            (
                path.to_string(),
                vec![
                    "-H".to_string(),
                    "0.0.0.0".to_string(),
                    "-p".to_string(),
                    port.to_string(),
                    "-d".to_string(),
                    "CUDA0".to_string(),
                ],
            )
        } else {
            (
                path.to_string(),
                vec![
                    "--rpc-server-host".to_string(),
                    "0.0.0.0".to_string(),
                    "--rpc-server-port".to_string(),
                    port.to_string(),
                ],
            )
        }
    };

    let lib_path = format!(
        "{}:/usr/local/lib/ollama/cuda_v12",
        crate::cluster_engine::llama_bin_dir().display()
    );

    info!("Starting llama.cpp RPC server on port {}", port);
    info!("Binary: {}", bin_path);

    let mut cmd = Command::new(&bin_path);
    if !cfg!(target_os = "windows") {
        cmd.env("LD_LIBRARY_PATH", lib_path);
    }

    let child = cmd
        .args(&rpc_args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("Failed to spawn RPC server on port {}", port))?;

    info!("RPC server started with PID: {}", child.id());
    Ok(child)
}

pub fn stop_rpc_server(pid: u32) {
    info!("Stopping RPC server PID: {}", pid);
    if cfg!(target_os = "windows") {
        let _ = std::process::Command::new("taskkill")
            .args(&["/F", "/PID", &pid.to_string()])
            .status();
    } else {
        let _ = std::process::Command::new("kill")
            .arg("-15")
            .arg(pid.to_string())
            .status();
        std::thread::sleep(std::time::Duration::from_millis(300));
        let _ = std::process::Command::new("kill")
            .arg("-9")
            .arg(pid.to_string())
            .status();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_probe_live_pc() {
        let ip = get_local_ip();
        println!("DETECTED LOCAL IP: {}", ip);
        let res = probe_remote_giver("10.158.255.229:50051", "vkremote-secret").await;
        println!("LIVE PC PROBE RESULT: {:#?}", res);
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_scan_lan() {
        let ip = get_local_ip();
        println!("SCANNING LAN FROM LOCAL IP: {}", ip);
        let found = scan_lan_for_givers(&ip, 50051).await;
        println!("FOUND GIVERS: {:?}", found);
    }
}
