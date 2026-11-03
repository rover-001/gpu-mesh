use anyhow::{Context, Result};
use std::fs;
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use vkremote_wire::framing::{client_handshake, read_msg, write_msg, DEFAULT_MAX_FRAME_SIZE};
use vkremote_wire::types::{Request, Response};

#[cfg(windows)]
extern "system" {
    fn SetConsoleTitleW(lpConsoleTitle: *const u16) -> i32;
}

fn set_window_title(title: &str) {
    #[cfg(windows)]
    unsafe {
        let wide: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
        SetConsoleTitleW(wide.as_ptr());
    }
    print!("\x1B]0;{}\x07", title);
}

#[tokio::main]
async fn main() -> Result<()> {
    let server_addr = std::env::var("VKREMOTE_SERVER")
        .ok()
        .or_else(|| {
            fs::read_to_string("vkremote_server.txt")
                .ok()
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "127.0.0.1:50051".to_string());

    let token = std::env::var("VKREMOTE_TOKEN")
        .ok()
        .or_else(|| {
            fs::read_to_string("vkremote_token.txt")
                .ok()
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "vkremote-secret".to_string());

    println!("==================================================");
    println!("   vkremote - Remote GPU Monitor & Service        ");
    println!("==================================================");
    println!("Connecting to host at {}...", server_addr);

    let mut stream = TcpStream::connect(&server_addr)
        .await
        .with_context(|| format!("Failed to connect to {}", server_addr))?;

    client_handshake(&mut stream, token).await?;

    // Request snapshot
    write_msg(&mut stream, &Request::GetDeviceSnapshot).await?;
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await?;

    let snapshot = match resp {
        Response::DeviceSnapshot(s) => s,
        _ => return Err(anyhow::anyhow!("Unexpected response from server")),
    };

    let gpu_name = format!("{} (Remote)", snapshot.properties.device_name);
    let vram_mb = snapshot.memory_properties.memory_heaps[0].size / (1024 * 1024);

    // Set process title in Windows Task Manager / Console
    let title = format!("vkremote - {}", gpu_name);
    set_window_title(&title);

    println!("\n[CONNECTED] Remote GPU: {}", gpu_name);
    println!(
        "            VRAM:       {:.2} GB ({} MB)",
        vram_mb as f64 / 1024.0,
        vram_mb
    );
    println!(
        "            API:        {}.{}.{}",
        ash::vk::api_version_major(snapshot.properties.api_version),
        ash::vk::api_version_minor(snapshot.properties.api_version),
        ash::vk::api_version_patch(snapshot.properties.api_version)
    );
    println!("\nThis process is now active in Windows Task Manager under Processes:");
    println!("  >> \"{}\"", title);
    println!("Monitoring connection health and telemetry...\n");

    let mut pings = 0u64;
    loop {
        let start = Instant::now();
        // Ping with snapshot check
        let req = Request::GetDeviceSnapshot;
        if write_msg(&mut stream, &req).await.is_err() {
            println!("\n[DISCONNECTED] Host connection lost. Retrying in 2s...");
            tokio::time::sleep(Duration::from_secs(2)).await;
            continue;
        }

        let _ = read_msg::<_, Response>(&mut stream, DEFAULT_MAX_FRAME_SIZE).await;
        let rtt = start.elapsed();
        pings += 1;

        print!(
            "\r[HEARTBEAT #{}] Host: {} | RTT: {:.2}ms | Status: ACTIVE & HEALTHY   ",
            pings,
            server_addr,
            rtt.as_secs_f64() * 1000.0
        );
        std::io::Write::flush(&mut std::io::stdout())?;

        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}
