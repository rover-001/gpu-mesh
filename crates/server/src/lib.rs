pub mod engine;
pub mod session;

use anyhow::{Context, Result};
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tracing::{error, info, warn};
use vkremote_wire::framing::{read_msg, server_handshake, write_msg, DEFAULT_MAX_FRAME_SIZE};
use vkremote_wire::types::{Request, Response};

pub use engine::{Engine, MockEngine, VulkanEngine};
pub use session::{MockSession, SessionHandler, VulkanSession};

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub token: String,
    pub gpu_index: Option<usize>,
    pub force_mock: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            token: "vkremote-secret".to_string(),
            gpu_index: None,
            force_mock: false,
        }
    }
}

/// Run server with basic parameters
pub async fn run_server(
    listener: TcpListener,
    token: String,
    gpu_index: Option<usize>,
) -> Result<()> {
    let config = ServerConfig {
        token,
        gpu_index,
        force_mock: false,
    };
    run_server_with_config(listener, config).await
}

/// Run server with full configuration
pub async fn run_server_with_config(listener: TcpListener, config: ServerConfig) -> Result<()> {
    let engine = Engine::new(config.gpu_index, config.force_mock)?;
    let local_addr = listener.local_addr()?;
    info!("vkremote-server listening on {}", local_addr);

    let token = Arc::new(config.token);

    loop {
        match listener.accept().await {
            Ok((socket, peer_addr)) => {
                info!("Accepted connection from {}", peer_addr);
                let conn_token = token.clone();
                let conn_engine = engine.clone();

                tokio::spawn(async move {
                    if let Err(e) = handle_connection(socket, conn_token, conn_engine).await {
                        warn!("Connection from {} closed with error: {:#}", peer_addr, e);
                    } else {
                        info!("Connection from {} closed cleanly", peer_addr);
                    }
                });
            }
            Err(e) => {
                error!("Failed to accept incoming connection: {:#}", e);
            }
        }
    }
}

/// Handle a single client connection session
/// Per-connection request timing, reported every 5 s when `VKREMOTE_PERF=1`.
struct ServerPerf {
    names: std::collections::HashMap<std::mem::Discriminant<Request>, String>,
    stats: std::collections::HashMap<String, (u64, u64, u64)>, // count, handle_us, write_us
    started: std::time::Instant,
}

impl ServerPerf {
    fn new() -> Self {
        Self {
            names: Default::default(),
            stats: Default::default(),
            started: std::time::Instant::now(),
        }
    }

    fn record(&mut self, name: String, handle_us: u64, write_us: u64) {
        let s = self.stats.entry(name).or_default();
        s.0 += 1;
        s.1 += handle_us;
        s.2 += write_us;
        let elapsed = self.started.elapsed();
        if elapsed.as_secs() < 5 {
            return;
        }
        let secs = elapsed.as_secs_f64();
        let (mut handle, mut write, mut count) = (0u64, 0u64, 0u64);
        let mut rows: Vec<_> = self.stats.drain().collect();
        for (_, s) in &rows {
            count += s.0;
            handle += s.1;
            write += s.2;
        }
        rows.sort_by(|a, b| (b.1 .1 + b.1 .2).cmp(&(a.1 .1 + a.1 .2)));
        let top: Vec<String> = rows
            .iter()
            .take(6)
            .map(|(n, s)| {
                format!(
                    "{}x{}={:.0}+{:.0}ms",
                    n,
                    s.0,
                    s.1 as f64 / 1000.0,
                    s.2 as f64 / 1000.0
                )
            })
            .collect();
        eprintln!(
            "[vkremote-server-perf] window={:.1}s reqs={} ({:.0}/s) handle={:.0}ms ({:.0}%) write={:.0}ms ({:.0}%) | top(handle+write): {}",
            secs,
            count,
            count as f64 / secs,
            handle as f64 / 1000.0,
            handle as f64 / secs / 1e4, // µs over the window, as % of wall time
            write as f64 / 1000.0,
            write as f64 / secs / 1e4,
            top.join(" ")
        );
        self.started = std::time::Instant::now();
    }
}

pub async fn handle_connection(
    mut stream: TcpStream,
    token: Arc<String>,
    engine: Engine,
) -> Result<()> {
    // Tune socket for low-latency inference: no Nagle, large buffers
    let _ = stream.set_nodelay(true);
    // Set large socket buffers via the underlying std socket
    #[cfg(unix)]
    {
        let fd = stream.as_raw_fd();
        let buf_size: libc::c_int = 256 * 1024;
        unsafe {
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_SNDBUF,
                &buf_size as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_RCVBUF,
                &buf_size as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
        }
    }
    // 1. Authenticate connection with handshake
    let _client_req = server_handshake(&mut stream, |req| {
        if req.token == *token {
            (true, "Authentication successful".to_string())
        } else {
            (false, "Invalid authentication token".to_string())
        }
    })
    .await
    .context("Handshake failed")?;

    info!("Client handshake succeeded");

    // 2. Instantiate session state
    let mut session = SessionHandler::new(&engine);

    // 3. Process requests
    let mut perf = ServerPerf::new();
    loop {
        let req: Request = match read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await {
            Ok(msg) => msg,
            Err(vkremote_wire::ProtocolError::Io(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof
                    || e.kind() == std::io::ErrorKind::ConnectionReset =>
            {
                break;
            }
            Err(e) => {
                return Err(e.into());
            }
        };

        if !vkremote_wire::perf::enabled() {
            let resp: Response = session.handle_request(req);
            write_msg(&mut stream, &resp).await?;
            continue;
        }

        let kind = std::mem::discriminant(&req);
        let name = perf.names.entry(kind).or_insert_with(|| vkremote_wire::perf::variant_name(&req)).clone();
        let t_handle = std::time::Instant::now();
        let resp: Response = session.handle_request(req);
        let handle_us = t_handle.elapsed().as_micros() as u64;
        let t_write = std::time::Instant::now();
        write_msg(&mut stream, &resp).await?;
        let write_us = t_write.elapsed().as_micros() as u64;
        perf.record(name, handle_us, write_us);
    }

    Ok(())
}
