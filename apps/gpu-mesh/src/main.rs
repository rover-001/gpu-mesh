mod cluster_engine;
mod mesh;
mod net_setup;
mod network;
mod remote_llm;
mod vulkan_local;

use cluster_engine::{ClusterEngineManager, EngineStatus};
use gpui::*;
use mesh::RemoteGiver;
use network::{
    get_host_name, get_local_ip, ping_remote_giver, probe_remote_giver, scan_lan_for_givers,
    start_giver_server, start_rpc_server, stop_rpc_server,
};
use parking_lot::Mutex;
use remote_llm::{
    largest_fitting_worker, start_remote_session, start_worker_server, ModelManifest, RemoteSession,
};
use std::path::Path;
use std::sync::{Arc, OnceLock};
use tokio::runtime::Runtime;
use tokio::sync::oneshot;
use vulkan_local::{detect_local_gpus, LocalGpuInfo};

static TOKIO_RT: OnceLock<Runtime> = OnceLock::new();

pub fn tokio_rt() -> &'static Runtime {
    TOKIO_RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime")
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnectMode {
    AutoFetch,
    Manual,
}

struct AppState {
    local_name: String,
    local_ip: String,
    local_port: u16,
    giver_enabled: bool,
    giver_status_text: String,
    giver_stop_tx: Option<oneshot::Sender<()>>,
    worker_stop_tx: Option<oneshot::Sender<()>>,
    rpc_server_pid: Option<u32>,
    local_gpus: Vec<LocalGpuInfo>,

    target_endpoint: String,
    candidate_endpoints: Vec<String>,
    manual_ip_input: String,
    connect_mode: ConnectMode,
    remote_peers: Vec<RemoteGiver>,
    is_connecting: bool,
    is_scanning: bool,
    is_refreshing: bool,

    engine_status: EngineStatus,
    engine_pid: Option<u32>,
    remote_session: Option<Arc<tokio::sync::Mutex<RemoteSession>>>,
    web_proxy_stop_tx: Option<oneshot::Sender<()>>,
    available_models: Vec<ModelManifest>,
    selected_model_idx: usize,
    test_running: bool,
    test_prompt: String,
    test_result: Option<String>,
    test_speed: Option<String>,

    status_message: String,
    event_log: Vec<String>,
}

impl AppState {
    fn new() -> Self {
        let local_name = get_host_name();
        let local_ip = get_local_ip();
        let local_gpus = match detect_local_gpus() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[gpu-mesh] Failed to detect local GPUs: {:#}", e);
                Vec::new()
            }
        };
        let local_port = 50051;

        // Initial candidate endpoints: localhost loopback
        let candidates = vec!["127.0.0.1:50051".to_string()];
        let target_endpoint = "127.0.0.1:50051".to_string();

        // Auto-start Giver sharing on startup so the node is immediately available on mesh
        let worker_stop_tx = match start_worker_server(50053, "vkremote-secret".to_string()) {
            Ok(stop_tx) => Some(stop_tx),
            Err(e) => {
                eprintln!("[gpu-mesh] LLM worker unavailable at startup: {e:#}");
                None
            }
        };

        let (giver_enabled, giver_status_text, giver_stop_tx, rpc_server_pid) =
            match start_giver_server(local_port, "vkremote-secret".to_string(), None) {
                Ok(tx) => {
                    let rpc_pid = match start_rpc_server(50052) {
                        Ok(child) => {
                            eprintln!(
                                "[gpu-mesh] Auto-started RPC server on port 50052 (PID {})",
                                child.id()
                            );
                            Some(child.id())
                        }
                        Err(e) => {
                            eprintln!("[gpu-mesh] RPC server unavailable at startup: {:#}", e);
                            None
                        }
                    };
                    (
                        true,
                        format!("LISTENING on 0.0.0.0:{}", local_port),
                        Some(tx),
                        rpc_pid,
                    )
                }
                Err(e) => (
                    false,
                    format!("Server Stopped (Port {} idle: {})", local_port, e),
                    None,
                    None,
                ),
            };

        let status_message = if giver_enabled {
            format!(
                "Node active: {} ({}) • Sharing GPU on :{} • Ready",
                local_name, local_ip, local_port
            )
        } else {
            format!("Node ready: {} ({}) • 0 remote peers", local_name, local_ip)
        };

        Self {
            local_name: local_name.clone(),
            local_ip: local_ip.clone(),
            local_port,
            giver_enabled,
            giver_status_text,
            giver_stop_tx,
            worker_stop_tx,
            rpc_server_pid,
            local_gpus,

            target_endpoint,
            candidate_endpoints: candidates,
            manual_ip_input: String::new(),
            connect_mode: ConnectMode::AutoFetch,
            remote_peers: Vec::new(), // Zero fake data: starts empty
            is_connecting: false,
            is_scanning: false,
            is_refreshing: false,

            engine_status: EngineStatus::Stopped,
            engine_pid: None,
            remote_session: None,
            web_proxy_stop_tx: None,
            available_models: vec![
                ModelManifest::default_qwen_7b(),
                ModelManifest::qwen_14b_coder(),
            ],
            selected_model_idx: 0,
            test_running: false,
            test_prompt: "Calculate 42 * 1337 and explain in 1 sentence:".to_string(),
            test_result: None,
            test_speed: None,

            status_message,
            event_log: vec![format!(
                "Initialized host node {} @ {}",
                local_name, local_ip
            )],
        }
    }

    fn local_vram_mb(&self) -> u64 {
        for gpu in &self.local_gpus {
            if gpu.is_discrete {
                return gpu.vram_mb;
            }
        }
        self.local_gpus.first().map(|g| g.vram_mb).unwrap_or(0)
    }

    fn remote_vram_mb(&self) -> u64 {
        self.remote_peers.iter().map(|p| p.effective_vram_mb()).sum()
    }

    fn total_vram_mb(&self) -> u64 {
        self.local_vram_mb() + self.remote_vram_mb()
    }
}

impl Drop for AppState {
    fn drop(&mut self) {
        // Clean up llama-server when app exits
        if let Some(pid) = self.engine_pid.take() {
            eprintln!(
                "[gpu-mesh] App exiting - stopping llama-server PID: {}",
                pid
            );
            ClusterEngineManager::stop_server(pid);
        }
        self.remote_session = None;
        if let Some(tx) = self.worker_stop_tx.take() {
            let _ = tx.send(());
        }

        // Stop RPC server
        if let Some(pid) = self.rpc_server_pid.take() {
            eprintln!("[gpu-mesh] App exiting - stopping RPC server PID: {}", pid);
            stop_rpc_server(pid);
        }

        // Stop giver server
        if let Some(tx) = self.giver_stop_tx.take() {
            let _ = tx.send(());
        }
    }
}

struct GpuMeshApp {
    state: Arc<Mutex<AppState>>,
    focus_handle: FocusHandle,
}

impl GpuMeshApp {
    fn new(cx: &mut Context<Self>) -> Self {
        let app = Self {
            state: Arc::new(Mutex::new(AppState::new())),
            focus_handle: cx.focus_handle(),
        };
        // Per-device setup: open this machine's firewall for peer ports (prompts
        // once via the OS only if rules are missing). Results go to the event log.
        let state_fw = Arc::clone(&app.state);
        std::thread::spawn(move || {
            let mut lines = Vec::new();
            let orphans = ClusterEngineManager::kill_orphaned_engines();
            if orphans > 0 {
                lines.push(format!(
                    "Stopped {orphans} llama-server left running by a previous gpu-mesh session"
                ));
            }
            lines.extend(crate::net_setup::ensure_peer_ports_open());
            let mut s = state_fw.lock();
            for line in lines {
                eprintln!("[gpu-mesh] {line}");
                s.event_log.push(line);
            }
        });
        let state_poll = Arc::clone(&app.state);
        tokio_rt().spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
                // Only reflect a server this instance owns (started locally or a remote
                // session). A foreign process on :8080 must not look like "auto-started",
                // and an idle app makes no health requests at all.
                let owned = {
                    let s = state_poll.lock();
                    s.engine_pid.is_some() || s.remote_session.is_some()
                };
                if !owned {
                    continue;
                }
                // Blocking socket call: keep it off the async worker threads.
                let health = tokio::task::spawn_blocking(
                    crate::cluster_engine::ClusterEngineManager::check_health,
                )
                .await
                .unwrap_or_else(|e| Err(e.to_string()));
                if let Ok(health) = health {
                    if health == "ok" {
                        let mut s = state_poll.lock();
                        if s.engine_status != EngineStatus::Running {
                            s.engine_status = EngineStatus::Running;
                            s.status_message = "Cluster AI server ACTIVE on http://127.0.0.1:8080".to_string();
                        }
                    } else if health == "loading" || health.contains("loading") {
                        let mut s = state_poll.lock();
                        if matches!(s.engine_status, EngineStatus::Stopped) {
                            s.engine_status = EngineStatus::Starting("Loading model to VRAM...".to_string());
                        }
                    }
                }
            }
        });
        app
    }
}

impl Render for GpuMeshApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (
            local_name,
            local_ip,
            local_port,
            giver_enabled,
            _giver_status_text,
            local_gpus,
            target_endpoint,
            candidate_endpoints,
            manual_ip_input,
            connect_mode,
            remote_peers,
            is_connecting,
            is_scanning,
            is_refreshing,
            status_message,
            local_vram_mb,
            remote_vram_mb,
            total_vram_mb,
            engine_status,
            _remote_session_active,
            test_running,
            test_prompt,
            test_result,
            test_speed,
            available_models,
            selected_model_idx,
        ) = {
            let s = self.state.lock();
            (
                s.local_name.clone(),
                s.local_ip.clone(),
                s.local_port,
                s.giver_enabled,
                s.giver_status_text.clone(),
                s.local_gpus.clone(),
                s.target_endpoint.clone(),
                s.candidate_endpoints.clone(),
                s.manual_ip_input.clone(),
                s.connect_mode,
                s.remote_peers.clone(),
                s.is_connecting,
                s.is_scanning,
                s.is_refreshing,
                s.status_message.clone(),
                s.local_vram_mb(),
                s.remote_vram_mb(),
                s.total_vram_mb(),
                s.engine_status.clone(),
                s.remote_session.is_some(),
                s.test_running,
                s.test_prompt.clone(),
                s.test_result.clone(),
                s.test_speed.clone(),
                s.available_models.clone(),
                s.selected_model_idx,
            )
        };

        let state_arc = Arc::clone(&self.state);

        let local_gb = local_vram_mb as f64 / 1024.0;
        let remote_gb = remote_vram_mb as f64 / 1024.0;
        let total_gb = total_vram_mb as f64 / 1024.0;

        let (local_pct, remote_pct, local_fraction) = if total_vram_mb > 0 {
            let l_pct = (local_vram_mb as f64 / total_vram_mb as f64) * 100.0;
            let r_pct = (remote_vram_mb as f64 / total_vram_mb as f64) * 100.0;
            let frac = (local_vram_mb as f32 / total_vram_mb as f32).clamp(0.05, 0.95);
            (l_pct, r_pct, frac)
        } else {
            (100.0, 0.0, 1.0)
        };

        div()
            .size_full()
            .bg(rgb(0x09090b))
            .flex()
            .flex_col()
            .text_color(rgb(0xf4f4f5))
            // ------------------ HEADER ------------------
            .child(
                div()
                    .flex()
                    .flex_row()
                    .justify_between()
                    .items_center()
                    .px_6()
                    .py_4()
                    .border_b_1()
                    .border_color(rgb(0x27272a))
                    .bg(rgb(0x111114))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .flex()
                                    .flex_row()
                                    .items_center()
                                    .gap_3()
                                    .child(
                                        div()
                                            .text_xl()
                                            .font_weight(FontWeight::BOLD)
                                            .text_color(rgb(0xffffff))
                                            .child("rgpu // GPU Mesh Monitor")
                                    )
                                    .child(
                                        div()
                                            .px_2()
                                            .py_0p5()
                                            .rounded_md()
                                            .bg(if giver_enabled { rgb(0x064e3b) } else { rgb(0x27272a) })
                                            .border_1()
                                            .border_color(if giver_enabled { rgb(0x10b981) } else { rgb(0x3f3f46) })
                                            .text_xs()
                                            .text_color(if giver_enabled { rgb(0x34d399) } else { rgb(0xa1a1aa) })
                                            .child(if giver_enabled { "● GIVER ACTIVE (:50051)" } else { "○ GIVER IDLE" })
                                    )
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(rgb(0x71717a))
                                    .child(format!("Node: {} • IP: {} • vkremote Wire Protocol v1", local_name, local_ip))
                            )
                    )
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap_3()
                            .child(
                                div()
                                    .id("btn-ping-all")
                                    .px_3()
                                    .py_1p5()
                                    .rounded_md()
                                    .bg(if is_refreshing { rgb(0x3f3f46) } else { rgb(0x27272a) })
                                    .border_1()
                                    .border_color(rgb(0x3f3f46))
                                    .text_xs()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(rgb(0xe4e4e7))
                                    .cursor_pointer()
                                    .on_mouse_down(MouseButton::Left, {
                                        let state = Arc::clone(&state_arc);
                                        cx.listener(move |_this, _, _, cx| {
                                            let endpoints: Vec<String> = {
                                                let mut s = state.lock();
                                                if s.remote_peers.is_empty() || s.is_refreshing {
                                                    s.status_message = "No remote peers to ping".to_string();
                                                    return;
                                                }
                                                s.is_refreshing = true;
                                                s.status_message = "Pinging all connected peers...".to_string();
                                                s.remote_peers.iter().map(|p| p.endpoint.clone()).collect()
                                            };
                                            let state_bg = Arc::clone(&state);
                                            tokio_rt().spawn(async move {
                                                for ep in endpoints {
                                                    let outcome = ping_remote_giver(&ep, "vkremote-secret").await;
                                                    let mut s = state_bg.lock();
                                                    if let Some(peer) = s.remote_peers.iter_mut().find(|p| p.endpoint == ep) {
                                                        match outcome {
                                                            Ok(lat) => {
                                                                peer.latency_ms = lat;
                                                                peer.status = mesh::ConnectionStatus::Connected;
                                                            }
                                                            Err(_) => {
                                                                peer.status = mesh::ConnectionStatus::Disconnected;
                                                            }
                                                        }
                                                    }
                                                }
                                                let mut s = state_bg.lock();
                                                s.is_refreshing = false;
                                                s.status_message = "Ping refresh complete".to_string();
                                            });
                                            cx.notify();
                                        })
                                    })
                                    .child(if is_refreshing { "Pinging..." } else { "↻ Ping All" })
                            )
                    )
            )
            // ------------------ SCROLLABLE WORKSPACE CONTAINER ------------------
            .child(
                div()
                    .id("workspace-scroll-container")
                    .flex_1()
                    .overflow_y_scroll()
                    .flex()
                    .flex_col()
                    // ------------------ DYNAMIC METRICS BAR ------------------
                    .child(
                div()
                    .px_6()
                    .py_5()
                    .bg(rgb(0x131316))
                    .border_b_1()
                    .border_color(rgb(0x27272a))
                    .flex()
                    .flex_col()
                    .gap_4()
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .gap_4()
                            // Metric 1: Local VRAM
                            .child(
                                div()
                                    .flex_1()
                                    .p_4()
                                    .rounded_lg()
                                    .bg(rgb(0x18181b))
                                    .border_1()
                                    .border_color(rgb(0x27272a))
                                    .flex()
                                    .flex_col()
                                    .gap_1()
                                    .child(
                                        div()
                                            .flex()
                                            .flex_row()
                                            .justify_between()
                                            .items_center()
                                            .child(div().text_xs().font_weight(FontWeight::MEDIUM).text_color(rgb(0xa1a1aa)).child("LOCAL VRAM"))
                                            .child(div().px_2().py_0p5().rounded_sm().bg(rgb(0x0c4a6e)).text_xs().text_color(rgb(0x38bdf8)).child("THIS HOST"))
                                    )
                                    .child(
                                        div()
                                            .text_2xl()
                                            .font_weight(FontWeight::BOLD)
                                            .text_color(rgb(0x38bdf8))
                                            .child(format!("{:.2} GB", local_gb))
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(rgb(0x71717a))
                                            .child(local_gpus.first().map(|g| g.name.clone()).unwrap_or_else(|| "No GPU detected".to_string()))
                                    )
                            )
                            // Metric 2: Remote VRAM
                            .child(
                                div()
                                    .flex_1()
                                    .p_4()
                                    .rounded_lg()
                                    .bg(rgb(0x18181b))
                                    .border_1()
                                    .border_color(rgb(0x27272a))
                                    .flex()
                                    .flex_col()
                                    .gap_1()
                                    .child(
                                        div()
                                            .flex()
                                            .flex_row()
                                            .justify_between()
                                            .items_center()
                                            .child(div().text_xs().font_weight(FontWeight::MEDIUM).text_color(rgb(0xa1a1aa)).child("REMOTE VRAM (AGGREGATED)"))
                                            .child(div().px_2().py_0p5().rounded_sm().bg(rgb(0x2e1065)).text_xs().text_color(rgb(0xc084fc)).child(format!("{} PEER(S)", remote_peers.len())))
                                    )
                                    .child(
                                        div()
                                            .text_2xl()
                                            .font_weight(FontWeight::BOLD)
                                            .text_color(rgb(0xa78bfa))
                                            .child(format!("{:.2} GB", remote_gb))
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(rgb(0x71717a))
                                            .child(if remote_peers.is_empty() { "No remote peers connected".to_string() } else { format!("{} MB from active remote givers", remote_vram_mb) })
                                    )
                            )
                            // Metric 3: Total Cluster Pool
                            .child(
                                div()
                                    .flex_1()
                                    .p_4()
                                    .rounded_lg()
                                    .bg(rgb(0x18181b))
                                    .border_1()
                                    .border_color(rgb(0x27272a))
                                    .flex()
                                    .flex_col()
                                    .gap_1()
                                    .child(
                                        div()
                                            .flex()
                                            .flex_row()
                                            .justify_between()
                                            .items_center()
                                            .child(div().text_xs().font_weight(FontWeight::MEDIUM).text_color(rgb(0xa1a1aa)).child("TOTAL CLUSTER POOL"))
                                            .child(div().px_2().py_0p5().rounded_sm().bg(rgb(0x064e3b)).text_xs().text_color(rgb(0x34d399)).child("UNIFIED"))
                                    )
                                    .child(
                                        div()
                                            .text_2xl()
                                            .font_weight(FontWeight::BOLD)
                                            .text_color(rgb(0x10b981))
                                            .child(format!("{:.2} GB", total_gb))
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(rgb(0x71717a))
                                            .child(format!("{} MB combined pool across {} GPU(s)", total_vram_mb, local_gpus.len() + remote_peers.len()))
                                    )
                            )
                    )
                    // Real Dynamic Proportional VRAM Bar
                    .child(
                        if remote_vram_mb == 0 {
                            div()
                                .h(px(8.0))
                                .w_full()
                                .rounded_full()
                                .bg(rgb(0x0284c7))
                        } else {
                            div()
                                .h(px(8.0))
                                .w_full()
                                .rounded_full()
                                .bg(rgb(0x27272a))
                                .overflow_hidden()
                                .flex()
                                .flex_row()
                                .child(
                                    div()
                                        .h_full()
                                        .w(relative(local_fraction))
                                        .bg(rgb(0x0284c7))
                                )
                                .child(
                                    div()
                                        .h_full()
                                        .w(relative(1.0 - local_fraction))
                                        .bg(rgb(0x8b5cf6))
                                )
                        }
                    )
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .justify_between()
                            .text_xs()
                            .text_color(rgb(0x71717a))
                            .child(
                                div()
                                    .flex()
                                    .flex_row()
                                    .items_center()
                                    .gap_2()
                                    .child(div().w(px(8.0)).h(px(8.0)).rounded_sm().bg(rgb(0x0284c7)))
                                    .child(format!("Local ({}): {:.1} GB ({:.1}%)", local_name, local_gb, local_pct))
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_row()
                                    .items_center()
                                    .gap_2()
                                    .child(div().w(px(8.0)).h(px(8.0)).rounded_sm().bg(rgb(0x8b5cf6)))
                                    .child(if remote_vram_mb == 0 {
                                        "Remote: 0.0 GB (0%) • Standby".to_string()
                                    } else {
                                        format!("Remote ({} peer(s)): {:.1} GB ({:.1}%)", remote_peers.len(), remote_gb, remote_pct)
                                    })
                            )
                    )
            )
            // ------------------ MAIN WORKSPACE (2 COLUMNS) ------------------
            .child(
                div()
                    .flex_1()
                    .flex()
                    .flex_row()
                    .p_6()
                    .gap_6()
                    // ===== LEFT COLUMN: LOCAL HOST & GIVER CONTROL =====
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap_4()
                            // Local Giver Control Card
                            .child(
                                div()
                                    .p_5()
                                    .rounded_lg()
                                    .bg(rgb(0x18181b))
                                    .border_1()
                                    .border_color(rgb(0x27272a))
                                    .flex()
                                    .flex_col()
                                    .gap_4()
                                    .child(
                                        div()
                                            .flex()
                                            .flex_row()
                                            .justify_between()
                                            .items_center()
                                            .child(
                                                div()
                                                    .text_base()
                                                    .font_weight(FontWeight::BOLD)
                                                    .text_color(rgb(0xffffff))
                                                    .child("Share GPU")
                                            )
                                            .child(
                                                div()
                                                    .px_2()
                                                    .py_0p5()
                                                    .rounded_md()
                                                    .bg(rgb(0x27272a))
                                                    .text_xs()
                                                    .text_color(rgb(0xa1a1aa))
                                                    .child(format!("{}:{}", local_ip, local_port))
                                            )
                                    )
                                    // REAL TOGGLE SWITCH ROW
                                    .child(
                                        div()
                                            .flex()
                                            .flex_row()
                                            .justify_between()
                                            .items_center()
                                            .p_4()
                                            .rounded_md()
                                            .bg(rgb(0x202024))
                                            .border_1()
                                            .border_color(if giver_enabled { rgb(0x059669) } else { rgb(0x27272a) })
                                            .child(
                                                div()
                                                    .flex()
                                                    .flex_col()
                                                    .gap_1()
                                                    .child(
                                                        div()
                                                            .text_sm()
                                                            .font_weight(FontWeight::SEMIBOLD)
                                                            .text_color(rgb(0xffffff))
                                                            .child("Share This GPU")
                                                    )
                                                    .child(
                                                        div()
                                                            .text_xs()
                                                            .text_color(if giver_enabled { rgb(0x34d399) } else { rgb(0x71717a) })
                                                            .child(if giver_enabled { format!("Actively sharing on port {}", local_port) } else { "GPU sharing is stopped".to_string() })
                                                    )
                                            )
                                            .child(
                                                // Modern clickable toggle pill
                                                div()
                                                    .id("btn-toggle-giver")
                                                    .cursor_pointer()
                                                    .flex()
                                                    .flex_row()
                                                    .items_center()
                                                    .gap_2()
                                                    .on_mouse_down(MouseButton::Left, {
                                                        let state = Arc::clone(&state_arc);
                                                        cx.listener(move |_this, _, _, cx| {
                                                            let mut s = state.lock();
                                                            if !s.giver_enabled {
                                                                let port = s.local_port;
                                                                let worker_tx = match start_worker_server(50053, "vkremote-secret".to_string()) {
                                                                    Ok(tx) => Some(tx),
                                                                    Err(e) => {
                                                                        eprintln!("[gpu-mesh] Worker server start error: {e:#}");
                                                                        None
                                                                    }
                                                                };
                                                                s.worker_stop_tx = worker_tx;
                                                                match start_giver_server(port, "vkremote-secret".to_string(), None) {
                                                                    Ok(stop_tx) => {
                                                                        s.giver_enabled = true;
                                                                        s.giver_stop_tx = Some(stop_tx);
                                                                        s.giver_status_text = format!("LISTENING on 0.0.0.0:{}", port);
                                                                        s.status_message = format!("Giver server ACTIVE on 0.0.0.0:{}", port);
                                                                        s.event_log.push(format!("Started Giver server on port {}", port));
                                                                        
                                                                        // Also start RPC server for llama.cpp mesh
                                                                        match start_rpc_server(50052) {
                                                                            Ok(child) => {
                                                                                s.rpc_server_pid = Some(child.id());
                                                                                s.event_log.push("Started RPC server on port 50052".to_string());
                                                                                std::mem::forget(child); // Prevent blocking
                                                                            }
                                                                            Err(e) => {
                                                                                eprintln!("[gpu-mesh] Failed to start RPC server: {:#}", e);
                                                                                s.event_log.push(format!("RPC server failed: {}", e));
                                                                            }
                                                                        }
                                                                    }
                                                                    Err(e) => {
                                                                        s.giver_enabled = false;
                                                                        s.giver_status_text = format!("Failed to bind port {}: {}", port, e);
                                                                        s.status_message = format!("Error starting Giver: {}", e);
                                                                        s.event_log.push(format!("Bind error: {}", e));
                                                                    }
                                                                }
                                                            } else {
                                                                if let Some(tx) = s.giver_stop_tx.take() {
                                                                    let _ = tx.send(());
                                                                }
                                                                if let Some(tx) = s.worker_stop_tx.take() {
                                                                    let _ = tx.send(());
                                                                }
                                                                if let Some(pid) = s.rpc_server_pid.take() {
                                                                    stop_rpc_server(pid);
                                                                    s.event_log.push("Stopped RPC server".to_string());
                                                                }
                                                                if cfg!(target_os = "windows") {
                                                                    let _ = std::process::Command::new("taskkill")
                                                                        .args(&["/F", "/IM", "ggml-rpc-server.exe"])
                                                                        .status();
                                                                } else {
                                                                    let _ = std::process::Command::new("pkill")
                                                                        .arg("-9")
                                                                        .arg("ggml-rpc-server")
                                                                        .status();
                                                                }
                                                                s.giver_enabled = false;
                                                                s.giver_status_text = format!("Server Stopped (Port {} idle)", s.local_port);
                                                                s.status_message = "GPU sharing disabled".to_string();
                                                                s.event_log.push("Disabled GPU sharing".to_string());
                                                            }
                                                            cx.notify();
                                                        })
                                                    })
                                                    .child(
                                                        div()
                                                            .w(px(46.0))
                                                            .h(px(24.0))
                                                            .rounded_full()
                                                            .bg(if giver_enabled { rgb(0x10b981) } else { rgb(0x3f3f46) })
                                                            .p(px(2.0))
                                                            .flex()
                                                            .items_center()
                                                            .child(
                                                                div()
                                                                    .w(px(20.0))
                                                                    .h(px(20.0))
                                                                    .rounded_full()
                                                                    .bg(rgb(0xffffff))
                                                                    .ml(if giver_enabled { px(22.0) } else { px(0.0) })
                                                            )
                                                    )
                                                    .child(
                                                        div()
                                                            .text_xs()
                                                            .font_weight(FontWeight::BOLD)
                                                            .text_color(if giver_enabled { rgb(0x10b981) } else { rgb(0x71717a) })
                                                            .child(if giver_enabled { "ON" } else { "OFF" })
                                                    )
                                            )
                                    )
                            )
                            // Local Hardware GPUs List Card
                            .child(
                                div()
                                    .p_5()
                                    .rounded_lg()
                                    .bg(rgb(0x18181b))
                                    .border_1()
                                    .border_color(rgb(0x27272a))
                                    .flex()
                                    .flex_col()
                                    .gap_3()
                                    .child(
                                        div()
                                            .flex()
                                            .flex_row()
                                            .justify_between()
                                            .items_center()
                                            .child(
                                                div()
                                                    .text_sm()
                                                    .font_weight(FontWeight::BOLD)
                                                    .text_color(rgb(0xffffff))
                                                    .child(format!("Local Vulkan Hardware ({} device{})", local_gpus.len(), if local_gpus.len() == 1 { "" } else { "s" }))
                                            )
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .text_color(rgb(0x71717a))
                                                    .child(local_name.clone())
                                            )
                                    )
                                    .children(
                                        local_gpus.iter().map(|gpu| {
                                            div()
                                                .p_3()
                                                .rounded_md()
                                                .bg(rgb(0x111113))
                                                .border_1()
                                                .border_color(rgb(0x27272a))
                                                .flex()
                                                .flex_row()
                                                .justify_between()
                                                .items_center()
                                                .child(
                                                    div()
                                                        .flex()
                                                        .flex_col()
                                                        .gap_1()
                                                        .child(
                                                            div()
                                                                .text_sm()
                                                                .font_weight(FontWeight::SEMIBOLD)
                                                                .text_color(rgb(0xffffff))
                                                                .child(gpu.name.clone())
                                                        )
                                                        .child(
                                                            div()
                                                                .text_xs()
                                                                .text_color(rgb(0x71717a))
                                                                .child(format!("Vulkan {} • Driver {}", gpu.api_version, gpu.driver_version))
                                                        )
                                                )
                                                .child(
                                                    div()
                                                        .flex()
                                                        .flex_col()
                                                        .items_end()
                                                        .child(
                                                            div()
                                                                .text_sm()
                                                                .font_weight(FontWeight::BOLD)
                                                                .text_color(rgb(0x38bdf8))
                                                                .child(format!("{:.2} GB", gpu.vram_mb as f64 / 1024.0))
                                                        )
                                                        .child(
                                                            div()
                                                                .px_1p5()
                                                                .py_0p5()
                                                                .rounded_sm()
                                                                .bg(if gpu.is_discrete { rgb(0x064e3b) } else { rgb(0x27272a) })
                                                                .text_xs()
                                                                .text_color(if gpu.is_discrete { rgb(0x34d399) } else { rgb(0x71717a) })
                                                                .child(if gpu.is_discrete { "Discrete" } else { "Integrated" })
                                                        )
                                                )
                                        })
                                    )
                            )
                            // AI CLUSTER INFERENCE & WEB UI CARD
                            .child(
                                div()
                                    .p_5()
                                    .rounded_lg()
                                    .bg(rgb(0x18181b))
                                    .border_1()
                                    .border_color(rgb(0x27272a))
                                    .flex()
                                    .flex_col()
                                    .gap_3()
                                    .child(
                                        div()
                                            .flex()
                                            .flex_row()
                                            .justify_between()
                                            .items_center()
                                            .child(
                                                div()
                                                    .text_base()
                                                    .font_weight(FontWeight::BOLD)
                                                    .text_color(rgb(0xffffff))
                                                    .child("Distributed AI Engine")
                                            )
                                            .child(
                                                match &engine_status {
                                                    EngineStatus::Stopped => div().px_2().py_0p5().rounded_md().bg(rgb(0x27272a)).text_xs().text_color(rgb(0xa1a1aa)).child("● Stopped"),
                                                    EngineStatus::Starting(_) => div().px_2().py_0p5().rounded_md().bg(rgb(0x78350f)).text_xs().text_color(rgb(0xfbbf24)).child("● Loading..."),
                                                    EngineStatus::Running => {
                                                        div().px_2().py_0p5().rounded_md().bg(rgb(0x064e3b)).text_xs().text_color(rgb(0x34d399)).child("● Running on :8080")
                                                    },
                                                    EngineStatus::Failed(_) => div().px_2().py_0p5().rounded_md().bg(rgb(0x7f1d1d)).text_xs().text_color(rgb(0xf87171)).child("● Failed"),
                                                }
                                            )
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(rgb(0x71717a))
                                            .child(format!("Cluster Pool: {:.1} GB • Distributed across {} local + {} remote GPU", total_gb, local_name, if remote_peers.is_empty() { "0".to_string() } else { remote_peers.len().to_string() }))
                                    )
                                    .child(
                                        div()
                                            .flex()
                                            .flex_row()
                                            .flex_wrap()
                                            .items_center()
                                            .gap_2()
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .text_color(rgb(0xa1a1aa))
                                                    .font_weight(FontWeight::MEDIUM)
                                                    .child("Model:")
                                            )
                                            .children(available_models.iter().enumerate().map(|(idx, m)| {
                                                let is_selected = idx == selected_model_idx;
                                                let is_disabled = matches!(engine_status, EngineStatus::Running | EngineStatus::Starting(_));
                                                let state = Arc::clone(&state_arc);
                                                let label = if m.name.contains("7B") {
                                                    "Qwen 2.5 Coder 7B".to_string()
                                                } else if m.name.contains("14B") {
                                                    "Qwen 2.5 Coder 14B".to_string()
                                                } else {
                                                    m.name.clone()
                                                };
                                                div()
                                                    .id(("model-btn", idx))
                                                    .px_2p5()
                                                    .py_1()
                                                    .rounded_md()
                                                    .cursor_pointer()
                                                    .bg(if is_selected { rgb(0x2e1065) } else { rgb(0x27272a) })
                                                    .border_1()
                                                    .border_color(if is_selected { rgb(0x7c3aed) } else { rgb(0x3f3f46) })
                                                    .text_xs()
                                                    .font_weight(if is_selected { FontWeight::BOLD } else { FontWeight::NORMAL })
                                                    .text_color(if is_selected { rgb(0xd8b4fe) } else { rgb(0xa1a1aa) })
                                                    .on_mouse_down(MouseButton::Left, {
                                                        let state = Arc::clone(&state);
                                                        cx.listener(move |_this, _, _, cx| {
                                                            if is_disabled { return; }
                                                            let mut s = state.lock();
                                                            s.selected_model_idx = idx;
                                                            s.status_message = format!("Active model set to: {}", s.available_models[idx].name);
                                                            cx.notify();
                                                        })
                                                    })
                                                    .child(label)
                                            }))
                                    )
                                    .child(
                                        div()
                                            .flex()
                                            .flex_row()
                                            .gap_2()
                                            .child(
                                                div()
                                                    .id("btn-toggle-engine")
                                                    .px_3()
                                                    .py_2()
                                                    .rounded_md()
                                                    .bg(match &engine_status {
                                                        EngineStatus::Running => rgb(0xdc2626),
                                                        EngineStatus::Starting(_) => rgb(0x4b5563),
                                                        _ => rgb(0x059669),
                                                    })
                                                    .text_xs()
                                                    .font_weight(FontWeight::BOLD)
                                                    .text_color(rgb(0xffffff))
                                                    .cursor_pointer()
                                                    .on_mouse_down(MouseButton::Left, {
                                                        let state = Arc::clone(&state_arc);
                                                        cx.listener(move |_this, _, _, cx| {
                                                            let mut s = state.lock();
                                                            if s.engine_status == EngineStatus::Stopped || matches!(s.engine_status, EngineStatus::Failed(_)) {
                                                                s.engine_status = EngineStatus::Starting("Allocating cluster VRAM...".to_string());
                                                                s.status_message = "Starting distributed AI server on :8080...".to_string();
                                                                
                                                                // Build RPC endpoint list from all connected remote peers
                                                                let rpc_endpoints: Vec<String> = s.remote_peers.iter()
                                                                    .filter_map(|peer| {
                                                                        let host = peer.endpoint.split(':').next().unwrap_or("");
                                                                        if !host.is_empty() && host != "127.0.0.1" && host != "localhost" {
                                                                            Some(format!("{}:50052", host))
                                                                        } else {
                                                                            None
                                                                        }
                                                                    })
                                                                    .collect();
                                                                
                                                                let rpc_target = rpc_endpoints.join(",");
                                                                
                                                                let manifest = s.available_models[s.selected_model_idx].clone();
                                                                let model = ClusterEngineManager::get_model_path_for_manifest(&manifest);
                                                                let model_size_mb = manifest.size / (1024 * 1024);
                                                                let remote_worker = largest_fitting_worker(&s.remote_peers, model_size_mb);
                                                                let local_vram = s.local_vram_mb();
                                                                // Collect VRAM for each remote peer in the same order as rpc_target
                                                                let remote_vram_list: Vec<u64> = s.remote_peers.iter()
                                                                    .filter_map(|peer| {
                                                                        let host = peer.endpoint.split(':').next().unwrap_or("");
                                                                        if !host.is_empty() && host != "127.0.0.1" && host != "localhost" {
                                                                            Some(peer.effective_vram_mb())
                                                                        } else {
                                                                            None
                                                                        }
                                                                    })
                                                                    .collect();
                                                                let state_bg = Arc::clone(&state);
                                                                tokio_rt().spawn(async move {
                                                                    if let Some(endpoint) = remote_worker {
                                                                        {
                                                                            let mut s = state_bg.lock();
                                                                            s.engine_status = EngineStatus::Starting(
                                                                                format!("Connecting to worker {}...", endpoint),
                                                                            );
                                                                        }
                                                                        match start_remote_session(
                                                                            &endpoint,
                                                                            "vkremote-secret",
                                                                            &manifest,
                                                                        ).await {
                                                                            Ok(session) => {
                                                                                let host = endpoint.split(':').next().unwrap_or("127.0.0.1").to_string();
                                                                                let target_addr = format!("{}:18080", host);
                                                                                let (proxy_tx, mut proxy_rx) = tokio::sync::oneshot::channel::<()>();
                                                                                tokio_rt().spawn(async move {
                                                                                    let Ok(listener) = tokio::net::TcpListener::bind("127.0.0.1:8080").await else {
                                                                                        eprintln!("[gpu-mesh] Proxy bind on 127.0.0.1:8080 failed");
                                                                                        return;
                                                                                    };
                                                                                    loop {
                                                                                        tokio::select! {
                                                                                            _ = &mut proxy_rx => break,
                                                                                            res = listener.accept() => {
                                                                                                let Ok((mut client, _)) = res else { continue };
                                                                                                let target = target_addr.clone();
                                                                                                tokio::spawn(async move {
                                                                                                    if let Ok(mut server) = tokio::net::TcpStream::connect(&target).await {
                                                                                                        let _ = tokio::io::copy_bidirectional(&mut client, &mut server).await;
                                                                                                    }
                                                                                                });
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                });
                                                                                let mut s = state_bg.lock();
                                                                                s.web_proxy_stop_tx = Some(proxy_tx);
                                                                                s.remote_session = Some(Arc::new(tokio::sync::Mutex::new(session)));
                                                                                s.engine_status = EngineStatus::Running;
                                                                                s.status_message = format!(
                                                                                    "Cluster AI server ACTIVE on http://127.0.0.1:8080 (PC: {})",
                                                                                    endpoint
                                                                                );
                                                                                return;
                                                                            }
                                                                            Err(err) => {
                                                                                let mut s = state_bg.lock();
                                                                                s.event_log.push(format!(
                                                                                    "Remote worker unavailable, falling back to RPC: {}",
                                                                                    err
                                                                                ));
                                                                            }
                                                                        }
                                                                    }
                                                                    let res = ClusterEngineManager::spawn_server(&rpc_target, &model, local_vram, &remote_vram_list);
                                                                    match res {
                                                                        Ok(pid) => {
                                                                            {
                                                                                let mut s = state_bg.lock();
                                                                                s.engine_pid = Some(pid);
                                                                            }
                                                                            
                                                                            let mut server_ready = false;
                                                                            for attempt in 1..=360 {
                                                                                tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                                                                                
                                                                                if let Some(reason) = ClusterEngineManager::engine_exit_reason(pid) {
                                                                                    let mut s = state_bg.lock();
                                                                                    s.engine_pid = None;
                                                                                    s.engine_status = EngineStatus::Failed(reason.clone());
                                                                                    s.status_message = format!("AI server startup failed: {}", reason);
                                                                                    s.event_log.push(reason);
                                                                                    server_ready = true; // handled: skip the timeout message
                                                                                    break;
                                                                                }
                                                                                match ClusterEngineManager::check_health() {
                                                                                    Ok(h) if h == "ok" => {
                                                                                        let mut s = state_bg.lock();
                                                                                        s.engine_status = EngineStatus::Running;
                                                                                        s.status_message = "Cluster AI server ACTIVE on http://127.0.0.1:8080".to_string();
                                                                                        server_ready = true;
                                                                                        break;
                                                                                    }
                                                                                    Ok(h) if h == "loading" || h.contains("loading") => {
                                                                                        let mut s = state_bg.lock();
                                                                                        s.engine_status = EngineStatus::Starting(format!("Loading VRAM... {}s", attempt));
                                                                                        s.status_message = format!("Loading model into cluster VRAM across GPUs... ({}s)", attempt);
                                                                                    }
                                                                                    Ok(_) => {
                                                                                        let mut s = state_bg.lock();
                                                                                        s.engine_status = EngineStatus::Starting(format!("Server initializing... {}s", attempt));
                                                                                        s.status_message = format!("Server initializing... ({}s)", attempt);
                                                                                    }
                                                                                    Err(_) => {
                                                                                        let mut s = state_bg.lock();
                                                                                        s.engine_status = EngineStatus::Starting(format!("Allocating VRAM... {}s", attempt));
                                                                                        s.status_message = format!("Starting distributed server and allocating VRAM... ({}s)", attempt);
                                                                                    }
                                                                                }
                                                                            }
                                                                            
                                                                            if !server_ready {
                                                                                let mut s = state_bg.lock();
                                                                                s.engine_status = EngineStatus::Failed(
                                                                                    "Server did not become ready within 360 seconds. Check /tmp/llama-server.log for details.".to_string()
                                                                                );
                                                                                s.status_message = "AI server startup timeout - check logs at /tmp/llama-server.log".to_string();
                                                                            }
                                                                        }
                                                                        Err(err) => {
                                                                            let mut s = state_bg.lock();
                                                                            s.engine_status = EngineStatus::Failed(err.clone());
                                                                            s.status_message = format!("AI server startup failed: {}", err);
                                                                        }
                                                                    }
                                                                });
                                                            } else {
                                                                if let Some(tx) = s.web_proxy_stop_tx.take() {
                                                                    let _ = tx.send(());
                                                                }
                                                                if let Some(session_arc) = s.remote_session.take() {
                                                                    tokio_rt().spawn(async move {
                                                                        let mut session = session_arc.lock().await;
                                                                        let _ = session.stop().await;
                                                                    });
                                                                }
                                                                if let Some(pid) = s.engine_pid.take() {
                                                                    ClusterEngineManager::stop_server(pid);
                                                                }
                                                                s.engine_status = EngineStatus::Stopped;
                                                                s.status_message = "Cluster AI server stopped".to_string();
                                                            }
                                                            cx.notify();
                                                        })
                                                    })
                                                    .child(match &engine_status {
                                                        EngineStatus::Running => "⏹ Stop Server",
                                                        EngineStatus::Starting(_) => "⏳ Loading...",
                                                        _ => "▶ Start Server",
                                                    })
                                            )
                                            .child(
                                                div()
                                                    .id("btn-run-test")
                                                    .px_3()
                                                    .py_2()
                                                    .rounded_md()
                                                    .bg(if test_running { rgb(0x4b5563) } else { rgb(0x2563eb) })
                                                    .text_xs()
                                                    .font_weight(FontWeight::BOLD)
                                                    .text_color(rgb(0xffffff))
                                                    .cursor_pointer()
                                                    .on_mouse_down(MouseButton::Left, {
                                                        let state = Arc::clone(&state_arc);
                                                        cx.listener(move |_this, _, _, cx| {
                                                            let mut s = state.lock();
                                                            if s.test_running { return; }
                                                            s.test_running = true;
                                                            s.test_result = None;
                                                            s.test_speed = None;
                                                            s.status_message = "Executing test prompt across GPU cluster...".to_string();
                                                            let prompt = s.test_prompt.clone();
                                                            let state_bg = Arc::clone(&state);
                                                            tokio_rt().spawn(async move {
                                                                let remote_session = {
                                                                    let s = state_bg.lock();
                                                                    s.remote_session.clone()
                                                                };
                                                                if let Some(session) = remote_session {
                                                                    let outcome = {
                                                                        let mut session = session.lock().await;
                                                                        session.prompt(prompt.clone(), 48).await
                                                                    };
                                                                    let mut s = state_bg.lock();
                                                                    s.test_running = false;
                                                                    match outcome {
                                                                        Ok(body) => {
                                                                            let resp_str = String::from_utf8_lossy(&body);
                                                                            let (content, speed) = ClusterEngineManager::parse_completion_response(&resp_str);
                                                                            s.test_result = Some(content);
                                                                            s.test_speed = Some(speed.clone());
                                                                            s.status_message = format!("Remote inference complete! Speed: {}", speed);
                                                                        }
                                                                        Err(e) => {
                                                                            s.test_result = Some(format!("Error: {e}"));
                                                                            s.status_message = format!("Inference error: {e}");
                                                                        }
                                                                    }
                                                                    return;
                                                                }
                                                                let outcome = ClusterEngineManager::execute_prompt(&prompt);
                                                                let mut s = state_bg.lock();
                                                                s.test_running = false;
                                                                match outcome {
                                                                    Ok((content, speed)) => {
                                                                        s.test_result = Some(content);
                                                                        s.test_speed = Some(speed.clone());
                                                                        s.status_message = format!("Inference test complete! Speed: {}", speed);
                                                                    }
                                                                    Err(e) => {
                                                                        s.test_result = Some(format!("Error (is server started?): {}", e));
                                                                        s.status_message = format!("Inference error: {}", e);
                                                                    }
                                                                }
                                                            });
                                                            cx.notify();
                                                        })
                                                    })
                                                    .child(if test_running { "Running Test..." } else { "⚡ Test Cluster" })
                                            )
                                            .child(
                                                div()
                                                    .id("btn-open-web")
                                                    .px_3()
                                                    .py_2()
                                                    .rounded_md()
                                                    .bg(rgb(0x27272a))
                                                    .text_xs()
                                                    .font_weight(FontWeight::SEMIBOLD)
                                                    .text_color(rgb(0xe4e4e7))
                                                    .cursor_pointer()
                                                    .on_mouse_down(MouseButton::Left, {
                                                        let _state = Arc::clone(&state_arc);
                                                        cx.listener(move |_this, _, _, cx| {
                                                            ClusterEngineManager::open_browser();
                                                            cx.notify();
                                                        })
                                                    })
                                                    .child("🌐 Open Web UI")
                                            )
                                    )
                                    .child(
                                        div()
                                            .p_3()
                                            .rounded_md()
                                            .bg(rgb(0x111113))
                                            .border_1()
                                            .border_color(rgb(0x27272a))
                                            .flex()
                                            .flex_col()
                                            .gap_2()
                                            .child(
                                                div()
                                                    .flex()
                                                    .flex_row()
                                                    .justify_between()
                                                    .items_center()
                                                    .child(
                                                        div()
                                                            .text_xs()
                                                            .font_weight(FontWeight::MEDIUM)
                                                            .text_color(rgb(0xa1a1aa))
                                                            .child(format!("Prompt: \"{}\"", test_prompt))
                                                    )
                                                    .child(
                                                        if let Some(speed) = &test_speed {
                                                            div()
                                                                .px_2()
                                                                .py_0p5()
                                                                .rounded_sm()
                                                                .bg(rgb(0x064e3b))
                                                                .text_xs()
                                                                .text_color(rgb(0x34d399))
                                                                .child(speed.clone())
                                                        } else {
                                                            div()
                                                        }
                                                    )
                                            )
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .text_color(if test_running { rgb(0xfbbf24) } else if test_result.is_some() { rgb(0xffffff) } else { rgb(0x71717a) })
                                                    .child(
                                                        if test_running {
                                                            "Executing inference across GPU cluster...".to_string()
                                                        } else if let Some(res) = &test_result {
                                                            res.clone()
                                                        } else {
                                                            "Click '⚡ Test Cluster' to run inference and measure generation speed.".to_string()
                                                        }
                                                    )
                                            )
                                    )
                            )
                    )
                    // ===== RIGHT COLUMN: REMOTE MESH & CONNECT PEERS =====
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap_4()
                            // Connect Peer Card
                            .child(
                                div()
                                    .p_5()
                                    .rounded_lg()
                                    .bg(rgb(0x18181b))
                                    .border_1()
                                    .border_color(rgb(0x27272a))
                                    .flex()
                                    .flex_col()
                                    .gap_3()
                                    .child(
                                        div()
                                            .flex()
                                            .flex_row()
                                            .justify_between()
                                            .items_center()
                                            .child(
                                                div()
                                                    .text_base()
                                                    .font_weight(FontWeight::BOLD)
                                                    .text_color(rgb(0xffffff))
                                                    .child("Connect Peer")
                                            )
                                            // Two Option Tab Switcher: [ Auto-Fetch ] / [ Manual ]
                                            .child(
                                                div()
                                                    .flex()
                                                    .flex_row()
                                                    .gap_1()
                                                    .bg(rgb(0x111113))
                                                    .p(px(2.0))
                                                    .rounded_md()
                                                    .border_1()
                                                    .border_color(rgb(0x27272a))
                                                    .child(
                                                        div()
                                                            .id("tab-auto-fetch")
                                                            .px_2p5()
                                                            .py_1()
                                                            .rounded_sm()
                                                            .bg(if connect_mode == ConnectMode::AutoFetch { rgb(0x1e3a5f) } else { rgb(0x111113) })
                                                            .border_1()
                                                            .border_color(if connect_mode == ConnectMode::AutoFetch { rgb(0x3b82f6) } else { rgb(0x111113) })
                                                            .text_xs()
                                                            .font_weight(if connect_mode == ConnectMode::AutoFetch { FontWeight::BOLD } else { FontWeight::NORMAL })
                                                            .text_color(if connect_mode == ConnectMode::AutoFetch { rgb(0x93c5fd) } else { rgb(0x71717a) })
                                                            .cursor_pointer()
                                                            .on_mouse_down(MouseButton::Left, {
                                                                let state = Arc::clone(&state_arc);
                                                                cx.listener(move |_this, _, _, cx| {
                                                                    let mut s = state.lock();
                                                                    s.connect_mode = ConnectMode::AutoFetch;
                                                                    cx.notify();
                                                                })
                                                            })
                                                            .child("Auto-Fetch")
                                                    )
                                                    .child(
                                                        div()
                                                            .id("tab-manual-add")
                                                            .px_2p5()
                                                            .py_1()
                                                            .rounded_sm()
                                                            .bg(if connect_mode == ConnectMode::Manual { rgb(0x1e3a5f) } else { rgb(0x111113) })
                                                            .border_1()
                                                            .border_color(if connect_mode == ConnectMode::Manual { rgb(0x3b82f6) } else { rgb(0x111113) })
                                                            .text_xs()
                                                            .font_weight(if connect_mode == ConnectMode::Manual { FontWeight::BOLD } else { FontWeight::NORMAL })
                                                            .text_color(if connect_mode == ConnectMode::Manual { rgb(0x93c5fd) } else { rgb(0x71717a) })
                                                            .cursor_pointer()
                                                            .on_mouse_down(MouseButton::Left, {
                                                                let state = Arc::clone(&state_arc);
                                                                cx.listener(move |_this, _, _, cx| {
                                                                    let mut s = state.lock();
                                                                    s.connect_mode = ConnectMode::Manual;
                                                                    cx.notify();
                                                                })
                                                            })
                                                            .child("Manual")
                                                    )
                                            )
                                    )
                                    // Mode Container
                                    .child(
                                        match connect_mode {
                                             ConnectMode::AutoFetch => {
                                                div()
                                                    .p_3()
                                                    .rounded_md()
                                                    .bg(rgb(0x111113))
                                                    .border_1()
                                                    .border_color(rgb(0x27272a))
                                                    .flex()
                                                    .flex_col()
                                                    .gap_2()
                                                    .child(
                                                        div()
                                                            .flex()
                                                            .flex_row()
                                                            .justify_between()
                                                            .items_center()
                                                            .child(
                                                                div()
                                                                    .text_xs()
                                                                    .text_color(rgb(0xa1a1aa))
                                                                    .child(format!("Subnet: {}.0/24", {
                                                                        let parts: Vec<&str> = local_ip.split('.').collect();
                                                                        if parts.len() == 4 {
                                                                            format!("{}.{}.{}", parts[0], parts[1], parts[2])
                                                                        } else {
                                                                            "LAN".to_string()
                                                                        }
                                                                    }))
                                                            )
                                                            .child(
                                                                div()
                                                                    .id("btn-scan-lan-action")
                                                                    .px_3()
                                                                    .py_1p5()
                                                                    .rounded_md()
                                                                    .bg(if is_scanning { rgb(0x3f3f46) } else { rgb(0x0284c7) })
                                                                    .text_xs()
                                                                    .font_weight(FontWeight::BOLD)
                                                                    .text_color(rgb(0xffffff))
                                                                    .cursor_pointer()
                                                                    .on_mouse_down(MouseButton::Left, {
                                                                        let state = Arc::clone(&state_arc);
                                                                        cx.listener(move |_this, _, _, cx| {
                                                                            let (ip, port) = {
                                                                                let mut s = state.lock();
                                                                                if s.is_scanning { return; }
                                                                                s.is_scanning = true;
                                                                                // Re-fetch interface IP dynamically in case Wi-Fi changed!
                                                                                s.local_ip = get_local_ip();
                                                                                s.status_message = format!("Scanning subnet {}/24 on :{}...", s.local_ip, s.local_port);
                                                                                (s.local_ip.clone(), s.local_port)
                                                                            };
                                                                            let state_bg = Arc::clone(&state);
                                                                            tokio_rt().spawn(async move {
                                                                                let found = scan_lan_for_givers(&ip, port).await;
                                                                                let mut s = state_bg.lock();
                                                                                s.is_scanning = false;
                                                                                if !found.is_empty() {
                                                                                    s.target_endpoint = found[0].clone();
                                                                                    for f in &found {
                                                                                        if !s.candidate_endpoints.contains(f) {
                                                                                            s.candidate_endpoints.insert(0, f.clone());
                                                                                        }
                                                                                    }
                                                                                    s.status_message = format!("Auto-Fetch found {} active giver(s) on LAN!", found.len());
                                                                                } else {
                                                                                    s.status_message = format!("Auto-Fetch complete: No other givers found on {}/24. Try 'Manually Add'.", ip);
                                                                                }
                                                                            });
                                                                            cx.notify();
                                                                        })
                                                                    })
                                                                    .child(if is_scanning { "Scanning..." } else { "🔍 Scan LAN" })
                                                            )
                                                    )
                                            }
                                            ConnectMode::Manual => {
                                                div()
                                                    .p_3()
                                                    .rounded_md()
                                                    .bg(rgb(0x111113))
                                                    .border_1()
                                                    .border_color(rgb(0x27272a))
                                                    .flex()
                                                    .flex_col()
                                                    .gap_2()
                                                    // Input text box
                                                    .child(
                                                        div()
                                                            .id("manual-input-box")
                                                            .p_2()
                                                            .rounded_md()
                                                            .bg(rgb(0x18181b))
                                                            .border_1()
                                                            .border_color(rgb(0x3f3f46))
                                                            .flex()
                                                            .flex_row()
                                                            .items_center()
                                                            .justify_between()
                                                            .track_focus(&self.focus_handle)
                                                            .on_mouse_down(MouseButton::Left, {
                                                                let focus = self.focus_handle.clone();
                                                                cx.listener(move |_this, _, window, cx| {
                                                                    window.focus(&focus);
                                                                    cx.notify();
                                                                })
                                                            })
                                                            .on_key_down(cx.listener({
                                                                let state = Arc::clone(&state_arc);
                                                                move |_this, event: &KeyDownEvent, _window, cx| {
                                                                    let key = &event.keystroke.key;
                                                                    let mut s = state.lock();
                                                                    if key == "backspace" {
                                                                        s.manual_ip_input.pop();
                                                                        cx.notify();
                                                                    } else if key == "enter" {
                                                                        let raw = s.manual_ip_input.trim().to_string();
                                                                        if !raw.is_empty() {
                                                                            let ep = if raw.contains(':') { raw } else { format!("{}:50051", raw) };
                                                                            s.target_endpoint = ep.clone();
                                                                            if !s.candidate_endpoints.contains(&ep) {
                                                                                s.candidate_endpoints.insert(0, ep);
                                                                            }
                                                                            s.status_message = format!("Target set to {}", s.target_endpoint);
                                                                        }
                                                                        cx.notify();
                                                                    } else if key.len() == 1 {
                                                                        let c = key.chars().next().unwrap();
                                                                        if c.is_ascii_digit() || c == '.' || c == ':' {
                                                                            s.manual_ip_input.push(c);
                                                                            cx.notify();
                                                                        }
                                                                    }
                                                                }
                                                            }))
                                                            .child(
                                                                div()
                                                                    .text_xs()
                                                                    .font_weight(FontWeight::MEDIUM)
                                                                    .child(
                                                                        if manual_ip_input.is_empty() {
                                                                            div().text_color(rgb(0x71717a)).child("Click to type IP (e.g. 10.158.255.229:50051)...")
                                                                        } else {
                                                                            div()
                                                                                .flex()
                                                                                .flex_row()
                                                                                .items_center()
                                                                                .text_color(rgb(0xffffff))
                                                                                .child(manual_ip_input.clone())
                                                                                .child(div().text_color(rgb(0x38bdf8)).child(" ▏"))
                                                                        }
                                                                    )
                                                            )
                                                            .child(
                                                                div()
                                                                    .flex()
                                                                    .flex_row()
                                                                    .gap_1()
                                                                    // Paste Button
                                                                    .child(
                                                                        div()
                                                                            .id("btn-paste-clip")
                                                                            .px_2()
                                                                            .py_0p5()
                                                                            .rounded_sm()
                                                                            .bg(rgb(0x27272a))
                                                                            .text_xs()
                                                                            .text_color(rgb(0x38bdf8))
                                                                            .cursor_pointer()
                                                                            .on_mouse_down(MouseButton::Left, {
                                                                                let state = Arc::clone(&state_arc);
                                                                                cx.listener(move |_this, _, _window, cx| {
                                                                                    if let Some(item) = cx.read_from_clipboard() {
                                                                                        if let Some(t) = item.text() {
                                                                                            let clean: String = t.trim().chars().filter(|c| c.is_ascii_digit() || *c == '.' || *c == ':').collect();
                                                                                            if !clean.is_empty() {
                                                                                                let mut s = state.lock();
                                                                                                s.manual_ip_input = clean;
                                                                                                cx.notify();
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                })
                                                                            })
                                                                            .child("📋 Paste")
                                                                    )
                                                                    // Add & Set Target Button
                                                                    .child(
                                                                        div()
                                                                            .id("btn-commit-manual")
                                                                            .px_2()
                                                                            .py_0p5()
                                                                            .rounded_sm()
                                                                            .bg(rgb(0x2563eb))
                                                                            .text_xs()
                                                                            .font_weight(FontWeight::BOLD)
                                                                            .text_color(rgb(0xffffff))
                                                                            .cursor_pointer()
                                                                            .on_mouse_down(MouseButton::Left, {
                                                                                let state = Arc::clone(&state_arc);
                                                                                cx.listener(move |_this, _, _, cx| {
                                                                                    let mut s = state.lock();
                                                                                    let raw = s.manual_ip_input.trim().to_string();
                                                                                    if !raw.is_empty() {
                                                                                        let ep = if raw.contains(':') { raw } else { format!("{}:50051", raw) };
                                                                                        s.target_endpoint = ep.clone();
                                                                                        if !s.candidate_endpoints.contains(&ep) {
                                                                                            s.candidate_endpoints.insert(0, ep);
                                                                                        }
                                                                                        s.status_message = format!("Target set to {}", s.target_endpoint);
                                                                                        cx.notify();
                                                                                    }
                                                                                })
                                                                            })
                                                                            .child("➕ Add")
                                                                    )
                                                                    // Clear Button
                                                                    .child(
                                                                        div()
                                                                            .id("btn-clear-manual")
                                                                            .px_2()
                                                                            .py_0p5()
                                                                            .rounded_sm()
                                                                            .bg(rgb(0x27272a))
                                                                            .text_xs()
                                                                            .text_color(rgb(0x71717a))
                                                                            .cursor_pointer()
                                                                            .on_mouse_down(MouseButton::Left, {
                                                                                let state = Arc::clone(&state_arc);
                                                                                cx.listener(move |_this, _, _, cx| {
                                                                                    let mut s = state.lock();
                                                                                    s.manual_ip_input.clear();
                                                                                    cx.notify();
                                                                                })
                                                                            })
                                                                            .child("⌫")
                                                                    )
                                                            )
                                                    )
                                            }
                                        }
                                    )
                                    // Target Endpoint Selector / Presets
                                    .child(
                                        div()
                                            .flex()
                                            .flex_col()
                                            .gap_2()
                                            .child(
                                                div().text_xs().text_color(rgb(0xa1a1aa)).child("TARGET PEER ENDPOINT:")
                                            )
                                            .child(
                                                div()
                                                    .flex()
                                                    .flex_row()
                                                    .gap_2()
                                                    .children(
                                                        candidate_endpoints.iter().enumerate().map(|(idx, ep)| {
                                                            let is_selected = ep == &target_endpoint;
                                                            let ep_clone = ep.clone();
                                                            let state = Arc::clone(&state_arc);
                                                            div()
                                                                .id(("select-ep", idx))
                                                                .px_3()
                                                                .py_1p5()
                                                                .rounded_md()
                                                                .bg(if is_selected { rgb(0x1e3a5f) } else { rgb(0x27272a) })
                                                                .border_1()
                                                                .border_color(if is_selected { rgb(0x3b82f6) } else { rgb(0x3f3f46) })
                                                                .text_xs()
                                                                .font_weight(if is_selected { FontWeight::BOLD } else { FontWeight::NORMAL })
                                                                .text_color(if is_selected { rgb(0x93c5fd) } else { rgb(0xd4d4d8) })
                                                                .cursor_pointer()
                                                                .on_mouse_down(MouseButton::Left, cx.listener(move |_this, _, _, cx| {
                                                                    let mut s = state.lock();
                                                                    s.target_endpoint = ep_clone.clone();
                                                                    cx.notify();
                                                                }))
                                                                .child(ep.clone())
                                                        })
                                                    )
                                            )
                                    )
                                    // Primary Connect Button
                                    .child(
                                        div()
                                            .id("btn-connect-peer")
                                            .px_4()
                                            .py_2p5()
                                            .rounded_md()
                                            .bg(if is_connecting { rgb(0x4b5563) } else { rgb(0x2563eb) })
                                            .text_xs()
                                            .font_weight(FontWeight::BOLD)
                                            .text_color(rgb(0xffffff))
                                            .cursor_pointer()
                                            .on_mouse_down(MouseButton::Left, {
                                                let state = Arc::clone(&state_arc);
                                                cx.listener(move |_this, _, _, cx| {
                                                    let target = {
                                                        let mut s = state.lock();
                                                        if s.is_connecting { return; }
                                                        s.is_connecting = true;
                                                        s.status_message = format!("Connecting to remote giver at {}...", s.target_endpoint);
                                                        s.target_endpoint.clone()
                                                    };
                                                    let state_bg = Arc::clone(&state);
                                                    let state_bg2 = Arc::clone(&state);
                                                    tokio_rt().spawn(async move {
                                                        let res = probe_remote_giver(&target, "vkremote-secret").await;
                                                        let mut s = state_bg.lock();
                                                        s.is_connecting = false;
                                                        match res {
                                                            Ok(giver) => {
                                                                let dev_name = giver.gpus.first().map(|g| g.name.clone()).unwrap_or_else(|| "Unknown GPU".to_string());
                                                                let vram_gb = giver.total_vram_mb as f64 / 1024.0;
                                                                let lat = giver.latency_ms;
                                                                s.remote_peers.retain(|p| p.endpoint != target);
                                                                s.remote_peers.push(giver);
                                                                s.status_message = format!("CONNECTED to {}! Discovered: {} ({:.1} GB, {:.2}ms)", target, dev_name, vram_gb, lat);
                                                                s.event_log.push(format!("Connected to {} ({:.1} GB, {:.2}ms)", target, vram_gb, lat));

                                                                // If the AI server is already running local-only, restart it
                                                                // now with the new peer included in the RPC split.
                                                                let should_restart = s.engine_status == EngineStatus::Running;
                                                                if should_restart {
                                                                    // Stop current server
                                                                    if let Some(pid) = s.engine_pid.take() {
                                                                        ClusterEngineManager::stop_server(pid);
                                                                    }
                                                                    s.engine_status = EngineStatus::Starting("Restarting with remote GPU...".to_string());
                                                                    s.status_message = format!("Restarting AI server with {} ({:.1} GB) in cluster...", dev_name, vram_gb);

                                                                    let rpc_endpoints: Vec<String> = s.remote_peers.iter()
                                                                        .filter_map(|peer| {
                                                                            let host = peer.endpoint.split(':').next().unwrap_or("");
                                                                            if !host.is_empty() && host != "127.0.0.1" && host != "localhost" {
                                                                                Some(format!("{}:50052", host))
                                                                            } else {
                                                                                None
                                                                            }
                                                                        })
                                                                        .collect();
                                                                    let rpc_target = rpc_endpoints.join(",");
                                                                    let model = ClusterEngineManager::get_default_model_path();
                                                                    let local_vram = s.local_vram_mb();
                                                                    let remote_vram_list: Vec<u64> = s.remote_peers.iter()
                                                                        .filter_map(|peer| {
                                                                            let host = peer.endpoint.split(':').next().unwrap_or("");
                                                                            if !host.is_empty() && host != "127.0.0.1" && host != "localhost" {
                                                                                Some(peer.total_vram_mb)
                                                                            } else {
                                                                                None
                                                                            }
                                                                        })
                                                                        .collect();

                                                                    let state_restart = Arc::clone(&state_bg2);
                                                                    tokio_rt().spawn(async move {
                                                                        let res = ClusterEngineManager::spawn_server(&rpc_target, &model, local_vram, &remote_vram_list);
                                                                        match res {
                                                                            Ok(pid) => {
                                                                                {
                                                                                    let mut s = state_restart.lock();
                                                                                    s.engine_pid = Some(pid);
                                                                                }
                                                                                for _ in 1..=120 {
                                                                                    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
                                                                                    if let Some(reason) = ClusterEngineManager::engine_exit_reason(pid) {
                                                                                        let mut s = state_restart.lock();
                                                                                        s.engine_pid = None;
                                                                                        s.engine_status = EngineStatus::Failed(reason.clone());
                                                                                        s.status_message = format!("Restart failed: {}", reason);
                                                                                        break;
                                                                                    }
                                                                                    if let Ok(h) = ClusterEngineManager::check_health() {
                                                                                        if h == "ok" {
                                                                                            let mut s = state_restart.lock();
                                                                                            s.engine_status = EngineStatus::Running;
                                                                                            s.status_message = "Cluster AI server ACTIVE (all GPUs) on http://127.0.0.1:8080".to_string();
                                                                                            break;
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                            Err(e) => {
                                                                                let mut s = state_restart.lock();
                                                                                s.engine_status = EngineStatus::Failed(e.clone());
                                                                                s.status_message = format!("Restart failed: {}", e);
                                                                            }
                                                                        }
                                                                    });
                                                                }
                                                            }
                                                            Err(e) => {
                                                                s.status_message = format!("Connection failed to {}: {}", target, e);
                                                                s.event_log.push(format!("Failed to connect {}: {}", target, e));
                                                            }
                                                        }
                                                    });
                                                    cx.notify();
                                                })
                                            })
                                            .child(if is_connecting {
                                                format!("Connecting to {}...", target_endpoint)
                                            } else {
                                                format!("⚡ Connect to {}", target_endpoint)
                                            })
                                    )
                            )
                            // Active Remote Connected Peers Card
                            .child(
                                div()
                                    .p_5()
                                    .rounded_lg()
                                    .bg(rgb(0x18181b))
                                    .border_1()
                                    .border_color(rgb(0x27272a))
                                    .flex()
                                    .flex_col()
                                    .gap_3()
                                    .child(
                                        div()
                                            .flex()
                                            .flex_row()
                                            .justify_between()
                                            .items_center()
                                            .child(
                                                div()
                                                    .text_sm()
                                                    .font_weight(FontWeight::BOLD)
                                                    .text_color(rgb(0xffffff))
                                                    .child(format!("Active Remote Peers ({})", remote_peers.len()))
                                            )
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .text_color(if remote_peers.is_empty() { rgb(0x71717a) } else { rgb(0xa78bfa) })
                                                    .child(format!("+{:.2} GB VRAM Incoming", remote_gb))
                                            )
                                    )
                                    .child(
                                        if remote_peers.is_empty() {
                                            div()
                                                .p_6()
                                                .rounded_md()
                                                .border_1()
                                                .border_color(rgb(0x27272a))
                                                .bg(rgb(0x111113))
                                                .flex()
                                                .flex_col()
                                                .items_center()
                                                .justify_center()
                                                .gap_2()
                                                .child(div().text_sm().font_weight(FontWeight::MEDIUM).text_color(rgb(0x71717a)).child("No remote peers connected"))
                                                .child(div().text_xs().text_color(rgb(0x52525b)).child("Start sharing on another machine and connect to expand VRAM pool."))
                                        } else {
                                            div()
                                                .flex()
                                                .flex_col()
                                                .gap_2()
                                                .children(
                                                    remote_peers.iter().enumerate().map(|(idx, peer)| {
                                                        div()
                                                            .p_3()
                                                            .rounded_md()
                                                            .bg(rgb(0x111113))
                                                            .border_1()
                                                            .border_color(rgb(0x27272a))
                                                            .flex()
                                                            .flex_row()
                                                            .justify_between()
                                                            .items_center()
                                                            .child(
                                                                div()
                                                                    .flex()
                                                                    .flex_col()
                                                                    .gap_1()
                                                                    .child(
                                                                        div()
                                                                            .flex()
                                                                            .flex_row()
                                                                            .items_center()
                                                                            .gap_2()
                                                                            .child(div().w(px(8.0)).h(px(8.0)).rounded_full().bg(rgb(0x10b981)))
                                                                            .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).text_color(rgb(0xffffff)).child(peer.endpoint.clone()))
                                                                    )
                                                                    .child(
                                                                        div()
                                                                            .text_xs()
                                                                            .text_color(rgb(0x71717a))
                                                                            .child(format!("{} • {:.2}ms ping", peer.gpus.first().map(|g| g.name.as_str()).unwrap_or("Vulkan Device"), peer.latency_ms))
                                                                    )
                                                            )
                                                            .child(
                                                                div()
                                                                    .flex()
                                                                    .flex_row()
                                                                    .items_center()
                                                                    .gap_3()
                                                                    .child({
                                                                        let effective_gb = peer.effective_vram_mb() as f64 / 1024.0;
                                                                        let total_gb = peer.total_vram_mb as f64 / 1024.0;
                                                                        let ep_alloc = peer.endpoint.clone();
                                                                        div()
                                                                            .flex()
                                                                            .flex_row()
                                                                            .items_center()
                                                                            .gap_1p5()
                                                                            .bg(rgb(0x18181b))
                                                                            .border_1()
                                                                            .border_color(rgb(0x27272a))
                                                                            .rounded_md()
                                                                            .px_2()
                                                                            .py_0p5()
                                                                            .child(
                                                                                div()
                                                                                    .id(("vram-dec", idx))
                                                                                    .px_1p5()
                                                                                    .py_0p5()
                                                                                    .rounded_sm()
                                                                                    .cursor_pointer()
                                                                                    .bg(rgb(0x27272a))
                                                                                    .text_xs()
                                                                                    .font_weight(FontWeight::BOLD)
                                                                                    .text_color(rgb(0xd4d4d8))
                                                                                    .on_mouse_down(MouseButton::Left, {
                                                                                        let ep = ep_alloc.clone();
                                                                                        let state = Arc::clone(&state_arc);
                                                                                        cx.listener(move |_this, _, _, cx| {
                                                                                            let mut s = state.lock();
                                                                                            if let Some(p) = s.remote_peers.iter_mut().find(|p| p.endpoint == ep) {
                                                                                                let current = p.effective_vram_mb();
                                                                                                let next = current.saturating_sub(1024).max(1024);
                                                                                                p.allocated_vram_mb = Some(next);
                                                                                                s.status_message = format!("Allocated {:.1} GB from peer {}", next as f64 / 1024.0, ep);
                                                                                            }
                                                                                            cx.notify();
                                                                                        })
                                                                                    })
                                                                                    .child("-")
                                                                            )
                                                                            .child(
                                                                                div()
                                                                                    .flex()
                                                                                    .flex_row()
                                                                                    .items_baseline()
                                                                                    .gap_1()
                                                                                    .child(
                                                                                        div()
                                                                                            .text_xs()
                                                                                            .font_weight(FontWeight::BOLD)
                                                                                            .text_color(rgb(0xa78bfa))
                                                                                            .child(format!("{:.1} GB", effective_gb))
                                                                                    )
                                                                                    .child(
                                                                                        div()
                                                                                            .text_xs()
                                                                                            .text_color(rgb(0x71717a))
                                                                                            .child(format!("/ {:.0}G", total_gb))
                                                                                    )
                                                                            )
                                                                            .child(
                                                                                div()
                                                                                    .id(("vram-inc", idx))
                                                                                    .px_1p5()
                                                                                    .py_0p5()
                                                                                    .rounded_sm()
                                                                                    .cursor_pointer()
                                                                                    .bg(rgb(0x27272a))
                                                                                    .text_xs()
                                                                                    .font_weight(FontWeight::BOLD)
                                                                                    .text_color(rgb(0xd4d4d8))
                                                                                    .on_mouse_down(MouseButton::Left, {
                                                                                        let ep = ep_alloc.clone();
                                                                                        let state = Arc::clone(&state_arc);
                                                                                        cx.listener(move |_this, _, _, cx| {
                                                                                            let mut s = state.lock();
                                                                                            if let Some(p) = s.remote_peers.iter_mut().find(|p| p.endpoint == ep) {
                                                                                                let current = p.effective_vram_mb();
                                                                                                let next = (current + 1024).min(p.total_vram_mb);
                                                                                                p.allocated_vram_mb = Some(next);
                                                                                                s.status_message = format!("Allocated {:.1} GB from peer {}", next as f64 / 1024.0, ep);
                                                                                            }
                                                                                            cx.notify();
                                                                                        })
                                                                                    })
                                                                                    .child("+")
                                                                            )
                                                                    })
                                                                    .child(
                                                                        div()
                                                                            .id(("btn-disconnect", idx))
                                                                            .px_2()
                                                                            .py_1()
                                                                            .rounded_sm()
                                                                            .bg(rgb(0x27272a))
                                                                            .text_xs()
                                                                            .text_color(rgb(0xf87171))
                                                                            .cursor_pointer()
                                                                            .on_mouse_down(MouseButton::Left, {
                                                                                let ep_disconnect = peer.endpoint.clone();
                                                                                let state = Arc::clone(&state_arc);
                                                                                cx.listener(move |_this, _, _, cx| {
                                                                                    let mut s = state.lock();
                                                                                    s.remote_peers.retain(|p| p.endpoint != ep_disconnect);
                                                                                    s.status_message = format!("Disconnected peer {}", ep_disconnect);
                                                                                    cx.notify();
                                                                                })
                                                                            })
                                                                            .child("Disconnect")
                                                                    )
                                                            )
                                                    })
                                                )
                                        }
                                    )
                            )
                    )
                )
            )
            // ------------------ FOOTER STATUS ------------------
            .child(
                div()
                    .px_6()
                    .py_2()
                    .border_t_1()
                    .border_color(rgb(0x27272a))
                    .bg(rgb(0x111114))
                    .flex()
                    .flex_row()
                    .justify_between()
                    .items_center()
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0xa1a1aa))
                            .child(status_message)
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(rgb(0x71717a))
                            .child("vkremote mesh v0.2.0 • Real Telemetry")
                    )
            )
    }
}

fn main() {
    // Purge any residual Vulkan loader or driver overrides from terminal environment
    std::env::remove_var("VK_LOADER_DRIVERS_SELECT");
    std::env::remove_var("VK_LOADER_LAYERS_ENABLE");
    std::env::remove_var("VK_ADD_DRIVER_FILES");
    std::env::remove_var("VK_DRIVER_FILES");
    std::env::remove_var("VK_ICD_FILENAMES");
    std::env::remove_var("DXVK_FILTER_DEVICE_NAME");
    std::env::remove_var("GALLIUM_DRIVER");
    std::env::remove_var("MESA_LOADER_DRIVER_OVERRIDE");
    std::env::remove_var("MESA_VK_DEVICE_SELECT");

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    Application::new().run(|cx: &mut App| {
        let options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds {
                origin: Point::default(),
                size: Size {
                    width: px(1240.0),
                    height: px(880.0),
                },
            })),
            titlebar: Some(TitlebarOptions {
                title: Some("rgpu // GPU Mesh Monitor".into()),
                appears_transparent: false,
                traffic_light_position: None,
            }),
            ..Default::default()
        };

        cx.open_window(options, |_window, cx| cx.new(|cx| GpuMeshApp::new(cx)))
            .unwrap();
    });
}

#[cfg(test)]
mod test_vk {
    #[test]
    fn test_vk_engine_on_pc() {
        let eng = vkremote_server::engine::VulkanEngine::new(None);
        match eng {
            Ok(ref e) => println!(
                "VULKAN ENGINE SUCCEEDED: {}",
                e.snapshot.properties.device_name
            ),
            Err(ref e) => println!("VULKAN ENGINE FAILED: {:#}", e),
        }
        assert!(eng.is_ok());
    }

    #[test]
    fn test_wire_probe_pc() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let target = "10.158.255.229:50051";
        let res = rt.block_on(crate::network::probe_remote_giver(
            target,
            "vkremote-secret",
        ));
        println!("PROBE TO PC RESULT: {:?}", res);
        assert!(res.is_ok());
        let peer = res.unwrap();
        println!("PEER NAME: {}, GPUS: {:?}", peer.name, peer.gpus);
        assert!(!peer.gpus.is_empty());
        assert_eq!(peer.gpus[0].name, "NVIDIA GeForce RTX 3060");
        assert_eq!(peer.gpus[0].vram_mb, 12115);
    }
}
