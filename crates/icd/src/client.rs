use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;
use vkremote_wire::framing::{
    client_handshake_sync, read_msg_sync_with_len, write_msg_sync, DEFAULT_MAX_FRAME_SIZE,
};
use vkremote_wire::sync::MemorySyncTracker;
use vkremote_wire::types::{MemoryChunkDiff, PhysicalDeviceSnapshot, Request, Response};
use vkremote_wire::ProtocolError;

// ---------------------------------------------------------------------------
// Per-token telemetry
// ---------------------------------------------------------------------------

/// Accumulated stats for one decode token (reset after each print).
#[derive(Default)]
struct TokenStats {
    round_trips: u64,
    bytes_sent: u64,
    bytes_recv: u64,
    serialize_us: u64, // µs spent in postcard serialisation
    blocked_us: u64,   // µs waiting for network response
}

/// Global telemetry state shared across all calls.
struct Telemetry {
    current: Mutex<TokenStats>,
    token_count: AtomicU64,
    // lifetime totals for the summary line
    total_round_trips: AtomicU64,
    total_bytes_sent: AtomicU64,
    total_bytes_recv: AtomicU64,
}

impl Telemetry {
    fn new() -> Self {
        Self {
            current: Mutex::new(TokenStats::default()),
            token_count: AtomicU64::new(0),
            total_round_trips: AtomicU64::new(0),
            total_bytes_sent: AtomicU64::new(0),
            total_bytes_recv: AtomicU64::new(0),
        }
    }

    /// Record one round trip. Call *after* the response arrives.
    fn record_rtt(&self, bytes_sent: u64, bytes_recv: u64, serialize_us: u64, blocked_us: u64) {
        let mut s = self.current.lock().unwrap();
        s.round_trips += 1;
        s.bytes_sent += bytes_sent;
        s.bytes_recv += bytes_recv;
        s.serialize_us += serialize_us;
        s.blocked_us += blocked_us;
        self.total_round_trips.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_sent
            .fetch_add(bytes_sent, Ordering::Relaxed);
        self.total_bytes_recv
            .fetch_add(bytes_recv, Ordering::Relaxed);
    }

    /// Call at the end of each token (after WaitForFences / QueueWaitIdle returns).
    /// Prints a summary line every 32 tokens.
    fn end_token(&self) {
        let n = self.token_count.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 32 == 0 {
            let s = self.current.lock().unwrap();
            eprintln!(
                "[vkremote-telemetry] tok={} | rtt={} | sent={:.1}KB recv={:.1}KB | \
                 ser={:.2}ms blocked={:.2}ms | \
                 total_rtt={} total_sent={:.1}MB total_recv={:.1}MB",
                n,
                s.round_trips,
                s.bytes_sent as f64 / 1024.0,
                s.bytes_recv as f64 / 1024.0,
                s.serialize_us as f64 / 1000.0,
                s.blocked_us as f64 / 1000.0,
                self.total_round_trips.load(Ordering::Relaxed),
                self.total_bytes_sent.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
                self.total_bytes_recv.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0),
            );
        }
        // Reset per-token counters
        *self.current.lock().unwrap() = TokenStats::default();
    }
}

static TELEMETRY: std::sync::OnceLock<Telemetry> = std::sync::OnceLock::new();
fn telemetry() -> &'static Telemetry {
    TELEMETRY.get_or_init(Telemetry::new)
}

/// Call once per decoded token (at WaitForFences / QueueWaitIdle boundary).
pub fn end_token() {
    telemetry().end_token();
}

pub trait Transport: Send + Sync {
    fn roundtrip(&self, req: &Request) -> Result<Response, ProtocolError>;
}

pub struct TcpTransport {
    server_addr: String,
    token: String,
    stream: Mutex<Option<std::net::TcpStream>>,
}

impl TcpTransport {
    pub fn new(server_addr: String, token: String) -> Self {
        Self {
            server_addr,
            token,
            stream: Mutex::new(None),
        }
    }

    fn ensure_stream(&self) -> Result<(), ProtocolError> {
        let mut guard = self.stream.lock().unwrap();
        if guard.is_none() {
            let addr = self.server_addr.clone();
            let token = self.token.clone();
            eprintln!(
                "[vkremote-icd] Connecting to remote server at {}...",
                addr
            );
            let mut s = std::net::TcpStream::connect(&addr).map_err(|e| {
                eprintln!("[vkremote-icd] Failed to connect to {}: {}", addr, e);
                ProtocolError::Io(e)
            })?;
            // Minimise latency: no Nagle, large socket buffers (256 KB each)
            let _ = s.set_nodelay(true);
            #[cfg(unix)]
            {
                use std::os::unix::io::AsRawFd;
                let fd = s.as_raw_fd();
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
            client_handshake_sync(&mut s, token).map_err(|e| {
                eprintln!("[vkremote-icd] Handshake failed with {}: {}", addr, e);
                e
            })?;
            eprintln!(
                "[vkremote-icd] Successfully connected to remote server at {}!",
                self.server_addr
            );
            *guard = Some(s);
        }
        Ok(())
    }
}

impl Transport for TcpTransport {
    fn roundtrip(&self, req: &Request) -> Result<Response, ProtocolError> {
        self.ensure_stream()?;
        let mut guard = self.stream.lock().unwrap();
        let stream = guard.as_mut().unwrap();

        // Telemetry: measure serialise cost + bytes + blocked time
        let t_ser_start = Instant::now();
        let bytes_sent = match write_msg_sync(stream, req) {
            Ok(bytes) => bytes as u64,
            Err(e) => {
                eprintln!(
                    "[vkremote-icd] write_msg_sync FAILED (request kind {:?}): {:?}",
                    std::mem::discriminant(req),
                    e
                );
                *guard = None;
                return Err(e);
            }
        };
        let serialize_us = t_ser_start.elapsed().as_micros() as u64;

        let t_blocked = Instant::now();
        let (resp, bytes_recv) = match read_msg_sync_with_len(stream, DEFAULT_MAX_FRAME_SIZE) {
            Ok((r, len)) => (r, len as u64),
            Err(e) => {
                eprintln!(
                    "[vkremote-icd] read_msg_sync FAILED (request kind {:?}): {:?}",
                    std::mem::discriminant(req),
                    e
                );
                *guard = None;
                return Err(e);
            }
        };
        let blocked_us = t_blocked.elapsed().as_micros() as u64;

        telemetry().record_rtt(bytes_sent, bytes_recv, serialize_us, blocked_us);
        crate::perf::record_rpc(req, bytes_sent, bytes_recv, serialize_us + blocked_us);

        Ok(resp)
    }
}

pub struct MockTransport {
    handler: Box<dyn Fn(&Request) -> Result<Response, ProtocolError> + Send + Sync>,
}

impl MockTransport {
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(&Request) -> Result<Response, ProtocolError> + Send + Sync + 'static,
    {
        Self {
            handler: Box::new(handler),
        }
    }
}

impl Transport for MockTransport {
    fn roundtrip(&self, req: &Request) -> Result<Response, ProtocolError> {
        (self.handler)(req)
    }
}

pub struct BufferInfo {
    pub size: u64,
    pub usage: u32,
    pub bound_memory: Option<(u64, u64)>, // memory_id, offset
}

#[derive(Debug, Clone)]
pub enum WindowHandle {
    Win32 {
        hwnd: ash::vk::HWND,
        hinstance: ash::vk::HINSTANCE,
    },
    Wayland {
        display: usize,
        surface: usize,
    },
    Xcb {
        connection: usize,
        window: u32,
    },
    Xlib {
        display: usize,
        window: usize,
    },
}

#[derive(Debug, Clone)]
pub struct SurfaceInfo {
    pub id: u64,
    pub window: WindowHandle,
}

#[derive(Debug)]
pub struct SwapchainInfo {
    pub id: u64,
    pub surface_id: u64,
    pub width: u32,
    pub height: u32,
    pub format: ash::vk::Format,
    pub image_ids: Vec<u64>,
    pub current_index: std::sync::atomic::AtomicU32,
}

#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub id: u64,
    pub format: ash::vk::Format,
    pub extent: ash::vk::Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub usage: ash::vk::ImageUsageFlags,
    pub bound_memory: Option<(u64, u64)>, // memory_id, offset
}

pub struct AllocationInfo {
    pub size: u64,
    pub memory_type_index: u32,
    pub is_host_visible: bool,
    pub shadow_buffer: Option<crate::shadow::ShadowBuffer>,
    pub mapped_range: Option<(u64, u64)>, // offset, size
    pub has_been_mapped: bool,
}

pub struct Client {
    transport: Box<dyn Transport>,
    next_id: AtomicU64,
    snapshot: Mutex<Option<Arc<PhysicalDeviceSnapshot>>>,
    allocations: Mutex<HashMap<u64, AllocationInfo>>,
    buffers: Mutex<HashMap<u64, BufferInfo>>,
    surfaces: Mutex<HashMap<u64, SurfaceInfo>>,
    swapchains: Mutex<HashMap<u64, SwapchainInfo>>,
    images: Mutex<HashMap<u64, ImageInfo>>,
    recorded_commands: Mutex<HashMap<u64, Vec<Request>>>,
    sync_tracker: Mutex<MemorySyncTracker>,
    timeline_values: Mutex<HashMap<u64, u64>>,
    timeline_changed: Condvar,
    pending_destructions: Mutex<Vec<Request>>,
}

impl Client {
    pub fn new_with_transport(transport: Box<dyn Transport>) -> Arc<Self> {
        Arc::new(Self {
            transport,
            next_id: AtomicU64::new(1),
            snapshot: Mutex::new(None),
            allocations: Mutex::new(HashMap::new()),
            buffers: Mutex::new(HashMap::new()),
            surfaces: Mutex::new(HashMap::new()),
            swapchains: Mutex::new(HashMap::new()),
            images: Mutex::new(HashMap::new()),
            recorded_commands: Mutex::new(HashMap::new()),
            sync_tracker: Mutex::new(MemorySyncTracker::new()),
            timeline_values: Mutex::new(HashMap::new()),
            timeline_changed: Condvar::new(),
            pending_destructions: Mutex::new(Vec::new()),
        })
    }

    pub fn from_env() -> Result<Arc<Self>, ProtocolError> {
        let server_addr = std::env::var("VKREMOTE_SERVER")
            .ok()
            .or_else(|| {
                std::env::var("HOME").ok().and_then(|h| {
                    std::fs::read_to_string(format!("{}/.config/vkremote/server.conf", h))
                        .ok()
                        .map(|s| s.trim().to_string())
                })
            })
            .or_else(|| {
                std::fs::read_to_string("vkremote_server.txt")
                    .ok()
                    .map(|s| s.trim().to_string())
            })
            .or_else(|| {
                std::fs::read_to_string(r"C:\rgpu\vkremote_server.txt")
                    .ok()
                    .map(|s| s.trim().to_string())
            })
            .or_else(|| {
                std::fs::read_to_string(r"C:\ProgramData\vkremote\server.txt")
                    .ok()
                    .map(|s| s.trim().to_string())
            })
            .or_else(|| {
                std::fs::read_to_string("/etc/vkremote/server.conf")
                    .ok()
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "192.168.1.9:50051".to_string());
        let token = std::env::var("VKREMOTE_TOKEN")
            .ok()
            .or_else(|| {
                std::env::var("HOME").ok().and_then(|h| {
                    std::fs::read_to_string(format!("{}/.config/vkremote/token.conf", h))
                        .ok()
                        .map(|s| s.trim().to_string())
                })
            })
            .or_else(|| {
                std::fs::read_to_string("vkremote_token.txt")
                    .ok()
                    .map(|s| s.trim().to_string())
            })
            .or_else(|| {
                std::fs::read_to_string(r"C:\rgpu\vkremote_token.txt")
                    .ok()
                    .map(|s| s.trim().to_string())
            })
            .or_else(|| {
                std::fs::read_to_string(r"C:\ProgramData\vkremote\token.txt")
                    .ok()
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "vkremote-secret".to_string());

        let tcp_transport = TcpTransport::new(server_addr, token);
        Ok(Self::new_with_transport(Box::new(tcp_transport)))
    }

    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    pub fn register_timeline(&self, id: u64, value: u64) {
        self.timeline_values.lock().unwrap().insert(id, value);
    }

    pub fn remove_timeline(&self, id: u64) {
        self.timeline_values.lock().unwrap().remove(&id);
    }

    pub fn timeline_value(&self, id: u64) -> Option<u64> {
        self.timeline_values.lock().unwrap().get(&id).copied()
    }

    pub fn signal_timeline(&self, id: u64, value: u64) -> ash::vk::Result {
        let mut values = self.timeline_values.lock().unwrap();
        match values.get_mut(&id) {
            Some(current) if value > *current => *current = value,
            _ => return ash::vk::Result::ERROR_UNKNOWN,
        }
        self.timeline_changed.notify_all();
        ash::vk::Result::SUCCESS
    }

    pub fn wait_timelines(&self, waits: &[(u64, u64)], any: bool, timeout: u64) -> ash::vk::Result {
        let started = Instant::now();
        let mut values = self.timeline_values.lock().unwrap();
        loop {
            if waits.iter().any(|(id, _)| !values.contains_key(id)) {
                return ash::vk::Result::ERROR_UNKNOWN;
            }
            let ready = |(id, value): &(u64, u64)| values[id] >= *value;
            if waits.is_empty()
                || if any {
                    waits.iter().any(ready)
                } else {
                    waits.iter().all(ready)
                }
            {
                return ash::vk::Result::SUCCESS;
            }
            if timeout == u64::MAX {
                values = self.timeline_changed.wait(values).unwrap();
            } else {
                let remaining =
                    std::time::Duration::from_nanos(timeout).saturating_sub(started.elapsed());
                if remaining.is_zero() {
                    return ash::vk::Result::TIMEOUT;
                }
                values = self
                    .timeline_changed
                    .wait_timeout(values, remaining)
                    .unwrap()
                    .0;
            }
        }
    }

    #[inline]
    fn is_deferrable_command(req: &Request) -> bool {
        match req {
            Request::DestroyResource { .. }
            | Request::DestroyImageView { .. }
            | Request::DestroyFramebuffer { .. }
            | Request::DestroyRenderPass { .. }
            | Request::DestroySampler { .. }
            | Request::DestroyImage { .. }
            | Request::BindBufferMemory { .. }
            | Request::CreateFramebuffer { .. }
            | Request::CreateImageView { .. }
            | Request::UpdateDescriptorSets { .. }
            // Client-assigned ids; replies carry nothing but success/error. Deferring
            // removes one Wi-Fi round trip each (Blender issues hundreds per frame).
            | Request::CreateShaderModule { .. }
            | Request::CreatePipelineLayout { .. }
            | Request::AllocateDescriptorSets { .. }
            | Request::CreateRenderPass { .. }
            // Client-assigned pipeline ids; the reply is only logged. Blender
            // compiles pipelines lazily, ~15-50 per frame while warming up.
            | Request::CreateGraphicsPipelines { .. }
            // Replies ignored (client computes the image index / uses own ids); the
            // server still executes them in order before the next synchronous call.
            | Request::AcquireNextImage { .. }
            | Request::BindImageMemory { .. }
            | Request::ResetFences { .. }
            | Request::CreateDescriptorPool { .. }
            | Request::CommandBatch { .. } => true,
            // Same opt-in as CreateBuffer: deferring hides creation failures.
            Request::CreateBuffer { .. } | Request::CreateImage { .. } => {
                static BATCH_CREATE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                static INIT: std::sync::Once = std::sync::Once::new();
                INIT.call_once(|| {
                    // Opt-in: deferring hides CreateBuffer failures from the caller.
                    if let Ok(val) = std::env::var("VKREMOTE_BATCH_CREATE_BUFFER") {
                        BATCH_CREATE.store(val != "0", std::sync::atomic::Ordering::Relaxed);
                    }
                });
                BATCH_CREATE.load(std::sync::atomic::Ordering::Relaxed)
            }
            _ => false,
        }
    }

    /// `VKREMOTE_PIGGYBACK_SYNC=1`: append a synchronous request to the pending
    /// deferred batch instead of flushing it in a separate round trip.
    fn piggyback_sync() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("VKREMOTE_PIGGYBACK_SYNC").map_or(false, |v| v != "0"))
    }

    #[inline]
    fn requires_flush_before(req: &Request) -> bool {
        matches!(
            req,
            Request::QueueSubmit { .. }
                | Request::QueuePresent { .. }
                | Request::BeginCommandBuffer { .. }
                | Request::WriteMemoryRanges { .. }
                | Request::ReadMemoryRanges { .. }
                | Request::CreateGraphicsPipelines { .. }
                | Request::CreateComputePipeline { .. }
                | Request::CreateRenderPass { .. }
        )
    }

    pub fn flush_deferred_commands(&self) -> Result<(), ProtocolError> {
        let batch = {
            let mut guard = self.pending_destructions.lock().unwrap();
            if guard.is_empty() {
                return Ok(());
            }
            std::mem::take(&mut *guard)
        };
        let mut flat = Vec::new();
        for req in batch {
            match req {
                Request::CommandBatch { commands } => flat.extend(commands),
                other => flat.push(other),
            }
        }
        for chunk in flat.chunks(512) {
            let _ = self.transport.roundtrip(&Request::CommandBatch {
                commands: chunk.to_vec(),
            })?;
        }
        Ok(())
    }

    #[inline]
    pub fn flush_destructions(&self) -> Result<(), ProtocolError> {
        self.flush_deferred_commands()
    }

    /// `VKREMOTE_ASYNC_SUBMIT=1`: vkQueueSubmit queues the submit instead of waiting.
    pub fn async_submit() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("VKREMOTE_ASYNC_SUBMIT").map_or(false, |v| v != "0"))
    }

    /// Queue a request without waiting; it is sent in order ahead of the next
    /// synchronous request (or when the queue reaches 512 entries).
    pub fn defer_request(&self, req: Request) -> Result<(), ProtocolError> {
        let to_flush = {
            let mut guard = self.pending_destructions.lock().unwrap();
            guard.push(req);
            // Bound one batch message: flush early once queued memory uploads
            // exceed 16 MiB (each WriteMemoryRanges can carry ~2 MiB).
            let queued_bytes: usize = guard
                .iter()
                .map(|r| match r {
                    Request::WriteMemoryRanges { diffs } => {
                        diffs.iter().map(|d| d.data.len()).sum::<usize>()
                    }
                    _ => 0,
                })
                .sum();
            if guard.len() < 512 && queued_bytes < 16 * 1024 * 1024 {
                return Ok(());
            }
            std::mem::take(&mut *guard)
        };
        let mut flat = Vec::new();
        for r in to_flush {
            match r {
                Request::CommandBatch { commands } => flat.extend(commands),
                other => flat.push(other),
            }
        }
        for chunk in flat.chunks(512) {
            let _ = self.transport.roundtrip(&Request::CommandBatch {
                commands: chunk.to_vec(),
            })?;
        }
        Ok(())
    }

    pub fn send_request(&self, req: Request) -> Result<Response, ProtocolError> {
        if Self::is_deferrable_command(&req) {
            let mut guard = self.pending_destructions.lock().unwrap();
            guard.push(req);
            if guard.len() >= 512 {
                let to_flush = std::mem::take(&mut *guard);
                drop(guard);
                let mut flat = Vec::new();
                for r in to_flush {
                    match r {
                        Request::CommandBatch { commands } => flat.extend(commands),
                        other => flat.push(other),
                    }
                }
                for chunk in flat.chunks(512) {
                    let _ = self.transport.roundtrip(&Request::CommandBatch {
                        commands: chunk.to_vec(),
                    })?;
                }
            }
            return Ok(Response::Success);
        }

        // Any request that waits for a reply may depend on deferred creations
        // (e.g. buffer memory requirements after a deferred CreateBuffer), so drain
        // the deferred queue first to preserve submission order.
        let _ = Self::requires_flush_before(&req);
        if Self::piggyback_sync() && !matches!(req, Request::CommandBatch { .. }) {
            let pending = std::mem::take(&mut *self.pending_destructions.lock().unwrap());
            if !pending.is_empty() {
                // One round trip: deferred commands + this request, server replies
                // with the last command's response.
                let mut flat = Vec::new();
                for r in pending {
                    match r {
                        Request::CommandBatch { commands } => flat.extend(commands),
                        other => flat.push(other),
                    }
                }
                let split = flat.len().saturating_sub(511);
                let tail = flat.split_off(split);
                for chunk in flat.chunks(512) {
                    let _ = self.transport.roundtrip(&Request::CommandBatch {
                        commands: chunk.to_vec(),
                    })?;
                }
                let mut last = tail;
                last.push(req);
                return self.transport.roundtrip(&Request::CommandBatch { commands: last });
            }
        }
        let _ = self.flush_deferred_commands();

        static REQUEST_COUNT: AtomicU64 = AtomicU64::new(0);
        let count = REQUEST_COUNT.fetch_add(1, Ordering::Relaxed);
        if count % 1024 == 0 {
            eprintln!(
                "[vkremote-icd] request #{count} kind {:?}",
                std::mem::discriminant(&req)
            );
            if let Request::BindBufferMemory {
                buffer_id,
                memory_id,
                offset,
            } = &req
            {
                eprintln!(
                    "[vkremote-icd] bind buffer={buffer_id} memory={memory_id} offset={offset}"
                );
            }
        }
        self.transport.roundtrip(&req)
    }

    pub fn get_or_fetch_snapshot(&self) -> Result<Arc<PhysicalDeviceSnapshot>, ProtocolError> {
        let mut guard = self.snapshot.lock().unwrap();
        if let Some(snapshot) = &*guard {
            return Ok(Arc::clone(snapshot));
        }

        match self.send_request(Request::GetDeviceSnapshot)? {
            Response::DeviceSnapshot(snapshot) => {
                let arc_snap = Arc::new(snapshot);
                *guard = Some(Arc::clone(&arc_snap));
                Ok(arc_snap)
            }
            Response::Error { code, message } => {
                tracing::error!("Failed to get device snapshot: {} ({})", message, code);
                Err(ProtocolError::HandshakeRejected(message))
            }
            other => {
                tracing::error!("Unexpected snapshot response: {:?}", other);
                Err(ProtocolError::HandshakeRejected(format!(
                    "unexpected response: {:?}",
                    other
                )))
            }
        }
    }

    pub fn register_buffer(&self, id: u64, size: u64, usage: u32) {
        let mut guard = self.buffers.lock().unwrap();
        guard.insert(
            id,
            BufferInfo {
                size,
                usage,
                bound_memory: None,
            },
        );
    }

    pub fn unregister_buffer(&self, id: u64) {
        let mut guard = self.buffers.lock().unwrap();
        guard.remove(&id);
    }

    pub fn get_buffer_size(&self, id: u64) -> Option<u64> {
        let guard = self.buffers.lock().unwrap();
        guard.get(&id).map(|b| b.size)
    }

    pub fn bind_buffer_memory(&self, buffer_id: u64, memory_id: u64, offset: u64) {
        let mut guard = self.buffers.lock().unwrap();
        if let Some(buf) = guard.get_mut(&buffer_id) {
            buf.bound_memory = Some((memory_id, offset));
        }
    }

    pub fn register_surface(&self, id: u64, window: WindowHandle) {
        let mut guard = self.surfaces.lock().unwrap();
        guard.insert(id, SurfaceInfo { id, window });
    }

    pub fn unregister_surface(&self, id: u64) {
        let mut guard = self.surfaces.lock().unwrap();
        guard.remove(&id);
    }

    pub fn get_surface(&self, id: u64) -> Option<SurfaceInfo> {
        let guard = self.surfaces.lock().unwrap();
        guard.get(&id).cloned()
    }

    pub fn register_swapchain(
        &self,
        id: u64,
        surface_id: u64,
        width: u32,
        height: u32,
        format: ash::vk::Format,
        image_ids: Vec<u64>,
    ) {
        let mut guard = self.swapchains.lock().unwrap();
        guard.insert(
            id,
            SwapchainInfo {
                id,
                surface_id,
                width,
                height,
                format,
                image_ids,
                current_index: std::sync::atomic::AtomicU32::new(0),
            },
        );
    }

    pub fn unregister_swapchain(&self, id: u64) {
        let mut guard = self.swapchains.lock().unwrap();
        guard.remove(&id);
    }

    pub fn get_swapchain_images(&self, id: u64) -> Option<Vec<u64>> {
        let guard = self.swapchains.lock().unwrap();
        guard.get(&id).map(|s| s.image_ids.clone())
    }

    pub fn acquire_next_image_index(&self, id: u64) -> Option<u32> {
        let guard = self.swapchains.lock().unwrap();
        guard.get(&id).map(|s| {
            let count = s.image_ids.len() as u32;
            if count == 0 {
                0
            } else {
                s.current_index.fetch_add(1, Ordering::SeqCst) % count
            }
        })
    }

    pub fn get_swapchain_surface_id(&self, id: u64) -> Option<u64> {
        let guard = self.swapchains.lock().unwrap();
        guard.get(&id).map(|s| s.surface_id)
    }

    /// Starts local recording for a Vulkan command buffer. Command recording
    /// calls do not need a server response; the completed stream is delivered
    /// atomically before vkEndCommandBuffer returns.
    pub fn begin_command_recording(&self, command_buffer_id: u64) {
        self.recorded_commands
            .lock()
            .unwrap()
            .insert(command_buffer_id, Vec::new());
    }

    /// Returns true when the command was retained for the current recording.
    /// Returning false preserves the old immediate transport behavior for
    /// callers outside a begun command buffer.
    pub fn record_command(&self, command_buffer_id: u64, request: Request) -> bool {
        let mut recordings = self.recorded_commands.lock().unwrap();
        let Some(commands) = recordings.get_mut(&command_buffer_id) else {
            return false;
        };
        commands.push(request);
        true
    }

    pub fn finish_command_recording(&self, command_buffer_id: u64) -> Option<Vec<Request>> {
        self.recorded_commands
            .lock()
            .unwrap()
            .remove(&command_buffer_id)
    }

    pub fn discard_command_recording(&self, command_buffer_id: u64) {
        self.recorded_commands
            .lock()
            .unwrap()
            .remove(&command_buffer_id);
    }

    pub fn register_image(
        &self,
        id: u64,
        format: ash::vk::Format,
        extent: ash::vk::Extent3D,
        mip_levels: u32,
        array_layers: u32,
        usage: ash::vk::ImageUsageFlags,
    ) {
        let mut guard = self.images.lock().unwrap();
        guard.insert(
            id,
            ImageInfo {
                id,
                format,
                extent,
                mip_levels,
                array_layers,
                usage,
                bound_memory: None,
            },
        );
    }

    pub fn unregister_image(&self, id: u64) {
        let mut guard = self.images.lock().unwrap();
        guard.remove(&id);
    }

    pub fn get_image_info(&self, id: u64) -> Option<ImageInfo> {
        let guard = self.images.lock().unwrap();
        guard.get(&id).cloned()
    }

    pub fn bind_image_memory(&self, image_id: u64, memory_id: u64, offset: u64) {
        let mut guard = self.images.lock().unwrap();
        if let Some(img) = guard.get_mut(&image_id) {
            img.bound_memory = Some((memory_id, offset));
        }
    }

    pub fn register_allocation(
        &self,
        id: u64,
        size: u64,
        memory_type_index: u32,
        is_host_visible: bool,
    ) {
        let shadow_buffer = if is_host_visible {
            Some(crate::shadow::ShadowBuffer::new(size as usize))
        } else {
            None
        };

        let mut guard = self.allocations.lock().unwrap();
        guard.insert(
            id,
            AllocationInfo {
                size,
                memory_type_index,
                is_host_visible,
                shadow_buffer,
                mapped_range: None,
                has_been_mapped: false,
            },
        );
        if is_host_visible {
            let mut tracker = self.sync_tracker.lock().unwrap();
            tracker.init_clean_chunks(id, size as usize);
        }
    }

    pub fn unregister_allocation(&self, id: u64) {
        let mut guard = self.allocations.lock().unwrap();
        guard.remove(&id);
        let mut tracker = self.sync_tracker.lock().unwrap();
        tracker.remove_allocation(id);
    }

    pub fn map_memory(&self, id: u64, offset: u64, size: u64) -> Result<*mut u8, ash::vk::Result> {
        let mut guard = self.allocations.lock().unwrap();
        let alloc = guard
            .get_mut(&id)
            .ok_or(ash::vk::Result::ERROR_MEMORY_MAP_FAILED)?;

        if !alloc.is_host_visible || alloc.shadow_buffer.is_none() {
            return Err(ash::vk::Result::ERROR_MEMORY_MAP_FAILED);
        }

        let actual_size = if size == ash::vk::WHOLE_SIZE {
            alloc.size.saturating_sub(offset)
        } else {
            size
        };

        if offset + actual_size > alloc.size {
            return Err(ash::vk::Result::ERROR_MEMORY_MAP_FAILED);
        }

        alloc.mapped_range = Some((offset, actual_size));
        alloc.has_been_mapped = true;
        let shadow = alloc.shadow_buffer.as_mut().unwrap();
        // SAFETY: offset was verified within alloc.size above
        let ptr = unsafe { shadow.as_mut_ptr().add(offset as usize) };
        Ok(ptr)
    }

    pub fn unmap_memory(&self, id: u64) {
        let mut guard = self.allocations.lock().unwrap();
        if let Some(alloc) = guard.get_mut(&id) {
            alloc.mapped_range = None;
        }
    }

    /// Extract dirty chunks across all host-visible memory allocations that have been mapped.
    pub fn collect_dirty_memory(&self) -> Vec<MemoryChunkDiff> {
        let _span = crate::perf::Span::new(&crate::perf::DIRTY_SCAN_US);
        let mut guard = self.allocations.lock().unwrap();
        let mut tracker = self.sync_tracker.lock().unwrap();
        let mut all_diffs = Vec::new();

        for (&alloc_id, alloc) in guard.iter_mut() {
            if alloc.is_host_visible && alloc.has_been_mapped {
                if let Some(shadow) = &alloc.shadow_buffer {
                    let chunk = vkremote_wire::sync::CHUNK_SIZE;
                    let diffs = match shadow.dirty_chunks(chunk) {
                        // OS write tracking: hash only chunks the app touched.
                        Some(offsets) => {
                            crate::perf::DIRTY_SCAN_BYTES.add((offsets.len() * chunk) as u64);
                            tracker.compute_dirty_chunks_at(alloc_id, shadow, &offsets)
                        }
                        None => {
                            crate::perf::DIRTY_SCAN_BYTES.add(shadow.len() as u64);
                            tracker.compute_dirty_chunks(alloc_id, shadow)
                        }
                    };
                    all_diffs.extend(diffs);
                }
            }
        }

        crate::perf::DIRTY_SENT_BYTES.add(all_diffs.iter().map(|d| d.data.len() as u64).sum());
        all_diffs
    }

    /// Apply updated memory diffs from server to client shadow buffers.
    pub fn apply_memory_diffs(&self, diffs: &[MemoryChunkDiff]) {
        let mut guard = self.allocations.lock().unwrap();
        let mut tracker = self.sync_tracker.lock().unwrap();

        for diff in diffs {
            crate::perf::DIFF_APPLY_BYTES.add(diff.data.len() as u64);
            if let Some(alloc) = guard.get_mut(&diff.allocation_id) {
                if let Some(shadow) = &mut alloc.shadow_buffer {
                    let start = diff.offset as usize;
                    let end = start + diff.data.len();
                    if end <= shadow.len() {
                        shadow[start..end].copy_from_slice(&diff.data);
                        tracker.mark_clean(diff.allocation_id, diff.offset, &diff.data);
                    }
                }
            }
        }
    }

    /// Pull remote memory ranges for all currently host-visible or mapped allocations.
    pub fn sync_remote_memory_to_shadow(&self) -> Result<(), ProtocolError> {
        let host_visible_ids: Vec<u64> = {
            let guard = self.allocations.lock().unwrap();
            guard
                .iter()
                .filter(|(_, a)| a.is_host_visible)
                .map(|(&id, _)| id)
                .collect()
        };

        if host_visible_ids.is_empty() {
            return Ok(());
        }

        self.sync_remote_allocations_to_shadow(host_visible_ids)
    }

    /// Pull the allocations explicitly requested by a Vulkan memory invalidate.
    pub fn sync_remote_allocations_to_shadow(
        &self,
        allocation_ids: Vec<u64>,
    ) -> Result<(), ProtocolError> {
        if allocation_ids.is_empty() {
            return Ok(());
        }

        let resp = self.send_request(Request::ReadMemoryRanges { allocation_ids })?;

        if let Response::MemoryData { diffs } = resp {
            self.apply_memory_diffs(&diffs);
        }
        Ok(())
    }
}
