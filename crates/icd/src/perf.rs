//! Opt-in per-frame profiler for the ICD (`VKREMOTE_PERF=1`).
//!
//! Counters are always cheap atomics; the report line is printed only when
//! profiling is enabled, once per 60 presented frames or every 5 seconds.

use std::collections::HashMap;
use std::mem::Discriminant;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::Instant;
use vkremote_wire::types::Request;

pub use vkremote_wire::perf::enabled;

#[derive(Default)]
pub struct Counter(AtomicU64);

impl Counter {
    #[inline]
    pub fn add(&self, v: u64) {
        self.0.fetch_add(v, Relaxed);
    }
    fn take(&self) -> u64 {
        self.0.swap(0, Relaxed)
    }
}

pub static RPC_COUNT: Counter = Counter(AtomicU64::new(0));
pub static RPC_BYTES_SENT: Counter = Counter(AtomicU64::new(0));
pub static RPC_BYTES_RECV: Counter = Counter(AtomicU64::new(0));
pub static RPC_BLOCKED_US: Counter = Counter(AtomicU64::new(0));
pub static DIRTY_SCAN_US: Counter = Counter(AtomicU64::new(0));
pub static DIRTY_SCAN_BYTES: Counter = Counter(AtomicU64::new(0));
pub static DIRTY_SENT_BYTES: Counter = Counter(AtomicU64::new(0));
pub static DIFF_APPLY_BYTES: Counter = Counter(AtomicU64::new(0));
pub static SUBMIT_US: Counter = Counter(AtomicU64::new(0));
pub static FENCE_WAIT_US: Counter = Counter(AtomicU64::new(0));
pub static PRESENT_US: Counter = Counter(AtomicU64::new(0));
pub static BLIT_US: Counter = Counter(AtomicU64::new(0));

/// Adds the elapsed wall time to a counter when dropped.
pub struct Span {
    counter: &'static Counter,
    start: Instant,
}

impl Span {
    #[inline]
    pub fn new(counter: &'static Counter) -> Self {
        Self {
            counter,
            start: Instant::now(),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        self.counter.add(self.start.elapsed().as_micros() as u64);
    }
}

#[derive(Default)]
struct KindStats {
    name: String,
    count: u64,
    blocked_us: u64,
    bytes: u64,
}

struct Window {
    started: Instant,
    frames: u64,
    kinds: HashMap<Discriminant<Request>, KindStats>,
}

static WINDOW: Mutex<Option<Window>> = Mutex::new(None);

fn with_window<R>(f: impl FnOnce(&mut Window) -> R) -> R {
    let mut guard = WINDOW.lock().unwrap();
    let w = guard.get_or_insert_with(|| Window {
        started: Instant::now(),
        frames: 0,
        kinds: HashMap::new(),
    });
    f(w)
}

/// Record one completed round trip.
pub fn record_rpc(req: &Request, bytes_sent: u64, bytes_recv: u64, blocked_us: u64) {
    RPC_COUNT.add(1);
    RPC_BYTES_SENT.add(bytes_sent);
    RPC_BYTES_RECV.add(bytes_recv);
    RPC_BLOCKED_US.add(blocked_us);
    if !enabled() {
        return;
    }
    // A piggybacked batch ends with the synchronous request it carries; attribute
    // the round trip to that request so the report shows what still blocks.
    let req = match req {
        Request::CommandBatch { commands } => commands.last().unwrap_or(req),
        other => other,
    };
    with_window(|w| {
        let k = w
            .kinds
            .entry(std::mem::discriminant(req))
            .or_insert_with(|| KindStats {
                name: vkremote_wire::perf::variant_name(req),
                ..Default::default()
            });
        k.count += 1;
        k.blocked_us += blocked_us;
        k.bytes += bytes_sent + bytes_recv;
    });
}

/// Mark the end of one presented frame and print a report when the window closes.
pub fn end_frame() {
    if !enabled() {
        return;
    }
    let report = with_window(|w| {
        w.frames += 1;
        let elapsed = w.started.elapsed();
        if w.frames < 60 && elapsed.as_secs() < 5 {
            return None;
        }
        let frames = w.frames.max(1);
        let per = |v: u64| v as f64 / frames as f64 / 1000.0; // µs total -> ms/frame
        let kb = |v: u64| v as f64 / frames as f64 / 1024.0; // bytes -> KB/frame
        let mut kinds: Vec<_> = w.kinds.drain().map(|(_, k)| k).collect();
        kinds.sort_by(|a, b| b.blocked_us.cmp(&a.blocked_us));
        let top: Vec<String> = kinds
            .iter()
            .take(6)
            .map(|k| {
                format!(
                    "{}x{:.1}={:.1}ms/{:.0}KB",
                    k.name,
                    k.count as f64 / frames as f64,
                    per(k.blocked_us),
                    kb(k.bytes)
                )
            })
            .collect();
        let line = format!(
            "[vkremote-perf] fps={:.2} ms/frame={:.1} | rpc={:.1}/f blocked={:.1}ms sent={:.0}KB recv={:.0}KB | \
             submit={:.1}ms fence={:.1}ms present={:.1}ms blit={:.1}ms | \
             dirty_scan={:.1}ms over {:.0}KB sent={:.0}KB apply={:.0}KB | top: {}",
            frames as f64 / elapsed.as_secs_f64().max(1e-9),
            elapsed.as_secs_f64() * 1000.0 / frames as f64,
            RPC_COUNT.take() as f64 / frames as f64,
            per(RPC_BLOCKED_US.take()),
            kb(RPC_BYTES_SENT.take()),
            kb(RPC_BYTES_RECV.take()),
            per(SUBMIT_US.take()),
            per(FENCE_WAIT_US.take()),
            per(PRESENT_US.take()),
            per(BLIT_US.take()),
            per(DIRTY_SCAN_US.take()),
            kb(DIRTY_SCAN_BYTES.take()),
            kb(DIRTY_SENT_BYTES.take()),
            kb(DIFF_APPLY_BYTES.take()),
            top.join(" "),
        );
        w.started = Instant::now();
        w.frames = 0;
        Some(line)
    });
    if let Some(line) = report {
        eprintln!("{line}");
    }
}
