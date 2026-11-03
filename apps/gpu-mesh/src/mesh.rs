use serde::{Deserialize, Serialize};
use std::time::Instant;

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    Giver,
    Taker,
    GiverAndTaker,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionStatus {
    Connected,
    Connecting,
    Listening,
    Disconnected,
    Standby,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuDevice {
    pub name: String,
    pub vram_mb: u64,
    pub is_discrete: bool,
    pub is_shared: bool,
    pub driver_version: String,
    pub api_version: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct RemoteGiver {
    pub id: String,
    pub name: String,
    pub endpoint: String,
    pub status: ConnectionStatus,
    pub latency_ms: f32,
    pub gpus: Vec<GpuDevice>,
    pub last_ping: Option<Instant>,
    pub total_vram_mb: u64,
    pub allocated_vram_mb: Option<u64>,
}

impl RemoteGiver {
    pub fn effective_vram_mb(&self) -> u64 {
        self.allocated_vram_mb.unwrap_or(self.total_vram_mb)
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct LocalNode {
    pub name: String,
    pub ip: String,
    pub port: u16,
    pub giver_enabled: bool,
    pub taker_enabled: bool,
    pub active_clients_count: usize,
    pub gpus: Vec<GpuDevice>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct ClusterSummary {
    pub total_vram_mb: u64,
    pub local_vram_mb: u64,
    pub remote_vram_mb: u64,
    pub total_gpus_count: usize,
    pub active_givers_count: usize,
    pub avg_latency_ms: f32,
}

#[allow(dead_code)]
impl ClusterSummary {
    pub fn total_vram_gb(&self) -> f64 {
        self.total_vram_mb as f64 / 1024.0
    }

    pub fn local_vram_gb(&self) -> f64 {
        self.local_vram_mb as f64 / 1024.0
    }

    pub fn remote_vram_gb(&self) -> f64 {
        self.remote_vram_mb as f64 / 1024.0
    }
}
