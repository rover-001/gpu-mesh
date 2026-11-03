pub mod mock;
pub mod vulkan;

use anyhow::Result;
use std::sync::Arc;
use tracing::info;
use vkremote_wire::types::PhysicalDeviceSnapshot;

pub use mock::MockEngine;
pub use vulkan::VulkanEngine;

#[derive(Clone)]
pub enum Engine {
    Vulkan(Arc<VulkanEngine>),
    Mock(Arc<MockEngine>),
}

impl Engine {
    pub fn new(gpu_index: Option<usize>, force_mock: bool) -> Result<Self> {
        if force_mock {
            info!("Force mock enabled: initializing Mock Engine");
            return Ok(Engine::Mock(Arc::new(MockEngine::new())));
        }

        let vk = VulkanEngine::new(gpu_index)?;
        info!("Successfully initialized real Vulkan ash backend");
        Ok(Engine::Vulkan(vk))
    }

    pub fn snapshot(&self) -> PhysicalDeviceSnapshot {
        match self {
            Engine::Vulkan(v) => v.snapshot.clone(),
            Engine::Mock(m) => m.snapshot.clone(),
        }
    }
}
