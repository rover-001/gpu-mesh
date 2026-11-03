use vkremote_wire::types::*;

pub struct MockEngine {
    pub snapshot: PhysicalDeviceSnapshot,
}

impl MockEngine {
    pub fn new() -> Self {
        let limits = PhysicalDeviceLimitsData::default();
        let features = PhysicalDeviceFeaturesData {
            robust_buffer_access: true,
            shader_float64: true,
            shader_int64: true,
            shader_int16: true,
            ..Default::default()
        };

        let memory_properties = PhysicalDeviceMemoryPropertiesData {
            memory_types: vec![
                MemoryTypeData {
                    // VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 1
                    property_flags: 1,
                    heap_index: 0,
                },
                MemoryTypeData {
                    // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT (2) | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT (4)
                    property_flags: 2 | 4,
                    heap_index: 1,
                },
            ],
            memory_heaps: vec![
                MemoryHeapData {
                    size: 8 * 1024 * 1024 * 1024, // 8 GB VRAM
                    flags: 1,                     // DEVICE_LOCAL
                },
                MemoryHeapData {
                    size: 16 * 1024 * 1024 * 1024, // 16 GB Host RAM
                    flags: 0,
                },
            ],
        };

        let queue_families = vec![QueueFamilyPropertiesData {
            // VK_QUEUE_GRAPHICS_BIT (1) | VK_QUEUE_COMPUTE_BIT (2) | VK_QUEUE_TRANSFER_BIT (4)
            queue_flags: 1 | 2 | 4,
            queue_count: 1,
            timestamp_valid_bits: 64,
            min_image_transfer_granularity: [1, 1, 1],
        }];

        let properties = PhysicalDevicePropertiesData {
            api_version: 0x00402000, // Vulkan 1.2
            driver_version: 1,
            vendor_id: 0x10de, // Mock NVIDIA
            device_id: 0x28a1,
            device_type: 2, // DISCRETE_GPU
            device_name: "Mock NVIDIA Vulkan Compute Accelerator".to_string(),
            pipeline_cache_uuid: [0x42; 16],
            limits,
        };

        let snapshot = PhysicalDeviceSnapshot {
            properties,
            features,
            memory_properties,
            queue_families,
        };

        Self { snapshot }
    }
}

impl Default for MockEngine {
    fn default() -> Self {
        Self::new()
    }
}
