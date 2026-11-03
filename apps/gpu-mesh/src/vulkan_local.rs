use anyhow::{Context, Result};
use ash::vk;
use std::ffi::CStr;

#[derive(Clone, Debug)]
pub struct LocalGpuInfo {
    pub name: String,
    #[allow(dead_code)]
    pub device_type: String,
    pub vram_mb: u64,
    pub driver_version: String,
    pub api_version: String,
    #[allow(dead_code)]
    pub vendor_id: u32,
    #[allow(dead_code)]
    pub device_id: u32,
    pub is_discrete: bool,
    #[allow(dead_code)]
    pub is_shared: bool,
}

pub fn detect_local_gpus() -> Result<Vec<LocalGpuInfo>> {
    // Purge any stale Vulkan driver overrides to ensure real hardware is enumerated
    std::env::remove_var("VK_LOADER_DRIVERS_SELECT");
    std::env::remove_var("VK_LOADER_LAYERS_ENABLE");
    std::env::remove_var("VK_ADD_DRIVER_FILES");
    std::env::remove_var("VK_DRIVER_FILES");
    std::env::remove_var("VK_ICD_FILENAMES");
    std::env::remove_var("DXVK_FILTER_DEVICE_NAME");
    std::env::remove_var("GALLIUM_DRIVER");
    std::env::remove_var("MESA_LOADER_DRIVER_OVERRIDE");
    std::env::remove_var("MESA_VK_DEVICE_SELECT");

    let entry = unsafe { ash::Entry::load().context("Failed to load Vulkan loader")? };

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"vkremote-mesh")
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(c"vkremote")
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_2);

    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .context("Failed to create Vulkan instance")?
    };

    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .context("Failed to enumerate physical devices")?
    };

    let mut gpus = Vec::new();

    for &pdevice in &physical_devices {
        let props = unsafe { instance.get_physical_device_properties(pdevice) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pdevice) };

        let raw_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
        let device_name = raw_name.to_str().unwrap_or("Unknown GPU").to_string();

        let (device_type, is_discrete) = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => ("Discrete GPU".to_string(), true),
            vk::PhysicalDeviceType::INTEGRATED_GPU => ("Integrated GPU".to_string(), false),
            vk::PhysicalDeviceType::CPU => ("CPU Fallback".to_string(), false),
            vk::PhysicalDeviceType::VIRTUAL_GPU => ("Virtual GPU".to_string(), false),
            _ => ("Other".to_string(), false),
        };

        // Find device local heap size
        let mut vram_bytes: u64 = 0;
        for i in 0..mem_props.memory_heap_count as usize {
            let heap = mem_props.memory_heaps[i];
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                if heap.size > vram_bytes {
                    vram_bytes = heap.size;
                }
            }
        }
        if vram_bytes == 0 && mem_props.memory_heap_count > 0 {
            vram_bytes = mem_props.memory_heaps[0].size;
        }

        let vram_mb = vram_bytes / (1024 * 1024);

        let api_version = format!(
            "{}.{}.{}",
            ash::vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version),
            vk::api_version_patch(props.api_version)
        );

        let driver_version = format!(
            "{}.{}.{}",
            (props.driver_version >> 22) & 0x3ff,
            (props.driver_version >> 12) & 0x3ff,
            props.driver_version & 0xfff
        );

        gpus.push(LocalGpuInfo {
            name: device_name,
            device_type,
            vram_mb,
            driver_version,
            api_version,
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            is_discrete,
            is_shared: is_discrete, // discrete GPUs shared by default
        });
    }

    // Prioritize discrete GPUs first
    gpus.sort_by_key(|g| !g.is_discrete);

    unsafe { instance.destroy_instance(None) };

    Ok(gpus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect() {
        let res = detect_local_gpus();
        println!("DETECT RESULT: {:?}", res);
        assert!(res.is_ok());
    }
}
