use anyhow::{Context, Result};
use ash::vk;
use std::ffi::CStr;

fn main() -> Result<()> {
    println!("=== vkremote Vulkan Device Probe ===");

    // SAFETY: Loading dynamic Vulkan loader library
    let entry = unsafe { ash::Entry::load().context("Failed to load Vulkan loader library")? };

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"vkremote-probe")
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(c"vkremote")
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_2);

    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    // SAFETY: Creating Vulkan instance using validated create_info
    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .context("Failed to create Vulkan instance")?
    };

    // SAFETY: instance is valid
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .context("Failed to enumerate physical devices")?
    };

    println!("Found {} physical device(s):\n", physical_devices.len());

    for (idx, &pdevice) in physical_devices.iter().enumerate() {
        // SAFETY: instance and pdevice are valid
        let props = unsafe { instance.get_physical_device_properties(pdevice) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(pdevice) };

        let raw_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
        let device_name = raw_name.to_str().unwrap_or("Unknown Device");

        let device_type_str = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
            vk::PhysicalDeviceType::CPU => "CPU",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
            _ => "Other",
        };

        let api_version = format!(
            "{}.{}.{}",
            vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version),
            vk::api_version_patch(props.api_version)
        );

        println!("--------------------------------------------------");
        println!("Device [{}]: {}", idx, device_name);
        println!("  Type:           {}", device_type_str);
        println!("  API Version:    {}", api_version);
        println!(
            "  Driver Version: {:#x} ({}.{}.{})",
            props.driver_version,
            (props.driver_version >> 22) & 0x3ff,
            (props.driver_version >> 12) & 0x3ff,
            props.driver_version & 0xfff
        );
        println!("  Vendor / DevID: {:#06x} / {:#06x}", props.vendor_id, props.device_id);
        let feats = unsafe { instance.get_physical_device_features(pdevice) };
        println!("  Features:");
        println!("    shader_sampled_image_array_dynamic_indexing: {}", feats.shader_sampled_image_array_dynamic_indexing);
        println!("    sampler_anisotropy: {}", feats.sampler_anisotropy);
        println!("    texture_compression_bc: {}", feats.texture_compression_bc);
        println!("    multi_draw_indirect: {}", feats.multi_draw_indirect);
        println!("    draw_indirect_first_instance: {}", feats.draw_indirect_first_instance);
        println!("  Limits:");
        println!("    max_per_stage_descriptor_sampled_images: {}", props.limits.max_per_stage_descriptor_sampled_images);
        println!("    max_per_stage_resources: {}", props.limits.max_per_stage_resources);
        println!("    max_descriptor_set_sampled_images: {}", props.limits.max_descriptor_set_sampled_images);
        println!("    max_descriptor_set_samplers: {}", props.limits.max_descriptor_set_samplers);
        println!("    max_storage_buffer_range: {}", props.limits.max_storage_buffer_range);
        println!("    max_uniform_buffer_range: {}", props.limits.max_uniform_buffer_range);

        println!("  Memory Heaps ({}):", mem_props.memory_heap_count);
        for h in 0..mem_props.memory_heap_count as usize {
            let heap = mem_props.memory_heaps[h];
            let size_mb = heap.size as f64 / (1024.0 * 1024.0);
            let size_gb = size_mb / 1024.0;
            let mut flags = Vec::new();
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                flags.push("DEVICE_LOCAL");
            }
            if heap.flags.contains(vk::MemoryHeapFlags::MULTI_INSTANCE) {
                flags.push("MULTI_INSTANCE");
            }
            let flags_str = if flags.is_empty() {
                "NONE".to_string()
            } else {
                flags.join(" | ")
            };
            println!(
                "    Heap #{}: {:.2} MB ({:.2} GB) [{}]",
                h, size_mb, size_gb, flags_str
            );
        }

        println!("  Memory Types ({}):", mem_props.memory_type_count);
        for t in 0..mem_props.memory_type_count as usize {
            let mem_type = mem_props.memory_types[t];
            let mut flags = Vec::new();
            if mem_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            {
                flags.push("DEVICE_LOCAL");
            }
            if mem_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            {
                flags.push("HOST_VISIBLE");
            }
            if mem_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            {
                flags.push("HOST_COHERENT");
            }
            if mem_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_CACHED)
            {
                flags.push("HOST_CACHED");
            }
            if mem_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED)
            {
                flags.push("LAZILY_ALLOCATED");
            }
            let flags_str = if flags.is_empty() {
                "NONE".to_string()
            } else {
                flags.join(" | ")
            };
            println!(
                "    Type #{:2}: Heap #{} [{}]",
                t, mem_type.heap_index, flags_str
            );
        }

        println!("  Queue Families ({}):", queue_families.len());
        for (qf_idx, qf) in queue_families.iter().enumerate() {
            let mut flags = Vec::new();
            if qf.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                flags.push("GRAPHICS");
            }
            if qf.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                flags.push("COMPUTE");
            }
            if qf.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                flags.push("TRANSFER");
            }
            if qf.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
                flags.push("SPARSE_BINDING");
            }
            let flags_str = if flags.is_empty() {
                "NONE".to_string()
            } else {
                flags.join(" | ")
            };
            println!(
                "    Family #{}: count = {}, flags = [{}]",
                qf_idx, qf.queue_count, flags_str
            );
        }
    }

    // SAFETY: instance is destroyed at end of probe
    unsafe {
        instance.destroy_instance(None);
    }

    Ok(())
}
