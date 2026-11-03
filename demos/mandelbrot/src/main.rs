use anyhow::{anyhow, Context, Result};
use ash::vk;
use std::ffi::CStr;
use std::time::Instant;

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..mem_props.memory_type_count).find(|&i| {
        (type_filter & (1 << i)) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(properties)
    })
}

fn main() -> Result<()> {
    println!("=== vkremote Mandelbrot Fractal Compute Demo ===");

    let width: usize = std::env::var("MANDEL_WIDTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(640);
    let height: usize = std::env::var("MANDEL_HEIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(360);
    let max_iterations: u32 = std::env::var("MANDEL_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);

    let total_pixels = width * height;
    let buffer_size = (total_pixels * std::mem::size_of::<u32>()) as vk::DeviceSize;

    println!(
        "Rendering resolution: {}x{} ({} pixels, max_iters: {})",
        width, height, total_pixels, max_iterations
    );

    // SAFETY: Loading dynamic Vulkan loader library
    let entry = unsafe { ash::Entry::load().context("Failed to load Vulkan loader")? };

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"vkremote-mandelbrot")
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(c"vkremote")
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_2);

    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    // SAFETY: Creating Vulkan instance
    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .context("Failed to create instance")?
    };

    // SAFETY: instance is valid
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    if physical_devices.is_empty() {
        return Err(anyhow!("No physical devices found"));
    }

    let mut selected_device = None;
    let mut selected_queue_family = None;

    for &pdevice in &physical_devices {
        // SAFETY: instance and pdevice are valid
        let props = unsafe { instance.get_physical_device_properties(pdevice) };
        let q_props = unsafe { instance.get_physical_device_queue_family_properties(pdevice) };
        let raw_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
        let name = raw_name.to_str().unwrap_or("");

        for (q_idx, q) in q_props.iter().enumerate() {
            if q.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && (name.contains("(Remote)") || selected_device.is_none())
            {
                selected_device = Some(pdevice);
                selected_queue_family = Some(q_idx as u32);
                if name.contains("(Remote)") {
                    break;
                }
            }
        }
    }

    let pdevice = selected_device.ok_or_else(|| anyhow!("No device with compute queue found"))?;
    let queue_family_index = selected_queue_family.unwrap();

    // SAFETY: querying device properties
    let dev_props = unsafe { instance.get_physical_device_properties(pdevice) };
    let dev_name = unsafe { CStr::from_ptr(dev_props.device_name.as_ptr()) }
        .to_str()
        .unwrap_or("Unknown");
    println!("Selected Physical Device: {}", dev_name);

    // Create logical device
    let queue_priorities = [1.0f32];
    let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities)];

    let device_create_info =
        vk::DeviceCreateInfo::default().queue_create_infos(&queue_create_infos);

    // SAFETY: creating logical device
    let device = unsafe {
        instance
            .create_device(pdevice, &device_create_info, None)
            .context("Failed to create device")?
    };

    // SAFETY: getting queue
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
    // SAFETY: querying memory properties
    let mem_props = unsafe { instance.get_physical_device_memory_properties(pdevice) };

    // Create output buffer
    let buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    // SAFETY: creating buffer
    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
    // SAFETY: querying buffer memory requirements
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

    let mem_type_index = find_memory_type(
        &mem_props,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| anyhow!("Suitable host visible memory type not found"))?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size.max(buffer_size))
        .memory_type_index(mem_type_index);

    // SAFETY: allocating device memory
    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    // SAFETY: binding buffer memory
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

    // Load SPIR-V shader
    let spv_bytes = include_bytes!("../shaders/mandelbrot.spv");
    let spv_u32 = ash::util::read_spv(&mut std::io::Cursor::new(spv_bytes))
        .context("Failed to read SPIR-V")?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_u32);
    // SAFETY: creating shader module
    let shader_module = unsafe { device.create_shader_module(&shader_info, None)? };

    // Descriptor set layout: binding 0 is storage buffer
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: creating descriptor set layout
    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

    // Push constant range: uint width, uint height, uint max_iterations -> 12 bytes
    let push_constant_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(12)];

    let set_layouts = [descriptor_set_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_constant_ranges);
    // SAFETY: creating pipeline layout
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(c"main");

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(pipeline_layout);
    // SAFETY: creating compute pipeline
    let pipeline = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_, err)| anyhow!("Pipeline creation failed: {:?}", err))?[0]
    };

    // Descriptor pool & set
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    // SAFETY: creating descriptor pool
    let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&set_layouts);
    // SAFETY: allocating descriptor set
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };

    let buffer_info_desc = [vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(0)
        .range(buffer_size)];

    let writes = [vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&buffer_info_desc)];
    // SAFETY: updating descriptor sets
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    // Command pool and buffer
    let cmd_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    // SAFETY: creating command pool
    let command_pool = unsafe { device.create_command_pool(&cmd_pool_info, None)? };

    let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: allocating command buffer
    let command_buffer = unsafe { device.allocate_command_buffers(&cmd_alloc_info)?[0] };

    let push_data = [width as u32, height as u32, max_iterations];
    let push_bytes = unsafe {
        std::slice::from_raw_parts(
            push_data.as_ptr() as *const u8,
            std::mem::size_of_val(&push_data),
        )
    };

    let group_x = width.div_ceil(16) as u32;
    let group_y = height.div_ceil(16) as u32;

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    // SAFETY: recording command buffer
    unsafe {
        device.begin_command_buffer(command_buffer, &begin_info)?;
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push_bytes,
        );
        device.cmd_dispatch(command_buffer, group_x, group_y, 1);
        device.end_command_buffer(command_buffer)?;
    }

    // Fence & Submit
    let fence_info = vk::FenceCreateInfo::default();
    // SAFETY: creating fence
    let fence = unsafe { device.create_fence(&fence_info, None)? };
    let submit_info =
        vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

    println!(
        "Dispatching Mandelbrot compute (grid: {}x{}, local: 16x16)...",
        group_x, group_y
    );
    let start_time = Instant::now();

    // SAFETY: submitting queue and waiting for fence
    unsafe {
        device.queue_submit(queue, &[submit_info], fence)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;
    }

    let elapsed = start_time.elapsed();
    let elapsed_sec = elapsed.as_secs_f64();
    let mpixels_per_sec = (total_pixels as f64 / 1e6) / elapsed_sec;

    // Read back buffer
    // SAFETY: mapping memory
    let ptr = unsafe { device.map_memory(memory, 0, buffer_size, vk::MemoryMapFlags::empty())? }
        as *const u32;

    let mut non_zero_pixels = 0usize;
    for i in 0..total_pixels {
        // SAFETY: pointer valid for total_pixels u32 values
        let pixel = unsafe { *ptr.add(i) };
        if pixel != 0 {
            non_zero_pixels += 1;
        }
    }

    println!(
        "[MANDELBROT] Rendered in {:.3} ms ({:.2} MPix/s), non-zero pixels: {} / {}",
        elapsed_sec * 1000.0,
        mpixels_per_sec,
        non_zero_pixels,
        total_pixels
    );

    if non_zero_pixels == 0 {
        return Err(anyhow!("Rendered buffer is all zeros!"));
    }

    // Generate ASCII preview (64 columns x 20 rows)
    println!("\n=== ASCII Fractal Preview ===");
    let preview_cols = 64usize;
    let preview_rows = 20usize;
    let ascii_palette = b" .:-=+*#%@";

    for row in 0..preview_rows {
        let mut line = String::with_capacity(preview_cols);
        let y = (row * height) / preview_rows;
        for col in 0..preview_cols {
            let x = (col * width) / preview_cols;
            let pixel = unsafe { *ptr.add(y * width + x) };
            let r = (pixel & 0xFF) as usize;
            let g = ((pixel >> 8) & 0xFF) as usize;
            let b = ((pixel >> 16) & 0xFF) as usize;
            let brightness = (r + g + b) / 3;
            let char_idx = (brightness * (ascii_palette.len() - 1)) / 255;
            line.push(ascii_palette[char_idx] as char);
        }
        println!("{}", line);
    }
    println!("=============================\n");

    println!("[MANDELBROT] SUCCESS: Fractal successfully generated and verified!");

    // Clean up
    // SAFETY: destroying all resources
    unsafe {
        device.unmap_memory(memory);
        device.destroy_fence(fence, None);
        device.destroy_command_pool(command_pool, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
        device.destroy_shader_module(shader_module, None);
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    Ok(())
}
