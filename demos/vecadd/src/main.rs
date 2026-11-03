use anyhow::{anyhow, Context, Result};
use ash::vk;
use std::ffi::CStr;

const NUM_ELEMENTS: usize = 1024;
const BUFFER_SIZE: vk::DeviceSize = (NUM_ELEMENTS * std::mem::size_of::<f32>()) as vk::DeviceSize;

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
    println!("=== vkremote VecAdd Compute Demo ===");

    // SAFETY: Loading dynamic Vulkan loader library
    let entry = unsafe { ash::Entry::load().context("Failed to load Vulkan loader")? };

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"vkremote-vecadd")
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

    // Prefer remote device if present, or first with compute support
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

    // SAFETY: Creating logical device
    let device = unsafe {
        instance
            .create_device(pdevice, &device_create_info, None)
            .context("Failed to create device")?
    };

    // SAFETY: getting queue
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    // SAFETY: querying memory properties
    let mem_props = unsafe { instance.get_physical_device_memory_properties(pdevice) };

    // Helper to allocate and bind buffer
    let create_buffer_and_memory =
        |usage: vk::BufferUsageFlags| -> Result<(vk::Buffer, vk::DeviceMemory)> {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(BUFFER_SIZE)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            // SAFETY: creating buffer
            let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

            // SAFETY: querying requirements
            let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

            let mem_type_index = find_memory_type(
                &mem_props,
                reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or_else(|| anyhow!("Suitable host visible memory type not found"))?;

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size.max(BUFFER_SIZE))
                .memory_type_index(mem_type_index);

            // SAFETY: allocating device memory
            let memory = unsafe { device.allocate_memory(&alloc_info, None)? };

            // SAFETY: binding buffer memory
            unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

            Ok((buffer, memory))
        };

    let (buffer_a, memory_a) = create_buffer_and_memory(vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let (buffer_b, memory_b) = create_buffer_and_memory(vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let (buffer_c, memory_c) = create_buffer_and_memory(vk::BufferUsageFlags::STORAGE_BUFFER)?;

    // Map A and B, fill A[i] = i, B[i] = 2*i
    // SAFETY: mapping memory_a
    let ptr_a = unsafe { device.map_memory(memory_a, 0, BUFFER_SIZE, vk::MemoryMapFlags::empty())? }
        as *mut f32;
    // SAFETY: mapping memory_b
    let ptr_b = unsafe { device.map_memory(memory_b, 0, BUFFER_SIZE, vk::MemoryMapFlags::empty())? }
        as *mut f32;

    for i in 0..NUM_ELEMENTS {
        // SAFETY: pointers valid and sized for NUM_ELEMENTS floats
        unsafe {
            *ptr_a.add(i) = i as f32;
            *ptr_b.add(i) = 2.0 * (i as f32);
        }
    }

    // Flush or keep mapped
    let spv_bytes = include_bytes!("../shaders/vecadd.spv");
    let spv_u32 = ash::util::read_spv(&mut std::io::Cursor::new(spv_bytes))
        .context("Failed to read SPIR-V")?;

    let shader_info = vk::ShaderModuleCreateInfo::default().code(&spv_u32);
    // SAFETY: creating shader module
    let shader_module = unsafe { device.create_shader_module(&shader_info, None)? };

    // Descriptor set layout: 3 storage buffers at bindings 0, 1, 2
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    // SAFETY: creating descriptor set layout
    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

    // Pipeline layout
    let set_layouts = [descriptor_set_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    // SAFETY: creating pipeline layout
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

    // Compute pipeline
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

    // Descriptor Pool & Set
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(3)];
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

    // Update descriptor set
    let buffer_info_a = [vk::DescriptorBufferInfo::default()
        .buffer(buffer_a)
        .offset(0)
        .range(BUFFER_SIZE)];
    let buffer_info_b = [vk::DescriptorBufferInfo::default()
        .buffer(buffer_b)
        .offset(0)
        .range(BUFFER_SIZE)];
    let buffer_info_c = [vk::DescriptorBufferInfo::default()
        .buffer(buffer_c)
        .offset(0)
        .range(BUFFER_SIZE)];

    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_info_a),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_info_b),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_info_c),
    ];
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

    // Record command buffer
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
        // 1024 elements / 64 local_size_x = 16 workgroups
        device.cmd_dispatch(command_buffer, 16, 1, 1);
        device.end_command_buffer(command_buffer)?;
    }

    // Fence & Submit
    let fence_info = vk::FenceCreateInfo::default();
    // SAFETY: creating fence
    let fence = unsafe { device.create_fence(&fence_info, None)? };

    let submit_info =
        vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
    // SAFETY: submitting queue
    unsafe {
        device.queue_submit(queue, &[submit_info], fence)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;
    }

    // Map Buffer C and verify C[i] == 3*i
    // SAFETY: mapping memory_c
    let ptr_c = unsafe { device.map_memory(memory_c, 0, BUFFER_SIZE, vk::MemoryMapFlags::empty())? }
        as *const f32;

    for i in 0..NUM_ELEMENTS {
        let expected = 3.0 * (i as f32);
        // SAFETY: pointer ptr_c is valid for NUM_ELEMENTS floats
        let actual = unsafe { *ptr_c.add(i) };
        if (actual - expected).abs() > 1e-4 {
            return Err(anyhow!(
                "Mismatch at index {}: got {}, expected {}",
                i,
                actual,
                expected
            ));
        }
    }

    let c0 = unsafe { *ptr_c };
    let c_last = unsafe { *ptr_c.add(NUM_ELEMENTS - 1) };
    println!(
        "[VECADD] SUCCESS: C[0]={:.1}, C[{}]={:.1}",
        c0,
        NUM_ELEMENTS - 1,
        c_last
    );

    // Clean up
    // SAFETY: destroying resources in reverse order of creation
    unsafe {
        device.unmap_memory(memory_a);
        device.unmap_memory(memory_b);
        device.unmap_memory(memory_c);
        device.destroy_fence(fence, None);
        device.destroy_command_pool(command_pool, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
        device.destroy_shader_module(shader_module, None);
        device.destroy_buffer(buffer_a, None);
        device.free_memory(memory_a, None);
        device.destroy_buffer(buffer_b, None);
        device.free_memory(memory_b, None);
        device.destroy_buffer(buffer_c, None);
        device.free_memory(memory_c, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    Ok(())
}
