use crate::abi::DeviceDispatchable;
use ash::vk::Handle;
use vkremote_wire::types::{Request, ResourceType, Response};

#[no_mangle]
pub unsafe extern "system" fn vkCreateBuffer(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::BufferCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_buffer: *mut ash::vk::Buffer,
) -> ash::vk::Result {
    if device.is_null() || p_create_info.is_null() || p_buffer.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let create_info = &*p_create_info;
    let size = create_info.size;
    let usage = create_info.usage.as_raw();

    let buffer_id = dev.client.next_id();
    dev.client.register_buffer(buffer_id, size, usage);

    let req = Request::CreateBuffer {
        buffer_id,
        size,
        usage,
    };
    match dev.client.send_request(req) {
        Ok(Response::Success) => {}
        Ok(Response::Error { code, message }) => {
            dev.client.unregister_buffer(buffer_id);
            tracing::error!("CreateBuffer failed: {} ({})", message, code);
            return ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
        }
        other => {
            dev.client.unregister_buffer(buffer_id);
            tracing::error!("CreateBuffer transport/protocol failure: {:?}", other);
            return ash::vk::Result::ERROR_DEVICE_LOST;
        }
    }

    *p_buffer = ash::vk::Buffer::from_raw(buffer_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyBuffer(
    device: ash::vk::Device,
    buffer: ash::vk::Buffer,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || buffer.is_null() {
        return;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let buffer_id = buffer.as_raw();

    dev.client.unregister_buffer(buffer_id);
    let _ = dev.client.send_request(Request::DestroyResource {
        handle_id: buffer_id,
        resource_type: ResourceType::Buffer,
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkGetBufferMemoryRequirements(
    device: ash::vk::Device,
    buffer: ash::vk::Buffer,
    p_memory_requirements: *mut ash::vk::MemoryRequirements,
) {
    if device.is_null() || p_memory_requirements.is_null() {
        return;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let reqs = &mut *p_memory_requirements;

    let alignment = 256u64;
    let size = dev.client.get_buffer_size(buffer.as_raw()).unwrap_or(64);

    reqs.alignment = alignment;
    reqs.size = ((size + alignment - 1) / alignment) * alignment;
    // 0x3E masks out Type #0 (flags: NONE) on NVIDIA GPUs to prevent MMU page faults
    // while providing valid bits for DEVICE_LOCAL and HOST_VISIBLE memory heaps.
    reqs.memory_type_bits = 0x3E;
}

#[no_mangle]
pub unsafe extern "system" fn vkGetBufferMemoryRequirements2(
    device: ash::vk::Device,
    p_info: *const ash::vk::BufferMemoryRequirementsInfo2,
    p_memory_requirements: *mut ash::vk::MemoryRequirements2,
) {
    if p_info.is_null() || p_memory_requirements.is_null() {
        return;
    }
    // SAFETY: caller provided valid pointers
    let info = &*p_info;
    let reqs2 = &mut *p_memory_requirements;
    vkGetBufferMemoryRequirements(device, info.buffer, &mut reqs2.memory_requirements);

    let mut next = reqs2.p_next as *mut ash::vk::BaseOutStructure;
    while !next.is_null() {
        match (*next).s_type {
            ash::vk::StructureType::MEMORY_DEDICATED_REQUIREMENTS => {
                let dedicated = &mut *(next as *mut ash::vk::MemoryDedicatedRequirements);
                dedicated.prefers_dedicated_allocation = ash::vk::FALSE;
                dedicated.requires_dedicated_allocation = ash::vk::FALSE;
            }
            _ => {}
        }
        next = (*next).p_next;
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetBufferMemoryRequirements2KHR(
    device: ash::vk::Device,
    p_info: *const ash::vk::BufferMemoryRequirementsInfo2,
    p_memory_requirements: *mut ash::vk::MemoryRequirements2,
) {
    vkGetBufferMemoryRequirements2(device, p_info, p_memory_requirements);
}

#[no_mangle]
pub unsafe extern "system" fn vkAllocateMemory(
    device: ash::vk::Device,
    p_allocate_info: *const ash::vk::MemoryAllocateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_memory: *mut ash::vk::DeviceMemory,
) -> ash::vk::Result {
    if device.is_null() || p_allocate_info.is_null() || p_memory.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let alloc_info = &*p_allocate_info;
    let size = alloc_info.allocation_size;
    let memory_type_index = alloc_info.memory_type_index;

    let phys = &*dev.physical_device;
    let is_host_visible = phys
        .snapshot
        .memory_properties
        .memory_types
        .get(memory_type_index as usize)
        .map(|mt| (mt.property_flags & ash::vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()) != 0)
        .unwrap_or(true);

    let memory_id = dev.client.next_id();
    dev.client
        .register_allocation(memory_id, size, memory_type_index, is_host_visible);

    let req = Request::AllocateMemory {
        memory_id,
        size,
        memory_type_index,
    };
    match dev.client.send_request(req) {
        Ok(Response::Success) => {}
        Ok(Response::Error { code, message }) => {
            dev.client.unregister_allocation(memory_id);
            eprintln!(
                "[vkremote-icd] AllocateMemory server error: code={} message={}",
                code, message
            );
            return ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
        }
        Ok(other) => {
            dev.client.unregister_allocation(memory_id);
            eprintln!(
                "[vkremote-icd] AllocateMemory unexpected response: {:?}",
                other
            );
            return ash::vk::Result::ERROR_DEVICE_LOST;
        }
        Err(e) => {
            dev.client.unregister_allocation(memory_id);
            eprintln!("[vkremote-icd] AllocateMemory transport error: {}", e);
            tracing::error!("AllocateMemory request failed: {}", e);
            return ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }

    *p_memory = ash::vk::DeviceMemory::from_raw(memory_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkFreeMemory(
    device: ash::vk::Device,
    memory: ash::vk::DeviceMemory,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || memory.is_null() {
        return;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let memory_id = memory.as_raw();

    dev.client.unregister_allocation(memory_id);
    let _ = dev.client.send_request(Request::DestroyResource {
        handle_id: memory_id,
        resource_type: ResourceType::DeviceMemory,
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkBindBufferMemory(
    device: ash::vk::Device,
    buffer: ash::vk::Buffer,
    memory: ash::vk::DeviceMemory,
    memory_offset: ash::vk::DeviceSize,
) -> ash::vk::Result {
    if device.is_null() || buffer.is_null() || memory.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let buffer_id = buffer.as_raw();
    let memory_id = memory.as_raw();

    dev.client
        .bind_buffer_memory(buffer_id, memory_id, memory_offset);

    let req = Request::BindBufferMemory {
        buffer_id,
        memory_id,
        offset: memory_offset,
    };
    if let Err(e) = dev.client.send_request(req) {
        tracing::error!("BindBufferMemory request failed: {}", e);
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkBindBufferMemory2(
    device: ash::vk::Device,
    bind_info_count: u32,
    p_bind_infos: *const ash::vk::BindBufferMemoryInfo,
) -> ash::vk::Result {
    if device.is_null() || p_bind_infos.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    for i in 0..bind_info_count as usize {
        // SAFETY: caller provided bind_info_count elements
        let info = &*p_bind_infos.add(i);
        let res = vkBindBufferMemory(device, info.buffer, info.memory, info.memory_offset);
        if res != ash::vk::Result::SUCCESS {
            return res;
        }
    }

    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkBindBufferMemory2KHR(
    device: ash::vk::Device,
    bind_info_count: u32,
    p_bind_infos: *const ash::vk::BindBufferMemoryInfo,
) -> ash::vk::Result {
    vkBindBufferMemory2(device, bind_info_count, p_bind_infos)
}

#[no_mangle]
pub unsafe extern "system" fn vkMapMemory(
    device: ash::vk::Device,
    memory: ash::vk::DeviceMemory,
    offset: ash::vk::DeviceSize,
    size: ash::vk::DeviceSize,
    _flags: ash::vk::MemoryMapFlags,
    pp_data: *mut *mut std::os::raw::c_void,
) -> ash::vk::Result {
    if device.is_null() || memory.is_null() || pp_data.is_null() {
        return ash::vk::Result::ERROR_MEMORY_MAP_FAILED;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    match dev.client.map_memory(memory.as_raw(), offset, size) {
        Ok(ptr) => {
            *pp_data = ptr as *mut std::os::raw::c_void;
            ash::vk::Result::SUCCESS
        }
        Err(e) => e,
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkUnmapMemory(
    device: ash::vk::Device,
    memory: ash::vk::DeviceMemory,
) {
    if device.is_null() || memory.is_null() {
        return;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    dev.client.unmap_memory(memory.as_raw());
}

#[no_mangle]
pub unsafe extern "system" fn vkFlushMappedMemoryRanges(
    _device: ash::vk::Device,
    _memory_range_count: u32,
    _p_memory_ranges: *const ash::vk::MappedMemoryRange,
) -> ash::vk::Result {
    // Memory sync is executed automatically on queue submit / fence wait
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkInvalidateMappedMemoryRanges(
    device: ash::vk::Device,
    memory_range_count: u32,
    p_memory_ranges: *const ash::vk::MappedMemoryRange,
) -> ash::vk::Result {
    if device.is_null() || (memory_range_count > 0 && p_memory_ranges.is_null()) {
        return ash::vk::Result::ERROR_MEMORY_MAP_FAILED;
    }

    if memory_range_count == 0 {
        return ash::vk::Result::SUCCESS;
    }

    let dev = DeviceDispatchable::from_vk(device);
    let mut allocation_ids = Vec::with_capacity(memory_range_count as usize);
    for index in 0..memory_range_count as usize {
        let range = &*p_memory_ranges.add(index);
        let id = range.memory.as_raw();
        if id != 0 && !allocation_ids.contains(&id) {
            allocation_ids.push(id);
        }
    }

    match dev.client.sync_remote_allocations_to_shadow(allocation_ids) {
        Ok(()) => ash::vk::Result::SUCCESS,
        Err(error) => {
            tracing::error!("InvalidateMappedMemoryRanges network error: {error}");
            ash::vk::Result::ERROR_DEVICE_LOST
        }
    }
}
