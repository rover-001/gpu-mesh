use crate::abi::DeviceDispatchable;
use ash::vk::Handle;
use vkremote_wire::types::{Request, ResourceType, Response};

#[no_mangle]
pub unsafe extern "system" fn vkCreateFence(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::FenceCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_fence: *mut ash::vk::Fence,
) -> ash::vk::Result {
    if device.is_null() || p_create_info.is_null() || p_fence.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let info = &*p_create_info;
    let signaled =
        (info.flags & ash::vk::FenceCreateFlags::SIGNALED) != ash::vk::FenceCreateFlags::empty();

    let fence_id = dev.client.next_id();
    let req = Request::CreateFence { fence_id, signaled };

    if let Err(e) = dev.client.send_request(req) {
        tracing::error!("CreateFence request failed: {}", e);
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    *p_fence = ash::vk::Fence::from_raw(fence_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyFence(
    device: ash::vk::Device,
    fence: ash::vk::Fence,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || fence.is_null() {
        return;
    }
    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let _ = dev.client.send_request(Request::DestroyResource {
        handle_id: fence.as_raw(),
        resource_type: ResourceType::Fence,
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateSemaphore(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::SemaphoreCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_semaphore: *mut ash::vk::Semaphore,
) -> ash::vk::Result {
    if device.is_null() || p_semaphore.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let sem_id = dev.client.next_id();
    if !p_create_info.is_null() {
        let mut next = (*p_create_info).p_next as *const ash::vk::BaseInStructure;
        while !next.is_null() {
            if (*next).s_type == ash::vk::StructureType::SEMAPHORE_TYPE_CREATE_INFO {
                let info = &*(next as *const ash::vk::SemaphoreTypeCreateInfo);
                if info.semaphore_type == ash::vk::SemaphoreType::TIMELINE {
                    dev.client.register_timeline(sem_id, info.initial_value);
                }
            }
            next = (*next).p_next;
        }
    }
    *p_semaphore = ash::vk::Semaphore::from_raw(sem_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroySemaphore(
    device: ash::vk::Device,
    semaphore: ash::vk::Semaphore,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if !device.is_null() {
        DeviceDispatchable::from_vk(device)
            .client
            .remove_timeline(semaphore.as_raw());
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkWaitSemaphores(
    device: ash::vk::Device,
    p_wait_info: *const ash::vk::SemaphoreWaitInfo,
    timeout: u64,
) -> ash::vk::Result {
    if device.is_null() || p_wait_info.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let info = &*p_wait_info;
    if info.semaphore_count > 0 && (info.p_semaphores.is_null() || info.p_values.is_null()) {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let waits: Vec<_> = (0..info.semaphore_count as usize)
        .map(|i| ((*info.p_semaphores.add(i)).as_raw(), *info.p_values.add(i)))
        .collect();
    DeviceDispatchable::from_vk(device).client.wait_timelines(
        &waits,
        info.flags.contains(ash::vk::SemaphoreWaitFlags::ANY),
        timeout,
    )
}

#[no_mangle]
pub unsafe extern "system" fn vkWaitSemaphoresKHR(
    device: ash::vk::Device,
    p_wait_info: *const ash::vk::SemaphoreWaitInfo,
    timeout: u64,
) -> ash::vk::Result {
    vkWaitSemaphores(device, p_wait_info, timeout)
}

#[no_mangle]
pub unsafe extern "system" fn vkSignalSemaphore(
    device: ash::vk::Device,
    p_signal_info: *const ash::vk::SemaphoreSignalInfo,
) -> ash::vk::Result {
    if device.is_null() || p_signal_info.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let info = &*p_signal_info;
    DeviceDispatchable::from_vk(device)
        .client
        .signal_timeline(info.semaphore.as_raw(), info.value)
}

#[no_mangle]
pub unsafe extern "system" fn vkSignalSemaphoreKHR(
    device: ash::vk::Device,
    p_signal_info: *const ash::vk::SemaphoreSignalInfo,
) -> ash::vk::Result {
    vkSignalSemaphore(device, p_signal_info)
}

#[no_mangle]
pub unsafe extern "system" fn vkGetSemaphoreCounterValue(
    device: ash::vk::Device,
    semaphore: ash::vk::Semaphore,
    p_value: *mut u64,
) -> ash::vk::Result {
    if device.is_null() || p_value.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    match DeviceDispatchable::from_vk(device)
        .client
        .timeline_value(semaphore.as_raw())
    {
        Some(value) => {
            *p_value = value;
            ash::vk::Result::SUCCESS
        }
        None => ash::vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetSemaphoreCounterValueKHR(
    device: ash::vk::Device,
    semaphore: ash::vk::Semaphore,
    p_value: *mut u64,
) -> ash::vk::Result {
    vkGetSemaphoreCounterValue(device, semaphore, p_value)
}

#[no_mangle]
pub unsafe extern "system" fn vkResetQueryPool(
    _device: ash::vk::Device,
    _query_pool: ash::vk::QueryPool,
    _first_query: u32,
    _query_count: u32,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkResetQueryPoolEXT(
    device: ash::vk::Device,
    query_pool: ash::vk::QueryPool,
    first_query: u32,
    query_count: u32,
) {
    vkResetQueryPool(device, query_pool, first_query, query_count)
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateQueryPool(
    device: ash::vk::Device,
    _p_create_info: *const ash::vk::QueryPoolCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_query_pool: *mut ash::vk::QueryPool,
) -> ash::vk::Result {
    if device.is_null() || p_query_pool.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let id = dev.client.next_id();
    *p_query_pool = ash::vk::QueryPool::from_raw(id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyQueryPool(
    _device: ash::vk::Device,
    _query_pool: ash::vk::QueryPool,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkGetQueryPoolResults(
    _device: ash::vk::Device,
    _query_pool: ash::vk::QueryPool,
    _first_query: u32,
    _query_count: u32,
    data_size: usize,
    p_data: *mut std::os::raw::c_void,
    _stride: ash::vk::DeviceSize,
    _flags: ash::vk::QueryResultFlags,
) -> ash::vk::Result {
    if !p_data.is_null() && data_size > 0 {
        std::ptr::write_bytes(p_data as *mut u8, 0, data_size);
    }
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdResetQueryPool(
    _command_buffer: ash::vk::CommandBuffer,
    _query_pool: ash::vk::QueryPool,
    _first_query: u32,
    _query_count: u32,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdWriteTimestamp(
    _command_buffer: ash::vk::CommandBuffer,
    _pipeline_stage: ash::vk::PipelineStageFlags,
    _query_pool: ash::vk::QueryPool,
    _query: u32,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdWriteTimestamp2(
    _command_buffer: ash::vk::CommandBuffer,
    _stage: ash::vk::PipelineStageFlags2,
    _query_pool: ash::vk::QueryPool,
    _query: u32,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdWriteTimestamp2KHR(
    command_buffer: ash::vk::CommandBuffer,
    stage: ash::vk::PipelineStageFlags2,
    query_pool: ash::vk::QueryPool,
    query: u32,
) {
    vkCmdWriteTimestamp2(command_buffer, stage, query_pool, query);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetBufferDeviceAddress(
    _device: ash::vk::Device,
    _p_info: *const ash::vk::BufferDeviceAddressInfo,
) -> ash::vk::DeviceAddress {
    0
}

#[no_mangle]
pub unsafe extern "system" fn vkGetBufferDeviceAddressKHR(
    device: ash::vk::Device,
    p_info: *const ash::vk::BufferDeviceAddressInfo,
) -> ash::vk::DeviceAddress {
    vkGetBufferDeviceAddress(device, p_info)
}

#[no_mangle]
pub unsafe extern "system" fn vkGetBufferDeviceAddressEXT(
    device: ash::vk::Device,
    p_info: *const ash::vk::BufferDeviceAddressInfo,
) -> ash::vk::DeviceAddress {
    vkGetBufferDeviceAddress(device, p_info)
}

#[no_mangle]
pub unsafe extern "system" fn vkGetBufferOpaqueCaptureAddress(
    _device: ash::vk::Device,
    _p_info: *const ash::vk::BufferDeviceAddressInfo,
) -> u64 {
    0
}

#[no_mangle]
pub unsafe extern "system" fn vkGetBufferOpaqueCaptureAddressKHR(
    device: ash::vk::Device,
    p_info: *const ash::vk::BufferDeviceAddressInfo,
) -> u64 {
    vkGetBufferOpaqueCaptureAddress(device, p_info)
}

#[no_mangle]
pub unsafe extern "system" fn vkGetDeviceMemoryOpaqueCaptureAddress(
    _device: ash::vk::Device,
    _p_info: *const ash::vk::DeviceMemoryOpaqueCaptureAddressInfo,
) -> u64 {
    0
}

#[no_mangle]
pub unsafe extern "system" fn vkGetDeviceMemoryOpaqueCaptureAddressKHR(
    device: ash::vk::Device,
    p_info: *const ash::vk::DeviceMemoryOpaqueCaptureAddressInfo,
) -> u64 {
    vkGetDeviceMemoryOpaqueCaptureAddress(device, p_info)
}

#[no_mangle]
pub unsafe extern "system" fn vkWaitForFences(
    device: ash::vk::Device,
    fence_count: u32,
    p_fences: *const ash::vk::Fence,
    wait_all: ash::vk::Bool32,
    timeout: u64,
) -> ash::vk::Result {
    if device.is_null() || p_fences.is_null() || fence_count == 0 {
        return ash::vk::Result::SUCCESS;
    }
    let _perf_span = crate::perf::Span::new(&crate::perf::FENCE_WAIT_US);

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let mut fence_ids = Vec::with_capacity(fence_count as usize);

    for i in 0..fence_count as usize {
        let f = *p_fences.add(i);
        fence_ids.push(f.as_raw());
    }

    let req = Request::WaitForFences {
        fence_ids,
        wait_all: wait_all != 0,
        timeout_ns: timeout,
    };

    match dev.client.send_request(req) {
        Ok(Response::MemoryData { diffs }) => {
            dev.client.apply_memory_diffs(&diffs);
            // Memory diffs piggybacked on fence signal — no extra round trip needed
            crate::client::end_token();
            ash::vk::Result::SUCCESS
        }
        Ok(Response::WaitResult {
            success: _,
            timeout: is_timeout,
        }) => {
            if is_timeout {
                ash::vk::Result::TIMEOUT
            } else {
                // P2: server should piggyback diffs in MemoryData, not WaitResult.
                // If we get a bare WaitResult the server is old — do the fallback pull.
                let _ = dev.client.sync_remote_memory_to_shadow();
                crate::client::end_token();
                ash::vk::Result::SUCCESS
            }
        }
        // A fence completion does not imply the application will read mapped
        // memory. Readback is performed only for an explicit invalidate call.
        Ok(Response::Success) => {
            crate::client::end_token();
            ash::vk::Result::SUCCESS
        }
        Ok(Response::Error { code, message }) => {
            tracing::error!("WaitForFences failed: {} ({})", message, code);
            ash::vk::Result::ERROR_DEVICE_LOST
        }
        Err(e) => {
            tracing::error!("WaitForFences network error: {}", e);
            ash::vk::Result::ERROR_DEVICE_LOST
        }
        _ => ash::vk::Result::SUCCESS,
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkResetFences(
    device: ash::vk::Device,
    fence_count: u32,
    p_fences: *const ash::vk::Fence,
) -> ash::vk::Result {
    if device.is_null() || p_fences.is_null() || fence_count == 0 {
        return ash::vk::Result::SUCCESS;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let mut fence_ids = Vec::with_capacity(fence_count as usize);

    for i in 0..fence_count as usize {
        let f = *p_fences.add(i);
        fence_ids.push(f.as_raw());
    }

    let req = Request::ResetFences { fence_ids };
    if let Err(e) = dev.client.send_request(req) {
        tracing::error!("ResetFences request failed: {}", e);
        return ash::vk::Result::ERROR_DEVICE_LOST;
    }

    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkGetFenceStatus(
    device: ash::vk::Device,
    fence: ash::vk::Fence,
) -> ash::vk::Result {
    if device.is_null() || fence.is_null() {
        return ash::vk::Result::ERROR_DEVICE_LOST;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let req = Request::GetFenceStatus {
        fence_id: fence.as_raw(),
    };

    match dev.client.send_request(req) {
        Ok(Response::FenceStatus { signaled }) => {
            if signaled {
                ash::vk::Result::SUCCESS
            } else {
                ash::vk::Result::NOT_READY
            }
        }
        _ => ash::vk::Result::NOT_READY,
    }
}
