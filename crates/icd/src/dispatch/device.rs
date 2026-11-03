use crate::abi::{CommandBufferDispatchable, DeviceDispatchable, QueueDispatchable};
use ash::vk::Handle;
use vkremote_wire::types::{Request, ResourceType, Response};

#[no_mangle]
pub unsafe extern "system" fn vkDestroyDevice(
    device: ash::vk::Device,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() {
        return;
    }
    // SAFETY: device was created by vkCreateDevice
    let dev_ptr = device.as_raw() as usize as *mut DeviceDispatchable;
    let dev = &*dev_ptr;
    let dev_id = dev.id;
    let client = dev.client.clone();

    if !dev.queue.is_null() {
        let _ = Box::from_raw(dev.queue);
    }
    let _ = Box::from_raw(dev_ptr);

    let _ = client.send_request(Request::DestroyResource {
        handle_id: dev_id,
        resource_type: ResourceType::Device,
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkGetDeviceQueue(
    device: ash::vk::Device,
    _queue_family_index: u32,
    _queue_index: u32,
    p_queue: *mut ash::vk::Queue,
) {
    if device.is_null() || p_queue.is_null() {
        return;
    }
    // SAFETY: device was created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    *p_queue = QueueDispatchable::to_vk(dev.queue);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetDeviceQueue2(
    device: ash::vk::Device,
    _p_queue_info: *const ash::vk::DeviceQueueInfo2,
    p_queue: *mut ash::vk::Queue,
) {
    if device.is_null() || p_queue.is_null() {
        return;
    }
    let dev = DeviceDispatchable::from_vk(device);
    *p_queue = QueueDispatchable::to_vk(dev.queue);
}

#[no_mangle]
pub unsafe extern "system" fn vkQueueSubmit(
    queue: ash::vk::Queue,
    submit_count: u32,
    p_submits: *const ash::vk::SubmitInfo,
    fence: ash::vk::Fence,
) -> ash::vk::Result {
    if queue.is_null() {
        return ash::vk::Result::ERROR_DEVICE_LOST;
    }

    let _perf_span = crate::perf::Span::new(&crate::perf::SUBMIT_US);
    // SAFETY: queue is a valid handle created by vkCreateDevice
    let q = QueueDispatchable::from_vk(queue);
    // Preserve submission boundaries: a later submit may wait on a timeline
    // value signalled by an earlier submit in this same call.
    if submit_count > 1 && !p_submits.is_null() {
        for i in 0..submit_count {
            let result = vkQueueSubmit(
                queue,
                1,
                p_submits.add(i as usize),
                if i + 1 == submit_count {
                    fence
                } else {
                    ash::vk::Fence::null()
                },
            );
            if result != ash::vk::Result::SUCCESS {
                return result;
            }
        }
        return ash::vk::Result::SUCCESS;
    }
    let mut timeline_signals = Vec::new();
    if submit_count == 1 && !p_submits.is_null() {
        let submit = &*p_submits;
        let mut next = submit.p_next as *const ash::vk::BaseInStructure;
        while !next.is_null() {
            if (*next).s_type == ash::vk::StructureType::TIMELINE_SEMAPHORE_SUBMIT_INFO {
                let timeline = &*(next as *const ash::vk::TimelineSemaphoreSubmitInfo);
                for i in 0..submit.signal_semaphore_count as usize {
                    let id = (*submit.p_signal_semaphores.add(i)).as_raw();
                    if q.client.timeline_value(id).is_some() {
                        timeline_signals.push((id, *timeline.p_signal_semaphore_values.add(i)));
                    }
                }
            }
            next = (*next).p_next;
        }
    }
    let mut command_buffer_ids = Vec::new();

    if !p_submits.is_null() && submit_count > 0 {
        for i in 0..submit_count as usize {
            let submit = &*p_submits.add(i);
            if !submit.p_command_buffers.is_null() && submit.command_buffer_count > 0 {
                for j in 0..submit.command_buffer_count as usize {
                    let cb_handle = *submit.p_command_buffers.add(j);
                    if !cb_handle.is_null() {
                        let cb = CommandBufferDispatchable::from_vk(cb_handle);
                        command_buffer_ids.push(cb.id);
                    }
                }
            }
        }
    }

    let fence_id = if fence.is_null() {
        None
    } else {
        Some(fence.as_raw())
    };

    // Extract dirty memory chunks from mapped host-visible allocations
    let dirty_memory = q.client.collect_dirty_memory();
    // Keep each request bounded even when wgpu initializes several large arenas.
    // Individual diff chunks are 64 KiB; 32 chunks remain well below the wire cap.
    for batch in dirty_memory.chunks(32) {
        match q.client.send_request(Request::WriteMemoryRanges {
            diffs: batch.to_vec(),
        }) {
            Ok(Response::Success) => {}
            other => {
                tracing::error!("Memory upload failed: {:?}", other);
                return ash::vk::Result::ERROR_DEVICE_LOST;
            }
        }
    }

    let req = Request::QueueSubmit {
        queue_id: q.id,
        command_buffer_ids,
        fence_id,
        dirty_memory: Vec::new(),
    };

    match q.client.send_request(req) {
        Ok(Response::Success) => {
            if !timeline_signals.is_empty() {
                // Signal timeline semaphores after QueueSubmit confirms the server
                // has accepted and executed the command buffers. QueueWaitIdle is
                // now a no-op on the client side, so signal immediately — the server
                // already ran queue_wait_idle before returning Success.
                for (id, value) in timeline_signals {
                    let result = q.client.signal_timeline(id, value);
                    if result != ash::vk::Result::SUCCESS {
                        return result;
                    }
                }
            }
            ash::vk::Result::SUCCESS
        }
        Ok(Response::Error { code, message }) => {
            tracing::error!("QueueSubmit failed: {} ({})", message, code);
            ash::vk::Result::ERROR_UNKNOWN
        }
        Err(e) => {
            tracing::error!("QueueSubmit network error: {}", e);
            ash::vk::Result::ERROR_DEVICE_LOST
        }
        _ => ash::vk::Result::SUCCESS,
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkQueueWaitIdle(queue: ash::vk::Queue) -> ash::vk::Result {
    if queue.is_null() {
        return ash::vk::Result::ERROR_DEVICE_LOST;
    }
    // The server already performs queue_wait_idle inside QueuePresent and QueueSubmit
    // (for timeline signals). A separate client-initiated wait is a redundant RTT
    // that stalls the render thread without providing additional correctness.
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDeviceWaitIdle(device: ash::vk::Device) -> ash::vk::Result {
    if device.is_null() {
        return ash::vk::Result::ERROR_DEVICE_LOST;
    }
    // Client-side wait is a no-op: the server enforces GPU sync inside QueuePresent
    // and QueueSubmit (timeline signals). Each roundtrip here costs ~10-20 ms over
    // LAN and collapses frame rate from steady rendering to one frame per second.
    ash::vk::Result::SUCCESS
}
