use crate::abi::{CommandBufferDispatchable, DeviceDispatchable};
use ash::vk::Handle;
use vkremote_wire::types::{BufferCopyData, Request, ResourceType};

fn layers(value: ash::vk::ImageSubresourceLayers) -> [u32; 4] {
    [
        value.aspect_mask.as_raw(),
        value.mip_level,
        value.base_array_layer,
        value.layer_count,
    ]
}
fn offset3(value: ash::vk::Offset3D) -> [i32; 3] {
    [value.x, value.y, value.z]
}
fn extent3(value: ash::vk::Extent3D) -> [u32; 3] {
    [value.width, value.height, value.depth]
}
fn image_range(
    value: &ash::vk::ImageSubresourceRange,
) -> vkremote_wire::types::ImageSubresourceRange {
    vkremote_wire::types::ImageSubresourceRange {
        aspect_mask: value.aspect_mask.as_raw().into(),
        base_mip_level: value.base_mip_level,
        level_count: value.level_count,
        base_array_layer: value.base_array_layer,
        layer_count: value.layer_count,
    }
}
fn buffer_image_region(r: &ash::vk::BufferImageCopy) -> vkremote_wire::types::BufferImageCopyData {
    vkremote_wire::types::BufferImageCopyData {
        buffer_offset: r.buffer_offset,
        buffer_row_length: r.buffer_row_length,
        buffer_image_height: r.buffer_image_height,
        image_subresource: layers(r.image_subresource),
        image_offset: offset3(r.image_offset),
        image_extent: extent3(r.image_extent),
    }
}

#[inline]
fn record_or_send(cb: &CommandBufferDispatchable, request: Request) {
    if !cb.client.record_command(cb.id, request.clone()) {
        let _ = cb.client.send_request(request);
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateCommandPool(
    device: ash::vk::Device,
    _p_create_info: *const ash::vk::CommandPoolCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_command_pool: *mut ash::vk::CommandPool,
) -> ash::vk::Result {
    if device.is_null() || p_command_pool.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let pool_id = dev.client.next_id();

    let req = Request::CreateCommandPool { pool_id };
    if let Err(e) = dev.client.send_request(req) {
        tracing::error!("CreateCommandPool request failed: {}", e);
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    *p_command_pool = ash::vk::CommandPool::from_raw(pool_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyCommandPool(
    device: ash::vk::Device,
    command_pool: ash::vk::CommandPool,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || command_pool.is_null() {
        return;
    }
    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let _ = dev.client.send_request(Request::DestroyResource {
        handle_id: command_pool.as_raw(),
        resource_type: ResourceType::CommandPool,
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkAllocateCommandBuffers(
    device: ash::vk::Device,
    p_allocate_info: *const ash::vk::CommandBufferAllocateInfo,
    p_command_buffers: *mut ash::vk::CommandBuffer,
) -> ash::vk::Result {
    if device.is_null() || p_allocate_info.is_null() || p_command_buffers.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let dev_ptr = device.as_raw() as usize as *mut DeviceDispatchable;
    let info = &*p_allocate_info;
    let count = info.command_buffer_count as usize;

    let mut buffer_ids = Vec::with_capacity(count);

    for i in 0..count {
        let cb_id = dev.client.next_id();
        buffer_ids.push(cb_id);

        let cb = CommandBufferDispatchable::new(
            cb_id,
            dev_ptr,
            dev.client.clone(),
            info.command_pool.as_raw(),
        );
        let cb_ptr = Box::into_raw(cb);
        *p_command_buffers.add(i) = CommandBufferDispatchable::to_vk(cb_ptr);
    }

    let req = Request::AllocateCommandBuffers {
        pool_id: info.command_pool.as_raw(),
        buffer_ids,
    };

    if let Err(e) = dev.client.send_request(req) {
        tracing::error!("AllocateCommandBuffers request failed: {}", e);
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkFreeCommandBuffers(
    device: ash::vk::Device,
    _command_pool: ash::vk::CommandPool,
    command_buffer_count: u32,
    p_command_buffers: *const ash::vk::CommandBuffer,
) {
    if device.is_null() || p_command_buffers.is_null() || command_buffer_count == 0 {
        return;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);

    for i in 0..command_buffer_count as usize {
        let handle = *p_command_buffers.add(i);
        if !handle.is_null() {
            let cb_ptr = handle.as_raw() as usize as *mut CommandBufferDispatchable;
            let cb_id = (*cb_ptr).id;
            (*cb_ptr).client.discard_command_recording(cb_id);
            let _ = Box::from_raw(cb_ptr);
            let _ = dev.client.send_request(Request::DestroyResource {
                handle_id: cb_id,
                resource_type: ResourceType::CommandBuffer,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkResetCommandBuffer(
    _command_buffer: ash::vk::CommandBuffer,
    _flags: ash::vk::CommandBufferResetFlags,
) -> ash::vk::Result {
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkBeginCommandBuffer(
    command_buffer: ash::vk::CommandBuffer,
    _p_begin_info: *const ash::vk::CommandBufferBeginInfo,
) -> ash::vk::Result {
    if command_buffer.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: command_buffer is a valid handle created by vkAllocateCommandBuffers
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    cb.client.begin_command_recording(cb.id);
    cb.client
        .record_command(cb.id, Request::BeginCommandBuffer { buffer_id: cb.id });

    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkEndCommandBuffer(
    command_buffer: ash::vk::CommandBuffer,
) -> ash::vk::Result {
    if command_buffer.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: command_buffer is a valid handle created by vkAllocateCommandBuffers
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    cb.client
        .record_command(cb.id, Request::EndCommandBuffer { buffer_id: cb.id });

    if let Some(commands) = cb.client.finish_command_recording(cb.id) {
        if !commands.is_empty() {
            if let Err(e) = cb.client.send_request(Request::CommandBatch { commands }) {
                tracing::error!("Command batch request failed: {}", e);
                return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
            }
        }
    }

    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBindPipeline(
    command_buffer: ash::vk::CommandBuffer,
    pipeline_bind_point: ash::vk::PipelineBindPoint,
    pipeline: ash::vk::Pipeline,
) {
    if command_buffer.is_null() || pipeline.is_null() {
        return;
    }

    // SAFETY: command_buffer is a valid handle created by vkAllocateCommandBuffers
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let req = Request::CmdBindPipeline {
        buffer_id: cb.id,
        pipeline_bind_point: (pipeline_bind_point.as_raw() as u32).into(),
        pipeline_id: pipeline.as_raw(),
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBindDescriptorSets(
    command_buffer: ash::vk::CommandBuffer,
    pipeline_bind_point: ash::vk::PipelineBindPoint,
    layout: ash::vk::PipelineLayout,
    first_set: u32,
    descriptor_set_count: u32,
    p_descriptor_sets: *const ash::vk::DescriptorSet,
    dynamic_offset_count: u32,
    p_dynamic_offsets: *const u32,
) {
    if command_buffer.is_null() || p_descriptor_sets.is_null() || descriptor_set_count == 0 {
        return;
    }

    // SAFETY: command_buffer is a valid handle created by vkAllocateCommandBuffers
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let mut set_ids = Vec::with_capacity(descriptor_set_count as usize);

    for i in 0..descriptor_set_count as usize {
        set_ids.push((*p_descriptor_sets.add(i)).as_raw());
    }

    let mut dynamic_offsets = Vec::with_capacity(dynamic_offset_count as usize);
    if !p_dynamic_offsets.is_null() && dynamic_offset_count > 0 {
        for i in 0..dynamic_offset_count as usize {
            dynamic_offsets.push(*p_dynamic_offsets.add(i));
        }
    }

    let req = Request::CmdBindDescriptorSets {
        buffer_id: cb.id,
        pipeline_bind_point: (pipeline_bind_point.as_raw() as u32).into(),
        layout_id: layout.as_raw(),
        first_set,
        set_ids,
        dynamic_offsets,
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdPushConstants(
    command_buffer: ash::vk::CommandBuffer,
    layout: ash::vk::PipelineLayout,
    stage_flags: ash::vk::ShaderStageFlags,
    offset: u32,
    size: u32,
    p_values: *const std::os::raw::c_void,
) {
    if command_buffer.is_null() || p_values.is_null() || size == 0 {
        return;
    }

    // SAFETY: command_buffer is a valid handle created by vkAllocateCommandBuffers
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let values = std::slice::from_raw_parts(p_values as *const u8, size as usize).to_vec();

    let req = Request::CmdPushConstants {
        buffer_id: cb.id,
        layout_id: layout.as_raw(),
        stage_flags: stage_flags.as_raw(),
        offset,
        values,
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDispatch(
    command_buffer: ash::vk::CommandBuffer,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
) {
    if command_buffer.is_null() {
        return;
    }

    // SAFETY: command_buffer is a valid handle created by vkAllocateCommandBuffers
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let req = Request::CmdDispatch {
        buffer_id: cb.id,
        group_count_x,
        group_count_y,
        group_count_z,
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyBuffer(
    command_buffer: ash::vk::CommandBuffer,
    src_buffer: ash::vk::Buffer,
    dst_buffer: ash::vk::Buffer,
    region_count: u32,
    p_regions: *const ash::vk::BufferCopy,
) {
    if command_buffer.is_null() || p_regions.is_null() || region_count == 0 {
        return;
    }

    // SAFETY: command_buffer is a valid handle created by vkAllocateCommandBuffers
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let mut regions = Vec::with_capacity(region_count as usize);

    for i in 0..region_count as usize {
        let r = &*p_regions.add(i);
        regions.push(BufferCopyData {
            src_offset: r.src_offset,
            dst_offset: r.dst_offset,
            size: r.size,
        });
    }

    let req = Request::CmdCopyBuffer {
        buffer_id: cb.id,
        src_buffer_id: src_buffer.as_raw(),
        dst_buffer_id: dst_buffer.as_raw(),
        regions,
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdFillBuffer(
    command_buffer: ash::vk::CommandBuffer,
    dst_buffer: ash::vk::Buffer,
    dst_offset: ash::vk::DeviceSize,
    size: ash::vk::DeviceSize,
    data: u32,
) {
    if command_buffer.is_null() {
        return;
    }

    // SAFETY: command_buffer is a valid handle created by vkAllocateCommandBuffers
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let req = Request::CmdFillBuffer {
        buffer_id: cb.id,
        dst_buffer_id: dst_buffer.as_raw(),
        dst_offset,
        size,
        data,
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdPipelineBarrier(
    command_buffer: ash::vk::CommandBuffer,
    src_stage_mask: ash::vk::PipelineStageFlags,
    dst_stage_mask: ash::vk::PipelineStageFlags,
    dependency_flags: ash::vk::DependencyFlags,
    memory_barrier_count: u32,
    p_memory_barriers: *const ash::vk::MemoryBarrier,
    buffer_memory_barrier_count: u32,
    p_buffer_memory_barriers: *const ash::vk::BufferMemoryBarrier,
    image_memory_barrier_count: u32,
    p_image_memory_barriers: *const ash::vk::ImageMemoryBarrier,
) {
    if command_buffer.is_null() {
        return;
    }

    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let mut memory_barriers = Vec::new();
    if !p_memory_barriers.is_null() && memory_barrier_count > 0 {
        for i in 0..memory_barrier_count as usize {
            let b = &*p_memory_barriers.add(i);
            memory_barriers.push(vkremote_wire::types::MemoryBarrierData {
                src_access_mask: b.src_access_mask.as_raw(),
                dst_access_mask: b.dst_access_mask.as_raw(),
            });
        }
    }

    let mut buffer_memory_barriers = Vec::new();
    if !p_buffer_memory_barriers.is_null() && buffer_memory_barrier_count > 0 {
        for i in 0..buffer_memory_barrier_count as usize {
            let b = &*p_buffer_memory_barriers.add(i);
            buffer_memory_barriers.push(vkremote_wire::types::BufferMemoryBarrierData {
                src_access_mask: b.src_access_mask.as_raw(),
                dst_access_mask: b.dst_access_mask.as_raw(),
                src_queue_family_index: b.src_queue_family_index,
                dst_queue_family_index: b.dst_queue_family_index,
                buffer_id: b.buffer.as_raw(),
                offset: b.offset,
                size: b.size,
            });
        }
    }

    let mut image_memory_barriers = Vec::new();
    if !p_image_memory_barriers.is_null() && image_memory_barrier_count > 0 {
        for i in 0..image_memory_barrier_count as usize {
            let b = &*p_image_memory_barriers.add(i);
            image_memory_barriers.push(vkremote_wire::types::ImageMemoryBarrierData {
                src_access_mask: b.src_access_mask.as_raw(),
                dst_access_mask: b.dst_access_mask.as_raw(),
                old_layout: (b.old_layout.as_raw() as u32).into(),
                new_layout: (b.new_layout.as_raw() as u32).into(),
                src_queue_family_index: b.src_queue_family_index,
                dst_queue_family_index: b.dst_queue_family_index,
                image_id: b.image.as_raw(),
                subresource_range: vkremote_wire::types::ImageSubresourceRange {
                    aspect_mask: b.subresource_range.aspect_mask.as_raw().into(),
                    base_mip_level: b.subresource_range.base_mip_level,
                    level_count: b.subresource_range.level_count,
                    base_array_layer: b.subresource_range.base_array_layer,
                    layer_count: b.subresource_range.layer_count,
                },
            });
        }
    }

    let req = Request::CmdPipelineBarrier {
        buffer_id: cb.id,
        src_stage_mask: src_stage_mask.as_raw(),
        dst_stage_mask: dst_stage_mask.as_raw(),
        dependency_flags: dependency_flags.as_raw(),
        memory_barriers,
        buffer_memory_barriers,
        image_memory_barriers,
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBeginRenderPass(
    command_buffer: ash::vk::CommandBuffer,
    p_render_pass_begin: *const ash::vk::RenderPassBeginInfo,
    contents: ash::vk::SubpassContents,
) {
    if command_buffer.is_null() || p_render_pass_begin.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let begin = &*p_render_pass_begin;

    let render_area = vkremote_wire::types::Rect2D {
        offset: vkremote_wire::types::Offset2D {
            x: begin.render_area.offset.x,
            y: begin.render_area.offset.y,
        },
        extent: vkremote_wire::types::Extent2D {
            width: begin.render_area.extent.width,
            height: begin.render_area.extent.height,
        },
    };

    let mut clear_values = Vec::with_capacity(begin.clear_value_count as usize);
    if !begin.p_clear_values.is_null() && begin.clear_value_count > 0 {
        for i in 0..begin.clear_value_count as usize {
            let cv = &*begin.p_clear_values.add(i);
            let f = cv.color.float32;
            clear_values.push(vkremote_wire::types::ClearValueData::Color(
                vkremote_wire::types::ClearColorValue::Float([
                    f[0].into(),
                    f[1].into(),
                    f[2].into(),
                    f[3].into(),
                ]),
            ));
        }
    }

    let req = Request::CmdBeginRenderPass {
        buffer_id: cb.id,
        render_pass_id: begin.render_pass.as_raw(),
        framebuffer_id: begin.framebuffer.as_raw(),
        render_area,
        clear_values,
        contents: (contents.as_raw() as u32).into(),
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdEndRenderPass(command_buffer: ash::vk::CommandBuffer) {
    if command_buffer.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let req = Request::CmdEndRenderPass { buffer_id: cb.id };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBindVertexBuffers(
    command_buffer: ash::vk::CommandBuffer,
    first_binding: u32,
    binding_count: u32,
    p_buffers: *const ash::vk::Buffer,
    p_offsets: *const ash::vk::DeviceSize,
) {
    if command_buffer.is_null() || p_buffers.is_null() || p_offsets.is_null() || binding_count == 0
    {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let mut buffers = Vec::with_capacity(binding_count as usize);
    let mut offsets = Vec::with_capacity(binding_count as usize);
    for i in 0..binding_count as usize {
        buffers.push((*p_buffers.add(i)).as_raw());
        offsets.push(*p_offsets.add(i));
    }

    let req = Request::CmdBindVertexBuffers {
        buffer_id: cb.id,
        first_binding,
        buffer_ids: buffers,
        offsets,
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBindIndexBuffer(
    command_buffer: ash::vk::CommandBuffer,
    buffer: ash::vk::Buffer,
    offset: ash::vk::DeviceSize,
    index_type: ash::vk::IndexType,
) {
    if command_buffer.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let req = Request::CmdBindIndexBuffer {
        buffer_id: cb.id,
        index_buffer_id: buffer.as_raw(),
        offset,
        index_type: (index_type.as_raw() as u32).into(),
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetViewport(
    command_buffer: ash::vk::CommandBuffer,
    first_viewport: u32,
    viewport_count: u32,
    p_viewports: *const ash::vk::Viewport,
) {
    if command_buffer.is_null() || p_viewports.is_null() || viewport_count == 0 {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let mut viewports = Vec::with_capacity(viewport_count as usize);
    for i in 0..viewport_count as usize {
        let vp = &*p_viewports.add(i);
        viewports.push(vkremote_wire::types::Viewport::new(
            vp.x,
            vp.y,
            vp.width,
            vp.height,
            vp.min_depth,
            vp.max_depth,
        ));
    }

    let req = Request::CmdSetViewport {
        buffer_id: cb.id,
        first_viewport,
        viewports,
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetScissor(
    command_buffer: ash::vk::CommandBuffer,
    first_scissor: u32,
    scissor_count: u32,
    p_scissors: *const ash::vk::Rect2D,
) {
    if command_buffer.is_null() || p_scissors.is_null() || scissor_count == 0 {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let mut scissors = Vec::with_capacity(scissor_count as usize);
    for i in 0..scissor_count as usize {
        let sc = &*p_scissors.add(i);
        scissors.push(vkremote_wire::types::Rect2D {
            offset: vkremote_wire::types::Offset2D {
                x: sc.offset.x,
                y: sc.offset.y,
            },
            extent: vkremote_wire::types::Extent2D {
                width: sc.extent.width,
                height: sc.extent.height,
            },
        });
    }

    let req = Request::CmdSetScissor {
        buffer_id: cb.id,
        first_scissor,
        scissors,
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDraw(
    command_buffer: ash::vk::CommandBuffer,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    if command_buffer.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let req = Request::CmdDraw {
        buffer_id: cb.id,
        vertex_count,
        instance_count,
        first_vertex,
        first_instance,
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDrawIndexed(
    command_buffer: ash::vk::CommandBuffer,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
) {
    if command_buffer.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let req = Request::CmdDrawIndexed {
        buffer_id: cb.id,
        index_count,
        instance_count,
        first_index,
        vertex_offset,
        first_instance,
    };
    record_or_send(cb, req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDrawIndirectCount(
    command_buffer: ash::vk::CommandBuffer,
    buffer: ash::vk::Buffer,
    offset: ash::vk::DeviceSize,
    count_buffer: ash::vk::Buffer,
    count_buffer_offset: ash::vk::DeviceSize,
    max_draw_count: u32,
    stride: u32,
) {
    if command_buffer.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    record_or_send(
        cb,
        Request::CmdDrawIndirectCount {
            buffer_id: cb.id,
            indirect_buffer_id: buffer.as_raw(),
            offset,
            count_buffer_id: count_buffer.as_raw(),
            count_buffer_offset,
            max_draw_count,
            stride,
            indexed: false,
        },
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDrawIndirectCountKHR(
    command_buffer: ash::vk::CommandBuffer,
    buffer: ash::vk::Buffer,
    offset: ash::vk::DeviceSize,
    count_buffer: ash::vk::Buffer,
    count_buffer_offset: ash::vk::DeviceSize,
    max_draw_count: u32,
    stride: u32,
) {
    vkCmdDrawIndirectCount(
        command_buffer,
        buffer,
        offset,
        count_buffer,
        count_buffer_offset,
        max_draw_count,
        stride,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDrawIndexedIndirectCount(
    command_buffer: ash::vk::CommandBuffer,
    buffer: ash::vk::Buffer,
    offset: ash::vk::DeviceSize,
    count_buffer: ash::vk::Buffer,
    count_buffer_offset: ash::vk::DeviceSize,
    max_draw_count: u32,
    stride: u32,
) {
    if command_buffer.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    record_or_send(
        cb,
        Request::CmdDrawIndirectCount {
            buffer_id: cb.id,
            indirect_buffer_id: buffer.as_raw(),
            offset,
            count_buffer_id: count_buffer.as_raw(),
            count_buffer_offset,
            max_draw_count,
            stride,
            indexed: true,
        },
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDrawIndexedIndirectCountKHR(
    command_buffer: ash::vk::CommandBuffer,
    buffer: ash::vk::Buffer,
    offset: ash::vk::DeviceSize,
    count_buffer: ash::vk::Buffer,
    count_buffer_offset: ash::vk::DeviceSize,
    max_draw_count: u32,
    stride: u32,
) {
    vkCmdDrawIndexedIndirectCount(
        command_buffer,
        buffer,
        offset,
        count_buffer,
        count_buffer_offset,
        max_draw_count,
        stride,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBeginRenderPass2(
    command_buffer: ash::vk::CommandBuffer,
    p_render_pass_begin: *const ash::vk::RenderPassBeginInfo,
    _p_subpass_begin_info: *const ash::vk::SubpassBeginInfo,
) {
    if !p_render_pass_begin.is_null() {
        vkCmdBeginRenderPass(
            command_buffer,
            p_render_pass_begin,
            ash::vk::SubpassContents::INLINE,
        );
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBeginRenderPass2KHR(
    command_buffer: ash::vk::CommandBuffer,
    p_render_pass_begin: *const ash::vk::RenderPassBeginInfo,
    p_subpass_begin_info: *const ash::vk::SubpassBeginInfo,
) {
    vkCmdBeginRenderPass2(command_buffer, p_render_pass_begin, p_subpass_begin_info);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdNextSubpass2(
    _command_buffer: ash::vk::CommandBuffer,
    _p_subpass_begin_info: *const ash::vk::SubpassBeginInfo,
    _p_subpass_end_info: *const ash::vk::SubpassEndInfo,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdNextSubpass2KHR(
    command_buffer: ash::vk::CommandBuffer,
    p_subpass_begin_info: *const ash::vk::SubpassBeginInfo,
    p_subpass_end_info: *const ash::vk::SubpassEndInfo,
) {
    vkCmdNextSubpass2(command_buffer, p_subpass_begin_info, p_subpass_end_info);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdEndRenderPass2(
    command_buffer: ash::vk::CommandBuffer,
    _p_subpass_end_info: *const ash::vk::SubpassEndInfo,
) {
    vkCmdEndRenderPass(command_buffer);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdEndRenderPass2KHR(
    command_buffer: ash::vk::CommandBuffer,
    p_subpass_end_info: *const ash::vk::SubpassEndInfo,
) {
    vkCmdEndRenderPass2(command_buffer, p_subpass_end_info);
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyBufferToImage(
    command_buffer: ash::vk::CommandBuffer,
    src_buffer: ash::vk::Buffer,
    dst_image: ash::vk::Image,
    dst_image_layout: ash::vk::ImageLayout,
    region_count: u32,
    p_regions: *const ash::vk::BufferImageCopy,
) {
    if command_buffer.is_null() {
        return;
    }
    if p_regions.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let regions = std::slice::from_raw_parts(p_regions, region_count as usize)
        .iter()
        .map(buffer_image_region)
        .collect();
    record_or_send(
        cb,
        Request::CmdCopyBufferToImage {
            buffer_id: cb.id,
            src_buffer_id: src_buffer.as_raw(),
            dst_image_id: dst_image.as_raw(),
            dst_layout: dst_image_layout.as_raw(),
            regions,
        },
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyImageToBuffer(
    command_buffer: ash::vk::CommandBuffer,
    src_image: ash::vk::Image,
    src_image_layout: ash::vk::ImageLayout,
    dst_buffer: ash::vk::Buffer,
    region_count: u32,
    p_regions: *const ash::vk::BufferImageCopy,
) {
    if command_buffer.is_null() {
        return;
    }
    if p_regions.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let regions = std::slice::from_raw_parts(p_regions, region_count as usize)
        .iter()
        .map(buffer_image_region)
        .collect();
    record_or_send(
        cb,
        Request::CmdCopyImageToBuffer {
            buffer_id: cb.id,
            src_image_id: src_image.as_raw(),
            src_layout: src_image_layout.as_raw(),
            dst_buffer_id: dst_buffer.as_raw(),
            regions,
        },
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyImage(
    command_buffer: ash::vk::CommandBuffer,
    src_image: ash::vk::Image,
    src_image_layout: ash::vk::ImageLayout,
    dst_image: ash::vk::Image,
    dst_image_layout: ash::vk::ImageLayout,
    region_count: u32,
    p_regions: *const ash::vk::ImageCopy,
) {
    if command_buffer.is_null() {
        return;
    }
    if p_regions.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let regions = std::slice::from_raw_parts(p_regions, region_count as usize)
        .iter()
        .map(|r| vkremote_wire::types::ImageCopyRegionData {
            src_subresource: layers(r.src_subresource),
            src_offset: offset3(r.src_offset),
            dst_subresource: layers(r.dst_subresource),
            dst_offset: offset3(r.dst_offset),
            extent: extent3(r.extent),
        })
        .collect();
    record_or_send(
        cb,
        Request::CmdCopyImage {
            buffer_id: cb.id,
            src_image_id: src_image.as_raw(),
            src_layout: src_image_layout.as_raw(),
            dst_image_id: dst_image.as_raw(),
            dst_layout: dst_image_layout.as_raw(),
            regions,
        },
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBlitImage(
    command_buffer: ash::vk::CommandBuffer,
    src_image: ash::vk::Image,
    src_image_layout: ash::vk::ImageLayout,
    dst_image: ash::vk::Image,
    dst_image_layout: ash::vk::ImageLayout,
    region_count: u32,
    p_regions: *const ash::vk::ImageBlit,
    filter: ash::vk::Filter,
) {
    if command_buffer.is_null() {
        return;
    }
    if p_regions.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let regions = std::slice::from_raw_parts(p_regions, region_count as usize)
        .iter()
        .map(|r| vkremote_wire::types::ImageBlitData {
            src_subresource: layers(r.src_subresource),
            src_offsets: r.src_offsets.map(offset3),
            dst_subresource: layers(r.dst_subresource),
            dst_offsets: r.dst_offsets.map(offset3),
        })
        .collect();
    record_or_send(
        cb,
        Request::CmdBlitImage {
            buffer_id: cb.id,
            src_image_id: src_image.as_raw(),
            src_layout: src_image_layout.as_raw(),
            dst_image_id: dst_image.as_raw(),
            dst_layout: dst_image_layout.as_raw(),
            regions,
            filter: filter.as_raw(),
        },
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdResolveImage(
    command_buffer: ash::vk::CommandBuffer,
    src_image: ash::vk::Image,
    src_image_layout: ash::vk::ImageLayout,
    dst_image: ash::vk::Image,
    dst_image_layout: ash::vk::ImageLayout,
    region_count: u32,
    p_regions: *const ash::vk::ImageResolve,
) {
    if command_buffer.is_null() {
        return;
    }
    if p_regions.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    let regions = std::slice::from_raw_parts(p_regions, region_count as usize)
        .iter()
        .map(|r| vkremote_wire::types::ImageCopyRegionData {
            src_subresource: layers(r.src_subresource),
            src_offset: offset3(r.src_offset),
            dst_subresource: layers(r.dst_subresource),
            dst_offset: offset3(r.dst_offset),
            extent: extent3(r.extent),
        })
        .collect();
    record_or_send(
        cb,
        Request::CmdResolveImage {
            buffer_id: cb.id,
            src_image_id: src_image.as_raw(),
            src_layout: src_image_layout.as_raw(),
            dst_image_id: dst_image.as_raw(),
            dst_layout: dst_image_layout.as_raw(),
            regions,
        },
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdUpdateBuffer(
    command_buffer: ash::vk::CommandBuffer,
    dst_buffer: ash::vk::Buffer,
    dst_offset: ash::vk::DeviceSize,
    data_size: ash::vk::DeviceSize,
    p_data: *const std::ffi::c_void,
) {
    if command_buffer.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    if p_data.is_null() || data_size > 65536 {
        return;
    }
    let data = std::slice::from_raw_parts(p_data as *const u8, data_size as usize).to_vec();
    record_or_send(
        cb,
        Request::CmdUpdateBuffer {
            buffer_id: cb.id,
            dst_buffer_id: dst_buffer.as_raw(),
            offset: dst_offset,
            data,
        },
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdClearColorImage(
    command_buffer: ash::vk::CommandBuffer,
    image: ash::vk::Image,
    image_layout: ash::vk::ImageLayout,
    p_color: *const ash::vk::ClearColorValue,
    range_count: u32,
    p_ranges: *const ash::vk::ImageSubresourceRange,
) {
    if command_buffer.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    if p_color.is_null() || p_ranges.is_null() {
        return;
    }
    let ranges = std::slice::from_raw_parts(p_ranges, range_count as usize)
        .iter()
        .map(image_range)
        .collect();
    record_or_send(
        cb,
        Request::CmdClearColorImage {
            buffer_id: cb.id,
            image_id: image.as_raw(),
            layout: image_layout.as_raw(),
            color: (*p_color).uint32,
            ranges,
        },
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdClearDepthStencilImage(
    command_buffer: ash::vk::CommandBuffer,
    image: ash::vk::Image,
    image_layout: ash::vk::ImageLayout,
    p_depth_stencil: *const ash::vk::ClearDepthStencilValue,
    range_count: u32,
    p_ranges: *const ash::vk::ImageSubresourceRange,
) {
    if command_buffer.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    if p_depth_stencil.is_null() || p_ranges.is_null() {
        return;
    }
    let ranges = std::slice::from_raw_parts(p_ranges, range_count as usize)
        .iter()
        .map(image_range)
        .collect();
    record_or_send(
        cb,
        Request::CmdClearDepthStencilImage {
            buffer_id: cb.id,
            image_id: image.as_raw(),
            layout: image_layout.as_raw(),
            depth_bits: (*p_depth_stencil).depth.to_bits(),
            stencil: (*p_depth_stencil).stencil,
            ranges,
        },
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdClearAttachments(
    _command_buffer: ash::vk::CommandBuffer,
    _attachment_count: u32,
    _p_attachments: *const ash::vk::ClearAttachment,
    _rect_count: u32,
    _p_rects: *const ash::vk::ClearRect,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetEvent(
    _command_buffer: ash::vk::CommandBuffer,
    _event: ash::vk::Event,
    _stage_mask: ash::vk::PipelineStageFlags,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdResetEvent(
    _command_buffer: ash::vk::CommandBuffer,
    _event: ash::vk::Event,
    _stage_mask: ash::vk::PipelineStageFlags,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdWaitEvents(
    _command_buffer: ash::vk::CommandBuffer,
    _event_count: u32,
    _p_events: *const ash::vk::Event,
    _src_stage_mask: ash::vk::PipelineStageFlags,
    _dst_stage_mask: ash::vk::PipelineStageFlags,
    _memory_barrier_count: u32,
    _p_memory_barriers: *const ash::vk::MemoryBarrier,
    _buffer_memory_barrier_count: u32,
    _p_buffer_memory_barriers: *const ash::vk::BufferMemoryBarrier,
    _image_memory_barrier_count: u32,
    _p_image_memory_barriers: *const ash::vk::ImageMemoryBarrier,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdBeginQuery(
    _command_buffer: ash::vk::CommandBuffer,
    _query_pool: ash::vk::QueryPool,
    _query: u32,
    _flags: ash::vk::QueryControlFlags,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdEndQuery(
    _command_buffer: ash::vk::CommandBuffer,
    _query_pool: ash::vk::QueryPool,
    _query: u32,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdCopyQueryPoolResults(
    _command_buffer: ash::vk::CommandBuffer,
    _query_pool: ash::vk::QueryPool,
    _first_query: u32,
    _query_count: u32,
    _dst_buffer: ash::vk::Buffer,
    _dst_offset: ash::vk::DeviceSize,
    _stride: ash::vk::DeviceSize,
    _flags: ash::vk::QueryResultFlags,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetLineWidth(
    _command_buffer: ash::vk::CommandBuffer,
    _line_width: f32,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetDepthBias(
    _command_buffer: ash::vk::CommandBuffer,
    _depth_bias_constant_factor: f32,
    _depth_bias_clamp: f32,
    _depth_bias_slope_factor: f32,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetBlendConstants(
    _command_buffer: ash::vk::CommandBuffer,
    _blend_constants: &[f32; 4],
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetDepthBounds(
    _command_buffer: ash::vk::CommandBuffer,
    _min_depth_bounds: f32,
    _max_depth_bounds: f32,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetStencilCompareMask(
    _command_buffer: ash::vk::CommandBuffer,
    _face_mask: ash::vk::StencilFaceFlags,
    _compare_mask: u32,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetStencilWriteMask(
    _command_buffer: ash::vk::CommandBuffer,
    _face_mask: ash::vk::StencilFaceFlags,
    _write_mask: u32,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdSetStencilReference(
    _command_buffer: ash::vk::CommandBuffer,
    _face_mask: ash::vk::StencilFaceFlags,
    _reference: u32,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDrawIndirect(
    command_buffer: ash::vk::CommandBuffer,
    buffer: ash::vk::Buffer,
    offset: ash::vk::DeviceSize,
    draw_count: u32,
    stride: u32,
) {
    if command_buffer.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    record_or_send(
        cb,
        Request::CmdDrawIndirect {
            buffer_id: cb.id,
            indirect_buffer_id: buffer.as_raw(),
            offset,
            draw_count,
            stride,
            indexed: false,
        },
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDrawIndexedIndirect(
    command_buffer: ash::vk::CommandBuffer,
    buffer: ash::vk::Buffer,
    offset: ash::vk::DeviceSize,
    draw_count: u32,
    stride: u32,
) {
    if command_buffer.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    record_or_send(
        cb,
        Request::CmdDrawIndirect {
            buffer_id: cb.id,
            indirect_buffer_id: buffer.as_raw(),
            offset,
            draw_count,
            stride,
            indexed: true,
        },
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdDispatchIndirect(
    command_buffer: ash::vk::CommandBuffer,
    buffer: ash::vk::Buffer,
    offset: ash::vk::DeviceSize,
) {
    if command_buffer.is_null() {
        return;
    }
    let cb = CommandBufferDispatchable::from_vk(command_buffer);
    record_or_send(
        cb,
        Request::CmdDispatchIndirect {
            buffer_id: cb.id,
            indirect_buffer_id: buffer.as_raw(),
            offset,
        },
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdNextSubpass(
    _command_buffer: ash::vk::CommandBuffer,
    _contents: ash::vk::SubpassContents,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkCmdExecuteCommands(
    _command_buffer: ash::vk::CommandBuffer,
    _command_buffer_count: u32,
    _p_command_buffers: *const ash::vk::CommandBuffer,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkResetCommandPool(
    _device: ash::vk::Device,
    _command_pool: ash::vk::CommandPool,
    _flags: ash::vk::CommandPoolResetFlags,
) -> ash::vk::Result {
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkGetRenderAreaGranularity(
    _device: ash::vk::Device,
    _render_pass: ash::vk::RenderPass,
    p_granularity: *mut ash::vk::Extent2D,
) {
    if !p_granularity.is_null() {
        *p_granularity = ash::vk::Extent2D {
            width: 1,
            height: 1,
        };
    }
}
