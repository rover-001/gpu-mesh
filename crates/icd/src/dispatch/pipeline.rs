use crate::abi::DeviceDispatchable;
use ash::vk::Handle;
use std::ffi::CStr;
use vkremote_wire::types::{
    DescriptorBufferInfoData, DescriptorPoolSizeData, DescriptorSetLayoutBindingData,
    PushConstantRangeData, Request, ResourceType, Response, WriteDescriptorSetData,
};

#[no_mangle]
pub unsafe extern "system" fn vkCreateShaderModule(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::ShaderModuleCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_shader_module: *mut ash::vk::ShaderModule,
) -> ash::vk::Result {
    if device.is_null() || p_create_info.is_null() || p_shader_module.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let info = &*p_create_info;
    let spirv = std::slice::from_raw_parts(info.p_code as *const u8, info.code_size).to_vec();

    let shader_id = dev.client.next_id();
    let req = Request::CreateShaderModule { shader_id, spirv };

    if let Err(e) = dev.client.send_request(req) {
        tracing::error!("CreateShaderModule request failed: {}", e);
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    *p_shader_module = ash::vk::ShaderModule::from_raw(shader_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyShaderModule(
    device: ash::vk::Device,
    shader_module: ash::vk::ShaderModule,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || shader_module.is_null() {
        return;
    }
    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let _ = dev.client.send_request(Request::DestroyResource {
        handle_id: shader_module.as_raw(),
        resource_type: ResourceType::ShaderModule,
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateDescriptorSetLayout(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::DescriptorSetLayoutCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_set_layout: *mut ash::vk::DescriptorSetLayout,
) -> ash::vk::Result {
    if device.is_null() || p_create_info.is_null() || p_set_layout.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let info = &*p_create_info;
    let mut bindings = Vec::with_capacity(info.binding_count as usize);

    if !info.p_bindings.is_null() && info.binding_count > 0 {
        for i in 0..info.binding_count as usize {
            let b = &*info.p_bindings.add(i);
            bindings.push(DescriptorSetLayoutBindingData {
                binding: b.binding,
                descriptor_type: b.descriptor_type.as_raw() as u32,
                descriptor_count: b.descriptor_count,
                stage_flags: b.stage_flags.as_raw(),
            });
        }
    }

    let layout_id = dev.client.next_id();
    let req = Request::CreateDescriptorSetLayout {
        layout_id,
        bindings,
    };

    if let Err(e) = dev.client.send_request(req) {
        tracing::error!("CreateDescriptorSetLayout request failed: {}", e);
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    *p_set_layout = ash::vk::DescriptorSetLayout::from_raw(layout_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyDescriptorSetLayout(
    device: ash::vk::Device,
    descriptor_set_layout: ash::vk::DescriptorSetLayout,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || descriptor_set_layout.is_null() {
        return;
    }
    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let _ = dev.client.send_request(Request::DestroyResource {
        handle_id: descriptor_set_layout.as_raw(),
        resource_type: ResourceType::DescriptorSetLayout,
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkCreatePipelineLayout(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::PipelineLayoutCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_pipeline_layout: *mut ash::vk::PipelineLayout,
) -> ash::vk::Result {
    if device.is_null() || p_create_info.is_null() || p_pipeline_layout.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let info = &*p_create_info;

    let mut set_layout_ids = Vec::with_capacity(info.set_layout_count as usize);
    if !info.p_set_layouts.is_null() && info.set_layout_count > 0 {
        for i in 0..info.set_layout_count as usize {
            let l = *info.p_set_layouts.add(i);
            set_layout_ids.push(l.as_raw());
        }
    }

    let mut push_constant_ranges = Vec::with_capacity(info.push_constant_range_count as usize);
    if !info.p_push_constant_ranges.is_null() && info.push_constant_range_count > 0 {
        for i in 0..info.push_constant_range_count as usize {
            let r = &*info.p_push_constant_ranges.add(i);
            push_constant_ranges.push(PushConstantRangeData {
                stage_flags: r.stage_flags.as_raw(),
                offset: r.offset,
                size: r.size,
            });
        }
    }

    let layout_id = dev.client.next_id();
    let req = Request::CreatePipelineLayout {
        layout_id,
        set_layout_ids,
        push_constant_ranges,
    };

    if let Err(e) = dev.client.send_request(req) {
        tracing::error!("CreatePipelineLayout request failed: {}", e);
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    *p_pipeline_layout = ash::vk::PipelineLayout::from_raw(layout_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyPipelineLayout(
    device: ash::vk::Device,
    pipeline_layout: ash::vk::PipelineLayout,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || pipeline_layout.is_null() {
        return;
    }
    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let _ = dev.client.send_request(Request::DestroyResource {
        handle_id: pipeline_layout.as_raw(),
        resource_type: ResourceType::PipelineLayout,
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateComputePipelines(
    device: ash::vk::Device,
    _pipeline_cache: ash::vk::PipelineCache,
    create_info_count: u32,
    p_create_infos: *const ash::vk::ComputePipelineCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_pipelines: *mut ash::vk::Pipeline,
) -> ash::vk::Result {
    if device.is_null() || p_create_infos.is_null() || p_pipelines.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);

    for i in 0..create_info_count as usize {
        let info = &*p_create_infos.add(i);
        let pipeline_id = dev.client.next_id();
        let layout_id = info.layout.as_raw();
        let shader_id = info.stage.module.as_raw();
        let entry_point = if !info.stage.p_name.is_null() {
            CStr::from_ptr(info.stage.p_name)
                .to_string_lossy()
                .into_owned()
        } else {
            "main".to_string()
        };

        let req = Request::CreateComputePipeline {
            pipeline_id,
            layout_id,
            shader_id,
            entry_point,
        };

        if let Err(e) = dev.client.send_request(req) {
            tracing::error!("CreateComputePipeline request failed: {}", e);
            return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
        }

        *p_pipelines.add(i) = ash::vk::Pipeline::from_raw(pipeline_id);
    }

    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyPipeline(
    device: ash::vk::Device,
    pipeline: ash::vk::Pipeline,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || pipeline.is_null() {
        return;
    }
    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let _ = dev.client.send_request(Request::DestroyResource {
        handle_id: pipeline.as_raw(),
        resource_type: ResourceType::Pipeline,
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateDescriptorPool(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::DescriptorPoolCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_descriptor_pool: *mut ash::vk::DescriptorPool,
) -> ash::vk::Result {
    if device.is_null() || p_create_info.is_null() || p_descriptor_pool.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let info = &*p_create_info;

    let mut pool_sizes = Vec::with_capacity(info.pool_size_count as usize);
    if !info.p_pool_sizes.is_null() && info.pool_size_count > 0 {
        for i in 0..info.pool_size_count as usize {
            let s = &*info.p_pool_sizes.add(i);
            pool_sizes.push(DescriptorPoolSizeData {
                descriptor_type: s.ty.as_raw() as u32,
                descriptor_count: s.descriptor_count,
            });
        }
    }

    let pool_id = dev.client.next_id();
    let req = Request::CreateDescriptorPool {
        pool_id,
        max_sets: info.max_sets,
        pool_sizes,
    };

    if let Err(e) = dev.client.send_request(req) {
        tracing::error!("CreateDescriptorPool request failed: {}", e);
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    *p_descriptor_pool = ash::vk::DescriptorPool::from_raw(pool_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyDescriptorPool(
    device: ash::vk::Device,
    descriptor_pool: ash::vk::DescriptorPool,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || descriptor_pool.is_null() {
        return;
    }
    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let _ = dev.client.send_request(Request::DestroyResource {
        handle_id: descriptor_pool.as_raw(),
        resource_type: ResourceType::DescriptorPool,
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkAllocateDescriptorSets(
    device: ash::vk::Device,
    p_allocate_info: *const ash::vk::DescriptorSetAllocateInfo,
    p_descriptor_sets: *mut ash::vk::DescriptorSet,
) -> ash::vk::Result {
    if device.is_null() || p_allocate_info.is_null() || p_descriptor_sets.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let info = &*p_allocate_info;
    let count = info.descriptor_set_count as usize;

    let mut layout_ids = Vec::with_capacity(count);
    let mut set_ids = Vec::with_capacity(count);

    for i in 0..count {
        let layout = *info.p_set_layouts.add(i);
        layout_ids.push(layout.as_raw());
        let set_id = dev.client.next_id();
        set_ids.push(set_id);
        *p_descriptor_sets.add(i) = ash::vk::DescriptorSet::from_raw(set_id);
    }

    let req = Request::AllocateDescriptorSets {
        pool_id: info.descriptor_pool.as_raw(),
        layout_ids,
        set_ids,
    };

    if let Err(e) = dev.client.send_request(req) {
        tracing::error!("AllocateDescriptorSets request failed: {}", e);
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkFreeDescriptorSets(
    _device: ash::vk::Device,
    _descriptor_pool: ash::vk::DescriptorPool,
    _descriptor_set_count: u32,
    _p_descriptor_sets: *const ash::vk::DescriptorSet,
) -> ash::vk::Result {
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkResetDescriptorPool(
    _device: ash::vk::Device,
    _descriptor_pool: ash::vk::DescriptorPool,
    _flags: ash::vk::DescriptorPoolResetFlags,
) -> ash::vk::Result {
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkUpdateDescriptorSets(
    device: ash::vk::Device,
    descriptor_write_count: u32,
    p_descriptor_writes: *const ash::vk::WriteDescriptorSet,
    _descriptor_copy_count: u32,
    _p_descriptor_copies: *const ash::vk::CopyDescriptorSet,
) {
    if device.is_null() || p_descriptor_writes.is_null() || descriptor_write_count == 0 {
        return;
    }

    // SAFETY: device is a valid handle created by vkCreateDevice
    let dev = DeviceDispatchable::from_vk(device);
    let mut writes = Vec::with_capacity(descriptor_write_count as usize);

    for i in 0..descriptor_write_count as usize {
        let w = &*p_descriptor_writes.add(i);
        let mut buffer_info = Vec::with_capacity(w.descriptor_count as usize);

        if !w.p_buffer_info.is_null() && w.descriptor_count > 0 {
            for j in 0..w.descriptor_count as usize {
                let bi = &*w.p_buffer_info.add(j);
                buffer_info.push(DescriptorBufferInfoData {
                    buffer_id: bi.buffer.as_raw(),
                    offset: bi.offset,
                    range: bi.range,
                });
            }
        }

        let mut image_info = Vec::with_capacity(w.descriptor_count as usize);
        if !w.p_image_info.is_null() && w.descriptor_count > 0 {
            for j in 0..w.descriptor_count as usize {
                let ii = &*w.p_image_info.add(j);
                image_info.push(vkremote_wire::types::DescriptorImageInfoData {
                    sampler_id: ii.sampler.as_raw(),
                    image_view_id: ii.image_view.as_raw(),
                    image_layout: ii.image_layout.as_raw() as u32,
                });
            }
        }

        writes.push(WriteDescriptorSetData {
            dst_set_id: w.dst_set.as_raw(),
            dst_binding: w.dst_binding,
            dst_array_element: w.dst_array_element,
            descriptor_type: w.descriptor_type.as_raw() as u32,
            buffer_info,
            image_info,
        });
    }

    let req = Request::UpdateDescriptorSets { writes };
    let _ = dev.client.send_request(req);
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateRenderPass(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::RenderPassCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_render_pass: *mut ash::vk::RenderPass,
) -> ash::vk::Result {
    if device.is_null() || p_create_info.is_null() || p_render_pass.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let info = &*p_create_info;
    let render_pass_id = dev.client.next_id();

    let mut attachments = Vec::with_capacity(info.attachment_count as usize);
    if !info.p_attachments.is_null() && info.attachment_count > 0 {
        for i in 0..info.attachment_count as usize {
            let a = &*info.p_attachments.add(i);
            attachments.push(vkremote_wire::types::AttachmentDescription {
                flags: a.flags.as_raw(),
                format: (a.format.as_raw() as u32).into(),
                samples: (a.samples.as_raw()).into(),
                load_op: (a.load_op.as_raw() as u32).into(),
                store_op: (a.store_op.as_raw() as u32).into(),
                stencil_load_op: (a.stencil_load_op.as_raw() as u32).into(),
                stencil_store_op: (a.stencil_store_op.as_raw() as u32).into(),
                initial_layout: (a.initial_layout.as_raw() as u32).into(),
                final_layout: (a.final_layout.as_raw() as u32).into(),
            });
        }
    }

    let mut subpasses = Vec::with_capacity(info.subpass_count as usize);
    if !info.p_subpasses.is_null() && info.subpass_count > 0 {
        for i in 0..info.subpass_count as usize {
            let s = &*info.p_subpasses.add(i);
            let mut color_attachments = Vec::new();
            if !s.p_color_attachments.is_null() && s.color_attachment_count > 0 {
                for c in 0..s.color_attachment_count as usize {
                    let ca = &*s.p_color_attachments.add(c);
                    color_attachments.push(vkremote_wire::types::AttachmentReference {
                        attachment: ca.attachment,
                        layout: (ca.layout.as_raw() as u32).into(),
                    });
                }
            }
            let depth_stencil_attachment = if !s.p_depth_stencil_attachment.is_null() {
                let da = &*s.p_depth_stencil_attachment;
                Some(vkremote_wire::types::AttachmentReference {
                    attachment: da.attachment,
                    layout: (da.layout.as_raw() as u32).into(),
                })
            } else {
                None
            };
            subpasses.push(vkremote_wire::types::SubpassDescription {
                flags: s.flags.as_raw(),
                pipeline_bind_point: (s.pipeline_bind_point.as_raw() as u32).into(),
                input_attachments: Vec::new(),
                color_attachments,
                resolve_attachments: Vec::new(),
                depth_stencil_attachment,
                preserve_attachments: Vec::new(),
            });
        }
    }

    let mut dependencies = Vec::with_capacity(info.dependency_count as usize);
    if !info.p_dependencies.is_null() && info.dependency_count > 0 {
        for i in 0..info.dependency_count as usize {
            let d = &*info.p_dependencies.add(i);
            dependencies.push(vkremote_wire::types::SubpassDependency {
                src_subpass: d.src_subpass,
                dst_subpass: d.dst_subpass,
                src_stage_mask: d.src_stage_mask.as_raw(),
                dst_stage_mask: d.dst_stage_mask.as_raw(),
                src_access_mask: d.src_access_mask.as_raw(),
                dst_access_mask: d.dst_access_mask.as_raw(),
                dependency_flags: d.dependency_flags.as_raw(),
            });
        }
    }

    let params = vkremote_wire::types::CreateRenderPassParams {
        attachments,
        subpasses,
        dependencies,
    };
    let req = Request::CreateRenderPass {
        render_pass_id,
        params,
    };
    let _ = dev.client.send_request(req);

    *p_render_pass = ash::vk::RenderPass::from_raw(render_pass_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateRenderPass2(
    device: ash::vk::Device,
    _p_create_info: *const ash::vk::RenderPassCreateInfo2,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_render_pass: *mut ash::vk::RenderPass,
) -> ash::vk::Result {
    if device.is_null() || p_render_pass.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let render_pass_id = dev.client.next_id();
    *p_render_pass = ash::vk::RenderPass::from_raw(render_pass_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateRenderPass2KHR(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::RenderPassCreateInfo2,
    p_allocator: *const ash::vk::AllocationCallbacks,
    p_render_pass: *mut ash::vk::RenderPass,
) -> ash::vk::Result {
    vkCreateRenderPass2(device, p_create_info, p_allocator, p_render_pass)
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyRenderPass(
    device: ash::vk::Device,
    render_pass: ash::vk::RenderPass,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || render_pass.is_null() {
        return;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let _ = dev.client.send_request(Request::DestroyRenderPass {
        render_pass_id: render_pass.as_raw(),
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateFramebuffer(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::FramebufferCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_framebuffer: *mut ash::vk::Framebuffer,
) -> ash::vk::Result {
    if device.is_null() || p_create_info.is_null() || p_framebuffer.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let info = &*p_create_info;
    let framebuffer_id = dev.client.next_id();

    let mut attachments = Vec::with_capacity(info.attachment_count as usize);
    if !info.p_attachments.is_null() && info.attachment_count > 0 {
        for i in 0..info.attachment_count as usize {
            attachments.push((*info.p_attachments.add(i)).as_raw());
        }
    }

    let params = vkremote_wire::types::CreateFramebufferParams {
        render_pass_id: info.render_pass.as_raw(),
        attachments,
        width: info.width,
        height: info.height,
        layers: info.layers,
    };
    let req = Request::CreateFramebuffer {
        framebuffer_id,
        params,
    };
    let _ = dev.client.send_request(req);

    *p_framebuffer = ash::vk::Framebuffer::from_raw(framebuffer_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyFramebuffer(
    device: ash::vk::Device,
    framebuffer: ash::vk::Framebuffer,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || framebuffer.is_null() {
        return;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let _ = dev.client.send_request(Request::DestroyFramebuffer {
        framebuffer_id: framebuffer.as_raw(),
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateGraphicsPipelines(
    device: ash::vk::Device,
    _pipeline_cache: ash::vk::PipelineCache,
    create_info_count: u32,
    p_create_infos: *const ash::vk::GraphicsPipelineCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_pipelines: *mut ash::vk::Pipeline,
) -> ash::vk::Result {
    if device.is_null()
        || p_create_infos.is_null()
        || p_pipelines.is_null()
        || create_info_count == 0
    {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let mut pipelines_data = Vec::with_capacity(create_info_count as usize);

    for i in 0..create_info_count as usize {
        let info = &*p_create_infos.add(i);
        let pipeline_id = dev.client.next_id();
        *p_pipelines.add(i) = ash::vk::Pipeline::from_raw(pipeline_id);

        let mut stages = Vec::with_capacity(info.stage_count as usize);
        if !info.p_stages.is_null() && info.stage_count > 0 {
            for s in 0..info.stage_count as usize {
                let stage_info = &*info.p_stages.add(s);
                let entry_name = if !stage_info.p_name.is_null() {
                    CStr::from_ptr(stage_info.p_name)
                        .to_string_lossy()
                        .into_owned()
                } else {
                    "main".to_string()
                };
                stages.push(vkremote_wire::types::PipelineShaderStageData {
                    stage: vkremote_wire::types::ShaderStageFlags(stage_info.stage.as_raw()),
                    module_id: stage_info.module.as_raw(),
                    entry_point: entry_name,
                });
            }
        }

        let vertex_input_state = if !info.p_vertex_input_state.is_null() {
            let vi = &*info.p_vertex_input_state;
            let mut bindings = Vec::new();
            let mut attributes = Vec::new();
            if !vi.p_vertex_binding_descriptions.is_null()
                && vi.vertex_binding_description_count > 0
            {
                for b in 0..vi.vertex_binding_description_count as usize {
                    let desc = &*vi.p_vertex_binding_descriptions.add(b);
                    bindings.push(vkremote_wire::types::VertexInputBindingDescription {
                        binding: desc.binding,
                        stride: desc.stride,
                        input_rate: (desc.input_rate.as_raw() as u32).into(),
                    });
                }
            }
            if !vi.p_vertex_attribute_descriptions.is_null()
                && vi.vertex_attribute_description_count > 0
            {
                for a in 0..vi.vertex_attribute_description_count as usize {
                    let desc = &*vi.p_vertex_attribute_descriptions.add(a);
                    attributes.push(vkremote_wire::types::VertexInputAttributeDescription {
                        location: desc.location,
                        binding: desc.binding,
                        format: (desc.format.as_raw() as u32).into(),
                        offset: desc.offset,
                    });
                }
            }
            vkremote_wire::types::PipelineVertexInputStateData {
                bindings,
                attributes,
            }
        } else {
            Default::default()
        };

        let input_assembly_state = if !info.p_input_assembly_state.is_null() {
            let ia = &*info.p_input_assembly_state;
            vkremote_wire::types::PipelineInputAssemblyStateData {
                topology: (ia.topology.as_raw() as u32).into(),
                primitive_restart_enable: ia.primitive_restart_enable != 0,
            }
        } else {
            Default::default()
        };

        let viewport_state = if !info.p_viewport_state.is_null() {
            let vp = &*info.p_viewport_state;
            let mut viewports = Vec::new();
            let mut scissors = Vec::new();
            if !vp.p_viewports.is_null() && vp.viewport_count > 0 {
                for v in 0..vp.viewport_count as usize {
                    let v_item = &*vp.p_viewports.add(v);
                    viewports.push(vkremote_wire::types::Viewport::new(
                        v_item.x,
                        v_item.y,
                        v_item.width,
                        v_item.height,
                        v_item.min_depth,
                        v_item.max_depth,
                    ));
                }
            }
            if !vp.p_scissors.is_null() && vp.scissor_count > 0 {
                for s in 0..vp.scissor_count as usize {
                    let s_item = &*vp.p_scissors.add(s);
                    scissors.push(vkremote_wire::types::Rect2D {
                        offset: vkremote_wire::types::Offset2D {
                            x: s_item.offset.x,
                            y: s_item.offset.y,
                        },
                        extent: vkremote_wire::types::Extent2D {
                            width: s_item.extent.width,
                            height: s_item.extent.height,
                        },
                    });
                }
            }
            Some(vkremote_wire::types::PipelineViewportStateData {
                viewports,
                scissors,
            })
        } else {
            None
        };

        let rasterization_state = if !info.p_rasterization_state.is_null() {
            let r = &*info.p_rasterization_state;
            vkremote_wire::types::PipelineRasterizationStateData {
                depth_clamp_enable: r.depth_clamp_enable != 0,
                rasterizer_discard_enable: r.rasterizer_discard_enable != 0,
                polygon_mode: (r.polygon_mode.as_raw() as u32).into(),
                cull_mode: vkremote_wire::types::CullModeFlags(r.cull_mode.as_raw()),
                front_face: (r.front_face.as_raw() as u32).into(),
                depth_bias_enable: r.depth_bias_enable != 0,
                depth_bias_constant_factor: r.depth_bias_constant_factor.into(),
                depth_bias_clamp: r.depth_bias_clamp.into(),
                depth_bias_slope_factor: r.depth_bias_slope_factor.into(),
                line_width: r.line_width.into(),
            }
        } else {
            vkremote_wire::types::PipelineRasterizationStateData {
                depth_clamp_enable: false,
                rasterizer_discard_enable: false,
                polygon_mode: Default::default(),
                cull_mode: vkremote_wire::types::CullModeFlags(0),
                front_face: Default::default(),
                depth_bias_enable: false,
                depth_bias_constant_factor: 0.0.into(),
                depth_bias_clamp: 0.0.into(),
                depth_bias_slope_factor: 0.0.into(),
                line_width: 1.0.into(),
            }
        };

        let multisample_state = if !info.p_multisample_state.is_null() {
            let ms = &*info.p_multisample_state;
            Some(vkremote_wire::types::PipelineMultisampleStateData {
                rasterization_samples: (ms.rasterization_samples.as_raw()).into(),
                sample_shading_enable: ms.sample_shading_enable != 0,
                min_sample_shading: ms.min_sample_shading.into(),
                sample_mask: Vec::new(),
                alpha_to_coverage_enable: ms.alpha_to_coverage_enable != 0,
                alpha_to_one_enable: ms.alpha_to_one_enable != 0,
            })
        } else {
            None
        };

        let depth_stencil_state = if !info.p_depth_stencil_state.is_null() {
            let ds = &*info.p_depth_stencil_state;
            Some(vkremote_wire::types::PipelineDepthStencilStateData {
                depth_test_enable: ds.depth_test_enable != 0,
                depth_write_enable: ds.depth_write_enable != 0,
                depth_compare_op: (ds.depth_compare_op.as_raw() as u32).into(),
                depth_bounds_test_enable: ds.depth_bounds_test_enable != 0,
                stencil_test_enable: ds.stencil_test_enable != 0,
                front: vkremote_wire::types::StencilOpState {
                    fail_op: (ds.front.fail_op.as_raw() as u32).into(),
                    pass_op: (ds.front.pass_op.as_raw() as u32).into(),
                    depth_fail_op: (ds.front.depth_fail_op.as_raw() as u32).into(),
                    compare_op: (ds.front.compare_op.as_raw() as u32).into(),
                    compare_mask: ds.front.compare_mask,
                    write_mask: ds.front.write_mask,
                    reference: ds.front.reference,
                },
                back: vkremote_wire::types::StencilOpState {
                    fail_op: (ds.back.fail_op.as_raw() as u32).into(),
                    pass_op: (ds.back.pass_op.as_raw() as u32).into(),
                    depth_fail_op: (ds.back.depth_fail_op.as_raw() as u32).into(),
                    compare_op: (ds.back.compare_op.as_raw() as u32).into(),
                    compare_mask: ds.back.compare_mask,
                    write_mask: ds.back.write_mask,
                    reference: ds.back.reference,
                },
                min_depth_bounds: ds.min_depth_bounds.into(),
                max_depth_bounds: ds.max_depth_bounds.into(),
            })
        } else {
            None
        };

        let color_blend_state = if !info.p_color_blend_state.is_null() {
            let cb = &*info.p_color_blend_state;
            let mut attachments = Vec::new();
            if !cb.p_attachments.is_null() && cb.attachment_count > 0 {
                for a in 0..cb.attachment_count as usize {
                    let att = &*cb.p_attachments.add(a);
                    attachments.push(vkremote_wire::types::ColorBlendAttachmentState {
                        blend_enable: att.blend_enable != 0,
                        src_color_blend_factor: (att.src_color_blend_factor.as_raw() as u32).into(),
                        dst_color_blend_factor: (att.dst_color_blend_factor.as_raw() as u32).into(),
                        color_blend_op: (att.color_blend_op.as_raw() as u32).into(),
                        src_alpha_blend_factor: (att.src_alpha_blend_factor.as_raw() as u32).into(),
                        dst_alpha_blend_factor: (att.dst_alpha_blend_factor.as_raw() as u32).into(),
                        alpha_blend_op: (att.alpha_blend_op.as_raw() as u32).into(),
                        color_write_mask: vkremote_wire::types::ColorComponentFlags(
                            att.color_write_mask.as_raw(),
                        ),
                    });
                }
            }
            Some(vkremote_wire::types::PipelineColorBlendStateData {
                logic_op_enable: cb.logic_op_enable != 0,
                logic_op: (cb.logic_op.as_raw() as u32).into(),
                attachments,
                blend_constants: [
                    cb.blend_constants[0].into(),
                    cb.blend_constants[1].into(),
                    cb.blend_constants[2].into(),
                    cb.blend_constants[3].into(),
                ],
            })
        } else {
            None
        };

        let dynamic_state = if !info.p_dynamic_state.is_null() {
            let ds = &*info.p_dynamic_state;
            let mut dynamic_states = Vec::new();
            if !ds.p_dynamic_states.is_null() && ds.dynamic_state_count > 0 {
                for d in 0..ds.dynamic_state_count as usize {
                    dynamic_states.push(((*ds.p_dynamic_states.add(d)).as_raw() as u32).into());
                }
            }
            Some(vkremote_wire::types::PipelineDynamicStateData { dynamic_states })
        } else {
            None
        };

        pipelines_data.push(vkremote_wire::types::GraphicsPipelineCreateInfoData {
            pipeline_id,
            stages,
            vertex_input_state,
            input_assembly_state,
            viewport_state,
            rasterization_state,
            multisample_state,
            depth_stencil_state,
            color_blend_state,
            dynamic_state,
            layout_id: info.layout.as_raw(),
            render_pass_id: info.render_pass.as_raw(),
            subpass: info.subpass,
            base_pipeline_id: if info.base_pipeline_handle.as_raw() != 0 {
                Some(info.base_pipeline_handle.as_raw())
            } else {
                None
            },
            base_pipeline_index: info.base_pipeline_index,
        });
    }

    let req = Request::CreateGraphicsPipelines {
        pipelines: pipelines_data,
    };
    let res = dev.client.send_request(req);
    eprintln!("[vkremote-icd] CreateGraphicsPipelines res: {:?}", res);
    match res {
        Ok(Response::GraphicsPipelinesCreated { .. }) | Ok(Response::Success) => {
            ash::vk::Result::SUCCESS
        }
        Ok(Response::Error { code, message }) => {
            eprintln!(
                "[vkremote-icd] CreateGraphicsPipelines server error: code={} msg={}",
                code, message
            );
            ash::vk::Result::ERROR_INITIALIZATION_FAILED
        }
        Err(e) => {
            eprintln!(
                "[vkremote-icd] CreateGraphicsPipelines transport error: {}",
                e
            );
            ash::vk::Result::ERROR_INITIALIZATION_FAILED
        }
        _ => ash::vk::Result::SUCCESS,
    }
}

// Pipeline caches are client-local placeholders: pipeline creation ignores the
// cache handle and the server compiles with a null cache, so there is no
// remote state to create, persist, or merge.
#[no_mangle]
pub unsafe extern "system" fn vkCreatePipelineCache(
    device: ash::vk::Device,
    _p_create_info: *const ash::vk::PipelineCacheCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_pipeline_cache: *mut ash::vk::PipelineCache,
) -> ash::vk::Result {
    if device.is_null() || p_pipeline_cache.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let dev = DeviceDispatchable::from_vk(device);
    *p_pipeline_cache = ash::vk::PipelineCache::from_raw(dev.client.next_id());
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyPipelineCache(
    _device: ash::vk::Device,
    _pipeline_cache: ash::vk::PipelineCache,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPipelineCacheData(
    _device: ash::vk::Device,
    _pipeline_cache: ash::vk::PipelineCache,
    p_data_size: *mut usize,
    _p_data: *mut std::ffi::c_void,
) -> ash::vk::Result {
    if !p_data_size.is_null() {
        *p_data_size = 0;
    }
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkMergePipelineCaches(
    _device: ash::vk::Device,
    _dst_cache: ash::vk::PipelineCache,
    _src_cache_count: u32,
    _p_src_caches: *const ash::vk::PipelineCache,
) -> ash::vk::Result {
    ash::vk::Result::SUCCESS
}
