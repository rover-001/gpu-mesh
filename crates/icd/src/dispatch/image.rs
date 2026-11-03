use crate::abi::DeviceDispatchable;
use ash::vk::Handle;
use vkremote_wire::types::{Request, Response};

#[no_mangle]
pub unsafe extern "system" fn vkCreateImage(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::ImageCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_image: *mut ash::vk::Image,
) -> ash::vk::Result {
    if device.is_null() || p_create_info.is_null() || p_image.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let dev = DeviceDispatchable::from_vk(device);
    let info = &*p_create_info;
    let image_id = dev.client.next_id();
    eprintln!(
        "[vkremote-icd] CreateImage image_id={} format={:?} extent={:?} usage={:?}",
        image_id, info.format, info.extent, info.usage
    );

    dev.client.register_image(
        image_id,
        info.format,
        info.extent,
        info.mip_levels,
        info.array_layers,
        info.usage,
    );

    let params = vkremote_wire::types::CreateImageParams {
        image_type: (info.image_type.as_raw() as u32).into(),
        format: (info.format.as_raw() as u32).into(),
        extent: vkremote_wire::types::Extent3D {
            width: info.extent.width,
            height: info.extent.height,
            depth: info.extent.depth,
        },
        mip_levels: info.mip_levels,
        array_layers: info.array_layers,
        samples: (info.samples.as_raw()).into(),
        tiling: (info.tiling.as_raw() as u32).into(),
        usage: info.usage.as_raw().into(),
        sharing_mode: (info.sharing_mode.as_raw() as u32).into(),
        queue_family_indices: Vec::new(),
        initial_layout: (info.initial_layout.as_raw() as u32).into(),
    };
    let req = Request::CreateImage { image_id, params };

    match dev.client.send_request(req) {
        Ok(Response::Success) | Ok(Response::ImageCreated { .. }) => {
            eprintln!("[vkremote-icd] CreateImage image_id={} OK", image_id);
        }
        Ok(Response::Error { code, message }) => {
            eprintln!(
                "[vkremote-icd] CreateImage server error: code={} message={}",
                code, message
            );
            return ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
        }
        Ok(other) => {
            eprintln!("[vkremote-icd] CreateImage unexpected response: {:?}", other);
        }
        Err(e) => {
            eprintln!("[vkremote-icd] CreateImage transport error: {}", e);
            tracing::error!("CreateImage request failed: {}", e);
            return ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }

    *p_image = ash::vk::Image::from_raw(image_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyImage(
    device: ash::vk::Device,
    image: ash::vk::Image,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || image.is_null() {
        return;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let image_id = image.as_raw();
    dev.client.unregister_image(image_id);
    let _ = dev.client.send_request(Request::DestroyImage { image_id });
}

#[no_mangle]
pub unsafe extern "system" fn vkGetImageMemoryRequirements(
    device: ash::vk::Device,
    image: ash::vk::Image,
    p_memory_requirements: *mut ash::vk::MemoryRequirements,
) {
    if device.is_null() || p_memory_requirements.is_null() {
        return;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let reqs = &mut *p_memory_requirements;

    let (width, height, depth, layers) =
        if let Some(info) = dev.client.get_image_info(image.as_raw()) {
            (
                info.extent.width.max(1),
                info.extent.height.max(1),
                info.extent.depth.max(1),
                info.array_layers.max(1),
            )
        } else {
            (64, 64, 1, 1)
        };

    let alignment = 64u64;
    let raw_size = (width as u64) * (height as u64) * (depth as u64) * (layers as u64) * 4;
    let aligned_size = ((raw_size + alignment - 1) / alignment) * alignment;

    reqs.alignment = alignment;
    reqs.size = aligned_size.max(256);
    reqs.memory_type_bits = 0xFFFF_FFFF;

    let req = Request::GetImageMemoryRequirements {
        image_id: image.as_raw(),
    };
    match dev.client.send_request(req) {
        Ok(Response::ImageMemoryRequirements(req_data)) => {
            reqs.size = req_data.size;
            reqs.alignment = req_data.alignment;
            reqs.memory_type_bits = req_data.memory_type_bits & !1;
        }
        Ok(Response::MemoryRequirements {
            size,
            alignment,
            memory_type_bits,
        }) => {
            reqs.size = size;
            reqs.alignment = alignment;
            reqs.memory_type_bits = memory_type_bits & !1;
        }
        other => {
            eprintln!(
                "[vkremote-icd] GetImageMemoryRequirements image={} resp: {:?}",
                image.as_raw(),
                other
            );
        }
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetImageMemoryRequirements2(
    device: ash::vk::Device,
    p_info: *const ash::vk::ImageMemoryRequirementsInfo2,
    p_memory_requirements: *mut ash::vk::MemoryRequirements2,
) {
    if p_info.is_null() || p_memory_requirements.is_null() {
        return;
    }
    let info = &*p_info;
    let reqs2 = &mut *p_memory_requirements;
    vkGetImageMemoryRequirements(device, info.image, &mut reqs2.memory_requirements);

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
pub unsafe extern "system" fn vkGetImageMemoryRequirements2KHR(
    device: ash::vk::Device,
    p_info: *const ash::vk::ImageMemoryRequirementsInfo2,
    p_memory_requirements: *mut ash::vk::MemoryRequirements2,
) {
    vkGetImageMemoryRequirements2(device, p_info, p_memory_requirements);
}

#[no_mangle]
pub unsafe extern "system" fn vkBindImageMemory(
    device: ash::vk::Device,
    image: ash::vk::Image,
    memory: ash::vk::DeviceMemory,
    memory_offset: ash::vk::DeviceSize,
) -> ash::vk::Result {
    if device.is_null() || image.is_null() || memory.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let dev = DeviceDispatchable::from_vk(device);
    dev.client
        .bind_image_memory(image.as_raw(), memory.as_raw(), memory_offset);

    let req = Request::BindImageMemory {
        image_id: image.as_raw(),
        memory_id: memory.as_raw(),
        offset: memory_offset,
    };
    let _ = dev.client.send_request(req);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkBindImageMemory2(
    device: ash::vk::Device,
    bind_info_count: u32,
    p_bind_infos: *const ash::vk::BindImageMemoryInfo,
) -> ash::vk::Result {
    if device.is_null() || p_bind_infos.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    for i in 0..bind_info_count as usize {
        let info = &*p_bind_infos.add(i);
        let res = vkBindImageMemory(device, info.image, info.memory, info.memory_offset);
        if res != ash::vk::Result::SUCCESS {
            return res;
        }
    }

    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkBindImageMemory2KHR(
    device: ash::vk::Device,
    bind_info_count: u32,
    p_bind_infos: *const ash::vk::BindImageMemoryInfo,
) -> ash::vk::Result {
    vkBindImageMemory2(device, bind_info_count, p_bind_infos)
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateImageView(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::ImageViewCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_view: *mut ash::vk::ImageView,
) -> ash::vk::Result {
    if device.is_null() || p_create_info.is_null() || p_view.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let info = &*p_create_info;
    let view_id = dev.client.next_id();

    let params = vkremote_wire::types::CreateImageViewParams {
        image_id: info.image.as_raw(),
        view_type: (info.view_type.as_raw() as u32).into(),
        format: (info.format.as_raw() as u32).into(),
        components: vkremote_wire::types::ComponentMapping {
            r: (info.components.r.as_raw() as u32).into(),
            g: (info.components.g.as_raw() as u32).into(),
            b: (info.components.b.as_raw() as u32).into(),
            a: (info.components.a.as_raw() as u32).into(),
        },
        subresource_range: vkremote_wire::types::ImageSubresourceRange {
            aspect_mask: info.subresource_range.aspect_mask.as_raw().into(),
            base_mip_level: info.subresource_range.base_mip_level,
            level_count: info.subresource_range.level_count,
            base_array_layer: info.subresource_range.base_array_layer,
            layer_count: info.subresource_range.layer_count,
        },
    };
    let req = Request::CreateImageView { view_id, params };
    let _ = dev.client.send_request(req);

    *p_view = ash::vk::ImageView::from_raw(view_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyImageView(
    device: ash::vk::Device,
    image_view: ash::vk::ImageView,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || image_view.is_null() {
        return;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let _ = dev.client.send_request(Request::DestroyImageView {
        view_id: image_view.as_raw(),
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateSampler(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::SamplerCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_sampler: *mut ash::vk::Sampler,
) -> ash::vk::Result {
    if device.is_null() || p_create_info.is_null() || p_sampler.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let info = &*p_create_info;
    let sampler_id = dev.client.next_id();

    let params = vkremote_wire::types::CreateSamplerParams {
        mag_filter: (info.mag_filter.as_raw() as u32).into(),
        min_filter: (info.min_filter.as_raw() as u32).into(),
        mipmap_mode: (info.mipmap_mode.as_raw() as u32).into(),
        address_mode_u: (info.address_mode_u.as_raw() as u32).into(),
        address_mode_v: (info.address_mode_v.as_raw() as u32).into(),
        address_mode_w: (info.address_mode_w.as_raw() as u32).into(),
        mip_lod_bias: info.mip_lod_bias.into(),
        anisotropy_enable: info.anisotropy_enable != 0,
        max_anisotropy: info.max_anisotropy.into(),
        compare_enable: info.compare_enable != 0,
        compare_op: (info.compare_op.as_raw() as u32).into(),
        min_lod: info.min_lod.into(),
        max_lod: info.max_lod.into(),
        border_color: (info.border_color.as_raw() as u32).into(),
        unnormalized_coordinates: info.unnormalized_coordinates != 0,
    };
    let req = Request::CreateSampler { sampler_id, params };
    let _ = dev.client.send_request(req);

    *p_sampler = ash::vk::Sampler::from_raw(sampler_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroySampler(
    device: ash::vk::Device,
    sampler: ash::vk::Sampler,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || sampler.is_null() {
        return;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let _ = dev.client.send_request(Request::DestroySampler {
        sampler_id: sampler.as_raw(),
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkGetImageSubresourceLayout(
    device: ash::vk::Device,
    image: ash::vk::Image,
    p_subresource: *const ash::vk::ImageSubresource,
    p_layout: *mut ash::vk::SubresourceLayout,
) {
    if device.is_null() || p_subresource.is_null() || p_layout.is_null() {
        return;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let sub = &*p_subresource;
    let out = &mut *p_layout;
    *out = ash::vk::SubresourceLayout::default();
    let req = Request::GetImageSubresourceLayout {
        image_id: image.as_raw(),
        aspect_mask: sub.aspect_mask.as_raw(),
        mip_level: sub.mip_level,
        array_layer: sub.array_layer,
    };
    match dev.client.send_request(req) {
        Ok(Response::ImageSubresourceLayout {
            offset,
            size,
            row_pitch,
            array_pitch,
            depth_pitch,
        }) => {
            out.offset = offset;
            out.size = size;
            out.row_pitch = row_pitch;
            out.array_pitch = array_pitch;
            out.depth_pitch = depth_pitch;
        }
        other => {
            // Tightly packed 4-byte texel fallback so callers never read garbage.
            eprintln!("[vkremote-icd] vkGetImageSubresourceLayout fallback: {:?}", other.err());
            if let Some(info) = dev.client.get_image_info(image.as_raw()) {
                let row = info.extent.width.max(1) as u64 * 4;
                let slice = row * info.extent.height.max(1) as u64;
                out.row_pitch = row;
                out.depth_pitch = slice;
                out.array_pitch = slice * info.extent.depth.max(1) as u64;
                out.size = out.array_pitch;
            }
        }
    }
}
