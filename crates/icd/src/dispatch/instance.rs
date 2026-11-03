use crate::abi::{
    DeviceDispatchable, InstanceDispatchable, PhysicalDeviceDispatchable, QueueDispatchable,
};
use crate::client::Client;
use ash::vk::Handle;
use std::sync::Arc;

fn bool_to_vk(b: bool) -> ash::vk::Bool32 {
    if b {
        ash::vk::TRUE
    } else {
        ash::vk::FALSE
    }
}

pub fn convert_features(
    f: &vkremote_wire::types::PhysicalDeviceFeaturesData,
) -> ash::vk::PhysicalDeviceFeatures {
    let mut out = ash::vk::PhysicalDeviceFeatures::default();
    out.robust_buffer_access = bool_to_vk(f.robust_buffer_access);
    out.full_draw_index_uint32 = bool_to_vk(f.full_draw_index_uint32);
    out.image_cube_array = bool_to_vk(f.image_cube_array);
    out.independent_blend = bool_to_vk(f.independent_blend);
    out.geometry_shader = bool_to_vk(f.geometry_shader);
    out.tessellation_shader = bool_to_vk(f.tessellation_shader);
    out.sample_rate_shading = bool_to_vk(f.sample_rate_shading);
    out.dual_src_blend = bool_to_vk(f.dual_src_blend);
    out.logic_op = bool_to_vk(f.logic_op);
    out.multi_draw_indirect = bool_to_vk(f.multi_draw_indirect);
    out.draw_indirect_first_instance = bool_to_vk(f.draw_indirect_first_instance);
    out.depth_clamp = bool_to_vk(f.depth_clamp);
    out.depth_bias_clamp = bool_to_vk(f.depth_bias_clamp);
    out.fill_mode_non_solid = bool_to_vk(f.fill_mode_non_solid);
    out.depth_bounds = bool_to_vk(f.depth_bounds);
    out.wide_lines = bool_to_vk(f.wide_lines);
    out.large_points = bool_to_vk(f.large_points);
    out.alpha_to_one = bool_to_vk(f.alpha_to_one);
    out.multi_viewport = bool_to_vk(f.multi_viewport);
    out.sampler_anisotropy = bool_to_vk(f.sampler_anisotropy);
    out.texture_compression_etc2 = bool_to_vk(f.texture_compression_etc2);
    out.texture_compression_astc_ldr = bool_to_vk(f.texture_compression_astc_ldr);
    out.texture_compression_bc = bool_to_vk(f.texture_compression_bc);
    out.occlusion_query_precise = bool_to_vk(f.occlusion_query_precise);
    out.pipeline_statistics_query = bool_to_vk(f.pipeline_statistics_query);
    out.vertex_pipeline_stores_and_atomics = bool_to_vk(f.vertex_pipeline_stores_and_atomics);
    out.fragment_stores_and_atomics = bool_to_vk(f.fragment_stores_and_atomics);
    out.shader_tessellation_and_geometry_point_size =
        bool_to_vk(f.shader_tessellation_and_geometry_point_size);
    out.shader_image_gather_extended = bool_to_vk(f.shader_image_gather_extended);
    out.shader_storage_image_extended_formats = bool_to_vk(f.shader_storage_image_extended_formats);
    out.shader_storage_image_multisample = bool_to_vk(f.shader_storage_image_multisample);
    out.shader_storage_image_read_without_format =
        bool_to_vk(f.shader_storage_image_read_without_format);
    out.shader_storage_image_write_without_format =
        bool_to_vk(f.shader_storage_image_write_without_format);
    out.shader_uniform_buffer_array_dynamic_indexing =
        bool_to_vk(f.shader_uniform_buffer_array_dynamic_indexing);
    out.shader_sampled_image_array_dynamic_indexing =
        bool_to_vk(f.shader_sampled_image_array_dynamic_indexing);
    out.shader_storage_buffer_array_dynamic_indexing =
        bool_to_vk(f.shader_storage_buffer_array_dynamic_indexing);
    out.shader_storage_image_array_dynamic_indexing =
        bool_to_vk(f.shader_storage_image_array_dynamic_indexing);
    out.shader_clip_distance = bool_to_vk(f.shader_clip_distance);
    out.shader_cull_distance = bool_to_vk(f.shader_cull_distance);
    out.shader_float64 = bool_to_vk(f.shader_float64);
    out.shader_int64 = bool_to_vk(f.shader_int64);
    out.shader_int16 = bool_to_vk(f.shader_int16);
    out.shader_resource_residency = bool_to_vk(f.shader_resource_residency);
    out.shader_resource_min_lod = bool_to_vk(f.shader_resource_min_lod);
    out.sparse_binding = bool_to_vk(f.sparse_binding);
    out.sparse_residency_buffer = bool_to_vk(f.sparse_residency_buffer);
    out.sparse_residency_image2_d = bool_to_vk(f.sparse_residency_image2_d);
    out.sparse_residency_image3_d = bool_to_vk(f.sparse_residency_image3_d);
    out.sparse_residency2_samples = bool_to_vk(f.sparse_residency2_samples);
    out.sparse_residency4_samples = bool_to_vk(f.sparse_residency4_samples);
    out.sparse_residency8_samples = bool_to_vk(f.sparse_residency8_samples);
    out.sparse_residency16_samples = bool_to_vk(f.sparse_residency16_samples);
    out.sparse_residency_aliased = bool_to_vk(f.sparse_residency_aliased);
    out.variable_multisample_rate = bool_to_vk(f.variable_multisample_rate);
    out.inherited_queries = bool_to_vk(f.inherited_queries);
    out
}

pub fn convert_limits(
    l: &vkremote_wire::types::PhysicalDeviceLimitsData,
) -> ash::vk::PhysicalDeviceLimits {
    let mut out = ash::vk::PhysicalDeviceLimits::default();
    out.max_image_dimension1_d = l.max_image_dimension1_d;
    out.max_image_dimension2_d = l.max_image_dimension2_d;
    out.max_image_dimension3_d = l.max_image_dimension3_d;
    out.max_image_dimension_cube = l.max_image_dimension_cube;
    out.max_image_array_layers = l.max_image_array_layers;
    out.max_texel_buffer_elements = l.max_texel_buffer_elements;
    out.max_uniform_buffer_range = l.max_uniform_buffer_range;
    out.max_storage_buffer_range = l.max_storage_buffer_range;
    out.max_push_constants_size = l.max_push_constants_size;
    out.max_memory_allocation_count = l.max_memory_allocation_count;
    out.max_sampler_allocation_count = l.max_sampler_allocation_count;
    out.buffer_image_granularity = l.buffer_image_granularity;
    out.sparse_address_space_size = l.sparse_address_space_size;
    out.max_bound_descriptor_sets = l.max_bound_descriptor_sets;
    out.max_per_stage_descriptor_samplers = l.max_per_stage_descriptor_samplers;
    out.max_per_stage_descriptor_uniform_buffers = l.max_per_stage_descriptor_uniform_buffers;
    out.max_per_stage_descriptor_storage_buffers = l.max_per_stage_descriptor_storage_buffers;
    out.max_per_stage_descriptor_sampled_images = l.max_per_stage_descriptor_sampled_images;
    out.max_per_stage_descriptor_storage_images = l.max_per_stage_descriptor_storage_images;
    out.max_per_stage_descriptor_input_attachments = l.max_per_stage_descriptor_input_attachments;
    out.max_per_stage_resources = l.max_per_stage_resources;
    out.max_descriptor_set_samplers = l.max_descriptor_set_samplers;
    out.max_descriptor_set_uniform_buffers = l.max_descriptor_set_uniform_buffers;
    out.max_descriptor_set_uniform_buffers_dynamic = l.max_descriptor_set_uniform_buffers_dynamic;
    out.max_descriptor_set_storage_buffers = l.max_descriptor_set_storage_buffers;
    out.max_descriptor_set_storage_buffers_dynamic = l.max_descriptor_set_storage_buffers_dynamic;
    out.max_descriptor_set_sampled_images = l.max_descriptor_set_sampled_images;
    out.max_descriptor_set_storage_images = l.max_descriptor_set_storage_images;
    out.max_descriptor_set_input_attachments = l.max_descriptor_set_input_attachments;
    out.max_vertex_input_attributes = l.max_vertex_input_attributes;
    out.max_vertex_input_bindings = l.max_vertex_input_bindings;
    out.max_vertex_input_attribute_offset = l.max_vertex_input_attribute_offset;
    out.max_vertex_input_binding_stride = l.max_vertex_input_binding_stride;
    out.max_vertex_output_components = l.max_vertex_output_components;
    out.max_tessellation_generation_level = l.max_tessellation_generation_level;
    out.max_tessellation_patch_size = l.max_tessellation_patch_size;
    out.max_tessellation_control_per_vertex_input_components =
        l.max_tessellation_control_per_vertex_input_components;
    out.max_tessellation_control_per_vertex_output_components =
        l.max_tessellation_control_per_vertex_output_components;
    out.max_tessellation_control_per_patch_output_components =
        l.max_tessellation_control_per_patch_output_components;
    out.max_tessellation_control_total_output_components =
        l.max_tessellation_control_total_output_components;
    out.max_tessellation_evaluation_input_components =
        l.max_tessellation_evaluation_input_components;
    out.max_tessellation_evaluation_output_components =
        l.max_tessellation_evaluation_output_components;
    out.max_geometry_shader_invocations = l.max_geometry_shader_invocations;
    out.max_geometry_input_components = l.max_geometry_input_components;
    out.max_geometry_output_components = l.max_geometry_output_components;
    out.max_geometry_output_vertices = l.max_geometry_output_vertices;
    out.max_geometry_total_output_components = l.max_geometry_total_output_components;
    out.max_fragment_input_components = l.max_fragment_input_components;
    out.max_fragment_output_attachments = l.max_fragment_output_attachments;
    out.max_fragment_dual_src_attachments = l.max_fragment_dual_src_attachments;
    out.max_fragment_combined_output_resources = l.max_fragment_combined_output_resources;
    out.max_compute_shared_memory_size = l.max_compute_shared_memory_size;
    out.max_compute_work_group_count = l.max_compute_work_group_count;
    out.max_compute_work_group_invocations = l.max_compute_work_group_invocations;
    out.max_compute_work_group_size = l.max_compute_work_group_size;
    out.sub_pixel_precision_bits = l.sub_pixel_precision_bits;
    out.sub_texel_precision_bits = l.sub_texel_precision_bits;
    out.mipmap_precision_bits = l.mipmap_precision_bits;
    out.max_draw_indexed_index_value = l.max_draw_indexed_index_value;
    out.max_draw_indirect_count = l.max_draw_indirect_count;
    out.max_sampler_lod_bias = f32::from_bits(l.max_sampler_lod_bias_bits);
    out.max_sampler_anisotropy = f32::from_bits(l.max_sampler_anisotropy_bits);
    out.max_viewports = l.max_viewports;
    out.max_viewport_dimensions = l.max_viewport_dimensions;
    out.viewport_bounds_range = [
        f32::from_bits(l.viewport_bounds_range_bits[0]),
        f32::from_bits(l.viewport_bounds_range_bits[1]),
    ];
    out.viewport_sub_pixel_bits = l.viewport_sub_pixel_bits;
    out.min_memory_map_alignment = l.min_memory_map_alignment;
    out.min_texel_buffer_offset_alignment = l.min_texel_buffer_offset_alignment;
    out.min_uniform_buffer_offset_alignment = l.min_uniform_buffer_offset_alignment;
    out.min_storage_buffer_offset_alignment = l.min_storage_buffer_offset_alignment;
    out.min_texel_offset = l.min_texel_offset;
    out.max_texel_offset = l.max_texel_offset;
    out.min_texel_gather_offset = l.min_texel_gather_offset;
    out.max_texel_gather_offset = l.max_texel_gather_offset;
    out.min_interpolation_offset = f32::from_bits(l.min_interpolation_offset_bits);
    out.max_interpolation_offset = f32::from_bits(l.max_interpolation_offset_bits);
    out.sub_pixel_interpolation_offset_bits = l.sub_pixel_interpolation_offset_bits;
    out.max_framebuffer_width = l.max_framebuffer_width;
    out.max_framebuffer_height = l.max_framebuffer_height;
    out.max_framebuffer_layers = l.max_framebuffer_layers;
    out.framebuffer_color_sample_counts =
        ash::vk::SampleCountFlags::from_raw(l.framebuffer_color_sample_counts);
    out.framebuffer_depth_sample_counts =
        ash::vk::SampleCountFlags::from_raw(l.framebuffer_depth_sample_counts);
    out.framebuffer_stencil_sample_counts =
        ash::vk::SampleCountFlags::from_raw(l.framebuffer_stencil_sample_counts);
    out.framebuffer_no_attachments_sample_counts =
        ash::vk::SampleCountFlags::from_raw(l.framebuffer_no_attachments_sample_counts);
    out.max_color_attachments = l.max_color_attachments;
    out.sampled_image_color_sample_counts =
        ash::vk::SampleCountFlags::from_raw(l.sampled_image_color_sample_counts);
    out.sampled_image_integer_sample_counts =
        ash::vk::SampleCountFlags::from_raw(l.sampled_image_integer_sample_counts);
    out.sampled_image_depth_sample_counts =
        ash::vk::SampleCountFlags::from_raw(l.sampled_image_depth_sample_counts);
    out.sampled_image_stencil_sample_counts =
        ash::vk::SampleCountFlags::from_raw(l.sampled_image_stencil_sample_counts);
    out.storage_image_sample_counts =
        ash::vk::SampleCountFlags::from_raw(l.storage_image_sample_counts);
    out.max_sample_mask_words = l.max_sample_mask_words;
    out.timestamp_compute_and_graphics = bool_to_vk(l.timestamp_compute_and_graphics);
    out.timestamp_period = f32::from_bits(l.timestamp_period_bits);
    out.max_clip_distances = l.max_clip_distances;
    out.max_cull_distances = l.max_cull_distances;
    out.max_combined_clip_and_cull_distances = l.max_combined_clip_and_cull_distances;
    out.discrete_queue_priorities = l.discrete_queue_priorities;
    out.point_size_range = [
        f32::from_bits(l.point_size_range_bits[0]),
        f32::from_bits(l.point_size_range_bits[1]),
    ];
    out.line_width_range = [
        f32::from_bits(l.line_width_range_bits[0]),
        f32::from_bits(l.line_width_range_bits[1]),
    ];
    out.point_size_granularity = f32::from_bits(l.point_size_granularity_bits);
    out.line_width_granularity = f32::from_bits(l.line_width_granularity_bits);
    out.strict_lines = bool_to_vk(l.strict_lines);
    out.standard_sample_locations = bool_to_vk(l.standard_sample_locations);
    out.optimal_buffer_copy_offset_alignment = l.optimal_buffer_copy_offset_alignment;
    out.optimal_buffer_copy_row_pitch_alignment = l.optimal_buffer_copy_row_pitch_alignment;
    out.non_coherent_atom_size = l.non_coherent_atom_size;

    // Sane fallbacks for robust Vulkan compliance
    if out.max_storage_buffer_range == 0 {
        out.max_storage_buffer_range = 1024 * 1024 * 1024; // 1 GB
    }
    if out.max_uniform_buffer_range == 0 {
        out.max_uniform_buffer_range = 65536;
    }
    if out.max_bound_descriptor_sets == 0 {
        out.max_bound_descriptor_sets = 8;
    }
    if out.max_per_stage_descriptor_uniform_buffers == 0 {
        out.max_per_stage_descriptor_uniform_buffers = 16;
    }
    if out.max_per_stage_descriptor_storage_buffers == 0 {
        out.max_per_stage_descriptor_storage_buffers = 16;
    }
    if out.max_per_stage_descriptor_sampled_images == 0 {
        out.max_per_stage_descriptor_sampled_images = 128;
    }
    if out.max_per_stage_descriptor_storage_images == 0 {
        out.max_per_stage_descriptor_storage_images = 16;
    }
    if out.max_per_stage_descriptor_samplers == 0 {
        out.max_per_stage_descriptor_samplers = 16;
    }
    if out.max_per_stage_resources == 0 {
        out.max_per_stage_resources = 1024;
    }
    if out.max_descriptor_set_samplers == 0 {
        out.max_descriptor_set_samplers = 1024;
    }
    if out.max_descriptor_set_uniform_buffers == 0 {
        out.max_descriptor_set_uniform_buffers = 1024;
    }
    if out.max_descriptor_set_storage_buffers == 0 {
        out.max_descriptor_set_storage_buffers = 1024;
    }
    if out.max_descriptor_set_sampled_images == 0 {
        out.max_descriptor_set_sampled_images = 1024;
    }
    if out.max_descriptor_set_storage_images == 0 {
        out.max_descriptor_set_storage_images = 1024;
    }
    if out.max_vertex_input_attributes == 0 {
        out.max_vertex_input_attributes = 32;
    }
    if out.max_vertex_input_bindings == 0 {
        out.max_vertex_input_bindings = 32;
    }

    out
}

#[no_mangle]
pub unsafe extern "system" fn vkEnumerateInstanceVersion(
    p_api_version: *mut u32,
) -> ash::vk::Result {
    if p_api_version.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    // SAFETY: caller provided a valid non-null pointer to u32
    *p_api_version = ash::vk::make_api_version(0, 1, 3, 0);
    ash::vk::Result::SUCCESS
}

fn fill_extension_properties(
    available: &[(&[u8], u32)],
    p_property_count: *mut u32,
    p_properties: *mut ash::vk::ExtensionProperties,
) -> ash::vk::Result {
    if p_property_count.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let count = available.len() as u32;
    if p_properties.is_null() {
        unsafe {
            *p_property_count = count;
        }
        return ash::vk::Result::SUCCESS;
    }
    let user_count = unsafe { *p_property_count };
    let to_write = (user_count as usize).min(available.len());
    for i in 0..to_write {
        let (name, ver) = available[i];
        let prop = unsafe { &mut *p_properties.add(i) };
        *prop = ash::vk::ExtensionProperties::default();
        for (j, &b) in name.iter().enumerate() {
            if j < prop.extension_name.len() {
                prop.extension_name[j] = b as std::os::raw::c_char;
            }
        }
        prop.spec_version = ver;
    }
    unsafe {
        *p_property_count = to_write as u32;
    }
    if (user_count as usize) < available.len() {
        ash::vk::Result::INCOMPLETE
    } else {
        ash::vk::Result::SUCCESS
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkEnumerateInstanceExtensionProperties(
    _p_layer_name: *const std::os::raw::c_char,
    p_property_count: *mut u32,
    p_properties: *mut ash::vk::ExtensionProperties,
) -> ash::vk::Result {
    const INSTANCE_EXTS: &[(&[u8], u32)] = &[
        (b"VK_KHR_get_physical_device_properties2\0", 2),
        (b"VK_KHR_surface\0", 25),
        (b"VK_KHR_win32_surface\0", 6),
        (b"VK_KHR_wayland_surface\0", 6),
        (b"VK_KHR_xcb_surface\0", 6),
        (b"VK_KHR_xlib_surface\0", 6),
    ];
    fill_extension_properties(INSTANCE_EXTS, p_property_count, p_properties)
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateInstance(
    p_create_info: *const ash::vk::InstanceCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_instance: *mut ash::vk::Instance,
) -> ash::vk::Result {
    if p_create_info.is_null() || p_instance.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let client = match Client::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "[vkremote-icd] vkCreateInstance: failed to initialize client: {}",
                e
            );
            tracing::warn!("vkCreateInstance: failed to initialize client: {}", e);
            return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
        }
    };

    let snapshot = match client.get_or_fetch_snapshot() {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "[vkremote-icd] vkCreateInstance: failed to fetch device snapshot: {}",
                e
            );
            tracing::warn!("vkCreateInstance: failed to fetch device snapshot: {}", e);
            return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
        }
    };

    let instance = InstanceDispatchable::new(client, snapshot);
    let ptr = Box::into_raw(instance);
    // SAFETY: caller provided a valid pointer to ash::vk::Instance
    *p_instance = InstanceDispatchable::to_vk(ptr);

    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroyInstance(
    instance: ash::vk::Instance,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if instance.is_null() {
        return;
    }
    // SAFETY: instance was allocated by vkCreateInstance as Box<InstanceDispatchable>
    let raw = instance.as_raw() as usize as *mut InstanceDispatchable;
    let inst = Box::from_raw(raw);
    let phys = inst
        .physical_device
        .swap(std::ptr::null_mut(), std::sync::atomic::Ordering::AcqRel);
    if !phys.is_null() {
        drop(Box::from_raw(phys));
    }
    drop(inst);
}

#[no_mangle]
pub unsafe extern "system" fn vkEnumeratePhysicalDevices(
    instance: ash::vk::Instance,
    p_physical_device_count: *mut u32,
    p_physical_devices: *mut ash::vk::PhysicalDevice,
) -> ash::vk::Result {
    if instance.is_null() || p_physical_device_count.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: caller passed a valid instance handle
    let inst = InstanceDispatchable::from_vk_mut(instance);
    let count = 1u32;

    if p_physical_devices.is_null() {
        *p_physical_device_count = count;
        return ash::vk::Result::SUCCESS;
    }

    if *p_physical_device_count < count {
        *p_physical_device_count = count;
        return ash::vk::Result::INCOMPLETE;
    }

    let mut ptr = inst.physical_device.load(std::sync::atomic::Ordering::Acquire);
    if ptr.is_null() {
        let phys = PhysicalDeviceDispatchable::new(
            inst as *mut _,
            Arc::clone(&inst.client),
            Arc::clone(&inst.snapshot),
        );
        let fresh = Box::into_raw(phys);
        match inst.physical_device.compare_exchange(
            std::ptr::null_mut(),
            fresh,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        ) {
            Ok(_) => ptr = fresh,
            Err(existing) => {
                // Another thread enumerated first; keep its handle.
                drop(Box::from_raw(fresh));
                ptr = existing;
            }
        }
    }
    *p_physical_devices = PhysicalDeviceDispatchable::to_vk(ptr);
    *p_physical_device_count = count;

    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceProperties(
    physical_device: ash::vk::PhysicalDevice,
    p_properties: *mut ash::vk::PhysicalDeviceProperties,
) {
    if physical_device.is_null() || p_properties.is_null() {
        return;
    }

    // SAFETY: physical_device is a valid handle created by this ICD
    let phys = PhysicalDeviceDispatchable::from_vk(physical_device);
    let props = &mut *p_properties;

    props.api_version = phys.snapshot.properties.api_version;
    props.driver_version = phys.snapshot.properties.driver_version;
    props.vendor_id = phys.snapshot.properties.vendor_id;
    props.device_id = phys.snapshot.properties.device_id;
    props.device_type =
        ash::vk::PhysicalDeviceType::from_raw(phys.snapshot.properties.device_type as i32);
    props.pipeline_cache_uuid = phys.snapshot.properties.pipeline_cache_uuid;

    // Requirement: format name as "<Original Name> (Remote)"
    let name = format!("{} (Remote)", phys.snapshot.properties.device_name);
    let bytes = name.as_bytes();
    let copy_len = bytes.len().min(props.device_name.len() - 1);
    for i in 0..copy_len {
        props.device_name[i] = bytes[i] as std::os::raw::c_char;
    }
    props.device_name[copy_len] = 0;

    props.limits = convert_limits(&phys.snapshot.properties.limits);
    eprintln!(
        "[vkremote-icd] limits: max_per_stage_sampled_images={}, max_per_stage_resources={}, max_descriptor_set_sampled_images={}, max_per_stage_samplers={}",
        props.limits.max_per_stage_descriptor_sampled_images,
        props.limits.max_per_stage_resources,
        props.limits.max_descriptor_set_sampled_images,
        props.limits.max_per_stage_descriptor_samplers,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceProperties2(
    physical_device: ash::vk::PhysicalDevice,
    p_properties: *mut ash::vk::PhysicalDeviceProperties2,
) {
    if p_properties.is_null() {
        return;
    }
    // SAFETY: caller passed a valid PhysicalDeviceProperties2 struct pointer
    vkGetPhysicalDeviceProperties(physical_device, &mut (*p_properties).properties);

    let phys = PhysicalDeviceDispatchable::from_vk(physical_device);
    let mut curr = (*p_properties).p_next as *mut ash::vk::BaseOutStructure;
    while !curr.is_null() {
        match (*curr).s_type {
            ash::vk::StructureType::PHYSICAL_DEVICE_DRIVER_PROPERTIES => {
                let d = &mut *(curr as *mut ash::vk::PhysicalDeviceDriverProperties);
                d.driver_id = ash::vk::DriverId::NVIDIA_PROPRIETARY;
                let name = b"NVIDIA\0";
                for (dst, &src) in d.driver_name.iter_mut().zip(name.iter()) {
                    *dst = src as std::os::raw::c_char;
                }
                let info = b"vkremote\0";
                for (dst, &src) in d.driver_info.iter_mut().zip(info.iter()) {
                    *dst = src as std::os::raw::c_char;
                }
                // Blender rejects NVIDIA drivers reporting conformance < 1.3.7.2 as having
                // known issues. The host NVIDIA 580.x driver is 1.4 conformant.
                d.conformance_version = ash::vk::ConformanceVersion {
                    major: 1,
                    minor: 4,
                    subminor: 0,
                    patch: 0,
                };
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_ID_PROPERTIES => {
                let id = &mut *(curr as *mut ash::vk::PhysicalDeviceIDProperties);
                id.device_uuid = phys.snapshot.properties.pipeline_cache_uuid;
                id.driver_uuid = phys.snapshot.properties.pipeline_cache_uuid;
                id.device_luid = [0; 8];
                id.device_node_mask = 1;
                id.device_luid_valid = ash::vk::FALSE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES => {
                let v = &mut *(curr as *mut ash::vk::PhysicalDeviceVulkan11Properties);
                v.subgroup_size = 32;
                v.subgroup_supported_stages = ash::vk::ShaderStageFlags::ALL;
                v.subgroup_supported_operations = ash::vk::SubgroupFeatureFlags::from_raw(0x1FF);
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES => {
                let v = &mut *(curr as *mut ash::vk::PhysicalDeviceVulkan12Properties);
                v.driver_id = ash::vk::DriverId::NVIDIA_PROPRIETARY;
                v.conformance_version = ash::vk::ConformanceVersion {
                    major: 1,
                    minor: 4,
                    subminor: 0,
                    patch: 0,
                };
                v.max_update_after_bind_descriptors_in_all_pools = 1048576;
                v.shader_uniform_buffer_array_non_uniform_indexing_native = ash::vk::TRUE;
                v.shader_sampled_image_array_non_uniform_indexing_native = ash::vk::TRUE;
                v.shader_storage_buffer_array_non_uniform_indexing_native = ash::vk::TRUE;
                v.shader_storage_image_array_non_uniform_indexing_native = ash::vk::TRUE;
                v.shader_input_attachment_array_non_uniform_indexing_native = ash::vk::TRUE;
                v.robust_buffer_access_update_after_bind = ash::vk::TRUE;
                v.quad_divergent_implicit_lod = ash::vk::TRUE;
                v.max_per_stage_descriptor_update_after_bind_samplers = 1048576;
                v.max_per_stage_descriptor_update_after_bind_uniform_buffers = 1048576;
                v.max_per_stage_descriptor_update_after_bind_storage_buffers = 1048576;
                v.max_per_stage_descriptor_update_after_bind_sampled_images = 1048576;
                v.max_per_stage_descriptor_update_after_bind_storage_images = 1048576;
                v.max_per_stage_descriptor_update_after_bind_input_attachments = 1048576;
                v.max_per_stage_update_after_bind_resources = 1048576;
                v.max_descriptor_set_update_after_bind_samplers = 1048576;
                v.max_descriptor_set_update_after_bind_uniform_buffers = 1048576;
                v.max_descriptor_set_update_after_bind_storage_buffers = 1048576;
                v.max_descriptor_set_update_after_bind_sampled_images = 1048576;
                v.max_descriptor_set_update_after_bind_storage_images = 1048576;
                v.max_descriptor_set_update_after_bind_input_attachments = 1048576;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES => {
                let d = &mut *(curr as *mut ash::vk::PhysicalDeviceDescriptorIndexingProperties);
                d.max_update_after_bind_descriptors_in_all_pools = 1048576;
                d.shader_uniform_buffer_array_non_uniform_indexing_native = ash::vk::TRUE;
                d.shader_sampled_image_array_non_uniform_indexing_native = ash::vk::TRUE;
                d.shader_storage_buffer_array_non_uniform_indexing_native = ash::vk::TRUE;
                d.shader_storage_image_array_non_uniform_indexing_native = ash::vk::TRUE;
                d.shader_input_attachment_array_non_uniform_indexing_native = ash::vk::TRUE;
                d.robust_buffer_access_update_after_bind = ash::vk::TRUE;
                d.quad_divergent_implicit_lod = ash::vk::TRUE;
                d.max_per_stage_descriptor_update_after_bind_samplers = 1048576;
                d.max_per_stage_descriptor_update_after_bind_uniform_buffers = 1048576;
                d.max_per_stage_descriptor_update_after_bind_storage_buffers = 1048576;
                d.max_per_stage_descriptor_update_after_bind_sampled_images = 1048576;
                d.max_per_stage_descriptor_update_after_bind_storage_images = 1048576;
                d.max_per_stage_descriptor_update_after_bind_input_attachments = 1048576;
                d.max_per_stage_update_after_bind_resources = 1048576;
                d.max_descriptor_set_update_after_bind_samplers = 1048576;
                d.max_descriptor_set_update_after_bind_uniform_buffers = 1048576;
                d.max_descriptor_set_update_after_bind_storage_buffers = 1048576;
                d.max_descriptor_set_update_after_bind_sampled_images = 1048576;
                d.max_descriptor_set_update_after_bind_storage_images = 1048576;
                d.max_descriptor_set_update_after_bind_input_attachments = 1048576;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES => {
                let v = &mut *(curr as *mut ash::vk::PhysicalDeviceVulkan13Properties);
                v.max_buffer_size = 1024 * 1024 * 1024 * 16; // 16 GB
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES => {
                let m = &mut *(curr as *mut ash::vk::PhysicalDeviceMaintenance3Properties);
                m.max_memory_allocation_size = 1024 * 1024 * 1024 * 16; // 16 GB
                m.max_per_set_descriptors = 1048576;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES => {
                let m = &mut *(curr as *mut ash::vk::PhysicalDeviceMaintenance4Properties);
                m.max_buffer_size = 1024 * 1024 * 1024 * 16; // 16 GB
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES => {
                let m = &mut *(curr as *mut ash::vk::PhysicalDeviceMultiviewProperties);
                m.max_multiview_view_count = 6;
                m.max_multiview_instance_index = 134217727;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_SUBGROUP_PROPERTIES => {
                let s = &mut *(curr as *mut ash::vk::PhysicalDeviceSubgroupProperties);
                s.subgroup_size = 32;
                s.supported_stages = ash::vk::ShaderStageFlags::ALL;
                s.supported_operations = ash::vk::SubgroupFeatureFlags::from_raw(0x1FF);
                s.quad_operations_in_all_stages = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES => {
                let s = &mut *(curr as *mut ash::vk::PhysicalDeviceSubgroupSizeControlProperties);
                s.min_subgroup_size = 32;
                s.max_subgroup_size = 32;
                s.max_compute_workgroup_subgroups = 32;
                s.required_subgroup_size_stages = ash::vk::ShaderStageFlags::ALL;
            }
            other => {
                eprintln!("[vkremote-icd] unhandled properties s_type: {:?}", other);
            }
        }
        curr = (*curr).p_next;
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceProperties2KHR(
    physical_device: ash::vk::PhysicalDevice,
    p_properties: *mut ash::vk::PhysicalDeviceProperties2,
) {
    vkGetPhysicalDeviceProperties2(physical_device, p_properties);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceFeatures(
    physical_device: ash::vk::PhysicalDevice,
    p_features: *mut ash::vk::PhysicalDeviceFeatures,
) {
    if physical_device.is_null() || p_features.is_null() {
        return;
    }
    // SAFETY: physical_device is a valid handle created by this ICD
    let phys = PhysicalDeviceDispatchable::from_vk(physical_device);
    *p_features = convert_features(&phys.snapshot.features);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceFeatures2(
    physical_device: ash::vk::PhysicalDevice,
    p_features: *mut ash::vk::PhysicalDeviceFeatures2,
) {
    if p_features.is_null() {
        return;
    }
    // SAFETY: caller passed a valid PhysicalDeviceFeatures2 struct pointer
    vkGetPhysicalDeviceFeatures(physical_device, &mut (*p_features).features);

    let mut curr = (*p_features).p_next as *mut ash::vk::BaseOutStructure;
    while !curr.is_null() {
        match (*curr).s_type {
            ash::vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceVulkan11Features);
                f.storage_buffer16_bit_access = ash::vk::TRUE;
                f.uniform_and_storage_buffer16_bit_access = ash::vk::TRUE;
                f.storage_push_constant16 = ash::vk::TRUE;
                f.storage_input_output16 = ash::vk::TRUE;
                f.multiview = ash::vk::TRUE;
                f.multiview_geometry_shader = ash::vk::TRUE;
                f.multiview_tessellation_shader = ash::vk::TRUE;
                f.variable_pointers_storage_buffer = ash::vk::TRUE;
                f.variable_pointers = ash::vk::TRUE;
                f.protected_memory = ash::vk::FALSE;
                f.sampler_ycbcr_conversion = ash::vk::TRUE;
                f.shader_draw_parameters = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceVulkan12Features);
                f.sampler_mirror_clamp_to_edge = ash::vk::TRUE;
                f.draw_indirect_count = ash::vk::TRUE;
                f.storage_buffer8_bit_access = ash::vk::TRUE;
                f.uniform_and_storage_buffer8_bit_access = ash::vk::TRUE;
                f.storage_push_constant8 = ash::vk::TRUE;
                f.shader_buffer_int64_atomics = ash::vk::TRUE;
                f.shader_shared_int64_atomics = ash::vk::TRUE;
                f.shader_float16 = ash::vk::TRUE;
                f.shader_int8 = ash::vk::TRUE;
                f.descriptor_indexing = ash::vk::TRUE;
                f.shader_input_attachment_array_dynamic_indexing = ash::vk::TRUE;
                f.shader_uniform_texel_buffer_array_dynamic_indexing = ash::vk::TRUE;
                f.shader_storage_texel_buffer_array_dynamic_indexing = ash::vk::TRUE;
                f.shader_uniform_buffer_array_non_uniform_indexing = ash::vk::TRUE;
                f.shader_sampled_image_array_non_uniform_indexing = ash::vk::TRUE;
                f.shader_storage_buffer_array_non_uniform_indexing = ash::vk::TRUE;
                f.shader_storage_image_array_non_uniform_indexing = ash::vk::TRUE;
                f.shader_input_attachment_array_non_uniform_indexing = ash::vk::TRUE;
                f.shader_uniform_texel_buffer_array_non_uniform_indexing = ash::vk::TRUE;
                f.shader_storage_texel_buffer_array_non_uniform_indexing = ash::vk::TRUE;
                f.descriptor_binding_uniform_buffer_update_after_bind = ash::vk::TRUE;
                f.descriptor_binding_sampled_image_update_after_bind = ash::vk::TRUE;
                f.descriptor_binding_storage_image_update_after_bind = ash::vk::TRUE;
                f.descriptor_binding_storage_buffer_update_after_bind = ash::vk::TRUE;
                f.descriptor_binding_uniform_texel_buffer_update_after_bind = ash::vk::TRUE;
                f.descriptor_binding_storage_texel_buffer_update_after_bind = ash::vk::TRUE;
                f.descriptor_binding_update_unused_while_pending = ash::vk::TRUE;
                f.descriptor_binding_partially_bound = ash::vk::TRUE;
                f.descriptor_binding_variable_descriptor_count = ash::vk::TRUE;
                f.runtime_descriptor_array = ash::vk::TRUE;
                f.sampler_filter_minmax = ash::vk::TRUE;
                f.scalar_block_layout = ash::vk::TRUE;
                f.imageless_framebuffer = ash::vk::TRUE;
                f.uniform_buffer_standard_layout = ash::vk::TRUE;
                f.shader_subgroup_extended_types = ash::vk::TRUE;
                f.separate_depth_stencil_layouts = ash::vk::TRUE;
                f.host_query_reset = ash::vk::TRUE;
                f.timeline_semaphore = ash::vk::TRUE;
                f.buffer_device_address = ash::vk::TRUE;
                f.buffer_device_address_capture_replay = ash::vk::FALSE;
                f.buffer_device_address_multi_device = ash::vk::FALSE;
                f.vulkan_memory_model = ash::vk::TRUE;
                f.vulkan_memory_model_device_scope = ash::vk::TRUE;
                f.vulkan_memory_model_availability_visibility_chains = ash::vk::TRUE;
                f.shader_output_viewport_index = ash::vk::TRUE;
                f.shader_output_layer = ash::vk::TRUE;
                f.subgroup_broadcast_dynamic_id = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceVulkan13Features);
                f.robust_image_access = ash::vk::TRUE;
                f.inline_uniform_block = ash::vk::TRUE;
                f.descriptor_binding_inline_uniform_block_update_after_bind = ash::vk::TRUE;
                f.pipeline_creation_cache_control = ash::vk::TRUE;
                f.private_data = ash::vk::TRUE;
                f.shader_demote_to_helper_invocation = ash::vk::TRUE;
                f.shader_terminate_invocation = ash::vk::TRUE;
                f.subgroup_size_control = ash::vk::TRUE;
                f.compute_full_subgroups = ash::vk::TRUE;
                // vkQueueSubmit2 / vkCmdPipelineBarrier2 are not implemented.
                f.synchronization2 = ash::vk::FALSE;
                f.texture_compression_astc_hdr = ash::vk::FALSE;
                f.shader_zero_initialize_workgroup_memory = ash::vk::TRUE;
                // vkCmdBeginRendering / vkCmdEndRendering are not implemented.
                f.dynamic_rendering = ash::vk::FALSE;
                f.shader_integer_dot_product = ash::vk::TRUE;
                f.maintenance4 = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceSynchronization2Features);
                f.synchronization2 = ash::vk::FALSE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceDynamicRenderingFeatures);
                f.dynamic_rendering = ash::vk::FALSE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT => {
                // Advertised so Blender's capability gate passes. The pipeline-state
                // struct is not forwarded yet, so the server uses first-vertex convention.
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceProvokingVertexFeaturesEXT);
                f.provoking_vertex_last = ash::vk::TRUE;
                f.transform_feedback_preserves_provoking_vertex = ash::vk::FALSE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT => {
                let f =
                    &mut *(curr as *mut ash::vk::PhysicalDeviceVertexAttributeDivisorFeaturesEXT);
                f.vertex_attribute_instance_rate_divisor = ash::vk::TRUE;
                f.vertex_attribute_instance_rate_zero_divisor = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceTimelineSemaphoreFeatures);
                f.timeline_semaphore = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceHostQueryResetFeatures);
                f.host_query_reset = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceShaderDrawParametersFeatures);
                f.shader_draw_parameters = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceDescriptorIndexingFeatures);
                f.shader_input_attachment_array_dynamic_indexing = ash::vk::TRUE;
                f.shader_uniform_texel_buffer_array_dynamic_indexing = ash::vk::TRUE;
                f.shader_storage_texel_buffer_array_dynamic_indexing = ash::vk::TRUE;
                f.shader_uniform_buffer_array_non_uniform_indexing = ash::vk::TRUE;
                f.shader_sampled_image_array_non_uniform_indexing = ash::vk::TRUE;
                f.shader_storage_buffer_array_non_uniform_indexing = ash::vk::TRUE;
                f.shader_storage_image_array_non_uniform_indexing = ash::vk::TRUE;
                f.shader_input_attachment_array_non_uniform_indexing = ash::vk::TRUE;
                f.shader_uniform_texel_buffer_array_non_uniform_indexing = ash::vk::TRUE;
                f.shader_storage_texel_buffer_array_non_uniform_indexing = ash::vk::TRUE;
                f.descriptor_binding_uniform_buffer_update_after_bind = ash::vk::TRUE;
                f.descriptor_binding_sampled_image_update_after_bind = ash::vk::TRUE;
                f.descriptor_binding_storage_image_update_after_bind = ash::vk::TRUE;
                f.descriptor_binding_storage_buffer_update_after_bind = ash::vk::TRUE;
                f.descriptor_binding_uniform_texel_buffer_update_after_bind = ash::vk::TRUE;
                f.descriptor_binding_storage_texel_buffer_update_after_bind = ash::vk::TRUE;
                f.descriptor_binding_update_unused_while_pending = ash::vk::TRUE;
                f.descriptor_binding_partially_bound = ash::vk::TRUE;
                f.descriptor_binding_variable_descriptor_count = ash::vk::TRUE;
                f.runtime_descriptor_array = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceVulkanMemoryModelFeatures);
                f.vulkan_memory_model = ash::vk::TRUE;
                f.vulkan_memory_model_device_scope = ash::vk::TRUE;
                f.vulkan_memory_model_availability_visibility_chains = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES => {
                let f = &mut *(curr
                    as *mut ash::vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures);
                f.shader_zero_initialize_workgroup_memory = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceMaintenance4Features);
                f.maintenance4 = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceShaderFloat16Int8Features);
                f.shader_float16 = ash::vk::TRUE;
                f.shader_int8 = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceShaderAtomicInt64Features);
                f.shader_buffer_int64_atomics = ash::vk::TRUE;
                f.shader_shared_int64_atomics = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceSamplerYcbcrConversionFeatures);
                f.sampler_ycbcr_conversion = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_MULTIVIEW_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceMultiviewFeatures);
                f.multiview = ash::vk::TRUE;
                f.multiview_geometry_shader = ash::vk::TRUE;
                f.multiview_tessellation_shader = ash::vk::TRUE;
            }
            ash::vk::StructureType::PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES => {
                let f = &mut *(curr as *mut ash::vk::PhysicalDeviceSubgroupSizeControlFeatures);
                f.subgroup_size_control = ash::vk::TRUE;
                f.compute_full_subgroups = ash::vk::TRUE;
            }
            other => {
                eprintln!("[vkremote-icd] unhandled features s_type: {:?}", other);
            }
        }
        curr = (*curr).p_next;
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceFeatures2KHR(
    physical_device: ash::vk::PhysicalDevice,
    p_features: *mut ash::vk::PhysicalDeviceFeatures2,
) {
    vkGetPhysicalDeviceFeatures2(physical_device, p_features);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceMemoryProperties(
    physical_device: ash::vk::PhysicalDevice,
    p_memory_properties: *mut ash::vk::PhysicalDeviceMemoryProperties,
) {
    if physical_device.is_null() || p_memory_properties.is_null() {
        return;
    }
    // SAFETY: physical_device is a valid handle created by this ICD
    let phys = PhysicalDeviceDispatchable::from_vk(physical_device);
    let mem = &mut *p_memory_properties;

    let types_len = phys
        .snapshot
        .memory_properties
        .memory_types
        .len()
        .min(mem.memory_types.len());
    mem.memory_type_count = types_len as u32;
    for i in 0..types_len {
        let t = &phys.snapshot.memory_properties.memory_types[i];
        mem.memory_types[i].property_flags =
            ash::vk::MemoryPropertyFlags::from_raw(t.property_flags);
        mem.memory_types[i].heap_index = t.heap_index;
    }

    let heaps_len = phys
        .snapshot
        .memory_properties
        .memory_heaps
        .len()
        .min(mem.memory_heaps.len());
    mem.memory_heap_count = heaps_len as u32;
    for i in 0..heaps_len {
        let h = &phys.snapshot.memory_properties.memory_heaps[i];
        mem.memory_heaps[i].size = h.size;
        mem.memory_heaps[i].flags = ash::vk::MemoryHeapFlags::from_raw(h.flags);
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceMemoryProperties2(
    physical_device: ash::vk::PhysicalDevice,
    p_memory_properties: *mut ash::vk::PhysicalDeviceMemoryProperties2,
) {
    if p_memory_properties.is_null() {
        return;
    }
    // SAFETY: caller passed a valid PhysicalDeviceMemoryProperties2 struct pointer
    vkGetPhysicalDeviceMemoryProperties(
        physical_device,
        &mut (*p_memory_properties).memory_properties,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceMemoryProperties2KHR(
    physical_device: ash::vk::PhysicalDevice,
    p_memory_properties: *mut ash::vk::PhysicalDeviceMemoryProperties2,
) {
    vkGetPhysicalDeviceMemoryProperties2(physical_device, p_memory_properties);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceQueueFamilyProperties(
    physical_device: ash::vk::PhysicalDevice,
    p_queue_family_property_count: *mut u32,
    p_queue_family_properties: *mut ash::vk::QueueFamilyProperties,
) {
    if physical_device.is_null() || p_queue_family_property_count.is_null() {
        return;
    }
    // SAFETY: physical_device is a valid handle created by this ICD
    let phys = PhysicalDeviceDispatchable::from_vk(physical_device);
    let total_count = phys.snapshot.queue_families.len() as u32;

    if p_queue_family_properties.is_null() {
        *p_queue_family_property_count = total_count;
        return;
    }

    let count_to_copy = (*p_queue_family_property_count).min(total_count);
    for i in 0..count_to_copy as usize {
        let qf = &phys.snapshot.queue_families[i];
        let dst = &mut *p_queue_family_properties.add(i);
        dst.queue_flags = ash::vk::QueueFlags::from_raw(qf.queue_flags);
        dst.queue_count = qf.queue_count;
        dst.timestamp_valid_bits = qf.timestamp_valid_bits;
        dst.min_image_transfer_granularity = ash::vk::Extent3D {
            width: qf.min_image_transfer_granularity[0],
            height: qf.min_image_transfer_granularity[1],
            depth: qf.min_image_transfer_granularity[2],
        };
    }
    *p_queue_family_property_count = count_to_copy;
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceQueueFamilyProperties2(
    physical_device: ash::vk::PhysicalDevice,
    p_queue_family_property_count: *mut u32,
    p_queue_family_properties: *mut ash::vk::QueueFamilyProperties2,
) {
    if physical_device.is_null() || p_queue_family_property_count.is_null() {
        return;
    }
    // SAFETY: physical_device is a valid handle created by this ICD
    let phys = PhysicalDeviceDispatchable::from_vk(physical_device);
    let total_count = phys.snapshot.queue_families.len() as u32;

    if p_queue_family_properties.is_null() {
        *p_queue_family_property_count = total_count;
        return;
    }

    let count_to_copy = (*p_queue_family_property_count).min(total_count);
    for i in 0..count_to_copy as usize {
        let qf = &phys.snapshot.queue_families[i];
        let dst = &mut (*p_queue_family_properties.add(i)).queue_family_properties;
        dst.queue_flags = ash::vk::QueueFlags::from_raw(qf.queue_flags);
        dst.queue_count = qf.queue_count;
        dst.timestamp_valid_bits = qf.timestamp_valid_bits;
        dst.min_image_transfer_granularity = ash::vk::Extent3D {
            width: qf.min_image_transfer_granularity[0],
            height: qf.min_image_transfer_granularity[1],
            depth: qf.min_image_transfer_granularity[2],
        };
    }
    *p_queue_family_property_count = count_to_copy;
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceQueueFamilyProperties2KHR(
    physical_device: ash::vk::PhysicalDevice,
    p_queue_family_property_count: *mut u32,
    p_queue_family_properties: *mut ash::vk::QueueFamilyProperties2,
) {
    vkGetPhysicalDeviceQueueFamilyProperties2(
        physical_device,
        p_queue_family_property_count,
        p_queue_family_properties,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceFormatProperties(
    _physical_device: ash::vk::PhysicalDevice,
    _format: ash::vk::Format,
    p_format_properties: *mut ash::vk::FormatProperties,
) {
    if p_format_properties.is_null() {
        return;
    }
    // SAFETY: caller provided a valid FormatProperties struct pointer
    let props = &mut *p_format_properties;
    props.linear_tiling_features = ash::vk::FormatFeatureFlags::TRANSFER_SRC
        | ash::vk::FormatFeatureFlags::TRANSFER_DST
        | ash::vk::FormatFeatureFlags::SAMPLED_IMAGE
        | ash::vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
        | ash::vk::FormatFeatureFlags::COLOR_ATTACHMENT
        | ash::vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
        | ash::vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
        | ash::vk::FormatFeatureFlags::BLIT_SRC
        | ash::vk::FormatFeatureFlags::BLIT_DST;
    props.optimal_tiling_features = props.linear_tiling_features
        | ash::vk::FormatFeatureFlags::STORAGE_IMAGE;
    props.buffer_features = ash::vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER
        | ash::vk::FormatFeatureFlags::UNIFORM_TEXEL_BUFFER
        | ash::vk::FormatFeatureFlags::VERTEX_BUFFER;
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceFormatProperties2(
    physical_device: ash::vk::PhysicalDevice,
    format: ash::vk::Format,
    p_format_properties: *mut ash::vk::FormatProperties2,
) {
    if p_format_properties.is_null() {
        return;
    }
    vkGetPhysicalDeviceFormatProperties(
        physical_device,
        format,
        &mut (*p_format_properties).format_properties,
    );
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceFormatProperties2KHR(
    physical_device: ash::vk::PhysicalDevice,
    format: ash::vk::Format,
    p_format_properties: *mut ash::vk::FormatProperties2,
) {
    vkGetPhysicalDeviceFormatProperties2(physical_device, format, p_format_properties);
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceImageFormatProperties(
    _physical_device: ash::vk::PhysicalDevice,
    _format: ash::vk::Format,
    _image_type: ash::vk::ImageType,
    _tiling: ash::vk::ImageTiling,
    _usage: ash::vk::ImageUsageFlags,
    _flags: ash::vk::ImageCreateFlags,
    p_image_format_properties: *mut ash::vk::ImageFormatProperties,
) -> ash::vk::Result {
    if p_image_format_properties.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let props = &mut *p_image_format_properties;
    props.max_extent = ash::vk::Extent3D {
        width: 4096,
        height: 4096,
        depth: 1,
    };
    props.max_mip_levels = 1;
    props.max_array_layers = 1;
    props.sample_counts = ash::vk::SampleCountFlags::TYPE_1;
    props.max_resource_size = 1 << 30;
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceSparseImageFormatProperties(
    _physical_device: ash::vk::PhysicalDevice,
    _format: ash::vk::Format,
    _type: ash::vk::ImageType,
    _samples: ash::vk::SampleCountFlags,
    _usage: ash::vk::ImageUsageFlags,
    _tiling: ash::vk::ImageTiling,
    p_property_count: *mut u32,
    _p_properties: *mut ash::vk::SparseImageFormatProperties,
) {
    if !p_property_count.is_null() {
        *p_property_count = 0;
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkEnumerateDeviceLayerProperties(
    _physical_device: ash::vk::PhysicalDevice,
    p_property_count: *mut u32,
    _p_properties: *mut ash::vk::LayerProperties,
) -> ash::vk::Result {
    if p_property_count.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    *p_property_count = 0;
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkEnumerateDeviceExtensionProperties(
    _physical_device: ash::vk::PhysicalDevice,
    _p_layer_name: *const std::os::raw::c_char,
    p_property_count: *mut u32,
    p_properties: *mut ash::vk::ExtensionProperties,
) -> ash::vk::Result {
    const DEVICE_EXTS: &[(&[u8], u32)] = &[
        (b"VK_KHR_swapchain\0", 70),
        (b"VK_KHR_timeline_semaphore\0", 2),
        (b"VK_KHR_buffer_device_address\0", 1),
        (b"VK_KHR_shader_draw_parameters\0", 1),
        (b"VK_EXT_host_query_reset\0", 1),
        (b"VK_KHR_host_query_reset\0", 1),
        (b"VK_EXT_descriptor_indexing\0", 2),
        (b"VK_KHR_create_renderpass2\0", 1),
        (b"VK_KHR_depth_stencil_resolve\0", 1),
        (b"VK_KHR_maintenance1\0", 2),
        (b"VK_KHR_maintenance2\0", 1),
        (b"VK_KHR_maintenance3\0", 1),
        (b"VK_KHR_maintenance4\0", 2),
        (b"VK_EXT_vertex_attribute_divisor\0", 3),
        (b"VK_EXT_debug_utils\0", 2),
        (b"VK_EXT_provoking_vertex\0", 1),
    ];
    fill_extension_properties(DEVICE_EXTS, p_property_count, p_properties)
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateDevice(
    physical_device: ash::vk::PhysicalDevice,
    p_create_info: *const ash::vk::DeviceCreateInfo,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_device: *mut ash::vk::Device,
) -> ash::vk::Result {
    if physical_device.is_null() || p_create_info.is_null() || p_device.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    // SAFETY: physical_device is a valid handle created by this ICD
    let phys = PhysicalDeviceDispatchable::from_vk_mut(physical_device);
    let client = Arc::clone(&phys.client);

    let device_id = client.next_id();
    let queue_id = client.next_id();

    let dev = DeviceDispatchable::new(device_id, Arc::clone(&client), phys as *mut _);
    let dev_ptr = Box::into_raw(dev);

    let queue = QueueDispatchable::new(queue_id, dev_ptr, Arc::clone(&client), 0, 0);
    let queue_ptr = Box::into_raw(queue);
    (*dev_ptr).queue = queue_ptr;

    let req = vkremote_wire::types::Request::CreateDevice {
        device_id,
        queue_id,
    };
    if let Err(e) = client.send_request(req) {
        tracing::error!("CreateDevice remote request failed: {}", e);
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    *p_device = DeviceDispatchable::to_vk(dev_ptr);
    ash::vk::Result::SUCCESS
}
