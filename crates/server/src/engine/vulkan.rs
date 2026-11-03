use anyhow::{anyhow, Context, Result};
use ash::vk;
use std::ffi::CStr;
use std::sync::Arc;
use tracing::{info, warn};
use vkremote_wire::types::*;

pub struct VulkanEngine {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: ash::vk::PhysicalDevice,
    pub queue_family_index: u32,
    pub snapshot: PhysicalDeviceSnapshot,
    pub memory_properties: ash::vk::PhysicalDeviceMemoryProperties,
}

impl VulkanEngine {
    pub fn new(gpu_index: Option<usize>) -> Result<Arc<Self>> {
        // Purge any stale Vulkan driver overrides to ensure real hardware is loaded
        std::env::remove_var("VK_LOADER_DRIVERS_SELECT");
        std::env::remove_var("VK_LOADER_LAYERS_ENABLE");
        std::env::remove_var("VK_ADD_DRIVER_FILES");
        std::env::remove_var("VK_DRIVER_FILES");
        std::env::remove_var("VK_ICD_FILENAMES");
        std::env::remove_var("DXVK_FILTER_DEVICE_NAME");
        std::env::remove_var("GALLIUM_DRIVER");
        std::env::remove_var("MESA_LOADER_DRIVER_OVERRIDE");
        std::env::remove_var("MESA_VK_DEVICE_SELECT");

        // 1. Load Vulkan entry points
        // Safety: dynamically loads libvulkan.so.1 / vulkan-1.dll into process memory.
        let entry = unsafe {
            ash::Entry::load().context("Failed to dynamically load Vulkan entry loader")?
        };

        // 2. Create Vulkan Instance
        let app_name = CStr::from_bytes_with_nul(b"vkremote-server\0").unwrap();
        let engine_name = CStr::from_bytes_with_nul(b"vkremote\0").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let instance_create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        // Safety: parameters are valid C strings and structs for instance creation.
        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .context("Failed to create Vulkan instance")?
        };

        // 3. Enumerate physical devices
        // Safety: instance is a valid handle created above.
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .context("Failed to enumerate Vulkan physical devices")?
        };

        if physical_devices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            return Err(anyhow!("No Vulkan physical devices found on host"));
        }

        info!("Found {} Vulkan physical device(s)", physical_devices.len());

        // 4. Select physical device and compute queue family
        let (selected_device, queue_family_index) = match gpu_index {
            Some(idx) if idx < physical_devices.len() => {
                let pdev = physical_devices[idx];
                let queue_props =
                    unsafe { instance.get_physical_device_queue_family_properties(pdev) };
                let qfam = find_best_queue_family(&queue_props).ok_or_else(|| {
                    anyhow!("Physical device {} has no suitable queue family", idx)
                })?;
                (pdev, qfam)
            }
            _ => {
                let mut best_device = None;
                let mut best_score = 0;
                let mut best_queue = 0;

                for (i, &pdev) in physical_devices.iter().enumerate() {
                    let props = unsafe { instance.get_physical_device_properties(pdev) };
                    let queue_props =
                        unsafe { instance.get_physical_device_queue_family_properties(pdev) };
                    let dev_name = unsafe {
                        CStr::from_ptr(props.device_name.as_ptr())
                            .to_string_lossy()
                            .into_owned()
                    };

                    info!(
                        "Evaluating GPU #{}: {} (type: {:?})",
                        i, dev_name, props.device_type
                    );

                    if let Some(qfam) = find_best_queue_family(&queue_props) {
                        let score = score_device(&props);
                        if score > best_score || best_device.is_none() {
                            best_score = score;
                            best_device = Some(pdev);
                            best_queue = qfam;
                        }
                    } else {
                        warn!(
                            "GPU #{}: {} has no suitable queue family, skipping",
                            i, dev_name
                        );
                    }
                }

                let pdev = best_device
                    .ok_or_else(|| anyhow!("No suitable Vulkan physical device found"))?;
                (pdev, best_queue)
            }
        };

        // 5. Query device details and build snapshot
        // Safety: instance and selected_device are valid handles.
        let props = unsafe { instance.get_physical_device_properties(selected_device) };
        let features = unsafe { instance.get_physical_device_features(selected_device) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(selected_device) };
        let queue_props =
            unsafe { instance.get_physical_device_queue_family_properties(selected_device) };

        // Safety: device_name is a null-terminated C string in properties.
        let dev_name = unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        info!(
            "Selected Vulkan device: {} (queue family index: {})",
            dev_name, queue_family_index
        );

        let snapshot = build_snapshot(&props, &features, &memory_properties, &queue_props);

        Ok(Arc::new(Self {
            entry,
            instance,
            physical_device: selected_device,
            queue_family_index,
            snapshot,
            memory_properties,
        }))
    }

    /// Create a logical device and retrieve its compute queue
    pub fn create_device(&self) -> Result<(ash::Device, ash::vk::Queue)> {
        let priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(self.queue_family_index)
            .queue_priorities(&priorities);

        // Replay shaders and indirect commands must have the features advertised
        // by the physical-device snapshot enabled on the real logical device.
        // Previously no optional features were enabled at all.
        let supported = unsafe { self.instance.get_physical_device_features(self.physical_device) };
        let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
        if self.snapshot.properties.api_version >= vk::API_VERSION_1_2 {
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut vulkan12);
            unsafe { self.instance.get_physical_device_features2(self.physical_device, &mut features2); }
        }
        let mut device_create_info = vk::DeviceCreateInfo::default()
            .enabled_features(&supported)
            .queue_create_infos(std::slice::from_ref(&queue_create_info));
        if self.snapshot.properties.api_version >= vk::API_VERSION_1_2 {
            device_create_info = device_create_info.push_next(&mut vulkan12);
        }

        // Safety: instance and physical_device are valid, queue priorities are within [0.0, 1.0].
        let device = unsafe {
            self.instance
                .create_device(self.physical_device, &device_create_info, None)
                .context("Failed to create logical Vulkan device")?
        };

        // Safety: queue family index and queue index 0 were verified to exist.
        let queue = unsafe { device.get_device_queue(self.queue_family_index, 0) };

        Ok((device, queue))
    }
}

impl Drop for VulkanEngine {
    fn drop(&mut self) {
        // Safety: instance is destroyed only once when the engine is dropped.
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

fn score_device(props: &vk::PhysicalDeviceProperties) -> u32 {
    match props.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 500,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 200,
        vk::PhysicalDeviceType::CPU => 100,
        _ => 10,
    }
}

fn find_best_queue_family(queue_families: &[vk::QueueFamilyProperties]) -> Option<u32> {
    // Prefer queue family with both GRAPHICS and COMPUTE
    for (i, q) in queue_families.iter().enumerate() {
        if q.queue_flags
            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        {
            return Some(i as u32);
        }
    }
    // Fallback to any queue family with COMPUTE
    for (i, q) in queue_families.iter().enumerate() {
        if q.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            return Some(i as u32);
        }
    }
    // Fallback to any queue family
    if !queue_families.is_empty() {
        return Some(0);
    }
    None
}

fn build_snapshot(
    props: &vk::PhysicalDeviceProperties,
    features: &vk::PhysicalDeviceFeatures,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    queue_props: &[vk::QueueFamilyProperties],
) -> PhysicalDeviceSnapshot {
    let dev_name = unsafe {
        CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned()
    };

    let properties_data = PhysicalDevicePropertiesData {
        api_version: props.api_version,
        driver_version: props.driver_version,
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        device_type: props.device_type.as_raw() as u32,
        device_name: dev_name,
        pipeline_cache_uuid: props.pipeline_cache_uuid,
        limits: convert_limits(&props.limits),
    };

    let features_data = convert_features(features);

    let mut memory_types = Vec::new();
    for i in 0..mem_props.memory_type_count as usize {
        memory_types.push(MemoryTypeData {
            property_flags: mem_props.memory_types[i].property_flags.as_raw(),
            heap_index: mem_props.memory_types[i].heap_index,
        });
    }

    let mut memory_heaps = Vec::new();
    for i in 0..mem_props.memory_heap_count as usize {
        memory_heaps.push(MemoryHeapData {
            size: mem_props.memory_heaps[i].size,
            flags: mem_props.memory_heaps[i].flags.as_raw(),
        });
    }

    let memory_properties = PhysicalDeviceMemoryPropertiesData {
        memory_types,
        memory_heaps,
    };

    let queue_families = queue_props
        .iter()
        .map(|q| QueueFamilyPropertiesData {
            queue_flags: q.queue_flags.as_raw(),
            queue_count: q.queue_count,
            timestamp_valid_bits: q.timestamp_valid_bits,
            min_image_transfer_granularity: [
                q.min_image_transfer_granularity.width,
                q.min_image_transfer_granularity.height,
                q.min_image_transfer_granularity.depth,
            ],
        })
        .collect();

    PhysicalDeviceSnapshot {
        properties: properties_data,
        features: features_data,
        memory_properties,
        queue_families,
    }
}

fn convert_limits(l: &vk::PhysicalDeviceLimits) -> PhysicalDeviceLimitsData {
    PhysicalDeviceLimitsData {
        max_image_dimension1_d: l.max_image_dimension1_d,
        max_image_dimension2_d: l.max_image_dimension2_d,
        max_image_dimension3_d: l.max_image_dimension3_d,
        max_image_dimension_cube: l.max_image_dimension_cube,
        max_image_array_layers: l.max_image_array_layers,
        max_texel_buffer_elements: l.max_texel_buffer_elements,
        max_uniform_buffer_range: l.max_uniform_buffer_range,
        max_storage_buffer_range: l.max_storage_buffer_range,
        max_push_constants_size: l.max_push_constants_size,
        max_memory_allocation_count: l.max_memory_allocation_count,
        max_sampler_allocation_count: l.max_sampler_allocation_count,
        buffer_image_granularity: l.buffer_image_granularity,
        sparse_address_space_size: l.sparse_address_space_size,
        max_bound_descriptor_sets: l.max_bound_descriptor_sets,
        max_per_stage_descriptor_samplers: l.max_per_stage_descriptor_samplers,
        max_per_stage_descriptor_uniform_buffers: l.max_per_stage_descriptor_uniform_buffers,
        max_per_stage_descriptor_storage_buffers: l.max_per_stage_descriptor_storage_buffers,
        max_per_stage_descriptor_sampled_images: l.max_per_stage_descriptor_sampled_images,
        max_per_stage_descriptor_storage_images: l.max_per_stage_descriptor_storage_images,
        max_per_stage_descriptor_input_attachments: l.max_per_stage_descriptor_input_attachments,
        max_per_stage_resources: l.max_per_stage_resources,
        max_descriptor_set_samplers: l.max_descriptor_set_samplers,
        max_descriptor_set_uniform_buffers: l.max_descriptor_set_uniform_buffers,
        max_descriptor_set_uniform_buffers_dynamic: l.max_descriptor_set_uniform_buffers_dynamic,
        max_descriptor_set_storage_buffers: l.max_descriptor_set_storage_buffers,
        max_descriptor_set_storage_buffers_dynamic: l.max_descriptor_set_storage_buffers_dynamic,
        max_descriptor_set_sampled_images: l.max_descriptor_set_sampled_images,
        max_descriptor_set_storage_images: l.max_descriptor_set_storage_images,
        max_descriptor_set_input_attachments: l.max_descriptor_set_input_attachments,
        max_vertex_input_attributes: l.max_vertex_input_attributes,
        max_vertex_input_bindings: l.max_vertex_input_bindings,
        max_vertex_input_attribute_offset: l.max_vertex_input_attribute_offset,
        max_vertex_input_binding_stride: l.max_vertex_input_binding_stride,
        max_vertex_output_components: l.max_vertex_output_components,
        max_tessellation_generation_level: l.max_tessellation_generation_level,
        max_tessellation_patch_size: l.max_tessellation_patch_size,
        max_tessellation_control_per_vertex_input_components: l
            .max_tessellation_control_per_vertex_input_components,
        max_tessellation_control_per_vertex_output_components: l
            .max_tessellation_control_per_vertex_output_components,
        max_tessellation_control_per_patch_output_components: l
            .max_tessellation_control_per_patch_output_components,
        max_tessellation_control_total_output_components: l
            .max_tessellation_control_total_output_components,
        max_tessellation_evaluation_input_components: l
            .max_tessellation_evaluation_input_components,
        max_tessellation_evaluation_output_components: l
            .max_tessellation_evaluation_output_components,
        max_geometry_shader_invocations: l.max_geometry_shader_invocations,
        max_geometry_input_components: l.max_geometry_input_components,
        max_geometry_output_components: l.max_geometry_output_components,
        max_geometry_output_vertices: l.max_geometry_output_vertices,
        max_geometry_total_output_components: l.max_geometry_total_output_components,
        max_fragment_input_components: l.max_fragment_input_components,
        max_fragment_output_attachments: l.max_fragment_output_attachments,
        max_fragment_dual_src_attachments: l.max_fragment_dual_src_attachments,
        max_fragment_combined_output_resources: l.max_fragment_combined_output_resources,
        max_compute_shared_memory_size: l.max_compute_shared_memory_size,
        max_compute_work_group_count: l.max_compute_work_group_count,
        max_compute_work_group_invocations: l.max_compute_work_group_invocations,
        max_compute_work_group_size: l.max_compute_work_group_size,
        sub_pixel_precision_bits: l.sub_pixel_precision_bits,
        sub_texel_precision_bits: l.sub_texel_precision_bits,
        mipmap_precision_bits: l.mipmap_precision_bits,
        max_draw_indexed_index_value: l.max_draw_indexed_index_value,
        max_draw_indirect_count: l.max_draw_indirect_count,
        max_sampler_lod_bias_bits: l.max_sampler_lod_bias.to_bits(),
        max_sampler_anisotropy_bits: l.max_sampler_anisotropy.to_bits(),
        max_viewports: l.max_viewports,
        max_viewport_dimensions: l.max_viewport_dimensions,
        viewport_bounds_range_bits: [
            l.viewport_bounds_range[0].to_bits(),
            l.viewport_bounds_range[1].to_bits(),
        ],
        viewport_sub_pixel_bits: l.viewport_sub_pixel_bits,
        min_memory_map_alignment: l.min_memory_map_alignment,
        min_texel_buffer_offset_alignment: l.min_texel_buffer_offset_alignment,
        min_uniform_buffer_offset_alignment: l.min_uniform_buffer_offset_alignment,
        min_storage_buffer_offset_alignment: l.min_storage_buffer_offset_alignment,
        min_texel_offset: l.min_texel_offset,
        max_texel_offset: l.max_texel_offset,
        min_texel_gather_offset: l.min_texel_gather_offset,
        max_texel_gather_offset: l.max_texel_gather_offset,
        min_interpolation_offset_bits: l.min_interpolation_offset.to_bits(),
        max_interpolation_offset_bits: l.max_interpolation_offset.to_bits(),
        sub_pixel_interpolation_offset_bits: l.sub_pixel_interpolation_offset_bits,
        max_framebuffer_width: l.max_framebuffer_width,
        max_framebuffer_height: l.max_framebuffer_height,
        max_framebuffer_layers: l.max_framebuffer_layers,
        framebuffer_color_sample_counts: l.framebuffer_color_sample_counts.as_raw(),
        framebuffer_depth_sample_counts: l.framebuffer_depth_sample_counts.as_raw(),
        framebuffer_stencil_sample_counts: l.framebuffer_stencil_sample_counts.as_raw(),
        framebuffer_no_attachments_sample_counts: l
            .framebuffer_no_attachments_sample_counts
            .as_raw(),
        max_color_attachments: l.max_color_attachments,
        sampled_image_color_sample_counts: l.sampled_image_color_sample_counts.as_raw(),
        sampled_image_integer_sample_counts: l.sampled_image_integer_sample_counts.as_raw(),
        sampled_image_depth_sample_counts: l.sampled_image_depth_sample_counts.as_raw(),
        sampled_image_stencil_sample_counts: l.sampled_image_stencil_sample_counts.as_raw(),
        storage_image_sample_counts: l.storage_image_sample_counts.as_raw(),
        max_sample_mask_words: l.max_sample_mask_words,
        timestamp_compute_and_graphics: l.timestamp_compute_and_graphics != 0,
        timestamp_period_bits: l.timestamp_period.to_bits(),
        max_clip_distances: l.max_clip_distances,
        max_cull_distances: l.max_cull_distances,
        max_combined_clip_and_cull_distances: l.max_combined_clip_and_cull_distances,
        discrete_queue_priorities: l.discrete_queue_priorities,
        point_size_range_bits: [
            l.point_size_range[0].to_bits(),
            l.point_size_range[1].to_bits(),
        ],
        line_width_range_bits: [
            l.line_width_range[0].to_bits(),
            l.line_width_range[1].to_bits(),
        ],
        point_size_granularity_bits: l.point_size_granularity.to_bits(),
        line_width_granularity_bits: l.line_width_granularity.to_bits(),
        strict_lines: l.strict_lines != 0,
        standard_sample_locations: l.standard_sample_locations != 0,
        optimal_buffer_copy_offset_alignment: l.optimal_buffer_copy_offset_alignment,
        optimal_buffer_copy_row_pitch_alignment: l.optimal_buffer_copy_row_pitch_alignment,
        non_coherent_atom_size: l.non_coherent_atom_size,
    }
}

fn convert_features(f: &vk::PhysicalDeviceFeatures) -> PhysicalDeviceFeaturesData {
    PhysicalDeviceFeaturesData {
        robust_buffer_access: f.robust_buffer_access != 0,
        full_draw_index_uint32: f.full_draw_index_uint32 != 0,
        image_cube_array: f.image_cube_array != 0,
        independent_blend: f.independent_blend != 0,
        geometry_shader: f.geometry_shader != 0,
        tessellation_shader: f.tessellation_shader != 0,
        sample_rate_shading: f.sample_rate_shading != 0,
        dual_src_blend: f.dual_src_blend != 0,
        logic_op: f.logic_op != 0,
        multi_draw_indirect: f.multi_draw_indirect != 0,
        draw_indirect_first_instance: f.draw_indirect_first_instance != 0,
        depth_clamp: f.depth_clamp != 0,
        depth_bias_clamp: f.depth_bias_clamp != 0,
        fill_mode_non_solid: f.fill_mode_non_solid != 0,
        depth_bounds: f.depth_bounds != 0,
        wide_lines: f.wide_lines != 0,
        large_points: f.large_points != 0,
        alpha_to_one: f.alpha_to_one != 0,
        multi_viewport: f.multi_viewport != 0,
        sampler_anisotropy: f.sampler_anisotropy != 0,
        texture_compression_etc2: f.texture_compression_etc2 != 0,
        texture_compression_astc_ldr: f.texture_compression_astc_ldr != 0,
        texture_compression_bc: f.texture_compression_bc != 0,
        occlusion_query_precise: f.occlusion_query_precise != 0,
        pipeline_statistics_query: f.pipeline_statistics_query != 0,
        vertex_pipeline_stores_and_atomics: f.vertex_pipeline_stores_and_atomics != 0,
        fragment_stores_and_atomics: f.fragment_stores_and_atomics != 0,
        shader_tessellation_and_geometry_point_size: f.shader_tessellation_and_geometry_point_size
            != 0,
        shader_image_gather_extended: f.shader_image_gather_extended != 0,
        shader_storage_image_extended_formats: f.shader_storage_image_extended_formats != 0,
        shader_storage_image_multisample: f.shader_storage_image_multisample != 0,
        shader_storage_image_read_without_format: f.shader_storage_image_read_without_format != 0,
        shader_storage_image_write_without_format: f.shader_storage_image_write_without_format != 0,
        shader_uniform_buffer_array_dynamic_indexing: f
            .shader_uniform_buffer_array_dynamic_indexing
            != 0,
        shader_sampled_image_array_dynamic_indexing: f.shader_sampled_image_array_dynamic_indexing
            != 0,
        shader_storage_buffer_array_dynamic_indexing: f
            .shader_storage_buffer_array_dynamic_indexing
            != 0,
        shader_storage_image_array_dynamic_indexing: f.shader_storage_image_array_dynamic_indexing
            != 0,
        shader_clip_distance: f.shader_clip_distance != 0,
        shader_cull_distance: f.shader_cull_distance != 0,
        shader_float64: f.shader_float64 != 0,
        shader_int64: f.shader_int64 != 0,
        shader_int16: f.shader_int16 != 0,
        shader_resource_residency: f.shader_resource_residency != 0,
        shader_resource_min_lod: f.shader_resource_min_lod != 0,
        sparse_binding: f.sparse_binding != 0,
        sparse_residency_buffer: f.sparse_residency_buffer != 0,
        sparse_residency_image2_d: f.sparse_residency_image2_d != 0,
        sparse_residency_image3_d: f.sparse_residency_image3_d != 0,
        sparse_residency2_samples: f.sparse_residency2_samples != 0,
        sparse_residency4_samples: f.sparse_residency4_samples != 0,
        sparse_residency8_samples: f.sparse_residency8_samples != 0,
        sparse_residency16_samples: f.sparse_residency16_samples != 0,
        sparse_residency_aliased: f.sparse_residency_aliased != 0,
        variable_multisample_rate: f.variable_multisample_rate != 0,
        inherited_queries: f.inherited_queries != 0,
    }
}
