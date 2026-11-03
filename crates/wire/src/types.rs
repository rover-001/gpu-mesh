use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HandleId(pub u64);

impl From<u64> for HandleId {
    fn from(id: u64) -> Self {
        HandleId(id)
    }
}

impl From<HandleId> for u64 {
    fn from(id: HandleId) -> Self {
        id.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    Instance,
    PhysicalDevice,
    Device,
    Queue,
    Buffer,
    DeviceMemory,
    ShaderModule,
    DescriptorSetLayout,
    PipelineLayout,
    Pipeline,
    DescriptorPool,
    DescriptorSet,
    CommandPool,
    CommandBuffer,
    Fence,
    Image,
    ImageView,
    Sampler,
    RenderPass,
    Framebuffer,
    Swapchain,
    Surface,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalDeviceLimitsData {
    pub max_image_dimension1_d: u32,
    pub max_image_dimension2_d: u32,
    pub max_image_dimension3_d: u32,
    pub max_image_dimension_cube: u32,
    pub max_image_array_layers: u32,
    pub max_texel_buffer_elements: u32,
    pub max_uniform_buffer_range: u32,
    pub max_storage_buffer_range: u32,
    pub max_push_constants_size: u32,
    pub max_memory_allocation_count: u32,
    pub max_sampler_allocation_count: u32,
    pub buffer_image_granularity: u64,
    pub sparse_address_space_size: u64,
    pub max_bound_descriptor_sets: u32,
    pub max_per_stage_descriptor_samplers: u32,
    pub max_per_stage_descriptor_uniform_buffers: u32,
    pub max_per_stage_descriptor_storage_buffers: u32,
    pub max_per_stage_descriptor_sampled_images: u32,
    pub max_per_stage_descriptor_storage_images: u32,
    pub max_per_stage_descriptor_input_attachments: u32,
    pub max_per_stage_resources: u32,
    pub max_descriptor_set_samplers: u32,
    pub max_descriptor_set_uniform_buffers: u32,
    pub max_descriptor_set_uniform_buffers_dynamic: u32,
    pub max_descriptor_set_storage_buffers: u32,
    pub max_descriptor_set_storage_buffers_dynamic: u32,
    pub max_descriptor_set_sampled_images: u32,
    pub max_descriptor_set_storage_images: u32,
    pub max_descriptor_set_input_attachments: u32,
    pub max_vertex_input_attributes: u32,
    pub max_vertex_input_bindings: u32,
    pub max_vertex_input_attribute_offset: u32,
    pub max_vertex_input_binding_stride: u32,
    pub max_vertex_output_components: u32,
    pub max_tessellation_generation_level: u32,
    pub max_tessellation_patch_size: u32,
    pub max_tessellation_control_per_vertex_input_components: u32,
    pub max_tessellation_control_per_vertex_output_components: u32,
    pub max_tessellation_control_per_patch_output_components: u32,
    pub max_tessellation_control_total_output_components: u32,
    pub max_tessellation_evaluation_input_components: u32,
    pub max_tessellation_evaluation_output_components: u32,
    pub max_geometry_shader_invocations: u32,
    pub max_geometry_input_components: u32,
    pub max_geometry_output_components: u32,
    pub max_geometry_output_vertices: u32,
    pub max_geometry_total_output_components: u32,
    pub max_fragment_input_components: u32,
    pub max_fragment_output_attachments: u32,
    pub max_fragment_dual_src_attachments: u32,
    pub max_fragment_combined_output_resources: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_compute_work_group_count: [u32; 3],
    pub max_compute_work_group_invocations: u32,
    pub max_compute_work_group_size: [u32; 3],
    pub sub_pixel_precision_bits: u32,
    pub sub_texel_precision_bits: u32,
    pub mipmap_precision_bits: u32,
    pub max_draw_indexed_index_value: u32,
    pub max_draw_indirect_count: u32,
    pub max_sampler_lod_bias_bits: u32,
    pub max_sampler_anisotropy_bits: u32,
    pub max_viewports: u32,
    pub max_viewport_dimensions: [u32; 2],
    pub viewport_bounds_range_bits: [u32; 2],
    pub viewport_sub_pixel_bits: u32,
    pub min_memory_map_alignment: usize,
    pub min_texel_buffer_offset_alignment: u64,
    pub min_uniform_buffer_offset_alignment: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub min_texel_offset: i32,
    pub max_texel_offset: u32,
    pub min_texel_gather_offset: i32,
    pub max_texel_gather_offset: u32,
    pub min_interpolation_offset_bits: u32,
    pub max_interpolation_offset_bits: u32,
    pub sub_pixel_interpolation_offset_bits: u32,
    pub max_framebuffer_width: u32,
    pub max_framebuffer_height: u32,
    pub max_framebuffer_layers: u32,
    pub framebuffer_color_sample_counts: u32,
    pub framebuffer_depth_sample_counts: u32,
    pub framebuffer_stencil_sample_counts: u32,
    pub framebuffer_no_attachments_sample_counts: u32,
    pub max_color_attachments: u32,
    pub sampled_image_color_sample_counts: u32,
    pub sampled_image_integer_sample_counts: u32,
    pub sampled_image_depth_sample_counts: u32,
    pub sampled_image_stencil_sample_counts: u32,
    pub storage_image_sample_counts: u32,
    pub max_sample_mask_words: u32,
    pub timestamp_compute_and_graphics: bool,
    pub timestamp_period_bits: u32,
    pub max_clip_distances: u32,
    pub max_cull_distances: u32,
    pub max_combined_clip_and_cull_distances: u32,
    pub discrete_queue_priorities: u32,
    pub point_size_range_bits: [u32; 2],
    pub line_width_range_bits: [u32; 2],
    pub point_size_granularity_bits: u32,
    pub line_width_granularity_bits: u32,
    pub strict_lines: bool,
    pub standard_sample_locations: bool,
    pub optimal_buffer_copy_offset_alignment: u64,
    pub optimal_buffer_copy_row_pitch_alignment: u64,
    pub non_coherent_atom_size: u64,
}

impl Default for PhysicalDeviceLimitsData {
    fn default() -> Self {
        Self {
            max_image_dimension1_d: 4096,
            max_image_dimension2_d: 4096,
            max_image_dimension3_d: 256,
            max_image_dimension_cube: 4096,
            max_image_array_layers: 256,
            max_texel_buffer_elements: 65536,
            max_uniform_buffer_range: 65536,
            max_storage_buffer_range: 134217728,
            max_push_constants_size: 128,
            max_memory_allocation_count: 4096,
            max_sampler_allocation_count: 4000,
            buffer_image_granularity: 1,
            sparse_address_space_size: 0,
            max_bound_descriptor_sets: 4,
            max_per_stage_descriptor_samplers: 16,
            max_per_stage_descriptor_uniform_buffers: 12,
            max_per_stage_descriptor_storage_buffers: 12,
            max_per_stage_descriptor_sampled_images: 16,
            max_per_stage_descriptor_storage_images: 4,
            max_per_stage_descriptor_input_attachments: 4,
            max_per_stage_resources: 128,
            max_descriptor_set_samplers: 96,
            max_descriptor_set_uniform_buffers: 72,
            max_descriptor_set_uniform_buffers_dynamic: 8,
            max_descriptor_set_storage_buffers: 24,
            max_descriptor_set_storage_buffers_dynamic: 4,
            max_descriptor_set_sampled_images: 96,
            max_descriptor_set_storage_images: 24,
            max_descriptor_set_input_attachments: 4,
            max_vertex_input_attributes: 16,
            max_vertex_input_bindings: 16,
            max_vertex_input_attribute_offset: 2047,
            max_vertex_input_binding_stride: 2048,
            max_vertex_output_components: 64,
            max_tessellation_generation_level: 0,
            max_tessellation_patch_size: 0,
            max_tessellation_control_per_vertex_input_components: 0,
            max_tessellation_control_per_vertex_output_components: 0,
            max_tessellation_control_per_patch_output_components: 0,
            max_tessellation_control_total_output_components: 0,
            max_tessellation_evaluation_input_components: 0,
            max_tessellation_evaluation_output_components: 0,
            max_geometry_shader_invocations: 0,
            max_geometry_input_components: 0,
            max_geometry_output_components: 0,
            max_geometry_output_vertices: 0,
            max_geometry_total_output_components: 0,
            max_fragment_input_components: 64,
            max_fragment_output_attachments: 4,
            max_fragment_dual_src_attachments: 0,
            max_fragment_combined_output_resources: 16,
            max_compute_shared_memory_size: 32768,
            max_compute_work_group_count: [65535, 65535, 65535],
            max_compute_work_group_invocations: 1024,
            max_compute_work_group_size: [1024, 1024, 64],
            sub_pixel_precision_bits: 4,
            sub_texel_precision_bits: 4,
            mipmap_precision_bits: 4,
            max_draw_indexed_index_value: u32::MAX,
            max_draw_indirect_count: 1,
            max_sampler_lod_bias_bits: 0,
            max_sampler_anisotropy_bits: 0,
            max_viewports: 1,
            max_viewport_dimensions: [4096, 4096],
            viewport_bounds_range_bits: [0, 0],
            viewport_sub_pixel_bits: 0,
            min_memory_map_alignment: 64,
            min_texel_buffer_offset_alignment: 16,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 64,
            min_texel_offset: -8,
            max_texel_offset: 7,
            min_texel_gather_offset: -8,
            max_texel_gather_offset: 7,
            min_interpolation_offset_bits: 0,
            max_interpolation_offset_bits: 0,
            sub_pixel_interpolation_offset_bits: 0,
            max_framebuffer_width: 4096,
            max_framebuffer_height: 4096,
            max_framebuffer_layers: 256,
            framebuffer_color_sample_counts: 1,
            framebuffer_depth_sample_counts: 1,
            framebuffer_stencil_sample_counts: 1,
            framebuffer_no_attachments_sample_counts: 1,
            max_color_attachments: 4,
            sampled_image_color_sample_counts: 1,
            sampled_image_integer_sample_counts: 1,
            sampled_image_depth_sample_counts: 1,
            sampled_image_stencil_sample_counts: 1,
            storage_image_sample_counts: 1,
            max_sample_mask_words: 1,
            timestamp_compute_and_graphics: true,
            timestamp_period_bits: 0,
            max_clip_distances: 0,
            max_cull_distances: 0,
            max_combined_clip_and_cull_distances: 0,
            discrete_queue_priorities: 2,
            point_size_range_bits: [0, 0],
            line_width_range_bits: [0, 0],
            point_size_granularity_bits: 0,
            line_width_granularity_bits: 0,
            strict_lines: false,
            standard_sample_locations: false,
            optimal_buffer_copy_offset_alignment: 64,
            optimal_buffer_copy_row_pitch_alignment: 64,
            non_coherent_atom_size: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalDevicePropertiesData {
    pub api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub device_name: String,
    pub pipeline_cache_uuid: [u8; 16],
    pub limits: PhysicalDeviceLimitsData,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PhysicalDeviceFeaturesData {
    pub robust_buffer_access: bool,
    pub full_draw_index_uint32: bool,
    pub image_cube_array: bool,
    pub independent_blend: bool,
    pub geometry_shader: bool,
    pub tessellation_shader: bool,
    pub sample_rate_shading: bool,
    pub dual_src_blend: bool,
    pub logic_op: bool,
    pub multi_draw_indirect: bool,
    pub draw_indirect_first_instance: bool,
    pub depth_clamp: bool,
    pub depth_bias_clamp: bool,
    pub fill_mode_non_solid: bool,
    pub depth_bounds: bool,
    pub wide_lines: bool,
    pub large_points: bool,
    pub alpha_to_one: bool,
    pub multi_viewport: bool,
    pub sampler_anisotropy: bool,
    pub texture_compression_etc2: bool,
    pub texture_compression_astc_ldr: bool,
    pub texture_compression_bc: bool,
    pub occlusion_query_precise: bool,
    pub pipeline_statistics_query: bool,
    pub vertex_pipeline_stores_and_atomics: bool,
    pub fragment_stores_and_atomics: bool,
    pub shader_tessellation_and_geometry_point_size: bool,
    pub shader_image_gather_extended: bool,
    pub shader_storage_image_extended_formats: bool,
    pub shader_storage_image_multisample: bool,
    pub shader_storage_image_read_without_format: bool,
    pub shader_storage_image_write_without_format: bool,
    pub shader_uniform_buffer_array_dynamic_indexing: bool,
    pub shader_sampled_image_array_dynamic_indexing: bool,
    pub shader_storage_buffer_array_dynamic_indexing: bool,
    pub shader_storage_image_array_dynamic_indexing: bool,
    pub shader_clip_distance: bool,
    pub shader_cull_distance: bool,
    pub shader_float64: bool,
    pub shader_int64: bool,
    pub shader_int16: bool,
    pub shader_resource_residency: bool,
    pub shader_resource_min_lod: bool,
    pub sparse_binding: bool,
    pub sparse_residency_buffer: bool,
    pub sparse_residency_image2_d: bool,
    pub sparse_residency_image3_d: bool,
    pub sparse_residency2_samples: bool,
    pub sparse_residency4_samples: bool,
    pub sparse_residency8_samples: bool,
    pub sparse_residency16_samples: bool,
    pub sparse_residency_aliased: bool,
    pub variable_multisample_rate: bool,
    pub inherited_queries: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryTypeData {
    pub property_flags: u32,
    pub heap_index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryHeapData {
    pub size: u64,
    pub flags: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalDeviceMemoryPropertiesData {
    pub memory_types: Vec<MemoryTypeData>,
    pub memory_heaps: Vec<MemoryHeapData>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueFamilyPropertiesData {
    pub queue_flags: u32,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: [u32; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalDeviceSnapshot {
    pub properties: PhysicalDevicePropertiesData,
    pub features: PhysicalDeviceFeaturesData,
    pub memory_properties: PhysicalDeviceMemoryPropertiesData,
    pub queue_families: Vec<QueueFamilyPropertiesData>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescriptorSetLayoutBindingData {
    pub binding: u32,
    pub descriptor_type: u32,
    pub descriptor_count: u32,
    pub stage_flags: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PushConstantRangeData {
    pub stage_flags: u32,
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescriptorPoolSizeData {
    pub descriptor_type: u32,
    pub descriptor_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescriptorBufferInfoData {
    pub buffer_id: u64,
    pub offset: u64,
    pub range: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescriptorImageInfoData {
    pub sampler_id: u64,
    pub image_view_id: u64,
    pub image_layout: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteDescriptorSetData {
    pub dst_set_id: u64,
    pub dst_binding: u32,
    pub dst_array_element: u32,
    pub descriptor_type: u32,
    pub buffer_info: Vec<DescriptorBufferInfoData>,
    pub image_info: Vec<DescriptorImageInfoData>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferCopyData {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryChunkDiff {
    pub allocation_id: u64,
    pub offset: u64,
    // serde_bytes: same postcard bytes as Vec<u8>, but one memcpy instead of per-byte serde.
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}
// ============================================================================
// Phase 3 Graphics: Basic Types & Formats
// ============================================================================

/// Float wrapper that implements Eq and Hash by bitwise comparison.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct F32(pub u32);

impl F32 {
    pub fn from_f32(v: f32) -> Self {
        Self(v.to_bits())
    }
    pub fn to_f32(self) -> f32 {
        f32::from_bits(self.0)
    }
}

impl From<f32> for F32 {
    fn from(v: f32) -> Self {
        Self::from_f32(v)
    }
}

impl From<F32> for f32 {
    fn from(v: F32) -> Self {
        v.to_f32()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Extent2D {
    pub width: u32,
    pub height: u32,
}

impl From<[u32; 2]> for Extent2D {
    fn from(v: [u32; 2]) -> Self {
        Self {
            width: v[0],
            height: v[1],
        }
    }
}

impl From<Extent2D> for [u32; 2] {
    fn from(e: Extent2D) -> Self {
        [e.width, e.height]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Extent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl From<[u32; 3]> for Extent3D {
    fn from(v: [u32; 3]) -> Self {
        Self {
            width: v[0],
            height: v[1],
            depth: v[2],
        }
    }
}

impl From<Extent3D> for [u32; 3] {
    fn from(e: Extent3D) -> Self {
        [e.width, e.height, e.depth]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Offset2D {
    pub x: i32,
    pub y: i32,
}

impl From<[i32; 2]> for Offset2D {
    fn from(v: [i32; 2]) -> Self {
        Self { x: v[0], y: v[1] }
    }
}

impl From<Offset2D> for [i32; 2] {
    fn from(o: Offset2D) -> Self {
        [o.x, o.y]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Offset3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Rect2D {
    pub offset: Offset2D,
    pub extent: Extent2D,
}

pub type Rect2DData = Rect2D;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Viewport {
    pub x: F32,
    pub y: F32,
    pub width: F32,
    pub height: F32,
    pub min_depth: F32,
    pub max_depth: F32,
}

impl Viewport {
    pub fn new(x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            width: width.into(),
            height: height.into(),
            min_depth: min_depth.into(),
            max_depth: max_depth.into(),
        }
    }
}

pub type ViewportData = Viewport;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Format {
    #[default]
    Undefined = 0,
    R8Unorm = 9,
    R8G8B8A8Unorm = 37,
    R8G8B8A8Srgb = 43,
    B8G8R8A8Unorm = 44,
    B8G8R8A8Srgb = 50,
    D16Unorm = 124,
    D32Sfloat = 126,
    S8Uint = 127,
    D16UnormS8Uint = 128,
    D24UnormS8Uint = 129,
    D32SfloatS8Uint = 130,
    Other(u32),
}

impl From<u32> for Format {
    fn from(v: u32) -> Self {
        match v {
            0 => Format::Undefined,
            9 => Format::R8Unorm,
            37 => Format::R8G8B8A8Unorm,
            43 => Format::R8G8B8A8Srgb,
            44 => Format::B8G8R8A8Unorm,
            50 => Format::B8G8R8A8Srgb,
            124 => Format::D16Unorm,
            126 => Format::D32Sfloat,
            127 => Format::S8Uint,
            128 => Format::D16UnormS8Uint,
            129 => Format::D24UnormS8Uint,
            130 => Format::D32SfloatS8Uint,
            other => Format::Other(other),
        }
    }
}

impl From<Format> for u32 {
    fn from(f: Format) -> Self {
        match f {
            Format::Undefined => 0,
            Format::R8Unorm => 9,
            Format::R8G8B8A8Unorm => 37,
            Format::R8G8B8A8Srgb => 43,
            Format::B8G8R8A8Unorm => 44,
            Format::B8G8R8A8Srgb => 50,
            Format::D16Unorm => 124,
            Format::D32Sfloat => 126,
            Format::S8Uint => 127,
            Format::D16UnormS8Uint => 128,
            Format::D24UnormS8Uint => 129,
            Format::D32SfloatS8Uint => 130,
            Format::Other(other) => other,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SampleCount {
    #[default]
    Count1 = 1,
    Count2 = 2,
    Count4 = 4,
    Count8 = 8,
    Count16 = 16,
    Count32 = 32,
    Count64 = 64,
}

impl From<u32> for SampleCount {
    fn from(v: u32) -> Self {
        match v {
            2 => SampleCount::Count2,
            4 => SampleCount::Count4,
            8 => SampleCount::Count8,
            16 => SampleCount::Count16,
            32 => SampleCount::Count32,
            64 => SampleCount::Count64,
            _ => SampleCount::Count1,
        }
    }
}

impl From<SampleCount> for u32 {
    fn from(s: SampleCount) -> Self {
        match s {
            SampleCount::Count1 => 1,
            SampleCount::Count2 => 2,
            SampleCount::Count4 => 4,
            SampleCount::Count8 => 8,
            SampleCount::Count16 => 16,
            SampleCount::Count32 => 32,
            SampleCount::Count64 => 64,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ImageType {
    Type1D = 0,
    #[default]
    Type2D = 1,
    Type3D = 2,
}

impl From<u32> for ImageType {
    fn from(v: u32) -> Self {
        match v {
            0 => ImageType::Type1D,
            1 => ImageType::Type2D,
            2 => ImageType::Type3D,
            _ => ImageType::Type2D,
        }
    }
}

impl From<ImageType> for u32 {
    fn from(t: ImageType) -> Self {
        t as u32
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ImageTiling {
    #[default]
    Optimal = 0,
    Linear = 1,
}

impl From<u32> for ImageTiling {
    fn from(v: u32) -> Self {
        match v {
            1 => ImageTiling::Linear,
            _ => ImageTiling::Optimal,
        }
    }
}

impl From<ImageTiling> for u32 {
    fn from(t: ImageTiling) -> Self {
        t as u32
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SharingMode {
    #[default]
    Exclusive = 0,
    Concurrent = 1,
}

impl From<u32> for SharingMode {
    fn from(v: u32) -> Self {
        match v {
            1 => SharingMode::Concurrent,
            _ => SharingMode::Exclusive,
        }
    }
}

impl From<SharingMode> for u32 {
    fn from(m: SharingMode) -> Self {
        m as u32
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ImageLayout {
    #[default]
    Undefined = 0,
    General = 1,
    ColorAttachmentOptimal = 2,
    DepthStencilAttachmentOptimal = 3,
    DepthStencilReadOnlyOptimal = 4,
    ShaderReadOnlyOptimal = 5,
    TransferSrcOptimal = 6,
    TransferDstOptimal = 7,
    Preinitialized = 8,
    PresentSrcKhr = 1000001002,
    Other(u32),
}

impl From<u32> for ImageLayout {
    fn from(v: u32) -> Self {
        match v {
            0 => ImageLayout::Undefined,
            1 => ImageLayout::General,
            2 => ImageLayout::ColorAttachmentOptimal,
            3 => ImageLayout::DepthStencilAttachmentOptimal,
            4 => ImageLayout::DepthStencilReadOnlyOptimal,
            5 => ImageLayout::ShaderReadOnlyOptimal,
            6 => ImageLayout::TransferSrcOptimal,
            7 => ImageLayout::TransferDstOptimal,
            8 => ImageLayout::Preinitialized,
            1000001002 => ImageLayout::PresentSrcKhr,
            other => ImageLayout::Other(other),
        }
    }
}

impl From<ImageLayout> for u32 {
    fn from(l: ImageLayout) -> Self {
        match l {
            ImageLayout::Undefined => 0,
            ImageLayout::General => 1,
            ImageLayout::ColorAttachmentOptimal => 2,
            ImageLayout::DepthStencilAttachmentOptimal => 3,
            ImageLayout::DepthStencilReadOnlyOptimal => 4,
            ImageLayout::ShaderReadOnlyOptimal => 5,
            ImageLayout::TransferSrcOptimal => 6,
            ImageLayout::TransferDstOptimal => 7,
            ImageLayout::Preinitialized => 8,
            ImageLayout::PresentSrcKhr => 1000001002,
            ImageLayout::Other(other) => other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ImageUsageFlags(pub u32);

impl ImageUsageFlags {
    pub const TRANSFER_SRC: u32 = 0x00000001;
    pub const TRANSFER_DST: u32 = 0x00000002;
    pub const SAMPLED: u32 = 0x00000004;
    pub const STORAGE: u32 = 0x00000008;
    pub const COLOR_ATTACHMENT: u32 = 0x00000010;
    pub const DEPTH_STENCIL_ATTACHMENT: u32 = 0x00000020;
    pub const TRANSIENT_ATTACHMENT: u32 = 0x00000040;
    pub const INPUT_ATTACHMENT: u32 = 0x00000080;

    pub fn contains(&self, flag: u32) -> bool {
        (self.0 & flag) == flag
    }
}

impl From<u32> for ImageUsageFlags {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<ImageUsageFlags> for u32 {
    fn from(f: ImageUsageFlags) -> Self {
        f.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateImageParams {
    pub image_type: ImageType,
    pub format: Format,
    pub extent: Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: SampleCount,
    pub tiling: ImageTiling,
    pub usage: ImageUsageFlags,
    pub sharing_mode: SharingMode,
    pub queue_family_indices: Vec<u32>,
    pub initial_layout: ImageLayout,
}

// ============================================================================
// Phase 3 Graphics: ImageView & Sampler
// ============================================================================

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ImageViewType {
    Type1D = 0,
    #[default]
    Type2D = 1,
    Type3D = 2,
    Cube = 3,
    Type1DArray = 4,
    Type2DArray = 5,
    CubeArray = 6,
}

impl From<u32> for ImageViewType {
    fn from(v: u32) -> Self {
        match v {
            0 => ImageViewType::Type1D,
            1 => ImageViewType::Type2D,
            2 => ImageViewType::Type3D,
            3 => ImageViewType::Cube,
            4 => ImageViewType::Type1DArray,
            5 => ImageViewType::Type2DArray,
            6 => ImageViewType::CubeArray,
            _ => ImageViewType::Type2D,
        }
    }
}

impl From<ImageViewType> for u32 {
    fn from(t: ImageViewType) -> Self {
        t as u32
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ComponentSwizzle {
    #[default]
    Identity = 0,
    Zero = 1,
    One = 2,
    R = 3,
    G = 4,
    B = 5,
    A = 6,
}

impl From<u32> for ComponentSwizzle {
    fn from(v: u32) -> Self {
        match v {
            1 => ComponentSwizzle::Zero,
            2 => ComponentSwizzle::One,
            3 => ComponentSwizzle::R,
            4 => ComponentSwizzle::G,
            5 => ComponentSwizzle::B,
            6 => ComponentSwizzle::A,
            _ => ComponentSwizzle::Identity,
        }
    }
}

impl From<ComponentSwizzle> for u32 {
    fn from(s: ComponentSwizzle) -> Self {
        s as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ComponentMapping {
    pub r: ComponentSwizzle,
    pub g: ComponentSwizzle,
    pub b: ComponentSwizzle,
    pub a: ComponentSwizzle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ImageAspectFlags(pub u32);

impl ImageAspectFlags {
    pub const COLOR: u32 = 0x00000001;
    pub const DEPTH: u32 = 0x00000002;
    pub const STENCIL: u32 = 0x00000004;
    pub const METADATA: u32 = 0x00000008;

    pub fn contains(&self, flag: u32) -> bool {
        (self.0 & flag) == flag
    }
}

impl From<u32> for ImageAspectFlags {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<ImageAspectFlags> for u32 {
    fn from(f: ImageAspectFlags) -> Self {
        f.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ImageSubresourceRange {
    pub aspect_mask: ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

pub type ImageSubresourceRangeData = ImageSubresourceRange;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateImageViewParams {
    pub image_id: u64,
    pub view_type: ImageViewType,
    pub format: Format,
    pub components: ComponentMapping,
    pub subresource_range: ImageSubresourceRange,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Filter {
    #[default]
    Nearest = 0,
    Linear = 1,
}

impl From<u32> for Filter {
    fn from(v: u32) -> Self {
        match v {
            1 => Filter::Linear,
            _ => Filter::Nearest,
        }
    }
}

impl From<Filter> for u32 {
    fn from(f: Filter) -> Self {
        f as u32
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SamplerMipmapMode {
    #[default]
    Nearest = 0,
    Linear = 1,
}

impl From<u32> for SamplerMipmapMode {
    fn from(v: u32) -> Self {
        match v {
            1 => SamplerMipmapMode::Linear,
            _ => SamplerMipmapMode::Nearest,
        }
    }
}

impl From<SamplerMipmapMode> for u32 {
    fn from(m: SamplerMipmapMode) -> Self {
        m as u32
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SamplerAddressMode {
    #[default]
    Repeat = 0,
    MirroredRepeat = 1,
    ClampToEdge = 2,
    ClampToBorder = 3,
    MirrorClampToEdge = 4,
}

impl From<u32> for SamplerAddressMode {
    fn from(v: u32) -> Self {
        match v {
            1 => SamplerAddressMode::MirroredRepeat,
            2 => SamplerAddressMode::ClampToEdge,
            3 => SamplerAddressMode::ClampToBorder,
            4 => SamplerAddressMode::MirrorClampToEdge,
            _ => SamplerAddressMode::Repeat,
        }
    }
}

impl From<SamplerAddressMode> for u32 {
    fn from(a: SamplerAddressMode) -> Self {
        a as u32
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum BorderColor {
    #[default]
    FloatTransparentBlack = 0,
    IntTransparentBlack = 1,
    FloatOpaqueBlack = 2,
    IntOpaqueBlack = 3,
    FloatOpaqueWhite = 4,
    IntOpaqueWhite = 5,
}

impl From<u32> for BorderColor {
    fn from(v: u32) -> Self {
        match v {
            1 => BorderColor::IntTransparentBlack,
            2 => BorderColor::FloatOpaqueBlack,
            3 => BorderColor::IntOpaqueBlack,
            4 => BorderColor::FloatOpaqueWhite,
            5 => BorderColor::IntOpaqueWhite,
            _ => BorderColor::FloatTransparentBlack,
        }
    }
}

impl From<BorderColor> for u32 {
    fn from(b: BorderColor) -> Self {
        b as u32
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum CompareOp {
    #[default]
    Never = 0,
    Less = 1,
    Equal = 2,
    LessOrEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterOrEqual = 6,
    Always = 7,
}

impl From<u32> for CompareOp {
    fn from(v: u32) -> Self {
        match v {
            1 => CompareOp::Less,
            2 => CompareOp::Equal,
            3 => CompareOp::LessOrEqual,
            4 => CompareOp::Greater,
            5 => CompareOp::NotEqual,
            6 => CompareOp::GreaterOrEqual,
            7 => CompareOp::Always,
            _ => CompareOp::Never,
        }
    }
}

impl From<CompareOp> for u32 {
    fn from(c: CompareOp) -> Self {
        c as u32
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateSamplerParams {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mipmap_mode: SamplerMipmapMode,
    pub address_mode_u: SamplerAddressMode,
    pub address_mode_v: SamplerAddressMode,
    pub address_mode_w: SamplerAddressMode,
    pub mip_lod_bias: F32,
    pub anisotropy_enable: bool,
    pub max_anisotropy: F32,
    pub compare_enable: bool,
    pub compare_op: CompareOp,
    pub min_lod: F32,
    pub max_lod: F32,
    pub border_color: BorderColor,
    pub unnormalized_coordinates: bool,
}

// ============================================================================
// Phase 3 Graphics: RenderPass & Framebuffer
// ============================================================================

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum AttachmentLoadOp {
    #[default]
    Load = 0,
    Clear = 1,
    DontCare = 2,
}

impl From<u32> for AttachmentLoadOp {
    fn from(v: u32) -> Self {
        match v {
            1 => AttachmentLoadOp::Clear,
            2 => AttachmentLoadOp::DontCare,
            _ => AttachmentLoadOp::Load,
        }
    }
}

impl From<AttachmentLoadOp> for u32 {
    fn from(o: AttachmentLoadOp) -> Self {
        o as u32
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum AttachmentStoreOp {
    #[default]
    Store = 0,
    DontCare = 1,
}

impl From<u32> for AttachmentStoreOp {
    fn from(v: u32) -> Self {
        match v {
            1 => AttachmentStoreOp::DontCare,
            _ => AttachmentStoreOp::Store,
        }
    }
}

impl From<AttachmentStoreOp> for u32 {
    fn from(o: AttachmentStoreOp) -> Self {
        o as u32
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttachmentDescription {
    pub flags: u32,
    pub format: Format,
    pub samples: SampleCount,
    pub load_op: AttachmentLoadOp,
    pub store_op: AttachmentStoreOp,
    pub stencil_load_op: AttachmentLoadOp,
    pub stencil_store_op: AttachmentStoreOp,
    pub initial_layout: ImageLayout,
    pub final_layout: ImageLayout,
}

pub type AttachmentDescriptionData = AttachmentDescription;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AttachmentReference {
    pub attachment: u32,
    pub layout: ImageLayout,
}

impl AttachmentReference {
    pub const UNUSED: u32 = !0;
}

pub type AttachmentReferenceData = AttachmentReference;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum PipelineBindPoint {
    #[default]
    Graphics = 0,
    Compute = 1,
}

impl From<u32> for PipelineBindPoint {
    fn from(v: u32) -> Self {
        match v {
            1 => PipelineBindPoint::Compute,
            _ => PipelineBindPoint::Graphics,
        }
    }
}

impl From<PipelineBindPoint> for u32 {
    fn from(p: PipelineBindPoint) -> Self {
        p as u32
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubpassDescription {
    pub flags: u32,
    pub pipeline_bind_point: PipelineBindPoint,
    pub input_attachments: Vec<AttachmentReference>,
    pub color_attachments: Vec<AttachmentReference>,
    pub resolve_attachments: Vec<AttachmentReference>,
    pub depth_stencil_attachment: Option<AttachmentReference>,
    pub preserve_attachments: Vec<u32>,
}

pub type SubpassDescriptionData = SubpassDescription;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubpassDependency {
    pub src_subpass: u32,
    pub dst_subpass: u32,
    pub src_stage_mask: u32,
    pub dst_stage_mask: u32,
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
    pub dependency_flags: u32,
}

impl SubpassDependency {
    pub const EXTERNAL: u32 = !0;
}

pub type SubpassDependencyData = SubpassDependency;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateRenderPassParams {
    pub attachments: Vec<AttachmentDescription>,
    pub subpasses: Vec<SubpassDescription>,
    pub dependencies: Vec<SubpassDependency>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateFramebufferParams {
    pub render_pass_id: u64,
    pub attachments: Vec<u64>,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClearColorValue {
    Float([F32; 4]),
    Int([i32; 4]),
    Uint([u32; 4]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClearDepthStencilValue {
    pub depth: F32,
    pub stencil: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClearValueData {
    Color(ClearColorValue),
    DepthStencil(ClearDepthStencilValue),
    ColorFloat([u32; 4]),
    ColorInt([i32; 4]),
    ColorUint([u32; 4]),
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum SubpassContents {
    #[default]
    Inline = 0,
    SecondaryCommandBuffers = 1,
}

impl From<u32> for SubpassContents {
    fn from(v: u32) -> Self {
        match v {
            1 => SubpassContents::SecondaryCommandBuffers,
            _ => SubpassContents::Inline,
        }
    }
}

impl From<SubpassContents> for u32 {
    fn from(c: SubpassContents) -> Self {
        c as u32
    }
}

// ============================================================================
// Phase 3 Graphics: Graphics Pipeline Parameters
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ShaderStageFlags(pub u32);

impl ShaderStageFlags {
    pub const VERTEX: u32 = 0x00000001;
    pub const TESSELLATION_CONTROL: u32 = 0x00000002;
    pub const TESSELLATION_EVALUATION: u32 = 0x00000004;
    pub const GEOMETRY: u32 = 0x00000008;
    pub const FRAGMENT: u32 = 0x00000010;
    pub const COMPUTE: u32 = 0x00000020;
    pub const ALL_GRAPHICS: u32 = 0x0000001F;
    pub const ALL: u32 = 0x7FFFFFFF;

    pub fn contains(&self, flag: u32) -> bool {
        (self.0 & flag) == flag
    }
}

impl From<u32> for ShaderStageFlags {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<ShaderStageFlags> for u32 {
    fn from(f: ShaderStageFlags) -> Self {
        f.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineShaderStageData {
    pub stage: ShaderStageFlags,
    pub module_id: u64,
    pub entry_point: String,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum VertexInputRate {
    #[default]
    Vertex = 0,
    Instance = 1,
}

impl From<u32> for VertexInputRate {
    fn from(v: u32) -> Self {
        match v {
            1 => VertexInputRate::Instance,
            _ => VertexInputRate::Vertex,
        }
    }
}

impl From<VertexInputRate> for u32 {
    fn from(r: VertexInputRate) -> Self {
        r as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct VertexInputBindingDescription {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: VertexInputRate,
}

pub type VertexInputBindingDescriptionData = VertexInputBindingDescription;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VertexInputAttributeDescription {
    pub location: u32,
    pub binding: u32,
    pub format: Format,
    pub offset: u32,
}

pub type VertexInputAttributeDescriptionData = VertexInputAttributeDescription;

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PipelineVertexInputStateData {
    pub bindings: Vec<VertexInputBindingDescription>,
    pub attributes: Vec<VertexInputAttributeDescription>,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum PrimitiveTopology {
    PointList = 0,
    LineList = 1,
    LineStrip = 2,
    #[default]
    TriangleList = 3,
    TriangleStrip = 4,
    TriangleFan = 5,
    LineListWithAdjacency = 6,
    LineStripWithAdjacency = 7,
    TriangleListWithAdjacency = 8,
    TriangleStripWithAdjacency = 9,
    PatchList = 10,
}

impl From<u32> for PrimitiveTopology {
    fn from(v: u32) -> Self {
        match v {
            0 => PrimitiveTopology::PointList,
            1 => PrimitiveTopology::LineList,
            2 => PrimitiveTopology::LineStrip,
            3 => PrimitiveTopology::TriangleList,
            4 => PrimitiveTopology::TriangleStrip,
            5 => PrimitiveTopology::TriangleFan,
            6 => PrimitiveTopology::LineListWithAdjacency,
            7 => PrimitiveTopology::LineStripWithAdjacency,
            8 => PrimitiveTopology::TriangleListWithAdjacency,
            9 => PrimitiveTopology::TriangleStripWithAdjacency,
            10 => PrimitiveTopology::PatchList,
            _ => PrimitiveTopology::TriangleList,
        }
    }
}

impl From<PrimitiveTopology> for u32 {
    fn from(t: PrimitiveTopology) -> Self {
        t as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct PipelineInputAssemblyStateData {
    pub topology: PrimitiveTopology,
    pub primitive_restart_enable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PipelineViewportStateData {
    pub viewports: Vec<Viewport>,
    pub scissors: Vec<Rect2D>,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum PolygonMode {
    #[default]
    Fill = 0,
    Line = 1,
    Point = 2,
}

impl From<u32> for PolygonMode {
    fn from(v: u32) -> Self {
        match v {
            1 => PolygonMode::Line,
            2 => PolygonMode::Point,
            _ => PolygonMode::Fill,
        }
    }
}

impl From<PolygonMode> for u32 {
    fn from(m: PolygonMode) -> Self {
        m as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct CullModeFlags(pub u32);

impl CullModeFlags {
    pub const NONE: u32 = 0;
    pub const FRONT: u32 = 0x00000001;
    pub const BACK: u32 = 0x00000002;
    pub const FRONT_AND_BACK: u32 = 0x00000003;

    pub fn contains(&self, flag: u32) -> bool {
        (self.0 & flag) == flag
    }
}

impl From<u32> for CullModeFlags {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<CullModeFlags> for u32 {
    fn from(f: CullModeFlags) -> Self {
        f.0
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum FrontFace {
    #[default]
    CounterClockwise = 0,
    Clockwise = 1,
}

impl From<u32> for FrontFace {
    fn from(v: u32) -> Self {
        match v {
            1 => FrontFace::Clockwise,
            _ => FrontFace::CounterClockwise,
        }
    }
}

impl From<FrontFace> for u32 {
    fn from(f: FrontFace) -> Self {
        f as u32
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineRasterizationStateData {
    pub depth_clamp_enable: bool,
    pub rasterizer_discard_enable: bool,
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullModeFlags,
    pub front_face: FrontFace,
    pub depth_bias_enable: bool,
    pub depth_bias_constant_factor: F32,
    pub depth_bias_clamp: F32,
    pub depth_bias_slope_factor: F32,
    pub line_width: F32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineMultisampleStateData {
    pub rasterization_samples: SampleCount,
    pub sample_shading_enable: bool,
    pub min_sample_shading: F32,
    pub sample_mask: Vec<u32>,
    pub alpha_to_coverage_enable: bool,
    pub alpha_to_one_enable: bool,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum StencilOp {
    #[default]
    Keep = 0,
    Zero = 1,
    Replace = 2,
    IncrementAndClamp = 3,
    DecrementAndClamp = 4,
    Invert = 5,
    IncrementAndWrap = 6,
    DecrementAndWrap = 7,
}

impl From<u32> for StencilOp {
    fn from(v: u32) -> Self {
        match v {
            1 => StencilOp::Zero,
            2 => StencilOp::Replace,
            3 => StencilOp::IncrementAndClamp,
            4 => StencilOp::DecrementAndClamp,
            5 => StencilOp::Invert,
            6 => StencilOp::IncrementAndWrap,
            7 => StencilOp::DecrementAndWrap,
            _ => StencilOp::Keep,
        }
    }
}

impl From<StencilOp> for u32 {
    fn from(o: StencilOp) -> Self {
        o as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct StencilOpState {
    pub fail_op: StencilOp,
    pub pass_op: StencilOp,
    pub depth_fail_op: StencilOp,
    pub compare_op: CompareOp,
    pub compare_mask: u32,
    pub write_mask: u32,
    pub reference: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineDepthStencilStateData {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: CompareOp,
    pub depth_bounds_test_enable: bool,
    pub stencil_test_enable: bool,
    pub front: StencilOpState,
    pub back: StencilOpState,
    pub min_depth_bounds: F32,
    pub max_depth_bounds: F32,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum BlendFactor {
    #[default]
    Zero = 0,
    One = 1,
    SrcColor = 2,
    OneMinusSrcColor = 3,
    DstColor = 4,
    OneMinusDstColor = 5,
    SrcAlpha = 6,
    OneMinusSrcAlpha = 7,
    DstAlpha = 8,
    OneMinusDstAlpha = 9,
    ConstantColor = 10,
    OneMinusConstantColor = 11,
    ConstantAlpha = 12,
    OneMinusConstantAlpha = 13,
    SrcAlphaSaturate = 14,
    Src1Color = 15,
    OneMinusSrc1Color = 16,
    Src1Alpha = 17,
    OneMinusSrc1Alpha = 18,
}

impl From<u32> for BlendFactor {
    fn from(v: u32) -> Self {
        match v {
            1 => BlendFactor::One,
            2 => BlendFactor::SrcColor,
            3 => BlendFactor::OneMinusSrcColor,
            4 => BlendFactor::DstColor,
            5 => BlendFactor::OneMinusDstColor,
            6 => BlendFactor::SrcAlpha,
            7 => BlendFactor::OneMinusSrcAlpha,
            8 => BlendFactor::DstAlpha,
            9 => BlendFactor::OneMinusDstAlpha,
            10 => BlendFactor::ConstantColor,
            11 => BlendFactor::OneMinusConstantColor,
            12 => BlendFactor::ConstantAlpha,
            13 => BlendFactor::OneMinusConstantAlpha,
            14 => BlendFactor::SrcAlphaSaturate,
            15 => BlendFactor::Src1Color,
            16 => BlendFactor::OneMinusSrc1Color,
            17 => BlendFactor::Src1Alpha,
            18 => BlendFactor::OneMinusSrc1Alpha,
            _ => BlendFactor::Zero,
        }
    }
}

impl From<BlendFactor> for u32 {
    fn from(b: BlendFactor) -> Self {
        b as u32
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum BlendOp {
    #[default]
    Add = 0,
    Subtract = 1,
    ReverseSubtract = 2,
    Min = 3,
    Max = 4,
}

impl From<u32> for BlendOp {
    fn from(v: u32) -> Self {
        match v {
            1 => BlendOp::Subtract,
            2 => BlendOp::ReverseSubtract,
            3 => BlendOp::Min,
            4 => BlendOp::Max,
            _ => BlendOp::Add,
        }
    }
}

impl From<BlendOp> for u32 {
    fn from(o: BlendOp) -> Self {
        o as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ColorComponentFlags(pub u32);

impl ColorComponentFlags {
    pub const R: u32 = 0x00000001;
    pub const G: u32 = 0x00000002;
    pub const B: u32 = 0x00000004;
    pub const A: u32 = 0x00000008;
    pub const ALL: u32 = 0x0000000F;

    pub fn contains(&self, flag: u32) -> bool {
        (self.0 & flag) == flag
    }
}

impl From<u32> for ColorComponentFlags {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<ColorComponentFlags> for u32 {
    fn from(f: ColorComponentFlags) -> Self {
        f.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ColorBlendAttachmentState {
    pub blend_enable: bool,
    pub src_color_blend_factor: BlendFactor,
    pub dst_color_blend_factor: BlendFactor,
    pub color_blend_op: BlendOp,
    pub src_alpha_blend_factor: BlendFactor,
    pub dst_alpha_blend_factor: BlendFactor,
    pub alpha_blend_op: BlendOp,
    pub color_write_mask: ColorComponentFlags,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum LogicOp {
    #[default]
    Clear = 0,
    And = 1,
    AndReverse = 2,
    Copy = 3,
    AndInverted = 4,
    NoOp = 5,
    Xor = 6,
    Or = 7,
    Nor = 8,
    Equivalent = 9,
    Invert = 10,
    OrReverse = 11,
    CopyInverted = 12,
    OrInverted = 13,
    Nand = 14,
    Set = 15,
}

impl From<u32> for LogicOp {
    fn from(v: u32) -> Self {
        match v {
            1 => LogicOp::And,
            2 => LogicOp::AndReverse,
            3 => LogicOp::Copy,
            4 => LogicOp::AndInverted,
            5 => LogicOp::NoOp,
            6 => LogicOp::Xor,
            7 => LogicOp::Or,
            8 => LogicOp::Nor,
            9 => LogicOp::Equivalent,
            10 => LogicOp::Invert,
            11 => LogicOp::OrReverse,
            12 => LogicOp::CopyInverted,
            13 => LogicOp::OrInverted,
            14 => LogicOp::Nand,
            15 => LogicOp::Set,
            _ => LogicOp::Clear,
        }
    }
}

impl From<LogicOp> for u32 {
    fn from(o: LogicOp) -> Self {
        o as u32
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineColorBlendStateData {
    pub logic_op_enable: bool,
    pub logic_op: LogicOp,
    pub attachments: Vec<ColorBlendAttachmentState>,
    pub blend_constants: [F32; 4],
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum DynamicState {
    #[default]
    Viewport = 0,
    Scissor = 1,
    LineWidth = 2,
    DepthBias = 3,
    BlendConstants = 4,
    DepthBounds = 5,
    StencilCompareMask = 6,
    StencilWriteMask = 7,
    StencilReference = 8,
}

impl From<u32> for DynamicState {
    fn from(v: u32) -> Self {
        match v {
            0 => DynamicState::Viewport,
            1 => DynamicState::Scissor,
            2 => DynamicState::LineWidth,
            3 => DynamicState::DepthBias,
            4 => DynamicState::BlendConstants,
            5 => DynamicState::DepthBounds,
            6 => DynamicState::StencilCompareMask,
            7 => DynamicState::StencilWriteMask,
            8 => DynamicState::StencilReference,
            _ => DynamicState::Viewport,
        }
    }
}

impl From<i32> for DynamicState {
    fn from(v: i32) -> Self {
        Self::from(v as u32)
    }
}

impl From<DynamicState> for u32 {
    fn from(d: DynamicState) -> Self {
        d as u32
    }
}

impl From<DynamicState> for i32 {
    fn from(d: DynamicState) -> Self {
        d as i32
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PipelineDynamicStateData {
    pub dynamic_states: Vec<DynamicState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphicsPipelineCreateInfoData {
    pub pipeline_id: u64,
    pub stages: Vec<PipelineShaderStageData>,
    pub vertex_input_state: PipelineVertexInputStateData,
    pub input_assembly_state: PipelineInputAssemblyStateData,
    pub viewport_state: Option<PipelineViewportStateData>,
    pub rasterization_state: PipelineRasterizationStateData,
    pub multisample_state: Option<PipelineMultisampleStateData>,
    pub depth_stencil_state: Option<PipelineDepthStencilStateData>,
    pub color_blend_state: Option<PipelineColorBlendStateData>,
    pub dynamic_state: Option<PipelineDynamicStateData>,
    pub layout_id: u64,
    pub render_pass_id: u64,
    pub subpass: u32,
    pub base_pipeline_id: Option<u64>,
    pub base_pipeline_index: i32,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum IndexType {
    #[default]
    Uint16 = 0,
    Uint32 = 1,
}

impl From<u32> for IndexType {
    fn from(v: u32) -> Self {
        match v {
            1 => IndexType::Uint32,
            _ => IndexType::Uint16,
        }
    }
}

impl From<IndexType> for u32 {
    fn from(i: IndexType) -> Self {
        i as u32
    }
}

// ============================================================================
// Phase 3 Graphics: WSI & Swapchain
// ============================================================================

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ColorSpace {
    #[default]
    SrgbNonlinear = 0,
    DisplayP3Nonlinear = 1000104001,
    ExtendedSrgbLinear = 1000104002,
    DisplayP3Linear = 1000104003,
    DciP3Nonlinear = 1000104004,
    Bt709Linear = 1000104005,
    Bt709Nonlinear = 1000104006,
    Bt2020Linear = 1000104007,
    Hdr10St2084 = 1000104008,
    DolbyVision = 1000104009,
    Hdr10Hlg = 1000104010,
    AdobergbLinear = 1000104011,
    AdobergbNonlinear = 1000104012,
    PassThrough = 1000104013,
    ExtendedSrgbNonlinear = 1000104014,
    Other(u32),
}

impl From<u32> for ColorSpace {
    fn from(v: u32) -> Self {
        match v {
            0 => ColorSpace::SrgbNonlinear,
            1000104001 => ColorSpace::DisplayP3Nonlinear,
            1000104002 => ColorSpace::ExtendedSrgbLinear,
            1000104003 => ColorSpace::DisplayP3Linear,
            1000104004 => ColorSpace::DciP3Nonlinear,
            1000104005 => ColorSpace::Bt709Linear,
            1000104006 => ColorSpace::Bt709Nonlinear,
            1000104007 => ColorSpace::Bt2020Linear,
            1000104008 => ColorSpace::Hdr10St2084,
            1000104009 => ColorSpace::DolbyVision,
            1000104010 => ColorSpace::Hdr10Hlg,
            1000104011 => ColorSpace::AdobergbLinear,
            1000104012 => ColorSpace::AdobergbNonlinear,
            1000104013 => ColorSpace::PassThrough,
            1000104014 => ColorSpace::ExtendedSrgbNonlinear,
            other => ColorSpace::Other(other),
        }
    }
}

impl From<ColorSpace> for u32 {
    fn from(c: ColorSpace) -> Self {
        match c {
            ColorSpace::SrgbNonlinear => 0,
            ColorSpace::DisplayP3Nonlinear => 1000104001,
            ColorSpace::ExtendedSrgbLinear => 1000104002,
            ColorSpace::DisplayP3Linear => 1000104003,
            ColorSpace::DciP3Nonlinear => 1000104004,
            ColorSpace::Bt709Linear => 1000104005,
            ColorSpace::Bt709Nonlinear => 1000104006,
            ColorSpace::Bt2020Linear => 1000104007,
            ColorSpace::Hdr10St2084 => 1000104008,
            ColorSpace::DolbyVision => 1000104009,
            ColorSpace::Hdr10Hlg => 1000104010,
            ColorSpace::AdobergbLinear => 1000104011,
            ColorSpace::AdobergbNonlinear => 1000104012,
            ColorSpace::PassThrough => 1000104013,
            ColorSpace::ExtendedSrgbNonlinear => 1000104014,
            ColorSpace::Other(other) => other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct SurfaceFormat {
    pub format: Format,
    pub color_space: ColorSpace,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum PresentMode {
    Immediate = 0,
    Mailbox = 1,
    #[default]
    Fifo = 2,
    FifoRelaxed = 3,
    SharedDemandRefresh = 1000111000,
    SharedContinuousRefresh = 1000111001,
    Other(u32),
}

impl From<u32> for PresentMode {
    fn from(v: u32) -> Self {
        match v {
            0 => PresentMode::Immediate,
            1 => PresentMode::Mailbox,
            2 => PresentMode::Fifo,
            3 => PresentMode::FifoRelaxed,
            1000111000 => PresentMode::SharedDemandRefresh,
            1000111001 => PresentMode::SharedContinuousRefresh,
            other => PresentMode::Other(other),
        }
    }
}

impl From<PresentMode> for u32 {
    fn from(p: PresentMode) -> Self {
        match p {
            PresentMode::Immediate => 0,
            PresentMode::Mailbox => 1,
            PresentMode::Fifo => 2,
            PresentMode::FifoRelaxed => 3,
            PresentMode::SharedDemandRefresh => 1000111000,
            PresentMode::SharedContinuousRefresh => 1000111001,
            PresentMode::Other(other) => other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct SurfaceTransformFlags(pub u32);

impl SurfaceTransformFlags {
    pub const IDENTITY: u32 = 0x00000001;
    pub const ROTATE_90: u32 = 0x00000002;
    pub const ROTATE_180: u32 = 0x00000004;
    pub const ROTATE_270: u32 = 0x00000008;
    pub const HORIZONTAL_MIRROR: u32 = 0x00000010;
    pub const HORIZONTAL_MIRROR_ROTATE_90: u32 = 0x00000020;
    pub const HORIZONTAL_MIRROR_ROTATE_180: u32 = 0x00000040;
    pub const HORIZONTAL_MIRROR_ROTATE_270: u32 = 0x00000080;
    pub const INHERIT: u32 = 0x00000100;

    pub fn contains(&self, flag: u32) -> bool {
        (self.0 & flag) == flag
    }
}

impl From<u32> for SurfaceTransformFlags {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<SurfaceTransformFlags> for u32 {
    fn from(f: SurfaceTransformFlags) -> Self {
        f.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct CompositeAlphaFlags(pub u32);

impl CompositeAlphaFlags {
    pub const OPAQUE: u32 = 0x00000001;
    pub const PRE_MULTIPLIED: u32 = 0x00000002;
    pub const POST_MULTIPLIED: u32 = 0x00000004;
    pub const INHERIT: u32 = 0x00000008;

    pub fn contains(&self, flag: u32) -> bool {
        (self.0 & flag) == flag
    }
}

impl From<u32> for CompositeAlphaFlags {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<CompositeAlphaFlags> for u32 {
    fn from(f: CompositeAlphaFlags) -> Self {
        f.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct SurfaceCapabilities {
    pub min_image_count: u32,
    pub max_image_count: u32,
    pub current_extent: Extent2D,
    pub min_image_extent: Extent2D,
    pub max_image_extent: Extent2D,
    pub max_image_array_layers: u32,
    pub supported_transforms: SurfaceTransformFlags,
    pub current_transform: SurfaceTransformFlags,
    pub supported_composite_alpha: CompositeAlphaFlags,
    pub supported_usage_flags: ImageUsageFlags,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateSwapchainParams {
    pub min_image_count: u32,
    pub image_format: Format,
    pub image_color_space: ColorSpace,
    pub image_extent: Extent2D,
    pub image_array_layers: u32,
    pub image_usage: ImageUsageFlags,
    pub image_sharing_mode: SharingMode,
    pub queue_family_indices: Vec<u32>,
    pub pre_transform: SurfaceTransformFlags,
    pub composite_alpha: CompositeAlphaFlags,
    pub present_mode: PresentMode,
    pub clipped: bool,
    pub old_swapchain_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresentParams {
    pub swapchain_id: u64,
    pub image_index: u32,
    pub wait_semaphores: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FramePacket {
    pub swapchain_id: u64,
    pub image_index: u32,
    pub width: u32,
    pub height: u32,
    pub format: Format,
    // serde_bytes: same postcard bytes as Vec<u8>, but one memcpy instead of per-byte serde.
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

pub type PresentFrameResponse = FramePacket;

// ============================================================================
// Phase 3 Graphics: Barriers & Memory Requirements
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct MemoryBarrierData {
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct BufferMemoryBarrierData {
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub buffer_id: u64,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageMemoryBarrierData {
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
    pub old_layout: ImageLayout,
    pub new_layout: ImageLayout,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub image_id: u64,
    pub subresource_range: ImageSubresourceRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct MemoryRequirementsData {
    pub size: u64,
    pub alignment: u64,
    pub memory_type_bits: u32,
}

// Backward-Compatibility Aliases
pub type SurfaceCapabilitiesData = SurfaceCapabilities;
pub type SurfaceFormatData = SurfaceFormat;

// Result Codes
pub type VkResultCode = i32;
pub const VK_SUCCESS: VkResultCode = 0;
pub const VK_NOT_READY: VkResultCode = 1;
pub const VK_TIMEOUT: VkResultCode = 2;
pub const VK_EVENT_SET: VkResultCode = 3;
pub const VK_EVENT_RESET: VkResultCode = 4;
pub const VK_INCOMPLETE: VkResultCode = 5;
pub const VK_SUBOPTIMAL_KHR: VkResultCode = 1000001003;
pub const VK_ERROR_OUT_OF_HOST_MEMORY: VkResultCode = -1;
pub const VK_ERROR_OUT_OF_DEVICE_MEMORY: VkResultCode = -2;
pub const VK_ERROR_INITIALIZATION_FAILED: VkResultCode = -3;
pub const VK_ERROR_DEVICE_LOST: VkResultCode = -4;
pub const VK_ERROR_MEMORY_MAP_FAILED: VkResultCode = -5;
pub const VK_ERROR_LAYER_NOT_PRESENT: VkResultCode = -6;
pub const VK_ERROR_EXTENSION_NOT_PRESENT: VkResultCode = -7;
pub const VK_ERROR_FEATURE_NOT_PRESENT: VkResultCode = -8;
pub const VK_ERROR_INCOMPATIBLE_DRIVER: VkResultCode = -9;
pub const VK_ERROR_TOO_MANY_OBJECTS: VkResultCode = -10;
pub const VK_ERROR_FORMAT_NOT_SUPPORTED: VkResultCode = -11;
pub const VK_ERROR_FRAGMENTED_POOL: VkResultCode = -12;
pub const VK_ERROR_SURFACE_LOST_KHR: VkResultCode = -1000000000;
pub const VK_ERROR_OUT_OF_DATE_KHR: VkResultCode = -1000001004;

// ============================================================================
// Protocol Requests & Responses
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferImageCopyData {
    pub buffer_offset: u64,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_subresource: [u32; 4],
    pub image_offset: [i32; 3],
    pub image_extent: [u32; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageCopyRegionData {
    pub src_subresource: [u32; 4],
    pub src_offset: [i32; 3],
    pub dst_subresource: [u32; 4],
    pub dst_offset: [i32; 3],
    pub extent: [u32; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageBlitData {
    pub src_subresource: [u32; 4],
    pub src_offsets: [[i32; 3]; 2],
    pub dst_subresource: [u32; 4],
    pub dst_offsets: [[i32; 3]; 2],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Request {
    GetDeviceSnapshot,
    CreateDevice {
        device_id: u64,
        queue_id: u64,
    },
    CreateBuffer {
        buffer_id: u64,
        size: u64,
        usage: u32,
    },
    AllocateMemory {
        memory_id: u64,
        size: u64,
        memory_type_index: u32,
    },
    BindBufferMemory {
        buffer_id: u64,
        memory_id: u64,
        offset: u64,
    },
    WriteMemoryRanges {
        diffs: Vec<MemoryChunkDiff>,
    },
    ReadMemoryRanges {
        allocation_ids: Vec<u64>,
    },
    CreateShaderModule {
        shader_id: u64,
        spirv: Vec<u8>,
    },
    CreateDescriptorSetLayout {
        layout_id: u64,
        bindings: Vec<DescriptorSetLayoutBindingData>,
    },
    CreatePipelineLayout {
        layout_id: u64,
        set_layout_ids: Vec<u64>,
        push_constant_ranges: Vec<PushConstantRangeData>,
    },
    CreateComputePipeline {
        pipeline_id: u64,
        layout_id: u64,
        shader_id: u64,
        entry_point: String,
    },
    CreateDescriptorPool {
        pool_id: u64,
        max_sets: u32,
        pool_sizes: Vec<DescriptorPoolSizeData>,
    },
    AllocateDescriptorSets {
        pool_id: u64,
        layout_ids: Vec<u64>,
        set_ids: Vec<u64>,
    },
    UpdateDescriptorSets {
        writes: Vec<WriteDescriptorSetData>,
    },
    CreateCommandPool {
        pool_id: u64,
    },
    AllocateCommandBuffers {
        pool_id: u64,
        buffer_ids: Vec<u64>,
    },
    BeginCommandBuffer {
        buffer_id: u64,
    },
    CmdBindPipeline {
        buffer_id: u64,
        pipeline_bind_point: PipelineBindPoint,
        pipeline_id: u64,
    },
    CmdBindDescriptorSets {
        buffer_id: u64,
        pipeline_bind_point: PipelineBindPoint,
        layout_id: u64,
        first_set: u32,
        set_ids: Vec<u64>,
        dynamic_offsets: Vec<u32>,
    },
    CmdPushConstants {
        buffer_id: u64,
        layout_id: u64,
        stage_flags: u32,
        offset: u32,
        values: Vec<u8>,
    },
    CmdDispatch {
        buffer_id: u64,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    },
    CmdCopyBuffer {
        buffer_id: u64,
        src_buffer_id: u64,
        dst_buffer_id: u64,
        regions: Vec<BufferCopyData>,
    },
    CmdFillBuffer {
        buffer_id: u64,
        dst_buffer_id: u64,
        dst_offset: u64,
        size: u64,
        data: u32,
    },
    CmdPipelineBarrier {
        buffer_id: u64,
        src_stage_mask: u32,
        dst_stage_mask: u32,
        dependency_flags: u32,
        memory_barriers: Vec<MemoryBarrierData>,
        buffer_memory_barriers: Vec<BufferMemoryBarrierData>,
        image_memory_barriers: Vec<ImageMemoryBarrierData>,
    },
    EndCommandBuffer {
        buffer_id: u64,
    },
    CreateFence {
        fence_id: u64,
        signaled: bool,
    },
    QueueSubmit {
        queue_id: u64,
        command_buffer_ids: Vec<u64>,
        fence_id: Option<u64>,
        dirty_memory: Vec<MemoryChunkDiff>,
    },
    WaitForFences {
        fence_ids: Vec<u64>,
        wait_all: bool,
        timeout_ns: u64,
    },
    GetFenceStatus {
        fence_id: u64,
    },
    ResetFences {
        fence_ids: Vec<u64>,
    },
    QueueWaitIdle {
        queue_id: u64,
    },
    DeviceWaitIdle {
        device_id: u64,
    },
    DestroyResource {
        handle_id: u64,
        resource_type: ResourceType,
    },

    // Phase 3 Graphics: Images & Views
    CreateImage {
        image_id: u64,
        params: CreateImageParams,
    },
    DestroyImage {
        image_id: u64,
    },
    GetImageMemoryRequirements {
        image_id: u64,
    },
    BindImageMemory {
        image_id: u64,
        memory_id: u64,
        offset: u64,
    },
    CreateImageView {
        view_id: u64,
        params: CreateImageViewParams,
    },
    DestroyImageView {
        view_id: u64,
    },

    // Phase 3 Graphics: Samplers
    CreateSampler {
        sampler_id: u64,
        params: CreateSamplerParams,
    },
    DestroySampler {
        sampler_id: u64,
    },

    // Phase 3 Graphics: RenderPass & Framebuffer
    CreateRenderPass {
        render_pass_id: u64,
        params: CreateRenderPassParams,
    },
    DestroyRenderPass {
        render_pass_id: u64,
    },
    CreateFramebuffer {
        framebuffer_id: u64,
        params: CreateFramebufferParams,
    },
    DestroyFramebuffer {
        framebuffer_id: u64,
    },

    // Phase 3 Graphics: Pipelines
    CreateGraphicsPipelines {
        pipelines: Vec<GraphicsPipelineCreateInfoData>,
    },

    // Phase 3 Graphics: Command Recording
    CmdBeginRenderPass {
        buffer_id: u64,
        render_pass_id: u64,
        framebuffer_id: u64,
        render_area: Rect2D,
        clear_values: Vec<ClearValueData>,
        contents: SubpassContents,
    },
    CmdEndRenderPass {
        buffer_id: u64,
    },
    CmdBindVertexBuffers {
        buffer_id: u64,
        first_binding: u32,
        buffer_ids: Vec<u64>,
        offsets: Vec<u64>,
    },
    CmdBindIndexBuffer {
        buffer_id: u64,
        index_buffer_id: u64,
        offset: u64,
        index_type: IndexType,
    },
    CmdSetViewport {
        buffer_id: u64,
        first_viewport: u32,
        viewports: Vec<Viewport>,
    },
    CmdSetScissor {
        buffer_id: u64,
        first_scissor: u32,
        scissors: Vec<Rect2D>,
    },
    CmdDraw {
        buffer_id: u64,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    CmdDrawIndexed {
        buffer_id: u64,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    },

    /// A completed command buffer's recording stream. Keeping the existing
    /// command request variants inside the batch makes this protocol extension
    /// additive while eliminating one network round trip per vkCmd* call.
    CommandBatch {
        commands: Vec<Request>,
    },

    // Phase 3 Graphics: WSI & Swapchain
    GetSurfaceCapabilities {
        surface_id: u64,
    },
    GetSurfaceFormats {
        surface_id: u64,
    },
    GetSurfacePresentModes {
        surface_id: u64,
    },
    CreateSwapchain {
        swapchain_id: u64,
        surface_id: u64,
        params: CreateSwapchainParams,
    },
    DestroySwapchain {
        swapchain_id: u64,
    },
    GetSwapchainImages {
        swapchain_id: u64,
    },
    AcquireNextImage {
        swapchain_id: u64,
        timeout_ns: u64,
        semaphore_id: Option<u64>,
        fence_id: Option<u64>,
    },
    QueuePresent {
        queue_id: u64,
        params: PresentParams,
    },
    CmdCopyBufferToImage {
        buffer_id: u64,
        src_buffer_id: u64,
        dst_image_id: u64,
        dst_layout: i32,
        regions: Vec<BufferImageCopyData>,
    },
    CmdCopyImageToBuffer {
        buffer_id: u64,
        src_image_id: u64,
        src_layout: i32,
        dst_buffer_id: u64,
        regions: Vec<BufferImageCopyData>,
    },
    CmdCopyImage {
        buffer_id: u64,
        src_image_id: u64,
        src_layout: i32,
        dst_image_id: u64,
        dst_layout: i32,
        regions: Vec<ImageCopyRegionData>,
    },
    CmdBlitImage {
        buffer_id: u64,
        src_image_id: u64,
        src_layout: i32,
        dst_image_id: u64,
        dst_layout: i32,
        regions: Vec<ImageBlitData>,
        filter: i32,
    },
    CmdResolveImage {
        buffer_id: u64,
        src_image_id: u64,
        src_layout: i32,
        dst_image_id: u64,
        dst_layout: i32,
        regions: Vec<ImageCopyRegionData>,
    },
    CmdDrawIndirect {
        buffer_id: u64,
        indirect_buffer_id: u64,
        offset: u64,
        draw_count: u32,
        stride: u32,
        indexed: bool,
    },
    CmdDrawIndirectCount {
        buffer_id: u64,
        indirect_buffer_id: u64,
        offset: u64,
        count_buffer_id: u64,
        count_buffer_offset: u64,
        max_draw_count: u32,
        stride: u32,
        indexed: bool,
    },
    CmdDispatchIndirect {
        buffer_id: u64,
        indirect_buffer_id: u64,
        offset: u64,
    },
    CmdUpdateBuffer {
        buffer_id: u64,
        dst_buffer_id: u64,
        offset: u64,
        data: Vec<u8>,
    },
    CmdClearColorImage {
        buffer_id: u64,
        image_id: u64,
        layout: i32,
        color: [u32; 4],
        ranges: Vec<ImageSubresourceRange>,
    },
    CmdClearDepthStencilImage {
        buffer_id: u64,
        image_id: u64,
        layout: i32,
        depth_bits: u32,
        stencil: u32,
        ranges: Vec<ImageSubresourceRange>,
    },
    GetBufferMemoryRequirements {
        buffer_id: u64,
    },
    /// vkGetImageSubresourceLayout. Appended last to keep earlier variant indices stable.
    GetImageSubresourceLayout {
        image_id: u64,
        aspect_mask: u32,
        mip_level: u32,
        array_layer: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Response {
    Success,
    DeviceSnapshot(PhysicalDeviceSnapshot),
    MemoryData {
        diffs: Vec<MemoryChunkDiff>,
    },
    FenceStatus {
        signaled: bool,
    },
    WaitResult {
        success: bool,
        timeout: bool,
    },
    Error {
        code: i32,
        message: String,
    },

    // Phase 3 Graphics Responses
    ImageCreated {
        id: u64,
    },
    ImageViewCreated {
        id: u64,
    },
    SamplerCreated {
        id: u64,
    },
    RenderPassCreated {
        id: u64,
    },
    FramebufferCreated {
        id: u64,
    },
    GraphicsPipelinesCreated {
        ids: Vec<u64>,
    },
    MemoryRequirements {
        size: u64,
        alignment: u64,
        memory_type_bits: u32,
    },
    ImageMemoryRequirements(MemoryRequirementsData),
    SurfaceCapabilities(SurfaceCapabilities),
    SurfaceFormats(Vec<SurfaceFormat>),
    SurfacePresentModes(Vec<u32>),
    SwapchainCreated {
        id: u64,
        image_ids: Vec<u64>,
    },
    SwapchainImages(Vec<u64>),
    AcquireNextImage {
        image_index: u32,
        suboptimal: bool,
    },
    AcquireNextImageResult {
        image_index: u32,
        suboptimal: bool,
    },
    QueuePresentResult {
        result: VkResultCode,
        frame: Option<FramePacket>,
    },
    /// Reply to `Request::GetImageSubresourceLayout`. Appended last for index stability.
    ImageSubresourceLayout {
        offset: u64,
        size: u64,
        row_pitch: u64,
        array_pitch: u64,
        depth_pitch: u64,
    },
    /// Like `QueuePresentResult`, but `frame.data` is the XOR of this frame with the
    /// previous frame sent for the same swapchain (same size). Unchanged pixels are
    /// zero, so zstd shrinks a mostly static UI frame from ~1 MB to a few KB.
    /// Only sent when the server runs with `VKREMOTE_FRAME_DELTA=1`.
    QueuePresentDelta {
        result: VkResultCode,
        frame: FramePacket,
    },
}
