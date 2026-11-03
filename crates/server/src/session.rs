use anyhow::{anyhow, Result};
use ash::vk;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::error;
use vkremote_wire::types::*;
use vkremote_wire::CHUNK_SIZE;

use crate::engine::{Engine, MockEngine, VulkanEngine};

/// Wrapper around a raw pointer to mapped host memory.
/// # Safety
/// VulkanSession is processed sequentially per client connection task, so
/// host-mapped pointers are accessed only by this connection without data races.
#[derive(Clone, Copy)]
pub struct SendPtr(pub *mut u8);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

#[inline]
fn to_vk_image_layout(layout: ImageLayout) -> vk::ImageLayout {
    vk::ImageLayout::from_raw(u32::from(layout) as i32)
}

fn vk_layers(v: [u32; 4]) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::from_raw(v[0]),
        mip_level: v[1],
        base_array_layer: v[2],
        layer_count: v[3],
    }
}
fn vk_offset(v: [i32; 3]) -> vk::Offset3D {
    vk::Offset3D {
        x: v[0],
        y: v[1],
        z: v[2],
    }
}
fn vk_extent(v: [u32; 3]) -> vk::Extent3D {
    vk::Extent3D {
        width: v[0],
        height: v[1],
        depth: v[2],
    }
}
fn vk_range(v: ImageSubresourceRange) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::from_raw(v.aspect_mask.0),
        base_mip_level: v.base_mip_level,
        level_count: v.level_count,
        base_array_layer: v.base_array_layer,
        layer_count: v.layer_count,
    }
}
fn vk_buffer_image(r: &BufferImageCopyData) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: r.buffer_offset,
        buffer_row_length: r.buffer_row_length,
        buffer_image_height: r.buffer_image_height,
        image_subresource: vk_layers(r.image_subresource),
        image_offset: vk_offset(r.image_offset),
        image_extent: vk_extent(r.image_extent),
    }
}

struct BufferState {
    vk_buffer: vk::Buffer,
    _size: u64,
    _usage: u32,
    _bound_memory: Option<(u64, u64)>,
}

struct MemoryState {
    vk_memory: vk::DeviceMemory,
    size: u64,
    _memory_type_index: u32,
    _is_host_visible: bool,
    is_host_coherent: bool,
    mapped_ptr: Option<SendPtr>,
}

struct ImageState {
    vk_image: vk::Image,
    _format: vk::Format,
    _extent: vk::Extent3D,
    _mip_levels: u32,
    _array_layers: u32,
    bound_memory: Option<(u64, u64)>,
    is_swapchain_image: bool,
    current_layout: vk::ImageLayout,
}

struct SwapchainState {
    _surface_id: u64,
    width: u32,
    height: u32,
    format: Format,
    images: Vec<(u64, vk::Image, vk::DeviceMemory)>,
    staging_buffers: Vec<(vk::Buffer, vk::DeviceMemory, SendPtr, u64)>,
    current_index: u32,
}

pub struct VulkanSession {
    engine: Arc<VulkanEngine>,
    device: Option<ash::Device>,
    queue: Option<vk::Queue>,
    internal_cmd_pool: Option<vk::CommandPool>,
    buffers: HashMap<u64, BufferState>,
    memories: HashMap<u64, MemoryState>,
    images: HashMap<u64, ImageState>,
    image_views: HashMap<u64, vk::ImageView>,
    samplers: HashMap<u64, vk::Sampler>,
    render_passes: HashMap<u64, vk::RenderPass>,
    framebuffers: HashMap<u64, vk::Framebuffer>,
    swapchains: HashMap<u64, SwapchainState>,
    image_view_images: HashMap<u64, u64>,
    render_pass_final_layouts: HashMap<u64, Vec<vk::ImageLayout>>,
    framebuffer_attachments: HashMap<u64, Vec<u64>>,
    cmd_buffer_render_passes: HashMap<u64, (u64, u64)>,
    shader_modules: HashMap<u64, vk::ShaderModule>,
    descriptor_set_layouts: HashMap<u64, vk::DescriptorSetLayout>,
    pipeline_layouts: HashMap<u64, vk::PipelineLayout>,
    pipelines: HashMap<u64, vk::Pipeline>,
    descriptor_pools: HashMap<u64, vk::DescriptorPool>,
    descriptor_sets: HashMap<u64, vk::DescriptorSet>,
    command_pools: HashMap<u64, vk::CommandPool>,
    command_buffers: HashMap<u64, vk::CommandBuffer>,
    fences: HashMap<u64, vk::Fence>,
}

impl VulkanSession {
    pub fn new(engine: Arc<VulkanEngine>) -> Self {
        Self {
            engine,
            device: None,
            queue: None,
            internal_cmd_pool: None,
            buffers: HashMap::new(),
            memories: HashMap::new(),
            images: HashMap::new(),
            image_views: HashMap::new(),
            samplers: HashMap::new(),
            render_passes: HashMap::new(),
            framebuffers: HashMap::new(),
            swapchains: HashMap::new(),
            image_view_images: HashMap::new(),
            render_pass_final_layouts: HashMap::new(),
            framebuffer_attachments: HashMap::new(),
            cmd_buffer_render_passes: HashMap::new(),
            shader_modules: HashMap::new(),
            descriptor_set_layouts: HashMap::new(),
            pipeline_layouts: HashMap::new(),
            pipelines: HashMap::new(),
            descriptor_pools: HashMap::new(),
            descriptor_sets: HashMap::new(),
            command_pools: HashMap::new(),
            command_buffers: HashMap::new(),
            fences: HashMap::new(),
        }
    }

    fn find_memory_type_index(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for (i, mem_type) in self
            .engine
            .memory_properties
            .memory_types
            .iter()
            .enumerate()
        {
            if (type_filter & (1 << i)) != 0 && (mem_type.property_flags & properties) == properties
            {
                return Some(i as u32);
            }
        }
        None
    }

    fn device(&self) -> Result<&ash::Device> {
        self.device
            .as_ref()
            .ok_or_else(|| anyhow!("Logical device has not been created yet"))
    }

    pub fn handle_request(&mut self, req: Request) -> Response {
        // Requests can contain complete memory chunks and frames. Logging their Debug
        // representation serializes megabytes to disk on the session's hot path.
        tracing::debug!(request = ?std::mem::discriminant(&req), "Received request");
        match self.process_request(req) {
            Ok(resp) => resp,
            Err(e) => {
                error!("Vulkan request processing error: {:#}", e);
                Response::Error {
                    code: -1,
                    message: format!("{:#}", e),
                }
            }
        }
    }

    fn process_request(&mut self, req: Request) -> Result<Response> {
        match req {
            Request::CmdCopyBufferToImage {
                buffer_id,
                src_buffer_id,
                dst_image_id,
                dst_layout,
                regions,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer not found"))?;
                let buffer = self
                    .buffers
                    .get(&src_buffer_id)
                    .ok_or_else(|| anyhow!("Buffer not found"))?
                    .vk_buffer;
                let image = self
                    .images
                    .get(&dst_image_id)
                    .ok_or_else(|| anyhow!("Image not found"))?
                    .vk_image;
                let regions: Vec<_> = regions.iter().map(vk_buffer_image).collect();
                unsafe {
                    dev.cmd_copy_buffer_to_image(
                        *cb,
                        buffer,
                        image,
                        vk::ImageLayout::from_raw(dst_layout),
                        &regions,
                    );
                }
                Ok(Response::Success)
            }
            Request::CmdCopyImageToBuffer {
                buffer_id,
                src_image_id,
                src_layout,
                dst_buffer_id,
                regions,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer not found"))?;
                let buffer = self
                    .buffers
                    .get(&dst_buffer_id)
                    .ok_or_else(|| anyhow!("Buffer not found"))?
                    .vk_buffer;
                let image = self
                    .images
                    .get(&src_image_id)
                    .ok_or_else(|| anyhow!("Image not found"))?
                    .vk_image;
                let regions: Vec<_> = regions.iter().map(vk_buffer_image).collect();
                unsafe {
                    dev.cmd_copy_image_to_buffer(
                        *cb,
                        image,
                        vk::ImageLayout::from_raw(src_layout),
                        buffer,
                        &regions,
                    );
                }
                Ok(Response::Success)
            }
            Request::CmdCopyImage {
                buffer_id,
                src_image_id,
                src_layout,
                dst_image_id,
                dst_layout,
                regions,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer not found"))?;
                let src = self
                    .images
                    .get(&src_image_id)
                    .ok_or_else(|| anyhow!("Source image not found"))?
                    .vk_image;
                let dst = self
                    .images
                    .get(&dst_image_id)
                    .ok_or_else(|| anyhow!("Destination image not found"))?
                    .vk_image;
                let regions: Vec<_> = regions
                    .iter()
                    .map(|r| vk::ImageCopy {
                        src_subresource: vk_layers(r.src_subresource),
                        src_offset: vk_offset(r.src_offset),
                        dst_subresource: vk_layers(r.dst_subresource),
                        dst_offset: vk_offset(r.dst_offset),
                        extent: vk_extent(r.extent),
                    })
                    .collect();
                tracing::info!("CmdCopyImage: src={} dst={}", src_image_id, dst_image_id);
                unsafe {
                    dev.cmd_copy_image(
                        *cb,
                        src,
                        vk::ImageLayout::from_raw(src_layout),
                        dst,
                        vk::ImageLayout::from_raw(dst_layout),
                        &regions,
                    );
                }
                Ok(Response::Success)
            }
            Request::CmdBlitImage {
                buffer_id,
                src_image_id,
                src_layout,
                dst_image_id,
                dst_layout,
                regions,
                filter,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer not found"))?;
                let src = self
                    .images
                    .get(&src_image_id)
                    .ok_or_else(|| anyhow!("Source image not found"))?
                    .vk_image;
                let dst = self
                    .images
                    .get(&dst_image_id)
                    .ok_or_else(|| anyhow!("Destination image not found"))?
                    .vk_image;
                let regions: Vec<_> = regions
                    .iter()
                    .map(|r| vk::ImageBlit {
                        src_subresource: vk_layers(r.src_subresource),
                        src_offsets: r.src_offsets.map(vk_offset),
                        dst_subresource: vk_layers(r.dst_subresource),
                        dst_offsets: r.dst_offsets.map(vk_offset),
                    })
                    .collect();
                tracing::info!("CmdBlitImage: src={} dst={}", src_image_id, dst_image_id);
                unsafe {
                    dev.cmd_blit_image(
                        *cb,
                        src,
                        vk::ImageLayout::from_raw(src_layout),
                        dst,
                        vk::ImageLayout::from_raw(dst_layout),
                        &regions,
                        vk::Filter::from_raw(filter),
                    );
                }
                Ok(Response::Success)
            }
            Request::CmdResolveImage {
                buffer_id,
                src_image_id,
                src_layout,
                dst_image_id,
                dst_layout,
                regions,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer not found"))?;
                let src = self
                    .images
                    .get(&src_image_id)
                    .ok_or_else(|| anyhow!("Source image not found"))?
                    .vk_image;
                let dst = self
                    .images
                    .get(&dst_image_id)
                    .ok_or_else(|| anyhow!("Destination image not found"))?
                    .vk_image;
                let regions: Vec<_> = regions
                    .iter()
                    .map(|r| vk::ImageResolve {
                        src_subresource: vk_layers(r.src_subresource),
                        src_offset: vk_offset(r.src_offset),
                        dst_subresource: vk_layers(r.dst_subresource),
                        dst_offset: vk_offset(r.dst_offset),
                        extent: vk_extent(r.extent),
                    })
                    .collect();
                tracing::info!("CmdResolveImage: src={} dst={}", src_image_id, dst_image_id);
                unsafe {
                    dev.cmd_resolve_image(
                        *cb,
                        src,
                        vk::ImageLayout::from_raw(src_layout),
                        dst,
                        vk::ImageLayout::from_raw(dst_layout),
                        &regions,
                    );
                }
                Ok(Response::Success)
            }
            Request::CmdDrawIndirect {
                buffer_id,
                indirect_buffer_id,
                offset,
                draw_count,
                stride,
                indexed,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer not found"))?;
                let buffer = self
                    .buffers
                    .get(&indirect_buffer_id)
                    .ok_or_else(|| anyhow!("Indirect buffer not found"))?
                    .vk_buffer;
                unsafe {
                    if indexed {
                        dev.cmd_draw_indexed_indirect(*cb, buffer, offset, draw_count, stride)
                    } else {
                        dev.cmd_draw_indirect(*cb, buffer, offset, draw_count, stride)
                    }
                }
                Ok(Response::Success)
            }
            Request::CmdDrawIndirectCount {
                buffer_id,
                indirect_buffer_id,
                offset,
                count_buffer_id,
                count_buffer_offset,
                max_draw_count,
                stride,
                indexed,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer not found"))?;
                let buffer = self
                    .buffers
                    .get(&indirect_buffer_id)
                    .ok_or_else(|| anyhow!("Indirect buffer not found"))?
                    .vk_buffer;
                let count = self
                    .buffers
                    .get(&count_buffer_id)
                    .ok_or_else(|| anyhow!("Count buffer not found"))?
                    .vk_buffer;
                unsafe {
                    if indexed {
                        dev.cmd_draw_indexed_indirect_count(
                            *cb,
                            buffer,
                            offset,
                            count,
                            count_buffer_offset,
                            max_draw_count,
                            stride,
                        )
                    } else {
                        dev.cmd_draw_indirect_count(
                            *cb,
                            buffer,
                            offset,
                            count,
                            count_buffer_offset,
                            max_draw_count,
                            stride,
                        )
                    }
                }
                Ok(Response::Success)
            }
            Request::CmdDispatchIndirect {
                buffer_id,
                indirect_buffer_id,
                offset,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer not found"))?;
                let buffer = self
                    .buffers
                    .get(&indirect_buffer_id)
                    .ok_or_else(|| anyhow!("Indirect buffer not found"))?
                    .vk_buffer;
                unsafe {
                    dev.cmd_dispatch_indirect(*cb, buffer, offset);
                }
                Ok(Response::Success)
            }
            Request::CmdUpdateBuffer {
                buffer_id,
                dst_buffer_id,
                offset,
                data,
            } => {
                if data.is_empty() || data.len() > 65536 || data.len() % 4 != 0 || offset % 4 != 0 {
                    return Err(anyhow!("Invalid update buffer alignment or size"));
                }
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer not found"))?;
                let buffer = self
                    .buffers
                    .get(&dst_buffer_id)
                    .ok_or_else(|| anyhow!("Destination buffer not found"))?
                    .vk_buffer;
                unsafe {
                    dev.cmd_update_buffer(*cb, buffer, offset, &data);
                }
                Ok(Response::Success)
            }
            Request::CmdClearColorImage {
                buffer_id,
                image_id,
                layout,
                color,
                ranges,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer not found"))?;
                let image = self
                    .images
                    .get(&image_id)
                    .ok_or_else(|| anyhow!("Image not found"))?
                    .vk_image;
                let ranges: Vec<_> = ranges.into_iter().map(vk_range).collect();
                unsafe {
                    dev.cmd_clear_color_image(
                        *cb,
                        image,
                        vk::ImageLayout::from_raw(layout),
                        &vk::ClearColorValue { uint32: color },
                        &ranges,
                    );
                }
                Ok(Response::Success)
            }
            Request::CmdClearDepthStencilImage {
                buffer_id,
                image_id,
                layout,
                depth_bits,
                stencil,
                ranges,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer not found"))?;
                let image = self
                    .images
                    .get(&image_id)
                    .ok_or_else(|| anyhow!("Image not found"))?
                    .vk_image;
                let ranges: Vec<_> = ranges.into_iter().map(vk_range).collect();
                unsafe {
                    dev.cmd_clear_depth_stencil_image(
                        *cb,
                        image,
                        vk::ImageLayout::from_raw(layout),
                        &vk::ClearDepthStencilValue {
                            depth: f32::from_bits(depth_bits),
                            stencil,
                        },
                        &ranges,
                    );
                }
                Ok(Response::Success)
            }
            Request::GetBufferMemoryRequirements { buffer_id } => {
                let dev = self.device()?;
                let buffer = self
                    .buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Buffer not found"))?
                    .vk_buffer;
                let req = unsafe { dev.get_buffer_memory_requirements(buffer) };
                Ok(Response::ImageMemoryRequirements(MemoryRequirementsData {
                    size: req.size,
                    alignment: req.alignment,
                    memory_type_bits: req.memory_type_bits,
                }))
            }
            Request::CommandBatch { commands } => {
                let cmd_count = commands.len();
                let has_draw = commands.iter().any(|c| matches!(c, Request::CmdDraw { .. } | Request::CmdDrawIndexed { .. }));
                let has_rp = commands.iter().any(|c| matches!(c, Request::CmdBeginRenderPass { .. }));
                tracing::debug!("CommandBatch: {} commands (has_rp={}, has_draw={})", cmd_count, has_rp, has_draw);
                // Each batched command is independent; a failure must not drop the
                // commands after it. Every failure is logged. The batch replies with the
                // LAST command's response so a client can piggyback one synchronous
                // request on the end of its deferred queue (one round trip, not two).
                let mut last: Result<Response> = Ok(Response::Success);
                for command in commands {
                    if matches!(command, Request::CommandBatch { .. }) {
                        return Err(anyhow!("Nested command batches are not supported"));
                    }
                    let kind = vkremote_wire::perf::variant_name(&command);
                    last = self.process_request(command);
                    match &last {
                        Err(e) => error!("Batched {kind} failed: {e:#}"),
                        Ok(Response::Error { code, message }) => {
                            error!("Batched {kind} failed ({code}): {message}")
                        }
                        // Deferred creations (CreateImageView, CreateFramebuffer, ...) use
                        // client-assigned ids and reply with an echo such as ImageViewCreated.
                        // The client never reads replies to batched commands, so any non-error
                        // reply is success. Aborting here dropped every command after the first
                        // creation in the batch (missing framebuffers -> black frames).
                        _ => {}
                    }
                }
                last
            }
            Request::GetDeviceSnapshot => {
                Ok(Response::DeviceSnapshot(self.engine.snapshot.clone()))
            }
            Request::CreateDevice {
                queue_id: _,
                device_id: _,
            } => {
                let (dev, q) = self.engine.create_device()?;
                let pool_info = vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.engine.queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
                let internal_cmd_pool = unsafe { dev.create_command_pool(&pool_info, None)? };
                self.device = Some(dev);
                self.queue = Some(q);
                self.internal_cmd_pool = Some(internal_cmd_pool);
                Ok(Response::Success)
            }
            Request::CreateBuffer {
                buffer_id,
                size,
                usage,
            } => {
                let dev = self.device()?;
                let create_info = vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::from_raw(usage))
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                // Safety: create_info contains valid buffer parameters and device is valid.
                let vk_buffer = unsafe { dev.create_buffer(&create_info, None)? };
                self.buffers.insert(
                    buffer_id,
                    BufferState {
                        vk_buffer,
                        _size: size,
                        _usage: usage,
                        _bound_memory: None,
                    },
                );
                Ok(Response::Success)
            }
            Request::AllocateMemory {
                memory_id,
                size,
                memory_type_index,
            } => {
                let dev = self.device()?;
                let alloc_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(size)
                    .memory_type_index(memory_type_index);
                // Safety: alloc_info contains a valid memory type index and size.
                let vk_memory = unsafe { dev.allocate_memory(&alloc_info, None)? };

                let mem_type = self
                    .engine
                    .memory_properties
                    .memory_types
                    .get(memory_type_index as usize)
                    .ok_or_else(|| anyhow!("Invalid memory type index {}", memory_type_index))?;

                let is_host_visible = mem_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
                let is_host_coherent = mem_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_COHERENT);

                let mapped_ptr = if is_host_visible {
                    // Safety: memory is host-visible and offset/size are within allocation bounds.
                    let ptr =
                        unsafe { dev.map_memory(vk_memory, 0, size, vk::MemoryMapFlags::empty())? };
                    Some(SendPtr(ptr as *mut u8))
                } else {
                    None
                };

                self.memories.insert(
                    memory_id,
                    MemoryState {
                        vk_memory,
                        size,
                        _memory_type_index: memory_type_index,
                        _is_host_visible: is_host_visible,
                        is_host_coherent,
                        mapped_ptr,
                    },
                );
                Ok(Response::Success)
            }
            Request::BindBufferMemory {
                buffer_id,
                memory_id,
                offset,
            } => {
                let dev = self.device()?.clone();
                let mem = self
                    .memories
                    .get(&memory_id)
                    .ok_or_else(|| anyhow!("Memory {} not found", memory_id))?;
                let mem_vk = mem.vk_memory;

                let buf = self
                    .buffers
                    .get_mut(&buffer_id)
                    .ok_or_else(|| anyhow!("Buffer {} not found", buffer_id))?;

                // Safety: buffer and device memory handles are valid and non-overlapping.
                unsafe {
                    dev.bind_buffer_memory(buf.vk_buffer, mem_vk, offset)?;
                }
                buf._bound_memory = Some((memory_id, offset));
                Ok(Response::Success)
            }
            Request::WriteMemoryRanges { diffs } => {
                tracing::info!("WriteMemoryRanges: {} diffs", diffs.len());
                let dev = self.device()?;
                for diff in diffs {
                    let mem = self
                        .memories
                        .get(&diff.allocation_id)
                        .ok_or_else(|| anyhow!("Memory {} not found", diff.allocation_id))?;
                    let ptr = mem.mapped_ptr.ok_or_else(|| {
                        anyhow!("Memory {} is not host mapped", diff.allocation_id)
                    })?;

                    if diff.offset + diff.data.len() as u64 > mem.size {
                        return Err(anyhow!("Write range exceeds allocation bounds"));
                    }

                    // Safety: bounds checked above; pointer is valid and non-overlapping.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            diff.data.as_ptr(),
                            ptr.0.add(diff.offset as usize),
                            diff.data.len(),
                        );
                    }

                    if !mem.is_host_coherent {
                        let range = vk::MappedMemoryRange::default()
                            .memory(mem.vk_memory)
                            .offset(diff.offset)
                            .size(diff.data.len() as u64);
                        // Safety: memory range is mapped and bounds checked.
                        unsafe {
                            dev.flush_mapped_memory_ranges(&[range])?;
                        }
                    }
                }
                Ok(Response::Success)
            }
            Request::ReadMemoryRanges { allocation_ids } => {
                let dev = self.device()?;
                let mut diffs = Vec::new();

                for alloc_id in allocation_ids {
                    let mem = self
                        .memories
                        .get(&alloc_id)
                        .ok_or_else(|| anyhow!("Memory {} not found", alloc_id))?;
                    let ptr = mem
                        .mapped_ptr
                        .ok_or_else(|| anyhow!("Memory {} is not host mapped", alloc_id))?;

                    if !mem.is_host_coherent {
                        let range = vk::MappedMemoryRange::default()
                            .memory(mem.vk_memory)
                            .offset(0)
                            .size(mem.size);
                        // Safety: memory range is mapped and bounds checked.
                        unsafe {
                            dev.invalidate_mapped_memory_ranges(&[range])?;
                        }
                    }

                    let mut data = vec![0u8; mem.size as usize];
                    // Safety: ptr.0 points to mapped host memory of at least mem.size bytes.
                    unsafe {
                        std::ptr::copy_nonoverlapping(ptr.0, data.as_mut_ptr(), mem.size as usize);
                    }

                    for (chunk_idx, chunk) in data.chunks(CHUNK_SIZE).enumerate() {
                        diffs.push(MemoryChunkDiff {
                            allocation_id: alloc_id,
                            offset: (chunk_idx * CHUNK_SIZE) as u64,
                            data: chunk.to_vec(),
                        });
                    }
                }
                Ok(Response::MemoryData { diffs })
            }
            Request::CreateShaderModule { shader_id, spirv } => {
                let dev = self.device()?;
                if spirv.len() % 4 != 0 {
                    return Err(anyhow!(
                        "SPIR-V length {} is not a multiple of 4",
                        spirv.len()
                    ));
                }
                let code: Vec<u32> = spirv
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
                // Safety: SPIR-V bytecode is aligned to 4 bytes and slice length is verified.
                let sm = unsafe { dev.create_shader_module(&create_info, None)? };
                self.shader_modules.insert(shader_id, sm);
                Ok(Response::Success)
            }
            Request::CreateDescriptorSetLayout {
                layout_id,
                bindings,
            } => {
                let dev = self.device()?;
                let vk_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
                    .iter()
                    .map(|b| {
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(b.binding)
                            .descriptor_type(vk::DescriptorType::from_raw(b.descriptor_type as i32))
                            .descriptor_count(b.descriptor_count)
                            .stage_flags(vk::ShaderStageFlags::from_raw(b.stage_flags))
                    })
                    .collect();
                let create_info =
                    vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);
                // Safety: bindings and device handles are valid.
                let layout = unsafe { dev.create_descriptor_set_layout(&create_info, None)? };
                self.descriptor_set_layouts.insert(layout_id, layout);
                Ok(Response::Success)
            }
            Request::CreatePipelineLayout {
                layout_id,
                set_layout_ids,
                push_constant_ranges,
            } => {
                let dev = self.device()?;
                let mut vk_set_layouts = Vec::new();
                for sid in &set_layout_ids {
                    let l = self
                        .descriptor_set_layouts
                        .get(sid)
                        .ok_or_else(|| anyhow!("Descriptor set layout {} not found", sid))?;
                    vk_set_layouts.push(*l);
                }
                let vk_push_constants: Vec<vk::PushConstantRange> = push_constant_ranges
                    .iter()
                    .map(|p| {
                        vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::from_raw(p.stage_flags))
                            .offset(p.offset)
                            .size(p.size)
                    })
                    .collect();
                let create_info = vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&vk_set_layouts)
                    .push_constant_ranges(&vk_push_constants);
                // Safety: descriptor set layouts and push constant ranges are valid.
                let layout = unsafe { dev.create_pipeline_layout(&create_info, None)? };
                self.pipeline_layouts.insert(layout_id, layout);
                Ok(Response::Success)
            }
            Request::CreateComputePipeline {
                pipeline_id,
                layout_id,
                shader_id,
                entry_point,
            } => {
                let dev = self.device()?;
                let sm = self
                    .shader_modules
                    .get(&shader_id)
                    .ok_or_else(|| anyhow!("Shader module {} not found", shader_id))?;
                let layout = self
                    .pipeline_layouts
                    .get(&layout_id)
                    .ok_or_else(|| anyhow!("Pipeline layout {} not found", layout_id))?;
                let c_entry = std::ffi::CString::new(entry_point)?;
                let stage_info = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(*sm)
                    .name(&c_entry);
                let create_info = vk::ComputePipelineCreateInfo::default()
                    .stage(stage_info)
                    .layout(*layout);
                // Safety: shader module, layout and entry point are valid compute stages.
                let pipelines = unsafe {
                    dev.create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
                        .map_err(|(_, err)| {
                            anyhow!("Failed to create compute pipeline: {:?}", err)
                        })?
                };
                self.pipelines.insert(pipeline_id, pipelines[0]);
                Ok(Response::Success)
            }
            Request::CreateDescriptorPool {
                pool_id,
                max_sets,
                pool_sizes,
            } => {
                let dev = self.device()?;
                let vk_sizes: Vec<vk::DescriptorPoolSize> = pool_sizes
                    .iter()
                    .map(|s| {
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::from_raw(s.descriptor_type as i32))
                            .descriptor_count(s.descriptor_count)
                    })
                    .collect();
                let create_info = vk::DescriptorPoolCreateInfo::default()
                    .max_sets(max_sets)
                    .pool_sizes(&vk_sizes)
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
                // Safety: descriptor pool parameters and types are valid.
                let pool = unsafe { dev.create_descriptor_pool(&create_info, None)? };
                self.descriptor_pools.insert(pool_id, pool);
                Ok(Response::Success)
            }
            Request::AllocateDescriptorSets {
                pool_id,
                layout_ids,
                set_ids,
            } => {
                let dev = self.device()?;
                let pool = self
                    .descriptor_pools
                    .get(&pool_id)
                    .ok_or_else(|| anyhow!("Descriptor pool {} not found", pool_id))?;
                let mut layouts = Vec::new();
                for lid in &layout_ids {
                    let l = self
                        .descriptor_set_layouts
                        .get(lid)
                        .ok_or_else(|| anyhow!("Descriptor set layout {} not found", lid))?;
                    layouts.push(*l);
                }
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(*pool)
                    .set_layouts(&layouts);
                // Safety: descriptor pool and layouts are valid.
                let sets = unsafe { dev.allocate_descriptor_sets(&alloc_info)? };
                for (sid, set) in set_ids.into_iter().zip(sets) {
                    self.descriptor_sets.insert(sid, set);
                }
                Ok(Response::Success)
            }
            Request::UpdateDescriptorSets { writes } => {
                let dev = self.device()?;
                tracing::info!("UpdateDescriptorSets: {} writes", writes.len());
                for w in &writes {
                    tracing::info!(
                        "  write: set={} binding={} type={} bufs={} imgs={}",
                        w.dst_set_id, w.dst_binding, w.descriptor_type, w.buffer_info.len(), w.image_info.len()
                    );
                }
                let mut all_buffer_infos: Vec<Vec<vk::DescriptorBufferInfo>> = Vec::new();
                let mut all_image_infos: Vec<Vec<vk::DescriptorImageInfo>> = Vec::new();
                for w in &writes {
                    let mut buf_infos = Vec::new();
                    for b in &w.buffer_info {
                        let buf = self
                            .buffers
                            .get(&b.buffer_id)
                            .ok_or_else(|| anyhow!("Buffer {} not found", b.buffer_id))?;
                        buf_infos.push(vk::DescriptorBufferInfo {
                            buffer: buf.vk_buffer,
                            offset: b.offset,
                            range: b.range,
                        });
                    }
                    all_buffer_infos.push(buf_infos);

                    let mut img_infos = Vec::new();
                    for img in &w.image_info {
                        let sampler = if img.sampler_id != 0 {
                            *self.samplers.get(&img.sampler_id).unwrap_or(&vk::Sampler::null())
                        } else {
                            vk::Sampler::null()
                        };
                        let image_view = if img.image_view_id != 0 {
                            *self.image_views.get(&img.image_view_id).unwrap_or(&vk::ImageView::null())
                        } else {
                            vk::ImageView::null()
                        };
                        let layout = vk::ImageLayout::from_raw(img.image_layout as i32);
                        img_infos.push(vk::DescriptorImageInfo {
                            sampler,
                            image_view,
                            image_layout: layout,
                        });
                    }
                    all_image_infos.push(img_infos);
                }

                let mut vk_writes = Vec::new();
                for (i, w) in writes.iter().enumerate() {
                    let set = self
                        .descriptor_sets
                        .get(&w.dst_set_id)
                        .ok_or_else(|| anyhow!("Descriptor set {} not found", w.dst_set_id))?;
                    let mut write = vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(w.dst_binding)
                        .dst_array_element(w.dst_array_element)
                        .descriptor_type(vk::DescriptorType::from_raw(w.descriptor_type as i32));
                    if !all_buffer_infos[i].is_empty() {
                        write = write.buffer_info(&all_buffer_infos[i]);
                    }
                    if !all_image_infos[i].is_empty() {
                        write = write.image_info(&all_image_infos[i]);
                    }
                    vk_writes.push(write);
                }
                // Safety: descriptor sets, buffer infos and image infos are valid and point to active resources.
                unsafe {
                    dev.update_descriptor_sets(&vk_writes, &[]);
                }
                Ok(Response::Success)
            }
            Request::CreateCommandPool { pool_id } => {
                let dev = self.device()?;
                let create_info = vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.engine.queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
                // Safety: queue family index and device handle are valid.
                let pool = unsafe { dev.create_command_pool(&create_info, None)? };
                self.command_pools.insert(pool_id, pool);
                Ok(Response::Success)
            }
            Request::AllocateCommandBuffers {
                pool_id,
                buffer_ids,
            } => {
                let dev = self.device()?;
                let pool = self
                    .command_pools
                    .get(&pool_id)
                    .ok_or_else(|| anyhow!("Command pool {} not found", pool_id))?;
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(*pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(buffer_ids.len() as u32);
                // Safety: command pool is valid and command_buffer_count matches slice.
                let bufs = unsafe { dev.allocate_command_buffers(&alloc_info)? };
                for (bid, b) in buffer_ids.into_iter().zip(bufs) {
                    self.command_buffers.insert(bid, b);
                }
                Ok(Response::Success)
            }
            Request::BeginCommandBuffer { buffer_id } => {
                tracing::info!("BeginCommandBuffer: cb={}", buffer_id);
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                // Safety: command buffer handle is valid and in initial state.
                unsafe {
                    dev.begin_command_buffer(*cb, &begin_info)?;
                }
                Ok(Response::Success)
            }
            Request::CmdBindPipeline {
                buffer_id,
                pipeline_bind_point,
                pipeline_id,
            } => {
                tracing::info!("CmdBindPipeline: cb={} pipe={}", buffer_id, pipeline_id);
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                let pipe = self
                    .pipelines
                    .get(&pipeline_id)
                    .ok_or_else(|| anyhow!("Pipeline {} not found", pipeline_id))?;
                let bind_point = match pipeline_bind_point {
                    PipelineBindPoint::Graphics => vk::PipelineBindPoint::GRAPHICS,
                    PipelineBindPoint::Compute => vk::PipelineBindPoint::COMPUTE,
                };
                unsafe {
                    dev.cmd_bind_pipeline(*cb, bind_point, *pipe);
                }
                Ok(Response::Success)
            }
            Request::CmdBindDescriptorSets {
                buffer_id,
                pipeline_bind_point,
                layout_id,
                first_set,
                set_ids,
                dynamic_offsets,
            } => {
                tracing::info!(
                    "CmdBindDescriptorSets: cb={} bind_point={:?} sets={:?} dyn_offsets={:?}",
                    buffer_id, pipeline_bind_point, set_ids, dynamic_offsets
                );
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                let layout = self
                    .pipeline_layouts
                    .get(&layout_id)
                    .ok_or_else(|| anyhow!("Pipeline layout {} not found", layout_id))?;
                let mut vk_sets = Vec::new();
                for sid in &set_ids {
                    let set = self
                        .descriptor_sets
                        .get(sid)
                        .ok_or_else(|| anyhow!("Descriptor set {} not found", sid))?;
                    vk_sets.push(*set);
                }
                let bind_point = match pipeline_bind_point {
                    PipelineBindPoint::Graphics => vk::PipelineBindPoint::GRAPHICS,
                    PipelineBindPoint::Compute => vk::PipelineBindPoint::COMPUTE,
                };
                unsafe {
                    dev.cmd_bind_descriptor_sets(
                        *cb,
                        bind_point,
                        *layout,
                        first_set,
                        &vk_sets,
                        &dynamic_offsets,
                    );
                }
                Ok(Response::Success)
            }
            Request::CmdPushConstants {
                buffer_id,
                layout_id,
                stage_flags,
                offset,
                values,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                let layout = self
                    .pipeline_layouts
                    .get(&layout_id)
                    .ok_or_else(|| anyhow!("Pipeline layout {} not found", layout_id))?;
                // Safety: push constant offset and size are within pipeline layout limits.
                unsafe {
                    dev.cmd_push_constants(
                        *cb,
                        *layout,
                        vk::ShaderStageFlags::from_raw(stage_flags),
                        offset,
                        &values,
                    );
                }
                Ok(Response::Success)
            }
            Request::CmdDispatch {
                buffer_id,
                group_count_x,
                group_count_y,
                group_count_z,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                // Safety: command buffer is recording and compute pipeline is bound.
                unsafe {
                    dev.cmd_dispatch(*cb, group_count_x, group_count_y, group_count_z);
                }
                Ok(Response::Success)
            }
            Request::CmdCopyBuffer {
                buffer_id,
                src_buffer_id,
                dst_buffer_id,
                regions,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                let src = self
                    .buffers
                    .get(&src_buffer_id)
                    .ok_or_else(|| anyhow!("Source buffer {} not found", src_buffer_id))?;
                let dst = self
                    .buffers
                    .get(&dst_buffer_id)
                    .ok_or_else(|| anyhow!("Destination buffer {} not found", dst_buffer_id))?;
                let vk_regions: Vec<vk::BufferCopy> = regions
                    .iter()
                    .map(|r| vk::BufferCopy {
                        src_offset: r.src_offset,
                        dst_offset: r.dst_offset,
                        size: r.size,
                    })
                    .collect();
                // Safety: source and destination buffers are valid and regions are within bounds.
                unsafe {
                    dev.cmd_copy_buffer(*cb, src.vk_buffer, dst.vk_buffer, &vk_regions);
                }
                Ok(Response::Success)
            }
            Request::CmdFillBuffer {
                buffer_id,
                dst_buffer_id,
                dst_offset,
                size,
                data,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                let dst = self
                    .buffers
                    .get(&dst_buffer_id)
                    .ok_or_else(|| anyhow!("Destination buffer {} not found", dst_buffer_id))?;
                // Safety: destination buffer is valid and offset/size are 4-byte aligned.
                unsafe {
                    dev.cmd_fill_buffer(*cb, dst.vk_buffer, dst_offset, size, data);
                }
                Ok(Response::Success)
            }
            Request::CmdPipelineBarrier {
                buffer_id,
                src_stage_mask,
                dst_stage_mask,
                dependency_flags,
                memory_barriers,
                buffer_memory_barriers,
                image_memory_barriers,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;

                let vk_memory_barriers: Vec<vk::MemoryBarrier> = memory_barriers
                    .iter()
                    .map(|b| {
                        vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::from_raw(b.src_access_mask))
                            .dst_access_mask(vk::AccessFlags::from_raw(b.dst_access_mask))
                    })
                    .collect();

                let mut vk_buffer_barriers: Vec<vk::BufferMemoryBarrier> = Vec::new();
                for b in &buffer_memory_barriers {
                    let buf = self
                        .buffers
                        .get(&b.buffer_id)
                        .ok_or_else(|| anyhow!("Buffer {} not found", b.buffer_id))?;
                    vk_buffer_barriers.push(
                        vk::BufferMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::from_raw(b.src_access_mask))
                            .dst_access_mask(vk::AccessFlags::from_raw(b.dst_access_mask))
                            .src_queue_family_index(b.src_queue_family_index)
                            .dst_queue_family_index(b.dst_queue_family_index)
                            .buffer(buf.vk_buffer)
                            .offset(b.offset)
                            .size(b.size),
                    );
                }

                let mut vk_image_barriers: Vec<vk::ImageMemoryBarrier> = Vec::new();
                let mut layout_updates = Vec::new();
                for b in &image_memory_barriers {
                    let img = self
                        .images
                        .get(&b.image_id)
                        .ok_or_else(|| anyhow!("Image {} not found", b.image_id))?;
                    let subresource = vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::from_raw(
                            b.subresource_range.aspect_mask.0,
                        ))
                        .base_mip_level(b.subresource_range.base_mip_level)
                        .level_count(b.subresource_range.level_count)
                        .base_array_layer(b.subresource_range.base_array_layer)
                        .layer_count(b.subresource_range.layer_count);
                    vk_image_barriers.push(
                        vk::ImageMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::from_raw(b.src_access_mask))
                            .dst_access_mask(vk::AccessFlags::from_raw(b.dst_access_mask))
                            .old_layout(to_vk_image_layout(b.old_layout))
                            .new_layout(to_vk_image_layout(b.new_layout))
                            .src_queue_family_index(b.src_queue_family_index)
                            .dst_queue_family_index(b.dst_queue_family_index)
                            .image(img.vk_image)
                            .subresource_range(subresource),
                    );
                    layout_updates.push((b.image_id, to_vk_image_layout(b.new_layout)));
                }

                // Sanitize pipeline stage masks: graphics command buffers cannot execute host operations (VUID-vkCmdPipelineBarrier-dstStageMask-04090)
                let sanitized_src = src_stage_mask & !vk::PipelineStageFlags::HOST.as_raw();
                let sanitized_dst = dst_stage_mask & !vk::PipelineStageFlags::HOST.as_raw();
                let mut src_stage = if sanitized_src == 0 {
                    vk::PipelineStageFlags::ALL_COMMANDS
                } else {
                    vk::PipelineStageFlags::from_raw(sanitized_src)
                };
                let mut dst_stage = if sanitized_dst == 0 {
                    vk::PipelineStageFlags::ALL_COMMANDS
                } else {
                    vk::PipelineStageFlags::from_raw(sanitized_dst)
                };

                // VUID-vkCmdPipelineBarrier-dstStageMask-01188:
                // dstStageMask must not be TOP_OF_PIPE if performing an image layout transition
                if !vk_image_barriers.is_empty() && dst_stage == vk::PipelineStageFlags::TOP_OF_PIPE {
                    dst_stage = vk::PipelineStageFlags::ALL_COMMANDS;
                }
                // VUID-vkCmdPipelineBarrier-srcAccessMask-02815:
                // If non-empty src_access_mask is used, srcStageMask cannot be TOP_OF_PIPE
                let has_src_access = vk_memory_barriers.iter().any(|b| !b.src_access_mask.is_empty())
                    || vk_buffer_barriers.iter().any(|b| !b.src_access_mask.is_empty())
                    || vk_image_barriers.iter().any(|b| !b.src_access_mask.is_empty());
                if has_src_access && src_stage == vk::PipelineStageFlags::TOP_OF_PIPE {
                    src_stage = vk::PipelineStageFlags::ALL_COMMANDS;
                }

                tracing::debug!(
                    "CmdPipelineBarrier: cb={} src_stage={:?} dst_stage={:?} mem_barriers={} buf_barriers={} img_barriers={}",
                    buffer_id, src_stage, dst_stage, vk_memory_barriers.len(), vk_buffer_barriers.len(), vk_image_barriers.len()
                );
                for b in &image_memory_barriers {
                    tracing::debug!(
                        "  img_barrier: image={} old_layout={:?} new_layout={:?} src_access=0x{:x} dst_access=0x{:x}",
                        b.image_id, b.old_layout, b.new_layout, b.src_access_mask, b.dst_access_mask
                    );
                }

                if vk_memory_barriers.is_empty()
                    && vk_buffer_barriers.is_empty()
                    && vk_image_barriers.is_empty()
                {
                    let default_barrier = vk::MemoryBarrier::default()
                        .src_access_mask(
                            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                                | vk::AccessFlags::SHADER_WRITE
                                | vk::AccessFlags::MEMORY_WRITE
                                | vk::AccessFlags::TRANSFER_WRITE,
                        )
                        .dst_access_mask(
                            vk::AccessFlags::COLOR_ATTACHMENT_READ
                                | vk::AccessFlags::SHADER_READ
                                | vk::AccessFlags::MEMORY_READ
                                | vk::AccessFlags::TRANSFER_READ,
                        );
                    unsafe {
                        dev.cmd_pipeline_barrier(
                            *cb,
                            src_stage,
                            dst_stage,
                            vk::DependencyFlags::from_raw(dependency_flags),
                            &[default_barrier],
                            &[],
                            &[],
                        );
                    }
                } else {
                    unsafe {
                        dev.cmd_pipeline_barrier(
                            *cb,
                            src_stage,
                            dst_stage,
                            vk::DependencyFlags::from_raw(dependency_flags),
                            &vk_memory_barriers,
                            &vk_buffer_barriers,
                            &vk_image_barriers,
                        );
                    }
                }
                for (image_id, new_layout) in layout_updates {
                    if let Some(image) = self.images.get_mut(&image_id) {
                        image.current_layout = new_layout;
                    }
                }
                Ok(Response::Success)
            }
            Request::EndCommandBuffer { buffer_id } => {
                tracing::info!("EndCommandBuffer: cb={}", buffer_id);
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                // Safety: command buffer is recording and has unmatched begin call.
                unsafe {
                    dev.end_command_buffer(*cb)?;
                }
                Ok(Response::Success)
            }
            Request::CreateFence { fence_id, signaled } => {
                let dev = self.device()?;
                let flags = if signaled {
                    vk::FenceCreateFlags::SIGNALED
                } else {
                    vk::FenceCreateFlags::empty()
                };
                let create_info = vk::FenceCreateInfo::default().flags(flags);
                // Safety: fence parameters are valid.
                let fence = unsafe { dev.create_fence(&create_info, None)? };
                self.fences.insert(fence_id, fence);
                Ok(Response::Success)
            }
            Request::QueueSubmit {
                queue_id,
                command_buffer_ids,
                fence_id,
                dirty_memory,
            } => {
                tracing::info!(
                    "QueueSubmit: queue={} cbs={:?} fence={:?} dirty_chunks={}",
                    queue_id, command_buffer_ids, fence_id, dirty_memory.len()
                );
                let dev = self.device()?;
                let queue = self.queue.ok_or_else(|| anyhow!("Queue not initialized"))?;

                // Apply incoming memory diffs
                for diff in dirty_memory {
                    let mem = self
                        .memories
                        .get(&diff.allocation_id)
                        .ok_or_else(|| anyhow!("Memory {} not found", diff.allocation_id))?;
                    let ptr = mem.mapped_ptr.ok_or_else(|| {
                        anyhow!("Memory {} is not host mapped", diff.allocation_id)
                    })?;

                    if diff.offset + diff.data.len() as u64 > mem.size {
                        return Err(anyhow!("Write range exceeds allocation bounds"));
                    }

                    // Safety: bounds checked above; pointer is valid and non-overlapping.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            diff.data.as_ptr(),
                            ptr.0.add(diff.offset as usize),
                            diff.data.len(),
                        );
                    }

                    if !mem.is_host_coherent {
                        let range = vk::MappedMemoryRange::default()
                            .memory(mem.vk_memory)
                            .offset(diff.offset)
                            .size(diff.data.len() as u64);
                        // Safety: memory range is mapped and bounds checked.
                        unsafe {
                            dev.flush_mapped_memory_ranges(&[range])?;
                        }
                    }
                }

                let mut vk_cbs = Vec::new();
                for cbid in &command_buffer_ids {
                    let cb = self
                        .command_buffers
                        .get(cbid)
                        .ok_or_else(|| anyhow!("Command buffer {} not found", cbid))?;
                    vk_cbs.push(*cb);
                }

                let fence = match fence_id {
                    Some(fid) => *self
                        .fences
                        .get(&fid)
                        .ok_or_else(|| anyhow!("Fence {} not found", fid))?,
                    None => vk::Fence::null(),
                };

                for (idx, &cb_handle) in vk_cbs.iter().enumerate() {
                    let sub_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb_handle));
                    let f = if idx + 1 == vk_cbs.len() { fence } else { vk::Fence::null() };
                    unsafe {
                        dev.queue_submit(queue, &[sub_info], f)?;
                        dev.queue_wait_idle(queue)?;
                    }
                    tracing::info!("Individual cb completed: id={} handle={:?}", command_buffer_ids[idx], cb_handle);
                }
                tracing::info!("QueueSubmit completed: cbs={:?}", command_buffer_ids);
                Ok(Response::Success)
            }
            Request::WaitForFences {
                fence_ids,
                wait_all,
                timeout_ns,
            } => {
                let dev = self.device()?;
                let mut vk_fences = Vec::new();
                for fid in &fence_ids {
                    let f = self
                        .fences
                        .get(fid)
                        .ok_or_else(|| anyhow!("Fence {} not found", fid))?;
                    vk_fences.push(*f);
                }
                // Safety: fences are valid handles created on this device.
                let res = unsafe { dev.wait_for_fences(&vk_fences, wait_all, timeout_ns) };
                match res {
                    // Fence completion establishes GPU ordering only. Sending all
                    // persistently mapped allocations here made Minecraft download
                    // roughly 134 MiB over the LAN per fence, starving the GPU.
                    Ok(()) => Ok(Response::Success),
                    Err(vk::Result::TIMEOUT) => Ok(Response::WaitResult {
                        success: false,
                        timeout: true,
                    }),
                    Err(e) => Ok(Response::Error {
                        code: e.as_raw(),
                        message: format!("{:?}", e),
                    }),
                }
            }
            Request::GetFenceStatus { fence_id } => {
                let dev = self.device()?;
                let fence = self
                    .fences
                    .get(&fence_id)
                    .ok_or_else(|| anyhow!("Fence {} not found", fence_id))?;
                // Safety: fence is a valid handle created on this device.
                let status = unsafe { dev.get_fence_status(*fence) };
                match status {
                    Ok(signaled) => Ok(Response::FenceStatus { signaled }),
                    Err(vk::Result::NOT_READY) => Ok(Response::FenceStatus { signaled: false }),
                    Err(vk::Result::SUCCESS) => Ok(Response::FenceStatus { signaled: true }),
                    Err(e) => Ok(Response::Error {
                        code: e.as_raw(),
                        message: format!("{:?}", e),
                    }),
                }
            }
            Request::ResetFences { fence_ids } => {
                let dev = self.device()?;
                let mut vk_fences = Vec::new();
                for fid in &fence_ids {
                    let f = self
                        .fences
                        .get(fid)
                        .ok_or_else(|| anyhow!("Fence {} not found", fid))?;
                    vk_fences.push(*f);
                }
                // Safety: fences are valid handles created on this device.
                unsafe {
                    dev.reset_fences(&vk_fences)?;
                }
                Ok(Response::Success)
            }
            Request::QueueWaitIdle { queue_id: _ } => {
                let dev = self.device()?;
                let queue = self.queue.ok_or_else(|| anyhow!("Queue not initialized"))?;
                // Safety: queue is a valid queue handle created with the device.
                unsafe {
                    dev.queue_wait_idle(queue)?;
                }
                Ok(Response::Success)
            }
            Request::DeviceWaitIdle { device_id: _ } => {
                let dev = self.device()?;
                // Safety: device is a valid device handle.
                unsafe {
                    dev.device_wait_idle()?;
                }
                Ok(Response::Success)
            }
            Request::DestroyResource {
                handle_id,
                resource_type,
            } => {
                if let Some(dev) = &self.device {
                    match resource_type {
                        ResourceType::Buffer => {
                            if let Some(buf) = self.buffers.remove(&handle_id) {
                                // Safety: buffer is no longer in use and device is valid.
                                unsafe {
                                    dev.destroy_buffer(buf.vk_buffer, None);
                                }
                            }
                        }
                        ResourceType::DeviceMemory => {
                            if let Some(mut mem) = self.memories.remove(&handle_id) {
                                if mem.mapped_ptr.is_some() {
                                    // Safety: memory is mapped and being freed.
                                    unsafe {
                                        dev.unmap_memory(mem.vk_memory);
                                    }
                                    mem.mapped_ptr = None;
                                }
                                // Safety: memory is unmapped and no longer bound to active operations.
                                unsafe {
                                    dev.free_memory(mem.vk_memory, None);
                                }
                            }
                        }
                        ResourceType::ShaderModule => {
                            if let Some(sm) = self.shader_modules.remove(&handle_id) {
                                // Safety: shader module is valid and being destroyed.
                                unsafe {
                                    dev.destroy_shader_module(sm, None);
                                }
                            }
                        }
                        ResourceType::DescriptorSetLayout => {
                            if let Some(l) = self.descriptor_set_layouts.remove(&handle_id) {
                                // Safety: descriptor set layout is valid.
                                unsafe {
                                    dev.destroy_descriptor_set_layout(l, None);
                                }
                            }
                        }
                        ResourceType::PipelineLayout => {
                            if let Some(l) = self.pipeline_layouts.remove(&handle_id) {
                                // Safety: pipeline layout is valid.
                                unsafe {
                                    dev.destroy_pipeline_layout(l, None);
                                }
                            }
                        }
                        ResourceType::Pipeline => {
                            if let Some(p) = self.pipelines.remove(&handle_id) {
                                // Safety: pipeline is valid.
                                unsafe {
                                    dev.destroy_pipeline(p, None);
                                }
                            }
                        }
                        ResourceType::DescriptorPool => {
                            if let Some(p) = self.descriptor_pools.remove(&handle_id) {
                                // Safety: descriptor pool and child sets are destroyed cleanly.
                                unsafe {
                                    dev.destroy_descriptor_pool(p, None);
                                }
                            }
                        }
                        ResourceType::DescriptorSet => {
                            self.descriptor_sets.remove(&handle_id);
                        }
                        ResourceType::CommandPool => {
                            if let Some(p) = self.command_pools.remove(&handle_id) {
                                // Safety: command pool and child buffers are destroyed cleanly.
                                unsafe {
                                    dev.destroy_command_pool(p, None);
                                }
                            }
                        }
                        ResourceType::CommandBuffer => {
                            self.command_buffers.remove(&handle_id);
                        }
                        ResourceType::Fence => {
                            if let Some(f) = self.fences.remove(&handle_id) {
                                // Safety: fence is valid and being destroyed.
                                unsafe {
                                    dev.destroy_fence(f, None);
                                }
                            }
                        }
                        ResourceType::Image => {
                            if let Some(img) = self.images.remove(&handle_id) {
                                if !img.is_swapchain_image {
                                    unsafe {
                                        dev.destroy_image(img.vk_image, None);
                                    }
                                }
                            }
                        }
                        ResourceType::ImageView => {
                            if let Some(v) = self.image_views.remove(&handle_id) {
                                unsafe {
                                    dev.destroy_image_view(v, None);
                                }
                            }
                        }
                        ResourceType::Sampler => {
                            if let Some(s) = self.samplers.remove(&handle_id) {
                                unsafe {
                                    dev.destroy_sampler(s, None);
                                }
                            }
                        }
                        ResourceType::RenderPass => {
                            if let Some(rp) = self.render_passes.remove(&handle_id) {
                                unsafe {
                                    dev.destroy_render_pass(rp, None);
                                }
                            }
                        }
                        ResourceType::Framebuffer => {
                            if let Some(fb) = self.framebuffers.remove(&handle_id) {
                                unsafe {
                                    dev.destroy_framebuffer(fb, None);
                                }
                            }
                        }
                        ResourceType::Swapchain => {
                            if let Some(sw) = self.swapchains.remove(&handle_id) {
                                unsafe {
                                    for (buf, mem, _, _) in sw.staging_buffers {
                                        dev.unmap_memory(mem);
                                        dev.destroy_buffer(buf, None);
                                        dev.free_memory(mem, None);
                                    }
                                    for (img_id, img, mem) in sw.images {
                                        self.images.remove(&img_id);
                                        dev.destroy_image(img, None);
                                        dev.free_memory(mem, None);
                                    }
                                }
                            }
                        }
                        ResourceType::Device => {
                            self.destroy_session_resources();
                        }
                        _ => {}
                    }
                }
                Ok(Response::Success)
            }
            Request::CreateImage { image_id, params } => {
                let dev = self.device()?;
                let fmt_u32: u32 = params.format.into();
                let create_info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::from_raw(params.image_type as i32))
                    .format(vk::Format::from_raw(fmt_u32 as i32))
                    .extent(vk::Extent3D {
                        width: params.extent.width,
                        height: params.extent.height,
                        depth: params.extent.depth,
                    })
                    .mip_levels(params.mip_levels)
                    .array_layers(params.array_layers)
                    .samples(vk::SampleCountFlags::from_raw(params.samples as u32))
                    .tiling(vk::ImageTiling::from_raw(params.tiling as i32))
                    .usage(vk::ImageUsageFlags::from_raw(params.usage.0))
                    .sharing_mode(vk::SharingMode::from_raw(params.sharing_mode as i32))
                    .initial_layout(to_vk_image_layout(params.initial_layout));
                tracing::info!(
                    "CreateImage: id={} fmt={:?} extent={}x{}x{} usage=0x{:x}",
                    image_id, create_info.format, create_info.extent.width, create_info.extent.height, create_info.extent.depth, params.usage.0
                );
                let vk_image = unsafe { dev.create_image(&create_info, None)? };
                self.images.insert(
                    image_id,
                    ImageState {
                        vk_image,
                        _format: vk::Format::from_raw(fmt_u32 as i32),
                        _extent: vk::Extent3D {
                            width: params.extent.width,
                            height: params.extent.height,
                            depth: params.extent.depth,
                        },
                        _mip_levels: params.mip_levels,
                        _array_layers: params.array_layers,
                        bound_memory: None,
                        is_swapchain_image: false,
                        current_layout: to_vk_image_layout(params.initial_layout),
                    },
                );
                Ok(Response::ImageCreated { id: image_id })
            }
            Request::DestroyImage { image_id } => {
                let dev = self.device()?.clone();
                if let Some(img) = self.images.remove(&image_id) {
                    if !img.is_swapchain_image {
                        unsafe {
                            dev.destroy_image(img.vk_image, None);
                        }
                    }
                }
                Ok(Response::Success)
            }
            Request::GetImageMemoryRequirements { image_id } => {
                let dev = self.device()?;
                let img = self
                    .images
                    .get(&image_id)
                    .ok_or_else(|| anyhow!("Image {} not found", image_id))?;
                let reqs = unsafe { dev.get_image_memory_requirements(img.vk_image) };
                Ok(Response::ImageMemoryRequirements(MemoryRequirementsData {
                    size: reqs.size,
                    alignment: reqs.alignment,
                    memory_type_bits: reqs.memory_type_bits,
                }))
            }
            Request::GetImageSubresourceLayout {
                image_id,
                aspect_mask,
                mip_level,
                array_layer,
            } => {
                let dev = self.device()?;
                let img = self
                    .images
                    .get(&image_id)
                    .ok_or_else(|| anyhow!("Image {} not found", image_id))?;
                let sub = vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::from_raw(aspect_mask))
                    .mip_level(mip_level)
                    .array_layer(array_layer);
                let l = unsafe { dev.get_image_subresource_layout(img.vk_image, sub) };
                Ok(Response::ImageSubresourceLayout {
                    offset: l.offset,
                    size: l.size,
                    row_pitch: l.row_pitch,
                    array_pitch: l.array_pitch,
                    depth_pitch: l.depth_pitch,
                })
            }
            Request::BindImageMemory {
                image_id,
                memory_id,
                offset,
            } => {
                tracing::info!("BindImageMemory: img={} mem={} offset={}", image_id, memory_id, offset);
                let dev = self.device()?.clone();
                let mem = self
                    .memories
                    .get(&memory_id)
                    .ok_or_else(|| anyhow!("Memory {} not found", memory_id))?;
                let mem_vk = mem.vk_memory;
                let img = self
                    .images
                    .get_mut(&image_id)
                    .ok_or_else(|| anyhow!("Image {} not found", image_id))?;
                unsafe {
                    dev.bind_image_memory(img.vk_image, mem_vk, offset)?;
                }
                img.bound_memory = Some((memory_id, offset));
                Ok(Response::Success)
            }
            Request::CreateImageView { view_id, params } => {
                let dev = self.device()?;
                let img = self
                    .images
                    .get(&params.image_id)
                    .ok_or_else(|| anyhow!("Image {} not found", params.image_id))?;
                let fmt_u32: u32 = params.format.into();
                tracing::info!("CreateImageView: view={} img={} fmt={:?}", view_id, params.image_id, fmt_u32);
                let subresource = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::from_raw(
                        params.subresource_range.aspect_mask.0,
                    ))
                    .base_mip_level(params.subresource_range.base_mip_level)
                    .level_count(params.subresource_range.level_count)
                    .base_array_layer(params.subresource_range.base_array_layer)
                    .layer_count(params.subresource_range.layer_count);
                let components = vk::ComponentMapping::default()
                    .r(vk::ComponentSwizzle::from_raw(params.components.r as i32))
                    .g(vk::ComponentSwizzle::from_raw(params.components.g as i32))
                    .b(vk::ComponentSwizzle::from_raw(params.components.b as i32))
                    .a(vk::ComponentSwizzle::from_raw(params.components.a as i32));
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(img.vk_image)
                    .view_type(vk::ImageViewType::from_raw(params.view_type as i32))
                    .format(vk::Format::from_raw(fmt_u32 as i32))
                    .components(components)
                    .subresource_range(subresource);
                let vk_view = unsafe { dev.create_image_view(&create_info, None)? };
                self.image_views.insert(view_id, vk_view);
                self.image_view_images.insert(view_id, params.image_id);
                Ok(Response::ImageViewCreated { id: view_id })
            }
            Request::DestroyImageView { view_id } => {
                let dev = self.device()?.clone();
                self.image_view_images.remove(&view_id);
                if let Some(view) = self.image_views.remove(&view_id) {
                    unsafe {
                        dev.destroy_image_view(view, None);
                    }
                }
                Ok(Response::Success)
            }
            Request::CreateSampler { sampler_id, params } => {
                let dev = self.device()?;
                let create_info = vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::from_raw(params.mag_filter as i32))
                    .min_filter(vk::Filter::from_raw(params.min_filter as i32))
                    .mipmap_mode(vk::SamplerMipmapMode::from_raw(params.mipmap_mode as i32))
                    .address_mode_u(vk::SamplerAddressMode::from_raw(
                        params.address_mode_u as i32,
                    ))
                    .address_mode_v(vk::SamplerAddressMode::from_raw(
                        params.address_mode_v as i32,
                    ))
                    .address_mode_w(vk::SamplerAddressMode::from_raw(
                        params.address_mode_w as i32,
                    ))
                    .mip_lod_bias(params.mip_lod_bias.to_f32())
                    .anisotropy_enable(params.anisotropy_enable)
                    .max_anisotropy(params.max_anisotropy.to_f32())
                    .compare_enable(params.compare_enable)
                    .compare_op(vk::CompareOp::from_raw(params.compare_op as i32))
                    .min_lod(params.min_lod.to_f32())
                    .max_lod(params.max_lod.to_f32())
                    .border_color(vk::BorderColor::from_raw(params.border_color as i32))
                    .unnormalized_coordinates(params.unnormalized_coordinates);
                let vk_sampler = unsafe { dev.create_sampler(&create_info, None)? };
                self.samplers.insert(sampler_id, vk_sampler);
                Ok(Response::SamplerCreated { id: sampler_id })
            }
            Request::DestroySampler { sampler_id } => {
                let dev = self.device()?.clone();
                if let Some(sampler) = self.samplers.remove(&sampler_id) {
                    unsafe {
                        dev.destroy_sampler(sampler, None);
                    }
                }
                Ok(Response::Success)
            }
            Request::CreateRenderPass {
                render_pass_id,
                params,
            } => {
                let dev = self.device()?;
                let vk_attachments: Vec<vk::AttachmentDescription> = params
                    .attachments
                    .iter()
                    .map(|a| {
                        let fmt_u32: u32 = a.format.into();
                        vk::AttachmentDescription::default()
                            .flags(vk::AttachmentDescriptionFlags::from_raw(a.flags))
                            .format(vk::Format::from_raw(fmt_u32 as i32))
                            .samples(vk::SampleCountFlags::from_raw(a.samples as u32))
                            .load_op(vk::AttachmentLoadOp::from_raw(a.load_op as i32))
                            .store_op(vk::AttachmentStoreOp::from_raw(a.store_op as i32))
                            .stencil_load_op(vk::AttachmentLoadOp::from_raw(
                                a.stencil_load_op as i32,
                            ))
                            .stencil_store_op(vk::AttachmentStoreOp::from_raw(
                                a.stencil_store_op as i32,
                            ))
                            .initial_layout(to_vk_image_layout(a.initial_layout))
                            .final_layout(to_vk_image_layout(a.final_layout))
                    })
                    .collect();

                let mut color_attachment_refs: Vec<Vec<vk::AttachmentReference>> = Vec::new();
                let mut depth_attachment_refs: Vec<Option<vk::AttachmentReference>> = Vec::new();
                let mut input_attachment_refs: Vec<Vec<vk::AttachmentReference>> = Vec::new();
                let mut resolve_attachment_refs: Vec<Vec<vk::AttachmentReference>> = Vec::new();

                for s in &params.subpasses {
                    let colors: Vec<vk::AttachmentReference> = s
                        .color_attachments
                        .iter()
                        .map(|c| {
                            vk::AttachmentReference::default()
                                .attachment(c.attachment)
                                .layout(to_vk_image_layout(c.layout))
                        })
                        .collect();
                    color_attachment_refs.push(colors);

                    let inputs: Vec<vk::AttachmentReference> = s
                        .input_attachments
                        .iter()
                        .map(|c| {
                            vk::AttachmentReference::default()
                                .attachment(c.attachment)
                                .layout(to_vk_image_layout(c.layout))
                        })
                        .collect();
                    input_attachment_refs.push(inputs);

                    let resolves: Vec<vk::AttachmentReference> = s
                        .resolve_attachments
                        .iter()
                        .map(|c| {
                            vk::AttachmentReference::default()
                                .attachment(c.attachment)
                                .layout(to_vk_image_layout(c.layout))
                        })
                        .collect();
                    resolve_attachment_refs.push(resolves);

                    let depth = s.depth_stencil_attachment.map(|d| {
                        vk::AttachmentReference::default()
                            .attachment(d.attachment)
                            .layout(to_vk_image_layout(d.layout))
                    });
                    depth_attachment_refs.push(depth);
                }

                let mut vk_subpasses: Vec<vk::SubpassDescription> = Vec::new();
                for i in 0..params.subpasses.len() {
                    let bind_point = match params.subpasses[i].pipeline_bind_point {
                        PipelineBindPoint::Compute => vk::PipelineBindPoint::COMPUTE,
                        _ => vk::PipelineBindPoint::GRAPHICS,
                    };
                    let mut desc = vk::SubpassDescription::default()
                        .flags(vk::SubpassDescriptionFlags::from_raw(
                            params.subpasses[i].flags,
                        ))
                        .pipeline_bind_point(bind_point)
                        .color_attachments(&color_attachment_refs[i]);
                    if !input_attachment_refs[i].is_empty() {
                        desc = desc.input_attachments(&input_attachment_refs[i]);
                    }
                    if !resolve_attachment_refs[i].is_empty() {
                        desc = desc.resolve_attachments(&resolve_attachment_refs[i]);
                    }
                    if let Some(ref d) = depth_attachment_refs[i] {
                        desc = desc.depth_stencil_attachment(d);
                    }
                    if !params.subpasses[i].preserve_attachments.is_empty() {
                        desc = desc.preserve_attachments(&params.subpasses[i].preserve_attachments);
                    }
                    vk_subpasses.push(desc);
                }

                let vk_dependencies: Vec<vk::SubpassDependency> = params
                    .dependencies
                    .iter()
                    .map(|d| {
                        vk::SubpassDependency::default()
                            .src_subpass(d.src_subpass)
                            .dst_subpass(d.dst_subpass)
                            .src_stage_mask(vk::PipelineStageFlags::from_raw(d.src_stage_mask))
                            .dst_stage_mask(vk::PipelineStageFlags::from_raw(d.dst_stage_mask))
                            .src_access_mask(vk::AccessFlags::from_raw(d.src_access_mask))
                            .dst_access_mask(vk::AccessFlags::from_raw(d.dst_access_mask))
                            .dependency_flags(vk::DependencyFlags::from_raw(d.dependency_flags))
                    })
                    .collect();

                let create_info = vk::RenderPassCreateInfo::default()
                    .attachments(&vk_attachments)
                    .subpasses(&vk_subpasses)
                    .dependencies(&vk_dependencies);
                let vk_rp = unsafe { dev.create_render_pass(&create_info, None)? };
                self.render_passes.insert(render_pass_id, vk_rp);
                let final_layouts: Vec<vk::ImageLayout> = params
                    .attachments
                    .iter()
                    .map(|a| to_vk_image_layout(a.final_layout))
                    .collect();
                self.render_pass_final_layouts.insert(render_pass_id, final_layouts);
                Ok(Response::RenderPassCreated { id: render_pass_id })
            }
            Request::DestroyRenderPass { render_pass_id } => {
                let dev = self.device()?.clone();
                self.render_pass_final_layouts.remove(&render_pass_id);
                if let Some(rp) = self.render_passes.remove(&render_pass_id) {
                    unsafe {
                        dev.destroy_render_pass(rp, None);
                    }
                }
                Ok(Response::Success)
            }
            Request::CreateFramebuffer {
                framebuffer_id,
                params,
            } => {
                let dev = self.device()?;
                let rp = self
                    .render_passes
                    .get(&params.render_pass_id)
                    .ok_or_else(|| anyhow!("RenderPass {} not found", params.render_pass_id))?;
                let mut views = Vec::new();
                for aid in &params.attachments {
                    let v = self
                        .image_views
                        .get(aid)
                        .ok_or_else(|| anyhow!("ImageView {} not found", aid))?;
                    views.push(*v);
                }
                let create_info = vk::FramebufferCreateInfo::default()
                    .render_pass(*rp)
                    .attachments(&views)
                    .width(params.width)
                    .height(params.height)
                    .layers(params.layers);
                let vk_fb = unsafe { dev.create_framebuffer(&create_info, None)? };
                self.framebuffers.insert(framebuffer_id, vk_fb);
                let image_ids: Vec<u64> = params
                    .attachments
                    .iter()
                    .filter_map(|aid| self.image_view_images.get(aid).copied())
                    .collect();
                tracing::info!(
                    "CreateFramebuffer: fb={} views={:?} image_ids={:?}",
                    framebuffer_id, params.attachments, image_ids
                );
                self.framebuffer_attachments.insert(framebuffer_id, image_ids);
                Ok(Response::FramebufferCreated { id: framebuffer_id })
            }
            Request::DestroyFramebuffer { framebuffer_id } => {
                let dev = self.device()?.clone();
                self.framebuffer_attachments.remove(&framebuffer_id);
                if let Some(fb) = self.framebuffers.remove(&framebuffer_id) {
                    unsafe {
                        dev.destroy_framebuffer(fb, None);
                    }
                }
                Ok(Response::Success)
            }
            Request::CreateGraphicsPipelines { pipelines } => {
                let dev = self.device()?.clone();
                let mut created_ids = Vec::new();
                for p in pipelines {
                    let mut c_entry_points = Vec::new();
                    for s in &p.stages {
                        let c_entry = std::ffi::CString::new(s.entry_point.as_str())?;
                        c_entry_points.push(c_entry);
                    }
                    let mut vk_stages = Vec::new();
                    for (i, s) in p.stages.iter().enumerate() {
                        let sm = self
                            .shader_modules
                            .get(&s.module_id)
                            .ok_or_else(|| anyhow!("ShaderModule {} not found", s.module_id))?;
                        let c_ptr = c_entry_points[i].as_c_str();
                        let stage_info = vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::from_raw(s.stage.0))
                            .module(*sm)
                            .name(c_ptr);
                        vk_stages.push(stage_info);
                    }

                    let vk_bindings: Vec<vk::VertexInputBindingDescription> = p
                        .vertex_input_state
                        .bindings
                        .iter()
                        .map(|b| {
                            vk::VertexInputBindingDescription::default()
                                .binding(b.binding)
                                .stride(b.stride)
                                .input_rate(vk::VertexInputRate::from_raw(b.input_rate as i32))
                        })
                        .collect();

                    let vk_attrs: Vec<vk::VertexInputAttributeDescription> = p
                        .vertex_input_state
                        .attributes
                        .iter()
                        .map(|a| {
                            let fmt_u32: u32 = a.format.into();
                            vk::VertexInputAttributeDescription::default()
                                .location(a.location)
                                .binding(a.binding)
                                .format(vk::Format::from_raw(fmt_u32 as i32))
                                .offset(a.offset)
                        })
                        .collect();

                    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
                        .vertex_binding_descriptions(&vk_bindings)
                        .vertex_attribute_descriptions(&vk_attrs);

                    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                        .topology(vk::PrimitiveTopology::from_raw(
                            p.input_assembly_state.topology as i32,
                        ))
                        .primitive_restart_enable(p.input_assembly_state.primitive_restart_enable);

                    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
                    let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default()
                        .dynamic_states(&dynamic_states);

                    let dummy_viewport = [vk::Viewport::default()
                        .width(1.0)
                        .height(1.0)
                        .max_depth(1.0)];
                    let dummy_scissor = [vk::Rect2D::default().extent(vk::Extent2D {
                        width: 1,
                        height: 1,
                    })];
                    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                        .viewports(&dummy_viewport)
                        .scissors(&dummy_scissor);

                    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
                        .depth_clamp_enable(p.rasterization_state.depth_clamp_enable)
                        .rasterizer_discard_enable(p.rasterization_state.rasterizer_discard_enable)
                        .polygon_mode(vk::PolygonMode::from_raw(
                            p.rasterization_state.polygon_mode as i32,
                        ))
                        .line_width(p.rasterization_state.line_width.to_f32().max(1.0))
                        .cull_mode(vk::CullModeFlags::from_raw(
                            p.rasterization_state.cull_mode.0,
                        ))
                        .front_face(vk::FrontFace::from_raw(
                            p.rasterization_state.front_face as i32,
                        ))
                        .depth_bias_enable(p.rasterization_state.depth_bias_enable);

                    let multisampling = if let Some(ref ms) = p.multisample_state {
                        vk::PipelineMultisampleStateCreateInfo::default()
                            .rasterization_samples(vk::SampleCountFlags::from_raw(ms.rasterization_samples as u32))
                            .sample_shading_enable(ms.sample_shading_enable)
                            .min_sample_shading(ms.min_sample_shading.to_f32())
                            .alpha_to_coverage_enable(ms.alpha_to_coverage_enable)
                            .alpha_to_one_enable(ms.alpha_to_one_enable)
                    } else {
                        vk::PipelineMultisampleStateCreateInfo::default()
                            .sample_shading_enable(false)
                            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                    };

                    let depth_stencil = p.depth_stencil_state.as_ref().map(|ds| {
                        vk::PipelineDepthStencilStateCreateInfo::default()
                            .depth_test_enable(ds.depth_test_enable)
                            .depth_write_enable(ds.depth_write_enable)
                            .depth_compare_op(vk::CompareOp::from_raw(ds.depth_compare_op as i32))
                            .depth_bounds_test_enable(ds.depth_bounds_test_enable)
                            .stencil_test_enable(ds.stencil_test_enable)
                            .min_depth_bounds(ds.min_depth_bounds.to_f32())
                            .max_depth_bounds(ds.max_depth_bounds.to_f32())
                            .front(vk::StencilOpState {
                                fail_op: vk::StencilOp::from_raw(ds.front.fail_op as i32),
                                pass_op: vk::StencilOp::from_raw(ds.front.pass_op as i32),
                                depth_fail_op: vk::StencilOp::from_raw(ds.front.depth_fail_op as i32),
                                compare_op: vk::CompareOp::from_raw(ds.front.compare_op as i32),
                                compare_mask: ds.front.compare_mask,
                                write_mask: ds.front.write_mask,
                                reference: ds.front.reference,
                            })
                            .back(vk::StencilOpState {
                                fail_op: vk::StencilOp::from_raw(ds.back.fail_op as i32),
                                pass_op: vk::StencilOp::from_raw(ds.back.pass_op as i32),
                                depth_fail_op: vk::StencilOp::from_raw(ds.back.depth_fail_op as i32),
                                compare_op: vk::CompareOp::from_raw(ds.back.compare_op as i32),
                                compare_mask: ds.back.compare_mask,
                                write_mask: ds.back.write_mask,
                                reference: ds.back.reference,
                            })
                    });

                    let color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> =
                        if let Some(ref cb_state) = p.color_blend_state {
                            cb_state
                                .attachments
                                .iter()
                                .map(|a| {
                                    vk::PipelineColorBlendAttachmentState::default()
                                        .blend_enable(a.blend_enable)
                                        .src_color_blend_factor(vk::BlendFactor::from_raw(
                                             a.src_color_blend_factor as i32,
                                        ))
                                        .dst_color_blend_factor(vk::BlendFactor::from_raw(
                                            a.dst_color_blend_factor as i32,
                                        ))
                                        .color_blend_op(vk::BlendOp::from_raw(
                                            a.color_blend_op as i32,
                                        ))
                                        .src_alpha_blend_factor(vk::BlendFactor::from_raw(
                                            a.src_alpha_blend_factor as i32,
                                        ))
                                        .dst_alpha_blend_factor(vk::BlendFactor::from_raw(
                                            a.dst_alpha_blend_factor as i32,
                                        ))
                                        .alpha_blend_op(vk::BlendOp::from_raw(
                                            a.alpha_blend_op as i32,
                                        ))
                                        .color_write_mask(vk::ColorComponentFlags::from_raw(
                                            a.color_write_mask.0,
                                        ))
                                })
                                .collect()
                        } else {
                            vec![vk::PipelineColorBlendAttachmentState::default()
                                .color_write_mask(vk::ColorComponentFlags::RGBA)
                                .blend_enable(false)]
                        };

                    let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
                        .attachments(&color_blend_attachments);

                    let layout = self
                        .pipeline_layouts
                        .get(&p.layout_id)
                        .ok_or_else(|| anyhow!("PipelineLayout {} not found", p.layout_id))?;
                    let rp = self
                        .render_passes
                        .get(&p.render_pass_id)
                        .ok_or_else(|| anyhow!("RenderPass {} not found", p.render_pass_id))?;

                    let mut pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                        .stages(&vk_stages)
                        .vertex_input_state(&vertex_input_info)
                        .input_assembly_state(&input_assembly)
                        .viewport_state(&viewport_state)
                        .rasterization_state(&rasterizer)
                        .multisample_state(&multisampling)
                        .color_blend_state(&color_blending)
                        .dynamic_state(&dynamic_state_info)
                        .layout(*layout)
                        .render_pass(*rp)
                        .subpass(p.subpass);

                    let default_ds = vk::PipelineDepthStencilStateCreateInfo::default()
                        .depth_test_enable(false)
                        .depth_write_enable(false)
                        .depth_compare_op(vk::CompareOp::ALWAYS);
                    let ds_ref = depth_stencil.as_ref().unwrap_or(&default_ds);
                    pipeline_info = pipeline_info.depth_stencil_state(ds_ref);

                    let created = unsafe {
                        dev.create_graphics_pipelines(
                            vk::PipelineCache::null(),
                            &[pipeline_info],
                            None,
                        )
                        .map_err(|(_, err)| {
                            anyhow!("Failed to create graphics pipeline: {:?}", err)
                        })?
                    };
                    self.pipelines.insert(p.pipeline_id, created[0]);
                    created_ids.push(p.pipeline_id);
                }
                Ok(Response::GraphicsPipelinesCreated { ids: created_ids })
            }
            Request::CmdBeginRenderPass {
                buffer_id,
                render_pass_id,
                framebuffer_id,
                render_area,
                clear_values,
                contents,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                let rp = self
                    .render_passes
                    .get(&render_pass_id)
                    .ok_or_else(|| anyhow!("RenderPass {} not found", render_pass_id))?;
                let fb = self
                    .framebuffers
                    .get(&framebuffer_id)
                    .ok_or_else(|| anyhow!("Framebuffer {} not found", framebuffer_id))?;
                let vk_clears: Vec<vk::ClearValue> = clear_values
                    .iter()
                    .map(|c| match c {
                        ClearValueData::Color(col) => match col {
                            ClearColorValue::Float(f) => vk::ClearValue {
                                color: vk::ClearColorValue {
                                    float32: [
                                        f[0].to_f32(),
                                        f[1].to_f32(),
                                        f[2].to_f32(),
                                        f[3].to_f32(),
                                    ],
                                },
                            },
                            ClearColorValue::Int(i) => vk::ClearValue {
                                color: vk::ClearColorValue { int32: *i },
                            },
                            ClearColorValue::Uint(u) => vk::ClearValue {
                                color: vk::ClearColorValue { uint32: *u },
                            },
                        },
                        ClearValueData::DepthStencil(ds) => vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: ds.depth.to_f32(),
                                stencil: ds.stencil,
                            },
                        },
                        ClearValueData::ColorFloat(f) => vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [
                                    f32::from_bits(f[0]),
                                    f32::from_bits(f[1]),
                                    f32::from_bits(f[2]),
                                    f32::from_bits(f[3]),
                                ],
                            },
                        },
                        ClearValueData::ColorInt(i) => vk::ClearValue {
                            color: vk::ClearColorValue { int32: *i },
                        },
                        ClearValueData::ColorUint(u) => vk::ClearValue {
                            color: vk::ClearColorValue { uint32: *u },
                        },
                    })
                    .collect();

                let render_pass_info = vk::RenderPassBeginInfo::default()
                    .render_pass(*rp)
                    .framebuffer(*fb)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D {
                            x: render_area.offset.x,
                            y: render_area.offset.y,
                        },
                        extent: vk::Extent2D {
                            width: render_area.extent.width,
                            height: render_area.extent.height,
                        },
                    })
                    .clear_values(&vk_clears);

                tracing::info!(
                    "CmdBeginRenderPass: cb={} rp={} fb={}",
                    buffer_id, render_pass_id, framebuffer_id
                );
                unsafe {
                    dev.cmd_begin_render_pass(
                        *cb,
                        &render_pass_info,
                        vk::SubpassContents::from_raw(contents as i32),
                    );
                }
                self.cmd_buffer_render_passes
                    .insert(buffer_id, (render_pass_id, framebuffer_id));
                Ok(Response::Success)
            }
            Request::CmdEndRenderPass { buffer_id } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                unsafe {
                    dev.cmd_end_render_pass(*cb);
                }
                if let Some((rp_id, fb_id)) = self.cmd_buffer_render_passes.remove(&buffer_id) {
                    if let (Some(final_layouts), Some(image_ids)) = (
                        self.render_pass_final_layouts.get(&rp_id).cloned(),
                        self.framebuffer_attachments.get(&fb_id).cloned(),
                    ) {
                        tracing::info!(
                            "CmdEndRenderPass: cb={} rp={} fb={} updating image_ids={:?} to final_layouts={:?}",
                            buffer_id, rp_id, fb_id, image_ids, final_layouts
                        );
                        for (&img_id, &final_layout) in image_ids.iter().zip(final_layouts.iter()) {
                            if let Some(image) = self.images.get_mut(&img_id) {
                                image.current_layout = final_layout;
                            }
                        }
                    }
                }
                Ok(Response::Success)
            }
            Request::CmdBindVertexBuffers {
                buffer_id,
                first_binding,
                buffer_ids,
                offsets,
            } => {
                tracing::info!("CmdBindVertexBuffers: cb={} bufs={:?} offsets={:?}", buffer_id, buffer_ids, offsets);
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                let mut vk_bufs = Vec::new();
                for b in &buffer_ids {
                    let buf = self
                        .buffers
                        .get(b)
                        .ok_or_else(|| anyhow!("Buffer {} not found", b))?;
                    vk_bufs.push(buf.vk_buffer);
                }
                unsafe {
                    dev.cmd_bind_vertex_buffers(*cb, first_binding, &vk_bufs, &offsets);
                }
                Ok(Response::Success)
            }
            Request::CmdBindIndexBuffer {
                buffer_id,
                index_buffer_id,
                offset,
                index_type,
            } => {
                tracing::info!("CmdBindIndexBuffer: cb={} index_buf={} offset={}", buffer_id, index_buffer_id, offset);
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                let buf = self
                    .buffers
                    .get(&index_buffer_id)
                    .ok_or_else(|| anyhow!("Buffer {} not found", index_buffer_id))?;
                unsafe {
                    dev.cmd_bind_index_buffer(
                        *cb,
                        buf.vk_buffer,
                        offset,
                        vk::IndexType::from_raw(index_type as i32),
                    );
                }
                Ok(Response::Success)
            }
            Request::CmdSetViewport {
                buffer_id,
                first_viewport,
                viewports,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                let vk_viewports: Vec<vk::Viewport> = viewports
                    .iter()
                    .map(|v| vk::Viewport {
                        x: v.x.to_f32(),
                        y: v.y.to_f32(),
                        width: v.width.to_f32(),
                        height: v.height.to_f32(),
                        min_depth: v.min_depth.to_f32(),
                        max_depth: v.max_depth.to_f32(),
                    })
                    .collect();
                unsafe {
                    dev.cmd_set_viewport(*cb, first_viewport, &vk_viewports);
                }
                Ok(Response::Success)
            }
            Request::CmdSetScissor {
                buffer_id,
                first_scissor,
                scissors,
            } => {
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                let vk_scissors: Vec<vk::Rect2D> = scissors
                    .iter()
                    .map(|s| vk::Rect2D {
                        offset: vk::Offset2D {
                            x: s.offset.x,
                            y: s.offset.y,
                        },
                        extent: vk::Extent2D {
                            width: s.extent.width,
                            height: s.extent.height,
                        },
                    })
                    .collect();
                unsafe {
                    dev.cmd_set_scissor(*cb, first_scissor, &vk_scissors);
                }
                Ok(Response::Success)
            }
            Request::CmdDraw {
                buffer_id,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => {
                tracing::info!("CmdDraw: cb={} vertices={}", buffer_id, vertex_count);
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                unsafe {
                    dev.cmd_draw(
                        *cb,
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    );
                }
                Ok(Response::Success)
            }
            Request::CmdDrawIndexed {
                buffer_id,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            } => {
                tracing::info!("CmdDrawIndexed: cb={} index_count={}", buffer_id, index_count);
                let dev = self.device()?;
                let cb = self
                    .command_buffers
                    .get(&buffer_id)
                    .ok_or_else(|| anyhow!("Command buffer {} not found", buffer_id))?;
                unsafe {
                    dev.cmd_draw_indexed(
                        *cb,
                        index_count,
                        instance_count,
                        first_index,
                        vertex_offset,
                        first_instance,
                    );
                }
                Ok(Response::Success)
            }
            Request::GetSurfaceCapabilities { surface_id: _ } => {
                Ok(Response::SurfaceCapabilities(SurfaceCapabilities {
                    min_image_count: 1,
                    max_image_count: 8,
                    current_extent: Extent2D {
                        width: 1280,
                        height: 720,
                    },
                    min_image_extent: Extent2D {
                        width: 1,
                        height: 1,
                    },
                    max_image_extent: Extent2D {
                        width: 7680,
                        height: 4320,
                    },
                    max_image_array_layers: 1,
                    supported_transforms: SurfaceTransformFlags(SurfaceTransformFlags::IDENTITY),
                    current_transform: SurfaceTransformFlags(SurfaceTransformFlags::IDENTITY),
                    supported_composite_alpha: CompositeAlphaFlags(CompositeAlphaFlags::OPAQUE),
                    supported_usage_flags: ImageUsageFlags(
                        ImageUsageFlags::COLOR_ATTACHMENT
                            | ImageUsageFlags::TRANSFER_SRC
                            | ImageUsageFlags::TRANSFER_DST
                            | ImageUsageFlags::SAMPLED,
                    ),
                }))
            }
            Request::GetSurfaceFormats { surface_id: _ } => Ok(Response::SurfaceFormats(vec![
                SurfaceFormat {
                    format: Format::B8G8R8A8Unorm,
                    color_space: ColorSpace::SrgbNonlinear,
                },
                SurfaceFormat {
                    format: Format::R8G8B8A8Unorm,
                    color_space: ColorSpace::SrgbNonlinear,
                },
            ])),
            Request::GetSurfacePresentModes { surface_id: _ } => {
                Ok(Response::SurfacePresentModes(vec![0, 1, 2]))
            }
            Request::CreateSwapchain {
                swapchain_id,
                surface_id,
                params,
            } => {
                let dev = self.device()?.clone();
                let mut swapchain_images = Vec::new();
                let mut staging_buffers = Vec::new();

                let fmt_u32: u32 = params.image_format.into();
                let width = params.image_extent.width;
                let height = params.image_extent.height;
                let byte_size = (width * height * 4) as u64;

                for i in 0..params.min_image_count {
                    let img_create_info = vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::from_raw(fmt_u32 as i32))
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(params.image_array_layers.max(1))
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST
                                | vk::ImageUsageFlags::SAMPLED,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED);

                    let vk_image = unsafe { dev.create_image(&img_create_info, None)? };
                    let reqs = unsafe { dev.get_image_memory_requirements(vk_image) };
                    let mem_type_index = self
                        .find_memory_type_index(
                            reqs.memory_type_bits,
                            vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        )
                        .or_else(|| {
                            self.find_memory_type_index(
                                reqs.memory_type_bits,
                                vk::MemoryPropertyFlags::empty(),
                            )
                        })
                        .ok_or_else(|| anyhow!("No suitable memory type for swapchain image"))?;

                    let alloc_info = vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size)
                        .memory_type_index(mem_type_index);
                    let vk_memory = unsafe { dev.allocate_memory(&alloc_info, None)? };
                    unsafe {
                        dev.bind_image_memory(vk_image, vk_memory, 0)?;
                    }

                    let img_id = swapchain_id.wrapping_mul(1000) + i as u64 + 1;
                    self.images.insert(
                        img_id,
                        ImageState {
                            vk_image,
                            _format: vk::Format::from_raw(fmt_u32 as i32),
                            _extent: vk::Extent3D {
                                width,
                                height,
                                depth: 1,
                            },
                            _mip_levels: 1,
                            _array_layers: params.image_array_layers.max(1),
                            bound_memory: None,
                            is_swapchain_image: true,
                            current_layout: vk::ImageLayout::UNDEFINED,
                        },
                    );
                    swapchain_images.push((img_id, vk_image, vk_memory));

                    // Create staging readback buffer
                    let buf_create_info = vk::BufferCreateInfo::default()
                        .size(byte_size)
                        .usage(vk::BufferUsageFlags::TRANSFER_DST)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE);
                    let staging_buf = unsafe { dev.create_buffer(&buf_create_info, None)? };
                    let buf_reqs = unsafe { dev.get_buffer_memory_requirements(staging_buf) };
                    // The CPU reads every presented frame back from this buffer, so prefer
                    // HOST_CACHED: uncached/write-combined memory makes CPU reads ~10x slower.
                    let buf_mem_type_index = self
                        .find_memory_type_index(
                            buf_reqs.memory_type_bits,
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_CACHED,
                        )
                        .or_else(|| {
                            self.find_memory_type_index(
                                buf_reqs.memory_type_bits,
                                vk::MemoryPropertyFlags::HOST_VISIBLE
                                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                            )
                        })
                        .or_else(|| {
                            self.find_memory_type_index(
                                buf_reqs.memory_type_bits,
                                vk::MemoryPropertyFlags::HOST_VISIBLE,
                            )
                        })
                        .ok_or_else(|| anyhow!("No host-visible memory type for staging buffer"))?;

                    let buf_alloc_info = vk::MemoryAllocateInfo::default()
                        .allocation_size(buf_reqs.size)
                        .memory_type_index(buf_mem_type_index);
                    let staging_mem = unsafe { dev.allocate_memory(&buf_alloc_info, None)? };
                    unsafe {
                        dev.bind_buffer_memory(staging_buf, staging_mem, 0)?;
                    }
                    let ptr = unsafe {
                        dev.map_memory(staging_mem, 0, buf_reqs.size, vk::MemoryMapFlags::empty())?
                    };
                    staging_buffers.push((
                        staging_buf,
                        staging_mem,
                        SendPtr(ptr as *mut u8),
                        byte_size,
                    ));
                }

                let image_ids: Vec<u64> = swapchain_images.iter().map(|(id, _, _)| *id).collect();
                self.swapchains.insert(
                    swapchain_id,
                    SwapchainState {
                        _surface_id: surface_id,
                        width,
                        height,
                        format: params.image_format,
                        images: swapchain_images,
                        staging_buffers,
                        current_index: 0,
                    },
                );
                Ok(Response::SwapchainCreated {
                    id: swapchain_id,
                    image_ids,
                })
            }
            Request::DestroySwapchain { swapchain_id } => {
                let dev = self.device()?.clone();
                if let Some(sw) = self.swapchains.remove(&swapchain_id) {
                    unsafe {
                        for (buf, mem, _, _) in sw.staging_buffers {
                            dev.unmap_memory(mem);
                            dev.destroy_buffer(buf, None);
                            dev.free_memory(mem, None);
                        }
                        for (img_id, img, mem) in sw.images {
                            self.images.remove(&img_id);
                            dev.destroy_image(img, None);
                            dev.free_memory(mem, None);
                        }
                    }
                }
                Ok(Response::Success)
            }
            Request::GetSwapchainImages { swapchain_id } => {
                let sw = self
                    .swapchains
                    .get(&swapchain_id)
                    .ok_or_else(|| anyhow!("Swapchain {} not found", swapchain_id))?;
                let image_ids: Vec<u64> = sw.images.iter().map(|(id, _, _)| *id).collect();
                Ok(Response::SwapchainImages(image_ids))
            }
            Request::AcquireNextImage {
                swapchain_id,
                timeout_ns: _,
                semaphore_id: _,
                fence_id: _,
            } => {
                let sw = self
                    .swapchains
                    .get_mut(&swapchain_id)
                    .ok_or_else(|| anyhow!("Swapchain {} not found", swapchain_id))?;
                let idx = sw.current_index;
                sw.current_index = (sw.current_index + 1) % sw.images.len().max(1) as u32;
                Ok(Response::AcquireNextImage {
                    image_index: idx,
                    suboptimal: false,
                })
            }
            Request::QueuePresent {
                queue_id: _,
                params,
            } => {
                let dev = self.device()?.clone();
                let queue = self.queue.ok_or_else(|| anyhow!("Queue not initialized"))?;
                let internal_cmd_pool = self
                    .internal_cmd_pool
                    .ok_or_else(|| anyhow!("Internal command pool not initialized"))?;

                let sw = self
                    .swapchains
                    .get(&params.swapchain_id)
                    .ok_or_else(|| anyhow!("Swapchain {} not found", params.swapchain_id))?;

                let img_idx = params.image_index as usize;
                let (image_id, vk_img, _) = sw.images[img_idx % sw.images.len()];
                let (staging_buf, staging_mem, ptr, byte_size) =
                    sw.staging_buffers[img_idx % sw.staging_buffers.len()];
                let width = sw.width;
                let height = sw.height;
                let format = sw.format;
                let source_layout = self
                    .images
                    .get(&image_id)
                    .map(|image| image.current_layout)
                    // Render passes can perform an implicit attachment-layout
                    // transition, which does not travel as CmdPipelineBarrier.
                    // For swapchain presentation, the required layout is PRESENT_SRC_KHR.
                    .filter(|layout| *layout != vk::ImageLayout::UNDEFINED)
                    .unwrap_or(vk::ImageLayout::PRESENT_SRC_KHR);

                let raw_current_layout = self.images.get(&image_id).map(|img| img.current_layout);
                tracing::info!(
                    "QueuePresent: sw={} img_idx={} image_id={} raw_layout={:?} source_layout={:?}",
                    params.swapchain_id, img_idx, image_id, raw_current_layout, source_layout
                );

                // 1. Wait for render completion
                unsafe {
                    dev.queue_wait_idle(queue)?;
                }

                // 2. Allocate command buffer to copy image to staging buffer
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(internal_cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let cbs = unsafe { dev.allocate_command_buffers(&alloc_info)? };
                let cb = cbs[0];

                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                unsafe {
                    dev.begin_command_buffer(cb, &begin_info)?;

                    let subresource_range = vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1);

                    let barrier_to_src = vk::ImageMemoryBarrier::default()
                        .old_layout(source_layout)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_access_mask(
                            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                                | vk::AccessFlags::MEMORY_WRITE
                                | vk::AccessFlags::MEMORY_READ,
                        )
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .image(vk_img)
                        .subresource_range(subresource_range);

                    dev.cmd_pipeline_barrier(
                        cb,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier_to_src],
                    );

                    let region = vk::BufferImageCopy::default()
                        .buffer_offset(0)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(0)
                                .base_array_layer(0)
                                .layer_count(1),
                        )
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        });

                    dev.cmd_copy_image_to_buffer(
                        cb,
                        vk_img,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        staging_buf,
                        &[region],
                    );

                    let barrier_back = vk::ImageMemoryBarrier::default()
                        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .new_layout(source_layout)
                        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .dst_access_mask(
                            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                                | vk::AccessFlags::MEMORY_WRITE
                                | vk::AccessFlags::MEMORY_READ,
                        )
                        .image(vk_img)
                        .subresource_range(subresource_range);

                    let buffer_barrier = vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(staging_buf)
                        .offset(0)
                        .size(vk::WHOLE_SIZE);

                    dev.cmd_pipeline_barrier(
                        cb,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[buffer_barrier],
                        &[barrier_back],
                    );

                    dev.end_command_buffer(cb)?;

                    let cbs_slice = [cb];
                    let submit_info = vk::SubmitInfo::default().command_buffers(&cbs_slice);
                    dev.queue_submit(queue, &[submit_info], vk::Fence::null())?;
                    dev.queue_wait_idle(queue)?;
                    dev.free_command_buffers(internal_cmd_pool, &[cb]);
                }

                // Invalidate mapped memory range if not coherent
                let range = vk::MappedMemoryRange::default()
                    .memory(staging_mem)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                if let Err(e) = unsafe { dev.invalidate_mapped_memory_ranges(&[range]) } {
                    tracing::warn!("invalidate_mapped_memory_ranges warning: {:?}", e);
                }

                let data =
                    unsafe { std::slice::from_raw_parts(ptr.0, byte_size as usize).to_vec() };
                static PRESENT_LOG_COUNT: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let p_idx = PRESENT_LOG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if p_idx % 60 == 0 {
                    let non_zero = data.iter().filter(|&&b| b != 0).count();
                    tracing::info!(
                        "QueuePresent frame #{}: {}x{}, {} non-zero bytes / {} ({:.1}%)",
                        p_idx,
                        width,
                        height,
                        non_zero,
                        data.len(),
                        (non_zero as f64 / data.len().max(1) as f64) * 100.0
                    );
                }

                Ok(Response::QueuePresentResult {
                    result: 0,
                    frame: Some(FramePacket {
                        swapchain_id: params.swapchain_id,
                        image_index: params.image_index,
                        width,
                        height,
                        format,
                        data,
                    }),
                })
            }
        }
    }

    fn destroy_session_resources(&mut self) {
        if let Some(dev) = self.device.take() {
            // Safety: wait for device to be idle before freeing all allocated session resources.
            unsafe {
                let _ = dev.device_wait_idle();

                for (_, sw) in self.swapchains.drain() {
                    for (buf, mem, _, _) in sw.staging_buffers {
                        dev.unmap_memory(mem);
                        dev.destroy_buffer(buf, None);
                        dev.free_memory(mem, None);
                    }
                    for (img_id, img, mem) in sw.images {
                        self.images.remove(&img_id);
                        dev.destroy_image(img, None);
                        dev.free_memory(mem, None);
                    }
                }
                for (_, fb) in self.framebuffers.drain() {
                    dev.destroy_framebuffer(fb, None);
                }
                for (_, rp) in self.render_passes.drain() {
                    dev.destroy_render_pass(rp, None);
                }
                for (_, s) in self.samplers.drain() {
                    dev.destroy_sampler(s, None);
                }
                for (_, v) in self.image_views.drain() {
                    dev.destroy_image_view(v, None);
                }
                for (_, img) in self.images.drain() {
                    if !img.is_swapchain_image {
                        dev.destroy_image(img.vk_image, None);
                    }
                }
                if let Some(pool) = self.internal_cmd_pool.take() {
                    dev.destroy_command_pool(pool, None);
                }

                for (_, f) in self.fences.drain() {
                    dev.destroy_fence(f, None);
                }
                for (_, p) in self.pipelines.drain() {
                    dev.destroy_pipeline(p, None);
                }
                for (_, l) in self.pipeline_layouts.drain() {
                    dev.destroy_pipeline_layout(l, None);
                }
                for (_, l) in self.descriptor_set_layouts.drain() {
                    dev.destroy_descriptor_set_layout(l, None);
                }
                for (_, p) in self.descriptor_pools.drain() {
                    dev.destroy_descriptor_pool(p, None);
                }
                for (_, p) in self.command_pools.drain() {
                    dev.destroy_command_pool(p, None);
                }
                for (_, sm) in self.shader_modules.drain() {
                    dev.destroy_shader_module(sm, None);
                }
                for (_, buf) in self.buffers.drain() {
                    dev.destroy_buffer(buf.vk_buffer, None);
                }
                for (_, mut mem) in self.memories.drain() {
                    if mem.mapped_ptr.is_some() {
                        dev.unmap_memory(mem.vk_memory);
                        mem.mapped_ptr = None;
                    }
                    dev.free_memory(mem.vk_memory, None);
                }

                dev.destroy_device(None);
            }
        }
    }
}

impl Drop for VulkanSession {
    fn drop(&mut self) {
        self.destroy_session_resources();
    }
}

struct MockSwapchain {
    width: u32,
    height: u32,
    format: Format,
    images: Vec<u64>,
    current_index: u32,
}

pub struct MockSession {
    engine: Arc<MockEngine>,
    buffers: HashMap<u64, (u64, u32)>,
    memories: HashMap<u64, Vec<u8>>,
    _buffer_bindings: HashMap<u64, (u64, u64)>,
    images: HashSet<u64>,
    image_views: HashSet<u64>,
    samplers: HashSet<u64>,
    render_passes: HashSet<u64>,
    framebuffers: HashSet<u64>,
    swapchains: HashMap<u64, MockSwapchain>,
    shader_modules: HashSet<u64>,
    descriptor_set_layouts: HashSet<u64>,
    pipeline_layouts: HashSet<u64>,
    pipelines: HashSet<u64>,
    descriptor_pools: HashSet<u64>,
    descriptor_sets: HashSet<u64>,
    command_pools: HashSet<u64>,
    command_buffers: HashSet<u64>,
    fences: HashMap<u64, bool>,
}

impl MockSession {
    pub fn new(engine: Arc<MockEngine>) -> Self {
        Self {
            engine,
            buffers: HashMap::new(),
            memories: HashMap::new(),
            _buffer_bindings: HashMap::new(),
            images: HashSet::new(),
            image_views: HashSet::new(),
            samplers: HashSet::new(),
            render_passes: HashSet::new(),
            framebuffers: HashSet::new(),
            swapchains: HashMap::new(),
            shader_modules: HashSet::new(),
            descriptor_set_layouts: HashSet::new(),
            pipeline_layouts: HashSet::new(),
            pipelines: HashSet::new(),
            descriptor_pools: HashSet::new(),
            descriptor_sets: HashSet::new(),
            command_pools: HashSet::new(),
            command_buffers: HashSet::new(),
            fences: HashMap::new(),
        }
    }

    pub fn handle_request(&mut self, req: Request) -> Response {
        match req {
            Request::CommandBatch { commands } => {
                if commands
                    .iter()
                    .any(|command| matches!(command, Request::CommandBatch { .. }))
                {
                    Response::Error {
                        code: -1,
                        message: "Nested command batches are not supported".to_string(),
                    }
                } else {
                    Response::Success
                }
            }
            Request::GetDeviceSnapshot => Response::DeviceSnapshot(self.engine.snapshot.clone()),
            Request::CreateDevice { .. } => Response::Success,
            Request::CreateBuffer {
                buffer_id,
                size,
                usage,
            } => {
                self.buffers.insert(buffer_id, (size, usage));
                Response::Success
            }
            Request::AllocateMemory {
                memory_id, size, ..
            } => {
                self.memories.insert(memory_id, vec![0u8; size as usize]);
                Response::Success
            }
            Request::BindBufferMemory {
                buffer_id,
                memory_id,
                offset,
            } => {
                self._buffer_bindings.insert(buffer_id, (memory_id, offset));
                Response::Success
            }
            Request::WriteMemoryRanges { diffs } => {
                for diff in diffs {
                    if let Some(mem) = self.memories.get_mut(&diff.allocation_id) {
                        let start = diff.offset as usize;
                        let end = start + diff.data.len();
                        if end <= mem.len() {
                            mem[start..end].copy_from_slice(&diff.data);
                        }
                    }
                }
                Response::Success
            }
            Request::ReadMemoryRanges { allocation_ids } => {
                let mut diffs = Vec::new();
                for alloc_id in allocation_ids {
                    if let Some(mem) = self.memories.get(&alloc_id) {
                        for (chunk_idx, chunk) in mem.chunks(CHUNK_SIZE).enumerate() {
                            diffs.push(MemoryChunkDiff {
                                allocation_id: alloc_id,
                                offset: (chunk_idx * CHUNK_SIZE) as u64,
                                data: chunk.to_vec(),
                            });
                        }
                    }
                }
                Response::MemoryData { diffs }
            }
            Request::CreateShaderModule { shader_id, .. } => {
                self.shader_modules.insert(shader_id);
                Response::Success
            }
            Request::CreateDescriptorSetLayout { layout_id, .. } => {
                self.descriptor_set_layouts.insert(layout_id);
                Response::Success
            }
            Request::CreatePipelineLayout { layout_id, .. } => {
                self.pipeline_layouts.insert(layout_id);
                Response::Success
            }
            Request::CreateComputePipeline { pipeline_id, .. } => {
                self.pipelines.insert(pipeline_id);
                Response::Success
            }
            Request::CreateDescriptorPool { pool_id, .. } => {
                self.descriptor_pools.insert(pool_id);
                Response::Success
            }
            Request::AllocateDescriptorSets { set_ids, .. } => {
                for sid in set_ids {
                    self.descriptor_sets.insert(sid);
                }
                Response::Success
            }
            Request::UpdateDescriptorSets { .. } => Response::Success,
            Request::CreateCommandPool { pool_id } => {
                self.command_pools.insert(pool_id);
                Response::Success
            }
            Request::AllocateCommandBuffers { buffer_ids, .. } => {
                for bid in buffer_ids {
                    self.command_buffers.insert(bid);
                }
                Response::Success
            }
            Request::BeginCommandBuffer { .. } => Response::Success,
            Request::CmdBindPipeline { .. } => Response::Success,
            Request::CmdBindDescriptorSets { .. } => Response::Success,
            Request::CmdPushConstants { .. } => Response::Success,
            Request::CmdDispatch { .. } => Response::Success,
            Request::CmdCopyBuffer { .. } => Response::Success,
            Request::CmdFillBuffer {
                dst_buffer_id: _,
                dst_offset: _,
                size: _,
                data: _,
                ..
            } => Response::Success,
            Request::CmdPipelineBarrier { .. } => Response::Success,
            Request::EndCommandBuffer { .. } => Response::Success,
            Request::CreateFence { fence_id, signaled } => {
                self.fences.insert(fence_id, signaled);
                Response::Success
            }
            Request::QueueSubmit {
                fence_id,
                dirty_memory,
                ..
            } => {
                for diff in dirty_memory {
                    if let Some(mem) = self.memories.get_mut(&diff.allocation_id) {
                        let start = diff.offset as usize;
                        let end = start + diff.data.len();
                        if end <= mem.len() {
                            mem[start..end].copy_from_slice(&diff.data);
                        }
                    }
                }
                if let Some(fid) = fence_id {
                    self.fences.insert(fid, true);
                }
                Response::Success
            }
            Request::WaitForFences { .. } => Response::WaitResult {
                success: true,
                timeout: false,
            },
            Request::GetFenceStatus { fence_id } => {
                let signaled = self.fences.get(&fence_id).copied().unwrap_or(true);
                Response::FenceStatus { signaled }
            }
            Request::ResetFences { fence_ids } => {
                for fid in fence_ids {
                    self.fences.insert(fid, false);
                }
                Response::Success
            }
            Request::QueueWaitIdle { .. } => Response::Success,
            Request::DeviceWaitIdle { .. } => Response::Success,
            Request::DestroyResource {
                handle_id,
                resource_type,
            } => {
                match resource_type {
                    ResourceType::Buffer => {
                        self.buffers.remove(&handle_id);
                    }
                    ResourceType::DeviceMemory => {
                        self.memories.remove(&handle_id);
                    }
                    ResourceType::ShaderModule => {
                        self.shader_modules.remove(&handle_id);
                    }
                    ResourceType::DescriptorSetLayout => {
                        self.descriptor_set_layouts.remove(&handle_id);
                    }
                    ResourceType::PipelineLayout => {
                        self.pipeline_layouts.remove(&handle_id);
                    }
                    ResourceType::Pipeline => {
                        self.pipelines.remove(&handle_id);
                    }
                    ResourceType::DescriptorPool => {
                        self.descriptor_pools.remove(&handle_id);
                    }
                    ResourceType::DescriptorSet => {
                        self.descriptor_sets.remove(&handle_id);
                    }
                    ResourceType::CommandPool => {
                        self.command_pools.remove(&handle_id);
                    }
                    ResourceType::CommandBuffer => {
                        self.command_buffers.remove(&handle_id);
                    }
                    ResourceType::Fence => {
                        self.fences.remove(&handle_id);
                    }
                    ResourceType::Image => {
                        self.images.remove(&handle_id);
                    }
                    ResourceType::ImageView => {
                        self.image_views.remove(&handle_id);
                    }
                    ResourceType::Sampler => {
                        self.samplers.remove(&handle_id);
                    }
                    ResourceType::RenderPass => {
                        self.render_passes.remove(&handle_id);
                    }
                    ResourceType::Framebuffer => {
                        self.framebuffers.remove(&handle_id);
                    }
                    ResourceType::Swapchain => {
                        self.swapchains.remove(&handle_id);
                    }
                    _ => {}
                }
                Response::Success
            }
            Request::CreateImage { image_id, .. } => {
                self.images.insert(image_id);
                Response::Success
            }
            Request::DestroyImage { image_id } => {
                self.images.remove(&image_id);
                Response::Success
            }
            Request::GetImageMemoryRequirements { .. } => Response::MemoryRequirements {
                size: 65536,
                alignment: 256,
                memory_type_bits: 0x3,
            },
            Request::BindImageMemory { .. } => Response::Success,
            Request::GetImageSubresourceLayout { .. } => Response::ImageSubresourceLayout {
                offset: 0,
                size: 65536,
                row_pitch: 256,
                array_pitch: 65536,
                depth_pitch: 65536,
            },
            Request::CreateImageView { view_id, .. } => {
                self.image_views.insert(view_id);
                Response::Success
            }
            Request::DestroyImageView { view_id } => {
                self.image_views.remove(&view_id);
                Response::Success
            }
            Request::CreateSampler { sampler_id, .. } => {
                self.samplers.insert(sampler_id);
                Response::Success
            }
            Request::DestroySampler { sampler_id } => {
                self.samplers.remove(&sampler_id);
                Response::Success
            }
            Request::CreateRenderPass { render_pass_id, .. } => {
                self.render_passes.insert(render_pass_id);
                Response::Success
            }
            Request::DestroyRenderPass { render_pass_id } => {
                self.render_passes.remove(&render_pass_id);
                Response::Success
            }
            Request::CreateFramebuffer { framebuffer_id, .. } => {
                self.framebuffers.insert(framebuffer_id);
                Response::Success
            }
            Request::DestroyFramebuffer { framebuffer_id } => {
                self.framebuffers.remove(&framebuffer_id);
                Response::Success
            }
            Request::CreateGraphicsPipelines { pipelines } => {
                for p in pipelines {
                    self.pipelines.insert(p.pipeline_id);
                }
                Response::Success
            }
            Request::CmdBeginRenderPass { .. }
            | Request::CmdCopyBufferToImage { .. }
            | Request::CmdCopyImageToBuffer { .. }
            | Request::CmdCopyImage { .. }
            | Request::CmdBlitImage { .. }
            | Request::CmdResolveImage { .. }
            | Request::CmdDrawIndirect { .. }
            | Request::CmdDrawIndirectCount { .. }
            | Request::CmdDispatchIndirect { .. }
            | Request::CmdUpdateBuffer { .. }
            | Request::CmdClearColorImage { .. }
            | Request::CmdClearDepthStencilImage { .. }
            | Request::CmdEndRenderPass { .. }
            | Request::CmdBindVertexBuffers { .. }
            | Request::CmdBindIndexBuffer { .. }
            | Request::CmdSetViewport { .. }
            | Request::CmdSetScissor { .. }
            | Request::CmdDraw { .. }
            | Request::CmdDrawIndexed { .. } => Response::Success,
            Request::GetBufferMemoryRequirements { buffer_id } => {
                match self.buffers.get(&buffer_id) {
                    Some((size, _)) => Response::ImageMemoryRequirements(MemoryRequirementsData {
                        size: *size,
                        alignment: 64,
                        memory_type_bits: u32::MAX,
                    }),
                    None => Response::Error {
                        code: -1,
                        message: format!("Buffer {buffer_id} not found"),
                    },
                }
            }
            Request::GetSurfaceCapabilities { .. } => {
                Response::SurfaceCapabilities(SurfaceCapabilities {
                    min_image_count: 1,
                    max_image_count: 8,
                    current_extent: Extent2D {
                        width: 1280,
                        height: 720,
                    },
                    min_image_extent: Extent2D {
                        width: 1,
                        height: 1,
                    },
                    max_image_extent: Extent2D {
                        width: 7680,
                        height: 4320,
                    },
                    max_image_array_layers: 1,
                    supported_transforms: SurfaceTransformFlags(SurfaceTransformFlags::IDENTITY),
                    current_transform: SurfaceTransformFlags(SurfaceTransformFlags::IDENTITY),
                    supported_composite_alpha: CompositeAlphaFlags(CompositeAlphaFlags::OPAQUE),
                    supported_usage_flags: ImageUsageFlags(
                        ImageUsageFlags::COLOR_ATTACHMENT
                            | ImageUsageFlags::TRANSFER_SRC
                            | ImageUsageFlags::TRANSFER_DST
                            | ImageUsageFlags::SAMPLED,
                    ),
                })
            }
            Request::GetSurfaceFormats { .. } => Response::SurfaceFormats(vec![
                SurfaceFormat {
                    format: Format::B8G8R8A8Unorm,
                    color_space: ColorSpace::SrgbNonlinear,
                },
                SurfaceFormat {
                    format: Format::R8G8B8A8Unorm,
                    color_space: ColorSpace::SrgbNonlinear,
                },
            ]),
            Request::GetSurfacePresentModes { .. } => Response::SurfacePresentModes(vec![
                u32::from(PresentMode::Fifo),
                u32::from(PresentMode::Mailbox),
                u32::from(PresentMode::Immediate),
            ]),
            Request::CreateSwapchain {
                swapchain_id,
                params,
                ..
            } => {
                let mut images = Vec::new();
                for i in 0..params.min_image_count {
                    images.push(swapchain_id.wrapping_mul(1000) + i as u64 + 1);
                }
                self.swapchains.insert(
                    swapchain_id,
                    MockSwapchain {
                        width: params.image_extent.width,
                        height: params.image_extent.height,
                        format: params.image_format,
                        images: images.clone(),
                        current_index: 0,
                    },
                );
                Response::SwapchainCreated {
                    id: swapchain_id,
                    image_ids: images,
                }
            }
            Request::DestroySwapchain { swapchain_id } => {
                self.swapchains.remove(&swapchain_id);
                Response::Success
            }
            Request::GetSwapchainImages { swapchain_id } => {
                if let Some(sw) = self.swapchains.get(&swapchain_id) {
                    Response::SwapchainImages(sw.images.clone())
                } else {
                    Response::SwapchainImages(vec![])
                }
            }
            Request::AcquireNextImage { swapchain_id, .. } => {
                if let Some(sw) = self.swapchains.get_mut(&swapchain_id) {
                    let idx = sw.current_index;
                    sw.current_index = (sw.current_index + 1) % sw.images.len().max(1) as u32;
                    Response::AcquireNextImageResult {
                        image_index: idx,
                        suboptimal: false,
                    }
                } else {
                    Response::AcquireNextImageResult {
                        image_index: 0,
                        suboptimal: false,
                    }
                }
            }
            Request::QueuePresent { params, .. } => {
                let (w, h, fmt) = if let Some(sw) = self.swapchains.get(&params.swapchain_id) {
                    (sw.width, sw.height, sw.format)
                } else {
                    (64, 64, Format::B8G8R8A8Unorm)
                };
                let len = (w * h * 4) as usize;
                Response::QueuePresentResult {
                    result: 0,
                    frame: Some(FramePacket {
                        swapchain_id: params.swapchain_id,
                        image_index: params.image_index,
                        width: w,
                        height: h,
                        format: fmt,
                        data: vec![255; len],
                    }),
                }
            }
        }
    }
}

pub enum SessionHandler {
    Vulkan(VulkanSession),
    Mock(MockSession),
}

impl SessionHandler {
    pub fn new(engine: &Engine) -> Self {
        match engine {
            Engine::Vulkan(v) => SessionHandler::Vulkan(VulkanSession::new(v.clone())),
            Engine::Mock(m) => SessionHandler::Mock(MockSession::new(m.clone())),
        }
    }

    pub fn handle_request(&mut self, req: Request) -> Response {
        match self {
            SessionHandler::Vulkan(v) => v.handle_request(req),
            SessionHandler::Mock(m) => m.handle_request(req),
        }
    }
}
