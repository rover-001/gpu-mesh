#[cfg(test)]
mod tests {
    use tokio::io::duplex;
    use vkremote_wire::*;

    #[tokio::test]
    async fn test_length_prefixed_framing() {
        let (mut client, mut server) = duplex(1024);

        let req = Request::CreateBuffer {
            buffer_id: 42,
            size: 1024 * 1024,
            usage: 0x00000020,
        };

        // Write from client to server
        write_msg(&mut client, &req).await.unwrap();

        // Read at server
        let received: Request = read_msg(&mut server, DEFAULT_MAX_FRAME_SIZE).await.unwrap();

        assert_eq!(req, received);
    }

    #[tokio::test]
    async fn test_command_batch_framing() {
        let (mut client, mut server) = duplex(1024);
        let req = Request::CommandBatch {
            commands: vec![Request::CmdDispatch {
                buffer_id: 7,
                group_count_x: 1,
                group_count_y: 2,
                group_count_z: 3,
            }],
        };

        write_msg(&mut client, &req).await.unwrap();
        let received: Request = read_msg(&mut server, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
        assert_eq!(req, received);
    }

    #[tokio::test]
    async fn test_max_frame_size_limit() {
        let (mut client, mut server) = duplex(1024);

        let req = Request::GetDeviceSnapshot;
        write_msg(&mut client, &req).await.unwrap();

        // Reading with max_frame_size = 0 should fail with FrameTooLarge
        let err = read_msg::<_, Request>(&mut server, 0).await.unwrap_err();
        match err {
            ProtocolError::FrameTooLarge { size, max } => {
                assert!(size > 0);
                assert_eq!(max, 0);
            }
            other => panic!("Unexpected error: {:?}", other),
        }
    }

    #[test]
    fn test_default_frame_size_accepts_minecraft_memory_readback() {
        // Horizon requested allocations 38 and 30, whose combined `MemoryData`
        // response was 134,234,047 bytes.  Keep the default above that real
        // workload so a successful GPU session is never reset by the framing layer.
        const MINECRAFT_MEMORY_READBACK_BYTES: u32 = 134_234_047;
        assert!(DEFAULT_MAX_FRAME_SIZE >= MINECRAFT_MEMORY_READBACK_BYTES);
    }

    #[tokio::test]
    async fn test_handshake_flow_success() {
        let (mut client, mut server) = duplex(1024);

        let client_task =
            tokio::spawn(async move { client_handshake(&mut client, "secret-token").await });

        let server_task = tokio::spawn(async move {
            server_handshake(&mut server, |req| {
                if req.token == "secret-token" {
                    (true, "Welcome".to_string())
                } else {
                    (false, "Invalid token".to_string())
                }
            })
            .await
        });

        let (client_res, server_res) = tokio::join!(client_task, server_task);
        let server_handshake = client_res.unwrap().unwrap();
        let client_handshake = server_res.unwrap().unwrap();

        assert_eq!(server_handshake.accepted, true);
        assert_eq!(server_handshake.message, "Welcome");
        assert_eq!(client_handshake.token, "secret-token");
        assert_eq!(client_handshake.version, CURRENT_PROTOCOL_VERSION);
    }

    #[tokio::test]
    async fn test_handshake_flow_rejected() {
        let (mut client, mut server) = duplex(1024);

        let client_task =
            tokio::spawn(async move { client_handshake(&mut client, "wrong-token").await });

        let server_task = tokio::spawn(async move {
            server_handshake(&mut server, |req| {
                if req.token == "secret-token" {
                    (true, "Welcome".to_string())
                } else {
                    (false, "Invalid token".to_string())
                }
            })
            .await
        });

        let (client_res, server_res) = tokio::join!(client_task, server_task);
        assert!(client_res.unwrap().is_err());
        assert!(server_res.unwrap().is_err());
    }

    #[tokio::test]
    async fn test_handshake_flow_version_mismatch() {
        use vkremote_wire::framing::{read_msg, write_msg, ClientHandshake, MAGIC};

        let (mut client, mut server) = duplex(1024);

        let client_task = tokio::spawn(async move {
            // Emulate older v1 client
            tokio::io::AsyncWriteExt::write_all(&mut client, MAGIC).await.unwrap();
            tokio::io::AsyncWriteExt::flush(&mut client).await.unwrap();
            let mut s_magic = [0u8; 8];
            tokio::io::AsyncReadExt::read_exact(&mut client, &mut s_magic).await.unwrap();
            assert_eq!(&s_magic, MAGIC);

            let old_req = ClientHandshake {
                version: 1, // older version
                token: "secret-token".to_string(),
            };
            write_msg(&mut client, &old_req).await.unwrap();

            let resp: vkremote_wire::framing::ServerHandshake =
                read_msg(&mut client, vkremote_wire::framing::DEFAULT_MAX_FRAME_SIZE)
                    .await
                    .unwrap();
            resp
        });

        let server_task = tokio::spawn(async move {
            server_handshake(&mut server, |_| (true, "Welcome".to_string())).await
        });

        let (client_res, server_res) = tokio::join!(client_task, server_task);
        let resp = client_res.unwrap();
        assert!(!resp.accepted);
        assert!(resp.message.contains("Version mismatch"));
        let server_err = server_res.unwrap();
        assert!(matches!(
            server_err,
            Err(vkremote_wire::framing::ProtocolError::VersionMismatch {
                client_version: 1,
                server_version: CURRENT_PROTOCOL_VERSION,
            })
        ));
    }

    #[test]
    fn test_postcard_all_request_response_variants() {
        let requests = vec![
            Request::GetDeviceSnapshot,
            Request::CreateDevice {
                device_id: 1,
                queue_id: 2,
            },
            Request::CreateBuffer {
                buffer_id: 3,
                size: 4096,
                usage: 1,
            },
            Request::AllocateMemory {
                memory_id: 4,
                size: 8192,
                memory_type_index: 0,
            },
            Request::BindBufferMemory {
                buffer_id: 3,
                memory_id: 4,
                offset: 0,
            },
            Request::WriteMemoryRanges {
                diffs: vec![MemoryChunkDiff {
                    allocation_id: 4,
                    offset: 0,
                    data: vec![1, 2, 3, 4],
                }],
            },
            Request::ReadMemoryRanges {
                allocation_ids: vec![4],
            },
            Request::CreateShaderModule {
                shader_id: 5,
                spirv: vec![0x03, 0x02, 0x23, 0x07],
            },
            Request::CreateDescriptorSetLayout {
                layout_id: 6,
                bindings: vec![DescriptorSetLayoutBindingData {
                    binding: 0,
                    descriptor_type: 7,
                    descriptor_count: 1,
                    stage_flags: 32,
                }],
            },
            Request::CreatePipelineLayout {
                layout_id: 7,
                set_layout_ids: vec![6],
                push_constant_ranges: vec![PushConstantRangeData {
                    stage_flags: 32,
                    offset: 0,
                    size: 16,
                }],
            },
            Request::CreateComputePipeline {
                pipeline_id: 8,
                layout_id: 7,
                shader_id: 5,
                entry_point: "main".to_string(),
            },
            Request::CreateDescriptorPool {
                pool_id: 9,
                max_sets: 10,
                pool_sizes: vec![DescriptorPoolSizeData {
                    descriptor_type: 7,
                    descriptor_count: 10,
                }],
            },
            Request::AllocateDescriptorSets {
                pool_id: 9,
                layout_ids: vec![6],
                set_ids: vec![10],
            },
            Request::UpdateDescriptorSets {
                writes: vec![WriteDescriptorSetData {
                    dst_set_id: 10,
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_type: 7,
                    buffer_info: vec![DescriptorBufferInfoData {
                        buffer_id: 3,
                        offset: 0,
                        range: 4096,
                    }],
                    image_info: Vec::new(),
                }],
            },
            Request::CreateCommandPool { pool_id: 11 },
            Request::AllocateCommandBuffers {
                pool_id: 11,
                buffer_ids: vec![12],
            },
            Request::BeginCommandBuffer { buffer_id: 12 },
            Request::CmdBindPipeline {
                buffer_id: 12,
                pipeline_bind_point: PipelineBindPoint::Compute,
                pipeline_id: 8,
            },
            Request::CmdBindDescriptorSets {
                buffer_id: 12,
                pipeline_bind_point: PipelineBindPoint::Compute,
                layout_id: 7,
                first_set: 0,
                set_ids: vec![10],
                dynamic_offsets: vec![],
            },
            Request::CmdPushConstants {
                buffer_id: 12,
                layout_id: 7,
                stage_flags: 32,
                offset: 0,
                values: vec![1, 2, 3, 4],
            },
            Request::CmdDispatch {
                buffer_id: 12,
                group_count_x: 1,
                group_count_y: 1,
                group_count_z: 1,
            },
            Request::CmdCopyBuffer {
                buffer_id: 12,
                src_buffer_id: 3,
                dst_buffer_id: 13,
                regions: vec![BufferCopyData {
                    src_offset: 0,
                    dst_offset: 0,
                    size: 1024,
                }],
            },
            Request::CmdFillBuffer {
                buffer_id: 12,
                dst_buffer_id: 13,
                dst_offset: 0,
                size: 1024,
                data: 0xDEADBEEF,
            },
            Request::CmdPipelineBarrier {
                buffer_id: 12,
                src_stage_mask: 2,
                dst_stage_mask: 4,
                dependency_flags: 0,
                memory_barriers: vec![MemoryBarrierData {
                    src_access_mask: 1,
                    dst_access_mask: 2,
                }],
                buffer_memory_barriers: vec![BufferMemoryBarrierData {
                    src_access_mask: 1,
                    dst_access_mask: 2,
                    src_queue_family_index: 0,
                    dst_queue_family_index: 0,
                    buffer_id: 3,
                    offset: 0,
                    size: 1024,
                }],
                image_memory_barriers: vec![ImageMemoryBarrierData {
                    src_access_mask: 1,
                    dst_access_mask: 2,
                    old_layout: ImageLayout::Undefined,
                    new_layout: ImageLayout::ColorAttachmentOptimal,
                    src_queue_family_index: 0,
                    dst_queue_family_index: 0,
                    image_id: 20,
                    subresource_range: ImageSubresourceRange {
                        aspect_mask: ImageAspectFlags(ImageAspectFlags::COLOR),
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                }],
            },
            Request::EndCommandBuffer { buffer_id: 12 },
            Request::CreateFence {
                fence_id: 14,
                signaled: false,
            },
            Request::QueueSubmit {
                queue_id: 2,
                command_buffer_ids: vec![12],
                fence_id: Some(14),
                dirty_memory: vec![],
            },
            Request::WaitForFences {
                fence_ids: vec![14],
                wait_all: true,
                timeout_ns: 1000000,
            },
            Request::GetFenceStatus { fence_id: 14 },
            Request::ResetFences {
                fence_ids: vec![14],
            },
            Request::QueueWaitIdle { queue_id: 2 },
            Request::DeviceWaitIdle { device_id: 1 },
            Request::DestroyResource {
                handle_id: 14,
                resource_type: ResourceType::Fence,
            },
            // Phase 3 Graphics Requests
            Request::CreateImage {
                image_id: 20,
                params: CreateImageParams {
                    image_type: ImageType::Type2D,
                    format: Format::B8G8R8A8Unorm,
                    extent: Extent3D {
                        width: 800,
                        height: 600,
                        depth: 1,
                    },
                    mip_levels: 1,
                    array_layers: 1,
                    samples: SampleCount::Count1,
                    tiling: ImageTiling::Optimal,
                    usage: ImageUsageFlags(ImageUsageFlags::COLOR_ATTACHMENT),
                    sharing_mode: SharingMode::Exclusive,
                    queue_family_indices: vec![],
                    initial_layout: ImageLayout::Undefined,
                },
            },
            Request::DestroyImage { image_id: 20 },
            Request::GetImageMemoryRequirements { image_id: 20 },
            Request::BindImageMemory {
                image_id: 20,
                memory_id: 4,
                offset: 0,
            },
            Request::CreateImageView {
                view_id: 21,
                params: CreateImageViewParams {
                    image_id: 20,
                    view_type: ImageViewType::Type2D,
                    format: Format::B8G8R8A8Unorm,
                    components: ComponentMapping::default(),
                    subresource_range: ImageSubresourceRange {
                        aspect_mask: ImageAspectFlags(ImageAspectFlags::COLOR),
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                },
            },
            Request::DestroyImageView { view_id: 21 },
            Request::CreateSampler {
                sampler_id: 22,
                params: CreateSamplerParams {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    mipmap_mode: SamplerMipmapMode::Linear,
                    address_mode_u: SamplerAddressMode::Repeat,
                    address_mode_v: SamplerAddressMode::Repeat,
                    address_mode_w: SamplerAddressMode::Repeat,
                    mip_lod_bias: F32::from_f32(0.0),
                    anisotropy_enable: false,
                    max_anisotropy: F32::from_f32(1.0),
                    compare_enable: false,
                    compare_op: CompareOp::Never,
                    min_lod: F32::from_f32(0.0),
                    max_lod: F32::from_f32(0.0),
                    border_color: BorderColor::FloatTransparentBlack,
                    unnormalized_coordinates: false,
                },
            },
            Request::DestroySampler { sampler_id: 22 },
            Request::CreateRenderPass {
                render_pass_id: 23,
                params: CreateRenderPassParams {
                    attachments: vec![AttachmentDescription {
                        flags: 0,
                        format: Format::B8G8R8A8Unorm,
                        samples: SampleCount::Count1,
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        stencil_load_op: AttachmentLoadOp::DontCare,
                        stencil_store_op: AttachmentStoreOp::DontCare,
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::PresentSrcKhr,
                    }],
                    subpasses: vec![SubpassDescription {
                        flags: 0,
                        pipeline_bind_point: PipelineBindPoint::Graphics,
                        input_attachments: vec![],
                        color_attachments: vec![AttachmentReference {
                            attachment: 0,
                            layout: ImageLayout::ColorAttachmentOptimal,
                        }],
                        resolve_attachments: vec![],
                        depth_stencil_attachment: None,
                        preserve_attachments: vec![],
                    }],
                    dependencies: vec![SubpassDependency {
                        src_subpass: !0,
                        dst_subpass: 0,
                        src_stage_mask: 0x00000400,
                        dst_stage_mask: 0x00000400,
                        src_access_mask: 0,
                        dst_access_mask: 0x00000100,
                        dependency_flags: 0,
                    }],
                },
            },
            Request::DestroyRenderPass { render_pass_id: 23 },
            Request::CreateFramebuffer {
                framebuffer_id: 24,
                params: CreateFramebufferParams {
                    render_pass_id: 23,
                    attachments: vec![21],
                    width: 800,
                    height: 600,
                    layers: 1,
                },
            },
            Request::DestroyFramebuffer { framebuffer_id: 24 },
            Request::CreateGraphicsPipelines {
                pipelines: vec![GraphicsPipelineCreateInfoData {
                    pipeline_id: 25,
                    stages: vec![PipelineShaderStageData {
                        stage: ShaderStageFlags(ShaderStageFlags::VERTEX),
                        module_id: 5,
                        entry_point: "main".to_string(),
                    }],
                    vertex_input_state: PipelineVertexInputStateData {
                        bindings: vec![VertexInputBindingDescription {
                            binding: 0,
                            stride: 16,
                            input_rate: VertexInputRate::Vertex,
                        }],
                        attributes: vec![VertexInputAttributeDescription {
                            location: 0,
                            binding: 0,
                            format: Format::R8G8B8A8Unorm,
                            offset: 0,
                        }],
                    },
                    input_assembly_state: PipelineInputAssemblyStateData {
                        topology: PrimitiveTopology::TriangleList,
                        primitive_restart_enable: false,
                    },
                    viewport_state: Some(PipelineViewportStateData {
                        viewports: vec![Viewport {
                            x: F32::from_f32(0.0),
                            y: F32::from_f32(0.0),
                            width: F32::from_f32(800.0),
                            height: F32::from_f32(600.0),
                            min_depth: F32::from_f32(0.0),
                            max_depth: F32::from_f32(1.0),
                        }],
                        scissors: vec![Rect2D {
                            offset: Offset2D { x: 0, y: 0 },
                            extent: Extent2D {
                                width: 800,
                                height: 600,
                            },
                        }],
                    }),
                    rasterization_state: PipelineRasterizationStateData {
                        depth_clamp_enable: false,
                        rasterizer_discard_enable: false,
                        polygon_mode: PolygonMode::Fill,
                        cull_mode: CullModeFlags(CullModeFlags::BACK),
                        front_face: FrontFace::Clockwise,
                        depth_bias_enable: false,
                        depth_bias_constant_factor: F32::from_f32(0.0),
                        depth_bias_clamp: F32::from_f32(0.0),
                        depth_bias_slope_factor: F32::from_f32(0.0),
                        line_width: F32::from_f32(1.0),
                    },
                    multisample_state: Some(PipelineMultisampleStateData {
                        rasterization_samples: SampleCount::Count1,
                        sample_shading_enable: false,
                        min_sample_shading: F32::from_f32(1.0),
                        sample_mask: vec![],
                        alpha_to_coverage_enable: false,
                        alpha_to_one_enable: false,
                    }),
                    depth_stencil_state: None,
                    color_blend_state: Some(PipelineColorBlendStateData {
                        logic_op_enable: false,
                        logic_op: LogicOp::Copy,
                        attachments: vec![ColorBlendAttachmentState {
                            blend_enable: false,
                            src_color_blend_factor: BlendFactor::One,
                            dst_color_blend_factor: BlendFactor::Zero,
                            color_blend_op: BlendOp::Add,
                            src_alpha_blend_factor: BlendFactor::One,
                            dst_alpha_blend_factor: BlendFactor::Zero,
                            alpha_blend_op: BlendOp::Add,
                            color_write_mask: ColorComponentFlags(ColorComponentFlags::ALL),
                        }],
                        blend_constants: [
                            F32::from_f32(0.0),
                            F32::from_f32(0.0),
                            F32::from_f32(0.0),
                            F32::from_f32(0.0),
                        ],
                    }),
                    dynamic_state: Some(PipelineDynamicStateData {
                        dynamic_states: vec![DynamicState::Viewport, DynamicState::Scissor],
                    }),
                    layout_id: 7,
                    render_pass_id: 23,
                    subpass: 0,
                    base_pipeline_id: None,
                    base_pipeline_index: -1,
                }],
            },
            Request::CmdBeginRenderPass {
                buffer_id: 12,
                render_pass_id: 23,
                framebuffer_id: 24,
                render_area: Rect2D {
                    offset: Offset2D { x: 0, y: 0 },
                    extent: Extent2D {
                        width: 800,
                        height: 600,
                    },
                },
                clear_values: vec![ClearValueData::ColorFloat([0, 0, 0, 1])],
                contents: SubpassContents::Inline,
            },
            Request::CmdEndRenderPass { buffer_id: 12 },
            Request::CmdBindVertexBuffers {
                buffer_id: 12,
                first_binding: 0,
                buffer_ids: vec![3],
                offsets: vec![0],
            },
            Request::CmdBindIndexBuffer {
                buffer_id: 12,
                index_buffer_id: 3,
                offset: 0,
                index_type: IndexType::Uint32,
            },
            Request::CmdSetViewport {
                buffer_id: 12,
                first_viewport: 0,
                viewports: vec![Viewport {
                    x: F32::from_f32(0.0),
                    y: F32::from_f32(0.0),
                    width: F32::from_f32(800.0),
                    height: F32::from_f32(600.0),
                    min_depth: F32::from_f32(0.0),
                    max_depth: F32::from_f32(1.0),
                }],
            },
            Request::CmdSetScissor {
                buffer_id: 12,
                first_scissor: 0,
                scissors: vec![Rect2D {
                    offset: Offset2D { x: 0, y: 0 },
                    extent: Extent2D {
                        width: 800,
                        height: 600,
                    },
                }],
            },
            Request::CmdDraw {
                buffer_id: 12,
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            Request::CmdDrawIndexed {
                buffer_id: 12,
                index_count: 6,
                instance_count: 1,
                first_index: 0,
                vertex_offset: 0,
                first_instance: 0,
            },
            Request::GetSurfaceCapabilities { surface_id: 1 },
            Request::GetSurfaceFormats { surface_id: 1 },
            Request::GetSurfacePresentModes { surface_id: 1 },
            Request::CreateSwapchain {
                swapchain_id: 30,
                surface_id: 1,
                params: CreateSwapchainParams {
                    min_image_count: 2,
                    image_format: Format::B8G8R8A8Unorm,
                    image_color_space: ColorSpace::SrgbNonlinear,
                    image_extent: Extent2D {
                        width: 800,
                        height: 600,
                    },
                    image_array_layers: 1,
                    image_usage: ImageUsageFlags(ImageUsageFlags::COLOR_ATTACHMENT),
                    image_sharing_mode: SharingMode::Exclusive,
                    queue_family_indices: vec![],
                    pre_transform: SurfaceTransformFlags(SurfaceTransformFlags::IDENTITY),
                    composite_alpha: CompositeAlphaFlags(CompositeAlphaFlags::OPAQUE),
                    present_mode: PresentMode::Fifo,
                    clipped: true,
                    old_swapchain_id: None,
                },
            },
            Request::DestroySwapchain { swapchain_id: 30 },
            Request::GetSwapchainImages { swapchain_id: 30 },
            Request::AcquireNextImage {
                swapchain_id: 30,
                timeout_ns: 1000,
                semaphore_id: None,
                fence_id: None,
            },
            Request::QueuePresent {
                queue_id: 2,
                params: PresentParams {
                    swapchain_id: 30,
                    image_index: 0,
                    wait_semaphores: vec![],
                },
            },
        ];

        for req in requests {
            let encoded = postcard::to_allocvec(&req).expect("Failed to serialize request");
            let decoded: Request =
                postcard::from_bytes(&encoded).expect("Failed to deserialize request");
            assert_eq!(req, decoded);
        }

        let responses = vec![
            Response::Success,
            Response::DeviceSnapshot(PhysicalDeviceSnapshot {
                properties: PhysicalDevicePropertiesData {
                    api_version: 0x00401000,
                    driver_version: 1,
                    vendor_id: 0x10DE,
                    device_id: 0x2503,
                    device_type: 2,
                    device_name: "NVIDIA GeForce RTX 3060 (Remote)".to_string(),
                    pipeline_cache_uuid: [0; 16],
                    limits: PhysicalDeviceLimitsData::default(),
                },
                features: PhysicalDeviceFeaturesData::default(),
                memory_properties: PhysicalDeviceMemoryPropertiesData {
                    memory_types: vec![
                        MemoryTypeData {
                            property_flags: 1,
                            heap_index: 0,
                        },
                        MemoryTypeData {
                            property_flags: 6,
                            heap_index: 1,
                        },
                    ],
                    memory_heaps: vec![
                        MemoryHeapData {
                            size: 12 * 1024 * 1024 * 1024,
                            flags: 1,
                        },
                        MemoryHeapData {
                            size: 16 * 1024 * 1024 * 1024,
                            flags: 0,
                        },
                    ],
                },
                queue_families: vec![QueueFamilyPropertiesData {
                    queue_flags: 6, // COMPUTE | TRANSFER
                    queue_count: 1,
                    timestamp_valid_bits: 64,
                    min_image_transfer_granularity: [1, 1, 1],
                }],
            }),
            Response::MemoryData {
                diffs: vec![MemoryChunkDiff {
                    allocation_id: 1,
                    offset: 0,
                    data: vec![42; 64],
                }],
            },
            Response::FenceStatus { signaled: true },
            Response::WaitResult {
                success: true,
                timeout: false,
            },
            Response::MemoryRequirements {
                size: 65536,
                alignment: 256,
                memory_type_bits: 0x3,
            },
            Response::ImageMemoryRequirements(MemoryRequirementsData {
                size: 65536,
                alignment: 256,
                memory_type_bits: 0x3,
            }),
            Response::ImageCreated { id: 20 },
            Response::ImageViewCreated { id: 21 },
            Response::SamplerCreated { id: 22 },
            Response::RenderPassCreated { id: 23 },
            Response::FramebufferCreated { id: 24 },
            Response::GraphicsPipelinesCreated { ids: vec![25] },
            Response::SurfaceCapabilities(SurfaceCapabilities {
                min_image_count: 1,
                max_image_count: 8,
                current_extent: Extent2D {
                    width: 800,
                    height: 600,
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
                supported_usage_flags: ImageUsageFlags(0x1f),
            }),
            Response::SurfaceFormats(vec![SurfaceFormat {
                format: Format::B8G8R8A8Unorm,
                color_space: ColorSpace::SrgbNonlinear,
            }]),
            Response::SurfacePresentModes(vec![0, 1, 2]),
            Response::SwapchainCreated {
                id: 30,
                image_ids: vec![101, 102],
            },
            Response::SwapchainImages(vec![101, 102]),
            Response::AcquireNextImage {
                image_index: 0,
                suboptimal: false,
            },
            Response::AcquireNextImageResult {
                image_index: 0,
                suboptimal: false,
            },
            Response::QueuePresentResult {
                result: VK_SUCCESS,
                frame: Some(FramePacket {
                    swapchain_id: 30,
                    image_index: 0,
                    width: 800,
                    height: 600,
                    format: Format::B8G8R8A8Unorm,
                    data: vec![0xAA; 16],
                }),
            },
            Response::Error {
                code: -4,
                message: "Out of host memory".to_string(),
            },
        ];

        for resp in responses {
            let encoded = postcard::to_allocvec(&resp).expect("Failed to serialize response");
            let decoded: Response =
                postcard::from_bytes(&encoded).expect("Failed to deserialize response");
            assert_eq!(resp, decoded);
        }
    }

    #[test]
    fn test_zstd_compression_roundtrip() {
        let original_data = vec![0xAB; 256 * 1024]; // 256 KB of compressible data
        let compressed = compress(&original_data).expect("compression failed");
        assert!(compressed.len() < original_data.len() / 10);

        let decompressed =
            decompress(&compressed, original_data.len()).expect("decompression failed");
        assert_eq!(original_data, decompressed);
    }

    #[tokio::test]
    async fn test_wire_transparent_compression_large_frame() {
        let (mut client, mut server) = duplex(1024 * 1024);
        let frame_payload = vec![42u8; 100_000]; // 100 KB compressible frame
        let resp = Response::QueuePresentResult {
            result: 0,
            frame: Some(FramePacket {
                swapchain_id: 100,
                image_index: 1,
                width: 250,
                height: 100,
                format: Format::B8G8R8A8Unorm,
                data: frame_payload.clone(),
            }),
        };

        // Write asynchronously
        write_msg(&mut server, &resp).await.unwrap();

        // Read asynchronously
        let received: Response = read_msg(&mut client, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
        assert_eq!(resp, received);
    }

    #[test]
    fn test_dirty_chunk_detection() {
        let mut tracker = MemorySyncTracker::new();
        let alloc_id = 100u64;

        // 192 KB memory = 3 x 64KB chunks
        let mut memory = vec![0u8; 192 * 1024];

        // Initial scan: all 3 chunks should be reported as dirty
        let diffs1 = tracker.compute_dirty_chunks(alloc_id, &memory);
        assert_eq!(diffs1.len(), 3);
        assert_eq!(diffs1[0].offset, 0);
        assert_eq!(diffs1[1].offset, 64 * 1024);
        assert_eq!(diffs1[2].offset, 128 * 1024);

        // Second scan without modifications: 0 dirty chunks
        let diffs2 = tracker.compute_dirty_chunks(alloc_id, &memory);
        assert_eq!(diffs2.len(), 0);

        // Mutate only chunk 1 (offset 64KB to 128KB)
        memory[64 * 1024 + 50] = 0xFF;
        let diffs3 = tracker.compute_dirty_chunks(alloc_id, &memory);
        assert_eq!(diffs3.len(), 1);
        assert_eq!(diffs3[0].offset, 64 * 1024);
        assert_eq!(diffs3[0].data[50], 0xFF);

        // Next scan without modifications: 0 dirty chunks
        let diffs4 = tracker.compute_dirty_chunks(alloc_id, &memory);
        assert_eq!(diffs4.len(), 0);

        // Mutate chunk 0 and chunk 2
        memory[10] = 0xAA;
        memory[128 * 1024 + 100] = 0xBB;
        let diffs5 = tracker.compute_dirty_chunks(alloc_id, &memory);
        assert_eq!(diffs5.len(), 2);
        assert_eq!(diffs5[0].offset, 0);
        assert_eq!(diffs5[1].offset, 128 * 1024);
    }
}
