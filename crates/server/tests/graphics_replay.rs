use vkremote_server::{VulkanEngine, VulkanSession};
use vkremote_wire::types::*;

fn success(session: &mut VulkanSession, request: Request) {
    assert_eq!(session.handle_request(request), Response::Success);
}

fn requirements(response: Response) -> MemoryRequirementsData {
    match response {
        Response::ImageMemoryRequirements(value) => value,
        Response::MemoryRequirements {
            size,
            alignment,
            memory_type_bits,
        } => MemoryRequirementsData {
            size,
            alignment,
            memory_type_bits,
        },
        other => panic!("Unexpected requirements response: {other:?}"),
    }
}

#[test]
#[ignore = "requires real Vulkan hardware; exclude vkremote ICD to prevent recursive replay"]
fn texture_upload_and_readback_replays_on_real_gpu() {
    let engine = VulkanEngine::new(None).unwrap();
    assert!(!engine.snapshot.properties.device_name.contains("Remote"));
    let memory_types = engine.snapshot.memory_properties.memory_types.clone();
    let mut session = VulkanSession::new(engine);
    success(
        &mut session,
        Request::CreateDevice {
            device_id: 1,
            queue_id: 2,
        },
    );
    for (buffer_id, memory_id) in [(10, 11), (20, 21)] {
        success(
            &mut session,
            Request::CreateBuffer {
                buffer_id,
                size: 64,
                usage: 3,
            },
        );
        let req = requirements(
            session.handle_request(Request::GetBufferMemoryRequirements { buffer_id }),
        );
        let index = memory_types
            .iter()
            .enumerate()
            .find(|(i, t)| req.memory_type_bits & (1 << i) != 0 && t.property_flags & 6 == 6)
            .unwrap()
            .0 as u32;
        success(
            &mut session,
            Request::AllocateMemory {
                memory_id,
                size: req.size,
                memory_type_index: index,
            },
        );
        success(
            &mut session,
            Request::BindBufferMemory {
                buffer_id,
                memory_id,
                offset: 0,
            },
        );
    }
    assert!(matches!(
        session.handle_request(Request::CreateImage {
            image_id: 30,
            params: CreateImageParams {
                image_type: 1.into(),
                format: Format::R8G8B8A8Unorm,
                extent: Extent3D {
                    width: 4,
                    height: 4,
                    depth: 1
                },
                mip_levels: 1,
                array_layers: 1,
                samples: 1.into(),
                tiling: 0.into(),
                usage: 3.into(),
                sharing_mode: 0.into(),
                queue_family_indices: vec![],
                initial_layout: 0.into()
            }
        }),
        Response::ImageCreated { id: 30 }
    ));
    let req =
        requirements(session.handle_request(Request::GetImageMemoryRequirements { image_id: 30 }));
    let index = memory_types
        .iter()
        .enumerate()
        .find(|(i, _)| req.memory_type_bits & (1 << i) != 0)
        .unwrap()
        .0 as u32;
    success(
        &mut session,
        Request::AllocateMemory {
            memory_id: 31,
            size: req.size,
            memory_type_index: index,
        },
    );
    success(
        &mut session,
        Request::BindImageMemory {
            image_id: 30,
            memory_id: 31,
            offset: 0,
        },
    );
    success(&mut session, Request::CreateCommandPool { pool_id: 40 });
    success(
        &mut session,
        Request::AllocateCommandBuffers {
            pool_id: 40,
            buffer_ids: vec![41],
        },
    );
    success(&mut session, Request::BeginCommandBuffer { buffer_id: 41 });
    let range = ImageSubresourceRange {
        aspect_mask: 1.into(),
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let barrier = |old, new, src, dst| Request::CmdPipelineBarrier {
        buffer_id: 41,
        src_stage_mask: 0x10000,
        dst_stage_mask: 0x10000,
        dependency_flags: 0,
        memory_barriers: vec![],
        buffer_memory_barriers: vec![],
        image_memory_barriers: vec![ImageMemoryBarrierData {
            src_access_mask: src,
            dst_access_mask: dst,
            old_layout: old,
            new_layout: new,
            src_queue_family_index: u32::MAX,
            dst_queue_family_index: u32::MAX,
            image_id: 30,
            subresource_range: range,
        }],
    };
    success(&mut session, barrier(0.into(), 7.into(), 0, 0x1000));
    let region = BufferImageCopyData {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: [1, 0, 0, 1],
        image_offset: [0, 0, 0],
        image_extent: [4, 4, 1],
    };
    success(
        &mut session,
        Request::CmdCopyBufferToImage {
            buffer_id: 41,
            src_buffer_id: 10,
            dst_image_id: 30,
            dst_layout: 7,
            regions: vec![region.clone()],
        },
    );
    success(&mut session, barrier(7.into(), 6.into(), 0x1000, 0x800));
    success(
        &mut session,
        Request::CmdCopyImageToBuffer {
            buffer_id: 41,
            src_image_id: 30,
            src_layout: 6,
            dst_buffer_id: 20,
            regions: vec![region],
        },
    );
    success(
        &mut session,
        Request::CmdPipelineBarrier {
            buffer_id: 41,
            src_stage_mask: 0x1000,
            dst_stage_mask: 0x4000,
            dependency_flags: 0,
            memory_barriers: vec![MemoryBarrierData {
                src_access_mask: 0x1000,
                dst_access_mask: 0x2000,
            }],
            buffer_memory_barriers: vec![],
            image_memory_barriers: vec![],
        },
    );
    success(&mut session, Request::EndCommandBuffer { buffer_id: 41 });
    let data = [213, 31, 7, 255].repeat(16);
    success(
        &mut session,
        Request::QueueSubmit {
            queue_id: 2,
            command_buffer_ids: vec![41],
            fence_id: None,
            dirty_memory: vec![MemoryChunkDiff {
                allocation_id: 11,
                offset: 0,
                data: data.clone(),
            }],
        },
    );
    success(&mut session, Request::QueueWaitIdle { queue_id: 2 });
    match session.handle_request(Request::ReadMemoryRanges {
        allocation_ids: vec![21],
    }) {
        Response::MemoryData { diffs } => assert_eq!(&diffs[0].data[..64], &data),
        other => panic!("Unexpected readback response {other:?}"),
    }
}
