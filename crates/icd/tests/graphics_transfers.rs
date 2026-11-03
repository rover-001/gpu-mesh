use ash::vk::{self, Handle};
use std::sync::{Arc, Mutex};
use vkremote_icd::{
    abi::CommandBufferDispatchable,
    client::{Client, MockTransport},
    dispatch::*,
};
use vkremote_wire::types::{Request, Response};

#[test]
fn texture_upload_and_indirect_draws_are_recorded_with_original_parameters() {
    let received = Arc::new(Mutex::new(Vec::new()));
    let captured = received.clone();
    let client = Client::new_with_transport(Box::new(MockTransport::new(move |request| {
        captured.lock().unwrap().push(request.clone());
        Ok(Response::Success)
    })));
    let mut cb = CommandBufferDispatchable::new(41, std::ptr::null_mut(), client.clone(), 1);
    client.begin_command_recording(41);
    unsafe {
        let handle = CommandBufferDispatchable::to_vk(&mut *cb);
        let region = vk::BufferImageCopy::default()
            .buffer_offset(256)
            .buffer_row_length(32)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 2,
                base_array_layer: 4,
                layer_count: 3,
            })
            .image_offset(vk::Offset3D { x: -2, y: 3, z: 0 })
            .image_extent(vk::Extent3D {
                width: 8,
                height: 6,
                depth: 1,
            });
        vkCmdCopyBufferToImage(
            handle,
            vk::Buffer::from_raw(7),
            vk::Image::from_raw(9),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            1,
            &region,
        );
        vkCmdDrawIndexedIndirectCountKHR(
            handle,
            vk::Buffer::from_raw(11),
            128,
            vk::Buffer::from_raw(12),
            16,
            64,
            20,
        );
        vkCmdDispatchIndirect(handle, vk::Buffer::from_raw(13), 24);
        let data = [1u32, 2, 3, 4];
        vkCmdUpdateBuffer(
            handle,
            vk::Buffer::from_raw(14),
            32,
            16,
            data.as_ptr().cast(),
        );
    }
    assert!(
        received.lock().unwrap().is_empty(),
        "Commands must remain batched"
    );
    let commands = client.finish_command_recording(41).unwrap();
    assert_eq!(commands.len(), 4);
    match &commands[0] {
        Request::CmdCopyBufferToImage {
            buffer_id,
            src_buffer_id,
            dst_image_id,
            dst_layout,
            regions,
        } => {
            assert_eq!(
                (*buffer_id, *src_buffer_id, *dst_image_id, *dst_layout),
                (41, 7, 9, vk::ImageLayout::TRANSFER_DST_OPTIMAL.as_raw())
            );
            assert_eq!(regions[0].buffer_offset, 256);
            assert_eq!(regions[0].buffer_row_length, 32);
            assert_eq!(regions[0].image_subresource, [1, 2, 4, 3]);
            assert_eq!(regions[0].image_offset, [-2, 3, 0]);
            assert_eq!(regions[0].image_extent, [8, 6, 1]);
        }
        other => panic!("Unexpected {other:?}"),
    }
    assert_eq!(
        commands[1],
        Request::CmdDrawIndirectCount {
            buffer_id: 41,
            indirect_buffer_id: 11,
            offset: 128,
            count_buffer_id: 12,
            count_buffer_offset: 16,
            max_draw_count: 64,
            stride: 20,
            indexed: true
        }
    );
    assert_eq!(
        commands[2],
        Request::CmdDispatchIndirect {
            buffer_id: 41,
            indirect_buffer_id: 13,
            offset: 24
        }
    );
    for request in commands {
        let bytes = postcard::to_allocvec(&request).unwrap();
        assert_eq!(postcard::from_bytes::<Request>(&bytes).unwrap(), request);
    }
}
