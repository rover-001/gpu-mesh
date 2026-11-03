use vkremote_wire::types::*;

#[test]
fn new_graphics_commands_roundtrip_without_losing_regions_or_float_bits() {
    let region = ImageCopyRegionData {
        src_subresource: [1, 2, 3, 4],
        src_offset: [-1, 2, 3],
        dst_subresource: [1, 0, 5, 4],
        dst_offset: [4, 5, 6],
        extent: [16, 8, 1],
    };
    let requests = vec![
        Request::CmdCopyImage {
            buffer_id: 1,
            src_image_id: 2,
            src_layout: 6,
            dst_image_id: 3,
            dst_layout: 7,
            regions: vec![region.clone()],
        },
        Request::CmdResolveImage {
            buffer_id: 1,
            src_image_id: 2,
            src_layout: 6,
            dst_image_id: 3,
            dst_layout: 7,
            regions: vec![region],
        },
        Request::CmdBlitImage {
            buffer_id: 1,
            src_image_id: 2,
            src_layout: 6,
            dst_image_id: 3,
            dst_layout: 7,
            regions: vec![ImageBlitData {
                src_subresource: [1, 0, 0, 1],
                src_offsets: [[0, 0, 0], [32, 32, 1]],
                dst_subresource: [1, 1, 0, 1],
                dst_offsets: [[0, 0, 0], [16, 16, 1]],
            }],
            filter: 1,
        },
        Request::CmdCopyImageToBuffer {
            buffer_id: 1,
            src_image_id: 2,
            src_layout: 6,
            dst_buffer_id: 3,
            regions: vec![BufferImageCopyData {
                buffer_offset: 128,
                buffer_row_length: 64,
                buffer_image_height: 32,
                image_subresource: [1, 0, 0, 1],
                image_offset: [0, 0, 0],
                image_extent: [8, 8, 1],
            }],
        },
        Request::CmdClearColorImage {
            buffer_id: 1,
            image_id: 2,
            layout: 7,
            color: [0x7fc00001, 0x80000000, 0x3f800000, 0],
            ranges: vec![ImageSubresourceRange {
                aspect_mask: 1.into(),
                base_mip_level: 2,
                level_count: 4,
                base_array_layer: 3,
                layer_count: 6,
            }],
        },
        Request::CmdClearDepthStencilImage {
            buffer_id: 1,
            image_id: 2,
            layout: 7,
            depth_bits: 0.5f32.to_bits(),
            stencil: 7,
            ranges: vec![],
        },
        Request::CmdDrawIndirect {
            buffer_id: 1,
            indirect_buffer_id: 2,
            offset: u64::MAX - 4,
            draw_count: 32,
            stride: 16,
            indexed: false,
        },
        Request::GetBufferMemoryRequirements { buffer_id: 123 },
    ];
    for request in requests {
        let encoded = postcard::to_allocvec(&request).unwrap();
        let decoded: Request = postcard::from_bytes(&encoded).unwrap();
        assert_eq!(decoded, request);
    }
}
