use tokio::net::TcpStream;
use vkremote_wire::framing::{client_handshake, read_msg, write_msg, DEFAULT_MAX_FRAME_SIZE};
use vkremote_wire::types::*;

#[tokio::test]
async fn test_remote_swapchain_and_depth_image() {
    let endpoint = "10.158.255.229:50051";
    let token = "vkremote-secret";
    println!("Connecting to {}...", endpoint);
    let mut stream = TcpStream::connect(endpoint).await.expect("connect failed");
    client_handshake(&mut stream, token.to_string())
        .await
        .expect("handshake failed");
    println!("Handshake OK");

    // 1. GetSnapshot
    write_msg(&mut stream, &Request::GetDeviceSnapshot)
        .await
        .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE)
        .await
        .unwrap();
    println!("Snapshot: {:?}", resp);

    // 2. CreateDevice
    write_msg(
        &mut stream,
        &Request::CreateDevice {
            queue_id: 1,
            device_id: 1,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE)
        .await
        .unwrap();
    println!("CreateDevice: {:?}", resp);

    // Test DestroyResource
    println!("Testing DestroyResource for non-existent shader module...");
    write_msg(
        &mut stream,
        &Request::DestroyResource {
            handle_id: 99999,
            resource_type: ResourceType::ShaderModule,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE)
        .await
        .unwrap();
    println!("DestroyResource resp: {:?}", resp);

    // 3. CreateSwapchain
    println!("Sending CreateSwapchain 1920x1080...");
    let params = CreateSwapchainParams {
        min_image_count: 2,
        image_format: 44.into(), // B8G8R8A8_UNORM
        image_color_space: 0.into(),
        image_extent: Extent2D {
            width: 1920,
            height: 1080,
        },
        image_array_layers: 1,
        image_usage: 0x10.into(), // COLOR_ATTACHMENT
        image_sharing_mode: 0.into(),
        queue_family_indices: Vec::new(),
        pre_transform: 1.into(),
        composite_alpha: 1.into(),
        present_mode: 0.into(),
        clipped: true,
        old_swapchain_id: None,
    };
    write_msg(
        &mut stream,
        &Request::CreateSwapchain {
            swapchain_id: 10,
            surface_id: 1,
            params: params.clone(),
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE)
        .await
        .unwrap();
    println!("CreateSwapchain resp: {:?}", resp);

    // 4. CreateImage depth 1
    println!("Sending CreateImage D32_SFLOAT 1920x1080 (1)...");
    let img_params = CreateImageParams {
        image_type: 1.into(), // 2D
        format: 126.into(),   // D32_SFLOAT
        extent: Extent3D {
            width: 1920,
            height: 1080,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: 1.into(),
        tiling: 0.into(), // OPTIMAL
        usage: 0x20.into(), // DEPTH_STENCIL_ATTACHMENT
        sharing_mode: 0.into(),
        queue_family_indices: Vec::new(),
        initial_layout: 0.into(),
    };
    write_msg(
        &mut stream,
        &Request::CreateImage {
            image_id: 20,
            params: img_params.clone(),
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE)
        .await
        .unwrap();
    println!("CreateImage 1 resp: {:?}", resp);

    // 5. CreateSwapchain 2 (reconfiguration)
    println!("Sending CreateSwapchain 2 1920x1080...");
    let params2 = CreateSwapchainParams {
        old_swapchain_id: Some(10),
        ..params
    };
    write_msg(
        &mut stream,
        &Request::CreateSwapchain {
            swapchain_id: 11,
            surface_id: 1,
            params: params2,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE)
        .await
        .unwrap();
    println!("CreateSwapchain 2 resp: {:?}", resp);

    // 6. CreateImage depth 2
    println!("Sending CreateImage D32_SFLOAT 1920x1080 (2)...");
    write_msg(
        &mut stream,
        &Request::CreateImage {
            image_id: 21,
            params: img_params,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE)
        .await
        .unwrap();
    println!("CreateImage 2 resp: {:?}", resp);
}
