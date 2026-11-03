use tokio::net::{TcpListener, TcpStream};
use vkremote_server::{run_server_with_config, Engine, ServerConfig};
use vkremote_wire::framing::{client_handshake, read_msg, write_msg, DEFAULT_MAX_FRAME_SIZE};
use vkremote_wire::types::*;

async fn spawn_test_server(force_mock: bool) -> (String, String) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let token = "test-secret-token".to_string();

    let config = ServerConfig {
        token: token.clone(),
        gpu_index: None,
        force_mock,
    };

    tokio::spawn(async move {
        let _ = run_server_with_config(listener, config).await;
    });

    (addr, token)
}

#[tokio::test]
async fn test_handshake_authentication() {
    let (addr, token) = spawn_test_server(true).await;

    // Test 1: Bad token rejected
    {
        let mut stream = TcpStream::connect(&addr).await.unwrap();
        let err = client_handshake(&mut stream, "wrong-token").await;
        assert!(err.is_err(), "Handshake with wrong token should fail");
    }

    // Test 2: Correct token accepted
    {
        let mut stream = TcpStream::connect(&addr).await.unwrap();
        let resp = client_handshake(&mut stream, token).await;
        assert!(resp.is_ok(), "Handshake with correct token should succeed");
        let handshake = resp.unwrap();
        assert!(handshake.accepted);
    }
}

#[tokio::test]
async fn test_mock_full_replay_pipeline() {
    let (addr, token) = spawn_test_server(true).await;
    let mut stream = TcpStream::connect(&addr).await.unwrap();
    let _ = client_handshake(&mut stream, token).await.unwrap();

    // 1. Snapshot
    write_msg(&mut stream, &Request::GetDeviceSnapshot)
        .await
        .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    let snapshot = match resp {
        Response::DeviceSnapshot(s) => s,
        other => panic!("Expected DeviceSnapshot, got {:?}", other),
    };
    assert!(!snapshot.properties.device_name.is_empty());
    assert!(!snapshot.queue_families.is_empty());

    // 2. Create Device
    write_msg(
        &mut stream,
        &Request::CreateDevice {
            device_id: 1,
            queue_id: 2,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    // 3. Create Buffer & Allocate Memory
    write_msg(
        &mut stream,
        &Request::CreateBuffer {
            buffer_id: 10,
            size: 4096,
            usage: 0x80, // STORAGE_BUFFER
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    write_msg(
        &mut stream,
        &Request::AllocateMemory {
            memory_id: 20,
            size: 4096,
            memory_type_index: 1, // Host visible
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    write_msg(
        &mut stream,
        &Request::BindBufferMemory {
            buffer_id: 10,
            memory_id: 20,
            offset: 0,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    // 4. Write Memory Ranges
    let test_data = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
    write_msg(
        &mut stream,
        &Request::WriteMemoryRanges {
            diffs: vec![MemoryChunkDiff {
                allocation_id: 20,
                offset: 0,
                data: test_data.clone(),
            }],
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    // 5. Read Memory Ranges back
    write_msg(
        &mut stream,
        &Request::ReadMemoryRanges {
            allocation_ids: vec![20],
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    match resp {
        Response::MemoryData { diffs } => {
            assert!(!diffs.is_empty());
            assert_eq!(&diffs[0].data[..8], &test_data[..]);
        }
        other => panic!("Expected MemoryData, got {:?}", other),
    }

    // 6. Fence Lifecycle & Queue Submit
    write_msg(
        &mut stream,
        &Request::CreateFence {
            fence_id: 30,
            signaled: false,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    write_msg(&mut stream, &Request::GetFenceStatus { fence_id: 30 })
        .await
        .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::FenceStatus { signaled: false });

    // Command buffer recording
    write_msg(&mut stream, &Request::CreateCommandPool { pool_id: 40 })
        .await
        .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    write_msg(
        &mut stream,
        &Request::AllocateCommandBuffers {
            pool_id: 40,
            buffer_ids: vec![50],
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    write_msg(&mut stream, &Request::BeginCommandBuffer { buffer_id: 50 })
        .await
        .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    write_msg(
        &mut stream,
        &Request::CmdDispatch {
            buffer_id: 50,
            group_count_x: 1,
            group_count_y: 1,
            group_count_z: 1,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    write_msg(&mut stream, &Request::EndCommandBuffer { buffer_id: 50 })
        .await
        .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    // Queue Submit
    let extra_data = vec![99u8, 88, 77];
    write_msg(
        &mut stream,
        &Request::QueueSubmit {
            queue_id: 2,
            command_buffer_ids: vec![50],
            fence_id: Some(30),
            dirty_memory: vec![MemoryChunkDiff {
                allocation_id: 20,
                offset: 8,
                data: extra_data.clone(),
            }],
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    // Wait on Fence
    write_msg(
        &mut stream,
        &Request::WaitForFences {
            fence_ids: vec![30],
            wait_all: true,
            timeout_ns: 1_000_000_000,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(
        resp,
        Response::WaitResult {
            success: true,
            timeout: false
        }
    );

    // Reset Fence
    write_msg(
        &mut stream,
        &Request::ResetFences {
            fence_ids: vec![30],
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    // Teardown Resources
    for (handle_id, rtype) in [
        (30, ResourceType::Fence),
        (50, ResourceType::CommandBuffer),
        (40, ResourceType::CommandPool),
        (10, ResourceType::Buffer),
        (20, ResourceType::DeviceMemory),
        (1, ResourceType::Device),
    ] {
        write_msg(
            &mut stream,
            &Request::DestroyResource {
                handle_id,
                resource_type: rtype,
            },
        )
        .await
        .unwrap();
        let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
        assert_eq!(resp, Response::Success);
    }
}

#[tokio::test]
async fn test_auto_engine_or_vulkan_hardware() {
    // Test engine initialization on host (automatically picks Vulkan hardware if available, or mock)
    let engine = Engine::new(None, false).expect("Engine initialization should never fail");
    let snapshot = engine.snapshot();
    assert!(!snapshot.properties.device_name.is_empty());
    assert!(!snapshot.queue_families.is_empty());
    assert!(!snapshot.memory_properties.memory_types.is_empty());

    // Run a server with auto engine
    let (addr, token) = spawn_test_server(false).await;
    let mut stream = TcpStream::connect(&addr).await.unwrap();
    let _ = client_handshake(&mut stream, token).await.unwrap();

    write_msg(&mut stream, &Request::GetDeviceSnapshot)
        .await
        .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    match resp {
        Response::DeviceSnapshot(s) => {
            assert_eq!(s.properties.device_name, snapshot.properties.device_name);
        }
        other => panic!("Expected DeviceSnapshot, got {:?}", other),
    }

    // Create device on the real or mock engine
    write_msg(
        &mut stream,
        &Request::CreateDevice {
            device_id: 1,
            queue_id: 2,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    // Create a fence
    write_msg(
        &mut stream,
        &Request::CreateFence {
            fence_id: 100,
            signaled: true,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    // Wait on fence
    write_msg(
        &mut stream,
        &Request::WaitForFences {
            fence_ids: vec![100],
            wait_all: true,
            timeout_ns: 1_000_000,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert!(matches!(
        resp,
        Response::Success
            | Response::WaitResult {
                success: true,
                timeout: false
            }
    ));

    // Destroy device
    write_msg(
        &mut stream,
        &Request::DestroyResource {
            handle_id: 1,
            resource_type: ResourceType::Device,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);
}

#[tokio::test]
async fn test_concurrent_sessions() {
    let (addr, token) = spawn_test_server(true).await;

    let mut handles = Vec::new();
    for i in 0..3 {
        let addr = addr.clone();
        let token = token.clone();
        handles.push(tokio::spawn(async move {
            let mut stream = TcpStream::connect(&addr).await.unwrap();
            let _ = client_handshake(&mut stream, token).await.unwrap();

            write_msg(&mut stream, &Request::GetDeviceSnapshot)
                .await
                .unwrap();
            let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
            assert!(matches!(resp, Response::DeviceSnapshot(_)));

            write_msg(
                &mut stream,
                &Request::CreateDevice {
                    device_id: 100 + i,
                    queue_id: 200 + i,
                },
            )
            .await
            .unwrap();
            let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
            assert_eq!(resp, Response::Success);
        }));
    }

    for h in handles {
        h.await.unwrap();
    }
}

#[tokio::test]
async fn test_hardware_buffer_roundtrip() {
    let (addr, token) = spawn_test_server(false).await;
    let mut stream = TcpStream::connect(&addr).await.unwrap();
    let _ = client_handshake(&mut stream, token).await.unwrap();

    // 1. Get snapshot and find host-visible memory type index
    write_msg(&mut stream, &Request::GetDeviceSnapshot)
        .await
        .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    let snapshot = match resp {
        Response::DeviceSnapshot(s) => s,
        other => panic!("Expected DeviceSnapshot, got {:?}", other),
    };

    // Find host-visible memory type (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 2)
    let host_visible_type_idx = snapshot
        .memory_properties
        .memory_types
        .iter()
        .position(|m| (m.property_flags & 2) != 0)
        .expect("Host visible memory type required") as u32;

    // 2. Create logical device
    write_msg(
        &mut stream,
        &Request::CreateDevice {
            device_id: 500,
            queue_id: 501,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    // 3. Create buffer and allocate memory
    write_msg(
        &mut stream,
        &Request::CreateBuffer {
            buffer_id: 502,
            size: 2048,
            usage: 0x80, // STORAGE_BUFFER
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    write_msg(
        &mut stream,
        &Request::AllocateMemory {
            memory_id: 503,
            size: 2048,
            memory_type_index: host_visible_type_idx,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    write_msg(
        &mut stream,
        &Request::BindBufferMemory {
            buffer_id: 502,
            memory_id: 503,
            offset: 0,
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    // 4. Write data
    let payload = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
    write_msg(
        &mut stream,
        &Request::WriteMemoryRanges {
            diffs: vec![MemoryChunkDiff {
                allocation_id: 503,
                offset: 0,
                data: payload.clone(),
            }],
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    assert_eq!(resp, Response::Success);

    // 5. Read back data
    write_msg(
        &mut stream,
        &Request::ReadMemoryRanges {
            allocation_ids: vec![503],
        },
    )
    .await
    .unwrap();
    let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
    match resp {
        Response::MemoryData { diffs } => {
            assert!(!diffs.is_empty());
            assert_eq!(&diffs[0].data[..8], &payload[..]);
        }
        other => panic!("Expected MemoryData, got {:?}", other),
    }

    // 6. Cleanup
    for (handle_id, rtype) in [
        (502, ResourceType::Buffer),
        (503, ResourceType::DeviceMemory),
        (500, ResourceType::Device),
    ] {
        write_msg(
            &mut stream,
            &Request::DestroyResource {
                handle_id,
                resource_type: rtype,
            },
        )
        .await
        .unwrap();
        let resp: Response = read_msg(&mut stream, DEFAULT_MAX_FRAME_SIZE).await.unwrap();
        assert_eq!(resp, Response::Success);
    }
}
