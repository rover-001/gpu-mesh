use ash::vk::Handle;
use std::ffi::CString;
use std::mem::offset_of;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use vkremote_icd::abi::{
    CommandBufferDispatchable, DeviceDispatchable, InstanceDispatchable,
    PhysicalDeviceDispatchable, QueueDispatchable,
};
use vkremote_icd::client::{Client, MockTransport};
use vkremote_icd::entry::{
    vk_icdGetInstanceProcAddr, vk_icdGetPhysicalDeviceProcAddr,
    vk_icdNegotiateLoaderICDInterfaceVersion,
};
use vkremote_icd::manifest::generate_icd_json;
use vkremote_wire::types::{
    MemoryChunkDiff, MemoryHeapData, MemoryTypeData, PhysicalDeviceFeaturesData,
    PhysicalDeviceLimitsData, PhysicalDeviceMemoryPropertiesData, PhysicalDevicePropertiesData,
    PhysicalDeviceSnapshot, QueueFamilyPropertiesData, Request, Response,
};

#[test]
fn timeline_submission_advances_only_after_remote_completion() {
    use vkremote_icd::dispatch::{
        vkCreateSemaphore, vkGetSemaphoreCounterValue, vkQueueSubmit, vkWaitSemaphores,
    };
    // With the no-op QueueWaitIdle design, the ICD signals timeline semaphores
    // immediately after QueueSubmit returns Success (the server is synchronous).
    // No separate QueueWaitIdle request is sent over the wire.
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let client = Client::new_with_transport(Box::new(MockTransport::new(move |req| {
        match req {
            Request::QueueSubmit { .. } => {
                observed.fetch_add(1, Ordering::SeqCst);
            }
            Request::WriteMemoryRanges { .. } => {} // dirty memory flush, benign
            _ => panic!("Unexpected request"),
        }
        Ok(Response::Success)
    })));
    let mut device = DeviceDispatchable::new(1, client.clone(), std::ptr::null_mut());
    let handle = DeviceDispatchable::to_vk(&mut *device);
    let mut queue = QueueDispatchable::new(2, &mut *device, client.clone(), 0, 0);
    unsafe {
        let mut ty = ash::vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(ash::vk::SemaphoreType::TIMELINE)
            .initial_value(3);
        let info = ash::vk::SemaphoreCreateInfo::default().push_next(&mut ty);
        let mut semaphore = ash::vk::Semaphore::null();
        assert_eq!(
            vkCreateSemaphore(handle, &info, std::ptr::null(), &mut semaphore),
            ash::vk::Result::SUCCESS
        );
        let mut value = 0;
        assert_eq!(
            vkGetSemaphoreCounterValue(handle, semaphore, &mut value),
            ash::vk::Result::SUCCESS
        );
        assert_eq!(value, 3);
        let semaphores = [semaphore];
        let values = [7];
        let wait = ash::vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        assert_eq!(vkWaitSemaphores(handle, &wait, 0), ash::vk::Result::TIMEOUT);
        let mut timeline =
            ash::vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let submit = ash::vk::SubmitInfo::default()
            .signal_semaphores(&semaphores)
            .push_next(&mut timeline);
        assert_eq!(
            vkQueueSubmit(
                QueueDispatchable::to_vk(&mut *queue),
                1,
                &submit,
                ash::vk::Fence::null()
            ),
            ash::vk::Result::SUCCESS
        );
        // Only QueueSubmit is sent; QueueWaitIdle is a no-op on the client side.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            vkGetSemaphoreCounterValue(handle, semaphore, &mut value),
            ash::vk::Result::SUCCESS
        );
        assert_eq!(value, 7);
        assert_eq!(vkWaitSemaphores(handle, &wait, 0), ash::vk::Result::SUCCESS);
    }
}

#[test]
fn timeline_wait_all_any_and_host_signal() {
    let client = Client::new_with_transport(Box::new(MockTransport::new(|_| {
        panic!("Unexpected transport")
    })));
    client.register_timeline(1, 0);
    client.register_timeline(2, 2);
    assert_eq!(
        client.wait_timelines(&[(1, 1), (2, 2)], true, 0),
        ash::vk::Result::SUCCESS
    );
    assert_eq!(
        client.wait_timelines(&[(1, 1), (2, 2)], false, 0),
        ash::vk::Result::TIMEOUT
    );
    let signal = client.clone();
    let worker = std::thread::spawn(move || signal.signal_timeline(1, 4));
    assert_eq!(
        client.wait_timelines(&[(1, 4)], false, 1_000_000_000),
        ash::vk::Result::SUCCESS
    );
    assert_eq!(worker.join().unwrap(), ash::vk::Result::SUCCESS);
    assert_eq!(client.timeline_value(1), Some(4));
    client.remove_timeline(1);
    assert_eq!(client.timeline_value(1), None);
}

#[test]
fn large_memory_upload_is_split_before_submission() {
    let uploads = Arc::new(AtomicUsize::new(0));
    let observed = uploads.clone();
    let client = Client::new_with_transport(Box::new(MockTransport::new(move |req| {
        match req {
            Request::WriteMemoryRanges { diffs } => {
                assert!(diffs.len() <= 32);
                observed.fetch_add(diffs.len(), Ordering::SeqCst);
            }
            Request::QueueSubmit { dirty_memory, .. } => {
                assert!(dirty_memory.is_empty());
                assert_eq!(observed.load(Ordering::SeqCst), 33);
            }
            _ => panic!("Unexpected request"),
        }
        Ok(Response::Success)
    })));
    client.register_allocation(8, (33 * vkremote_wire::CHUNK_SIZE) as u64, 0, true);
    let ptr = client
        .map_memory(8, 0, (33 * vkremote_wire::CHUNK_SIZE) as u64)
        .unwrap();
    unsafe {
        for i in 0..33 {
            *ptr.add(i * vkremote_wire::CHUNK_SIZE) = (i + 1) as u8;
        }
    }
    let mut queue = QueueDispatchable::new(2, std::ptr::null_mut(), client, 0, 0);
    unsafe {
        assert_eq!(
            vkremote_icd::dispatch::vkQueueSubmit(
                QueueDispatchable::to_vk(&mut *queue),
                0,
                std::ptr::null(),
                ash::vk::Fence::null()
            ),
            ash::vk::Result::SUCCESS
        );
    }
    assert_eq!(uploads.load(Ordering::SeqCst), 33);
}

#[test]
fn failed_buffer_creation_does_not_publish_a_handle() {
    let client = Client::new_with_transport(Box::new(MockTransport::new(|_| {
        Ok(Response::Error {
            code: -2,
            message: "allocation failed".into(),
        })
    })));
    let mut device = DeviceDispatchable::new(1, client, std::ptr::null_mut());
    let mut buffer = ash::vk::Buffer::null();
    unsafe {
        assert_eq!(
            vkremote_icd::dispatch::vkCreateBuffer(
                DeviceDispatchable::to_vk(&mut *device),
                &ash::vk::BufferCreateInfo::default().size(64),
                std::ptr::null(),
                &mut buffer
            ),
            ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
        );
    }
    assert!(buffer.is_null());
}

#[test]
fn test_dispatchable_handle_offsets() {
    assert_eq!(offset_of!(InstanceDispatchable, loader_data), 0);
    assert_eq!(offset_of!(PhysicalDeviceDispatchable, loader_data), 0);
    assert_eq!(offset_of!(DeviceDispatchable, loader_data), 0);
    assert_eq!(offset_of!(QueueDispatchable, loader_data), 0);
    assert_eq!(offset_of!(CommandBufferDispatchable, loader_data), 0);
}

#[test]
fn test_loader_icd_negotiation() {
    unsafe {
        // Null pointer check
        let res = vk_icdNegotiateLoaderICDInterfaceVersion(std::ptr::null_mut());
        assert_eq!(res, ash::vk::Result::ERROR_INITIALIZATION_FAILED);

        // Version > 5 clamped to 5
        let mut ver = 7u32;
        let res = vk_icdNegotiateLoaderICDInterfaceVersion(&mut ver);
        assert_eq!(res, ash::vk::Result::SUCCESS);
        assert_eq!(ver, 5);

        // Version 5 kept at 5
        let mut ver = 5u32;
        let res = vk_icdNegotiateLoaderICDInterfaceVersion(&mut ver);
        assert_eq!(res, ash::vk::Result::SUCCESS);
        assert_eq!(ver, 5);

        // Version 3 kept at 3
        let mut ver = 3u32;
        let res = vk_icdNegotiateLoaderICDInterfaceVersion(&mut ver);
        assert_eq!(res, ash::vk::Result::SUCCESS);
        assert_eq!(ver, 3);
    }
}

fn mock_snapshot() -> PhysicalDeviceSnapshot {
    PhysicalDeviceSnapshot {
        properties: PhysicalDevicePropertiesData {
            api_version: ash::vk::make_api_version(0, 1, 3, 0),
            driver_version: 1,
            vendor_id: 0x10DE,
            device_id: 0x2684,
            device_type: ash::vk::PhysicalDeviceType::DISCRETE_GPU.as_raw() as u32,
            device_name: "RTX 4090".to_string(),
            pipeline_cache_uuid: [0u8; 16],
            limits: PhysicalDeviceLimitsData::default(),
        },
        features: PhysicalDeviceFeaturesData::default(),
        memory_properties: PhysicalDeviceMemoryPropertiesData {
            memory_types: vec![MemoryTypeData {
                property_flags: (ash::vk::MemoryPropertyFlags::DEVICE_LOCAL
                    | ash::vk::MemoryPropertyFlags::HOST_VISIBLE
                    | ash::vk::MemoryPropertyFlags::HOST_COHERENT)
                    .as_raw(),
                heap_index: 0,
            }],
            memory_heaps: vec![MemoryHeapData {
                size: 24 * 1024 * 1024 * 1024,
                flags: ash::vk::MemoryHeapFlags::DEVICE_LOCAL.as_raw(),
            }],
        },
        queue_families: vec![QueueFamilyPropertiesData {
            queue_flags: ash::vk::QueueFlags::COMPUTE.as_raw(),
            queue_count: 1,
            timestamp_valid_bits: 64,
            min_image_transfer_granularity: [1, 1, 1],
        }],
    }
}

#[test]
fn test_get_proc_addr_all_compute_functions() {
    let mock = MockTransport::new(|req| match req {
        Request::GetDeviceSnapshot => Ok(Response::DeviceSnapshot(mock_snapshot())),
        _ => Ok(Response::Success),
    });

    let client = Client::new_with_transport(Box::new(mock));
    let snapshot = Arc::new(mock_snapshot());
    let mut instance_box = InstanceDispatchable::new(client, snapshot);
    let instance_ptr = &mut *instance_box as *mut InstanceDispatchable;
    let instance_handle = InstanceDispatchable::to_vk(instance_ptr);

    // Global queries (with null instance)
    unsafe {
        let name = CString::new("vkCreateInstance").unwrap();
        assert!(vk_icdGetInstanceProcAddr(ash::vk::Instance::null(), name.as_ptr()).is_some());

        let name = CString::new("vkEnumerateInstanceVersion").unwrap();
        assert!(vk_icdGetInstanceProcAddr(ash::vk::Instance::null(), name.as_ptr()).is_some());

        let name = CString::new("vkEnumerateInstanceExtensionProperties").unwrap();
        assert!(vk_icdGetInstanceProcAddr(ash::vk::Instance::null(), name.as_ptr()).is_some());

        let name = CString::new("vk_icdNegotiateLoaderICDInterfaceVersion").unwrap();
        assert!(vk_icdGetInstanceProcAddr(ash::vk::Instance::null(), name.as_ptr()).is_some());

        // Non-global should be None with null instance
        let name = CString::new("vkCreateDevice").unwrap();
        assert!(vk_icdGetInstanceProcAddr(ash::vk::Instance::null(), name.as_ptr()).is_none());
    }

    // All ~60 compute functions queried with non-null instance
    let required_functions = [
        // Instance / Global
        "vkEnumerateInstanceVersion",
        "vkEnumerateInstanceExtensionProperties",
        "vkCreateInstance",
        "vkDestroyInstance",
        "vkEnumeratePhysicalDevices",
        "vkGetPhysicalDeviceProperties",
        "vkGetPhysicalDeviceProperties2",
        "vkGetPhysicalDeviceFeatures",
        "vkGetPhysicalDeviceFeatures2",
        "vkGetPhysicalDeviceMemoryProperties",
        "vkGetPhysicalDeviceMemoryProperties2",
        "vkGetPhysicalDeviceQueueFamilyProperties",
        "vkGetPhysicalDeviceQueueFamilyProperties2",
        "vkGetPhysicalDeviceFormatProperties",
        "vkEnumerateDeviceExtensionProperties",
        "vkCreateDevice",
        // Device / Queue
        "vkGetDeviceProcAddr",
        "vkDestroyDevice",
        "vkGetDeviceQueue",
        "vkQueueSubmit",
        "vkQueueWaitIdle",
        "vkDeviceWaitIdle",
        // Resources & Memory
        "vkCreateBuffer",
        "vkDestroyBuffer",
        "vkGetBufferMemoryRequirements",
        "vkGetBufferMemoryRequirements2",
        "vkGetBufferMemoryRequirements2KHR",
        "vkCreateImage",
        "vkDestroyImage",
        "vkGetImageMemoryRequirements",
        "vkGetImageSubresourceLayout",
        "vkGetImageMemoryRequirements2",
        "vkGetImageMemoryRequirements2KHR",
        "vkAllocateMemory",
        "vkFreeMemory",
        "vkBindBufferMemory",
        "vkBindBufferMemory2",
        "vkBindBufferMemory2KHR",
        "vkBindImageMemory",
        "vkBindImageMemory2",
        "vkBindImageMemory2KHR",
        "vkMapMemory",
        "vkUnmapMemory",
        "vkFlushMappedMemoryRanges",
        "vkInvalidateMappedMemoryRanges",
        // Shaders & Pipelines
        "vkCreateShaderModule",
        "vkDestroyShaderModule",
        "vkCreateDescriptorSetLayout",
        "vkDestroyDescriptorSetLayout",
        "vkCreatePipelineLayout",
        "vkDestroyPipelineLayout",
        "vkCreateComputePipelines",
        "vkDestroyPipeline",
        "vkCreateDescriptorPool",
        "vkDestroyDescriptorPool",
        "vkAllocateDescriptorSets",
        "vkUpdateDescriptorSets",
        // Commands
        "vkCreateCommandPool",
        "vkDestroyCommandPool",
        "vkAllocateCommandBuffers",
        "vkFreeCommandBuffers",
        "vkResetCommandBuffer",
        "vkBeginCommandBuffer",
        "vkEndCommandBuffer",
        "vkCmdBindPipeline",
        "vkCmdBindDescriptorSets",
        "vkCmdPushConstants",
        "vkCmdDispatch",
        "vkCmdCopyBuffer",
        "vkCmdFillBuffer",
        "vkCmdPipelineBarrier",
        // Synchronization
        "vkCreateFence",
        "vkDestroyFence",
        "vkWaitForFences",
        "vkResetFences",
        "vkGetFenceStatus",
    ];

    for func_name in required_functions {
        let c_name = CString::new(func_name).unwrap();
        let pfn = unsafe { vk_icdGetInstanceProcAddr(instance_handle, c_name.as_ptr()) };
        assert!(
            pfn.is_some(),
            "vk_icdGetInstanceProcAddr failed to resolve: {}",
            func_name
        );
    }

    // Also test physical device proc addr
    for func_name in [
        "vkGetPhysicalDeviceProperties",
        "vkGetPhysicalDeviceFeatures",
        "vkGetPhysicalDeviceMemoryProperties",
        "vkGetPhysicalDeviceQueueFamilyProperties",
        "vkCreateDevice",
    ] {
        let c_name = CString::new(func_name).unwrap();
        let pfn = unsafe { vk_icdGetPhysicalDeviceProcAddr(instance_handle, c_name.as_ptr()) };
        assert!(
            pfn.is_some(),
            "vk_icdGetPhysicalDeviceProcAddr failed to resolve: {}",
            func_name
        );
    }
}

#[test]
fn test_mock_client_and_shadow_memory_tracker() {
    let requests_received = Arc::new(AtomicUsize::new(0));
    let req_counter = Arc::clone(&requests_received);

    let mock = MockTransport::new(move |req| {
        req_counter.fetch_add(1, Ordering::SeqCst);
        match req {
            Request::GetDeviceSnapshot => Ok(Response::DeviceSnapshot(mock_snapshot())),
            Request::CreateDevice { .. } => Ok(Response::Success),
            Request::AllocateMemory { .. } => Ok(Response::Success),
            Request::QueueSubmit { .. } => Ok(Response::Success),
            Request::WaitForFences { .. } => Ok(Response::Success),
            Request::DestroyResource { .. } => Ok(Response::Success),
            _ => Ok(Response::Success),
        }
    });

    let client = Client::new_with_transport(Box::new(mock));
    let snap = client.get_or_fetch_snapshot().unwrap();

    let mut instance_box = InstanceDispatchable::new(Arc::clone(&client), snap);
    let instance_handle = InstanceDispatchable::to_vk(&mut *instance_box as *mut _);

    // Enumerate Physical Devices
    let mut phys_count = 0u32;
    unsafe {
        let res = vkremote_icd::dispatch::vkEnumeratePhysicalDevices(
            instance_handle,
            &mut phys_count,
            std::ptr::null_mut(),
        );
        assert_eq!(res, ash::vk::Result::SUCCESS);
        assert_eq!(phys_count, 1);

        let mut phys_handle = ash::vk::PhysicalDevice::null();
        let res = vkremote_icd::dispatch::vkEnumeratePhysicalDevices(
            instance_handle,
            &mut phys_count,
            &mut phys_handle,
        );
        assert_eq!(res, ash::vk::Result::SUCCESS);
        assert!(!phys_handle.is_null());

        // Check device name formatting "<Original Name> (Remote)"
        let mut props = ash::vk::PhysicalDeviceProperties::default();
        vkremote_icd::dispatch::vkGetPhysicalDeviceProperties(phys_handle, &mut props);
        let name = std::ffi::CStr::from_ptr(props.device_name.as_ptr())
            .to_str()
            .unwrap();
        assert_eq!(name, "RTX 4090 (Remote)");

        // Create Device
        let create_info = ash::vk::DeviceCreateInfo::default();
        let mut device_handle = ash::vk::Device::null();
        let res = vkremote_icd::dispatch::vkCreateDevice(
            phys_handle,
            &create_info,
            std::ptr::null(),
            &mut device_handle,
        );
        assert_eq!(res, ash::vk::Result::SUCCESS);
        assert!(!device_handle.is_null());

        // Allocate host-visible memory
        let alloc_size = 128 * 1024; // 128 KB (2 x 64KB chunks)
        let alloc_info = ash::vk::MemoryAllocateInfo::default()
            .allocation_size(alloc_size)
            .memory_type_index(0);

        let mut mem_handle = ash::vk::DeviceMemory::null();
        let res = vkremote_icd::dispatch::vkAllocateMemory(
            device_handle,
            &alloc_info,
            std::ptr::null(),
            &mut mem_handle,
        );
        assert_eq!(res, ash::vk::Result::SUCCESS);
        assert!(!mem_handle.is_null());

        // Map memory
        let mut mapped_ptr = std::ptr::null_mut();
        let res = vkremote_icd::dispatch::vkMapMemory(
            device_handle,
            mem_handle,
            0,
            ash::vk::WHOLE_SIZE,
            ash::vk::MemoryMapFlags::empty(),
            &mut mapped_ptr,
        );
        assert_eq!(res, ash::vk::Result::SUCCESS);
        assert!(!mapped_ptr.is_null());

        // Initially no diffs since all zeros
        let diffs = client.collect_dirty_memory();
        assert_eq!(diffs.len(), 0); // Initially zero diffs since untouched zeros

        // Second time without write: zero dirty chunks
        let diffs = client.collect_dirty_memory();
        assert_eq!(diffs.len(), 0);

        // Write to chunk 0
        let slice = std::slice::from_raw_parts_mut(mapped_ptr as *mut u8, alloc_size as usize);
        slice[10..20].copy_from_slice(&[42u8; 10]);

        // Only chunk 0 is dirty
        let diffs = client.collect_dirty_memory();
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].offset, 0);

        // Next check is clean
        let diffs = client.collect_dirty_memory();
        assert_eq!(diffs.len(), 0);

        // Write to chunk 1 (offset 65536)
        slice[70000] = 77;
        let diffs = client.collect_dirty_memory();
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].offset, 64 * 1024);

        // Server updates memory diffs (e.g. compute output)
        let remote_diff = MemoryChunkDiff {
            allocation_id: mem_handle.as_raw(),
            offset: 0,
            data: vec![99u8; 64 * 1024],
        };
        client.apply_memory_diffs(&[remote_diff]);

        // Verify local shadow memory got the server data
        assert_eq!(slice[0], 99);
        assert_eq!(slice[100], 99);

        // Clean up
        vkremote_icd::dispatch::vkUnmapMemory(device_handle, mem_handle);
        vkremote_icd::dispatch::vkFreeMemory(device_handle, mem_handle, std::ptr::null());
        vkremote_icd::dispatch::vkDestroyDevice(device_handle, std::ptr::null());
    }

    assert!(requests_received.load(Ordering::SeqCst) > 0);
}

#[test]
fn test_command_recording_defers_transport_until_command_buffer_end() {
    let requests_received = Arc::new(AtomicUsize::new(0));
    let request_counter = Arc::clone(&requests_received);
    let client = Client::new_with_transport(Box::new(MockTransport::new(move |_| {
        request_counter.fetch_add(1, Ordering::SeqCst);
        Ok(Response::Success)
    })));

    client.begin_command_recording(41);
    assert!(client.record_command(
        41,
        Request::CmdDispatch {
            buffer_id: 41,
            group_count_x: 1,
            group_count_y: 2,
            group_count_z: 3,
        }
    ));
    assert_eq!(requests_received.load(Ordering::SeqCst), 0);

    let commands = client.finish_command_recording(41).unwrap();
    assert_eq!(commands.len(), 1);
    assert!(matches!(commands[0], Request::CmdDispatch { .. }));
    assert!(!client.record_command(41, Request::CmdEndRenderPass { buffer_id: 41 }));
}

#[test]
fn test_generate_icd_json() {
    let p_linux = Path::new("/usr/lib/libvkremote_icd.so");
    let json_linux = generate_icd_json(p_linux);
    assert!(json_linux.contains("\"library_path\": \"/usr/lib/libvkremote_icd.so\""));
    assert!(json_linux.contains("\"file_format_version\": \"1.0.0\""));
    assert!(json_linux.contains("\"api_version\": \"1.3.0\""));

    let p_win = Path::new(r"C:\Windows\System32\vkremote_icd.dll");
    let json_win = generate_icd_json(p_win);
    assert!(json_win.contains("\"library_path\": \"C:/Windows/System32/vkremote_icd.dll\""));
}

#[test]
fn test_bind_buffer_and_image_memory_are_batched() {
    let batch_counts = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = batch_counts.clone();
    let client = Client::new_with_transport(Box::new(MockTransport::new(move |req| {
        match req {
            Request::CommandBatch { commands } => {
                observed.lock().unwrap().push(commands.len());
                Ok(Response::Success)
            }
            Request::BeginCommandBuffer { .. } => Ok(Response::Success),
            other => panic!("Unexpected request: {:?}", std::mem::discriminant(other)),
        }
    })));

    // Send 10 BindBufferMemory requests
    for i in 0..10 {
        let res = client.send_request(Request::BindBufferMemory {
            buffer_id: i,
            memory_id: 100,
            offset: i * 64,
        });
        assert!(matches!(res, Ok(Response::Success)));
    }
    // None should be sent yet
    assert_eq!(batch_counts.lock().unwrap().len(), 0);

    // Send a non-deferrable request (BeginCommandBuffer) which must trigger flush
    let _ = client.send_request(Request::BeginCommandBuffer { buffer_id: 999 });

    // Verify exactly 1 batch of 10 commands was flushed
    let counts = batch_counts.lock().unwrap().clone();
    assert_eq!(counts, vec![10]);
}
