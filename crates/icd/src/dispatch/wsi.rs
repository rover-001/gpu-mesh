use crate::abi::{DeviceDispatchable, InstanceDispatchable, QueueDispatchable};
use crate::client::{SurfaceInfo, WindowHandle};
use ash::vk::Handle;
use vkremote_wire::types::{FramePacket, Request, ResourceType, Response};

#[cfg(target_os = "linux")]
#[path = "linux_present.rs"]
mod linux_present;

#[cfg(target_os = "windows")]
unsafe fn blit_to_window(hwnd: ash::vk::HWND, frame: &FramePacket) {
    if hwnd == 0 {
        return;
    }

    #[repr(C)]
    struct BITMAPINFOHEADER {
        bi_size: u32,
        bi_width: i32,
        bi_height: i32,
        bi_planes: u16,
        bi_bit_count: u16,
        bi_compression: u32,
        bi_size_image: u32,
        bi_x_pels_per_meter: i32,
        bi_y_pels_per_meter: i32,
        bi_clr_used: u32,
        bi_clr_important: u32,
    }

    #[repr(C)]
    struct BITMAPINFO {
        bmi_header: BITMAPINFOHEADER,
        bmi_colors: [u32; 1],
    }

    #[link(name = "user32")]
    #[link(name = "gdi32")]
    extern "system" {
        fn GetDC(hwnd: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
        fn ReleaseDC(hwnd: *mut std::ffi::c_void, hdc: *mut std::ffi::c_void) -> i32;
        fn StretchDIBits(
            hdc: *mut std::ffi::c_void,
            x_dst: i32,
            y_dst: i32,
            w_dst: i32,
            h_dst: i32,
            x_src: i32,
            y_src: i32,
            w_src: i32,
            h_src: i32,
            lpv_bits: *const std::ffi::c_void,
            lpbmi: *const BITMAPINFO,
            color_use: u32,
            rop: u32,
        ) -> i32;
    }

    let hdc = GetDC(hwnd as *mut std::ffi::c_void);
    if !hdc.is_null() {
        let mut bmi: BITMAPINFO = std::mem::zeroed();
        bmi.bmi_header.bi_size = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        bmi.bmi_header.bi_width = frame.width as i32;
        bmi.bmi_header.bi_height = -(frame.height as i32); // top-down
        bmi.bmi_header.bi_planes = 1;
        bmi.bmi_header.bi_bit_count = 32;
        bmi.bmi_header.bi_compression = 0; // BI_RGB
        bmi.bmi_header.bi_size_image = (frame.width * frame.height * 4) as u32;

        // Scale to the whole client area (equal sizes are a 1:1 copy) so a window
        // resized before the app recreates its swapchain is still fully covered.
        let (dst_w, dst_h) = win32_client_size(hwnd).unwrap_or((frame.width, frame.height));
        StretchDIBits(
            hdc,
            0,
            0,
            dst_w as i32,
            dst_h as i32,
            0,
            0,
            frame.width as i32,
            frame.height as i32,
            frame.data.as_ptr() as *const std::ffi::c_void,
            &bmi,
            0,          // DIB_RGB_COLORS
            0x00CC0020, // SRCCOPY
        );
        ReleaseDC(hwnd as *mut std::ffi::c_void, hdc);
    }
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
unsafe fn blit_to_window(_hwnd: ash::vk::HWND, _frame: &FramePacket) {}

fn present_frame_to_window(surface: &SurfaceInfo, frame: &FramePacket) -> ash::vk::Result {
    let _perf_span = crate::perf::Span::new(&crate::perf::BLIT_US);
    #[cfg(target_os = "linux")]
    {
        let result = match surface.window {
            WindowHandle::Xlib { display, window } => unsafe { linux_present::present_xlib(display, window, frame) },
            WindowHandle::Xcb { connection, window } => unsafe { linux_present::present_xcb(connection, window, frame) },
            WindowHandle::Wayland { .. } => Err("native Wayland frame presentation is not implemented; launch with WINIT_UNIX_BACKEND=x11"),
            _ => Err("unsupported Linux window handle"),
        };
        if let Err(error) = result {
            eprintln!("[vkremote-icd] Frame presentation failed: {error}");
            return ash::vk::Result::ERROR_SURFACE_LOST_KHR;
        }
        static PRESENT_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let count = PRESENT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count % 60 == 0 {
            let non_zero_rgb = frame
                .data
                .chunks_exact(4)
                .filter(|px| px[0] != 0 || px[1] != 0 || px[2] != 0)
                .count();
            let total_pixels = (frame.width as usize) * (frame.height as usize);
            eprintln!(
                "[vkremote-icd] Frame #{} presented: {}x{} (non-zero RGB pixels: {}/{} [{:.1}%])",
                count,
                frame.width,
                frame.height,
                non_zero_rgb,
                total_pixels,
                (non_zero_rgb as f64 / total_pixels.max(1) as f64) * 100.0
            );
        }
        return ash::vk::Result::SUCCESS;
    }
    #[cfg(not(target_os = "linux"))]
    {
        match surface.window {
            WindowHandle::Win32 { hwnd, .. } => unsafe {
                blit_to_window(hwnd, frame);
            },
            WindowHandle::Wayland {
                display: wl_display,
                surface: wl_surf,
            } => {
                tracing::info!(
                "[vkremote-icd] Presented frame {}x{} ({} bytes) to Wayland surface 0x{:x} (display 0x{:x})",
                frame.width, frame.height, frame.data.len(), wl_surf as usize, wl_display as usize
            );
            }
            WindowHandle::Xcb { connection, window } => {
                tracing::info!(
                "[vkremote-icd] Presented frame {}x{} ({} bytes) to XCB window 0x{:x} (conn 0x{:x})",
                frame.width, frame.height, frame.data.len(), window as usize, connection as usize
            );
            }
            WindowHandle::Xlib {
                display: xlib_display,
                window,
            } => {
                tracing::info!(
                "[vkremote-icd] Presented frame {}x{} ({} bytes) to Xlib window 0x{:x} (display 0x{:x})",
                frame.width, frame.height, frame.data.len(), window as usize, xlib_display as usize
            );
            }
        }
        ash::vk::Result::SUCCESS
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateWin32SurfaceKHR(
    instance: ash::vk::Instance,
    p_create_info: *const ash::vk::Win32SurfaceCreateInfoKHR,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_surface: *mut ash::vk::SurfaceKHR,
) -> ash::vk::Result {
    if instance.is_null() || p_create_info.is_null() || p_surface.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let inst = InstanceDispatchable::from_vk(instance);
    let info = &*p_create_info;
    let surface_id = inst.client.next_id();
    eprintln!("[vkremote-icd] vkCreateWaylandSurfaceKHR surface_id={}", surface_id);

    inst.client.register_surface(
        surface_id,
        WindowHandle::Win32 {
            hwnd: info.hwnd,
            hinstance: info.hinstance,
        },
    );

    *p_surface = ash::vk::SurfaceKHR::from_raw(surface_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateWaylandSurfaceKHR(
    instance: ash::vk::Instance,
    p_create_info: *const ash::vk::WaylandSurfaceCreateInfoKHR,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_surface: *mut ash::vk::SurfaceKHR,
) -> ash::vk::Result {
    if instance.is_null() || p_create_info.is_null() || p_surface.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let inst = InstanceDispatchable::from_vk(instance);
    let info = &*p_create_info;
    let surface_id = inst.client.next_id();
    eprintln!("[vkremote-icd] vkCreateWaylandSurfaceKHR surface_id={}", surface_id);

    inst.client.register_surface(
        surface_id,
        WindowHandle::Wayland {
            display: info.display as usize,
            surface: info.surface as usize,
        },
    );

    *p_surface = ash::vk::SurfaceKHR::from_raw(surface_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateXcbSurfaceKHR(
    instance: ash::vk::Instance,
    p_create_info: *const ash::vk::XcbSurfaceCreateInfoKHR,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_surface: *mut ash::vk::SurfaceKHR,
) -> ash::vk::Result {
    if instance.is_null() || p_create_info.is_null() || p_surface.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let inst = InstanceDispatchable::from_vk(instance);
    let info = &*p_create_info;
    let surface_id = inst.client.next_id();
    eprintln!("[vkremote-icd] vkCreateWaylandSurfaceKHR surface_id={}", surface_id);

    inst.client.register_surface(
        surface_id,
        WindowHandle::Xcb {
            connection: info.connection as usize,
            window: info.window,
        },
    );

    *p_surface = ash::vk::SurfaceKHR::from_raw(surface_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateXlibSurfaceKHR(
    instance: ash::vk::Instance,
    p_create_info: *const ash::vk::XlibSurfaceCreateInfoKHR,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_surface: *mut ash::vk::SurfaceKHR,
) -> ash::vk::Result {
    if instance.is_null() || p_create_info.is_null() || p_surface.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let inst = InstanceDispatchable::from_vk(instance);
    let info = &*p_create_info;
    let surface_id = inst.client.next_id();
    eprintln!("[vkremote-icd] vkCreateWaylandSurfaceKHR surface_id={}", surface_id);

    inst.client.register_surface(
        surface_id,
        WindowHandle::Xlib {
            display: info.dpy as usize,
            window: info.window as usize,
        },
    );

    *p_surface = ash::vk::SurfaceKHR::from_raw(surface_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroySurfaceKHR(
    instance: ash::vk::Instance,
    surface: ash::vk::SurfaceKHR,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if instance.is_null() || surface.is_null() {
        return;
    }
    let inst = InstanceDispatchable::from_vk(instance);
    inst.client.unregister_surface(surface.as_raw());
    let _ = inst.client.send_request(Request::DestroyResource {
        handle_id: surface.as_raw(),
        resource_type: ResourceType::Surface,
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceSurfaceSupportKHR(
    _physical_device: ash::vk::PhysicalDevice,
    _queue_family_index: u32,
    _surface: ash::vk::SurfaceKHR,
    p_supported: *mut ash::vk::Bool32,
) -> ash::vk::Result {
    if p_supported.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    *p_supported = ash::vk::TRUE;
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceWin32PresentationSupportKHR(
    _physical_device: ash::vk::PhysicalDevice,
    _queue_family_index: u32,
) -> ash::vk::Bool32 {
    ash::vk::TRUE
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceXlibPresentationSupportKHR(
    _physical_device: ash::vk::PhysicalDevice,
    _queue_family_index: u32,
    _dpy: *mut std::ffi::c_void,
    _visual_id: usize,
) -> ash::vk::Bool32 {
    ash::vk::TRUE
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceXcbPresentationSupportKHR(
    _physical_device: ash::vk::PhysicalDevice,
    _queue_family_index: u32,
    _connection: *mut std::ffi::c_void,
    _visual_id: u32,
) -> ash::vk::Bool32 {
    ash::vk::TRUE
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceWaylandPresentationSupportKHR(
    _physical_device: ash::vk::PhysicalDevice,
    _queue_family_index: u32,
    _display: *mut std::ffi::c_void,
) -> ash::vk::Bool32 {
    ash::vk::TRUE
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
    physical_device: ash::vk::PhysicalDevice,
    surface: ash::vk::SurfaceKHR,
    p_surface_capabilities: *mut ash::vk::SurfaceCapabilitiesKHR,
) -> ash::vk::Result {
    if p_surface_capabilities.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let caps = &mut *p_surface_capabilities;
    caps.min_image_count = 2;
    caps.max_image_count = 8;
    caps.current_extent = ash::vk::Extent2D {
        width: u32::MAX,
        height: u32::MAX,
    };
    caps.min_image_extent = ash::vk::Extent2D {
        width: 1,
        height: 1,
    };
    caps.max_image_extent = ash::vk::Extent2D {
        width: 4096,
        height: 4096,
    };
    caps.max_image_array_layers = 1;
    caps.supported_transforms = ash::vk::SurfaceTransformFlagsKHR::IDENTITY;
    caps.current_transform = ash::vk::SurfaceTransformFlagsKHR::IDENTITY;
    caps.supported_composite_alpha =
        ash::vk::CompositeAlphaFlagsKHR::OPAQUE | ash::vk::CompositeAlphaFlagsKHR::INHERIT;
    caps.supported_usage_flags = ash::vk::ImageUsageFlags::COLOR_ATTACHMENT
        | ash::vk::ImageUsageFlags::TRANSFER_SRC
        | ash::vk::ImageUsageFlags::TRANSFER_DST;

    // Win32 requires currentExtent to be the window's client size (0xFFFFFFFF is a
    // Wayland-only convention); otherwise apps fall back to a default such as 1280x720
    // and a fullscreen window shows the image only in its top-left corner.
    #[cfg(target_os = "windows")]
    if !physical_device.is_null() {
        let phys = crate::abi::PhysicalDeviceDispatchable::from_vk(physical_device);
        if let Some(info) = phys.client.get_surface(surface.as_raw()) {
            if let WindowHandle::Win32 { hwnd, .. } = info.window {
                if let Some((w, h)) = win32_client_size(hwnd) {
                    let extent = ash::vk::Extent2D { width: w, height: h };
                    caps.current_extent = extent;
                    caps.min_image_extent = extent;
                    caps.max_image_extent = extent;
                }
            }
        }
    }
    #[cfg(not(target_os = "windows"))]
    let _ = (physical_device, surface);

    ash::vk::Result::SUCCESS
}

#[cfg(target_os = "windows")]
fn win32_client_size(hwnd: ash::vk::HWND) -> Option<(u32, u32)> {
    #[repr(C)]
    struct RECT {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }
    #[link(name = "user32")]
    extern "system" {
        fn GetClientRect(hwnd: *mut std::ffi::c_void, rect: *mut RECT) -> i32;
    }
    if hwnd == 0 {
        return None;
    }
    let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
    // SAFETY: hwnd is the window the app created the surface for.
    if unsafe { GetClientRect(hwnd as *mut std::ffi::c_void, &mut r) } == 0 {
        return None;
    }
    let (w, h) = ((r.right - r.left).max(1) as u32, (r.bottom - r.top).max(1) as u32);
    Some((w, h))
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceSurfaceFormatsKHR(
    _physical_device: ash::vk::PhysicalDevice,
    _surface: ash::vk::SurfaceKHR,
    p_surface_format_count: *mut u32,
    p_surface_formats: *mut ash::vk::SurfaceFormatKHR,
) -> ash::vk::Result {
    if p_surface_format_count.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let formats = [
        ash::vk::SurfaceFormatKHR {
            format: ash::vk::Format::B8G8R8A8_UNORM,
            color_space: ash::vk::ColorSpaceKHR::SRGB_NONLINEAR,
        },
        ash::vk::SurfaceFormatKHR {
            format: ash::vk::Format::R8G8B8A8_UNORM,
            color_space: ash::vk::ColorSpaceKHR::SRGB_NONLINEAR,
        },
        ash::vk::SurfaceFormatKHR {
            format: ash::vk::Format::B8G8R8A8_SRGB,
            color_space: ash::vk::ColorSpaceKHR::SRGB_NONLINEAR,
        },
        ash::vk::SurfaceFormatKHR {
            format: ash::vk::Format::R8G8B8A8_SRGB,
            color_space: ash::vk::ColorSpaceKHR::SRGB_NONLINEAR,
        },
    ];

    if p_surface_formats.is_null() {
        *p_surface_format_count = formats.len() as u32;
        return ash::vk::Result::SUCCESS;
    }

    let count = (*p_surface_format_count as usize).min(formats.len());
    for i in 0..count {
        *p_surface_formats.add(i) = formats[i];
    }

    if count < formats.len() {
        ash::vk::Result::INCOMPLETE
    } else {
        ash::vk::Result::SUCCESS
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetPhysicalDeviceSurfacePresentModesKHR(
    _physical_device: ash::vk::PhysicalDevice,
    _surface: ash::vk::SurfaceKHR,
    p_present_mode_count: *mut u32,
    p_present_modes: *mut ash::vk::PresentModeKHR,
) -> ash::vk::Result {
    if p_present_mode_count.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let modes = [
        ash::vk::PresentModeKHR::FIFO,
        ash::vk::PresentModeKHR::MAILBOX,
        ash::vk::PresentModeKHR::IMMEDIATE,
    ];

    if p_present_modes.is_null() {
        *p_present_mode_count = modes.len() as u32;
        return ash::vk::Result::SUCCESS;
    }

    let count = (*p_present_mode_count as usize).min(modes.len());
    for i in 0..count {
        *p_present_modes.add(i) = modes[i];
    }

    if count < modes.len() {
        ash::vk::Result::INCOMPLETE
    } else {
        ash::vk::Result::SUCCESS
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkCreateSwapchainKHR(
    device: ash::vk::Device,
    p_create_info: *const ash::vk::SwapchainCreateInfoKHR,
    _p_allocator: *const ash::vk::AllocationCallbacks,
    p_swapchain: *mut ash::vk::SwapchainKHR,
) -> ash::vk::Result {
    if device.is_null() || p_create_info.is_null() || p_swapchain.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let dev = DeviceDispatchable::from_vk(device);
    let info = &*p_create_info;
    let swapchain_id = dev.client.next_id();
    eprintln!(
        "[vkremote-icd] vkCreateSwapchainKHR: {}x{} format={:?}",
        info.image_extent.width, info.image_extent.height, info.image_format
    );

    let image_count = info.min_image_count.clamp(2, 4);

    let params = vkremote_wire::types::CreateSwapchainParams {
        min_image_count: image_count,
        image_format: (info.image_format.as_raw() as u32).into(),
        image_color_space: (info.image_color_space.as_raw() as u32).into(),
        image_extent: vkremote_wire::types::Extent2D {
            width: info.image_extent.width,
            height: info.image_extent.height,
        },
        image_array_layers: info.image_array_layers.max(1),
        image_usage: info.image_usage.as_raw().into(),
        image_sharing_mode: (info.image_sharing_mode.as_raw() as u32).into(),
        queue_family_indices: Vec::new(),
        pre_transform: info.pre_transform.as_raw().into(),
        composite_alpha: info.composite_alpha.as_raw().into(),
        present_mode: (info.present_mode.as_raw() as u32).into(),
        clipped: info.clipped != 0,
        old_swapchain_id: if info.old_swapchain.as_raw() != 0 {
            Some(info.old_swapchain.as_raw())
        } else {
            None
        },
    };
    let req = Request::CreateSwapchain {
        swapchain_id,
        surface_id: info.surface.as_raw(),
        params,
    };
    let res = dev.client.send_request(req);
    eprintln!("[vkremote-icd] vkCreateSwapchainKHR res: {:?}", res);

    let final_image_ids = match &res {
        Ok(Response::SwapchainCreated { image_ids, .. }) => image_ids.clone(),
        _ => {
            let mut ids = Vec::with_capacity(image_count as usize);
            for _ in 0..image_count {
                ids.push(dev.client.next_id());
            }
            ids
        }
    };

    for &img_id in &final_image_ids {
        dev.client.register_image(
            img_id,
            info.image_format,
            ash::vk::Extent3D {
                width: info.image_extent.width,
                height: info.image_extent.height,
                depth: 1,
            },
            1,
            info.image_array_layers.max(1),
            info.image_usage,
        );
    }

    dev.client.register_swapchain(
        swapchain_id,
        info.surface.as_raw(),
        info.image_extent.width,
        info.image_extent.height,
        info.image_format,
        final_image_ids,
    );

    *p_swapchain = ash::vk::SwapchainKHR::from_raw(swapchain_id);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkDestroySwapchainKHR(
    device: ash::vk::Device,
    swapchain: ash::vk::SwapchainKHR,
    _p_allocator: *const ash::vk::AllocationCallbacks,
) {
    if device.is_null() || swapchain.is_null() {
        return;
    }
    let dev = DeviceDispatchable::from_vk(device);
    let sc_id = swapchain.as_raw();

    if let Some(imgs) = dev.client.get_swapchain_images(sc_id) {
        for img in imgs {
            dev.client.unregister_image(img);
        }
    }
    dev.client.unregister_swapchain(sc_id);

    let _ = dev.client.send_request(Request::DestroySwapchain {
        swapchain_id: sc_id,
    });
}

#[no_mangle]
pub unsafe extern "system" fn vkGetSwapchainImagesKHR(
    device: ash::vk::Device,
    swapchain: ash::vk::SwapchainKHR,
    p_swapchain_image_count: *mut u32,
    p_swapchain_images: *mut ash::vk::Image,
) -> ash::vk::Result {
    if device.is_null() || swapchain.is_null() || p_swapchain_image_count.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let dev = DeviceDispatchable::from_vk(device);
    let sc_id = swapchain.as_raw();
    let image_ids = dev.client.get_swapchain_images(sc_id).unwrap_or_default();

    if p_swapchain_images.is_null() {
        *p_swapchain_image_count = image_ids.len() as u32;
        return ash::vk::Result::SUCCESS;
    }

    let count = (*p_swapchain_image_count as usize).min(image_ids.len());
    for i in 0..count {
        *p_swapchain_images.add(i) = ash::vk::Image::from_raw(image_ids[i]);
    }

    if count < image_ids.len() {
        ash::vk::Result::INCOMPLETE
    } else {
        ash::vk::Result::SUCCESS
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkAcquireNextImageKHR(
    device: ash::vk::Device,
    swapchain: ash::vk::SwapchainKHR,
    timeout: u64,
    semaphore: ash::vk::Semaphore,
    fence: ash::vk::Fence,
    p_image_index: *mut u32,
) -> ash::vk::Result {
    if device.is_null() || swapchain.is_null() || p_image_index.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let dev = DeviceDispatchable::from_vk(device);
    let sc_id = swapchain.as_raw();

    let next_index = dev.client.acquire_next_image_index(sc_id).unwrap_or(0);
    *p_image_index = next_index;

    let req = Request::AcquireNextImage {
        swapchain_id: sc_id,
        timeout_ns: timeout,
        semaphore_id: if semaphore.is_null() {
            None
        } else {
            Some(semaphore.as_raw())
        },
        fence_id: if fence.is_null() {
            None
        } else {
            Some(fence.as_raw())
        },
    };
    let _ = dev.client.send_request(req);

    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vkQueuePresentKHR(
    queue: ash::vk::Queue,
    p_present_info: *const ash::vk::PresentInfoKHR,
) -> ash::vk::Result {
    if queue.is_null() || p_present_info.is_null() {
        return ash::vk::Result::ERROR_DEVICE_LOST;
    }

    let present_span = crate::perf::Span::new(&crate::perf::PRESENT_US);
    let q = QueueDispatchable::from_vk(queue);
    let info = &*p_present_info;
    eprintln!(
        "[vkremote-icd] vkQueuePresentKHR: swapchain_count={}",
        info.swapchain_count
    );

    let mut wait_semaphores = Vec::new();
    if !info.p_wait_semaphores.is_null() && info.wait_semaphore_count > 0 {
        for i in 0..info.wait_semaphore_count as usize {
            let s = *info.p_wait_semaphores.add(i);
            if !s.is_null() {
                wait_semaphores.push(s.as_raw());
            }
        }
    }

    let mut overall_result = ash::vk::Result::SUCCESS;

    if !info.p_swapchains.is_null() && !info.p_image_indices.is_null() && info.swapchain_count > 0 {
        for i in 0..info.swapchain_count as usize {
            let sc_handle = *info.p_swapchains.add(i);
            let image_index = *info.p_image_indices.add(i);

            let params = vkremote_wire::types::PresentParams {
                swapchain_id: sc_handle.as_raw(),
                image_index,
                wait_semaphores: wait_semaphores.clone(),
            };
            let req = Request::QueuePresent {
                queue_id: q.id,
                params,
            };

            let mut item_result = ash::vk::Result::SUCCESS;

            match q.client.send_request(req) {
                Ok(Response::QueuePresentResult { result, frame }) => {
                    if result != 0 {
                        item_result = ash::vk::Result::from_raw(result);
                    }
                    if let Some(frame_data) = frame {
                        if let Some(surface_id) =
                            q.client.get_swapchain_surface_id(sc_handle.as_raw())
                        {
                            if let Some(surface) = q.client.get_surface(surface_id) {
                                item_result = present_frame_to_window(&surface, &frame_data);
                            }
                        }
                    }
                }
                Ok(Response::Success) => {}
                Ok(Response::Error { code, message }) => {
                    tracing::error!("QueuePresent error: {} ({})", message, code);
                    item_result = ash::vk::Result::ERROR_DEVICE_LOST;
                }
                Err(e) => {
                    tracing::error!("QueuePresent network error: {}", e);
                    item_result = ash::vk::Result::ERROR_DEVICE_LOST;
                }
                _ => {}
            }

            if !info.p_results.is_null() {
                *info.p_results.add(i) = item_result;
            }

            if item_result != ash::vk::Result::SUCCESS && overall_result == ash::vk::Result::SUCCESS
            {
                overall_result = item_result;
            }
        }
    }

    drop(present_span);
    crate::perf::end_frame();
    overall_result
}
