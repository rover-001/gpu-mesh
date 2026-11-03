//! CPU presentation of remote frames to X11/Xwayland windows.
use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::sync::Mutex;
use vkremote_wire::types::{Format, FramePacket};

static WINDOW_CACHE: Mutex<Option<HashMap<(usize, u32), (u32, u8)>>> = Mutex::new(None);

#[repr(C)]
#[derive(Clone, Copy)]
struct Cookie {
    sequence: u32,
}

#[repr(C)]
struct Geometry {
    response_type: u8,
    depth: u8,
    sequence: u16,
    length: u32,
    root: u32,
    x: i16,
    y: i16,
    width: u16,
    height: u16,
    border: u16,
    pad: [u8; 2],
}

#[link(name = "xcb")]
#[allow(dead_code)]
extern "C" {
    fn xcb_generate_id(connection: *mut c_void) -> u32;
    fn xcb_get_geometry(connection: *mut c_void, drawable: u32) -> Cookie;
    fn xcb_get_geometry_reply(
        connection: *mut c_void,
        cookie: Cookie,
        error: *mut *mut c_void,
    ) -> *mut Geometry;
    fn xcb_create_gc(
        connection: *mut c_void,
        gc: u32,
        drawable: u32,
        mask: u32,
        values: *const c_void,
    ) -> Cookie;
    fn xcb_free_gc(connection: *mut c_void, gc: u32) -> Cookie;
    fn xcb_put_image(
        connection: *mut c_void,
        format: u8,
        drawable: u32,
        gc: u32,
        width: u16,
        height: u16,
        x: i16,
        y: i16,
        pad: u8,
        depth: u8,
        len: u32,
        data: *const u8,
    ) -> Cookie;
    fn xcb_request_check(connection: *mut c_void, cookie: Cookie) -> *mut c_void;
    fn xcb_flush(connection: *mut c_void) -> c_int;
}

#[link(name = "X11-xcb")]
extern "C" {
    fn XGetXCBConnection(display: *mut c_void) -> *mut c_void;
}

fn pixels(frame: &FramePacket) -> Result<Vec<u8>, &'static str> {
    if frame.width == 0 || frame.height == 0 || frame.width > 32767 || frame.height > 32767 {
        return Err("invalid X11 frame dimensions");
    }
    let expected = (frame.width as usize)
        .checked_mul(frame.height as usize)
        .and_then(|size| size.checked_mul(4))
        .ok_or("frame size overflow")?;
    if frame.data.len() != expected {
        return Err("invalid X11 frame payload size");
    }
    let mut data = frame.data.clone();
    match frame.format {
        Format::B8G8R8A8Unorm | Format::B8G8R8A8Srgb => {}
        Format::R8G8B8A8Unorm | Format::R8G8B8A8Srgb => {
            for pixel in data.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        _ => return Err("unsupported X11 frame format"),
    }
    // A depth-32 Xwayland window is composited with its alpha channel. Remote
    // swapchains are opaque even when rendering leaves their alpha undefined.
    let u32_slice: &mut [u32] = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u32, data.len() / 4)
    };
    for pixel in u32_slice.iter_mut() {
        *pixel |= 0xFF00_0000;
    }
    Ok(data)
}

/// Present to a live XCB window. The caller owns the connection and window.
pub(super) unsafe fn present_xcb(
    connection: usize,
    window: u32,
    frame: &FramePacket,
) -> Result<(), &'static str> {
    if connection == 0 || window == 0 {
        return Err("invalid X11 window handle");
    }
    if cfg!(target_endian = "big") {
        return Err("X11 frame presentation requires little-endian pixels");
    }
    let data = pixels(frame)?;
    let connection = connection as *mut c_void;

    let (gc, depth) = {
        let mut guard = WINDOW_CACHE.lock().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        let key = (connection as usize, window);
        if let Some(&cached) = map.get(&key) {
            cached
        } else {
            let geometry = xcb_get_geometry_reply(
                connection,
                xcb_get_geometry(connection, window),
                std::ptr::null_mut(),
            );
            if geometry.is_null() {
                return Err("cannot query X11 window geometry");
            }
            let depth = (*geometry).depth;
            libc::free(geometry.cast());
            if depth != 24 && depth != 32 {
                return Err("unsupported X11 window depth");
            }
            let gc = xcb_generate_id(connection);
            xcb_create_gc(connection, gc, window, 0, std::ptr::null());
            map.insert(key, (gc, depth));
            (gc, depth)
        }
    };

    // Keep every request below the core X11 256 KiB request limit. XPutImage
    // equivalents must split tall frames, even when BIG-REQUESTS is unavailable.
    let row_bytes = frame.width as usize * 4;
    let rows_per_request = ((256 * 1024 - 64) / row_bytes).max(1);
    for y in (0..frame.height as usize).step_by(rows_per_request) {
        let rows = rows_per_request.min(frame.height as usize - y);
        xcb_put_image(
            connection,
            2,
            window,
            gc,
            frame.width as u16,
            rows as u16,
            0,
            y as i16,
            0,
            depth,
            (row_bytes * rows) as u32,
            data.as_ptr().add(y * row_bytes),
        );
    }
    xcb_flush(connection);
    Ok(())
}

/// Present using the XCB connection backing a live Xlib display.
pub(super) unsafe fn present_xlib(
    display: usize,
    window: usize,
    frame: &FramePacket,
) -> Result<(), &'static str> {
    if display == 0 || window > u32::MAX as usize {
        return Err("invalid Xlib window handle");
    }
    present_xcb(
        XGetXCBConnection(display as *mut c_void) as usize,
        window as u32,
        frame,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame(format: Format, data: Vec<u8>) -> FramePacket {
        FramePacket {
            swapchain_id: 1,
            image_index: 0,
            width: 1,
            height: 1,
            format,
            data,
        }
    }
    #[test]
    fn converts_rgba_and_preserves_bgra_with_opaque_alpha() {
        assert_eq!(
            pixels(&frame(Format::R8G8B8A8Unorm, vec![255, 20, 10, 0])).unwrap(),
            vec![10, 20, 255, 255]
        );
        assert_eq!(
            pixels(&frame(Format::B8G8R8A8Unorm, vec![10, 20, 255, 0])).unwrap(),
            vec![10, 20, 255, 255]
        );
    }
    #[test]
    fn rejects_invalid_payloads() {
        assert!(pixels(&frame(Format::B8G8R8A8Unorm, vec![0; 3])).is_err());
        let mut oversized = frame(Format::B8G8R8A8Unorm, vec![]);
        oversized.width = u32::MAX;
        assert!(pixels(&oversized).is_err());
    }

    #[test]
    #[ignore = "requires a live X11 display"]
    fn x11_roundtrip_draws_large_frame_with_correct_colors() {
        #[link(name = "X11")]
        extern "C" {
            fn XOpenDisplay(name: *const i8) -> *mut c_void;
            fn XDefaultRootWindow(display: *mut c_void) -> libc::c_ulong;
            fn XDefaultScreen(display: *mut c_void) -> c_int;
            fn XDefaultDepth(display: *mut c_void, screen: c_int) -> c_int;
            fn XCreatePixmap(
                display: *mut c_void,
                drawable: libc::c_ulong,
                width: u32,
                height: u32,
                depth: u32,
            ) -> libc::c_ulong;
            fn XGetImage(
                display: *mut c_void,
                drawable: libc::c_ulong,
                x: c_int,
                y: c_int,
                width: u32,
                height: u32,
                mask: libc::c_ulong,
                format: c_int,
            ) -> *mut c_void;
            fn XGetPixel(image: *mut c_void, x: c_int, y: c_int) -> libc::c_ulong;
            fn XDestroyImage(image: *mut c_void) -> c_int;
            fn XFreePixmap(display: *mut c_void, pixmap: libc::c_ulong) -> c_int;
            fn XSync(display: *mut c_void, discard: c_int) -> c_int;
            fn XCloseDisplay(display: *mut c_void) -> c_int;
        }
        unsafe {
            let display = XOpenDisplay(std::ptr::null());
            assert!(!display.is_null(), "XOpenDisplay failed");
            let depth = XDefaultDepth(display, XDefaultScreen(display));
            let drawable =
                XCreatePixmap(display, XDefaultRootWindow(display), 1920, 80, depth as u32);
            XSync(display, 0);
            let frame = FramePacket {
                swapchain_id: 1,
                image_index: 0,
                width: 1920,
                height: 80,
                format: Format::R8G8B8A8Unorm,
                data: [255, 20, 10, 0].repeat(1920 * 80),
            };
            let result = present_xlib(display as usize, drawable as usize, &frame);
            XSync(display, 0);
            let image = XGetImage(display, drawable, 1919, 79, 1, 1, libc::c_ulong::MAX, 2);
            let pixel = if image.is_null() {
                0
            } else {
                XGetPixel(image, 0, 0)
            };
            if !image.is_null() {
                XDestroyImage(image);
            }
            XFreePixmap(display, drawable);
            XCloseDisplay(display);
            assert_eq!(result, Ok(()));
            assert_eq!(pixel & 0xffffff, 0xff140a);
        }
    }
}
