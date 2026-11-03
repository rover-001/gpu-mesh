//! Client-side shadow copy of a host-visible allocation (what vkMapMemory returns).
//!
//! On Windows the buffer is allocated with `MEM_WRITE_WATCH`, so the OS records
//! which pages the application wrote. `dirty_chunks` then lets the submit path hash
//! only those chunks instead of every mapped byte (Blender maps 130-200 MB and the
//! full scan cost ~26 ms per frame). Elsewhere, or if the allocation fails, it is a
//! plain zeroed heap buffer and `dirty_chunks` returns `None` (scan everything).

use std::ops::{Deref, DerefMut};

pub struct ShadowBuffer {
    ptr: *mut u8,
    len: usize,
    write_watch: bool,
    _heap: Option<Box<[u8]>>,
}

// SAFETY: the buffer is uniquely owned; access is serialized by the caller's Mutex.
unsafe impl Send for ShadowBuffer {}
unsafe impl Sync for ShadowBuffer {}

#[cfg(target_os = "windows")]
mod win {
    pub const MEM_COMMIT: u32 = 0x1000;
    pub const MEM_RESERVE: u32 = 0x2000;
    pub const MEM_WRITE_WATCH: u32 = 0x0020_0000;
    pub const MEM_RELEASE: u32 = 0x8000;
    pub const PAGE_READWRITE: u32 = 0x04;
    pub const WRITE_WATCH_FLAG_RESET: u32 = 0x01;

    #[link(name = "kernel32")]
    extern "system" {
        pub fn VirtualAlloc(
            addr: *mut std::ffi::c_void,
            size: usize,
            alloc_type: u32,
            protect: u32,
        ) -> *mut std::ffi::c_void;
        pub fn VirtualFree(addr: *mut std::ffi::c_void, size: usize, free_type: u32) -> i32;
        pub fn GetWriteWatch(
            flags: u32,
            base: *mut std::ffi::c_void,
            size: usize,
            addresses: *mut *mut std::ffi::c_void,
            count: *mut usize,
            granularity: *mut u32,
        ) -> u32;
    }
}

impl ShadowBuffer {
    /// Zero-initialised buffer of `len` bytes.
    pub fn new(len: usize) -> Self {
        #[cfg(target_os = "windows")]
        if len > 0 {
            // SAFETY: plain reservation+commit; VirtualAlloc returns zeroed pages.
            let p = unsafe {
                win::VirtualAlloc(
                    std::ptr::null_mut(),
                    len,
                    win::MEM_RESERVE | win::MEM_COMMIT | win::MEM_WRITE_WATCH,
                    win::PAGE_READWRITE,
                )
            };
            if !p.is_null() {
                return Self {
                    ptr: p as *mut u8,
                    len,
                    write_watch: true,
                    _heap: None,
                };
            }
        }
        let mut heap = vec![0u8; len].into_boxed_slice();
        Self {
            ptr: heap.as_mut_ptr(),
            len,
            write_watch: false,
            _heap: Some(heap),
        }
    }

    /// Offsets (multiples of `chunk`) of chunks written since the previous call,
    /// sorted and de-duplicated, resetting the OS write tracking. `None` means
    /// write tracking is unavailable and the caller must scan the whole buffer.
    pub fn dirty_chunks(&self, chunk: usize) -> Option<Vec<usize>> {
        if !self.write_watch || chunk == 0 {
            return None;
        }
        #[cfg(target_os = "windows")]
        {
            let max_pages = self.len / 4096 + 1;
            let mut addrs: Vec<*mut std::ffi::c_void> = vec![std::ptr::null_mut(); max_pages];
            let mut count = max_pages;
            let mut granularity = 0u32;
            // SAFETY: ptr/len describe our MEM_WRITE_WATCH region; addrs holds max_pages.
            let rc = unsafe {
                win::GetWriteWatch(
                    win::WRITE_WATCH_FLAG_RESET,
                    self.ptr as *mut std::ffi::c_void,
                    self.len,
                    addrs.as_mut_ptr(),
                    &mut count,
                    &mut granularity,
                )
            };
            if rc != 0 {
                return None;
            }
            let base = self.ptr as usize;
            let mut out: Vec<usize> = addrs[..count]
                .iter()
                .map(|&a| ((a as usize - base) / chunk) * chunk)
                .collect();
            out.dedup();
            return Some(out);
        }
        #[allow(unreachable_code)]
        None
    }
}

impl Deref for ShadowBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: ptr is valid for len bytes for the lifetime of self.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for ShadowBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid for len bytes and uniquely borrowed through &mut self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for ShadowBuffer {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        if self.write_watch {
            // SAFETY: ptr came from VirtualAlloc in new().
            unsafe {
                win::VirtualFree(self.ptr as *mut std::ffi::c_void, 0, win::MEM_RELEASE);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_and_writable() {
        let mut b = ShadowBuffer::new(100_000);
        assert!(b.iter().all(|&x| x == 0));
        b[99_999] = 7;
        assert_eq!(b[99_999], 7);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn write_watch_reports_written_chunks() {
        let mut b = ShadowBuffer::new(1 << 20);
        let _ = b.dirty_chunks(65536);
        b[70_000] = 1;
        assert_eq!(b.dirty_chunks(65536), Some(vec![65536]));
        assert_eq!(b.dirty_chunks(65536), Some(vec![]));
    }
}
