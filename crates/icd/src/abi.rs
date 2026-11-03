use crate::client::Client;
use ash::vk::Handle;
use std::sync::Arc;
use vkremote_wire::PhysicalDeviceSnapshot;

/// Khronos Vulkan Loader dispatchable handle requirement:
/// The first field MUST be `loader_data: usize` with `#[repr(C)]`.

#[repr(C)]
pub struct InstanceDispatchable {
    pub loader_data: usize,
    pub client: Arc<Client>,
    pub snapshot: Arc<PhysicalDeviceSnapshot>,
    /// The instance's single physical device, created on first enumeration.
    /// Vulkan requires vkEnumeratePhysicalDevices to return the same handle on every
    /// call; a fresh handle makes the loader free the wrapper the app still holds
    /// ("Invalid physicalDevice" abort in Blender). Freed by vkDestroyInstance.
    pub physical_device: std::sync::atomic::AtomicPtr<PhysicalDeviceDispatchable>,
}

impl InstanceDispatchable {
    pub fn new(client: Arc<Client>, snapshot: Arc<PhysicalDeviceSnapshot>) -> Box<Self> {
        Box::new(Self {
            loader_data: 0,
            client,
            snapshot,
            physical_device: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
        })
    }

    pub fn to_vk(ptr: *mut Self) -> ash::vk::Instance {
        ash::vk::Instance::from_raw(ptr as usize as u64)
    }

    /// SAFETY: caller must provide a valid non-null Vulkan instance handle created by this ICD.
    pub unsafe fn from_vk<'a>(handle: ash::vk::Instance) -> &'a Self {
        &*(handle.as_raw() as usize as *const Self)
    }

    /// SAFETY: caller must provide a valid non-null Vulkan instance handle created by this ICD.
    pub unsafe fn from_vk_mut<'a>(handle: ash::vk::Instance) -> &'a mut Self {
        &mut *(handle.as_raw() as usize as *mut Self)
    }
}

#[repr(C)]
pub struct PhysicalDeviceDispatchable {
    pub loader_data: usize,
    pub instance: *mut InstanceDispatchable,
    pub client: Arc<Client>,
    pub snapshot: Arc<PhysicalDeviceSnapshot>,
}

impl PhysicalDeviceDispatchable {
    pub fn new(
        instance: *mut InstanceDispatchable,
        client: Arc<Client>,
        snapshot: Arc<PhysicalDeviceSnapshot>,
    ) -> Box<Self> {
        Box::new(Self {
            loader_data: 0,
            instance,
            client,
            snapshot,
        })
    }

    pub fn to_vk(ptr: *mut Self) -> ash::vk::PhysicalDevice {
        ash::vk::PhysicalDevice::from_raw(ptr as usize as u64)
    }

    /// SAFETY: caller must provide a valid non-null Vulkan physical device handle created by this ICD.
    pub unsafe fn from_vk<'a>(handle: ash::vk::PhysicalDevice) -> &'a Self {
        &*(handle.as_raw() as usize as *const Self)
    }

    /// SAFETY: caller must provide a valid non-null Vulkan physical device handle created by this ICD.
    pub unsafe fn from_vk_mut<'a>(handle: ash::vk::PhysicalDevice) -> &'a mut Self {
        &mut *(handle.as_raw() as usize as *mut Self)
    }
}

#[repr(C)]
pub struct DeviceDispatchable {
    pub loader_data: usize,
    pub id: u64,
    pub client: Arc<Client>,
    pub physical_device: *mut PhysicalDeviceDispatchable,
    pub queue: *mut QueueDispatchable,
}

impl DeviceDispatchable {
    pub fn new(
        id: u64,
        client: Arc<Client>,
        physical_device: *mut PhysicalDeviceDispatchable,
    ) -> Box<Self> {
        Box::new(Self {
            loader_data: 0,
            id,
            client,
            physical_device,
            queue: std::ptr::null_mut(),
        })
    }

    pub fn to_vk(ptr: *mut Self) -> ash::vk::Device {
        ash::vk::Device::from_raw(ptr as usize as u64)
    }

    /// SAFETY: caller must provide a valid non-null Vulkan device handle created by this ICD.
    pub unsafe fn from_vk<'a>(handle: ash::vk::Device) -> &'a Self {
        &*(handle.as_raw() as usize as *const Self)
    }

    /// SAFETY: caller must provide a valid non-null Vulkan device handle created by this ICD.
    pub unsafe fn from_vk_mut<'a>(handle: ash::vk::Device) -> &'a mut Self {
        &mut *(handle.as_raw() as usize as *mut Self)
    }
}

#[repr(C)]
pub struct QueueDispatchable {
    pub loader_data: usize,
    pub id: u64,
    pub device: *mut DeviceDispatchable,
    pub client: Arc<Client>,
    pub queue_family_index: u32,
    pub queue_index: u32,
}

impl QueueDispatchable {
    pub fn new(
        id: u64,
        device: *mut DeviceDispatchable,
        client: Arc<Client>,
        queue_family_index: u32,
        queue_index: u32,
    ) -> Box<Self> {
        Box::new(Self {
            loader_data: 0,
            id,
            device,
            client,
            queue_family_index,
            queue_index,
        })
    }

    pub fn to_vk(ptr: *mut Self) -> ash::vk::Queue {
        ash::vk::Queue::from_raw(ptr as usize as u64)
    }

    /// SAFETY: caller must provide a valid non-null Vulkan queue handle created by this ICD.
    pub unsafe fn from_vk<'a>(handle: ash::vk::Queue) -> &'a Self {
        &*(handle.as_raw() as usize as *const Self)
    }

    /// SAFETY: caller must provide a valid non-null Vulkan queue handle created by this ICD.
    pub unsafe fn from_vk_mut<'a>(handle: ash::vk::Queue) -> &'a mut Self {
        &mut *(handle.as_raw() as usize as *mut Self)
    }
}

#[repr(C)]
pub struct CommandBufferDispatchable {
    pub loader_data: usize,
    pub id: u64,
    pub device: *mut DeviceDispatchable,
    pub client: Arc<Client>,
    pub pool_id: u64,
}

impl CommandBufferDispatchable {
    pub fn new(
        id: u64,
        device: *mut DeviceDispatchable,
        client: Arc<Client>,
        pool_id: u64,
    ) -> Box<Self> {
        Box::new(Self {
            loader_data: 0,
            id,
            device,
            client,
            pool_id,
        })
    }

    pub fn to_vk(ptr: *mut Self) -> ash::vk::CommandBuffer {
        ash::vk::CommandBuffer::from_raw(ptr as usize as u64)
    }

    /// SAFETY: caller must provide a valid non-null Vulkan command buffer handle created by this ICD.
    pub unsafe fn from_vk<'a>(handle: ash::vk::CommandBuffer) -> &'a Self {
        &*(handle.as_raw() as usize as *const Self)
    }

    /// SAFETY: caller must provide a valid non-null Vulkan command buffer handle created by this ICD.
    pub unsafe fn from_vk_mut<'a>(handle: ash::vk::CommandBuffer) -> &'a mut Self {
        &mut *(handle.as_raw() as usize as *mut Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatchable_handles_loader_data_offset() {
        assert_eq!(std::mem::offset_of!(InstanceDispatchable, loader_data), 0);
        assert_eq!(
            std::mem::offset_of!(PhysicalDeviceDispatchable, loader_data),
            0
        );
        assert_eq!(std::mem::offset_of!(DeviceDispatchable, loader_data), 0);
        assert_eq!(std::mem::offset_of!(QueueDispatchable, loader_data), 0);
        assert_eq!(
            std::mem::offset_of!(CommandBufferDispatchable, loader_data),
            0
        );
    }
}
