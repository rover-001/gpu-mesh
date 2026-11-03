use crate::dispatch::*;
use ash::vk::Handle;
use std::ffi::CStr;

macro_rules! pfn {
    ($f:expr) => {
        // SAFETY: transmuting a concrete extern "system" fn pointer to Vulkan PFN_vkVoidFunction
        Some(unsafe {
            std::mem::transmute::<*const (), unsafe extern "system" fn()>($f as *const ())
        })
    };
}

#[no_mangle]
pub unsafe extern "system" fn vk_icdNegotiateLoaderICDInterfaceVersion(
    p_supported_version: *mut u32,
) -> ash::vk::Result {
    if p_supported_version.is_null() {
        return ash::vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    // SAFETY: caller provided a valid non-null pointer to u32
    let supported = *p_supported_version;
    *p_supported_version = supported.min(5);
    ash::vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn vk_icdGetPhysicalDeviceProcAddr(
    _instance: ash::vk::Instance,
    p_name: *const std::os::raw::c_char,
) -> ash::vk::PFN_vkVoidFunction {
    if p_name.is_null() {
        return None;
    }

    // SAFETY: p_name is a null-terminated C string provided by the caller
    let name = CStr::from_ptr(p_name).to_bytes();

    match name {
        b"vkGetPhysicalDeviceProperties" => pfn!(vkGetPhysicalDeviceProperties),
        b"vkGetPhysicalDeviceProperties2" => pfn!(vkGetPhysicalDeviceProperties2),
        b"vkGetPhysicalDeviceProperties2KHR" => pfn!(vkGetPhysicalDeviceProperties2KHR),
        b"vkGetPhysicalDeviceFeatures" => pfn!(vkGetPhysicalDeviceFeatures),
        b"vkGetPhysicalDeviceFeatures2" => pfn!(vkGetPhysicalDeviceFeatures2),
        b"vkGetPhysicalDeviceFeatures2KHR" => pfn!(vkGetPhysicalDeviceFeatures2KHR),
        b"vkGetPhysicalDeviceMemoryProperties" => pfn!(vkGetPhysicalDeviceMemoryProperties),
        b"vkGetPhysicalDeviceMemoryProperties2" => pfn!(vkGetPhysicalDeviceMemoryProperties2),
        b"vkGetPhysicalDeviceMemoryProperties2KHR" => pfn!(vkGetPhysicalDeviceMemoryProperties2KHR),
        b"vkGetPhysicalDeviceQueueFamilyProperties" => {
            pfn!(vkGetPhysicalDeviceQueueFamilyProperties)
        }
        b"vkGetPhysicalDeviceQueueFamilyProperties2" => {
            pfn!(vkGetPhysicalDeviceQueueFamilyProperties2)
        }
        b"vkGetPhysicalDeviceQueueFamilyProperties2KHR" => {
            pfn!(vkGetPhysicalDeviceQueueFamilyProperties2KHR)
        }
        b"vkGetPhysicalDeviceFormatProperties" => pfn!(vkGetPhysicalDeviceFormatProperties),
        b"vkGetPhysicalDeviceFormatProperties2" => pfn!(vkGetPhysicalDeviceFormatProperties2),
        b"vkGetPhysicalDeviceFormatProperties2KHR" => {
            pfn!(vkGetPhysicalDeviceFormatProperties2KHR)
        }
        b"vkGetPhysicalDeviceImageFormatProperties" => {
            pfn!(vkGetPhysicalDeviceImageFormatProperties)
        }
        b"vkGetPhysicalDeviceSparseImageFormatProperties" => {
            pfn!(vkGetPhysicalDeviceSparseImageFormatProperties)
        }
        b"vkEnumerateDeviceExtensionProperties" => pfn!(vkEnumerateDeviceExtensionProperties),
        b"vkEnumerateDeviceLayerProperties" => pfn!(vkEnumerateDeviceLayerProperties),
        b"vkCreateDevice" => pfn!(vkCreateDevice),
        b"vkGetPhysicalDeviceSurfaceSupportKHR" => pfn!(vkGetPhysicalDeviceSurfaceSupportKHR),
        b"vkGetPhysicalDeviceWin32PresentationSupportKHR" => {
            pfn!(vkGetPhysicalDeviceWin32PresentationSupportKHR)
        }
        b"vkGetPhysicalDeviceXlibPresentationSupportKHR" => {
            pfn!(vkGetPhysicalDeviceXlibPresentationSupportKHR)
        }
        b"vkGetPhysicalDeviceXcbPresentationSupportKHR" => {
            pfn!(vkGetPhysicalDeviceXcbPresentationSupportKHR)
        }
        b"vkGetPhysicalDeviceWaylandPresentationSupportKHR" => {
            pfn!(vkGetPhysicalDeviceWaylandPresentationSupportKHR)
        }
        b"vkGetPhysicalDeviceSurfaceCapabilitiesKHR" => {
            pfn!(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)
        }
        b"vkGetPhysicalDeviceSurfaceFormatsKHR" => pfn!(vkGetPhysicalDeviceSurfaceFormatsKHR),
        b"vkGetPhysicalDeviceSurfacePresentModesKHR" => {
            pfn!(vkGetPhysicalDeviceSurfacePresentModesKHR)
        }
        _ => None,
    }
}

#[no_mangle]
pub unsafe extern "system" fn vk_icdGetInstanceProcAddr(
    instance: ash::vk::Instance,
    p_name: *const std::os::raw::c_char,
) -> ash::vk::PFN_vkVoidFunction {
    vkGetInstanceProcAddr(instance, p_name)
}

#[no_mangle]
pub unsafe extern "system" fn vkGetInstanceProcAddr(
    instance: ash::vk::Instance,
    p_name: *const std::os::raw::c_char,
) -> ash::vk::PFN_vkVoidFunction {
    if p_name.is_null() {
        return None;
    }

    // SAFETY: p_name is a null-terminated C string provided by the caller
    let name = CStr::from_ptr(p_name).to_bytes();

    // Global functions accessible with or without instance
    let global_fn = match name {
        b"vkGetInstanceProcAddr" => pfn!(vkGetInstanceProcAddr),
        b"vk_icdGetInstanceProcAddr" => pfn!(vk_icdGetInstanceProcAddr),
        b"vk_icdGetPhysicalDeviceProcAddr" => pfn!(vk_icdGetPhysicalDeviceProcAddr),
        b"vk_icdNegotiateLoaderICDInterfaceVersion" => {
            pfn!(vk_icdNegotiateLoaderICDInterfaceVersion)
        }
        b"vkEnumerateInstanceVersion" => pfn!(vkEnumerateInstanceVersion),
        b"vkEnumerateInstanceExtensionProperties" => pfn!(vkEnumerateInstanceExtensionProperties),
        b"vkCreateInstance" => pfn!(vkCreateInstance),
        _ => None,
    };

    if global_fn.is_some() {
        return global_fn;
    }

    // If instance is null, only global functions can be queried
    if instance.is_null() {
        return None;
    }

    // Physical device / Instance functions
    match name {
        b"vkDestroyInstance" => pfn!(vkDestroyInstance),
        b"vkEnumeratePhysicalDevices" => pfn!(vkEnumeratePhysicalDevices),
        b"vkGetPhysicalDeviceProperties" => pfn!(vkGetPhysicalDeviceProperties),
        b"vkGetPhysicalDeviceProperties2" => pfn!(vkGetPhysicalDeviceProperties2),
        b"vkGetPhysicalDeviceProperties2KHR" => pfn!(vkGetPhysicalDeviceProperties2KHR),
        b"vkGetPhysicalDeviceFeatures" => pfn!(vkGetPhysicalDeviceFeatures),
        b"vkGetPhysicalDeviceFeatures2" => pfn!(vkGetPhysicalDeviceFeatures2),
        b"vkGetPhysicalDeviceFeatures2KHR" => pfn!(vkGetPhysicalDeviceFeatures2KHR),
        b"vkGetPhysicalDeviceMemoryProperties" => pfn!(vkGetPhysicalDeviceMemoryProperties),
        b"vkGetPhysicalDeviceMemoryProperties2" => pfn!(vkGetPhysicalDeviceMemoryProperties2),
        b"vkGetPhysicalDeviceMemoryProperties2KHR" => pfn!(vkGetPhysicalDeviceMemoryProperties2KHR),
        b"vkGetPhysicalDeviceQueueFamilyProperties" => {
            pfn!(vkGetPhysicalDeviceQueueFamilyProperties)
        }
        b"vkGetPhysicalDeviceQueueFamilyProperties2" => {
            pfn!(vkGetPhysicalDeviceQueueFamilyProperties2)
        }
        b"vkGetPhysicalDeviceQueueFamilyProperties2KHR" => {
            pfn!(vkGetPhysicalDeviceQueueFamilyProperties2KHR)
        }
        b"vkGetPhysicalDeviceFormatProperties" => pfn!(vkGetPhysicalDeviceFormatProperties),
        b"vkGetPhysicalDeviceFormatProperties2" => pfn!(vkGetPhysicalDeviceFormatProperties2),
        b"vkGetPhysicalDeviceFormatProperties2KHR" => {
            pfn!(vkGetPhysicalDeviceFormatProperties2KHR)
        }
        b"vkGetPhysicalDeviceImageFormatProperties" => {
            pfn!(vkGetPhysicalDeviceImageFormatProperties)
        }
        b"vkGetPhysicalDeviceSparseImageFormatProperties" => {
            pfn!(vkGetPhysicalDeviceSparseImageFormatProperties)
        }
        b"vkEnumerateDeviceExtensionProperties" => pfn!(vkEnumerateDeviceExtensionProperties),
        b"vkEnumerateDeviceLayerProperties" => pfn!(vkEnumerateDeviceLayerProperties),
        b"vkCreateDevice" => pfn!(vkCreateDevice),
        b"vkGetDeviceProcAddr" => pfn!(vkGetDeviceProcAddr),

        // WSI Surface functions
        b"vkCreateWin32SurfaceKHR" => pfn!(vkCreateWin32SurfaceKHR),
        b"vkCreateWaylandSurfaceKHR" => pfn!(vkCreateWaylandSurfaceKHR),
        b"vkCreateXcbSurfaceKHR" => pfn!(vkCreateXcbSurfaceKHR),
        b"vkCreateXlibSurfaceKHR" => pfn!(vkCreateXlibSurfaceKHR),
        b"vkDestroySurfaceKHR" => pfn!(vkDestroySurfaceKHR),
        b"vkGetPhysicalDeviceSurfaceSupportKHR" => pfn!(vkGetPhysicalDeviceSurfaceSupportKHR),
        b"vkGetPhysicalDeviceWin32PresentationSupportKHR" => {
            pfn!(vkGetPhysicalDeviceWin32PresentationSupportKHR)
        }
        b"vkGetPhysicalDeviceXlibPresentationSupportKHR" => {
            pfn!(vkGetPhysicalDeviceXlibPresentationSupportKHR)
        }
        b"vkGetPhysicalDeviceXcbPresentationSupportKHR" => {
            pfn!(vkGetPhysicalDeviceXcbPresentationSupportKHR)
        }
        b"vkGetPhysicalDeviceWaylandPresentationSupportKHR" => {
            pfn!(vkGetPhysicalDeviceWaylandPresentationSupportKHR)
        }
        b"vkGetPhysicalDeviceSurfaceCapabilitiesKHR" => {
            pfn!(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)
        }
        b"vkGetPhysicalDeviceSurfaceFormatsKHR" => pfn!(vkGetPhysicalDeviceSurfaceFormatsKHR),
        b"vkGetPhysicalDeviceSurfacePresentModesKHR" => {
            pfn!(vkGetPhysicalDeviceSurfacePresentModesKHR)
        }

        // Core 1.1 / KHR memory requirements & bind functions
        b"vkGetImageMemoryRequirements2" => pfn!(vkGetImageMemoryRequirements2),
        b"vkGetImageMemoryRequirements2KHR" => pfn!(vkGetImageMemoryRequirements2KHR),
        b"vkBindImageMemory2" => pfn!(vkBindImageMemory2),
        b"vkBindImageMemory2KHR" => pfn!(vkBindImageMemory2KHR),

        // Per Vulkan spec, vkGetInstanceProcAddr may also return device commands
        _ => vkGetDeviceProcAddr(ash::vk::Device::null(), p_name),
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetDeviceProcAddr(
    _device: ash::vk::Device,
    p_name: *const std::os::raw::c_char,
) -> ash::vk::PFN_vkVoidFunction {
    if p_name.is_null() {
        return None;
    }

    // SAFETY: p_name is a null-terminated C string provided by the caller
    let name = CStr::from_ptr(p_name).to_bytes();

    match name {
        // Device core
        b"vkGetDeviceProcAddr" => pfn!(vkGetDeviceProcAddr),
        b"vkDestroyDevice" => pfn!(vkDestroyDevice),
        b"vkGetDeviceQueue" => pfn!(vkGetDeviceQueue),
        b"vkGetDeviceQueue2" => pfn!(vkGetDeviceQueue2),
        b"vkQueueSubmit" => pfn!(vkQueueSubmit),
        b"vkQueueWaitIdle" => pfn!(vkQueueWaitIdle),
        b"vkDeviceWaitIdle" => pfn!(vkDeviceWaitIdle),

        // Images, Views, Samplers
        b"vkCreateImage" => pfn!(vkCreateImage),
        b"vkDestroyImage" => pfn!(vkDestroyImage),
        b"vkGetImageMemoryRequirements" => pfn!(vkGetImageMemoryRequirements),
        b"vkGetImageSubresourceLayout" => pfn!(vkGetImageSubresourceLayout),
        b"vkCreatePipelineCache" => pfn!(vkCreatePipelineCache),
        b"vkDestroyPipelineCache" => pfn!(vkDestroyPipelineCache),
        b"vkGetPipelineCacheData" => pfn!(vkGetPipelineCacheData),
        b"vkMergePipelineCaches" => pfn!(vkMergePipelineCaches),
        b"vkGetImageMemoryRequirements2" => pfn!(vkGetImageMemoryRequirements2),
        b"vkGetImageMemoryRequirements2KHR" => pfn!(vkGetImageMemoryRequirements2KHR),
        b"vkBindImageMemory" => pfn!(vkBindImageMemory),
        b"vkBindImageMemory2" => pfn!(vkBindImageMemory2),
        b"vkBindImageMemory2KHR" => pfn!(vkBindImageMemory2KHR),
        b"vkCreateImageView" => pfn!(vkCreateImageView),
        b"vkDestroyImageView" => pfn!(vkDestroyImageView),
        b"vkCreateSampler" => pfn!(vkCreateSampler),
        b"vkDestroySampler" => pfn!(vkDestroySampler),

        // Memory & Buffer
        b"vkCreateBuffer" => pfn!(vkCreateBuffer),
        b"vkDestroyBuffer" => pfn!(vkDestroyBuffer),
        b"vkGetBufferMemoryRequirements" => pfn!(vkGetBufferMemoryRequirements),
        b"vkGetBufferMemoryRequirements2" => pfn!(vkGetBufferMemoryRequirements2),
        b"vkGetBufferMemoryRequirements2KHR" => pfn!(vkGetBufferMemoryRequirements2KHR),
        b"vkAllocateMemory" => pfn!(vkAllocateMemory),
        b"vkFreeMemory" => pfn!(vkFreeMemory),
        b"vkBindBufferMemory" => pfn!(vkBindBufferMemory),
        b"vkBindBufferMemory2" => pfn!(vkBindBufferMemory2),
        b"vkBindBufferMemory2KHR" => pfn!(vkBindBufferMemory2KHR),
        b"vkMapMemory" => pfn!(vkMapMemory),
        b"vkUnmapMemory" => pfn!(vkUnmapMemory),
        b"vkFlushMappedMemoryRanges" => pfn!(vkFlushMappedMemoryRanges),
        b"vkInvalidateMappedMemoryRanges" => pfn!(vkInvalidateMappedMemoryRanges),

        // Shaders & Pipelines
        b"vkCreateShaderModule" => pfn!(vkCreateShaderModule),
        b"vkDestroyShaderModule" => pfn!(vkDestroyShaderModule),
        b"vkCreateDescriptorSetLayout" => pfn!(vkCreateDescriptorSetLayout),
        b"vkDestroyDescriptorSetLayout" => pfn!(vkDestroyDescriptorSetLayout),
        b"vkCreatePipelineLayout" => pfn!(vkCreatePipelineLayout),
        b"vkDestroyPipelineLayout" => pfn!(vkDestroyPipelineLayout),
        b"vkCreateComputePipelines" => pfn!(vkCreateComputePipelines),
        b"vkCreateGraphicsPipelines" => pfn!(vkCreateGraphicsPipelines),
        b"vkCreateRenderPass" => pfn!(vkCreateRenderPass),
        b"vkCreateRenderPass2" => pfn!(vkCreateRenderPass2),
        b"vkCreateRenderPass2KHR" => pfn!(vkCreateRenderPass2KHR),
        b"vkDestroyRenderPass" => pfn!(vkDestroyRenderPass),
        b"vkCreateFramebuffer" => pfn!(vkCreateFramebuffer),
        b"vkDestroyFramebuffer" => pfn!(vkDestroyFramebuffer),
        b"vkDestroyPipeline" => pfn!(vkDestroyPipeline),
        b"vkCreateDescriptorPool" => pfn!(vkCreateDescriptorPool),
        b"vkDestroyDescriptorPool" => pfn!(vkDestroyDescriptorPool),
        b"vkAllocateDescriptorSets" => pfn!(vkAllocateDescriptorSets),
        b"vkFreeDescriptorSets" => pfn!(vkFreeDescriptorSets),
        b"vkResetDescriptorPool" => pfn!(vkResetDescriptorPool),
        b"vkUpdateDescriptorSets" => pfn!(vkUpdateDescriptorSets),

        // Commands
        b"vkCreateCommandPool" => pfn!(vkCreateCommandPool),
        b"vkDestroyCommandPool" => pfn!(vkDestroyCommandPool),
        b"vkAllocateCommandBuffers" => pfn!(vkAllocateCommandBuffers),
        b"vkFreeCommandBuffers" => pfn!(vkFreeCommandBuffers),
        b"vkResetCommandBuffer" => pfn!(vkResetCommandBuffer),
        b"vkResetCommandPool" => pfn!(vkResetCommandPool),
        b"vkGetRenderAreaGranularity" => pfn!(vkGetRenderAreaGranularity),
        b"vkBeginCommandBuffer" => pfn!(vkBeginCommandBuffer),
        b"vkEndCommandBuffer" => pfn!(vkEndCommandBuffer),
        b"vkCmdBindPipeline" => pfn!(vkCmdBindPipeline),
        b"vkCmdBindDescriptorSets" => pfn!(vkCmdBindDescriptorSets),
        b"vkCmdPushConstants" => pfn!(vkCmdPushConstants),
        b"vkCmdDispatch" => pfn!(vkCmdDispatch),
        b"vkCmdCopyBuffer" => pfn!(vkCmdCopyBuffer),
        b"vkCmdCopyBufferToImage" => pfn!(vkCmdCopyBufferToImage),
        b"vkCmdCopyImageToBuffer" => pfn!(vkCmdCopyImageToBuffer),
        b"vkCmdCopyImage" => pfn!(vkCmdCopyImage),
        b"vkCmdBlitImage" => pfn!(vkCmdBlitImage),
        b"vkCmdResolveImage" => pfn!(vkCmdResolveImage),
        b"vkCmdUpdateBuffer" => pfn!(vkCmdUpdateBuffer),
        b"vkCmdFillBuffer" => pfn!(vkCmdFillBuffer),
        b"vkCmdClearColorImage" => pfn!(vkCmdClearColorImage),
        b"vkCmdClearDepthStencilImage" => pfn!(vkCmdClearDepthStencilImage),
        b"vkCmdClearAttachments" => pfn!(vkCmdClearAttachments),
        b"vkCmdSetEvent" => pfn!(vkCmdSetEvent),
        b"vkCmdResetEvent" => pfn!(vkCmdResetEvent),
        b"vkCmdWaitEvents" => pfn!(vkCmdWaitEvents),
        b"vkCmdBeginQuery" => pfn!(vkCmdBeginQuery),
        b"vkCmdEndQuery" => pfn!(vkCmdEndQuery),
        b"vkCmdCopyQueryPoolResults" => pfn!(vkCmdCopyQueryPoolResults),
        b"vkCmdPipelineBarrier" => pfn!(vkCmdPipelineBarrier),
        b"vkCmdBeginRenderPass" => pfn!(vkCmdBeginRenderPass),
        b"vkCmdBeginRenderPass2" => pfn!(vkCmdBeginRenderPass2),
        b"vkCmdBeginRenderPass2KHR" => pfn!(vkCmdBeginRenderPass2KHR),
        b"vkCmdNextSubpass" => pfn!(vkCmdNextSubpass),
        b"vkCmdNextSubpass2" => pfn!(vkCmdNextSubpass2),
        b"vkCmdNextSubpass2KHR" => pfn!(vkCmdNextSubpass2KHR),
        b"vkCmdEndRenderPass" => pfn!(vkCmdEndRenderPass),
        b"vkCmdEndRenderPass2" => pfn!(vkCmdEndRenderPass2),
        b"vkCmdEndRenderPass2KHR" => pfn!(vkCmdEndRenderPass2KHR),
        b"vkCmdExecuteCommands" => pfn!(vkCmdExecuteCommands),
        b"vkCmdBindVertexBuffers" => pfn!(vkCmdBindVertexBuffers),
        b"vkCmdBindIndexBuffer" => pfn!(vkCmdBindIndexBuffer),
        b"vkCmdSetViewport" => pfn!(vkCmdSetViewport),
        b"vkCmdSetScissor" => pfn!(vkCmdSetScissor),
        b"vkCmdSetLineWidth" => pfn!(vkCmdSetLineWidth),
        b"vkCmdSetDepthBias" => pfn!(vkCmdSetDepthBias),
        b"vkCmdSetBlendConstants" => pfn!(vkCmdSetBlendConstants),
        b"vkCmdSetDepthBounds" => pfn!(vkCmdSetDepthBounds),
        b"vkCmdSetStencilCompareMask" => pfn!(vkCmdSetStencilCompareMask),
        b"vkCmdSetStencilWriteMask" => pfn!(vkCmdSetStencilWriteMask),
        b"vkCmdSetStencilReference" => pfn!(vkCmdSetStencilReference),
        b"vkCmdDraw" => pfn!(vkCmdDraw),
        b"vkCmdDrawIndexed" => pfn!(vkCmdDrawIndexed),
        b"vkCmdDrawIndirect" => pfn!(vkCmdDrawIndirect),
        b"vkCmdDrawIndexedIndirect" => pfn!(vkCmdDrawIndexedIndirect),
        b"vkCmdDrawIndirectCount" => pfn!(vkCmdDrawIndirectCount),
        b"vkCmdDrawIndirectCountKHR" => pfn!(vkCmdDrawIndirectCountKHR),
        b"vkCmdDrawIndexedIndirectCount" => pfn!(vkCmdDrawIndexedIndirectCount),
        b"vkCmdDrawIndexedIndirectCountKHR" => pfn!(vkCmdDrawIndexedIndirectCountKHR),
        b"vkCmdDispatchIndirect" => pfn!(vkCmdDispatchIndirect),

        // Synchronization
        b"vkCreateFence" => pfn!(vkCreateFence),
        b"vkDestroyFence" => pfn!(vkDestroyFence),
        b"vkWaitForFences" => pfn!(vkWaitForFences),
        b"vkResetFences" => pfn!(vkResetFences),
        b"vkGetFenceStatus" => pfn!(vkGetFenceStatus),
        b"vkCreateSemaphore" => pfn!(vkCreateSemaphore),
        b"vkDestroySemaphore" => pfn!(vkDestroySemaphore),
        b"vkWaitSemaphores" => pfn!(vkWaitSemaphores),
        b"vkWaitSemaphoresKHR" => pfn!(vkWaitSemaphoresKHR),
        b"vkSignalSemaphore" => pfn!(vkSignalSemaphore),
        b"vkSignalSemaphoreKHR" => pfn!(vkSignalSemaphoreKHR),
        b"vkGetSemaphoreCounterValue" => pfn!(vkGetSemaphoreCounterValue),
        b"vkGetSemaphoreCounterValueKHR" => pfn!(vkGetSemaphoreCounterValueKHR),
        b"vkResetQueryPool" => pfn!(vkResetQueryPool),
        b"vkResetQueryPoolEXT" => pfn!(vkResetQueryPoolEXT),

        // Buffer Device Address
        b"vkGetBufferDeviceAddress" => pfn!(vkGetBufferDeviceAddress),
        b"vkGetBufferDeviceAddressKHR" => pfn!(vkGetBufferDeviceAddressKHR),
        b"vkGetBufferDeviceAddressEXT" => pfn!(vkGetBufferDeviceAddressEXT),
        b"vkGetBufferOpaqueCaptureAddress" => pfn!(vkGetBufferOpaqueCaptureAddress),
        b"vkGetBufferOpaqueCaptureAddressKHR" => pfn!(vkGetBufferOpaqueCaptureAddressKHR),
        b"vkGetDeviceMemoryOpaqueCaptureAddress" => pfn!(vkGetDeviceMemoryOpaqueCaptureAddress),
        b"vkGetDeviceMemoryOpaqueCaptureAddressKHR" => {
            pfn!(vkGetDeviceMemoryOpaqueCaptureAddressKHR)
        }

        // Query Pools & Timestamps
        b"vkCreateQueryPool" => pfn!(vkCreateQueryPool),
        b"vkDestroyQueryPool" => pfn!(vkDestroyQueryPool),
        b"vkGetQueryPoolResults" => pfn!(vkGetQueryPoolResults),
        b"vkCmdResetQueryPool" => pfn!(vkCmdResetQueryPool),
        b"vkCmdWriteTimestamp" => pfn!(vkCmdWriteTimestamp),
        b"vkCmdWriteTimestamp2" => pfn!(vkCmdWriteTimestamp2),
        b"vkCmdWriteTimestamp2KHR" => pfn!(vkCmdWriteTimestamp2KHR),

        // WSI Swapchain
        b"vkCreateSwapchainKHR" => pfn!(vkCreateSwapchainKHR),
        b"vkDestroySwapchainKHR" => pfn!(vkDestroySwapchainKHR),
        b"vkGetSwapchainImagesKHR" => pfn!(vkGetSwapchainImagesKHR),
        b"vkAcquireNextImageKHR" => pfn!(vkAcquireNextImageKHR),
        b"vkQueuePresentKHR" => pfn!(vkQueuePresentKHR),

        _ => None,
    }
}
