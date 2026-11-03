#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bugcodes.h>
#include <wdf.h>
#include <iddcx.h>

#define VKREMOTE_ADAPTER_NAME L"vkremote Virtual GPU Adapter"

// Context structures
struct DeviceContext
{
    IDDCX_ADAPTER Adapter;
};

WDF_DECLARE_CONTEXT_TYPE_WITH_NAME(DeviceContext, GetDeviceContext);

// Driver callbacks
extern "C" {
    DRIVER_INITIALIZE DriverEntry;
    EVT_WDF_DRIVER_DEVICE_ADD EvtDeviceAdd;
    EVT_IDD_CX_ADAPTER_INIT_FINISHED EvtAdapterInitFinished;
    EVT_IDD_CX_PARSE_MONITOR_DESCRIPTION EvtParseMonitorDescription;
    EVT_IDD_CX_MONITOR_GET_DEFAULT_DESCRIPTION_MODES EvtMonitorGetDefaultDescriptionModes;
    EVT_IDD_CX_MONITOR_QUERY_TARGET_MODES EvtMonitorQueryTargetModes;
    EVT_IDD_CX_MONITOR_ASSIGN_SWAPCHAIN EvtMonitorAssignSwapChain;
    EVT_IDD_CX_MONITOR_UNASSIGN_SWAPCHAIN EvtMonitorUnassignSwapChain;
}
