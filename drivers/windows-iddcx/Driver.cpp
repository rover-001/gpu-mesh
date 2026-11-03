#include "Driver.h"

extern "C" NTSTATUS DriverEntry(
    _In_ PDRIVER_OBJECT  DriverObject,
    _In_ PUNICODE_STRING RegistryPath
)
{
    WDF_DRIVER_CONFIG Config;
    WDF_DRIVER_CONFIG_INIT(&Config, EvtDeviceAdd);

    NTSTATUS Status = WdfDriverCreate(
        DriverObject,
        RegistryPath,
        WDF_NO_OBJECT_ATTRIBUTES,
        &Config,
        WDF_NO_HANDLE
    );

    return Status;
}

NTSTATUS EvtDeviceAdd(
    _In_ WDFDRIVER       Driver,
    _Inout_ PWDFDEVICE_INIT DeviceInit
)
{
    UNREFERENCED_PARAMETER(Driver);

    NTSTATUS Status = IddCxDeviceInitConfig(DeviceInit, nullptr);
    if (!NT_SUCCESS(Status))
    {
        return Status;
    }

    WDF_OBJECT_ATTRIBUTES DeviceAttributes;
    WDF_OBJECT_ATTRIBUTES_INIT_CONTEXT_TYPE(&DeviceAttributes, DeviceContext);

    WDFDEVICE Device = WDF_NO_HANDLE;
    Status = WdfDeviceCreate(&DeviceInit, &DeviceAttributes, &Device);
    if (!NT_SUCCESS(Status))
    {
        return Status;
    }

    Status = IddCxDeviceInitialize(Device);
    if (!NT_SUCCESS(Status))
    {
        return Status;
    }

    // Configure IddCx Adapter
    IDDCX_ADAPTER_CAPS AdapterCaps = {};
    AdapterCaps.Size = sizeof(AdapterCaps);
    AdapterCaps.Flags = IDDCX_ADAPTER_FLAGS_CAN_USE_MOVE_RECTS;
    AdapterCaps.MaxMonitorsSupported = 1;
    AdapterCaps.EndPointType = IDDCX_ENDPOINT_TYPE_WIRELESS;

    IDARG_IN_ADAPTER_INIT AdapterInit = {};
    AdapterInit.WdfDevice = Device;
    AdapterInit.pCaps = &AdapterCaps;
    AdapterInit.ObjectAttributes.Size = sizeof(WDF_OBJECT_ATTRIBUTES);

    IDARG_OUT_ADAPTER_INIT AdapterInitOut = {};
    Status = IddCxAdapterInitAsync(&AdapterInit, &AdapterInitOut);
    if (!NT_SUCCESS(Status))
    {
        return Status;
    }

    auto* Context = GetDeviceContext(Device);
    Context->Adapter = AdapterInitOut.AdapterObject;

    return STATUS_SUCCESS;
}

NTSTATUS EvtAdapterInitFinished(
    _In_ IDDCX_ADAPTER AdapterObject,
    _In_ const IDARG_IN_ADAPTER_INIT_FINISHED* pInArgs
)
{
    UNREFERENCED_PARAMETER(AdapterObject);
    UNREFERENCED_PARAMETER(pInArgs);
    return STATUS_SUCCESS;
}

NTSTATUS EvtParseMonitorDescription(
    _In_ const IDARG_IN_PARSEMONITORDESCRIPTION* pInArgs,
    _Out_ IDARG_OUT_PARSEMONITORDESCRIPTION* pOutArgs
)
{
    UNREFERENCED_PARAMETER(pInArgs);
    pOutArgs->MonitorModeBufferOutputCount = 0;
    return STATUS_SUCCESS;
}

NTSTATUS EvtMonitorGetDefaultDescriptionModes(
    _In_ IDDCX_MONITOR MonitorObject,
    _In_ const IDARG_IN_GETDEFAULTDESCRIPTIONMODES* pInArgs,
    _Out_ IDARG_OUT_GETDEFAULTDESCRIPTIONMODES* pOutArgs
)
{
    UNREFERENCED_PARAMETER(MonitorObject);
    UNREFERENCED_PARAMETER(pInArgs);
    pOutArgs->DefaultMonitorModeBufferOutputCount = 0;
    return STATUS_SUCCESS;
}

NTSTATUS EvtMonitorQueryTargetModes(
    _In_ IDDCX_MONITOR MonitorObject,
    _In_ const IDARG_IN_QUERYTARGETMODES* pInArgs,
    _Out_ IDARG_OUT_QUERYTARGETMODES* pOutArgs
)
{
    UNREFERENCED_PARAMETER(MonitorObject);
    UNREFERENCED_PARAMETER(pInArgs);
    pOutArgs->TargetModeBufferOutputCount = 0;
    return STATUS_SUCCESS;
}

NTSTATUS EvtMonitorAssignSwapChain(
    _In_ IDDCX_MONITOR MonitorObject,
    _In_ const IDARG_IN_SETSWAPCHAIN* pInArgs
)
{
    UNREFERENCED_PARAMETER(MonitorObject);
    UNREFERENCED_PARAMETER(pInArgs);
    return STATUS_SUCCESS;
}

NTSTATUS EvtMonitorUnassignSwapChain(
    _In_ IDDCX_MONITOR MonitorObject
)
{
    UNREFERENCED_PARAMETER(MonitorObject);
    return STATUS_SUCCESS;
}
