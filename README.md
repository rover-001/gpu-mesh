# rgpu (vkremote): High-Performance Remote GPU Engine & Distributed Mesh

`rgpu` is a cross-platform system written in Rust that exposes remote physical GPUs across a local network as native Vulkan physical devices and unified compute/VRAM clusters. Any unmodified Vulkan application running on a client machine executes its GPU workloads on a remote machine (e.g., a lightweight laptop offloading rendering and compute to an RTX 3060 desktop) with minimal latency and high throughput over standard Wi-Fi or Ethernet.

---

## Key Capabilities

- **Transparent Vulkan Remoting (ICD)**: Native Khronos-compliant Installable Client Driver (`libvkremote_icd.so` / `vkremote_icd.dll`) presenting `<GPU Name> (Remote)` to the Vulkan loader.
- **Graphics & Compute Support**: Supports Vulkan compute pipelines, graphics dispatch, swapchain/WSI presentation, and asset streaming (verified with remote Horizon / Minecraft offloading).
- **Sub-Second Asset Loading**: Batched `vkBindBufferMemory` and `vkCreateBuffer` pipeline reduction cutting thousands of round-trips from minutes to sub-second transfer.
- **Smart Memory Shadowing & Dirty-Chunk Sync**: Efficient 64 KB chunk tracking hashed via `xxh3` with `zstd` compression over length-prefixed `postcard` binary protocol frames.
- **Distributed GPU Mesh (`gpu-mesh`)**: Multi-host GPU clustering, peer discovery, proportional VRAM allocation, and resident LLM worker orchestration (llama.cpp / llama-server).
- **Windows Task Manager Integration (`iddcx`)**: Custom Indirect Display Driver Class eXtension (UMDF 2.x) driver registering remote GPU activity as a physical/virtual display adapter in Windows Device Manager and Task Manager.
- **Cross-Platform**: Seamless interoperability between Linux (client/server) and Windows (client/server).

---

## Architecture

```
                      Client Application (Unmodified Vulkan App)
                                        │
                                        ▼
                             Vulkan Loader (Khronos)
                                        │
                                        ▼
                          vkremote-icd (cdylib driver)
                           • Linux: libvkremote_icd.so
                           • Windows: vkremote_icd.dll
                           • Shadow host memory + xxh3 dirty chunk tracking
                           • Async command batching & local state caching
                                        │
                                        ▼ (TCP / Wire Protocol v2 + zstd)
                                  Local Network (Wi-Fi 5/6, LAN)
                                        │
                                        ▼
                          vkremote-server (Replay Daemon)
                           • Native execution via Ash (Vulkan)
                           • Timeline semaphore synchronization
                           • Queue submit & layout restoration
                                        │
                                        ▼
                               Physical Hardware GPU (e.g. RTX 3060)
```

---

## Workspace Layout

```
rgpu/
├── apps/
│   └── gpu-mesh/              # Multi-host GPU clustering, UI, and distributed LLM engine
├── crates/
│   ├── wire/                  # Wire protocol v2, framing, postcard serialization, zstd, xxh3
│   ├── icd/                   # Vulkan Installable Client Driver (Linux .so, Windows .dll)
│   └── server/                # Vulkan replay server daemon powered by Ash
├── demos/
│   ├── vecadd/                # 1024-float vector addition compute demo
│   ├── matmul/                # Matrix multiplication compute benchmark (GFLOPS)
│   └── mandelbrot/            # 640x360 compute shader fractal generator with ASCII preview
├── drivers/
│   └── windows-iddcx/         # Windows Indirect Display Driver (IddCx) for Task Manager telemetry
├── tools/
│   ├── probe/                 # Vulkan device enumerator and capability inspector
│   └── remote-gpu-service/    # Background daemon service runner
└── xtask/                     # Build scripts, manifest generators, and loopback verification
```

---

## Quickstart

### 1. Build the Workspace

```bash
cargo build --workspace
```

### 2. Automated Loopback Verification

Run the end-to-end loopback suite to verify the server daemon, ICD manifest generation, `probe`, `vecadd`, `matmul`, and `mandelbrot`:

```bash
cargo run -p xtask -- run-loopback
```

### 3. Running Across Separate Machines

#### On the Host / Server (e.g., Windows PC with Dedicated GPU):

```powershell
# Set optional shared authentication token
$env:VKREMOTE_TOKEN = "your-secret-token"

# Launch server on TCP 50051
cargo run -p vkremote-server -- --bind 0.0.0.0 --port 50051
```

#### On the Client (e.g., Linux Laptop):

1. Generate the ICD manifest pointing to the compiled driver:
   ```bash
   cargo run -p xtask -- install-icd
   ```

2. Point Vulkan to the ICD and target the server:
   ```bash
   export VK_DRIVER_FILES="$(pwd)/target/vkremote_icd.json"
   export VKREMOTE_SERVER="192.168.1.11:50051"
   export VKREMOTE_TOKEN="your-secret-token"
   ```

3. Run any Vulkan application or benchmark:
   ```bash
   # Enumerate devices — lists "<GPU Name> (Remote)"
   cargo run -p vkremote-probe

   # Run matrix multiplication compute benchmark
   cargo run -p matmul

   # Render Mandelbrot fractal
   cargo run -p mandelbrot
   ```

---

## Distributed GPU Mesh & LLM Clustering

The `gpu-mesh` application coordinates multiple GPUs across your LAN for shared VRAM inference and dynamic model offloading:

```bash
cargo run -p gpu-mesh
```

- **Dynamic Auto-Discovery & Manual Adding**: Probes local subnets or connects directly to remote IP peers.
- **Model-by-Reference**: Resolves model weights directly on remote workers from Hugging Face caches rather than streaming gigabytes across Wi-Fi.
- **Proportional VRAM Allocation**: Automatic or manual layer splitting between discrete desktop and laptop GPUs.

---

## Windows Virtual GPU Adapter (`windows-iddcx`)

For Windows hosts, the `drivers/windows-iddcx` package allows Windows to recognize the remote execution target as a virtual WDDM display adapter, rendering live engine utilization graphs inside Windows Task Manager's Performance tab.

Refer to [`bench.ps1`](file:///home/rover/Projects/rgpu/bench.ps1) for automated speed-lever profiling (Flash Attention, non-mmap cold loads, token throughput).

---

## License

Licensed under the MIT License.
