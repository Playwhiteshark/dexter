# Dexter runtime performance audit

Status: active performance work, 2026-08-24.

The goal is to improve **interactive performance first**. Idle-memory efficiency is a second pass, but the profiler records residency now so later changes can be measured against the same baseline.

## Current runtime topology

Dexter currently has these important runtime boundaries:

- **Ollama model runner** — owns the main Gemma model, primarily on the RTX GPU with some host-RAM spill depending on the loaded model/context.
- **Dexter Host** — Qt UI plus one persistent Moonshine `Transcriber`. STT is not a separate Python process.
- **NeuTTS worker** — one Host-owned child process using the same root uv/Python 3.13 environment as the Host. It owns the Paul NeuTTS model and streams PCM back to the Host.
- **Gateway** — small Host-owned Python HTTP proxy in front of Ollama.
- **Goose ACP** — separate agent process.
- **Dexter MCP tools** — separate Python service with its own dependency environment.

A virtual environment is primarily an on-disk dependency boundary. It does not itself consume a second copy of every package in RAM. RAM cost comes from live processes and the modules/model weights they actually import or allocate.

Separate processes still carry interpreter/private-heap overhead, so each process boundary must justify itself. The TTS boundary is currently justified by long-lived native inference, cancellation, failure isolation, and independent lifecycle. Gateway and Goose have protocol/runtime reasons to remain distinct. The MCP tools process should be reviewed for unnecessary eager imports.

## Known residency findings before measurement

### STT

Dexter creates one Moonshine `Transcriber` during `VoiceInput.start()` and reuses it for all PTT sessions. A PTT press creates a fresh Moonshine stream, not a second model. This is the correct low-duplication shape.

Moonshine's current shipped runtime is intentionally CPU-only. Its `.ort` artifacts contain heavily optimized/fused ONNX Runtime operators; Moonshine's maintainers measured accelerator execution-provider partitioning and found it too fragmented to be beneficial. Moving STT to Arc/NPU is therefore not a first-step optimization and would require a different model-conversion/runtime path, not just a device flag.

### TTS

NeuTTS-2E / Paul is functionally integrated. Current CPU results:

- Q8 GGUF: ~1.11 s first audio, RTF ~1.48, 8 chunks. Live playback starves and is audibly choppy.
- Q4 GGUF: ~1.10 s first audio, RTF ~1.71, 9 chunks. Q4 was slower on this machine.

The likely first accelerator experiment is therefore the **GGUF backbone on the Intel Arc iGPU**, not a larger playback buffer. `llama.cpp` supports Intel GPUs through Vulkan and SYCL. `llama-cpp-python` now publishes a prebuilt Windows Vulkan wheel, so Vulkan should be benchmarked before installing a compiler/oneAPI toolchain for a custom SYCL Python build.

The NeuCodec ONNX decoder is a separate stage. Upstream NeuTTS currently restricts its ONNX codec path to CPU. Moving that decoder to OpenVINO/Arc would require a targeted fork/adapter and separate measurements, so it should not be mixed into the first backbone experiment.

### NPU

The machine runs Windows 10. Intel's current NPU driver/documentation states that Core Ultra NPU workloads are supported on Windows 11, not Windows 10; Windows 10 may expose basic device/driver functionality but cannot run NPU workloads. Do not spend engineering time targeting Dexter workloads at the NPU on the current OS.

If the machine later moves to Windows 11, the NPU can be reconsidered through OpenVINO. Model/operator compatibility still has to be demonstrated per workload rather than assumed.

### MCP tools

`Runtime/Tools/dexter-tools/server.py` currently imports all tool modules when the MCP server starts. This includes modules for image decoding, PyAutoGUI/screen matching, and web search even when a session never invokes those tools.

This is a good candidate for idle-memory work:

1. measure the server's private memory as-is;
2. convert heavyweight optional tool implementations to lazy imports while keeping tool schemas registered;
3. if imported native libraries remain resident after first use and that matters, consider short-lived worker subprocesses only for those heavy operations.

Do not make the whole MCP server demand-start on first tool call unless Goose/ACP tool discovery semantics explicitly support that architecture. The agent must know the tool schema before it can choose a tool.

## Hardware priorities

1. **RTX 5050** — keep the main LLM on CUDA; it is the strongest accelerator and Ollama already uses it.
2. **Intel Arc integrated GPU** — first unused accelerator to benchmark, initially for the NeuTTS GGUF backbone via a prebuilt Vulkan llama.cpp Python wheel.
3. **CPU** — Moonshine STT remains here initially; also handles orchestration and any unsupported operators.
4. **NPU** — unavailable for inference on the current Windows 10 installation.

Do not move the main Gemma model from CUDA to the iGPU merely to use otherwise-idle hardware. Ollama can use Intel GPUs through Vulkan, but a shared-memory iGPU can be substantially slower than CUDA and consumes system RAM. Mixed-device changes need a controlled benchmark and must not make the main LLM slower just to reduce VRAM pressure.

## Measurement workflow

Run `Diagnostics/audit_runtime_performance.ps1` while Dexter is already running and send one representative prompt during the capture window.

The profiler records:

- exact CPU, OS, GPU and visible NPU hardware;
- each Dexter/Ollama/Goose/tool process command line and interpreter path;
- private memory and working set;
- normalized CPU utilization sampled over time;
- Windows GPU-engine activity by PID when counters are available;
- distinct Python executables observed, which exposes process/environment duplication;
- an ignored JSON report for before/after comparisons.

The first audit should be taken before changing MCP imports or the TTS backend so it remains a useful baseline.

## Optimization order after baseline

1. Verify which process is actually consuming CPU during choppy TTS and how much headroom remains.
2. Benchmark NeuTTS Q8 on the Arc iGPU through the prebuilt Vulkan `llama-cpp-python` path, keeping the ONNX decoder unchanged.
3. If Q8 becomes comfortably real-time, keep the quality-preserving Q8 model and integrate device selection explicitly.
4. Profile the MCP tool server's idle footprint; lazy-load heavyweight optional tool modules if meaningful.
5. Profile Host/STT memory and latency. Keep Moonshine resident if its load cost and PTT-latency benefit justify it; only consider unload/reload after measuring both costs.
6. Audit the main LLM's RTX/host-RAM split and context allocation separately. Do not mix that experiment with TTS changes.
7. Only after performance is stable, do the broader idle-memory/efficiency pass.
