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

## Baseline findings

### Machine

The first runtime audit identified:

- Windows 11 Home, build 26200;
- Intel Core Ultra 7 255H, 16 cores / 16 logical processors;
- NVIDIA GeForce RTX 5050 Laptop GPU;
- Intel Arc 140T integrated GPU;
- Intel AI Boost exposed as a healthy `ComputeAccelerator` NPU;
- 16 GB system RAM.

The earlier Windows-10 assumption was stale. The NPU is therefore a valid future OpenVINO target, subject to per-model/operator compatibility and benchmark results.

### STT

Dexter creates one Moonshine `Transcriber` during `VoiceInput.start()` and reuses it for all PTT sessions. A PTT press creates a fresh Moonshine stream, not a second model. This is the correct low-duplication shape.

Moonshine's current shipped runtime is intentionally CPU-only. Its `.ort` artifacts contain heavily optimized/fused ONNX Runtime operators; Moonshine's maintainers measured accelerator execution-provider partitioning and found it too fragmented to be beneficial. Moving STT to Arc/NPU is therefore not a first-step optimization and would require a different model-conversion/runtime path, not just a device flag.

### TTS

NeuTTS-2E / Paul is functionally integrated. Current CPU results:

- Q8 GGUF: ~1.11 s first audio, RTF ~1.48, 8 chunks.
- Q4 GGUF: ~1.10 s first audio, RTF ~1.71, 9 chunks. Q4 was slower on this machine.

The first full-app audit showed the NeuTTS worker reaching roughly 74% whole-machine CPU while speaking and no GPU-engine activity. This makes TTS the strongest first candidate for offloading to otherwise-idle accelerator hardware.

The audible stop/start needs to be isolated before changing playback policy. A dedicated stream-capture diagnostic records the exact yielded PCM without live playback and logs per-chunk arrival time versus audio duration. If the recorded WAV is choppy, the defect is already in the NeuTTS streaming/codec path; if the WAV is smooth, Dexter's playback scheduling is at fault.

The likely first accelerator experiment remains the **GGUF backbone on the Intel Arc iGPU**. `llama.cpp` supports Intel GPUs through Vulkan and SYCL. Vulkan should be benchmarked before installing a compiler/oneAPI toolchain for a custom SYCL Python build.

The NeuCodec ONNX decoder is a separate stage. Upstream NeuTTS currently restricts its ONNX codec path to CPU. Moving that decoder to OpenVINO/Arc/NPU would require a targeted adapter or fork and separate measurements, so it should not be mixed into the first backbone experiment.

### NPU

The Intel AI Boost NPU is present on Windows 11 and can be reconsidered through OpenVINO. Availability alone does not make a model a good NPU target: operator coverage, graph partitioning, conversion cost, latency and memory need to be measured per workload.

For the current stack, NeuTTS's GGUF backbone is a llama.cpp workload rather than an OpenVINO graph, so the Arc iGPU is the simpler first accelerator target. The ONNX codec and future converted models are more natural candidates for later NPU experiments.

### MCP tools

`Runtime/Tools/dexter-tools/server.py` currently imports all tool modules when the MCP server starts. This includes modules for image decoding, PyAutoGUI/screen matching, and web search even when a session never invokes those tools.

The first audit measured the real MCP tools Python process at roughly 565 MB private memory while idle. This is a meaningful candidate for idle-memory work.

A good direction is:

1. convert heavyweight optional tool implementations to lazy imports while keeping tool schemas registered;
2. measure the idle process again;
3. if imported native libraries remain resident after first use and that matters, consider short-lived worker subprocesses only for those heavy operations.

Do not make the whole MCP server demand-start on first tool call unless Goose/ACP tool-discovery semantics explicitly support that architecture. The agent must know the tool schema before it can choose a tool.

## Process-wrapper note

The audit shows tiny ~0.8 MB venv launcher processes plus the real base-Python child process for several uv/venv launches. Those launchers are negligible. The significant resident allocations are in the base interpreter/model-owning processes: roughly 1.4 GB private for Host+STT, roughly 3.0 GB private for TTS, and roughly 565 MB private for MCP tools in the first audit.

Do not interpret the number of `python.exe` entries alone as duplicated model weights; inspect command lines, parent PIDs and actual private-memory owners.

## Hardware priorities

1. **RTX 5050** — keep the main LLM on CUDA unless a controlled benchmark proves a better split.
2. **Intel Arc integrated GPU** — first unused accelerator to benchmark, initially for the NeuTTS GGUF backbone.
3. **Intel AI Boost NPU** — valid Windows 11 accelerator, to test through OpenVINO only where model/operator support makes sense.
4. **CPU** — Moonshine STT remains here initially; also handles orchestration and unsupported operators.

Do not move the main Gemma model from CUDA to the iGPU merely to use otherwise-idle hardware. A shared-memory iGPU consumes system RAM and may be substantially slower than CUDA. Mixed-device changes need a controlled benchmark and must not make the main LLM slower just to reduce VRAM pressure.

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

The profiler ending does not stop Dexter or TTS; it only stops recording samples.

## Optimization order after baseline

1. Isolate whether the current TTS choppiness is encoded in NeuTTS's yielded audio or introduced by live playback.
2. Benchmark NeuTTS Q8 on the Arc iGPU, keeping the ONNX decoder unchanged initially.
3. If Q8 becomes comfortably real-time and audio quality is clean, integrate explicit device selection.
4. Profile and lazy-load heavyweight optional MCP tool implementations.
5. Profile Host/STT memory and latency. Keep Moonshine resident if its load cost and PTT-latency benefit justify it; only consider unload/reload after measuring both costs.
6. Audit the main LLM's RTX/host-RAM split and context allocation separately. Do not mix that experiment with TTS changes.
7. Test NPU-suitable OpenVINO workloads only after the higher-confidence Arc/TTS path is understood.
8. Only after performance is stable, do the broader idle-memory/efficiency pass.
