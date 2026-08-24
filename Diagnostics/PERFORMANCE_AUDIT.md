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

NeuTTS-2E / Paul is integrated and the original choppy playback has now been traced to sustained synthesis starvation rather than corrupt PCM.

Initial CPU measurements:

- Q8 GGUF: roughly RTF 1.5-1.6, about 1.1 s first audio.
- Q4 GGUF: roughly RTF 1.7 and slower than Q8 on this machine.
- forcing 16 llama.cpp generation threads was slightly slower than NeuTTS's default eight threads.

A raw streamed-PCM capture sounded smooth, proving that NeuTTS/NeuCodec generated valid audio. The live cuts occurred because Q8 CPU generation produced each nominal 0.5 s audio block in roughly 0.7-0.85 s, so any finite playback reserve eventually ran dry.

Stage profiling then showed NeuCodec was only about 8% of synthesis time; the GGUF/token path dominated. That made the otherwise-idle Arc iGPU the highest-leverage accelerator target.

#### Verified Intel Arc path

The project now uses the prebuilt Vulkan `llama-cpp-python` wheel. Native llama.cpp logs verified the real accelerator path rather than relying on a device flag alone:

- `Vulkan0 = Intel(R) Arc(TM) 140T GPU`;
- `matrix cores: KHR_coopmat`;
- 29/29 model layers offloaded to Vulkan0;
- the KV cache is on Vulkan0;
- the RTX 5050 is not used by TTS.

Serial Arc + Perth improved the clean benchmark from roughly RTF 1.64 on CPU to **RTF 1.218**, with first audio around **0.703 s**. That was a substantial speedup but still not enough for indefinitely smooth playback.

A no-Perth diagnostic reached **RTF 0.826**, proving the Arc hardware itself was already fast enough and exposing synchronous Perth watermarking as the hidden serial bottleneck. Production does not remove Perth.

#### Pipelined Perth production path

NeuTTS 1.4.1 normally performs Arc GGUF generation, CPU ONNX decode, Perth watermarking, and overlap-add serially. These are partly independent stages on different hardware, so Dexter now pipelines them:

- a bounded producer thread continues GGUF speech-token generation on Arc;
- the consumer performs the same CPU NeuCodec decode;
- the same Perth watermark is applied;
- the same crop and overlap-add order is preserved;
- chunks remain ordered and cancellation bounds speculative work.

The verified pipelined result with Perth still enabled was:

- **RTF 0.782**;
- **first audio 0.775 s**;
- average post-first-chunk surplus about **0.123 s per nominal 0.5 s chunk**;
- worst measured steady-state deficit only **+0.004 s**.

This is the selected production TTS path: Q8 backbone on Intel Arc 140T Vulkan0, NeuCodec + Perth on CPU, pipelined scheduling.

Production live playback remains callback-driven through PortAudio, but the buffer is now a jitter reserve rather than a workaround for permanently slow synthesis. The default startup/rebuffer reserve was reduced from 1.0/0.75 s to **0.5/0.5 s**. PTT interruption still aborts output and clears buffered PCM immediately.

### NPU

The Intel AI Boost NPU is present on Windows 11 and can be reconsidered through OpenVINO. Availability alone does not make a model a good NPU target: operator coverage, graph partitioning, conversion cost, latency and memory need to be measured per workload.

For the current stack, NeuTTS's GGUF backbone is a llama.cpp workload rather than an OpenVINO graph. The ONNX codec is now a small minority of TTS time and no longer a priority after the Arc + pipeline result. Future converted models remain more natural NPU experiments.

### MCP tools

`Runtime/Tools/dexter-tools/server.py` currently imports all tool modules when the MCP server starts. This includes modules for image decoding, PyAutoGUI/screen matching, and web search even when a session never invokes those tools.

The first audit measured the real MCP tools Python process at roughly 565 MB private memory while idle. This is a meaningful candidate for idle-memory work.

A good direction is:

1. convert heavyweight optional tool implementations to lazy imports while keeping tool schemas registered;
2. measure the idle process again;
3. if imported native libraries remain resident after first use and that matters, consider short-lived worker subprocesses only for those heavy operations.

Do not make the whole MCP server demand-start on first tool call unless Goose/ACP tool-discovery semantics explicitly support that architecture. The agent must know the tool schema before it can choose a tool.

## Process-wrapper note

The audit shows tiny ~0.8 MB venv launcher processes plus the real base-Python child process for several uv/venv launches. Those launchers are negligible. The significant resident allocations are in the base interpreter/model-owning processes: roughly 1.4 GB private for Host+STT, roughly 3.0 GB private for TTS in the original CPU-era audit, and roughly 565 MB private for MCP tools.

Do not interpret the number of `python.exe` entries alone as duplicated model weights; inspect command lines, parent PIDs and actual private-memory owners.

A fresh full-app audit should be taken after the new Arc/pipelined TTS path is live because CPU utilization and TTS residency may now differ materially from the original baseline.

## Hardware priorities

1. **RTX 5050** — keep the main LLM on CUDA unless a controlled benchmark proves a better split.
2. **Intel Arc 140T** — now owns the NeuTTS Q8 GGUF backbone and has converted TTS from slower-than-real-time to RTF ~0.78 with pipelined Perth.
3. **Intel AI Boost NPU** — valid Windows 11 accelerator, to test through OpenVINO only where model/operator support makes sense.
4. **CPU** — Moonshine STT, NeuCodec, Perth watermarking, orchestration, and unsupported operators.

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

## Optimization order after TTS fix

1. Validate the production Arc + pipelined-Perth path in the live Host: smoothness, first-audio latency, cancellation, and response boundaries.
2. Take a fresh full-app performance audit with the new TTS path and compare CPU/RAM/GPU residency against the original baseline.
3. Profile and lazy-load heavyweight optional MCP tool implementations.
4. Profile Host/STT memory and latency. Keep Moonshine resident if its load cost and PTT-latency benefit justify it; only consider unload/reload after measuring both costs.
5. Audit the main LLM's RTX/host-RAM split and context allocation separately. Do not mix that experiment with TTS changes.
6. Test NPU-suitable OpenVINO workloads only after higher-confidence memory/latency wins are exhausted.
7. Only after interactive performance is stable, do the broader idle-memory/efficiency pass.
