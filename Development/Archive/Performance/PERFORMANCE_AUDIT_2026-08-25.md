# Dexter runtime performance audit

Status: active performance work, 2026-08-25.

The priority is **interactive performance first**, then idle-memory efficiency. This document records the current production topology and the next useful measurements; intermediate TTS experiments live in Git history rather than the active tree.

## Machine

- Windows 11 Home, build 26200
- Intel Core Ultra 7 255H, 16 cores / 16 logical processors
- Intel Arc 140T integrated GPU
- Intel AI Boost NPU
- NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM
- 16 GB system RAM

## Current runtime topology

- **Ollama model runner** — main Gemma model, primarily RTX/CUDA.
- **Dexter Host** — Qt UI plus one persistent Moonshine `Transcriber`.
- **NeuTTS worker** — one Host-owned child process using the same root uv/Python 3.13 environment as the Host **only while Voice output is enabled/warming**.
- **Gateway** — Host-owned Python HTTP proxy in front of Ollama.
- **Goose ACP** — separate agent process.
- **Dexter MCP tools** — separate Python service with its own dependency environment.

A virtual environment is an on-disk dependency boundary, not a RAM allocation by itself. Live process/module/model allocations are what matter.

## STT

Dexter creates one Moonshine `Transcriber` during `VoiceInput.start()` and reuses it for all PTT sessions. A PTT press creates a fresh stream, not another model.

Moonshine's current shipped runtime is intentionally CPU-oriented. Its fused ONNX artifacts have historically partitioned poorly across alternate execution providers, so Arc/NPU STT migration is not currently a first-priority optimization.

## TTS: selected production path

Production TTS is NeuTTS-2E / Paul with:

- Q8 GGUF backbone
- Intel Arc 140T via llama.cpp Vulkan device 0
- 29/29 model layers offloaded to Arc in verified native logs
- NeuCodec ONNX decoder on CPU
- Perth watermarking enabled on CPU
- Arc token generation pipelined with CPU decode/watermark work
- callback-driven 24 kHz mono PortAudio playback
- pre-opened/reused output stream
- 0.5 s playback jitter reserve
- 25-frame first streaming window, then 50-frame steady windows

The RTX 5050 remains reserved for the main LLM rather than sharing TTS load.

### Why this path

The original Q8 CPU path was slower than real time at roughly RTF 1.5-1.6. Q4 was slower still, and forcing 16 generation threads was slightly worse than NeuTTS's default eight.

Moving the GGUF backbone to Arc helped substantially, but serial Arc generation + CPU Perth remained slower than playback at roughly RTF 1.22. A no-Perth diagnostic proved Arc itself was fast enough. Pipelining independent Arc generation with CPU codec/Perth work produced a validated real-time path around **RTF 0.782** while retaining the watermark.

The overlap-add implementation was then changed from full-history recomputation to an incremental equivalent, with numerical regression tests against the reference behavior.

### Hybrid streaming windows

A later controlled benchmark measured:

- 25 frames / 0.5 s chunks: **RTF 0.805**, worst chunk deficit **+0.113 s**
- 40 frames / 0.8 s chunks: **RTF 0.773**, worst chunk deficit **-0.008 s**
- 50 frames / 1.0 s chunks: **RTF 0.733**, worst chunk deficit **-0.067 s**

Starting at 50 frames would add roughly 0.3 s to first audio, so production uses **25 frames for the first decode and 50 frames thereafter**. This keeps initial latency close to the faster 25-frame path while reducing repeated codec/watermark overhead enough to give sustained playback useful headroom.

### Playback latency

Live tracing previously showed almost a one-second gap between first PCM and actual playback because PortAudio opened lazily on the critical path. The production 24 kHz mono stream is now pre-opened during TTS startup and reused. A later live trace measured only about **15 ms** from first PCM to playback start.

Sentence boundaries can still incur a new NeuTTS prompt/prefill. `VoiceOutput` therefore non-blockingly coalesces already-queued complete sentences after an active synthesis request, without reintroducing the failed short-clause splitting experiment.

### Text normalization

NeuTTS-2E accepts raw written text rather than SSML/phoneme overrides. Production now places a very thin text-normalization wrapper immediately before synthesis:

- `utterwise==0.1.0`, pinned to the evaluated version
- pure-Python core with no runtime dependency stack
- semantic normalization for acronyms, numbers, dates, versions, currency, percentages, temperatures, etc.
- Dexter-only cleanup for single-underscore emphasis, `snake_case`, and word-style hyphens
- source punctuation retained because NeuTTS uses punctuation for prosody
- measured pronunciation policy: `AI -> A I`, `JSON -> jay son`; ordinary acronyms such as SQL/CPU/GPU remain Utterwise's job

There is no extra model, phonemizer, service, worker process, or persistent background component for this layer.

### Voice-off resource mode

Voice output is now a persistent resource mode rather than a volume mute. The secondary header overflow menu can fully disable TTS.

When disabled, `SpeechOutputService` immediately stops routing response text to speech and asynchronously stops the active `VoiceOutput`. The NeuTTS child process is shut down, so its model-owned Arc/CPU/RAM allocations are released. Re-enable creates a fresh output/worker and performs the normal warm-up; Dexter/Goose/STT stay usable while that happens.

The setting persists. If Voice was left off, the next Dexter launch does not start the NeuTTS worker at all. Warm-up can also be cancelled by turning Voice off while it is still starting. The UI does not expose OFF until the retired startup/shutdown path has completed, preventing old and new TTS workers from overlapping.

This creates two useful measurement baselines:

1. **Voice ON / ready** — full production Dexter with the NeuTTS worker resident.
2. **Voice OFF** — identical Dexter session after the worker/model has been unloaded.

The delta between those captures is now the cleanest way to measure TTS's real resident RAM/CPU/Arc cost instead of inferring it from process names.

## NPU

Intel AI Boost remains available for future OpenVINO experiments, but availability alone is not a reason to move a workload. The current NeuTTS backbone is a llama.cpp/Vulkan workload, while the ONNX codec is already a minority of TTS cost. NPU work should wait for a model/operator combination where the expected gain is material.

## MCP tools: next memory target

`Runtime/Tools/dexter-tools/server.py` still eagerly imports tool modules, including image/screen/automation-related dependencies that many sessions never use.

The previous audit measured the real MCP tools Python process at roughly **565 MB private memory while idle**. The next useful optimization pass is:

1. convert heavyweight optional implementations to lazy imports while keeping schemas registered;
2. take another idle measurement;
3. consider short-lived helper processes only where native heaps remain resident after first use and the memory saving justifies the complexity.

Do not demand-start the entire MCP server unless ACP/tool-discovery semantics are first verified to support it.

## Current measurement workflow

Run:

```powershell
powershell -ExecutionPolicy Bypass -File Diagnostics\audit_runtime_performance.ps1
```

while Dexter is already running, and send one representative prompt during the capture window.

The profiler records CPU/OS/hardware, process command lines/interpreters, private memory and working set, sampled CPU utilization, visible GPU-engine activity when Windows counters provide it, and an ignored JSON report for before/after comparisons.

For the next memory pass, capture once with **Voice ready**, then turn Voice off and wait until the menu reports it off / the console reports model resources released, and capture again under otherwise comparable idle conditions. That isolates the TTS resident-cost delta before changing any other subsystem.

For the production voice path specifically:

```bat
uv run python Runtime\Voice\neutts2e_runtime.py doctor
uv run python Runtime\Voice\neutts2e_runtime.py smoke
```

Set `DEXTER_TTS_TRACE=1` only when diagnosing live first-audio/underflow behavior. The trace reports the normalized synthesis text actually reaching NeuTTS.

## Optimization order

1. Measure current full-app **Voice ON vs Voice OFF** RAM/CPU/GPU residency now that TTS can be cleanly unloaded.
2. Measure and reduce MCP idle memory through lazy imports.
3. Re-profile Host/STT memory and latency after MCP work.
4. Audit the main LLM's RTX/host-RAM split separately; do not mix that experiment with voice changes.
5. Test NPU/Arc alternatives only when a specific workload/build has plausible operator coverage and a controlled benchmark plan.
6. After interactive performance is stable, do the broader idle-memory/efficiency pass.
