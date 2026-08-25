# Dexter TTS Optimization / Voice Runtime Handoff — 2026-08-26

This handoff records the production TTS implementation, performance work, RAM investigation, accepted/rejected optimizations, Windows Application Control fixes, and remaining Voice work completed through 2026-08-26.

**Canonical current functionality:** `Runtime/Voice/README.md`.

**Current branch:** `feat/dexter-v0.3-host` in `Playwhiteshark/dexter`.

## Target machine / constraints

- Windows 11 Home
- Intel Core Ultra 7 255H, 16 logical processors
- Intel Arc 140T iGPU (shared/UMA memory)
- Intel AI Boost NPU
- RTX 5050 Laptop GPU, 8 GB VRAM
- 16 GB system RAM
- Main LLM: Gemma 4 12B through Ollama, with the RTX intentionally reserved for the LLM
- Dexter root environment: uv-managed Python 3.13

Voice must remain free/local, low-resource, streaming, and able to unload completely when Voice Output is disabled. Do not move TTS onto the RTX while the main LLM depends on it without a broader architecture decision.

## Production TTS implementation

Dexter currently uses **NeuTTS-2E**, speaker `paul`, neutral, 24 kHz mono.

Current production path:

- backbone: `neuphonic/neutts-2e-q4-gguf`
- llama.cpp Vulkan on **Intel Arc 140T**, visible device `0`
- all Q4 model layers offloaded to Arc; RTX remains unused by TTS
- llama.cpp `n_batch=128`, `n_ubatch=128`
- upstream `use_mlock=True` retained
- generation threads: 8
- worker process: Windows `ABOVE_NORMAL_PRIORITY_CLASS`
- codec: `neuphonic/neucodec-onnx-decoder-int8` on CPU
- NeuCodec import: Dexter's ONNX-only compatibility module
- Transformers import: Dexter's quantized/GGUF-only compatibility module
- Perth watermark: required; encoder-only Dexter wrapper, detector network omitted
- Perth watermark inference: `torch.inference_mode()`
- Arc generation pipelined with CPU codec + watermark work
- streaming windows: 40 frames first, 50 steady
- playback reserve: 0.5 s
- output: SoundCard/Core Audio/WASAPI, following live Windows default render endpoint
- PTT: hard barge-in; queued/audible speech is invalidated before new capture
- Voice Output Off: shuts down the entire TTS worker and frees model/native runtime resources rather than merely muting audio

Startup should log:

```text
[TTS] NeuCodec ONNX path=lean-compat
[TTS] Transformers path=quantized-compat
[TTS] Perth path=encoder-only
[TTS] llama.cpp backbone batch=128, ubatch=128, mlock=upstream-default
```

Fallback environment switches exist for diagnostics:

```text
DEXTER_TTS_LEAN_NEUCODEC=0
DEXTER_TTS_LEAN_TRANSFORMERS=0
DEXTER_TTS_PERTH_ENCODER_ONLY=0
```

Do not treat those fallbacks as normal production alternatives on the current target machine; upstream Transformers currently reaches a SciPy native module blocked by Windows Application Control.

## Why the original TTS was choppy

The choppiness was real **PCM buffer starvation**, not a cosmetic playback problem.

Early simultaneous Gemma + TTS runs were around **RTF 1.7-1.8**, meaning TTS took ~1.7-1.8 seconds to generate one second of speech. Playback therefore inevitably caught up with generation and ran out of PCM.

The final performance work solved this by improving the actual synthesis pipeline rather than hiding the underrun.

### Performance changes that mattered

1. **Move Q4 GGUF to Arc/Vulkan.**
   - Kept RTX free for Gemma.
   - Native logs verified Arc selection and full layer offload.

2. **Run the real worker AboveNormal.**
   - Applying priority at child-process creation was a large concurrent win.
   - A later redundant self-priority call was removed.

3. **Keep 8 generation threads.**
   - Six threads benchmarked worse.

4. **Pipeline Arc generation with CPU codec/watermarking.**
   - NeuTTS originally serialized token generation and Perth work.
   - `neutts2e_pipelined_stream.py` overlaps independent stages while preserving decode/watermark/crop/overlap-add order.

5. **Constrain ONNX Runtime CPU contention.**
   - Process defaults are set to 2 intra-op / 1 inter-op.
   - Empirically this coincided with the biggest live improvement: roughly **1.26 RTF -> ~0.8 RTF** under concurrent load.
   - Important nuance: direct ORT `SessionOptions` properties still display `0/0`, so state only that the environment configuration was empirically validated; do not claim ORT itself reports 2/1.

6. **Use 40 first / 50 steady frames.**
   - Once steady-state synthesis was faster than realtime, the remaining underrun was an early-buffer problem.
   - A 40-frame initial window supplies ~0.8 s of PCM before switching to 50-frame steady windows.
   - Repeated concurrent tests showed zero measured underflow/refill events in the validated configuration.

Corrected sustained-Gemma batch test:

- 512/512: **0.807 RTF / 1.001 s TTFA**
- 256/256: **0.781 / 1.030 s**
- 128/128: **0.794 / 1.079 s**

128/128 was selected because it kept realtime headroom and saved the most memory.

On an idle final worker, power state matters to speed but not RAM: two identical-request runs measured roughly **0.671 average RTF on battery** versus **0.583 plugged in**. Memory figures were nearly identical, so this is a hardware power-budget effect rather than different Dexter configuration.

## Static RAM cleanup

The initial worker was far too large for a 16 GB system, so a staged memory audit was built rather than guessing from Task Manager.

### Diagnostic lesson: measure the whole process tree

A first whole-worker probe reported only ~5 MB. That was wrong: `Popen.pid` pointed to a tiny uv/Python launcher which spawned the heavy interpreter as a child.

Correct process topology looked like:

```text
small python.exe launcher (~5 MB WS)
└── heavy python.exe NeuTTS worker (~1.5-1.6+ GB WS)
```

All current memory diagnostics enumerate descendants and sum the complete TTS process tree.

### llama.cpp batch buffers

Fresh-process backbone A/B:

- 512/512, mlock true: ~1179.7 MB WS / 1451.5 MB private delta
- 256/256: ~961.7 / 1021.9 MB
- 128/128: ~855.2 / 807.3 MB

128 versus 512 saved about **324.5 MB working set / 644.2 MB private**.

`mlock=False` did **not** help; it slightly worsened measured working set and was rejected.

Production uses an isolated launcher patch rather than modifying installed NeuTTS/llama-cpp packages.

### NeuCodec import cleanup

The actual INT8 codec model is large and mostly unavoidable:

- cached ONNX model: ~297.8 MB
- ONNX Runtime session: ~311-312 MB

Raw ORT and upstream NeuCodec produced almost the same session allocation, proving the majority was real model/session memory rather than package bloat.

Turning off the ORT arena saved only about **0.6 MB WS / 2.5 MB private** and slightly slowed decode, so it was rejected.

However, upstream `neucodec` imports a large unused PyTorch/Transformers/local-model graph even when only the ONNX decoder is selected. `neucodec_onnx_compat.py` removes that eager import cost while exposing exactly the ONNX API NeuTTS needs.

Measured saving:

- ~**52.5 MB working set / 48.1 MB private**
- exact decoded PCM parity
- repeated sustained-Gemma median effectively unchanged: stock 0.772 RTF / 1.114 s TTFA, lean 0.777 / 1.089 s

### Perth detector removal

Stock Perth constructs both the encoder used to insert the watermark and a three-branch decoder used to detect it. Dexter never performs watermark detection in the TTS worker.

Encoder-only A/B:

- removed ~26.7 MB parameters
- saved ~**29.1 MB WS / 32.1 MB private**
- exact output parity
- no full-pipeline performance regression

This became `perth_encoder_only.py`.

### Transformers import decision

NeuTTS imports `AutoTokenizer` and `AutoModelForCausalLM` even when using GGUF/llama.cpp. `transformers_quantized_compat.py` satisfies those imports and fails loudly if the unused PyTorch backbone is requested.

Initial isolated memory result:

- ~37.1 MB WS / 114.2 MB private saved

But a repeated sustained-Gemma benchmark showed a small real median regression:

- full Transformers: 0.781 RTF
- lean shim: 0.813 RTF

Therefore the shim was **originally rejected**; RAM savings alone were not allowed to reduce realtime headroom.

Later, Windows Application Control blocked SciPy through:

```text
NeuTTS -> Transformers -> scikit-learn -> SciPy
```

At that point upstream Transformers became a startup failure on the target machine. Since the GGUF path does not use those HF model classes anyway, quantized compatibility became the correct production default. Re-run the final concurrent Gemma benchmark when convenient because the final production stack differs from the earlier full-Transformers benchmark.

## Windows Application Control / SciPy

Do **not** disable Smart App Control / Windows Application Control for Dexter.

Several third-party packages eagerly import SciPy native `.pyd` modules even though Dexter's production path does not require those code paths. The target machine blocks the unsigned/untrusted native extensions.

Fixes avoid unnecessary imports instead of weakening security:

- `Runtime/Voice/librosa.py`: minimal SciPy-free compatibility layer; Perth resampling uses Torchaudio sinc resampling.
- `perth_encoder_only.py`: Perth's utility module imports `scipy.signal.butter` and `filtfilt` even though insertion never calls them. During Perth import Dexter provides temporary fail-loud placeholders for those names and removes them immediately afterward. If they are ever actually used, Dexter raises instead of silently replacing the math.
- `transformers_quantized_compat.py`: avoids the unused HF/sklearn/SciPy graph entirely for the Q4 GGUF path.

## Repeated-request RAM leak investigation

After the static cleanup, a complete production process-tree measurement showed roughly:

- ready/warmed: ~1.60 GB WS / ~2.58 GB private
- after first synthesis: ~1.65 GB / ~2.68 GB

A five-request test then showed memory **continued rising by ~30-43 MB WS per response**, so this was not ordinary one-time warmup.

### Proving it was a real leak/accumulation

The investigation deliberately narrowed one uncertainty at a time:

1. **Different text/shapes hypothesis:** rejected by running the exact same deterministic utterance ten times.
   - same text
   - same seed 42
   - same sample count
   - same chunk count
   - same PCM SHA-256
   - still ~31-39 MB/request growth

2. **Python object/thread leak hypothesis:** rejected with a Python-vs-native probe.
   - OS growth over six requests: +214.6 MB WS / +236.1 MB private
   - live `tracemalloc` heap: only ~0.2 MB, peak ~0.9 MB
   - active threads: 1 -> 1
   - forced `gc.collect()` only recovered ~2.5 MB
   - therefore the dominant accumulation was native, not a Python queue/generator/thread leak

3. **Perth/autograd hypothesis:** confirmed conclusively.

### Root cause: Perth ran with autograd enabled

Perth was in `.eval()` mode, but `.eval()` does not disable PyTorch autograd. Each streaming watermark call built gradient-tracking state for a backward pass Dexter never performs.

Isolated A/B over 64 identical watermark calls:

- current/autograd enabled: **+244.7 MB WS / +258.4 MB private**
- `torch.inference_mode()`: **+0.3 MB / +0.3 MB**
- exact PCM output match: true
- inference mode was also slightly faster

Production fix: decorate `PerthEncoderOnlyWatermarker.apply_watermark()` with `@torch.inference_mode()`.

### Final whole-worker result after the leak fix

Two 10-request deterministic production runs:

Plugged in:

- ready/warmed: **1502.4 MB WS / 1942.9 MB private**
- request 1: **1533.1 / 2006.4 MB**
- request 10: **1556.1 / 2029.1 MB**
- late per-request growth: ~2.4-2.8 MB
- average RTF: ~0.583

Battery:

- ready/warmed: **1503.9 / 1942.5 MB**
- request 1: **1535.5 / 2006.6 MB**
- request 10: **1558.7 / 2029.6 MB**
- same ~2.5 MB late slope
- average RTF: ~0.671

So the old ~30-40 MB/response leak is fixed. A smaller, reproducible ~2.5 MB/identical-response slope remains.

## Current outstanding work

### P0: residual ~2.5 MB/request memory slope

Tracked in **GitHub issue #10**: `P0: Investigate residual ~2.5 MB/request TTS memory growth`.

For now this has deliberately been parked rather than continuing the investigation in this session.

Already ruled out for the old dominant leak:

- Python heap accumulation
- producer-thread accumulation
- simple GC delay
- varying text/audio/tensor shapes
- Perth autograd (fixed)

Recommended next investigation when returning to #10:

1. run 50-100 identical production requests and determine whether ~2.5 MB/request remains linear or eventually plateaus;
2. if linear, isolate **llama.cpp/Vulkan generation only** and **ONNX NeuCodec decode only** with identical inputs;
3. keep process-tree memory accounting and deterministic hashes;
4. fix the owning lifecycle/cache rather than using forced GC or working-set trimming;
5. re-run sustained Gemma + TTS after any production change.

### P0: text queue overflow/truncation

Still in umbrella issue **#9**.

`VoiceOutput._enqueue_unit()` currently sets the response-wide cancel event if the bounded text queue fills. That can truncate all remaining speech.

Desired behavior:

- bounded coalescing/backpressure/drop-oldest semantics;
- queue pressure must not cancel the entire assistant response;
- add a regression test that fills the queue and proves later speech survives.

### P1: final concurrent benchmark

The last fully corrected sustained-Gemma benchmark predates the final quantized-Transformers + Perth-inference-mode stack. The final stack is fast in idle tests, but run one final sustained Gemma + TTS benchmark and record final concurrent RTF/TTFA before considering performance measurements fully closed.

## Important non-regression decisions

- Keep TTS on Arc, not RTX, while Gemma needs RTX.
- Keep Q4 and 128/128 unless a new benchmark proves a better tradeoff.
- Keep 8 generation threads.
- Keep 40 first / 50 steady windows unless startup/underflow measurements justify a change.
- Keep Perth watermark enabled.
- Keep encoder-only Perth + `torch.inference_mode()`.
- Do not restore broad SciPy/librosa/Transformers imports just to match upstream package structure.
- Do not disable Windows security controls to make third-party `.pyd` files load.
- Do not claim the ONNX ~300 MB session is removable RAM; the model itself is ~298 MB.
- Do not disable ORT arena for a ~1-3 MB saving at the cost of decode speed.
- Do not disable `mlock`; measured no RAM win.
- When benchmarking memory, measure the **entire worker process tree**, not just `Popen.pid`.
- When claiming a performance/memory optimization, use the established workflow: isolate -> output parity -> repeated fresh-process A/B -> sustained Gemma sanity test -> productionize.

## Key files

- `Runtime/Voice/README.md` — canonical architecture, measurements and current Voice behavior.
- `Runtime/Voice/neutts2e_backend.py` — Host-facing backend and production environment defaults.
- `Runtime/Voice/neutts2e_worker_launcher.py` — isolated compatibility surfaces and llama.cpp 128/128 patch.
- `Runtime/Voice/neutts2e_worker.py` — persistent speech worker/protocol.
- `Runtime/Voice/neutts2e_pipelined_stream.py` — Arc generation / CPU codec+Perth pipeline.
- `Runtime/Voice/neucodec_onnx_compat.py` — lean ONNX-only NeuCodec.
- `Runtime/Voice/transformers_quantized_compat.py` — GGUF-only Transformers compatibility.
- `Runtime/Voice/perth_encoder_only.py` — encoder-only Perth, SciPy import avoidance, inference mode.
- `Runtime/Voice/librosa.py` — minimal SciPy-free compatibility surface.
- `Runtime/Voice/windows_default_audio_sink.py` — Core Audio/WASAPI default-device following.
- `Runtime/Voice/tts.py` — text queue / response speech engine; queue overflow bug remains here.
- `Diagnostics/` — retained A/B and process-tree diagnostics.

Particularly useful diagnostics:

- `tts_production_worker_tree_memory.py`
- `tts_production_worker_identical_repeat_memory.py`
- `tts_request_growth_python_native_probe.py`
- `tts_perth_autograd_growth_ab.py`
- batch/codec/Transformers/Perth A/B scripts retained in `Diagnostics/`

## Key commits from the optimization work

Selected landmarks, not an exhaustive history:

- `d51a8d68...` — smooth concurrent production settings / 40-first 50-steady work
- `1dda45ed...` — launch actual TTS worker AboveNormal; Q4 default
- `b025f6d3...` — SciPy-free librosa/Torchaudio compatibility path
- `a4b92fd3...` / `12d1870...` — isolated launcher + production 128/128 llama.cpp batch
- `1a785dfa...` — lean ONNX NeuCodec production path
- `66e834b5...` — encoder-only Perth production path
- `6ad17f66...` — avoid Perth's eager SciPy import under Windows Application Control
- `980fdd04...` — disable Perth autograd with `torch.inference_mode()`
- `92a6dfee...` — quantized-only Transformers production path
- `009a1c48...` — current Voice README documentation refresh

## Current tracker state

- **#10 — P0:** residual ~2.5 MB/request TTS memory growth.
- **#9 — Voice hardening umbrella:** major memory/performance work complete; queue-overflow fix remains P0; final-stack sustained-Gemma benchmark remains P1.
- **#7:** Windows output routing completed/closed.

## Recommended next order

1. Leave #10 parked unless long-running RAM growth becomes immediately painful.
2. Fix the **TTS text-queue overflow/truncation** behavior in #9.
3. Run one final sustained-Gemma benchmark on the exact final production stack.
4. Move on to the broader Dexter resource audit / generalized tools / context roadmap unless new Voice regressions appear.
