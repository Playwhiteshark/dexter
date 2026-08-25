# Dexter Voice Runtime

Dexter's selected production TTS is **NeuTTS-2E**, speaker **paul**, neutral emotion.

> **Current production status (2026-08-26):** the original live TTS choppiness/starvation problem is solved on the target machine, and the major TTS memory cleanup is now productionized. The current worker uses the Q4 GGUF backbone on the Intel Arc 140T, 128/128 llama.cpp batch/ubatch, 8 generation threads, Windows AboveNormal priority, a lean ONNX-only NeuCodec compatibility surface, a quantized-only Transformers compatibility surface, encoder-only Perth watermarking under `torch.inference_mode()`, 40-frame first / 50-frame steady streaming, and Core Audio/WASAPI default-device following through SoundCard. The production worker now starts around **1.50 GB working set / 1.94 GB private commit** and is around **1.53-1.56 GB / 2.01-2.03 GB** after repeated deterministic requests on the target machine. The former Perth/autograd leak of roughly **30-40 MB per spoken response** has been eliminated; a much smaller reproducible residual growth of about **2.5 MB per identical response** remains tracked as a high-priority follow-up.

## Runtime architecture

Dexter is a single uv-managed Python application pinned to **Python 3.13** at the repository root. Host, STT, and TTS dependencies are resolved from the root `pyproject.toml` into one root `.venv`.

TTS runs in a child process, but that child uses the **same project interpreter and dependency set as the Host**. On Windows/uv there can be a tiny launcher `python.exe` (about 5 MB working set) which owns the real heavy interpreter as a child. Memory diagnostics therefore measure the **complete process tree**, not only `Popen.pid`.

The process boundary exists so the Host owns TTS lifecycle/cancellation and native-runtime isolation; it is not a second Python installation or TTS-specific virtual environment.

Machine-local model/auth/cache data stays outside the repo at:

```text
%LOCALAPPDATA%\Dexter\Voice\neutts2e\cache
```

## Current production path

Production uses:

- backbone: `neuphonic/neutts-2e-q4-gguf`
- backbone device: **Intel Arc 140T** via llama.cpp Vulkan, visible device `0`
- RTX 5050: deliberately left to Gemma/Ollama; TTS does not use it
- llama.cpp batch / ubatch: **128 / 128**
- llama.cpp `use_mlock=True`: retained from upstream after A/B testing showed disabling it did not improve memory
- generation threads: **8**
- codec: `neuphonic/neucodec-onnx-decoder-int8` on CPU
- NeuCodec import path: Dexter's **ONNX-only compatibility surface** (`neucodec_onnx_compat.py`)
- Transformers import path: Dexter's **quantized-only compatibility surface** (`transformers_quantized_compat.py`); the GGUF path never uses `AutoTokenizer` or `AutoModelForCausalLM`
- Perth watermarking: enabled through Dexter's **encoder-only** wrapper; the unused detector network is not constructed
- Perth inference: wrapped in **`torch.inference_mode()`** so watermarking does not build autograd graphs
- scheduling: Arc speech-token generation is pipelined with CPU codec/watermark work
- ONNX Runtime environment defaults: **2 intra-op / 1 inter-op**; this setting was chosen from empirical target-machine A/B results, although direct `SessionOptions` introspection still reports `0/0`, so do not claim ORT itself reports the 2/1 values
- streaming windows: **40 frames first / 50 frames steady**
- worker priority: Windows `ABOVE_NORMAL_PRIORITY_CLASS` at process creation
- output routing: SoundCard/Core Audio/WASAPI follows the live Windows default render endpoint
- text normalization: `utterwise==0.1.1` plus a small Dexter pronunciation/markup policy
- speaker: `paul`
- emotion: `neutral`
- sample rate: **24 kHz mono**
- Perth watermark: required and preserved

The GGUF backbone is intentional. NeuTTS implements `infer_stream()` for its quantized llama.cpp path, whereas its PyTorch backbone path only returns complete utterances.

The production launcher logs its active compatibility/tuning choices, for example:

```text
[TTS] NeuCodec ONNX path=lean-compat
[TTS] Transformers path=quantized-compat
[TTS] Perth path=encoder-only
[TTS] llama.cpp backbone batch=128, ubatch=128, mlock=upstream-default
```

Diagnostic fallbacks remain available:

```text
DEXTER_TTS_LEAN_NEUCODEC=0
DEXTER_TTS_LEAN_TRANSFORMERS=0
DEXTER_TTS_PERTH_ENCODER_ONLY=0
```

The upstream Transformers fallback is not a usable normal path on the current target machine while Windows Application Control blocks the SciPy native extension reached through `Transformers -> scikit-learn -> SciPy`.

## Performance work and choppiness fix

The original CPU/Q8 and early concurrent configurations synthesized too slowly for uninterrupted playback. Under simultaneous Gemma load, TTS could reach roughly **RTF 1.7-1.8**, producing real PCM starvation rather than merely UI/audio timing jitter.

The production performance path was reached incrementally:

1. **Intel Arc Vulkan offload.** Native llama.cpp logs confirmed the Arc 140T was selected, 29/29 model layers were offloaded, KV lived on Vulkan, and the RTX 5050 stayed free for the main LLM.
2. **Q4 backbone.** Q4 was only a modest speed win versus Q8 by itself, but reduced pressure and became the production default.
3. **AboveNormal worker priority.** Creating the actual TTS worker at Windows `ABOVE_NORMAL_PRIORITY_CLASS` materially improved concurrent throughput. Six generation threads were worse than the production eight, so 8 remains the default.
4. **Pipeline independent devices.** NeuTTS originally serialized Arc token generation with CPU NeuCodec/Perth work. `neutts2e_pipelined_stream.py` overlaps those stages while preserving decode, watermark, crop and overlap-add ordering.
5. **Bound ONNX CPU contention.** The process sets ORT thread defaults to 2 intra-op / 1 inter-op. On the target machine this coincided with the largest remaining live improvement: concurrent TTS moved from roughly **1.26 RTF to around 0.8 RTF**.
6. **40/50 streaming windows.** Once overall synthesis was faster than realtime, the remaining audible issue was startup buffer starvation. A 40-frame first window gives about 0.8 s of initial PCM; 50-frame steady windows preserve throughput. This removed the measured early underflows while keeping first audio around one second under concurrent load.

A corrected sustained-contention benchmark where Gemma remained active throughout the complete TTS synthesis measured approximately:

- 512/512 batch: **0.807 RTF / 1.001 s TTFA**
- 256/256: **0.781 / 1.030 s**
- 128/128: **0.794 / 1.079 s**

Because 128/128 preserved realtime headroom while saving the most memory, it became production.

On an otherwise idle worker with the final stack, AC power materially improves throughput but not memory use. Two identical-request runs measured roughly **0.671 average RTF on battery** versus **0.583 plugged in**, while worker memory was effectively unchanged. This is consistent with laptop CPU/iGPU power limits rather than a software configuration change.

## Memory cleanup: accepted production changes

### llama.cpp batch / ubatch: 512 -> 128

A staged constructor probe identified the Q4 llama.cpp backbone as the largest avoidable allocation. Fresh-process A/B results showed:

- upstream-like 512/512: backbone delta roughly **1179.7 MB working set / 1451.5 MB private**
- 256/256: roughly **961.7 / 1021.9 MB**
- 128/128: roughly **855.2 / 807.3 MB**

Relative to 512/512, 128/128 saved about **324.5 MB working set / 644.2 MB private**. Isolated and corrected concurrent benchmarks showed no meaningful throughput regression, so 128/128 is now applied only inside the isolated TTS child by `neutts2e_worker_launcher.py`.

### Lean ONNX-only NeuCodec import path

Upstream `neucodec` eagerly imports Torch, Torchaudio, Transformers and local encoder/decoder modules even when NeuTTS selects only the precompiled ONNX decoder. Dexter's `neucodec_onnx_compat.py` exposes exactly the ONNX API NeuTTS needs and fails loudly if a PyTorch codec path is requested.

Measured isolated saving:

- about **52.5 MB working set / 48.1 MB private**
- byte-for-byte identical decoded PCM
- repeated sustained-Gemma full-pipeline medians effectively unchanged: stock **0.772 RTF / 1.114 s TTFA**, lean **0.777 / 1.089 s**

### Encoder-only Perth

Stock Perth constructs both:

- an encoder used to **insert** the watermark; and
- a three-branch decoder used to **detect/read** it.

Dexter only inserts watermarks. Loading only the encoder removed about **26.7 MB of unused parameters**, saving approximately **29.1 MB working set / 32.1 MB private**. Output was byte-for-byte identical, and sustained-Gemma testing was non-regressive.

### Perth autograd leak fix

This was the main long-running memory bug.

`Module.eval()` does **not** disable PyTorch gradient tracking. Perth's watermark insertion path therefore built autograd graphs on every streaming watermark invocation even though Dexter never trains the model or calls backward.

The full production worker initially showed a repeatable **30-40 MB working-set/private increase per spoken response**, even when ten requests used the exact same text, seed, sample count, chunk geometry and PCM hash.

An isolated Perth A/B reproduced the problem directly over 64 identical watermark calls:

- normal autograd-enabled execution: **+244.7 MB WS / +258.4 MB private**
- `torch.inference_mode()`: **+0.3 MB / +0.3 MB**
- exact watermark output: **identical**

`PerthEncoderOnlyWatermarker.apply_watermark()` is therefore permanently decorated with `@torch.inference_mode()`.

After the fix, the full production identical-request test changed from roughly **+31-39 MB per late response** to roughly **+2.4-2.8 MB per response**. The large leak is considered fixed; the smaller residual slope is tracked separately as a high-priority issue.

### Quantized-only Transformers compatibility

NeuTTS imports `AutoTokenizer` and `AutoModelForCausalLM` unconditionally, but the production GGUF path never uses them. Dexter already had a strict compatibility shim which exposes those two names and fails loudly if a PyTorch/Hugging Face backbone is attempted.

An earlier memory/performance A/B found the shim saved about **37.1 MB working set / 114.2 MB private**, but repeated sustained-Gemma testing showed a small median throughput regression (about **0.781 -> 0.813 RTF**), so it was initially rejected.

That tradeoff changed when Windows Application Control began blocking the SciPy native extension reached through:

```text
NeuTTS -> Transformers -> scikit-learn -> SciPy
```

Full Transformers therefore became a startup failure on the target machine. The quantized compatibility path is now production because it both matches the actual GGUF functionality and avoids importing unused sklearn/SciPy code. Re-run the sustained-Gemma benchmark if the final-stack throughput needs to be compared with the older full-Transformers result.

## What was measured and found not worth / not possible to save

Memory work deliberately used isolated A/B probes rather than assuming every large process number was removable.

### ONNX codec model/session is mostly irreducible

The cached INT8 NeuCodec ONNX model is about **297.8 MB**. Fresh-process measurements showed the ONNX Runtime session itself costs roughly **311-312 MB working set/private**, whether reached through stock NeuCodec or a raw ORT session. That means most of this allocation is the actual compiled model/session, not Python import waste.

Disabling the ORT CPU memory arena saved only around **0.6 MB working set / 2.5 MB private** and slightly slowed decoding, so it was rejected. The lean NeuCodec work targets eager Python/package imports, not the necessary ~300 MB decoder model.

### `use_mlock=False` did not help

Backbone A/B testing with `mlock` disabled did not reduce memory; in the measured configurations it slightly increased working set. Production therefore keeps the upstream `use_mlock=True` behavior.

### Transformers shim was originally rejected on performance

The quantized-only shim was not accepted simply because it saved RAM. It was initially rejected because the repeated concurrent benchmark showed a real ~4% median RTF penalty. It became production later only because the upstream dependency chain was blocked by Windows Application Control and was functionally unusable on the target machine.

### Process-tree accounting matters

An early whole-worker diagnostic incorrectly reported about **5 MB** because it measured the small Windows/uv launcher PID returned by `Popen`, not the heavy interpreter underneath it. The corrected diagnostics enumerate descendants and sum the complete process tree. Do not use only the root `Popen.pid` when assessing TTS memory on this setup.

### Python heap was not the repeated-request leak

When the worker was growing by hundreds of MB, `tracemalloc` showed only about **0.2 MB** of live Python-heap growth with a peak below **0.9 MB**. Active threads stayed **1 -> 1**, and forced `gc.collect()` reclaimed only about 2.5 MB. This proved the main accumulation lived in a native runtime rather than Python queues, generators or leaked producer threads.

Repeating the exact same deterministic utterance then ruled out varying shapes. Isolating Perth with and without autograd identified the native cause conclusively.

## Current measured footprint

With the final compatibility stack and Perth inference-mode fix, two complete identical-request production runs measured approximately:

- ready/warmed: **1502-1504 MB working set / ~1943 MB private**
- after request 1: **1533-1536 MB / ~2006-2007 MB**
- after request 10: **1556-1559 MB / ~2029-2030 MB**

The first real request still causes a normal one-time native-runtime increase of roughly **31 MB WS / 64 MB private**.

Requests 2-10 then show a small highly repeatable residual slope of roughly **2.5 MB per identical request**. This is dramatically smaller than the fixed Perth/autograd leak but is still real enough to warrant follow-up rather than being dismissed as working-set noise.

For historical comparison, an earlier staged full constructor/synthesis probe before the major cleanup ended around **1996 MB working set / 3364 MB private**. The exact figures are not perfectly additive across isolated experiments because Windows working sets, shared DLL pages, native allocators and Arc UMA interact, so current whole-worker measurements are the preferred final numbers.

## Windows Application Control / SciPy avoidance

Do **not** disable Smart App Control / Windows Application Control for Dexter.

Two unused eager dependency paths have triggered blocked unsigned SciPy `.pyd` files on the target machine:

1. real `librosa` / SciPy imports in the broader audio stack;
2. Perth's utility module eagerly importing `scipy.signal.butter`/`filtfilt`, plus upstream Transformers reaching SciPy through scikit-learn.

Production avoids those unnecessary imports instead:

- `Runtime/Voice/librosa.py` provides the tiny surface NeuTTS/Perth needs and uses Torchaudio sinc resampling;
- `perth_encoder_only.py` temporarily satisfies Perth's two unused Butterworth names with fail-loud placeholders during import, then removes those placeholder modules so they cannot shadow a legitimate later SciPy import;
- `transformers_quantized_compat.py` avoids the unused Hugging Face/sklearn/SciPy graph entirely for the GGUF path.

If Perth ever actually calls the placeholder Butterworth helpers, Dexter deliberately raises rather than silently changing watermark math.

## Speech text normalization

NeuTTS-2E accepts raw written text rather than phonemes/SSML. Production normalizes text immediately before synthesis.

`Runtime/Voice/text_normalization.py` is intentionally thin:

1. `SpeechFormatter` removes fenced code, Markdown presentation syntax and raw URLs that should not be read literally;
2. Dexter converts single-underscore emphasis, `snake_case`, and word-style hyphens to ordinary word boundaries while leaving negatives, dates and numeric hyphens available for semantic parsing;
3. Utterwise handles acronyms, numbers, dates, versions, percentages, currency, temperatures and similar written forms;
4. source punctuation is preserved because NeuTTS uses it for pauses and sentence prosody;
5. measured pronunciation policies remain deliberately small.

Utterwise is pinned to **0.1.1**. It adds no model or background service.

## Voice off / resource release

Voice output is a **resource mode**, not a volume mute. Turning Voice output off shuts down the NeuTTS child so the model-owned Arc/CPU/RAM resources are actually released.

When Voice output is turned off:

1. audible and queued TTS is invalidated immediately;
2. streamed response text stops entering the speech path and remains visual-only;
3. `SpeechOutputService` stops the active `VoiceOutput` off the Qt thread;
4. the NeuTTS child process is shut down;
5. the UI changes from `stopping...` to off only after shutdown completes.

Turning Voice output back on creates a fresh worker and performs asynchronous warm-up. Dexter/Goose/STT remain usable while Voice is off or warming. The preference is persisted with Qt `QSettings`; if Voice was deliberately left off, the next launch skips TTS warm-up.

## Pipelined Perth integration

`neutts2e_pipelined_stream.py` mirrors NeuTTS 1.4.1's private GGUF streaming method but changes scheduling only:

1. a bounded producer thread continues llama.cpp speech-token generation on Arc;
2. the consumer performs the same ONNX decode on CPU;
3. the same Perth watermark is applied through the encoder-only inference wrapper;
4. the same crop and overlap-add sequence is preserved;
5. chunks are yielded in the same order.

The queue is intentionally bounded so cancellation cannot accumulate a large amount of speculative work. Because this integration mirrors a private NeuTTS method, `neutts==1.4.1` must remain pinned until the pipeline is revalidated against any upgrade.

## Playback and interruption

The playback layer keeps a **0.5 s jitter reserve**. Production output uses `windows_default_audio_sink.py` with SoundCard/Core Audio/WASAPI instead of relying on PortAudio's cached default render endpoint. The sink follows the live Windows default output device when headphones/speakers change; Moonshine's microphone path is untouched.

PTT acts as a hard barge-in boundary and invalidates queued/audible TTS before the new voice capture begins.

## Dependency management

From the repository root:

```bat
uv sync
```

The root `.python-version` requests Python 3.13. `llama-cpp-python` is resolved from its prebuilt Vulkan wheel index, and the project forbids silently building it from source.

The validated project uses `llama-cpp-python 0.3.35` on Windows/Python 3.13. `soundcard==0.4.6` is a production dependency for live output-device following.

## Runtime validation

Authenticate/migrate model access if required:

```bat
uv run python Runtime\Voice\neutts2e_runtime.py migrate-auth
uv run python Runtime\Voice\neutts2e_runtime.py auth
```

Check the project interpreter, imports and gated model access:

```bat
uv run python Runtime\Voice\neutts2e_runtime.py doctor
```

Run the production streaming/cancellation smoke test:

```bat
uv run python Runtime\Voice\neutts2e_runtime.py smoke
```

For live timing diagnosis, set `DEXTER_TTS_TRACE=1` before launching the Host. Normal hidden-launch stderr is retained at:

```text
%LOCALAPPDATA%\Dexter\Diagnostics\host-stderr.log
```

## Useful memory/performance diagnostics

The active `Diagnostics/` directory contains the probes used to validate production changes. Particularly useful current ones include:

- `tts_production_worker_tree_memory.py` — measure the complete launcher + heavy-worker process tree rather than only `Popen.pid`;
- `tts_production_worker_identical_repeat_memory.py` — repeat exact deterministic requests and detect request-lifetime growth;
- `tts_request_growth_python_native_probe.py` — compare OS growth with `tracemalloc`, thread count and forced GC to separate Python from native retention;
- `tts_perth_autograd_growth_ab.py` — reproduce the Perth autograd leak and compare against inference-mode execution;
- codec / batch / compatibility A/B scripts — retain the evidence for accepted and rejected memory changes.

Do not infer a production optimization from one isolated run. The workflow used throughout this cleanup was: **isolate the allocation -> verify output parity -> repeat fresh-process measurements -> validate under sustained Gemma contention -> only then productionize**.

## Live Host behavior

When Voice output is enabled, `SpeechOutputService` starts Paul in parallel with Gateway and Moonshine. Responses remain text-only until TTS is ready. A TTS startup failure also leaves the Host running text-only.

Normal Goose `end_turn` responses flush the final speech fragment. PTT, typed prompts, agent cancellation, chat reset, non-normal response termination, Gateway/Goose failure, Voice-off, and Host shutdown interrupt audible and queued TTS.

Run Dexter from the root environment with:

```bat
uv run python -m Runtime.Host.app
```

## Files

- `tts.py` — engine-independent sentence buffering, formatting, playback queue and interruption.
- `text_normalization.py` — production speech-text normalizer/backend wrapper.
- `buffered_audio_sink.py` — legacy/diagnostic PortAudio jitter-buffer sink.
- `windows_default_audio_sink.py` — production Core Audio/WASAPI sink following the live default output.
- `output_service.py` — restartable Host-facing Voice lifecycle; async warm-up, unload, degraded mode and response gating.
- `neutts2e_backend.py` — `SpeechBackend` adapter and production Arc/pipeline defaults.
- `neutts2e_worker_launcher.py` — isolated production compatibility surfaces and llama.cpp constructor tuning.
- `neutts2e_worker.py` — persistent NeuTTS-2E / Paul child while Voice is enabled.
- `neutts2e_pipelined_stream.py` — Arc-generation / CPU-codec+Perth pipeline and hybrid streaming windows.
- `neucodec_onnx_compat.py` — minimal ONNX-only NeuCodec surface.
- `transformers_quantized_compat.py` — minimal GGUF-only Transformers import surface.
- `perth_encoder_only.py` — encoder-only Perth watermarking with `torch.inference_mode()` and SciPy-eager-import avoidance.
- `librosa.py` — minimal SciPy-free librosa compatibility surface.
- `neutts2e_runtime.py` — auth migration, doctor, streaming smoke and cancellation tests.
- `tts_trace.py` — opt-in latency/underflow tracing.
- `../Host/voice_output_control.py` — Voice resource toggle UI.

## Remaining TTS work

High priority:

1. investigate the remaining reproducible **~2.5 MB per identical spoken response** native-memory slope and prove whether it is bounded or indefinite;
2. fix text-queue overflow so queue pressure cannot cancel the rest of an assistant response;
3. re-run a sustained-Gemma benchmark on the exact final stack (quantized Transformers compatibility + Perth inference-mode) and record the final concurrent RTF/TTFA.

Lower priority:

- only revisit NPU/OpenVINO or other backend experiments if profiling demonstrates a concrete latency or memory benefit;
- do not move TTS onto the RTX 5050 while it is needed by the main LLM;
- do not reduce Gemma's context as an early TTS optimization.

The retired model-comparison harness is documented in `Development/Archive/TTS_SELECTION_2026-08-20.md`. Completed experiments remain recoverable through Git history and the active diagnostics rather than being treated as current production requirements.
