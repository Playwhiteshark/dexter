# Dexter Voice Runtime

Dexter's selected production TTS is **NeuTTS-2E**, speaker **paul**, neutral emotion.

> **Current production status (2026-08-25):** the throughput/choppiness problem is considered solved on the target machine. Production now uses the Q4 backbone, 8 generation threads, Windows AboveNormal process creation, NeuCodec ORT 2-intra/1-inter threading, 40-frame first / 50-frame steady streaming, and Core Audio/WASAPI default-device following via SoundCard. Repeated fully-concurrent test runs measured about **0.79-0.81 RTF with zero PCM underflows**. Older benchmark sections below are retained as the progression that led to these settings.

## Runtime architecture

Dexter is a single uv-managed Python application pinned to **Python 3.13** at the repository root. Host, STT, and TTS dependencies are resolved from the root `pyproject.toml` into one root `.venv`.

TTS runs in a child process, but that child uses the **same `sys.executable` and environment as the Host**. The process boundary exists so the Host owns TTS lifecycle/cancellation and so native inference failures do not run on the Qt thread; it is not a second Python installation or TTS-specific virtual environment.

Machine-local model/auth/cache data stays outside the repo at:

```text
%LOCALAPPDATA%\Dexter\Voice\neutts2e\cache
```

Production uses:

- backbone: `neuphonic/neutts-2e-q4-gguf`
- backbone device: Intel Arc 140T via llama.cpp Vulkan, visible device `0`
- codec: `neuphonic/neucodec-onnx-decoder-int8` on CPU
- Perth watermarking: enabled
- scheduling: Arc token generation is pipelined with CPU codec/watermark work
- generation threads: 8
- NeuCodec ONNX Runtime pool: 2 intra-op / 1 inter-op threads
- streaming windows: 40 frames for first audio, then 50 frames for steady-state headroom
- worker priority: Windows `ABOVE_NORMAL_PRIORITY_CLASS` at process creation
- output routing: SoundCard/Core Audio/WASAPI follows the live Windows default render endpoint
- text normalization: `utterwise==0.1.1` plus a very small Dexter pronunciation/markup policy
- speaker: `paul`
- emotion: `neutral`
- sample rate: 24 kHz

The GGUF backbone is intentional. NeuTTS implements `infer_stream()` for its quantized llama.cpp path, whereas the PyTorch path only returns complete utterances.

## Measured production path

The original CPU Q8 path synthesized slower than playback (roughly RTF 1.5-1.6), which guaranteed live starvation on longer speech. Profiling showed NeuCodec was only about 8% of synthesis time; the GGUF/token path dominated.

The Intel Arc 140T Vulkan path was then verified from native llama.cpp logs:

- `Vulkan0 = Intel(R) Arc(TM) 140T GPU`
- cooperative matrices available as `KHR_coopmat`
- 29/29 model layers offloaded
- KV cache on Vulkan0
- RTX 5050 not used by TTS

Serial Arc + Perth improved the clean benchmark to about **RTF 1.22**, but remained too slow for uninterrupted live playback. Profiling then showed that NeuTTS serialized Arc token generation with CPU Perth watermarking. Pipelining those independent stages while keeping Perth enabled produced a validated **RTF 0.782** path and removed the original sustained-starvation problem.

A later steady-window benchmark showed that keeping the upstream 25-frame / 0.5 s window forever could still occasionally miss a playback deadline under load:

- 25 frames: **RTF 0.805**, worst measured chunk deficit **+0.113 s**
- 40 frames: **RTF 0.773**, worst measured chunk deficit **-0.008 s**
- 50 frames: **RTF 0.733**, worst measured chunk deficit **-0.067 s**

During real simultaneous Gemma generation, an additional problem emerged: NeuCodec's default ONNX Runtime CPU pool competed aggressively with the Arc/Vulkan generation path. Constraining ORT to **2 intra-op / 1 inter-op threads** was the largest live win, dropping the first concurrent request from about **1.26 RTF to ~0.8 RTF**. With synthesis now comfortably faster than realtime, changing the first streaming window from 25 to **40 frames** removed the remaining measured startup underflows while keeping first audio around one second. Production therefore uses **40 first / 50 steady**.

The playback layer keeps a **0.5 s jitter reserve**. TTS output now uses `windows_default_audio_sink.py` with SoundCard/Core Audio/WASAPI instead of relying on PortAudio's cached default output. The sink checks the live Windows default render endpoint and reopens only TTS playback when headphones/speakers are plugged or unplugged; Moonshine's always-open PortAudio microphone is untouched. PTT interruption still invalidates queued/audible TTS immediately.

## Speech text normalization

NeuTTS-2E accepts raw written text rather than phonemes/SSML. Compact raw-text TTS models are unreliable around programmer typography, acronyms and semantic written forms, so production normalizes text immediately before synthesis.

`Runtime/Voice/text_normalization.py` is intentionally thin:

1. existing `SpeechFormatter` removes fenced code, Markdown presentation syntax and raw URLs that should not be read literally;
2. Dexter converts single-underscore emphasis, `snake_case`, and word-style hyphens to ordinary word boundaries while leaving negatives, dates and numeric hyphens available for semantic parsing;
3. Utterwise handles acronyms, numbers, dates, versions, percentages, currency, temperatures and similar written forms;
4. Dexter preserves the source punctuation instead of using Utterwise's punctuation-free plain renderer, because NeuTTS uses punctuation for pauses and sentence prosody;
5. measured pronunciation policies are deliberately tiny: `AI` becomes `A I`, `JSON` becomes `jay son`, while ordinary acronyms such as `SQL`, `CPU` and `GPU` remain Utterwise's responsibility.

Utterwise is pinned to **0.1.1**. Its core is pure Python and declares no runtime dependency stack. There is no extra model, phonemizer, service or background process for normalization.

## Voice off / resource release

Voice output is a **resource mode**, not a volume mute. The header overflow menu contains a checkable `Voice output` command rather than an easy-to-hit speaker button because disabling it has a meaningful reverse cost: Paul must warm again before speech is available.

When Voice output is turned off:

1. audible and queued TTS is invalidated immediately;
2. streamed response text stops entering the speech path and remains visual-only;
3. `SpeechOutputService` stops the active `VoiceOutput` off the Qt thread;
4. the NeuTTS child process is shut down, releasing the model-owned Arc/CPU/RAM resources;
5. the UI changes from `stopping…` to off only after shutdown completes.

Turning Voice output back on creates a **fresh `VoiceOutput` and NeuTTS worker** and performs the normal asynchronous warm-up. Dexter/Goose/STT remain usable while Voice is off or warming. Voice can be turned back off during warm-up, which cancels that startup rather than forcing the user to wait for it to finish.

The preference is persisted with Qt `QSettings`. If Voice was deliberately left off, the next Dexter launch skips the TTS warm-up entirely. A transient TTS startup failure does not overwrite the preference; the menu exposes a retry state instead.

## Pipelined Perth integration

`neutts2e_pipelined_stream.py` mirrors NeuTTS 1.4.1's private GGUF streaming method but changes scheduling only:

1. a bounded producer thread continues llama.cpp speech-token generation on Arc;
2. the consumer performs the same ONNX decode on CPU;
3. the same Perth watermark is applied;
4. the same crop and overlap-add sequence is preserved;
5. chunks are yielded in the same order.

The queue is intentionally bounded so cancellation cannot accumulate a large amount of speculative work. Because this integration mirrors a private NeuTTS method, `neutts==1.4.1` must remain pinned until the pipeline is revalidated against any upgrade.

Dexter provides a tiny `Runtime/Voice/librosa.py` compatibility surface so the production NeuTTS worker does not eagerly import the real librosa/SciPy stack. Perth's required sample-rate conversion is implemented through `torchaudio.functional.resample`, which is already present in the NeuTTS dependency stack. This avoids Windows Application Control blocking an unsigned SciPy `_rgi_cython*.pyd` while preserving Perth watermarking.

## Dependency management

From the repository root:

```bat
uv sync
```

The root `.python-version` requests Python 3.13. uv may install that project Python automatically without changing the system/default Python used by unrelated software.

`llama-cpp-python` is resolved from its prebuilt **Vulkan** wheel index. The root uv configuration forbids building it from source, so a missing wheel fails immediately instead of silently requiring CMake, NMake, or Visual Studio Build Tools.

The validated project currently resolves `llama-cpp-python 0.3.35` on Windows/Python 3.13. `soundcard==0.4.6` is a production dependency for live Windows output-device following.

## Runtime validation

If the earlier benchmark login is the only Hugging Face login available, migrate it once:

```bat
uv run python Runtime\Voice\neutts2e_runtime.py migrate-auth
```

Or authenticate directly inside Dexter's machine-local voice cache:

```bat
uv run python Runtime\Voice\neutts2e_runtime.py auth
```

Check the project interpreter, native imports, and gated model access:

```bat
uv run python Runtime\Voice\neutts2e_runtime.py doctor
```

Run the production Paul streaming/cancellation test:

```bat
uv run python Runtime\Voice\neutts2e_runtime.py smoke
```

For a live timing diagnosis, set `DEXTER_TTS_TRACE=1` before launching the Host. The `synthesis_request` trace reports the **post-normalization text actually sent to NeuTTS**. The WASAPI sink also forwards buffer, underflow and `audio_device_change` events into the same trace when enabled.

Normal hidden shortcut launches preserve Host/worker stderr at:

```text
%LOCALAPPDATA%\Dexter\Diagnostics\host-stderr.log
```

## Live Host behavior

When Voice output is enabled, `SpeechOutputService` starts Paul in parallel with Gateway and Moonshine. Responses remain text-only until TTS is ready. A TTS startup failure also leaves the Host running text-only.

Normal Goose `end_turn` responses flush the final speech fragment. PTT, typed prompts, agent cancellation, chat reset, non-normal response termination, Gateway/Goose failure, Voice-off, and Host shutdown interrupt audible and queued TTS. PTT therefore acts as a hard barge-in boundary before a new voice capture begins.

Run Dexter from the root environment with:

```bat
uv run python -m Runtime.Host.app
```

## Files

- `tts.py` — engine-independent sentence buffering, formatting, playback queue, and interruption.
- `text_normalization.py` — tiny production speech-text normalizer/backend wrapper.
- `buffered_audio_sink.py` — legacy/diagnostic PortAudio jitter-buffer sink.
- `windows_default_audio_sink.py` — production Core Audio/WASAPI sink that follows the live Windows default output.
- `output_service.py` — restartable Host-facing Voice lifecycle; async warm-up, disable/unload, degraded mode and response gating.
- `neutts2e_backend.py` — `SpeechBackend` adapter and production Arc/pipeline runtime defaults.
- `neutts2e_worker.py` — persistent NeuTTS-2E / Paul child process while Voice is enabled.
- `neutts2e_pipelined_stream.py` — validated Arc-generation / CPU-Perth pipeline and hybrid streaming windows.
- `neutts2e_runtime.py` — auth migration, runtime doctor, streaming smoke test, and cancellation test.
- `tts_trace.py` — opt-in live latency/underflow tracing.
- `../Host/voice_output_control.py` — secondary overflow UI for the expensive Voice resource toggle.

## Remaining TTS work

The live throughput/choppiness issue is solved for the current machine. Remaining high-value work is:

1. attribute and reduce the long-lived TTS worker's oversized RAM footprint;
2. fix text-queue overflow so queue pressure cannot cancel the rest of an assistant response;
3. only revisit NPU/OpenVINO/other backend experiments if profiling shows a concrete latency or memory benefit.

The retired model-comparison harness is documented in `Development/Archive/TTS_SELECTION_2026-08-20.md`; completed experimental scripts remain recoverable through Git history rather than living in the active tree.
