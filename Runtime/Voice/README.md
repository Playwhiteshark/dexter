# Dexter Voice Runtime

Dexter's selected production TTS is **NeuTTS-2E**, speaker **paul**, neutral emotion, CPU-only.

## Runtime architecture

Dexter is a single uv-managed Python application pinned to **Python 3.13** at the repository root. Host, STT, and TTS dependencies are resolved from the root `pyproject.toml` into one root `.venv`.

TTS runs in a child process, but that child uses the **same `sys.executable` and environment as the Host**. The process boundary exists so the Host owns TTS lifecycle/cancellation and so native inference failures do not run on the Qt thread; it is not a second Python installation or TTS-specific virtual environment.

Machine-local model/auth/cache data stays outside the repo at:

```text
%LOCALAPPDATA%\Dexter\Voice\neutts2e\cache
```

Production uses:

- backbone: `neuphonic/neutts-2e-q8-gguf`
- codec: `neuphonic/neucodec-onnx-decoder-int8`
- speaker: `paul`
- emotion: `neutral`
- sample rate: 24 kHz

The GGUF backbone is intentional. NeuTTS implements `infer_stream()` for its quantized llama.cpp path, whereas the PyTorch path only returns complete utterances. The measured PyTorch smoke test produced about 9 seconds to first audio for 3.7 seconds of speech (RTF ~2.43), so it is not suitable for Dexter's interactive output.

The target-machine Q8 smoke test passed with approximately **1.11 s first audio**, **8 streamed chunks**, **3.94 s of audio in 5.82 s** (RTF ~1.48), and successful mid-stream cancellation. Q8 is therefore the first live candidate. Because RTF is still above 1.0, listen for starvation/gaps on longer responses before declaring performance final; Q4 or a small measured prebuffer remain possible follow-ups.

## Dependency management

From the repository root:

```bat
uv sync
```

The root `.python-version` requests Python 3.13. uv may install that project Python automatically without changing the system/default Python used by unrelated software.

`llama-cpp-python` is resolved only from its official prebuilt CPU-wheel index. The root uv configuration explicitly forbids building `llama-cpp-python` from source, so a missing wheel fails immediately instead of silently requiring CMake, NMake, or Visual Studio Build Tools.

The previously pinned `llama-cpp-python==0.3.20` was removed: that exact pin did not resolve to the needed Windows CPU wheel and caused pip to fall back to a source build. The validated project currently resolves `llama-cpp-python 0.3.35` on Windows/Python 3.13.

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

## Live Host behavior

`SpeechOutputService` starts Paul in parallel with Gateway and Moonshine. The ~44-second model warmup does not block Dexter from becoming usable; responses remain text-only until TTS is ready. A TTS startup failure also leaves the Host running text-only.

Normal Goose `end_turn` responses flush the final speech fragment. PTT, typed prompts, agent cancellation, chat reset, non-normal response termination, Gateway/Goose failure, and Host shutdown interrupt audible and queued TTS. PTT therefore acts as a hard barge-in boundary before a new voice capture begins.

Run Dexter from the root environment with:

```bat
uv run python -m Runtime.Host.app
```

## Files

- `tts.py` — engine-independent sentence buffering, formatting, playback queue, and interruption.
- `output_service.py` — Host-facing async startup/degraded-mode/response lifecycle.
- `neutts2e_backend.py` — `SpeechBackend` adapter; launches the worker with the Host interpreter.
- `neutts2e_worker.py` — persistent NeuTTS-2E / Paul child process.
- `neutts2e_runtime.py` — auth migration, runtime doctor, streaming smoke test, and cancellation test.

The retired model-comparison harness is documented in `Development/Archive/TTS_SELECTION_2026-08-20.md` and remains available through Git history.
