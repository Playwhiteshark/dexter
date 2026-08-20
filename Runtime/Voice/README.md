# Dexter Voice Runtime

Dexter's selected production TTS is **NeuTTS-2E**, speaker **paul**, neutral emotion, CPU-only.

## Why it is isolated

NeuTTS requires Python 3.10-3.13 and brings a large PyTorch/Transformers stack. Dexter Host stays lean and does not import those packages. `neutts2e_backend.py` launches a persistent Python 3.13 worker and receives streamed float32 PCM chunks over a small local stdio protocol.

Production currently uses:

- backbone: `neuphonic/neutts-2e-q8-gguf`
- codec: `neuphonic/neucodec-onnx-decoder-int8`
- speaker: `paul`
- emotion: `neutral`
- sample rate: 24 kHz

The GGUF path is intentional: NeuTTS implements chunked streaming for GGUF backbones. Q8 is the initial quality-first choice; Q4 is the fallback if measured CPU latency justifies the quality tradeoff.

## One-time setup

From the repository root:

```bat
python3.13 Runtime\Voice\setup_neutts2e.py setup
```

The runtime is installed outside the repo under:

```text
%LOCALAPPDATA%\Dexter\Voice\neutts2e
```

If the earlier TTS benchmark Hugging Face login still exists, setup migrates that login into the production runtime automatically.

NeuTTS model repositories are gated on Hugging Face. If access has been accepted but the production runtime has no login, run:

```bat
python3.13 Runtime\Voice\setup_neutts2e.py auth
```

Check installation state:

```bat
python3.13 Runtime\Voice\setup_neutts2e.py status
```

Before wiring/changing live Host behavior, measure the production streaming path:

```bat
python3.13 Runtime\Voice\setup_neutts2e.py smoke
```

The smoke test reports warm startup time, time to first audio, generated audio duration, chunk count, and real-time factor for Paul.

## Files

- `tts.py` — engine-independent sentence buffering, formatting, playback queue, and interruption.
- `neutts2e_backend.py` — Host-side `SpeechBackend` adapter.
- `neutts2e_worker.py` — persistent Python 3.13 NeuTTS process.
- `setup_neutts2e.py` — isolated runtime setup/auth/status/smoke/cleanup.
- `neutts2e-constraints.txt` — pinned production ML dependency versions.

The retired model-comparison harness is documented in `Development/Archive/TTS_SELECTION_2026-08-20.md` and remains available through Git history.
