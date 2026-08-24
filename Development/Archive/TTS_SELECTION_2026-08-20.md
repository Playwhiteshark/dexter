# Dexter TTS selection — 2026-08-20

The exploratory conversational-TTS benchmark is concluded.

## Decision

- Selected engine: **NeuTTS-2E**
- Selected speaker: **paul**
- Default emotion: **neutral**
- Target device: **CPU only**
- No reference/cloned voice is required.

The user preferred Paul after listening to the generated Chatterbox Nano and NeuTTS-2E samples.

## Production findings

The listening benchmark originally used the PyTorch NeuTTS-2E backbone. A production smoke test later measured approximately **9.0 seconds to first audio** for **3.7 seconds of generated speech**, an RTF of roughly **2.43**. That is too slow for interactive Dexter speech, even with sentence-level pipelining.

NeuTTS's implementation only exposes `infer_stream()` for its quantized llama.cpp path. Production therefore uses the official **Q8 GGUF NeuTTS-2E backbone** with the **int8 ONNX NeuCodec decoder**, keeping Paul/neutral while enabling chunked CPU output.

The validated Windows/Python 3.13 Q8 smoke test produced approximately **1.11 seconds to first audio**, **8 streamed chunks**, and **3.94 seconds of audio in 5.82 seconds** (RTF ~**1.48**). Mid-stream cancellation also passed. Q8 is therefore the first live candidate. Because RTF remains above 1.0, longer real replies should be checked for audible starvation/gaps before performance is considered final; Q4 or a small measured prebuffer are follow-up options if needed.

An earlier attempt pinned `llama-cpp-python==0.3.20`. That exact pin failed to resolve a suitable prebuilt Windows CPU wheel and pip fell back to a local CMake/NMake build. The production project now resolves llama.cpp from its official CPU-wheel index and explicitly forbids source-building `llama-cpp-python`, so a missing wheel fails immediately rather than introducing compiler tooling. The validated environment resolves `llama-cpp-python 0.3.35` from a binary wheel.

## Runtime architecture

NeuTTS 1.4.1 officially requires Python `>=3.10,<3.14`. The rest of Dexter's current Host/STT dependencies support Python 3.13, so Dexter is standardized on **Python 3.13 as a project**, rather than running a Python 3.14 Host plus a separate Python 3.13 TTS environment.

The repository root `pyproject.toml` and `.python-version` define one uv-managed Host/STT/TTS environment. TTS remains a Host-owned child process for lifecycle, cancellation, and native-runtime isolation, but it launches with the same `sys.executable` as the Host. Model/auth/cache data remains machine-local outside the repository.

The separate `dexter-tools` environment remains an independent MCP service boundary with its own dependency surface; it is not a workaround for TTS or Python-version incompatibility.

## Retired benchmark

`Development/TTSBench/` was a disposable selection harness and has been removed from the active source tree after the decision. Its source remains available in Git history. Generated WAV results and temporary engine environments were intentionally never committed.

Do not reintroduce the broad model-comparison harness into production unless the TTS selection itself is reopened.
