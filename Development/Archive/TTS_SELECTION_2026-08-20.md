# Dexter TTS selection — 2026-08-20

The exploratory conversational-TTS benchmark is concluded.

## Decision

- Selected engine: **NeuTTS-2E**
- Selected speaker: **paul**
- Default emotion: **neutral**
- Target device: **CPU only**
- No reference/cloned voice is required.

The user preferred Paul after listening to the generated Chatterbox Nano and NeuTTS-2E samples.

## Production shape

Production intentionally uses the same **PyTorch NeuTTS-2E backbone** used in the listening test, paired with the **int8 ONNX NeuCodec decoder**. This keeps the selected voice path compiler-free on Windows and avoids introducing `llama-cpp-python` purely for within-utterance streaming.

Dexter's existing `VoiceOutput` already buffers streamed LLM text into sentence-sized units, synthesizes sentence N+1 while sentence N is playing, and owns playback interruption. That sentence-level pipeline is the initial production strategy. NeuTTS's GGUF-only chunk streaming can be revisited later only if measured sentence latency justifies the added native runtime complexity.

The heavy TTS dependency stack is intentionally isolated from Dexter Host in a persistent Python 3.13 worker under the machine-local Dexter voice runtime. The host-side `SpeechBackend` adapter exchanges float32 PCM with that worker while the existing `VoiceOutput` core owns sentence buffering, playback, and interruption.

## Retired benchmark

`Development/TTSBench/` was a disposable selection harness and has been removed from the active source tree after the decision. Its source remains available in Git history. Generated WAV results and temporary engine environments were intentionally never committed.

Do not reintroduce the broad model-comparison harness into production unless the TTS selection itself is reopened.
