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

The benchmark used the PyTorch NeuTTS-2E backbone for listening comparisons. Production uses the official **Q8 GGUF** NeuTTS-2E backbone with the **int8 ONNX NeuCodec decoder** because NeuTTS implements chunked streaming only for its GGUF path. Q8 is the initial choice to stay close to the quality of the selected voice; Q4 remains a simple fallback if CPU latency warrants it.

The heavy TTS dependency stack is intentionally isolated from Dexter Host in a persistent Python 3.13 worker under the machine-local Dexter voice runtime. The host-side `SpeechBackend` adapter exchanges streamed float32 PCM with that worker while the existing `VoiceOutput` core owns sentence buffering, playback, and interruption.

## Retired benchmark

`Development/TTSBench/` was a disposable selection harness and has been removed from the active source tree after the decision. Its source remains available in Git history. Generated WAV results and temporary engine environments were intentionally never committed.

Do not reintroduce the broad model-comparison harness into production unless the TTS selection itself is reopened.
