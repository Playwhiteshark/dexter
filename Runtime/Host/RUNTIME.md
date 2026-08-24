# Dexter Host runtime ownership

The Host owns the runtime processes it launches:

1. `Runtime/Model Gateway/gateway.py`
2. `goose acp`
3. Moonshine speech input
4. the NeuTTS-2E / Paul speech-output worker

Startup order for the model path is Gateway -> Goose ACP. Moonshine and TTS start independently in parallel. TTS warmup is intentionally non-blocking: Dexter can become usable before Paul is ready, and replies remain text-only until the TTS worker reports ready. A TTS startup failure also degrades to text-only operation rather than taking down the Host.

The TTS worker is a separate Host-owned child process so model/native inference never runs on the Qt thread and can be stopped independently. The child uses the same Python interpreter and dependency environment as the Host; this is process isolation, not environment isolation.

Dexter does **not** discover/adopt a process already listening on port 11435.
Its own Gateway child must bind successfully and print its normal readiness
line. If the port is occupied, the child exits and Dexter surfaces a Gateway
startup error.

Normal Host shutdown first stops audible/queued TTS and its worker, then stops Moonshine, Goose ACP, and the Gateway process that the Host created.

## Python/runtime environment

Dexter's Host, STT, and TTS dependencies share one uv-managed environment at the repository root. The project is pinned to Python 3.13 by `.python-version` and `pyproject.toml` because NeuTTS 1.4.1 requires Python `<3.14` while the current Host/STT stack supports 3.13.

Model/auth/cache data lives under `%LOCALAPPDATA%\Dexter\Voice\...`, outside the Python environment.

The root uv project explicitly resolves `llama-cpp-python` from its prebuilt CPU-wheel index and forbids source-building that package. A missing wheel must fail dependency resolution rather than silently requiring Visual Studio/CMake/NMake.

`Runtime/Tools/dexter-tools` remains a separate MCP service with its own dependency surface. That boundary predates TTS and is not a Python-version workaround.

## Speech output behavior

NeuTTS-2E uses the fixed `paul` speaker in neutral mode, with the Q8 GGUF backbone and int8 ONNX decoder on CPU. Goose response chunks are still rendered immediately in the UI; the same chunks are fed into `VoiceOutput`, which emits sentence-sized synthesis units and can synthesize ahead of playback.

PTT is a hard barge-in boundary. Starting a new voice capture immediately interrupts audible speech and invalidates queued TTS before listening begins. Typed prompts, agent cancellation, non-normal response termination, chat reset, Gateway/Goose failure, and Host shutdown also interrupt old speech. A normal `end_turn` flushes the final partial sentence so the complete answer can be spoken.

Typed prompts are available at the bottom of the overlay. Enter or the Send
button submits a single-line prompt through the same Goose ACP session as voice.

Each Dexter response can retain a bounded activity record from ACP. The
collapsed row summarizes tool calls, plans, and reasoning chunks; expanding it
shows the protocol-provided details. A separate badge says whether Dexter used
the `web_search` tool and whether that call completed or failed. Raw tool input,
output, and text are deliberately truncated before they enter the Qt history so
one large tool result cannot make the overlay unbounded.

Deleting or undoing only the most recent prompt/response pair is intentionally
deferred. ACP can cancel an active prompt and can delete an entire saved
session, but it does not expose stable mutation of a completed turn. Removing
only the visible cards would leave Goose's actual context unchanged and would
therefore be a misleading undo.

The Host also owns Dexter's per-chat tool approval policy. Classified Dexter
tools are auto-approved, unknown future tools default to Ask, and pre-existing
file writes use path-specific, SHA-256-verified trust that is cleared with the
chat. Approval decisions are never persisted outside the ACP chat.
