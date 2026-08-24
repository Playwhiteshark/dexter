# Dexter Host runtime ownership

The Host owns the runtime processes it launches:

1. `Runtime/Model Gateway/gateway.py`
2. `goose acp`
3. Moonshine speech input
4. the NeuTTS-2E speech-output worker once TTS wiring is enabled

Startup order for the model path is Gateway -> Goose ACP. TTS is a separate Host-owned child process so model inference never runs on the Qt thread and can be restarted/stopped independently.

Dexter does **not** discover/adopt a process already listening on port 11435.
Its own Gateway child must bind successfully and print its normal readiness
line. If the port is occupied, the child exits and Dexter surfaces a Gateway
startup error.

Normal Host shutdown stops Goose ACP and the Gateway process that the Host
created. Once TTS is live, shutdown also stops the TTS worker before the Host exits.

## Python/runtime environment

Dexter's Host, STT, and TTS dependencies share one uv-managed environment at the repository root. The project is pinned to Python 3.13 by `.python-version` and `pyproject.toml` because NeuTTS 1.4.1 requires Python `<3.14` while the current Host/STT stack supports 3.13.

TTS process isolation is therefore **not** environment isolation: the worker launches with the same `sys.executable` as the Host. Model/auth/cache data lives under `%LOCALAPPDATA%\Dexter\Voice\...`, outside the Python environment.

The root uv project explicitly resolves `llama-cpp-python` from its prebuilt CPU-wheel index and forbids source-building that package. A missing wheel must fail dependency resolution rather than silently requiring Visual Studio/CMake/NMake.

`Runtime/Tools/dexter-tools` remains a separate MCP service with its own dependency surface. That boundary predates TTS and is not a Python-version workaround.

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
