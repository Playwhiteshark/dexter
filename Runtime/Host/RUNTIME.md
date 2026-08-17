# Dexter Host runtime ownership

The Host owns the runtime processes it launches:

1. `Runtime/Model Gateway/gateway.py`
2. `goose acp`
3. Moonshine speech input

Startup order for the model path is Gateway -> Goose ACP.

Dexter does **not** discover/adopt a process already listening on port 11435.
Its own Gateway child must bind successfully and print its normal readiness
line. If the port is occupied, the child exits and Dexter surfaces a Gateway
startup error.

Normal Host shutdown stops Goose ACP and the Gateway process that the Host
created.

Typed prompts are available at the bottom of the overlay. Enter or the Send
button submits a single-line prompt through the same Goose ACP session as voice.

The Host also owns Dexter's per-chat tool approval policy. Classified Dexter
tools are auto-approved, unknown future tools default to Ask, and pre-existing
file writes use path-specific, SHA-256-verified trust that is cleared with the
chat. Approval decisions are never persisted outside the ACP chat.
