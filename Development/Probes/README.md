# Development probes

Small developer-only probes retained because they are useful when debugging Dexter's
model/prompt/tool boundary.

- `capture_goose_ollama.py` â€” capture the exact Goose â†’ Ollama request.
- `dexter_personality_probe.py` â€” compare personality/style behavior.
- `dexter_prompt_endpoint_matrix.py` â€” compare prompt/endpoint combinations.
- `dexter_tools_style_probe.py` â€” inspect tool-catalog/style interactions.

Generated result JSON and request captures are deliberately ignored by Git. These
scripts are not part of Dexter's normal runtime.
