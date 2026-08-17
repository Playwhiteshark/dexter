# Development diagnostics

Small developer-only probes retained because they are useful when debugging Dexter's
model/prompt/tool boundary.

- `capture_goose_ollama.py` — capture the exact Goose → Ollama request.
- `dexter_personality_probe.py` — compare personality/style behavior.
- `dexter_prompt_endpoint_matrix.py` — compare prompt/endpoint combinations.
- `dexter_tools_style_probe.py` — inspect tool-catalog/style interactions.

Generated result JSON and request captures are deliberately ignored by Git. These
scripts are not part of Dexter's normal runtime.
