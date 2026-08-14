# Dexter Tools v0.2 development candidate

Deterministic local-computer tools for Dexter. The direct tool catalog remains in place; the Model Gateway is unchanged. The model chooses tools while deterministic modules enforce permissions, exact execution, verification, and postconditions.

## Main changes

- `agent_results.py` keeps proof-rich internal verification while returning concise successful results to the model.
- `read_image(path)` is the single model-facing image loader. Pillow identifies the content, applies EXIF orientation, uses the first animation frame, fully decodes it, then emits static PNG bytes.
- Corrupt/unsafe images fail before MCP image content is created, preventing the provider-poisoned-session failure mode.
- Normal model-facing limits are smaller: directory 50, file search 25, content search 20 / 3 per file, reads 160 lines / 12k chars, clipboard 10k, processes/windows 30.
- Machine-specific roots and app aliases live in ignored `local_config.py`; copy `local_config.example.py` once on each machine.
- `schema_audit.py` measures the live MCP catalog instead of maintaining a duplicate schema catalog.
- `test_tools.py` includes dependency/import smoke tests, filesystem/result contracts, image regressions, background-job tests, MCP registration checks and multi-tool chains.
- `benchmark_server.py` + `benchmark_goose.py` provide an automated real-Goose/Gemma selection benchmark while simulating effectful tools.

No tool router, lazy schema facade, Gateway media layer, image service, custom approval layer, or runtime preflight daemon is introduced.

## Install/update dependencies

From the `dexter-tools` directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

## Local configuration

```powershell
Copy-Item .\local_config.example.py .\local_config.py
```

`local_config.py` is ignored by Git. Keep filesystem roots narrow and add only explicitly trusted app aliases.

## Deterministic validation

```powershell
.\.venv\Scripts\python.exe .\test_tools.py --suite all
.\.venv\Scripts\python.exe .\schema_audit.py
```

Automatic tests do not perform real mouse clicks or keyboard typing; desktop side effects are mocked at the tester layer.

## Real Goose/Gemma benchmark

Exit old Goose sessions first so they cannot retain old MCP schemas/processes. Then, with the normal Gateway/Ollama path running:

```powershell
.\.venv\Scripts\python.exe .\benchmark_goose.py
.\.venv\Scripts\python.exe .\benchmark_goose.py --case webp_vision
.\.venv\Scripts\python.exe .\benchmark_goose.py --case bad_image_recovery
.\.venv\Scripts\python.exe .\benchmark_goose.py --context
```

The benchmark uses an isolated Goose data root, `GOOSE_MODE=auto`, and a benchmark MCP server that reuses production schemas while simulating effectful tools. Defaults mirror the current Dexter stack and can be overridden with `DEXTER_BENCHMARK_PROVIDER`, `DEXTER_BENCHMARK_MODEL`, and `DEXTER_BENCHMARK_OLLAMA_HOST`. Production Goose remains `smart_approve`.

## Safety defaults

- Filesystem access is restricted to configured `ALLOWED_ROOTS`.
- Path resolution and recursive walking avoid traversing links/junctions outside allowed roots.
- Copy/move/rename refuse overwrite and retain deterministic postcondition verification.
- `delete_file` uses the Recycle Bin; permanent deletion is not exposed to the model.
- `open_app` accepts only configured aliases and never invokes a shell.
- App arguments are denied by default.
- `open_file` blocks executable/script/shortcut extensions so it cannot bypass the app allowlist.
- `open_url` accepts only HTTP/HTTPS.
- Window closing is graceful; `close_app` does not hard-kill processes.
- PyAutoGUI's corner fail-safe remains enabled.
- Repeating click jobs are cancellable and can have run/time/visual stop conditions.
