# Dexter

Dexter is the local Windows AI assistant project. This repository is the canonical source for its Host, model gateway, voice runtime, integrations, tools, diagnostics, harnesses, and model configuration.

The current normal Host is a root `uv` project pinned to Python 3.13. The canonical development launch from the repository root is:

```bat
uv sync
uv run python -m Runtime.Host.app
```

Normal desktop use is console-free: `Runtime/Host/configure_shortcut.ps1` creates/repairs the Dexter shortcut, which runs `Runtime/Host/launch_hidden.vbs` and then the root `.venv`'s `pythonw.exe` module entry point. See [`Runtime/Host/README.md`](Runtime/Host/README.md) and [`Runtime/Host/RUNTIME.md`](Runtime/Host/RUNTIME.md) for the current runtime contract, and [`Runtime/Voice/README.md`](Runtime/Voice/README.md) for the production voice stack.

## Repository ownership

This repository owns Dexter source, project documentation, issues, and product-specific lifecycle. Local Windows Tools may index a checkout, inspect Git state, measure its size, and consume the optional `.local-windows-tools.json` storage declaration, but it does not own Dexter source or user data.

Machine-specific absolute checkout paths belong in Local Windows Tools' machine-local Projects Index registry, not in this repository.

## Storage and cleanup ownership

Dexter intentionally distinguishes reproducible runtime material from persistent user/machine state. `.local-windows-tools.json` exposes the currently known ownership in machine-readable form; this section explains the same contract for maintainers.

### Logs / diagnostics

The experimental advisor/model-gateway path can write JSONL diagnostics in:

```text
Runtime/Model Gateway/ollama_advisor_trace.jsonl
Runtime/Model Gateway/ollama_advisor_debug.jsonl
Runtime/Model Gateway/ollama_proxy_trace.jsonl
```

Voice debugging can also write under:

```text
Runtime/Voice/debug/
Runtime/Voice/ptt_debug/
```

These locations contain diagnostic output rather than source or durable user state and are declared disposable logs.

The production `Runtime/Model Gateway/gateway.py` itself normally writes status to stdout/stderr rather than maintaining its own persistent log file.

### Cache / reproducible environment

The root `.venv/` is reproducible dependency state and may be rebuilt from `pyproject.toml` / `uv.lock`.

Voice model/auth/cache material is kept outside the repository under:

```text
%LOCALAPPDATA%\Dexter\Voice\neutts2e\cache
```

The existing runtime documentation describes that tree as model/auth/cache data. It is therefore **documented but not generic automatic cleanup**: deleting it may force model downloads or destroy local authentication/cache state, so a cache label alone is not permission to remove it.

### Persistent user data

Dexter Tools' operational/computer memory defaults to:

```text
%LOCALAPPDATA%\Dexter\state\computer_memory.json
```

This is persistent user-specific application data. It survives source updates and must never be treated as a disposable log/cache. `Runtime/Tools/dexter-tools/local_config.py` may override its state root; that machine policy file is itself local configuration and is gitignored.

### Configuration

Machine-specific Dexter Tools policy belongs in:

```text
Runtime/Tools/dexter-tools/local_config.py
```

The committed `local_config.example.py` documents the portable schema. Never commit real machine-specific trusted roots, private aliases, secrets, or other local policy merely to make Projects Index portable.

### Generated development artifacts

Dexter currently has several historical/development producers whose generated outputs are intentionally gitignored but are not yet consolidated under one bounded artifact root. Examples include benchmark-run folders, generated Diagnostics reports, root probe-result JSON files, and generated voice WAV/debug material.

For that reason `.local-windows-tools.json` deliberately reports **`artifacts: unknown`** instead of pretending the project has no generated artifacts. Projects Index should continue to show Dexter's storage declaration as incomplete until those outputs are normalized or fully enumerated without unsafe wildcards.

This is a documentation warning, not evidence that project files are unsafe or broken.

## Storage-documentation rule

When code starts writing a new persistent or generated location, classify it as one of:

- logs;
- cache;
- runtime/machine state;
- configuration;
- retained/generated artifacts;
- user-owned data.

Update the owning runtime documentation and `.local-windows-tools.json` together. Unknown/incomplete ownership must remain explicit rather than being changed to `none` merely to clear a Projects Index warning. Only genuinely disposable logs/cache may ever be marked safe for generic cleanup.

## Repository map

```text
Runtime/Host/                desktop Host and lifecycle
Runtime/Voice/               STT/TTS/audio runtime
Runtime/Model Gateway/       Ollama/model gateway and experiments
Runtime/Tools/dexter-tools/  local computer/workspace tools
Runtime/Integrations/        browser/editor integrations
Models/                      model configuration
Harnesses/                   evaluation/benchmark harnesses
Diagnostics/                 diagnostic scripts plus generated local reports
Development/                 probes, handoffs and historical development material
```

Before changing a subsystem, read the README/runtime notes in that subsystem and preserve its documented security/runtime boundaries.
