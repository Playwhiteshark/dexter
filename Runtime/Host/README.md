# Dexter Host

Dexter is managed as a uv project from the repository root. The Host, Moonshine STT, and NeuTTS-2E TTS share the root Python 3.13 environment.

From the repository root:

```bat
uv sync
```

Then validate the core Host/TTS code before launching the app:

```bat
uv run python -m unittest Runtime.Host.tests.test_tts -v
uv run python Runtime\Voice\neutts2e_runtime.py doctor
uv run python Runtime\Voice\neutts2e_runtime.py smoke
```

Launch the Host with:

```bat
uv run python Runtime\Host\app.py
```

Do not install Host dependencies from a separate `Runtime/Host/requirements.txt`; the root `pyproject.toml` is the dependency source of truth and `uv.lock` is the reproducible lockfile once generated.

## Assets

Place the Windows application/tray icon at:

```text
Runtime/Host/assets/dexter.ico
```

The Host uses it for the tray and window icon. If it is absent or invalid, Dexter falls back to a standard Qt system icon. A multi-resolution ICO is ideal, especially 16x16, 32x32, 48x48, and 256x256.
