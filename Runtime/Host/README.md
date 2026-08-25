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

## Launching Dexter

The canonical development/manual launch is the module entry point from the repository root:

```bat
uv run python -m Runtime.Host.app
```

Using `-m Runtime.Host.app` keeps the launch form aligned with Dexter's package/module imports and the root uv environment. Do not launch an older Host copy or a separate Python environment.

For a normal Windows desktop shortcut, do **not** spawn `uv` on every launch. After `uv sync`, the shortcut can call the already-synced root environment directly:

```text
Target:   <repo>\.venv\Scripts\pythonw.exe -m Runtime.Host.app
Start in: <repo>
```

`pythonw.exe` avoids opening an unnecessary console window while still using the same pinned Python 3.13 environment as `uv run`.

Create or update the current user's Desktop `Dexter.lnk` automatically with:

```powershell
powershell -ExecutionPolicy Bypass -File .\Runtime\Host\configure_shortcut.ps1
```

To update a shortcut stored somewhere else, pass its path explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File .\Runtime\Host\configure_shortcut.ps1 -ShortcutPath "C:\path\to\Dexter.lnk"
```

The helper sets the target, `-m Runtime.Host.app` arguments, repository working directory, description, and Dexter icon when available. It fails clearly if the root `.venv` has not been created yet.

Do not install Host dependencies from a separate `Runtime/Host/requirements.txt`; the root `pyproject.toml` is the dependency source of truth and `uv.lock` is the reproducible lockfile once generated.

## Assets

Place the Windows application/tray icon at:

```text
Runtime/Host/assets/dexter.ico
```

The Host uses it for the tray and window icon. If it is absent or invalid, Dexter falls back to a standard Qt system icon. A multi-resolution ICO is ideal, especially 16x16, 32x32, 48x48, and 256x256.
