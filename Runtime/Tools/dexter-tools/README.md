# Dexter Tools v0.1

Deterministic local tools for Dexter, split conceptually into three implementation batches.

## Install/update dependencies

From the `dexter-tools` directory:

```powershell
.\install_requirements.ps1
```

Or directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

## Tests

The tester is deliberately external to the tool implementation and can be run by batch:

```powershell
.\.venv\Scripts\python.exe .\test_tools.py --batch 1
.\.venv\Scripts\python.exe .\test_tools.py --batch 2
.\.venv\Scripts\python.exe .\test_tools.py --batch 3
.\.venv\Scripts\python.exe .\test_tools.py --batch chains
.\.venv\Scripts\python.exe .\test_tools.py --batch all
```

Batch 3 does not actually click or type during automatic tests. Its scheduler is exercised by monkey-patching the click operation at the tester layer, while screenshots/template matching use the live desktop.

## Safety defaults

- Filesystem access is restricted to `ALLOWED_ROOTS`.
- Path resolution prevents following an allowed path through a symlink/junction to an outside target.
- Copy/move/rename refuse overwrite and verify postconditions.
- `delete_file` uses the Recycle Bin; permanent deletion is not exposed to the model.
- `open_app` accepts only configured aliases and never invokes a shell.
- App arguments are denied by default.
- `open_file` blocks executable/script/shortcut extensions so it cannot bypass the app allowlist.
- `open_url` accepts only HTTP/HTTPS.
- Window closing is graceful (`WM_CLOSE`); `close_app` does not hard-kill processes.
- PyAutoGUI's corner fail-safe remains enabled.
- Repeating click jobs are cancellable and can have run/time/visual stop conditions.
