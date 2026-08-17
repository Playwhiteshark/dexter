# Dexter Tools v0.2

Deterministic local-computer tools for Dexter. The direct tool catalog remains in place; the Model Gateway is unchanged. The model chooses tools while deterministic modules enforce permissions, exact execution, verification, and postconditions.

## Main changes

- `agent_results.py` keeps proof-rich internal verification while returning concise successful results to the model.
- `read_image(path)` is the single model-facing image loader. Pillow identifies the content, applies EXIF orientation, uses the first animation frame, fully decodes it, then emits static PNG bytes.
- Corrupt/unsafe images fail before MCP image content is created, preventing the provider-poisoned-session failure mode.
- Normal model-facing limits are smaller: directory 50, file search 25, content search 20 / 3 per file, reads 160 lines / 12k chars, clipboard 10k, processes/windows 30.
- Machine-specific roots and optional app-alias overrides live in ignored `local_config.py`; ordinary installed Windows apps are discovered automatically by `open_app`.
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

Then edit `local_config.py` for this machine. The committed example is deny-by-default and contains no user-specific paths. `local_config.py` is ignored by Git; keep filesystem roots narrow. `APP_ALIASES` is now only for trusted overrides, special targets, or apps that need fixed trusted arguments; normal installed apps do not need entries.

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

The benchmark uses an isolated Goose data root, `GOOSE_MODE=auto`, and a benchmark MCP server that reuses production schemas while simulating effectful tools. The server writes a startup marker before any model call; if that marker is absent the runner reports a **HARNESS ERROR** instead of misgrading the model. Image cases explicitly target `dexter_tools` because Goose can expose another `read_image` with overlapping naming. Subprocess output is decoded as UTF-8 explicitly for Windows. Defaults mirror the current Dexter stack and can be overridden with `DEXTER_BENCHMARK_PROVIDER`, `DEXTER_BENCHMARK_MODEL`, and `DEXTER_BENCHMARK_OLLAMA_HOST`. Production Goose remains `smart_approve`.

## Safety defaults

- Filesystem access is restricted to configured `ALLOWED_ROOTS`.
- Path resolution and recursive walking avoid traversing links/junctions outside allowed roots.
- Copy/move/rename refuse overwrite and retain deterministic postcondition verification.
- `delete_file` uses the Recycle Bin; permanent deletion is not exposed to the model.
- `open_app` resolves configured aliases first, then strongly/uniquely matches the current user's Windows Start app catalog and Start Menu shortcuts. It never searches arbitrary executables or constructs a shell command.
- Model-provided app arguments are accepted only through an explicitly configured alias whose policy allows them; dynamically discovered apps are name-only launches.
- `open_file` blocks executable/script/shortcut extensions so it cannot bypass the app allowlist.
- `open_url` accepts only HTTP/HTTPS.
- Window closing is graceful; `close_app`, `close_all_windows_for_app`, and `close_every_app_on_active_desktop` never hard-kill processes. `close_all_windows_for_app(process_name)` closes every visible window for one exact process name on the active Windows virtual desktop; it is the scoped choice for requests such as “close every VS Code window.” `close_every_app_on_active_desktop` is reserved for an explicit request to close every app. Both desktop-scoped tools snapshot titled user-facing windows, exclude Dexter and Windows shell surfaces, and fail before closing anything if Windows cannot verify virtual-desktop membership.
- PyAutoGUI's corner fail-safe remains enabled.
- Repeating click jobs are cancellable and can have run/time/visual stop conditions.

The Host owns the approval policy above Goose's `smart_approve` requests. Every current tool is classified explicitly and future tools default to Ask. Ordinary reads and computer operations are automatically allowed. `write_file`, plus `screenshot(..., overwrite=True)` for an existing target, asks once per file per chat. Files created by Dexter in the current chat are trusted automatically. After each successful write the Host records the expected SHA-256; an out-of-band change revokes silent trust and asks again. Clear Chat removes all file trust.

## Compact web search

`web_search(query)` returns at most five records containing only `title`, `url`, and a bounded `snippet`. It uses DDGS backends one at a time with a fixed timeout and filters non-HTTP(S) URLs. It does not fetch result pages or expose an arbitrary crawler. Install/update the tool environment with `install_requirements.ps1` to add the `ddgs` dependency.

## Persistent computer map and workspaces

The model-facing catalog now includes a small persistent operational-memory layer:

- `remember_resource(name, kind, target, aliases=None, overwrite=False)` stores a named URL, folder, file, VS Code workspace/folder, Visual Studio solution, or Unity project.
- `search_saved(...)` retrieves only relevant saved resources/workspaces on demand.
- `open_resource(name)` resolves a saved item internally and opens it through the existing safe launch boundaries.
- `save_workspace(name)` captures the current user-facing top-level window layout.
- `restore_workspace(name)` repositions matching open windows and can relaunch missing configured apps or saved development resources when a safe launch hint exists.
- `forget_saved(category, name)` removes a stored resource/workspace.

Operational memory is stored outside the repository by default at `%LOCALAPPDATA%\Dexter\state\computer_memory.json`. This is intentionally separate from conversation history and future biographical/profile memory. `STATE_DIR` may be overridden in ignored `local_config.py`.

Resource paths remain subject to `ALLOWED_ROOTS`. Saving a resource does not create a new filesystem permission boundary.

### Installed app discovery

`open_app(name)` no longer requires one alias per application. Explicit `APP_ALIASES` entries are checked first as deterministic overrides. Otherwise Dexter builds a small in-process index from the current user's Windows Start app catalog (`Get-StartApps`) plus per-user/all-user Start Menu shortcuts, resolves only a strong unique name match, and launches that OS-provided entry. The fixed discovery command contains no model text. The index is cached briefly and refreshed automatically after a miss, so newly installed applications become available without maintaining a permanent app database. Ambiguous names fail with a short list of candidates instead of guessing. No additional MCP tool is exposed; requests such as "open OMEN Gaming Hub" remain a single `open_app` call.

Discovered apps intentionally do not accept model-provided command-line arguments. Apps requiring fixed trusted arguments or unusual launch targets should still use `APP_ALIASES` or a purpose-built internal provider.

### Development-project launch aliases

Saved VS Code, Visual Studio, and Unity projects use configured app aliases. By default Dexter expects aliases named `vscode`, `visual studio`, and `unity`. Override those names with `RESOURCE_APP_ALIASES` in `local_config.py` if needed. The project path is validated first and then passed through fixed app-specific arguments (`--new-window`, solution path, or `-projectPath`); arbitrary model-provided command arguments are not exposed.

### Workspace behavior and current limits

Workspace capture follows the established PowerToys Workspaces pattern: capture application windows, then restore by moving/resizing matching windows after they exist. Normal and snapped/arranged windows store DWM extended-frame bounds as their canonical visible geometry rather than raw `GetWindowRect` bounds, because modern `GetWindowRect` includes invisible resize borders. Floating/minimized/maximized windows use the verified `BeginDeferWindowPos` geometry path. Supported arranged groups are rebuilt through native Windows Snap input and then have their real shared Snap separators moved to the saved DWM physical-pixel coordinates. The Snap worker temporarily uses Per-Monitor-V2 DPI awareness, verifies physical cursor coordinates, and restores the user's original cursor position. Because the MCP server runs behind Dexter rather than as the foreground console, Snap-specific activation also has an attached-input fallback for Windows foreground-focus restrictions. One stubborn window is reported individually and does not abort the rest of the workspace. The Shell desktop window returned by `GetShellWindow` is excluded from snapshots.

If native Snap reconstruction fails, Dexter repairs the windows with exact ordinary geometry so the desktop is not left half-mutated, but that result is explicitly `degraded=true` and `complete=false`: visual adjacency is not treated as a successful restoration of native Snap semantics.

A minimized window is still capturable: Windows `GetWindowPlacement` exposes its normal restore rectangle while minimized. Dexter therefore stores both the minimized state and the size/position the window will return to when restored. A genuinely closed window is different: if it is already closed when a new workspace is captured, there is no live window to inspect. Covered/overlapping windows are separate top-level windows and are captured independently.

Dexter also exposes `set_window_state(hwnd, state)` for explicit `normal`/restore, `minimized`, and `maximized` transitions. This is intentionally separate from `close_window`; asking to minimize a window should never close it.

A genuinely closed window has no current HWND, so workspace restore now uses an internal provider layer before geometry restoration. The model still calls only `restore_workspace`; trusted code first reuses an existing matching HWND or safely relaunches the application, waits for a stable main window, and only then passes that HWND to the generic placement layer. Saved resources and configured app aliases remain the preferred launch hints. Built-in conservative providers can also reopen Windows Terminal, Explorer, VS Code, Visual Studio, Brave/Chrome/Edge, Firefox, and Blender when a safe fixed launch path is available. Unity without a saved Unity resource and Steam without a configured app alias are intentionally not guessed.

Provider-aware startup waiting is designed for splash/launcher applications. For known desktop applications, a newly appeared fixed-size/non-resizable card is treated as transient; Dexter keeps waiting for a stable resizable main window instead of moving/resizing the splash. If the real window does not appear before the timeout, the workspace result reports it as `pending` and leaves the transient window alone. The default launch timeout is 20 seconds and callers may request up to 120 seconds.

This stage does not yet snapshot application-internal state. Browser tabs, Explorer folder identity, VS Code project/editor state without an already saved resource, Terminal tabs/profiles/directories, unsaved documents, and arbitrary application command lines are not inferred from window titles. Those belong in app-specific capture/restore providers rather than the model prompt or generic window layer.

Virtual-desktop orchestration is deliberately not implemented in this version. DWM-cloaked windows are omitted from capture. Bulk close uses the documented `IVirtualDesktopManager::IsWindowOnCurrentVirtualDesktop` membership check so windows on other desktops are left alone, but Dexter does not enumerate, name, create, switch, or assign Windows virtual desktops.

### Why these are normal tools, not dynamically exposed tools

The operational-memory layer deliberately keeps a small stable model-facing API instead of changing the MCP tool list after `search_saved`. `open_resource` performs deterministic name/alias resolution internally, so a normal request such as "open my Dexter repo" can be one tool call; `search_saved` is mainly for discovery, ambiguity, or "what have I saved?" questions.

This avoids turning saved projects into separate tool definitions and keeps resource data out of the prompt until it is requested. If Dexter's total catalog later becomes large enough to hurt selection/context measurably, deferred/tool-search loading should be implemented as a dedicated Host/provider feature and benchmarked rather than as a side effect of a memory lookup.

### Browser and virtual-desktop follow-ups

`remember_resource` can store an explicit HTTP/HTTPS URL now, but Dexter does not yet have a reliable primitive for reading the active Brave/Chrome tab URL. Do not infer URLs from browser window titles. A future browser-context adapter can make requests such as "remember this tab as Fleckle" deterministic.

The current workspace snapshot is scoped to non-cloaked user-facing top-level windows. The small virtual-desktop module is membership-only and exists for active-desktop bulk closing; desktop enumeration/naming/switching remains deferred and must not be implemented through undocumented shell interfaces without a separate decision and regression coverage.

### Steam workspace reopening

Steam windows may be owned by either `steam.exe` or `steamwebhelper.exe`. Workspace restore never launches `steamwebhelper.exe` directly. It resolves the real `steam.exe` from the captured executable path first, then Steam's per-user registry values, common install folders, and finally `PATH`. After dispatch, the normal workspace startup listener ignores transient/fixed-size loading windows and waits for a stable resizable Steam main window before applying geometry.

`save_workspace` now returns compact `captured_windows` summaries so the model can report what was actually saved instead of inventing explanations from `window_count`.
