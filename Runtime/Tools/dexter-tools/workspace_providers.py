"""Internal application adapters for workspace reopening.

These helpers are intentionally not MCP tools. The model still sees only
save_workspace/restore_workspace; provider selection and launch policy stay in
trusted deterministic code.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from launcher import launch_vscode_project
from terminal_state import terminal_launch_command
from vscode_state import vscode_open_target


_PROVIDER_BY_PROCESS = {
    "windowsterminal.exe": "windows_terminal",
    "explorer.exe": "explorer",
    "code.exe": "vscode",
    "devenv.exe": "visual_studio",
    "brave.exe": "chromium_browser",
    "chrome.exe": "chromium_browser",
    "msedge.exe": "chromium_browser",
    "firefox.exe": "firefox",
    "unity.exe": "unity",
    "blender.exe": "blender",
    "steam.exe": "steam",
    "steamwebhelper.exe": "steam",
}

_PROVIDER_PROCESS_NAMES = {
    "windows_terminal": {"windowsterminal.exe"},
    "explorer": {"explorer.exe"},
    "vscode": {"code.exe"},
    "visual_studio": {"devenv.exe"},
    "chromium_browser": {"brave.exe", "chrome.exe", "msedge.exe"},
    "firefox": {"firefox.exe"},
    "unity": {"unity.exe"},
    "blender": {"blender.exe"},
    "steam": {"steam.exe", "steamwebhelper.exe"},
}

# These applications commonly display splash/launcher windows before the real
# workspace window exists. Reopening code waits for a stable resizable window
# instead of applying saved geometry to the first HWND that appears.
_RESIZABLE_MAIN_WINDOW_PROVIDERS = {
    "windows_terminal",
    "explorer",
    "vscode",
    "visual_studio",
    "chromium_browser",
    "firefox",
    "unity",
    "blender",
    "steam",
}

# Built-in direct reopening is deliberately conservative. Unity is omitted
# unless a saved Unity resource exists because launching Unity.exe without a
# project does not reconstruct the saved editor. Steam is handled separately:
# its visible UI may be owned by steamwebhelper.exe, but reopening must always
# dispatch the real Steam client (steam.exe), never the helper executable.
_DIRECT_EXE_PROVIDERS = {
    "visual_studio",
    "chromium_browser",
    "firefox",
    "blender",
}


def provider_for_process_name(process_name: str | None) -> str | None:
    if not process_name:
        return None
    return _PROVIDER_BY_PROCESS.get(process_name.strip().casefold())


def provider_for_saved_window(saved: dict) -> str | None:
    explicit = saved.get("workspace_provider")
    if isinstance(explicit, str) and explicit in _PROVIDER_PROCESS_NAMES:
        return explicit
    return provider_for_process_name(saved.get("process_name"))


def provider_matches_window(provider: str | None, window: dict) -> bool:
    if not provider:
        return False
    process_name = (window.get("process_name") or "").strip().casefold()
    return process_name in _PROVIDER_PROCESS_NAMES.get(provider, set())


def provider_requires_resizable_main(provider: str | None) -> bool:
    return bool(provider and provider in _RESIZABLE_MAIN_WINDOW_PROVIDERS)


def provider_label(provider: str | None) -> str:
    return provider or "generic"


def _validated_saved_executable(saved: dict, provider: str) -> str | None:
    """Return a previously captured executable only for the expected provider.

    This is not arbitrary model-controlled execution: the path comes from the
    persistent workspace snapshot, and the executable basename must still match
    the process/provider that Dexter captured from the live window.
    """
    exe = saved.get("exe")
    process_name = (saved.get("process_name") or "").strip().casefold()
    if not isinstance(exe, str) or not exe.strip() or not process_name:
        return None

    accepted = _PROVIDER_PROCESS_NAMES.get(provider, set())
    if process_name not in accepted:
        return None

    path = Path(exe)
    if not path.is_absolute() or not path.is_file():
        return None
    if path.name.casefold() != process_name:
        return None
    return str(path)



def _steam_executable_from_saved_window(saved: dict) -> str | None:
    """Resolve steam.exe from a captured Steam/steamwebhelper window.

    Steam's visible UI can be hosted by steamwebhelper.exe. Starting that helper
    directly is not a valid client reopen path, so walk upward from the captured
    executable and look for the actual Steam client.
    """
    exe = saved.get("exe")
    if not isinstance(exe, str) or not exe.strip():
        return None

    path = Path(exe)
    if not path.is_absolute():
        return None

    if path.name.casefold() == "steam.exe" and path.is_file():
        return str(path)

    if path.name.casefold() == "steamwebhelper.exe":
        for parent in (path.parent, *path.parents):
            candidate = parent / "steam.exe"
            if candidate.is_file():
                return str(candidate)
    return None


def _steam_executable_from_registry() -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:
        return None

    probes = [
        (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamExe", False),
        (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath", True),
    ]
    for hive, key_name, value_name, is_directory in probes:
        try:
            with winreg.OpenKey(hive, key_name) as key:
                value, _ = winreg.QueryValueEx(key, value_name)
        except OSError:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value.replace("/", "\\"))
        if is_directory:
            candidate = candidate / "steam.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def _find_steam_executable(saved: dict) -> str | None:
    captured = _steam_executable_from_saved_window(saved)
    if captured:
        return captured

    registered = _steam_executable_from_registry()
    if registered:
        return registered

    for env_name in ("ProgramFiles(x86)", "ProgramFiles"):
        root = os.environ.get(env_name)
        if not root:
            continue
        candidate = Path(root) / "Steam" / "steam.exe"
        if candidate.is_file():
            return str(candidate)

    discovered = shutil.which("steam.exe")
    if discovered and Path(discovered).is_file():
        return discovered
    return None


def _spawn(command: list[str], label: str) -> dict:
    proc = subprocess.Popen(
        command,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "status": "ok",
        "operation": "workspace_provider_launch",
        "provider": label,
        "command": command,
        "pid": proc.pid,
        "dispatched": True,
    }


def _launch_steam(saved: dict) -> dict:
    """Open Steam's main client window without ever launching steamwebhelper.

    Prefer Steam's registered URI because it also handles the common case where
    Steam is still running in the tray but its main window is closed. If the URI
    association is unavailable, fall back to the real steam.exe resolved from
    the saved installation.
    """
    uri = "steam://open/main"
    try:
        os.startfile(uri)
        return {
            "status": "ok",
            "operation": "workspace_provider_launch",
            "provider": "steam",
            "command": [uri],
            "transport": "registered_protocol",
            "pid": None,
            "dispatched": True,
        }
    except (AttributeError, OSError):
        steam_exe = _find_steam_executable(saved)
        if not steam_exe:
            raise FileNotFoundError(
                "Steam was recognized, but neither the registered steam:// protocol nor steam.exe "
                "could be used to reopen it."
            )
        return _spawn([steam_exe], "steam")


def launch_workspace_provider(saved: dict) -> dict | None:
    """Reopen a known application when no saved resource/app alias exists.

    Returns None when the provider intentionally has no safe generic reopen
    path. Callers should report that as missing rather than inventing arguments.
    """
    if os.name != "nt":
        raise RuntimeError("Workspace application reopening is currently Windows-only.")

    provider = provider_for_saved_window(saved)
    if provider == "windows_terminal":
        state = saved.get("app_state") if isinstance(saved.get("app_state"), dict) else {}
        return _spawn(terminal_launch_command(state), provider)
    if provider == "explorer":
        state = saved.get("app_state") if isinstance(saved.get("app_state"), dict) else {}
        folder = state.get("folder")
        if isinstance(folder, str) and folder:
            return _spawn(["explorer.exe", folder], provider)
        return _spawn(["explorer.exe"], provider)
    if provider == "steam":
        return _launch_steam(saved)

    if provider == "vscode":
        state = saved.get("app_state") if isinstance(saved.get("app_state"), dict) else {}
        target = vscode_open_target(state)
        if not target:
            return None
        return launch_vscode_project(target)

    if provider not in _DIRECT_EXE_PROVIDERS:
        return None

    exe = _validated_saved_executable(saved, provider)
    if not exe:
        return None

    # For stateful applications, do not force a fresh blank window here. Their
    # normal startup may restore the application's own previous session. A
    # later app-state provider can make that deterministic per Dexter workspace.
    return _spawn([exe], provider)
