"""Best-effort Windows Terminal session state.

Uses psutil when available to inspect descendant shell processes. This remains
optional: workspace capture/restore still works without psutil, but Terminal
cwd/profile recreation becomes less specific.
"""

from __future__ import annotations

import ntpath
import os
from pathlib import Path


_SHELL_PRIORITY = {
    "pwsh.exe": 100,
    "powershell.exe": 95,
    "cmd.exe": 90,
    "wsl.exe": 85,
    "bash.exe": 80,
    "ubuntu.exe": 80,
    "debian.exe": 80,
    "kali.exe": 80,
    "nu.exe": 75,
    "python.exe": 40,
    "python3.exe": 40,
}
_RESTORABLE_INTERACTIVE_SHELLS = {
    "pwsh.exe",
    "powershell.exe",
    "cmd.exe",
    "wsl.exe",
    "bash.exe",
    "ubuntu.exe",
    "debian.exe",
    "kali.exe",
    "nu.exe",
}



def _psutil():
    try:
        import psutil
    except ImportError:
        return None
    return psutil


def capture_terminal_state(window: dict) -> dict:
    state = {"kind": "windows_terminal"}
    psutil = _psutil()
    if psutil is None:
        state["capture_level"] = "window_only"
        return state

    pid = window.get("pid")
    if not isinstance(pid, int):
        state["capture_level"] = "window_only"
        return state

    try:
        root = psutil.Process(pid)
        descendants = root.children(recursive=True)
    except Exception:
        state["capture_level"] = "window_only"
        return state

    candidates = []
    for process in descendants:
        try:
            name = process.name().casefold()
        except Exception:
            continue
        priority = _SHELL_PRIORITY.get(name, 0)
        if priority <= 0:
            continue
        try:
            cwd = process.cwd() or None
        except Exception:
            cwd = None
        try:
            commandline = process.cmdline() or []
        except Exception:
            commandline = []
        try:
            environ = process.environ() or {}
        except Exception:
            environ = {}
        candidates.append(
            {
                "priority": priority,
                "pid": process.pid,
                "shell_name": name,
                "cwd": cwd,
                "commandline": commandline,
                "profile_id": environ.get("WT_PROFILE_ID"),
                "wt_session": environ.get("WT_SESSION"),
            }
        )

    if not candidates:
        state["capture_level"] = "window_only"
        return state

    # Prefer a recognized shell with a usable cwd, then the newest/highest PID
    # as a proxy for the currently active pane when multiple child shells exist.
    candidates.sort(
        key=lambda item: (
            item["priority"],
            1 if item.get("cwd") else 0,
            item["pid"],
        ),
        reverse=True,
    )
    chosen = candidates[0]
    state.update(
        {
            "capture_level": "shell_process",
            "shell_name": chosen.get("shell_name"),
            "cwd": chosen.get("cwd"),
            "commandline": chosen.get("commandline") or [],
            "profile_id": chosen.get("profile_id"),
        }
    )
    return state


def terminal_launch_command(state: dict) -> list[str]:
    command = ["wt.exe", "-w", "-1", "new-tab"]
    cwd = state.get("cwd")
    if isinstance(cwd, str) and cwd and Path(cwd).is_dir():
        command += ["--startingDirectory", cwd]

    # Windows Terminal's documented CLI profile argument takes a profile name,
    # not WT_PROFILE_ID. Until Dexter has a profile-ID->name resolver, preserve
    # the shell command and cwd rather than guessing a profile name.
    cmdline = state.get("commandline")
    if isinstance(cmdline, list) and cmdline:
        executable = str(cmdline[0])
        basename = ntpath.basename(executable).casefold()
        # Restore the interactive shell identity, not its transient arguments.
        # Replaying an old shell command line could re-run user commands or
        # scripts, which is not workspace restoration.
        if basename in _RESTORABLE_INTERACTIVE_SHELLS:
            command.append(executable)
    return command
