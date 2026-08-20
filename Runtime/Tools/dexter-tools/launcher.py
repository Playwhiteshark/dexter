from __future__ import annotations

import os
import shutil
import subprocess
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

import psutil

from app_discovery import (
    launch_installed_app,
    resolve_installed_app,
    resolve_installed_app_executable,
)
from config import APP_ALIASES, DANGEROUS_OPEN_FILE_EXTENSIONS
from filesystem import resolve_allowed_path


def list_app_aliases_configured() -> dict:
    return {
        "status": "ok",
        "operation": "list_app_aliases",
        "aliases": sorted(APP_ALIASES.keys()),
        "count": len(APP_ALIASES),
    }


def _resolve_app_alias(name: str) -> tuple[list[str], bool]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("App name must be a non-empty configured alias.")

    key = name.strip().lower()
    matches = {
        alias.lower(): config
        for alias, config in APP_ALIASES.items()
    }

    if key not in matches:
        raise ValueError(
            f"App alias is not configured: {name!r}. "
            f"Configured aliases: {', '.join(sorted(APP_ALIASES)) or '<none>'}"
        )

    config = matches[key]
    if isinstance(config, str):
        return [config], False
    if isinstance(config, (list, tuple)):
        return [str(part) for part in config], False
    if not isinstance(config, dict):
        raise ValueError(f"Invalid APP_ALIASES configuration for {name!r}.")

    command = config.get("command")
    if isinstance(command, str):
        parts = [command]
    elif isinstance(command, (list, tuple)) and command:
        parts = [str(part) for part in command]
    else:
        raise ValueError(f"Configured app {name!r} has no valid command.")

    return parts, bool(config.get("allow_arguments", False))


def _launch_configured_app(
    name: str,
    arguments: list[str] | None = None,
    *,
    require_argument_permission: bool,
) -> dict:
    command, allow_arguments = _resolve_app_alias(name)
    extra = arguments or []

    if extra and require_argument_permission and not allow_arguments:
        raise PermissionError(
            f"Arguments are disabled for app alias {name!r}. "
            "Use a dedicated Dexter tool such as open_file/open_folder instead."
        )

    if len(extra) > 32:
        raise ValueError("Too many app arguments; limit is 32.")

    for argument in extra:
        if not isinstance(argument, str):
            raise ValueError("Every app argument must be a string.")
        if "\x00" in argument:
            raise ValueError("App arguments may not contain NUL characters.")
        if len(argument) > 4096:
            raise ValueError("An app argument is unreasonably long.")

    executable = Path(command[0])
    if executable.is_absolute() and not executable.is_file():
        raise FileNotFoundError(f"Configured app executable does not exist: {executable}")

    proc = subprocess.Popen(
        command + extra,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(0.15)
    alive = psutil.pid_exists(proc.pid) and proc.poll() is None

    return {
        "status": "ok",
        "operation": "open_app",
        "app": name,
        "command": command,
        "arguments": extra,
        "pid": proc.pid,
        "process_alive_after_launch": alive,
        "dispatched": True,
        # Launchers such as Explorer may hand work to an existing process and
        # exit immediately; this is not necessarily a failure.
        "verified": alive,
    }


def _has_configured_app_alias(name: str) -> bool:
    if not isinstance(name, str) or not name.strip():
        return False
    key = name.strip().casefold()
    return any(alias.casefold() == key for alias in APP_ALIASES)


def launch_app(
    name: str,
    arguments: list[str] | None = None,
) -> dict:
    """Launch a configured override alias or a strongly matched installed app.

    Explicit APP_ALIASES remain authoritative and retain their argument policy.
    Ordinary installed applications are discovered internally from Windows'
    Start app catalog / Start Menu, so the model never needs executable paths or
    a manually curated alias for every application.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("App name must be a non-empty string.")

    if _has_configured_app_alias(name):
        result = _launch_configured_app(
            name,
            arguments,
            require_argument_permission=True,
        )
        result["app"] = name.strip()
        result["matched_name"] = name.strip()
        result["source"] = "configured_alias"
        return result

    if arguments:
        raise PermissionError(
            "Arguments are allowed only for an explicitly configured APP_ALIASES entry. "
            "Ordinary installed apps may be opened by name without arguments."
        )

    candidate = resolve_installed_app(name)
    return launch_installed_app(candidate)


def launch_app_with_trusted_arguments(
    name: str,
    arguments: list[str],
) -> dict:
    """Launch a configured app with arguments produced by trusted code.

    This is intentionally not model-facing. Callers must validate any paths or
    other values before constructing the fixed argument list.
    """
    return _launch_configured_app(
        name,
        arguments,
        require_argument_permission=False,
    )


def _find_vscode_executable() -> str:
    """Find Code.exe from Windows app discovery or standard installations."""
    try:
        return resolve_installed_app_executable("Visual Studio Code", {"Code.exe"})
    except (FileNotFoundError, RuntimeError, ValueError):
        pass

    candidates: list[Path] = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    program_files = os.environ.get("ProgramFiles")
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "Microsoft VS Code" / "Code.exe")
    for root in (program_files, program_files_x86):
        if root:
            candidates.append(Path(root) / "Microsoft VS Code" / "Code.exe")

    discovered = shutil.which("code.exe") or shutil.which("code")
    if discovered:
        candidates.append(Path(discovered))

    for candidate in candidates:
        if candidate.is_absolute() and candidate.is_file() and candidate.name.casefold() == "code.exe":
            return str(candidate)

    raise FileNotFoundError(
        "Visual Studio Code is installed but Dexter could not resolve Code.exe. "
        "A configured 'vscode' APP_ALIASES override can still be used for a nonstandard install."
    )


def launch_vscode_project(target: str, configured_alias: str = "vscode") -> dict:
    """Open one validated project and leave editor/session restore to VS Code."""
    project = Path(target).expanduser().resolve(strict=False)
    if not project.exists():
        raise FileNotFoundError(f"Saved VS Code project no longer exists: {project}")
    if not project.is_dir() and not (
        project.is_file() and project.suffix.casefold() == ".code-workspace"
    ):
        raise ValueError("Saved VS Code target must be a folder or .code-workspace file.")
    if os.name != "nt":
        raise RuntimeError("VS Code project launching is currently implemented for Windows only.")

    if _has_configured_app_alias(configured_alias):
        return _launch_configured_app(
            configured_alias,
            ["--new-window", str(project)],
            require_argument_permission=False,
        )

    executable = _find_vscode_executable()
    command = [executable, "--new-window", str(project)]
    proc = subprocess.Popen(
        command,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "status": "ok",
        "operation": "open_vscode_project",
        "app": "Visual Studio Code",
        "source": "installed_app",
        "command": command,
        "pid": proc.pid,
        "dispatched": True,
    }


def configured_app_alias_for_executable(exe_path: str | None) -> str | None:
    """Return one configured alias whose executable matches exe_path."""
    if not exe_path:
        return None

    target = os.path.normcase(os.path.abspath(exe_path))
    target_name = os.path.basename(target)
    matches: list[str] = []

    for alias in APP_ALIASES:
        try:
            command, _allow_arguments = _resolve_app_alias(alias)
        except ValueError:
            continue
        first = command[0]
        if not first:
            continue
        candidate = os.path.normcase(first)
        if os.path.isabs(candidate):
            if os.path.normcase(os.path.abspath(candidate)) == target:
                matches.append(alias)
        elif os.path.basename(candidate) == target_name:
            matches.append(alias)

    return sorted(matches, key=lambda value: (len(value), value.casefold()))[0] if matches else None


def open_allowed_file(path: str) -> dict:
    target = resolve_allowed_path(path)
    if not target.exists():
        raise FileNotFoundError(f"File does not exist: {target}")
    if not target.is_file():
        raise ValueError(f"Path is not a regular file: {target}")
    if target.suffix.lower() in DANGEROUS_OPEN_FILE_EXTENSIONS:
        raise PermissionError(
            f"open_file refuses executable/script/shortcut type {target.suffix!r}. "
            "Use a configured open_app alias or a purpose-built tool instead."
        )
    if os.name != "nt":
        raise RuntimeError("open_file is currently implemented for Windows only.")

    os.startfile(str(target))
    return {
        "status": "ok",
        "operation": "open_file",
        "path": str(target),
        "dispatched": True,
    }


def open_allowed_folder(path: str) -> dict:
    target = resolve_allowed_path(path)
    if not target.exists():
        raise FileNotFoundError(f"Folder does not exist: {target}")
    if not target.is_dir():
        raise ValueError(f"Path is not a directory: {target}")
    if os.name != "nt":
        raise RuntimeError("open_folder is currently implemented for Windows only.")

    os.startfile(str(target))
    return {
        "status": "ok",
        "operation": "open_folder",
        "path": str(target),
        "dispatched": True,
    }


def open_trusted_folder(path: str) -> dict:
    """Open an exact folder selected by trusted persisted-resource code.

    This deliberately does not consult ``ALLOWED_ROOTS``. It is not model-
    facing and must only receive a previously validated/saved resource target;
    opening the folder does not grant filesystem-tool access to its contents.
    """
    target = Path(path).expanduser().resolve(strict=False)
    if not target.exists():
        raise FileNotFoundError(f"Saved resource folder no longer exists: {target}")
    if not target.is_dir():
        raise ValueError(f"Saved resource location is not a directory: {target}")
    if os.name != "nt":
        raise RuntimeError("Saved resource location opening is currently Windows-only.")

    os.startfile(str(target))
    return {
        "status": "ok",
        "operation": "open_resource_location",
        "path": str(target),
        "dispatched": True,
    }


def open_http_url(url: str) -> dict:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")
    if len(url) > 8192:
        raise ValueError("URL is unreasonably long.")

    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are allowed.")
    if not parsed.netloc:
        raise ValueError("URL must include a host.")

    accepted = bool(webbrowser.open(url, new=2, autoraise=True))

    return {
        "status": "ok",
        "operation": "open_url",
        "url": url,
        "browser_accepted_request": accepted,
        "dispatched": True,
    }
