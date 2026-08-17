from __future__ import annotations

import os
import subprocess
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

import psutil

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


def launch_app(
    name: str,
    arguments: list[str] | None = None,
) -> dict:
    return _launch_configured_app(
        name,
        arguments,
        require_argument_permission=True,
    )


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
