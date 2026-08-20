"""Host-owned ACP subprocess launcher.

The ACP SDK does not currently expose Windows process creation flags through
its stdio transport. Dexter needs Goose to remain a stdio ACP child without
opening a console window when the Host is launched with pythonw.exe.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from acp.client.connection import ClientSideConnection
from acp.interfaces import Client
from acp.transports import default_environment


SHUTDOWN_TIMEOUT_SECONDS = 2.0
_WINDOWS_DESKTOP_ENV_VARS = (
    "WINDIR",
    "ComSpec",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "ProgramW6432",
)


def _windows_creation_flags() -> int:
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW
    return 0


def _agent_environment() -> dict[str, str]:
    """Keep ACP's trimmed env plus standard Windows desktop runtime paths."""
    env = default_environment()
    if sys.platform == "win32":
        for key in _WINDOWS_DESKTOP_ENV_VARS:
            value = os.environ.get(key)
            if value:
                env[key] = value
    return env


@asynccontextmanager
async def spawn_host_agent_process(
    client: Client,
    command: str,
    *args: str,
    cwd: str | Path | None = None,
) -> AsyncIterator[tuple[ClientSideConnection, asyncio.subprocess.Process]]:
    """Spawn an ACP agent owned by Dexter's Host."""
    kwargs = {
        "stdin": asyncio.subprocess.PIPE,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "env": _agent_environment(),
        "cwd": str(cwd) if cwd is not None else None,
    }
    creationflags = _windows_creation_flags()
    if creationflags:
        kwargs["creationflags"] = creationflags

    process = await asyncio.create_subprocess_exec(command, *args, **kwargs)
    if process.stdin is None or process.stdout is None:
        process.kill()
        await process.wait()
        raise RuntimeError("ACP child requires stdin/stdout pipes.")

    conn = ClientSideConnection(client, process.stdin, process.stdout)
    try:
        yield conn, process
    finally:
        try:
            await conn.close()
        finally:
            if process.returncode is None:
                try:
                    await asyncio.wait_for(
                        process.wait(),
                        timeout=SHUTDOWN_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    process.terminate()
                    try:
                        await asyncio.wait_for(
                            process.wait(),
                            timeout=SHUTDOWN_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
