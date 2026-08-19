"""Host-owned ACP subprocess launcher.

The ACP SDK does not currently expose Windows process creation flags through
its stdio transport. Dexter needs Goose to remain a stdio ACP child without
opening a console window when the Host is launched with pythonw.exe.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from acp.client.connection import ClientSideConnection
from acp.interfaces import Client
from acp.transports import default_environment


SHUTDOWN_TIMEOUT_SECONDS = 2.0


def _windows_creation_flags() -> int:
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW
    return 0


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
        "env": default_environment(),
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
        with contextlib.suppress(Exception):
            await conn.close()

        if process.returncode is not None:
            return

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
