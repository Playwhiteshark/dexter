import asyncio
import shutil
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.interfaces import Client


class ProbeClient(Client):
    """Minimal, deliberately safe ACP client for validating Goose."""

    async def request_permission(
        self,
        session_id,
        tool_call,
        options,
        **kwargs: Any,
    ):
        print(f"[PERMISSION] Unexpected request in session {session_id}")
        print(f"  tool_call: {tool_call}")
        print(f"  options:   {options}")
        print("  -> cancelling/rejecting for this smoke test")
        return {"outcome": {"outcome": "cancelled"}}

    async def session_update(
        self,
        session_id,
        update,
        **kwargs: Any,
    ):
        print(f"[UPDATE {session_id}] {type(update).__name__}: {update}")


async def main() -> None:
    goose = shutil.which("goose")
    if not goose:
        raise RuntimeError("Could not find 'goose' on PATH.")

    cwd = str(Path.cwd())

    print(f"Goose executable: {goose}")
    print(f"Session cwd:      {cwd}")
    print(f"ACP protocol:     {PROTOCOL_VERSION}")
    print()

    client = ProbeClient()

    async with spawn_agent_process(
        client,
        goose,
        "acp",
    ) as (conn, proc):
        init = await conn.initialize(protocol_version=PROTOCOL_VERSION)
        print(f"[OK] initialize: {init}")

        session = await conn.new_session(
            cwd=cwd,
            mcp_servers=[],
        )
        print(f"[OK] new_session: {session.session_id}")

        print()
        print("[PROMPT] Reply with exactly: ACP_OK")
        result = await conn.prompt(
            session_id=session.session_id,
            prompt=[
                text_block(
                    "Reply with exactly: ACP_OK. Do not call any tools."
                )
            ],
        )

        print()
        print(f"[OK] prompt finished: {result}")


if __name__ == "__main__":
    asyncio.run(main())
