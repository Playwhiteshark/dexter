from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Runtime.Host.goose_acp import GooseACP


ALLOW = SimpleNamespace(option_id="allow", name="Allow", kind="allow_once")
REJECT = SimpleNamespace(option_id="reject", name="Reject", kind="reject_once")


def tool(title: str, call_id: str, raw_input: dict | None = None):
    return SimpleNamespace(
        title=title,
        tool_call_id=call_id,
        raw_input=raw_input or {},
        kind="edit",
    )


class GoosePermissionFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.requests = []
        self.goose = GooseACP(
            cwd=self.root,
            on_permission_request=self.requests.append,
        )
        self.goose._session_id = "session"

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    async def test_known_operation_is_auto_allowed(self) -> None:
        outcome = await self.goose._request_permission(
            "session",
            tool("dexter tools: close every app on active desktop", "close-1"),
            [ALLOW, REJECT],
        )
        self.assertEqual(outcome["outcome"]["optionId"], "allow")
        self.assertEqual(self.requests, [])

    async def test_existing_file_uses_file_specific_session_choice(self) -> None:
        target = self.root / "existing.txt"
        target.write_text("original", encoding="utf-8")
        pending = asyncio.create_task(
            self.goose._request_permission(
                "session",
                tool(
                    "write_file · existing.txt",
                    "write-1",
                    {"path": str(target), "overwrite": True},
                ),
                [ALLOW, REJECT],
            )
        )
        await asyncio.sleep(0)

        request = self.requests[-1]
        self.assertEqual(
            [choice.label for choice in request.options],
            ["Reject", "Allow once", "Allow this file for this chat"],
        )
        self.goose._resolve_permission_on_loop(self.goose._ALLOW_FILE_ID)
        outcome = await pending
        self.assertEqual(outcome["outcome"]["optionId"], "allow")

        target.write_text("Dexter edit", encoding="utf-8")
        self.goose._permission_policy.observe_tool_update(
            SimpleNamespace(tool_call_id="write-1", status="completed")
        )
        repeat = await self.goose._request_permission(
            "session",
            tool(
                "write_file · existing.txt",
                "write-2",
                {"path": str(target), "overwrite": True},
            ),
            [ALLOW, REJECT],
        )
        self.assertEqual(repeat["outcome"]["optionId"], "allow")
        self.assertEqual(len(self.requests), 1)

    async def test_unknown_tool_has_no_persistent_allow_choice(self) -> None:
        pending = asyncio.create_task(
            self.goose._request_permission(
                "session",
                tool("future_tool", "future-1"),
                [ALLOW, REJECT],
            )
        )
        await asyncio.sleep(0)
        request = self.requests[-1]
        self.assertEqual(
            [choice.label for choice in request.options],
            ["Reject", "Allow once"],
        )
        self.goose._resolve_permission_on_loop("reject")
        await pending


if __name__ == "__main__":
    unittest.main()
