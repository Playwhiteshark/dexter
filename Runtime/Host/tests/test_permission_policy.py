from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Runtime.Host.permission_policy import (
    PermissionDecision,
    ToolPermissionPolicy,
)


def call(
    title: str,
    tool_call_id: str,
    raw_input: dict | None = None,
    status: str | None = None,
):
    return SimpleNamespace(
        title=title,
        tool_call_id=tool_call_id,
        raw_input=raw_input or {},
        status=status,
    )


class ToolPermissionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.policy = ToolPermissionPolicy(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_known_tools_auto_approve_and_unknown_tools_ask(self) -> None:
        known = self.policy.evaluate(call("close_all_apps", "known"))
        unknown = self.policy.evaluate(call("future_dangerous_tool", "unknown"))
        self.assertIs(known.decision, PermissionDecision.AUTO_ALLOW)
        self.assertIs(unknown.decision, PermissionDecision.ASK_ONCE)

    def test_new_file_is_trusted_after_success_and_hash_checked(self) -> None:
        target = self.root / "new.txt"
        first = call("dexter_tools__write_file · new.txt", "write-1", {"path": str(target)})
        self.assertIs(
            self.policy.evaluate(first).decision,
            PermissionDecision.AUTO_ALLOW,
        )

        target.write_text("one", encoding="utf-8")
        self.policy.observe_tool_update(call("write_file", "write-1", status="completed"))

        second = call(
            "write_file · new.txt",
            "write-2",
            {"path": str(target), "overwrite": True},
        )
        self.assertIs(
            self.policy.evaluate(second).decision,
            PermissionDecision.AUTO_ALLOW,
        )
        target.write_text("two", encoding="utf-8")
        self.policy.observe_tool_update(call("write_file", "write-2", status="completed"))

        target.write_text("changed elsewhere", encoding="utf-8")
        third = call(
            "write_file · new.txt",
            "write-3",
            {"path": str(target), "overwrite": True},
        )
        self.assertIs(
            self.policy.evaluate(third).decision,
            PermissionDecision.ASK_FILE,
        )

    def test_existing_file_can_be_allowed_for_chat_then_reset(self) -> None:
        target = self.root / "existing.txt"
        target.write_text("original", encoding="utf-8")
        first = call(
            "write_file · existing.txt",
            "write-1",
            {"path": str(target), "overwrite": True},
        )
        evaluation = self.policy.evaluate(first)
        self.assertIs(evaluation.decision, PermissionDecision.ASK_FILE)

        self.policy.approve_file_for_chat(target, "write-1")
        target.write_text("approved edit", encoding="utf-8")
        self.policy.observe_tool_update(call("write_file", "write-1", status="completed"))
        self.assertIs(
            self.policy.evaluate(
                call(
                    "write_file · existing.txt",
                    "write-2",
                    {"path": str(target), "overwrite": True},
                )
            ).decision,
            PermissionDecision.AUTO_ALLOW,
        )

        self.policy.reset()
        self.assertIs(
            self.policy.evaluate(
                call(
                    "write_file · existing.txt",
                    "write-3",
                    {"path": str(target), "overwrite": True},
                )
            ).decision,
            PermissionDecision.ASK_FILE,
        )

    def test_existing_file_without_overwrite_auto_allows_expected_refusal(self) -> None:
        target = self.root / "protected.txt"
        target.write_text("original", encoding="utf-8")
        evaluation = self.policy.evaluate(
            call("write_file · protected.txt", "write-1", {"path": str(target)})
        )
        self.assertIs(evaluation.decision, PermissionDecision.AUTO_ALLOW)

    def test_unprompted_new_file_tool_update_is_observed(self) -> None:
        target = self.root / "observed.png"
        started = call(
            "screenshot · observed.png",
            "shot-1",
            {"output_path": str(target), "overwrite": False},
            "in_progress",
        )
        self.policy.observe_tool_update(started)
        target.write_bytes(b"png")
        self.policy.observe_tool_update(call("screenshot", "shot-1", status="completed"))

        next_capture = call(
            "screenshot · observed.png",
            "shot-2",
            {"output_path": str(target), "overwrite": True},
        )
        self.assertIs(
            self.policy.evaluate(next_capture).decision,
            PermissionDecision.AUTO_ALLOW,
        )


if __name__ == "__main__":
    unittest.main()
