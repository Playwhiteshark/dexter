from __future__ import annotations

import unittest

from Runtime.Host.activity_events import (
    ActivityRecord,
    normalize_acp_activity,
)


class ActivityEventTests(unittest.TestCase):
    def test_web_tool_call_is_detected_and_bounded(self) -> None:
        update = normalize_acp_activity(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "web-1",
                "title": "dexter tools: web search",
                "kind": "search",
                "status": "pending",
                "rawInput": {"query": "UAE news today"},
            }
        )
        self.assertIsNotNone(update)
        assert update is not None
        self.assertEqual(update.activity_id, "tool:web-1")
        self.assertEqual(update.tool_name, "web_search")
        self.assertTrue(update.is_web_search)
        self.assertIn("UAE news today", update.input_text or "")

        record = ActivityRecord.from_update(update)
        self.assertTrue(record.is_web_search)
        self.assertEqual(record.status, "pending")

        completed = normalize_acp_activity(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "web-1",
                "status": "completed",
                "rawOutput": {"results": [{"title": "Current result"}]},
            }
        )
        assert completed is not None
        record.apply(completed)
        self.assertEqual(record.status, "completed")
        self.assertTrue(record.is_web_search)
        self.assertIn("Current result", record.details())

    def test_thought_chunks_are_appended_by_message_id(self) -> None:
        first = normalize_acp_activity(
            {
                "sessionUpdate": "agent_thought_chunk",
                "messageId": "thought-7",
                "content": {"type": "text", "text": "Checking "},
            }
        )
        second = normalize_acp_activity(
            {
                "sessionUpdate": "agent_thought_chunk",
                "messageId": "thought-7",
                "content": {"type": "text", "text": "the options."},
            }
        )
        assert first is not None and second is not None
        record = ActivityRecord.from_update(first)
        record.apply(second)
        self.assertEqual(record.activity_id, "thought:thought-7")
        self.assertEqual(record.content_text, "Checking the options.")

    def test_large_tool_payload_is_bounded_before_ui_storage(self) -> None:
        update = normalize_acp_activity(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "large-1",
                "title": "dexter tools: read file",
                "rawOutput": {"text": "x" * 10_000},
            }
        )
        assert update is not None
        self.assertIsNotNone(update.output_text)
        self.assertLessEqual(len(update.output_text or ""), 3_000)
        self.assertTrue((update.output_text or "").endswith("…"))

    def test_plan_entries_become_one_compact_record(self) -> None:
        update = normalize_acp_activity(
            {
                "sessionUpdate": "plan",
                "entries": [
                    {"status": "completed", "content": "Inspect state"},
                    {"status": "in_progress", "content": "Open project"},
                ],
            }
        )
        assert update is not None
        record = ActivityRecord.from_update(update)
        self.assertEqual(record.category, "plan")
        self.assertIn("completed: Inspect state", record.content_text)
        self.assertIn("in_progress: Open project", record.content_text)

    def test_non_activity_update_is_ignored(self) -> None:
        self.assertIsNone(
            normalize_acp_activity(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Hello"},
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
