"""Bounded, presentation-neutral records for Goose ACP activity updates.

ACP emits much more than assistant text: thought chunks, tool-call starts,
tool-call updates, and plans.  The Host keeps those protocol objects out of Qt
and turns them into a deliberately small record that can be retained with one
visible chat turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from Runtime.Host.permission_policy import tool_name_from_call


ActivityCategory = Literal["tool", "thought", "plan"]

_MAX_JSON_CHARS = 3_000
_MAX_CONTENT_CHARS = 6_000


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return default


def _normalized(value: Any) -> str:
    raw = _value(value, "value", default=value)
    text = str(raw or "").strip().casefold().replace("-", "_")
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _bounded(text: str, limit: int) -> str:
    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 1)].rstrip() + "…"


def _bounded_stream_text(text: str, limit: int) -> str:
    """Bound streamed text without eating meaningful chunk whitespace."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(by_alias=True, exclude_none=True)
        except TypeError:
            return value.model_dump()
    if hasattr(value, "dict"):
        try:
            return value.dict(by_alias=True, exclude_none=True)
        except TypeError:
            return value.dict()
    return value


def _bounded_json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        rendered = json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    except (TypeError, ValueError):
        rendered = str(value)
    return _bounded(rendered, _MAX_JSON_CHARS)


def _content_text(value: Any) -> str | None:
    """Extract only human-readable text from ACP content blocks."""
    if value is None:
        return None
    if isinstance(value, str):
        return (
            _bounded_stream_text(value, _MAX_CONTENT_CHARS)
            if value.strip()
            else None
        )
    if isinstance(value, (list, tuple)):
        parts = [part for item in value if (part := _content_text(item))]
        return _bounded("\n".join(parts), _MAX_CONTENT_CHARS) if parts else None

    text = _value(value, "text")
    if isinstance(text, str) and text.strip():
        return _bounded_stream_text(text, _MAX_CONTENT_CHARS)

    # Tool-call content wraps normal ACP content in a second ``content`` field.
    nested = _value(value, "content")
    if nested is not None and nested is not value:
        return _content_text(nested)
    return None


def _plan_text(update: Any) -> str | None:
    entries = _value(update, "entries")
    if isinstance(entries, (list, tuple)):
        lines: list[str] = []
        for entry in entries:
            content = str(_value(entry, "content", default="") or "").strip()
            if not content:
                continue
            status = _normalized(_value(entry, "status"))
            lines.append(f"{status or 'pending'}: {content}")
        if lines:
            return _bounded("\n".join(lines), _MAX_CONTENT_CHARS)

    content = _value(update, "content")
    return _content_text(content)


@dataclass(frozen=True, slots=True)
class ActivityUpdate:
    activity_id: str
    category: ActivityCategory
    title: str | None = None
    status: str | None = None
    kind: str | None = None
    tool_name: str | None = None
    input_text: str | None = None
    output_text: str | None = None
    content_text: str | None = None
    append_content: bool = False
    is_web_search: bool | None = None


@dataclass(slots=True)
class ActivityRecord:
    activity_id: str
    category: ActivityCategory
    title: str = ""
    status: str = ""
    kind: str = ""
    tool_name: str = ""
    input_text: str = ""
    output_text: str = ""
    content_text: str = ""
    is_web_search: bool = False

    @classmethod
    def from_update(cls, update: ActivityUpdate) -> "ActivityRecord":
        record = cls(update.activity_id, update.category)
        record.apply(update)
        return record

    def apply(self, update: ActivityUpdate) -> None:
        if update.title is not None:
            self.title = update.title.strip()
        if update.status is not None:
            self.status = update.status.strip()
        if update.kind is not None:
            self.kind = update.kind.strip()
        if update.tool_name is not None:
            self.tool_name = update.tool_name.strip()
        if update.input_text is not None:
            self.input_text = update.input_text
        if update.output_text is not None:
            self.output_text = update.output_text
        if update.content_text is not None:
            if update.append_content:
                self.content_text = _bounded_stream_text(
                    self.content_text + update.content_text,
                    _MAX_CONTENT_CHARS,
                )
            else:
                self.content_text = update.content_text
        if update.is_web_search is not None:
            self.is_web_search = update.is_web_search

    @property
    def display_title(self) -> str:
        if self.category == "thought":
            return "Reasoning"
        if self.category == "plan":
            return "Plan"
        return self.tool_name or self.title or "Tool"

    def heading(self) -> str:
        suffix = self.status.replace("_", " ").strip()
        return f"{self.display_title} · {suffix}" if suffix else self.display_title

    def details(self) -> str:
        parts: list[str] = []
        if self.category == "tool" and self.title and self.title != self.display_title:
            parts.append(self.title)
        if self.input_text:
            parts.append(f"Input\n{self.input_text}")
        if self.content_text:
            parts.append(self.content_text)
        if self.output_text:
            parts.append(f"Result\n{self.output_text}")
        return "\n\n".join(parts)


def normalize_acp_activity(update: Any) -> ActivityUpdate | None:
    """Convert one generated ACP update object (or test dict) into UI data."""
    update_kind = _normalized(
        _value(update, "session_update", "sessionUpdate")
    )
    if not update_kind:
        class_name = type(update).__name__
        aliases = {
            "AgentThoughtChunk": "agent_thought_chunk",
            "ToolCall": "tool_call",
            "ToolCallUpdate": "tool_call_update",
            "Plan": "plan",
        }
        update_kind = aliases.get(class_name, "")

    if update_kind == "agent_thought_chunk":
        text = _content_text(_value(update, "content"))
        if not text:
            return None
        message_id = str(
            _value(update, "message_id", "messageId", default="current")
            or "current"
        )
        return ActivityUpdate(
            activity_id=f"thought:{message_id}",
            category="thought",
            title="Reasoning",
            content_text=text,
            append_content=True,
        )

    if update_kind in {"tool_call", "tool_call_update"}:
        call_id = str(
            _value(update, "tool_call_id", "toolCallId", default="") or ""
        ).strip()
        if not call_id:
            return None
        title_value = _value(update, "title")
        title = str(title_value).strip() if title_value is not None else None
        tool_name = tool_name_from_call(update)
        status_value = _value(update, "status")
        kind_value = _value(update, "kind")
        raw_input = _value(update, "raw_input", "rawInput")
        raw_output = _value(update, "raw_output", "rawOutput")
        return ActivityUpdate(
            activity_id=f"tool:{call_id}",
            category="tool",
            title=title,
            status=_normalized(status_value) if status_value is not None else None,
            kind=_normalized(kind_value) if kind_value is not None else None,
            tool_name=tool_name,
            input_text=_bounded_json(raw_input),
            output_text=_bounded_json(raw_output),
            content_text=_content_text(_value(update, "content")),
            is_web_search=(tool_name == "web_search") if tool_name else None,
        )

    if update_kind in {"plan", "plan_update"}:
        text = _plan_text(update)
        if not text:
            return None
        plan_id = str(
            _value(update, "plan_id", "planId", default="current") or "current"
        )
        return ActivityUpdate(
            activity_id=f"plan:{plan_id}",
            category="plan",
            title="Plan",
            content_text=text,
        )

    return None
