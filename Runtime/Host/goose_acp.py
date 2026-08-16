"""Persistent Goose ACP client for the Dexter host.

Stage 4A responsibilities only:
- spawn one local ``goose acp`` process over stdio;
- initialize ACP and create one persistent Goose session;
- accept finalized text prompts through a FIFO;
- stream agent text chunks back through callbacks;
- cancel any permission request until Stage 4B wires the approval UI.

No Qt imports live here. The host is responsible for marshalling callbacks onto
the GUI thread.
"""

from __future__ import annotations

import asyncio
import json
import queue
import shutil
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.interfaces import Client


class GooseState(str, Enum):
    STARTING = "starting"
    IDLE = "idle"
    THINKING = "thinking"
    RESETTING = "resetting"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PermissionChoice:
    option_id: str
    label: str
    kind: str


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    title: str
    detail: str
    options: tuple[PermissionChoice, ...]


StateCallback = Callable[[GooseState], None]
TextCallback = Callable[[str], None]
StringCallback = Callable[[str], None]
VoidCallback = Callable[[], None]
PermissionCallback = Callable[[PermissionRequest], None]


class _ACPClient(Client):
    """Protocol callbacks invoked by the ACP SDK on the ACP event-loop thread."""

    def __init__(self, owner: "GooseACP") -> None:
        self._owner = owner

    async def request_permission(
        self,
        session_id,
        tool_call,
        options,
        **kwargs: Any,
    ):
        return await self._owner._request_permission(
            session_id,
            tool_call,
            options,
        )

    async def session_update(
        self,
        session_id,
        update,
        **kwargs: Any,
    ) -> None:
        self._owner._session_update(session_id, update)


class GooseACP:
    """One persistent local Goose ACP process and session."""

    _STOP = object()
    _ALLOW_CHAT_ID = "__dexter_allow_this_chat__"

    def __init__(
        self,
        *,
        cwd: Path,
        on_state: StateCallback | None = None,
        on_response_start: VoidCallback | None = None,
        on_response_text: TextCallback | None = None,
        on_response_end: StringCallback | None = None,
        on_error: StringCallback | None = None,
        on_permission_request: PermissionCallback | None = None,
        on_permission_cleared: VoidCallback | None = None,
        on_chat_cleared: VoidCallback | None = None,
    ) -> None:
        self.cwd = Path(cwd).resolve()

        self._on_state = on_state
        self._on_response_start = on_response_start
        self._on_response_text = on_response_text
        self._on_response_end = on_response_end
        self._on_error = on_error
        self._on_permission_request = on_permission_request
        self._on_permission_cleared = on_permission_cleared
        self._on_chat_cleared = on_chat_cleared

        self._commands: queue.Queue[str | object] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._process = None
        self._conn = None
        self._session_id: str | None = None
        self._cancel_requested = threading.Event()

        # Permission futures live exclusively on the ACP asyncio thread.
        # UI selections are marshalled onto that loop with call_soon_threadsafe.
        self._permission_future: asyncio.Future | None = None
        self._permission_option_ids: set[str] = set()
        self._permission_scope_key: str | None = None
        self._permission_allow_once_id: str | None = None
        self._session_allowed_scopes: set[str] = set()

        self._started = threading.Event()
        self._stopped = threading.Event()
        self._stop_requested = threading.Event()

        self._state = GooseState.STOPPED
        self._state_lock = threading.Lock()

    @property
    def state(self) -> GooseState:
        with self._state_lock:
            return self._state

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def start(self) -> None:
        """Start the ACP worker. Safe to call once; repeated calls are ignored."""
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_requested.clear()
            self._stopped.clear()
            self._started.clear()

            # Publish STARTING before the thread begins so a very early prompt
            # can be queued rather than mistaken for a stopped client.
            self._set_state(GooseState.STARTING)

            self._thread = threading.Thread(
                target=self._thread_main,
                name="DexterGooseACP",
                daemon=True,
            )
            self._thread.start()

    def submit_prompt(self, text: str) -> bool:
        """Submit one prompt only when the session is fully idle.

        There is intentionally no user-visible prompt queue. The state change
        to THINKING is reserved atomically before placing the command onto the
        worker FIFO, so a second caller cannot sneak another turn in during the
        small cross-thread handoff window.
        """
        prompt = text.strip()
        if not prompt or self._stop_requested.is_set():
            return False

        with self._state_lock:
            if self._state is not GooseState.IDLE:
                return False
            self._state = GooseState.THINKING

        self._cancel_requested.clear()
        self._emit(self._on_state, GooseState.THINKING)
        self._commands.put(prompt)
        return True

    def new_chat(self) -> bool:
        """Create a fresh ACP session while keeping the Goose process alive."""
        loop = self._loop
        conn = self._conn
        if (
            self.state is not GooseState.IDLE
            or loop is None
            or not loop.is_running()
            or conn is None
            or self._stop_requested.is_set()
        ):
            return False

        with self._state_lock:
            if self._state is not GooseState.IDLE:
                return False
            self._state = GooseState.RESETTING

        self._emit(self._on_state, GooseState.RESETTING)
        future = asyncio.run_coroutine_threadsafe(
            self._new_chat_on_loop(),
            loop,
        )
        future.add_done_callback(self._new_chat_done)
        return True

    async def _new_chat_on_loop(self) -> None:
        conn = self._conn
        if conn is None:
            raise RuntimeError("Goose ACP connection is not ready.")

        session = await conn.new_session(
            cwd=str(self.cwd),
            mcp_servers=[],
        )
        self._session_id = session.session_id

        # Dexter's "Always allow this chat" memory lives only here.
        self._session_allowed_scopes.clear()
        self._permission_scope_key = None
        self._permission_allow_once_id = None

    def _new_chat_done(self, future) -> None:
        try:
            future.result()
        except Exception as exc:
            self._emit_error(f"Could not start a new Dexter chat: {exc}")
            self._set_state(GooseState.IDLE)
            return

        self._set_state(GooseState.IDLE)
        self._emit(self._on_chat_cleared)

    def resolve_permission(self, option_id: str) -> bool:
        """Resolve the currently visible ACP permission request.

        The option id must be one of the exact ids supplied by Goose. The
        persistent ACP session remains alive regardless of allow/reject choice.
        """
        selected = option_id.strip()
        loop = self._loop
        if (
            not selected
            or loop is None
            or not loop.is_running()
            or self.state is not GooseState.THINKING
        ):
            return False

        loop.call_soon_threadsafe(
            self._resolve_permission_on_loop,
            selected,
        )
        return True

    def cancel_current(self) -> bool:
        """Request ACP cancellation of the active prompt turn.

        This keeps the same ACP process and session alive. It does not restart
        Goose or create a replacement conversation.
        """
        if self.state is not GooseState.THINKING:
            return False

        self._cancel_requested.set()

        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._cancel_permission_on_loop)

        conn = self._conn
        session_id = self._session_id
        if (
            loop is None
            or not loop.is_running()
            or conn is None
            or session_id is None
        ):
            # The turn may still be in the tiny handoff between submit_prompt()
            # and _run_prompt(); the event above makes _run_prompt cancel it
            # before sending anything to Goose.
            return True

        future = asyncio.run_coroutine_threadsafe(
            conn.cancel(session_id=session_id),
            loop,
        )
        future.add_done_callback(self._cancel_done)
        return True

    def _cancel_done(self, future) -> None:
        try:
            future.result()
        except Exception as exc:
            if not self._stop_requested.is_set():
                self._emit_error(f"Failed to cancel Goose prompt: {exc}")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop accepting prompts and tear down the Goose child process."""
        self._stop_requested.set()
        self._commands.put(self._STOP)

        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._cancel_permission_on_loop)
            loop.call_soon_threadsafe(self._terminate_child)

        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=timeout)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            # _run() handles normal ACP/runtime errors itself. This is only a
            # last-resort guard around event-loop setup/teardown.
            self._emit_error(f"ACP worker failed: {exc}")
            self._set_state(GooseState.ERROR)
        finally:
            self._loop = None
            self._process = None
            self._conn = None
            self._session_id = None
            self._stopped.set()
            if self.state is not GooseState.ERROR:
                self._set_state(GooseState.STOPPED)

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()

        goose = shutil.which("goose")
        if not goose:
            raise RuntimeError("Could not find 'goose' on PATH.")

        if not self.cwd.is_dir():
            raise RuntimeError(f"ACP session directory does not exist: {self.cwd}")

        client = _ACPClient(self)

        try:
            async with spawn_agent_process(
                client,
                goose,
                "acp",
            ) as (conn, process):
                self._process = process
                self._conn = conn

                await conn.initialize(protocol_version=PROTOCOL_VERSION)
                session = await conn.new_session(
                    cwd=str(self.cwd),
                    mcp_servers=[],
                )
                self._session_id = session.session_id
                self._started.set()
                self._set_state(GooseState.IDLE)

                while not self._stop_requested.is_set():
                    command = await asyncio.to_thread(self._commands.get)
                    if command is self._STOP:
                        break

                    assert isinstance(command, str)
                    await self._run_prompt(conn, command)

        except Exception as exc:
            if not self._stop_requested.is_set():
                self._emit_error(f"Goose ACP connection failed: {exc}")
                self._set_state(GooseState.ERROR)
        finally:
            self._process = None

    async def _run_prompt(self, conn, prompt: str) -> None:
        session_id = self._session_id
        if session_id is None:
            self._emit_error("Goose ACP session is not ready.")
            self._cancel_requested.clear()
            self._set_state(GooseState.IDLE)
            return

        # If Cancel was clicked immediately after submission, it may arrive
        # before Goose has an active run to cancel. Treat that as a clean local
        # cancellation rather than accidentally starting the turn anyway.
        if self._cancel_requested.is_set():
            self._cancel_requested.clear()
            self._emit(self._on_response_end, "cancelled")
            if not self._stop_requested.is_set():
                self._set_state(GooseState.IDLE)
            return

        self._emit(self._on_response_start)

        try:
            result = await conn.prompt(
                session_id=session_id,
                prompt=[text_block(prompt)],
            )
        except Exception as exc:
            cancelled = self._cancel_requested.is_set()
            self._cancel_requested.clear()

            if not self._stop_requested.is_set():
                if cancelled:
                    self._emit(self._on_response_end, "cancelled")
                else:
                    self._emit_error(f"Goose prompt failed: {exc}")
                    self._emit(self._on_response_end, "error")
                self._set_state(GooseState.IDLE)
            return

        self._cancel_requested.clear()
        stop_reason = str(getattr(result, "stop_reason", "end_turn"))
        self._emit(self._on_response_end, stop_reason)

        if not self._stop_requested.is_set():
            self._set_state(GooseState.IDLE)

    def _session_update(self, session_id, update) -> None:
        if self._session_id is not None and session_id != self._session_id:
            return

        # Goose 1.45.0 + ACP v1 emitted AgentMessageChunk in the compatibility
        # probe. Match the protocol discriminator too so this remains tolerant
        # of generated-class naming changes within the same schema.
        update_kind = getattr(update, "session_update", None)
        if update_kind != "agent_message_chunk" and type(update).__name__ != "AgentMessageChunk":
            return

        content = getattr(update, "content", None)
        text = getattr(content, "text", None)
        if isinstance(text, str) and text:
            self._emit(self._on_response_text, text)

    @staticmethod
    def _normalize_permission_kind(value) -> str:
        text = str(value or "").strip().lower().replace("-", "_")
        if "." in text:
            text = text.rsplit(".", 1)[-1]
        return text

    @staticmethod
    def _permission_scope(tool_call) -> str:
        """Best stable per-tool key available in ACP v1's permission payload."""
        title = str(getattr(tool_call, "title", "") or "").strip()
        # Goose's Dexter tool titles use a middle-dot suffix for the target
        # path/value. Strip that volatile portion so e.g. create_directory is
        # remembered independently of the specific directory.
        stable_title = title.split(" · ", 1)[0].strip().casefold()
        kind = str(getattr(tool_call, "kind", "") or "").strip().casefold()
        return f"{kind}|{stable_title}"

    @staticmethod
    def _selected_outcome(option_id: str):
        return {
            "outcome": {
                "outcome": "selected",
                "optionId": option_id,
            }
        }

    async def _request_permission(
        self,
        session_id,
        tool_call,
        options,
    ):
        """Bridge one ACP permission request to the Host UI and await a choice."""
        if (
            self._stop_requested.is_set()
            or self._cancel_requested.is_set()
            or (
                self._session_id is not None
                and session_id != self._session_id
            )
        ):
            return {"outcome": {"outcome": "cancelled"}}

        if self._permission_future is not None:
            self._emit_error(
                "Received another permission request while one is pending."
            )
            return {"outcome": {"outcome": "cancelled"}}

        actual: dict[str, PermissionChoice] = {}
        for option in options:
            option_id = str(getattr(option, "option_id", "") or "").strip()
            label = str(getattr(option, "name", "") or "").strip()
            kind = self._normalize_permission_kind(
                getattr(option, "kind", "")
            )
            if option_id and label and kind:
                actual[kind] = PermissionChoice(
                    option_id=option_id,
                    label=label,
                    kind=kind,
                )

        allow_once = actual.get("allow_once")
        reject_once = actual.get("reject_once")
        scope_key = self._permission_scope(tool_call)

        # "Always allow this chat" is intentionally Dexter-local. We never send
        # Goose's protocol-level allow_always here, because ACP describes it as
        # remembered without defining a persistence scope. Reusing allow_once
        # gives us guaranteed session-only behavior.
        if (
            scope_key in self._session_allowed_scopes
            and allow_once is not None
        ):
            return self._selected_outcome(allow_once.option_id)

        choices: list[PermissionChoice] = []
        if reject_once is not None:
            choices.append(
                PermissionChoice(
                    option_id=reject_once.option_id,
                    label="Reject",
                    kind="reject_once",
                )
            )
        if allow_once is not None:
            choices.append(
                PermissionChoice(
                    option_id=allow_once.option_id,
                    label="Allow once",
                    kind="allow_once",
                )
            )
            choices.append(
                PermissionChoice(
                    option_id=self._ALLOW_CHAT_ID,
                    label="Always allow this chat",
                    kind="allow_session",
                )
            )

        if not choices:
            self._emit_error(
                "Goose requested permission without usable allow/reject options."
            )
            return {"outcome": {"outcome": "cancelled"}}

        title = str(getattr(tool_call, "title", "") or "").strip()
        if not title:
            title = "Permission required"

        detail_parts: list[str] = []
        tool_kind = str(getattr(tool_call, "kind", "") or "").strip()
        if tool_kind and tool_kind != "other":
            detail_parts.append(f"Action type: {tool_kind}")

        raw_input = getattr(tool_call, "raw_input", None)
        if raw_input:
            try:
                rendered = json.dumps(
                    raw_input,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            except Exception:
                rendered = str(raw_input)

            if len(rendered) > 1200:
                rendered = rendered[:1197] + "..."
            detail_parts.append(rendered)

        if not detail_parts:
            tool_call_id = str(
                getattr(tool_call, "tool_call_id", "") or ""
            ).strip()
            if tool_call_id:
                detail_parts.append(f"Tool call: {tool_call_id}")

        request = PermissionRequest(
            title=title,
            detail="\n\n".join(detail_parts),
            options=tuple(choices),
        )

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._permission_future = future
        self._permission_option_ids = {
            choice.option_id for choice in choices
        }
        self._permission_scope_key = scope_key
        self._permission_allow_once_id = (
            allow_once.option_id if allow_once is not None else None
        )

        self._emit(self._on_permission_request, request)

        try:
            return await future
        finally:
            if self._permission_future is future:
                self._permission_future = None
                self._permission_option_ids.clear()
                self._permission_scope_key = None
                self._permission_allow_once_id = None
            self._emit(self._on_permission_cleared)

    def _resolve_permission_on_loop(self, option_id: str) -> None:
        future = self._permission_future
        if future is None or future.done():
            return

        if option_id not in self._permission_option_ids:
            self._emit_error(
                f"Refusing unknown ACP permission option: {option_id}"
            )
            return

        if option_id == self._ALLOW_CHAT_ID:
            scope_key = self._permission_scope_key
            allow_once_id = self._permission_allow_once_id
            if not scope_key or not allow_once_id:
                self._emit_error(
                    "Session-scoped permission could not be resolved safely."
                )
                return

            self._session_allowed_scopes.add(scope_key)
            future.set_result(self._selected_outcome(allow_once_id))
            return

        future.set_result(self._selected_outcome(option_id))

    def _cancel_permission_on_loop(self) -> None:
        future = self._permission_future
        if future is None or future.done():
            return

        future.set_result({"outcome": {"outcome": "cancelled"}})

    def _terminate_child(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.returncode is None:
                process.terminate()
        except ProcessLookupError:
            pass

    def _set_state(self, state: GooseState) -> None:
        with self._state_lock:
            if self._state is state:
                return
            self._state = state
        self._emit(self._on_state, state)

    def _emit_error(self, message: str) -> None:
        self._emit(self._on_error, message)

    @staticmethod
    def _emit(callback, *args) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            # UI/debug callbacks must never kill the ACP transport thread.
            pass
