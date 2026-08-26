"""Owned lifecycle wrapper for Dexter's Model Gateway.

The host never discovers or adopts an arbitrary service already listening on
11435. It launches the repository's gateway.py as its own child and proves
startup from that child's stdout. If the configured port is occupied, the
child fails to bind and Dexter reports the startup failure.

Normal host shutdown terminates only the Gateway process Dexter created.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from Runtime.Host.agent_profile import AgentProfile, load_profile


class GatewayState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    ERROR = "error"


StateCallback = Callable[[GatewayState], None]
StringCallback = Callable[[str], None]


class GatewayProcess:
    READY_PREFIX = "Dexter Model Gateway listening on "

    def __init__(
        self,
        script: Path,
        *,
        profile: AgentProfile | None = None,
        on_state: StateCallback | None = None,
        on_error: StringCallback | None = None,
        on_log: StringCallback | None = None,
    ) -> None:
        self.script = Path(script).resolve()
        self.profile = profile or load_profile()
        self._on_state = on_state
        self._on_error = on_error
        self._on_log = on_log

        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._tail: deque[str] = deque(maxlen=12)
        self._stop_requested = threading.Event()

        self._state = GatewayState.STOPPED
        self._state_lock = threading.Lock()

    @property
    def state(self) -> GatewayState:
        with self._state_lock:
            return self._state

    @property
    def pid(self) -> int | None:
        process = self._process
        return None if process is None else process.pid

    def _gateway_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env["DEXTER_MODEL"] = self.profile.model.name
        env["DEXTER_REASONING_EFFORT"] = self.profile.model.reasoning_effort
        return env

    def start(self, timeout: float = 8.0) -> None:
        if not self.script.is_file():
            raise RuntimeError(f"Gateway script not found: {self.script}")

        process = self._process
        if process is not None and process.poll() is None:
            return

        self._stop_requested.clear()
        self._tail.clear()
        self._set_state(GatewayState.STARTING)

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        # -u makes gateway.py's existing "listening on ..." print immediately
        # visible through the pipe. That line is emitted only after the HTTP
        # server successfully binds, so it proves *our child* owns the port.
        self._process = subprocess.Popen(
            [sys.executable, "-u", str(self.script)],
            cwd=str(self.script.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=self._gateway_environment(),
            creationflags=creationflags,
        )

        self._reader_thread = threading.Thread(
            target=self._read_output,
            name="DexterGatewayOutput",
            daemon=True,
        )
        self._reader_thread.start()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._stop_requested.is_set():
                raise RuntimeError("Gateway startup was cancelled.")

            process = self._process
            if process is None:
                raise RuntimeError("Gateway process disappeared during startup.")

            return_code = process.poll()
            if return_code is not None:
                message = self._startup_failure_message(return_code)
                self._set_state(GatewayState.ERROR)
                raise RuntimeError(message)

            try:
                line = self._lines.get(timeout=0.10)
            except queue.Empty:
                continue

            if line.startswith(self.READY_PREFIX):
                self._set_state(GatewayState.READY)
                self._monitor_thread = threading.Thread(
                    target=self._monitor_process,
                    name="DexterGatewayMonitor",
                    daemon=True,
                )
                self._monitor_thread.start()
                return

        self.stop()
        self._set_state(GatewayState.ERROR)
        raise RuntimeError(
            "Dexter's Gateway did not report ready within "
            f"{timeout:.1f} seconds."
        )

    def stop(self, timeout: float = 4.0) -> None:
        self._stop_requested.set()
        process = self._process

        if process is None:
            if self.state is not GatewayState.ERROR:
                self._set_state(GatewayState.STOPPED)
            return

        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
            except ProcessLookupError:
                pass

        self._process = None
        if self.state is not GatewayState.ERROR:
            self._set_state(GatewayState.STOPPED)

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return

        try:
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                self._tail.append(line)
                self._lines.put(line)
                self._emit(self._on_log, line)
        finally:
            try:
                process.stdout.close()
            except Exception:
                pass

    def _monitor_process(self) -> None:
        process = self._process
        if process is None:
            return

        return_code = process.wait()
        if self._stop_requested.is_set():
            return

        message = (
            f"Dexter's Gateway exited unexpectedly with code {return_code}."
        )
        if self._tail:
            message += "\n" + "\n".join(self._tail)

        self._set_state(GatewayState.ERROR)
        self._emit(self._on_error, message)

    def _startup_failure_message(self, return_code: int) -> str:
        message = (
            f"Dexter could not start its Model Gateway "
            f"(exit code {return_code})."
        )

        tail = "\n".join(self._tail)
        lowered = tail.lower()
        if (
            "address already in use" in lowered
            or "only one usage of each socket address" in lowered
            or "winerror 10048" in lowered
        ):
            message += (
                " Port 11435 appears to already be in use. Dexter will not "
                "attach to an existing Gateway or other process; close the "
                "conflicting process and start Dexter again."
            )

        if tail:
            message += "\n\nGateway output:\n" + tail
        return message

    def _set_state(self, state: GatewayState) -> None:
        with self._state_lock:
            self._state = state
        self._emit(self._on_state, state)

    @staticmethod
    def _emit(callback, *args) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            # Runtime status callbacks must not kill process supervision.
            pass
