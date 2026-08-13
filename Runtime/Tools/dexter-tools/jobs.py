from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from config import (
    MAX_BACKGROUND_JOBS,
    MIN_CLICK_INTERVAL_SECONDS,
)
from computer_tools import click_mouse
from screen_tools import find_image_on_screen


JobState = Literal["running", "completed", "cancelled", "failed"]


@dataclass
class ClickJob:
    job_id: str
    interval_seconds: float
    button: str
    position_mode: str
    fixed_x: int | None
    fixed_y: int | None
    max_runs: int | None
    max_seconds: float | None
    stop_image_path: str | None
    stop_image_confidence: float
    created_monotonic: float = field(default_factory=time.monotonic)
    started_monotonic: float | None = None
    ended_monotonic: float | None = None
    runs: int = 0
    state: JobState = "running"
    stop_reason: str | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def snapshot(self) -> dict:
        now = time.monotonic()
        started = self.started_monotonic or self.created_monotonic
        ended = self.ended_monotonic or now
        return {
            "job_id": self.job_id,
            "kind": "click_interval",
            "state": self.state,
            "interval_seconds": self.interval_seconds,
            "button": self.button,
            "position_mode": self.position_mode,
            "fixed_position": (
                {"x": self.fixed_x, "y": self.fixed_y}
                if self.fixed_x is not None and self.fixed_y is not None
                else None
            ),
            "runs": self.runs,
            "max_runs": self.max_runs,
            "max_seconds": self.max_seconds,
            "stop_image_path": self.stop_image_path,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "elapsed_seconds": round(max(0.0, ended - started), 3),
        }


_LOCK = threading.RLock()
_JOBS: dict[str, ClickJob] = {}


def _active_count() -> int:
    return sum(1 for job in _JOBS.values() if job.state == "running")


def _run_click_job(job: ClickJob) -> None:
    job.started_monotonic = time.monotonic()

    try:
        while True:
            if job.cancel_event.is_set():
                job.state = "cancelled"
                job.stop_reason = "cancelled_by_user"
                break

            elapsed = time.monotonic() - job.started_monotonic
            if job.max_seconds is not None and elapsed >= job.max_seconds:
                job.state = "completed"
                job.stop_reason = "max_seconds_reached"
                break

            if job.max_runs is not None and job.runs >= job.max_runs:
                job.state = "completed"
                job.stop_reason = "max_runs_reached"
                break

            if job.stop_image_path:
                visual = find_image_on_screen(
                    template_path=job.stop_image_path,
                    confidence=job.stop_image_confidence,
                )
                if visual["found"]:
                    job.state = "completed"
                    job.stop_reason = "stop_image_found"
                    break

            if job.position_mode == "current":
                click_mouse(button=job.button)
            else:
                click_mouse(
                    button=job.button,
                    x=job.fixed_x,
                    y=job.fixed_y,
                )

            job.runs += 1

            if job.cancel_event.wait(job.interval_seconds):
                job.state = "cancelled"
                job.stop_reason = "cancelled_by_user"
                break

        if job.state == "running":
            job.state = "completed"
            job.stop_reason = job.stop_reason or "completed"

    except Exception as exc:
        job.state = "failed"
        job.stop_reason = "error"
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.ended_monotonic = time.monotonic()


def start_click_interval_job(
    interval_seconds: float,
    position_mode: Literal["current", "fixed"] = "current",
    x: int | None = None,
    y: int | None = None,
    button: Literal["left", "right", "middle"] = "left",
    max_runs: int | None = None,
    max_seconds: float | None = None,
    stop_image_path: str | None = None,
    stop_image_confidence: float = 0.90,
) -> dict:
    if interval_seconds < MIN_CLICK_INTERVAL_SECONDS or interval_seconds > 3600:
        raise ValueError(
            f"interval_seconds must be between "
            f"{MIN_CLICK_INTERVAL_SECONDS} and 3600."
        )
    if position_mode not in {"current", "fixed"}:
        raise ValueError("position_mode must be current or fixed.")
    if button not in {"left", "right", "middle"}:
        raise ValueError("button must be left, right, or middle.")
    if max_runs is not None and (max_runs < 1 or max_runs > 1_000_000):
        raise ValueError("max_runs must be between 1 and 1,000,000.")
    if max_seconds is not None and (max_seconds <= 0 or max_seconds > 7 * 24 * 3600):
        raise ValueError("max_seconds must be positive and at most 7 days.")
    if not 0 < stop_image_confidence <= 1:
        raise ValueError("stop_image_confidence must be greater than 0 and at most 1.")

    if position_mode == "current":
        if x is not None or y is not None:
            raise ValueError(
                "For position_mode='current', omit x and y. "
                "Each click will occur at the cursor's current position without moving it."
            )
        fixed_x = fixed_y = None
    else:
        if (x is None) != (y is None):
            raise ValueError("For fixed mode, supply both x and y or neither.")
        if x is None:
            # Capture the current point once, then anchor all future clicks there.
            from computer_tools import perform_computer_action
            position = perform_computer_action("cursor_position")
            fixed_x = int(position["x"])
            fixed_y = int(position["y"])
        else:
            # click_mouse validates the actual coordinate before the first click;
            # this preflight keeps invalid jobs from being registered.
            from computer_tools import _validate_coordinate
            _validate_coordinate(x, y)
            fixed_x, fixed_y = x, y

    if stop_image_path:
        # Validate path/confidence before starting a thread. "Not found" is fine.
        find_image_on_screen(
            template_path=stop_image_path,
            confidence=stop_image_confidence,
        )

    with _LOCK:
        if _active_count() >= MAX_BACKGROUND_JOBS:
            raise RuntimeError(
                f"Dexter already has {MAX_BACKGROUND_JOBS} active background jobs."
            )

        job_id = uuid.uuid4().hex[:12]
        job = ClickJob(
            job_id=job_id,
            interval_seconds=interval_seconds,
            button=button,
            position_mode=position_mode,
            fixed_x=fixed_x,
            fixed_y=fixed_y,
            max_runs=max_runs,
            max_seconds=max_seconds,
            stop_image_path=stop_image_path,
            stop_image_confidence=stop_image_confidence,
        )
        _JOBS[job_id] = job

        thread = threading.Thread(
            target=_run_click_job,
            args=(job,),
            name=f"dexter-click-{job_id}",
            daemon=True,
        )
        job.thread = thread
        thread.start()

    return {
        "status": "ok",
        "operation": "start_click_job",
        "job": job.snapshot(),
        "started": True,
    }


def cancel_background_job(job_id: str, wait_seconds: float = 2.0) -> dict:
    if not job_id:
        raise ValueError("job_id must not be empty.")
    if wait_seconds < 0 or wait_seconds > 30:
        raise ValueError("wait_seconds must be between 0 and 30.")

    with _LOCK:
        job = _JOBS.get(job_id)

    if job is None:
        raise KeyError(f"Unknown background job: {job_id}")

    if job.state != "running":
        return {
            "status": "ok",
            "operation": "cancel_job",
            "already_finished": True,
            "job": job.snapshot(),
        }

    job.cancel_event.set()
    if job.thread and wait_seconds:
        job.thread.join(timeout=wait_seconds)

    return {
        "status": "ok",
        "operation": "cancel_job",
        "already_finished": False,
        "job": job.snapshot(),
    }


def get_job_status(job_id: str) -> dict:
    with _LOCK:
        job = _JOBS.get(job_id)

    if job is None:
        raise KeyError(f"Unknown background job: {job_id}")

    return {
        "status": "ok",
        "operation": "job_status",
        "job": job.snapshot(),
    }


def list_background_jobs(include_finished: bool = False) -> dict:
    with _LOCK:
        jobs = list(_JOBS.values())

    snapshots = [
        job.snapshot()
        for job in jobs
        if include_finished or job.state == "running"
    ]

    return {
        "status": "ok",
        "operation": "list_jobs",
        "jobs": snapshots,
        "count": len(snapshots),
    }
