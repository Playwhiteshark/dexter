"""Dexter SpeechBackend adapter for the persistent NeuTTS-2E worker."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
from pathlib import Path

import numpy as np

from Runtime.Voice.tts import AudioChunk

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = Path(__file__).resolve().parent / "neutts2e_worker.py"
RUNTIME_ROOT = Path(
    os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(REPO_ROOT)
) / "Dexter" / "Voice" / "neutts2e"


class NeuTTS2EBackend:
    """Stream Paul/neutral PCM from a dedicated Python 3.13 NeuTTS process."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._request_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self.ready_info: dict[str, object] | None = None

    @staticmethod
    def runtime_python() -> Path:
        suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        return RUNTIME_ROOT / ".venv" / suffix

    @staticmethod
    def runtime_env() -> dict[str, str]:
        cache = RUNTIME_ROOT / "cache"
        hf_home = cache / "huggingface"
        torch_home = cache / "torch"
        xdg_home = cache / "xdg"
        for path in (hf_home, torch_home, xdg_home):
            path.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.pop("PIP_CONSTRAINT", None)
        env.update(
            {
                "HF_HOME": str(hf_home),
                "HF_HUB_CACHE": str(hf_home / "hub"),
                "TORCH_HOME": str(torch_home),
                "XDG_CACHE_HOME": str(xdg_home),
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        return env

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("NeuTTS-2E backend is already running")

        python = self.runtime_python()
        if not python.exists():
            raise RuntimeError(
                "NeuTTS-2E runtime is not installed. Run: "
                "python3.13 Runtime\\Voice\\setup_neutts2e.py setup"
            )
        if not WORKER_PATH.exists():
            raise RuntimeError(f"Missing NeuTTS-2E worker: {WORKER_PATH}")

        process = subprocess.Popen(
            [str(python), "-u", str(WORKER_PATH)],
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=self.runtime_env(),
        )
        self._process = process

        message = self._read_message()
        if message.get("type") != "ready":
            self.stop()
            if message.get("type") == "fatal":
                raise RuntimeError(f"NeuTTS-2E failed to start: {message.get('message')}")
            raise RuntimeError(f"Unexpected NeuTTS-2E startup message: {message}")
        self.ready_info = message

    def synthesize(
        self,
        text: str,
        cancel_event: threading.Event,
    ):
        process = self._require_process()
        if process.poll() is not None:
            raise RuntimeError("NeuTTS-2E worker exited unexpectedly")

        with self._request_lock:
            self._request_id += 1
            request_id = self._request_id
            self._send({"cmd": "synthesize", "id": request_id, "text": text})

            finished = threading.Event()

            def watch_cancel() -> None:
                while not finished.wait(0.025):
                    if cancel_event.is_set():
                        try:
                            self._send({"cmd": "cancel", "id": request_id})
                        except Exception:
                            pass
                        return

            watcher = threading.Thread(
                target=watch_cancel,
                name=f"DexterNeuTTSCancel-{request_id}",
                daemon=True,
            )
            watcher.start()

            try:
                while True:
                    message = self._read_message()
                    message_id = message.get("id")
                    if message_id is not None and int(message_id) != request_id:
                        raise RuntimeError(
                            "NeuTTS-2E protocol desynchronized: "
                            f"expected request {request_id}, got {message_id}"
                        )

                    kind = message.get("type")
                    if kind == "audio":
                        if cancel_event.is_set():
                            continue
                        raw = base64.b64decode(message["pcm_f32le"])
                        samples = np.frombuffer(raw, dtype="<f4").copy()
                        yield AudioChunk(samples=samples, sample_rate=int(message["sample_rate"]))
                    elif kind == "done":
                        return
                    elif kind in {"error", "fatal"}:
                        raise RuntimeError(str(message.get("message") or "NeuTTS-2E worker error"))
                    else:
                        raise RuntimeError(f"Unexpected NeuTTS-2E message: {message}")
            finally:
                finished.set()

    def stop(self) -> None:
        process = self._process
        self._process = None
        self.ready_info = None
        if process is None:
            return

        if process.poll() is None:
            try:
                self._send_to_process(process, {"cmd": "shutdown"})
                process.wait(timeout=5.0)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    def _require_process(self) -> subprocess.Popen[str]:
        process = self._process
        if process is None:
            raise RuntimeError("NeuTTS-2E backend is not running")
        return process

    def _send(self, payload: dict[str, object]) -> None:
        self._send_to_process(self._require_process(), payload)

    def _send_to_process(
        self,
        process: subprocess.Popen[str],
        payload: dict[str, object],
    ) -> None:
        stdin = process.stdin
        if stdin is None:
            raise RuntimeError("NeuTTS-2E worker stdin is unavailable")
        data = json.dumps(payload, separators=(",", ":")) + "\n"
        with self._write_lock:
            stdin.write(data)
            stdin.flush()

    def _read_message(self) -> dict[str, object]:
        process = self._require_process()
        stdout = process.stdout
        if stdout is None:
            raise RuntimeError("NeuTTS-2E worker stdout is unavailable")
        line = stdout.readline()
        if not line:
            code = process.poll()
            raise RuntimeError(f"NeuTTS-2E worker closed its protocol stream (exit={code})")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid NeuTTS-2E worker message: {line!r}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Invalid NeuTTS-2E worker payload: {value!r}")
        return value
