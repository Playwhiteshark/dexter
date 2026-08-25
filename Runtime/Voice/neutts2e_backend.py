"""Dexter SpeechBackend adapter for the persistent NeuTTS-2E worker."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from Runtime.Voice.tts import AudioChunk

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = Path(__file__).resolve().parent / "neutts2e_worker_launcher.py"
RUNTIME_ROOT = Path(
    os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(REPO_ROOT)
) / "Dexter" / "Voice" / "neutts2e"


class NeuTTS2EBackend:
    """Stream Paul/neutral PCM from a child process in Dexter's own environment."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._request_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self.ready_info: dict[str, object] | None = None
        self.last_synthesis_stats: dict[str, object] | None = None

    @staticmethod
    def runtime_python() -> Path:
        # TTS is process-isolated, not environment-isolated. uv pins the entire
        # Dexter application to Python 3.13, so the worker uses the exact same
        # interpreter and installed dependency set as the Host.
        return Path(sys.executable)

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

        # Validated production path on the target machine:
        # Q4 backbone -> Intel Arc 140T Vulkan0; ONNX codec + Perth -> CPU.
        # GGUF generation and CPU post-processing are pipelined so Perth remains
        # enabled without forcing the two independent devices to run serially.
        # 128/128 llama.cpp batch/ubatch reduced the backbone by ~325 MB working
        # set / ~644 MB private versus upstream 512/512 while sustained Gemma +
        # TTS remained faster than realtime (RTF 0.794 vs 0.807 at 512).
        # Preserve explicit developer overrides when troubleshooting.
        env.setdefault("DEXTER_TTS_VARIANT", "q4")
        env.setdefault("DEXTER_TTS_BACKBONE_DEVICE", "gpu")
        env.setdefault("DEXTER_TTS_VULKAN_DEVICE", "0")
        env.setdefault("DEXTER_TTS_PIPELINE_WATERMARK", "1")
        env.setdefault("DEXTER_TTS_BACKBONE_BATCH", "128")
        env.setdefault("DEXTER_TTS_BACKBONE_UBATCH", "128")
        env.setdefault("GGML_VK_VISIBLE_DEVICES", env["DEXTER_TTS_VULKAN_DEVICE"])
        return env

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("NeuTTS-2E backend is already running")

        python = self.runtime_python()
        if sys.version_info[:2] != (3, 13):
            raise RuntimeError(
                "Dexter NeuTTS-2E requires the project Python 3.13 environment. "
                "Run Dexter with `uv run`, after `uv sync`."
            )
        if not WORKER_PATH.exists():
            raise RuntimeError(f"Missing NeuTTS-2E worker launcher: {WORKER_PATH}")

        # Start the latency-sensitive worker at AboveNormal directly. This is
        # more reliable on Windows than changing priority from Python after
        # process creation, and preserves the ~30-40% concurrent-generation win
        # measured on the target machine. Never use HIGH/REALTIME here.
        creationflags = (
            subprocess.ABOVE_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
        )
        worker_env = self.runtime_env()

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
            env=worker_env,
            creationflags=creationflags,
        )
        self._process = process

        message = self._read_message()
        if message.get("type") != "ready":
            self.stop()
            if message.get("type") == "fatal":
                raise RuntimeError(f"NeuTTS-2E failed to start: {message.get('message')}")
            raise RuntimeError(f"Unexpected NeuTTS-2E startup message: {message}")
        # The wrapper owns these llama.cpp constructor settings; surface them in
        # the same ready-info dictionary used by diagnostics even though the
        # upstream worker itself does not know about the wrapper.
        message["backbone_batch"] = worker_env["DEXTER_TTS_BACKBONE_BATCH"]
        message["backbone_ubatch"] = worker_env["DEXTER_TTS_BACKBONE_UBATCH"]
        self.ready_info = message
        self.last_synthesis_stats = None

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
            self.last_synthesis_stats = None
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
                        yield AudioChunk(
                            samples=samples,
                            sample_rate=int(message["sample_rate"]),
                        )
                    elif kind == "done":
                        # Keep the worker's optional stage profile available to
                        # diagnostics without changing the SpeechBackend API.
                        self.last_synthesis_stats = dict(message)
                        return
                    elif kind in {"error", "fatal"}:
                        raise RuntimeError(
                            str(message.get("message") or "NeuTTS-2E worker error")
                        )
                    else:
                        raise RuntimeError(f"Unexpected NeuTTS-2E message: {message}")
            finally:
                finished.set()

    def stop(self) -> None:
        process = self._process
        self._process = None
        self.ready_info = None
        self.last_synthesis_stats = None
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
            raise RuntimeError(
                f"NeuTTS-2E worker closed its protocol stream (exit={code})"
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid NeuTTS-2E worker message: {line!r}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Invalid NeuTTS-2E worker payload: {value!r}")
        return value
