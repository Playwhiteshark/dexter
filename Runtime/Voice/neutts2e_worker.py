"""Persistent Python 3.13 NeuTTS-2E worker for Dexter.

The Dexter host deliberately does not import NeuTTS. This worker owns the heavy
ML runtime in a separate process and speaks a tiny JSON-lines protocol over
stdin/stdout. Model/library logging is redirected to stderr so stdout remains a
clean protocol channel.

The worker intentionally uses the PyTorch NeuTTS-2E backbone selected in the
listening test. Dexter's VoiceOutput already pipelines sentence-sized synthesis
with playback, so production does not require llama-cpp-python or a compiler.
"""

from __future__ import annotations

import base64
import contextlib
import json
import queue
import sys
import threading
import time
import traceback
from typing import Any

MODEL_REPO = "neuphonic/neutts-2e"
CODEC_REPO = "neuphonic/neucodec-onnx-decoder-int8"
SPEAKER = "paul"
EMOTION = "neutral"
SAMPLE_RATE = 24_000

_PROTOCOL_OUT = sys.stdout


def _emit(payload: dict[str, Any]) -> None:
    _PROTOCOL_OUT.write(json.dumps(payload, separators=(",", ":")) + "\n")
    _PROTOCOL_OUT.flush()


def _command_reader(
    commands: queue.Queue[dict[str, Any] | None],
    cancelled: set[int],
    cancel_lock: threading.Lock,
    stopping: threading.Event,
) -> None:
    try:
        for raw in sys.stdin:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                print(f"Ignoring malformed Dexter TTS command: {raw!r}", file=sys.stderr)
                continue

            command = message.get("cmd")
            if command == "cancel":
                try:
                    request_id = int(message["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                with cancel_lock:
                    cancelled.add(request_id)
            elif command == "shutdown":
                stopping.set()
                commands.put(None)
                return
            elif command == "synthesize":
                commands.put(message)
    finally:
        stopping.set()
        commands.put(None)


def main() -> int:
    # Imports stay in the worker so the normal Dexter host environment remains
    # independent of NeuTTS, PyTorch, Transformers, and ONNX Runtime.
    import numpy as np
    from neutts import NeuTTS2E

    commands: queue.Queue[dict[str, Any] | None] = queue.Queue()
    cancelled: set[int] = set()
    cancel_lock = threading.Lock()
    stopping = threading.Event()

    reader = threading.Thread(
        target=_command_reader,
        args=(commands, cancelled, cancel_lock, stopping),
        name="DexterNeuTTSCommandReader",
        daemon=True,
    )
    reader.start()

    try:
        print(
            f"[TTS] Loading {MODEL_REPO} / {CODEC_REPO} on CPU...",
            file=sys.stderr,
        )
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            tts = NeuTTS2E(
                backbone_repo=MODEL_REPO,
                backbone_device="cpu",
                codec_repo=CODEC_REPO,
                codec_device="cpu",
                seed=42,
            )
            # Warm the exact selected speaker during Dexter startup so the first
            # answer does not pay model/reference setup costs.
            tts.infer(
                "Warming up.",
                speaker=SPEAKER,
                emotion=EMOTION,
                temperature=1.0,
                top_k=50,
            )
        load_seconds = time.perf_counter() - started
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit({"type": "fatal", "message": str(exc)})
        return 1

    _emit(
        {
            "type": "ready",
            "speaker": SPEAKER,
            "emotion": EMOTION,
            "sample_rate": SAMPLE_RATE,
            "model": MODEL_REPO,
            "codec": CODEC_REPO,
            "mode": "sentence",
            "load_seconds": load_seconds,
        }
    )

    while not stopping.is_set():
        message = commands.get()
        if message is None:
            break

        try:
            request_id = int(message["id"])
            text = str(message["text"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            _emit({"type": "error", "id": message.get("id"), "message": str(exc)})
            continue

        if not text:
            _emit({"type": "done", "id": request_id, "cancelled": False})
            continue

        with cancel_lock:
            was_cancelled = request_id in cancelled
        if was_cancelled:
            _emit({"type": "done", "id": request_id, "cancelled": True})
            with cancel_lock:
                cancelled.discard(request_id)
            continue

        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                wav = tts.infer(
                    text,
                    speaker=SPEAKER,
                    emotion=EMOTION,
                    temperature=1.0,
                    top_k=50,
                )

            with cancel_lock:
                cancelled_after_generation = request_id in cancelled
            if cancelled_after_generation:
                _emit(
                    {
                        "type": "done",
                        "id": request_id,
                        "cancelled": True,
                        "generation_seconds": time.perf_counter() - started,
                        "chunks": 0,
                    }
                )
                continue

            samples = np.asarray(wav, dtype=np.float32).reshape(-1)
            elapsed = time.perf_counter() - started
            if samples.size:
                _emit(
                    {
                        "type": "audio",
                        "id": request_id,
                        "sample_rate": SAMPLE_RATE,
                        "pcm_f32le": base64.b64encode(samples.tobytes()).decode("ascii"),
                    }
                )
            _emit(
                {
                    "type": "done",
                    "id": request_id,
                    "cancelled": False,
                    "first_audio_seconds": elapsed if samples.size else None,
                    "generation_seconds": elapsed,
                    "chunks": 1 if samples.size else 0,
                }
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _emit({"type": "error", "id": request_id, "message": str(exc)})
        finally:
            with cancel_lock:
                cancelled.discard(request_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
