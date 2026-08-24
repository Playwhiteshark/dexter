"""Persistent NeuTTS-2E / Paul worker for Dexter.

The Host and this worker share Dexter's uv-managed Python 3.13 environment.
The process boundary exists for lifecycle, cancellation, and native-runtime
isolation; it is not a second Python environment.

stdout is reserved for a tiny JSON-lines audio protocol. Model/library logging
is redirected to stderr.
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

MODEL_REPO = "neuphonic/neutts-2e-q8-gguf"
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
            # NeuTTS2E.warmup() uses infer_stream() for quantized backbones.
            tts.warmup()
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
            "mode": "stream",
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
            if request_id in cancelled:
                cancelled.discard(request_id)
                _emit({"type": "done", "id": request_id, "cancelled": True})
                continue

        started = time.perf_counter()
        first_audio_seconds: float | None = None
        chunk_count = 0
        cancelled_during_run = False

        try:
            with contextlib.redirect_stdout(sys.stderr):
                stream = tts.infer_stream(
                    text,
                    speaker=SPEAKER,
                    emotion=EMOTION,
                    temperature=1.0,
                    top_k=50,
                )
                for wav in stream:
                    with cancel_lock:
                        if request_id in cancelled:
                            cancelled_during_run = True
                            break

                    samples = np.asarray(wav, dtype=np.float32).reshape(-1)
                    if samples.size == 0:
                        continue
                    if first_audio_seconds is None:
                        first_audio_seconds = time.perf_counter() - started
                    chunk_count += 1
                    _emit(
                        {
                            "type": "audio",
                            "id": request_id,
                            "sample_rate": SAMPLE_RATE,
                            "pcm_f32le": base64.b64encode(
                                samples.tobytes()
                            ).decode("ascii"),
                        }
                    )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _emit({"type": "error", "id": request_id, "message": str(exc)})
        else:
            _emit(
                {
                    "type": "done",
                    "id": request_id,
                    "cancelled": cancelled_during_run,
                    "first_audio_seconds": first_audio_seconds,
                    "generation_seconds": time.perf_counter() - started,
                    "chunks": chunk_count,
                }
            )
        finally:
            with cancel_lock:
                cancelled.discard(request_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
