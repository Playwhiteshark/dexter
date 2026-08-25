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
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

# The Host/runtime helpers launch this worker by file path. In that execution
# mode Python puts Runtime/Voice (not the repository root) on sys.path, which
# would make production package imports such as Runtime.Voice.* fail. Bootstrap
# the repo root once so direct-script and package execution behave identically.
REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root_text = str(REPO_ROOT)
if _repo_root_text not in sys.path:
    sys.path.insert(0, _repo_root_text)

MODEL_REPOS = {
    "q8": "neuphonic/neutts-2e-q8-gguf",
    "q4": "neuphonic/neutts-2e-q4-gguf",
}
VARIANT = os.environ.get("DEXTER_TTS_VARIANT", "q4").strip().lower()
if VARIANT not in MODEL_REPOS:
    raise SystemExit(
        f"Unsupported DEXTER_TTS_VARIANT={VARIANT!r}; expected one of {sorted(MODEL_REPOS)}"
    )
MODEL_REPO = MODEL_REPOS[VARIANT]
CODEC_REPO = "neuphonic/neucodec-onnx-decoder-int8"
SPEAKER = "paul"
EMOTION = "neutral"
SAMPLE_RATE = 24_000

BACKBONE_DEVICE = os.environ.get("DEXTER_TTS_BACKBONE_DEVICE", "cpu").strip().lower()
if BACKBONE_DEVICE not in {"cpu", "gpu"}:
    raise SystemExit(
        "DEXTER_TTS_BACKBONE_DEVICE must be 'cpu' or 'gpu', "
        f"got {BACKBONE_DEVICE!r}"
    )

VULKAN_DEVICE = os.environ.get("DEXTER_TTS_VULKAN_DEVICE", "").strip()
if VULKAN_DEVICE:
    # llama.cpp reads this before backend initialization. Keeping selection in
    # the worker environment prevents TTS from silently choosing the RTX that
    # Ollama already owns.
    os.environ["GGML_VK_VISIBLE_DEVICES"] = VULKAN_DEVICE

PIPELINE_WATERMARK = (
    os.environ.get("DEXTER_TTS_PIPELINE_WATERMARK", "0").strip() == "1"
)

# NeuCodec's upstream ONNX session leaves ORT's CPU pool at its machine-sized
# default. On Dexter's 16-core hybrid CPU that pool heavily contends with the
# Arc/Vulkan TTS backbone while Gemma is generating. Live A/B measurements on
# the target machine improved fully-concurrent TTS from ~1.26 RTF to ~0.8 RTF
# by limiting ORT to two intra-op threads and one inter-op thread. ORT 1.29+
# reads these process defaults when the session is created. Explicit user
# settings still win because setdefault preserves inherited environment values.
CODEC_THREADS_RAW = os.environ.get("DEXTER_TTS_CODEC_THREADS", "2").strip()
try:
    CODEC_THREADS = int(CODEC_THREADS_RAW)
except ValueError as exc:
    raise SystemExit("DEXTER_TTS_CODEC_THREADS must be an integer") from exc
if CODEC_THREADS < 1:
    raise SystemExit("DEXTER_TTS_CODEC_THREADS must be positive")
os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", str(CODEC_THREADS))
os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", "1")

# Once synthesis was comfortably faster than realtime, the remaining audible
# glitch was the 25 -> 50 frame startup transition: only 0.5 s of initial PCM
# could be exhausted before the first 1.0 s steady chunk arrived. A 40-frame
# first window gives 0.8 s of initial audio and eliminated the measured startup
# underflows while adding only ~0.18-0.20 s to first-audio latency. Keep the
# steady window at 50 frames for throughput. Environment overrides remain
# available for diagnostics.
os.environ.setdefault("DEXTER_TTS_FIRST_STREAM_FRAMES", "40")
os.environ.setdefault("DEXTER_TTS_STEADY_STREAM_FRAMES", "50")

THREAD_OVERRIDE_RAW = os.environ.get("DEXTER_TTS_THREADS", "").strip()
THREAD_OVERRIDE: int | None = None
if THREAD_OVERRIDE_RAW:
    try:
        THREAD_OVERRIDE = int(THREAD_OVERRIDE_RAW)
    except ValueError as exc:
        raise SystemExit("DEXTER_TTS_THREADS must be an integer") from exc
    if THREAD_OVERRIDE < 1:
        raise SystemExit("DEXTER_TTS_THREADS must be positive")

PROFILE_STAGES = os.environ.get("DEXTER_TTS_PROFILE_STAGES", "0").strip() == "1"
PROFILE_LLAMA_STAGES = (
    os.environ.get("DEXTER_TTS_PROFILE_LLAMA_STAGES", "0").strip() == "1"
)

_PROTOCOL_OUT = sys.stdout


def _apply_windows_priority() -> None:
    """Protect latency-sensitive synthesis from concurrent LLM CPU work.

    Live profiling on Dexter's target Windows machine showed that the default
    process priority lets Ollama/llama-server starve the NeuTTS worker badly
    enough to push synthesis from faster-than-realtime to ~1.7x realtime.
    AboveNormal restored concurrent synthesis to roughly realtime without using
    HIGH/REALTIME priority classes that could compromise system responsiveness.
    """
    if os.name != "nt":
        return
    try:
        import ctypes

        above_normal_priority_class = 0x00008000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        process = kernel32.GetCurrentProcess()
        if not kernel32.SetPriorityClass(process, above_normal_priority_class):
            error = ctypes.get_last_error()
            print(
                f"[TTS] Warning: could not set AboveNormal process priority (error={error})",
                file=sys.stderr,
            )
    except Exception as exc:
        print(
            f"[TTS] Warning: could not set AboveNormal process priority: {exc}",
            file=sys.stderr,
        )


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
    _apply_windows_priority()

    import numpy as np
    import llama_cpp
    import neutts.neutts as neutts_core

    if PIPELINE_WATERMARK:
        from Runtime.Voice.neutts2e_pipelined_stream import install

        install()

    from neutts import NeuTTS2E

    if THREAD_OVERRIDE is not None:
        # NeuTTS 1.4.1 defaults Windows/Linux generation to half os.cpu_count().
        # This override remains diagnostic-only; the measured production default
        # of eight generation threads is faster than forcing all 16 CPU cores.
        neutts_core._n_perf_cores = lambda: THREAD_OVERRIDE

    try:
        gpu_offload_supported = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        gpu_offload_supported = False

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

    device_detail = BACKBONE_DEVICE
    if BACKBONE_DEVICE == "gpu" and VULKAN_DEVICE:
        device_detail += f" (Vulkan visible device {VULKAN_DEVICE})"
    if PIPELINE_WATERMARK:
        device_detail += ", pipelined Perth"
    if THREAD_OVERRIDE is not None:
        device_detail += f", threads={THREAD_OVERRIDE}"

    try:
        print(
            f"[TTS] Loading {MODEL_REPO} / {CODEC_REPO}; backbone={device_detail}, codec=cpu...",
            file=sys.stderr,
        )
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            tts = NeuTTS2E(
                backbone_repo=MODEL_REPO,
                backbone_device=BACKBONE_DEVICE,
                codec_repo=CODEC_REPO,
                codec_device="cpu",
                seed=42,
            )
            tts.warmup()
        load_seconds = time.perf_counter() - started
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit({"type": "fatal", "message": str(exc)})
        return 1

    requested_gpu_layers: int | None = None
    actual_threads: int | None = None
    try:
        requested_gpu_layers = int(tts.backbone.model_params.n_gpu_layers)
    except Exception:
        pass
    try:
        actual_threads = int(tts.backbone.n_threads)
    except Exception:
        pass

    decode_profile = {"seconds": 0.0, "calls": 0}
    llama_profile = {
        "eval_seconds": 0.0,
        "eval_calls": 0,
        "sample_seconds": 0.0,
        "sample_calls": 0,
        "tokenize_seconds": 0.0,
        "tokenize_calls": 0,
        "detokenize_seconds": 0.0,
        "detokenize_calls": 0,
    }

    if PROFILE_STAGES:
        original_decode = tts._decode

        def profiled_decode(*args: Any, **kwargs: Any):
            decode_started = time.perf_counter()
            try:
                return original_decode(*args, **kwargs)
            finally:
                decode_profile["seconds"] += time.perf_counter() - decode_started
                decode_profile["calls"] += 1

        tts._decode = profiled_decode

    if PROFILE_LLAMA_STAGES:
        for method_name in ("eval", "sample", "tokenize", "detokenize"):
            original = getattr(tts.backbone, method_name)
            seconds_key = f"{method_name}_seconds"
            calls_key = f"{method_name}_calls"

            def make_profiled(method, seconds_name: str, calls_name: str):
                def profiled(*args: Any, **kwargs: Any):
                    call_started = time.perf_counter()
                    try:
                        return method(*args, **kwargs)
                    finally:
                        llama_profile[seconds_name] += time.perf_counter() - call_started
                        llama_profile[calls_name] += 1

                return profiled

            setattr(
                tts.backbone,
                method_name,
                make_profiled(original, seconds_key, calls_key),
            )

    _emit(
        {
            "type": "ready",
            "speaker": SPEAKER,
            "emotion": EMOTION,
            "sample_rate": SAMPLE_RATE,
            "model": MODEL_REPO,
            "variant": VARIANT,
            "codec": CODEC_REPO,
            "mode": "stream",
            "load_seconds": load_seconds,
            "backbone_device": BACKBONE_DEVICE,
            "vulkan_device": VULKAN_DEVICE or None,
            "pipeline_watermark": PIPELINE_WATERMARK,
            "actual_threads": actual_threads,
            "codec_ort_intra_threads": os.environ.get("ORT_INTRA_OP_NUM_THREADS"),
            "codec_ort_inter_threads": os.environ.get("ORT_INTER_OP_NUM_THREADS"),
            "first_stream_frames": os.environ.get("DEXTER_TTS_FIRST_STREAM_FRAMES"),
            "steady_stream_frames": os.environ.get("DEXTER_TTS_STEADY_STREAM_FRAMES"),
            "gpu_offload_supported": gpu_offload_supported,
            "requested_gpu_layers": requested_gpu_layers,
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

        decode_profile["seconds"] = 0.0
        decode_profile["calls"] = 0
        for key in llama_profile:
            llama_profile[key] = 0.0 if key.endswith("_seconds") else 0

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
                try:
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
                                "pcm_f32le": base64.b64encode(samples.tobytes()).decode("ascii"),
                            }
                        )
                finally:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _emit({"type": "error", "id": request_id, "message": str(exc)})
        else:
            payload: dict[str, Any] = {
                "type": "done",
                "id": request_id,
                "cancelled": cancelled_during_run,
                "first_audio_seconds": first_audio_seconds,
                "generation_seconds": time.perf_counter() - started,
                "chunks": chunk_count,
                "decode_seconds": decode_profile["seconds"] if PROFILE_STAGES else None,
                "decode_calls": decode_profile["calls"] if PROFILE_STAGES else None,
            }
            if PROFILE_LLAMA_STAGES:
                payload.update(llama_profile)
            _emit(payload)
        finally:
            with cancel_lock:
                cancelled.discard(request_id)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
