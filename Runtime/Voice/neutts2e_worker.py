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
import ctypes
import json
import os
import queue
import sys
import threading
import time
import traceback
from typing import Any

MODEL_REPOS = {
    "q8": "neuphonic/neutts-2e-q8-gguf",
    "q4": "neuphonic/neutts-2e-q4-gguf",
}
VARIANT = os.environ.get("DEXTER_TTS_VARIANT", "q8").strip().lower()
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
    # the worker environment prevents a Vulkan build from silently choosing the
    # RTX that Ollama already owns.
    os.environ["GGML_VK_VISIBLE_DEVICES"] = VULKAN_DEVICE

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


def _emit(payload: dict[str, Any]) -> None:
    _PROTOCOL_OUT.write(json.dumps(payload, separators=(",", ":")) + "\n")
    _PROTOCOL_OUT.flush()


def _decode_c_string(value: bytes | None) -> str:
    if not value:
        return ""
    return value.decode("utf-8", errors="replace")


def _enumerate_backend_devices() -> tuple[list[dict[str, Any]], str | None]:
    """Best-effort ggml device enumeration for newer llama.cpp bindings."""
    try:
        from llama_cpp import llama_cpp as llama_low

        lib = llama_low._lib
        dev_count = lib.ggml_backend_dev_count
        dev_count.argtypes = []
        dev_count.restype = ctypes.c_size_t

        dev_get = lib.ggml_backend_dev_get
        dev_get.argtypes = [ctypes.c_size_t]
        dev_get.restype = ctypes.c_void_p

        dev_name = lib.ggml_backend_dev_name
        dev_name.argtypes = [ctypes.c_void_p]
        dev_name.restype = ctypes.c_char_p

        dev_description = lib.ggml_backend_dev_description
        dev_description.argtypes = [ctypes.c_void_p]
        dev_description.restype = ctypes.c_char_p

        dev_type = lib.ggml_backend_dev_type
        dev_type.argtypes = [ctypes.c_void_p]
        dev_type.restype = ctypes.c_int

        dev_backend_reg = lib.ggml_backend_dev_backend_reg
        dev_backend_reg.argtypes = [ctypes.c_void_p]
        dev_backend_reg.restype = ctypes.c_void_p

        reg_name = lib.ggml_backend_reg_name
        reg_name.argtypes = [ctypes.c_void_p]
        reg_name.restype = ctypes.c_char_p

        type_names = {
            0: "cpu",
            1: "gpu",
            2: "igpu",
            3: "accelerator",
            4: "meta",
        }
        devices: list[dict[str, Any]] = []
        for index in range(int(dev_count())):
            device = dev_get(index)
            if not device:
                continue
            backend = dev_backend_reg(device)
            raw_type = int(dev_type(device))
            devices.append(
                {
                    "index": index,
                    "name": _decode_c_string(dev_name(device)),
                    "description": _decode_c_string(dev_description(device)),
                    "type": type_names.get(raw_type, f"unknown-{raw_type}"),
                    "backend": _decode_c_string(reg_name(backend)) if backend else "",
                }
            )
        return devices, None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


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
    import llama_cpp
    import neutts.neutts as neutts_core
    from neutts import NeuTTS2E

    if THREAD_OVERRIDE is not None:
        # NeuTTS 1.4.1 defaults Windows/Linux generation to half os.cpu_count().
        # This opt-in override exists only for controlled diagnostics.
        neutts_core._n_perf_cores = lambda: THREAD_OVERRIDE

    try:
        gpu_offload_supported = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        gpu_offload_supported = False
    try:
        raw_system_info = llama_cpp.llama_print_system_info()
        if isinstance(raw_system_info, bytes):
            llama_system_info = raw_system_info.decode("utf-8", errors="replace")
        else:
            llama_system_info = str(raw_system_info)
    except Exception as exc:
        llama_system_info = f"unavailable: {type(exc).__name__}: {exc}"

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
            # NeuTTS2E.warmup() uses infer_stream() for quantized backbones.
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

    backend_devices, backend_device_error = _enumerate_backend_devices()

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
        # llama-cpp-python's high-level streaming completion calls these instance
        # methods in its hot loop. Timing them lets us distinguish native model
        # eval from sampling and Python token/string handling without changing
        # the generated token sequence.
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
            "threads": THREAD_OVERRIDE,
            "actual_threads": actual_threads,
            "gpu_offload_supported": gpu_offload_supported,
            "requested_gpu_layers": requested_gpu_layers,
            "llama_system_info": llama_system_info,
            "backend_devices": backend_devices,
            "backend_device_error": backend_device_error,
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
