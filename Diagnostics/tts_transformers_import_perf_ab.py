"""Repeated sustained-contention stock-vs-lean Transformers A/B for Dexter TTS.

Both variants mirror current production: Q4 NeuTTS-2E on Intel Arc/Vulkan0,
llama.cpp 128/128, lean ONNX-only NeuCodec, ORT 2/1, Perth watermarking and
40/50 streaming windows. The only difference is whether NeuTTS imports the full
Transformers package or Dexter's guarded quantized-only compatibility surface.

The parent runs three fresh-process trials per variant and alternates order.
Every measured take is rejected unless gemma4-jarvis:12b remains actively
generating for the full TTS timing window.

Run with Dexter fully quit, but leave Ollama running:

    uv run python Diagnostics/tts_transformers_import_perf_ab.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = REPO_ROOT / "Runtime" / "Voice"
MODEL_REPO = "neuphonic/neutts-2e-q4-gguf"
CODEC_REPO = "neuphonic/neucodec-onnx-decoder-int8"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma4-jarvis:12b"
SAMPLE_RATE = 24_000
TEST_TEXT = (
    "Dexter is measuring the complete speech pipeline while the language model "
    "is generating at the same time, so this is a realistic contention test."
)


def _configure_environment() -> None:
    local = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or REPO_ROOT)
    cache = local / "Dexter" / "Voice" / "neutts2e" / "cache"
    hf_home = cache / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("GGML_VK_VISIBLE_DEVICES", "0")
    os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "2")
    os.environ.setdefault("ORT_INTER_OP_NUM_THREADS", "1")
    os.environ.setdefault("DEXTER_TTS_FIRST_STREAM_FRAMES", "40")
    os.environ.setdefault("DEXTER_TTS_STEADY_STREAM_FRAMES", "50")

    for path in (str(REPO_ROOT), str(VOICE_DIR)):
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(VOICE_DIR))


def _install_batch128() -> None:
    import llama_cpp

    original_init = llama_cpp.Llama.__init__

    def tuned_init(self, *args: Any, **kwargs: Any):
        kwargs["n_batch"] = 128
        kwargs["n_ubatch"] = 128
        return original_init(self, *args, **kwargs)

    llama_cpp.Llama.__init__ = tuned_init


def _synthesize(tts) -> dict[str, float | int]:
    started = time.perf_counter()
    first_audio: float | None = None
    samples = 0
    chunks = 0
    for wav in tts.infer_stream(
        TEST_TEXT,
        speaker="paul",
        emotion="neutral",
        temperature=1.0,
        top_k=50,
    ):
        if wav is None:
            continue
        if first_audio is None:
            first_audio = time.perf_counter() - started
        chunks += 1
        samples += int(len(wav))
    generation_seconds = time.perf_counter() - started
    audio_seconds = samples / SAMPLE_RATE
    return {
        "generation_seconds": round(generation_seconds, 4),
        "audio_seconds": round(audio_seconds, 4),
        "rtf": round(generation_seconds / audio_seconds, 4) if audio_seconds else 0.0,
        "first_audio_seconds": round(first_audio or 0.0, 4),
        "chunks": chunks,
    }


def _ollama_request(payload: dict[str, object], timeout: float = 240.0) -> bytes:
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _preload_ollama() -> None:
    _ollama_request(
        {
            "model": OLLAMA_MODEL,
            "prompt": "Reply with ready.",
            "stream": False,
            "keep_alive": "5m",
            "options": {"num_predict": 4},
        }
    )


def _gemma_load(dispatched: threading.Event, finished: threading.Event, stats: dict[str, object]) -> None:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": (
            "Write a long detailed technical essay about transformer inference, quantization, "
            "KV caches, GPU offload, CPU scheduling, and memory bandwidth. Continue in depth."
        ),
        "stream": False,
        "keep_alive": "5m",
        "options": {"num_predict": 1024},
    }
    started = time.perf_counter()
    stats["request_started"] = started
    dispatched.set()
    try:
        raw = _ollama_request(payload)
        stats["request_completed"] = time.perf_counter()
        try:
            data = json.loads(raw)
            stats["eval_count"] = int(data.get("eval_count", 0) or 0)
        except Exception:
            stats["response_bytes"] = len(raw)
    except BaseException as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
        stats["request_completed"] = time.perf_counter()
    finally:
        finished.set()


def _concurrent_synthesize(tts) -> dict[str, float | int | bool]:
    _preload_ollama()
    dispatched = threading.Event()
    finished = threading.Event()
    stats: dict[str, object] = {}
    thread = threading.Thread(
        target=_gemma_load,
        args=(dispatched, finished, stats),
        daemon=True,
        name="DexterTransformersABGemmaLoad",
    )
    thread.start()
    if not dispatched.wait(timeout=10.0):
        raise RuntimeError("Gemma contention request was not dispatched")
    time.sleep(1.0)
    if finished.is_set():
        if "error" in stats:
            raise RuntimeError(f"Gemma load failed: {stats['error']}")
        raise RuntimeError("Gemma finished before TTS began; invalid contention run")

    result = _synthesize(tts)
    result["gemma_active_at_tts_end"] = not finished.is_set()
    if not result["gemma_active_at_tts_end"]:
        raise RuntimeError("Gemma finished before TTS completed; refusing invalid result")
    return result


def _child(mode: str) -> int:
    _configure_environment()

    # Current production path already uses the lean ONNX-only NeuCodec surface.
    from Runtime.Voice.neucodec_onnx_compat import install_as_neucodec

    install_as_neucodec()
    if mode == "lean":
        from Runtime.Voice.transformers_quantized_compat import install_as_transformers

        install_as_transformers()

    _install_batch128()

    from Runtime.Voice.neutts2e_pipelined_stream import install

    install()
    from neutts import NeuTTS2E

    load_started = time.perf_counter()
    tts = NeuTTS2E(
        backbone_repo=MODEL_REPO,
        backbone_device="gpu",
        codec_repo=CODEC_REPO,
        codec_device="cpu",
        seed=42,
    )
    tts.warmup()
    load_seconds = time.perf_counter() - load_started

    # Stabilize the Arc/ORT/Perth path before the measured concurrent take.
    warm = _synthesize(tts)
    measured = _concurrent_synthesize(tts)
    print(
        "DEXTER_TRANSFORMERS_PERF_RESULT="
        + json.dumps(
            {
                "mode": mode,
                "load_seconds": round(load_seconds, 3),
                "warm": warm,
                "concurrent": measured,
                "transformers_path": getattr(sys.modules.get("transformers"), "__file__", None),
            }
        ),
        flush=True,
    )
    return 0


def _run(mode: str) -> dict[str, object]:
    creationflags = subprocess.ABOVE_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", "--mode", mode],
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        check=False,
        creationflags=creationflags,
    )
    if completed.returncode != 0:
        print(completed.stdout, end="")
        raise SystemExit(f"{mode} failed with exit code {completed.returncode}")
    marker = "DEXTER_TRANSFORMERS_PERF_RESULT="
    line = next((ln for ln in completed.stdout.splitlines() if ln.startswith(marker)), None)
    if line is None:
        print(completed.stdout, end="")
        raise SystemExit(f"{mode} returned no result marker")
    return json.loads(line[len(marker):])


def _parent() -> int:
    print("Dexter repeated full-pipeline stock-vs-lean Transformers A/B")
    print(
        "Quit Dexter, leave Ollama running. Three fresh-process trials per variant; "
        "order alternates to reduce run-order bias.\n"
    )

    schedule = [
        (1, "stock"), (1, "lean"),
        (2, "lean"), (2, "stock"),
        (3, "stock"), (3, "lean"),
    ]
    results: dict[str, list[dict[str, object]]] = {"stock": [], "lean": []}

    for trial, mode in schedule:
        print(f"Trial {trial}: running {mode} ...", flush=True)
        result = _run(mode)
        results[mode].append(result)
        take = result["concurrent"]
        print(
            f"  load={result['load_seconds']:>6.2f}s  "
            f"RTF={take['rtf']:.3f} TTFA={take['first_audio_seconds']:.3f}s  "
            f"Gemma active through TTS={take['gemma_active_at_tts_end']}"
        )

    medians: dict[str, dict[str, float]] = {}
    print("\nMedians across three sustained-contention trials:")
    for mode in ("stock", "lean"):
        rtfs = [float(r["concurrent"]["rtf"]) for r in results[mode]]
        ttfas = [float(r["concurrent"]["first_audio_seconds"]) for r in results[mode]]
        medians[mode] = {
            "rtf": statistics.median(rtfs),
            "ttfa": statistics.median(ttfas),
        }
        print(
            f"  {mode:<5} RTF={medians[mode]['rtf']:.3f}  "
            f"TTFA={medians[mode]['ttfa']:.3f}s  "
            f"RTFs={[round(v, 3) for v in rtfs]}"
        )

    print("\nLean relative to stock median (lower is better):")
    print(f"  dRTF={medians['lean']['rtf'] - medians['stock']['rtf']:+.3f}")
    print(f"  dTTFA={medians['lean']['ttfa'] - medians['stock']['ttfa']:+.3f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=("stock", "lean"), default="stock")
    args = parser.parse_args()
    if args.child:
        return _child(args.mode)
    return _parent()


if __name__ == "__main__":
    raise SystemExit(main())
