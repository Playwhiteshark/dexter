"""Isolated NeuTTS batch/ubatch performance A/B for Dexter.

Each configuration runs in a fresh Python process and mirrors the validated
production voice path: Q4 GGUF on Intel Arc/Vulkan0, ONNX codec on CPU with
ORT 2/1 threads, Perth watermarking, and Dexter's 40/50 streaming pipeline.
Only llama.cpp n_batch/n_ubatch changes.

Normal isolated test:

    uv run python Diagnostics/tts_batch_perf_ab.py

Final production-like concurrency check (Gemma generating via Ollama while TTS
runs):

    uv run python Diagnostics/tts_batch_perf_ab.py --concurrent
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = REPO_ROOT / "Runtime" / "Voice"
MODEL_REPO = "neuphonic/neutts-2e-q4-gguf"
CODEC_REPO = "neuphonic/neucodec-onnx-decoder-int8"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma4-jarvis:12b"
SAMPLE_RATE = 24_000
TEST_TEXT = (
    "The quick brown fox jumps over the lazy dog while Dexter explains the result "
    "clearly and naturally."
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

    repo = str(REPO_ROOT)
    voice = str(VOICE_DIR)
    if repo in sys.path:
        sys.path.remove(repo)
    sys.path.insert(0, repo)
    if voice in sys.path:
        sys.path.remove(voice)
    sys.path.insert(0, voice)


def _install_batch_override(batch: int, ubatch: int) -> None:
    """Change only llama.cpp constructor batch kwargs, preserving NeuTTS loader."""
    import llama_cpp

    original_init = llama_cpp.Llama.__init__

    def tuned_init(self, *args, **kwargs):
        kwargs["n_batch"] = batch
        kwargs["n_ubatch"] = ubatch
        return original_init(self, *args, **kwargs)

    llama_cpp.Llama.__init__ = tuned_init


def _synthesize(tts) -> dict[str, float | int]:
    started = time.perf_counter()
    first_audio_seconds: float | None = None
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
        if first_audio_seconds is None:
            first_audio_seconds = time.perf_counter() - started
        chunks += 1
        samples += int(len(wav))
    generation_seconds = time.perf_counter() - started
    audio_seconds = samples / SAMPLE_RATE
    return {
        "generation_seconds": round(generation_seconds, 4),
        "audio_seconds": round(audio_seconds, 4),
        "rtf": round(generation_seconds / audio_seconds, 4) if audio_seconds else 0.0,
        "first_audio_seconds": round(first_audio_seconds or 0.0, 4),
        "chunks": chunks,
    }


def _ollama_load(stop: threading.Event, active: threading.Event, stats: dict[str, object]) -> None:
    """Keep the production Gemma model actively generating during TTS."""
    prompt = (
        "Write a long, detailed technical essay explaining transformer inference, "
        "quantization, KV caches, mixture-of-experts routing, and GPU/CPU offload. "
        "Continue in depth until the requested token budget is exhausted."
    )
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "5m",
            "options": {"num_predict": 1024},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    lines = 0
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for raw in response:
                if raw.strip():
                    lines += 1
                    active.set()
                if stop.is_set():
                    break
        stats["lines"] = lines
        stats["seconds"] = round(time.perf_counter() - started, 3)
    except BaseException as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
        active.set()


def _concurrent_synthesize(tts) -> tuple[dict[str, float | int], dict[str, object]]:
    stop = threading.Event()
    active = threading.Event()
    stats: dict[str, object] = {}
    thread = threading.Thread(
        target=_ollama_load,
        args=(stop, active, stats),
        name="DexterBatchABGemmaLoad",
        daemon=True,
    )
    thread.start()
    if not active.wait(timeout=180):
        stop.set()
        raise RuntimeError("Gemma/Ollama did not begin streaming within 180 seconds")
    if "error" in stats:
        stop.set()
        raise RuntimeError(f"Gemma/Ollama load failed: {stats['error']}")

    # Give generation a short head start so TTS begins while Gemma is already
    # exercising its normal CPU/GPU offload path rather than during first-token setup.
    time.sleep(0.35)
    try:
        result = _synthesize(tts)
    finally:
        stop.set()
        thread.join(timeout=5.0)
    result["ollama_lines_during_window"] = int(stats.get("lines", 0))
    return result, stats


def _child(label: str, batch: int, ubatch: int, concurrent: bool) -> int:
    _configure_environment()
    _install_batch_override(batch, ubatch)

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

    # One no-load take stabilizes model/session caches before the measured run.
    warm_take = _synthesize(tts)
    if concurrent:
        measured, ollama_stats = _concurrent_synthesize(tts)
        result = {
            "label": label,
            "batch": batch,
            "ubatch": ubatch,
            "load_seconds": round(load_seconds, 3),
            "warm_take": warm_take,
            "concurrent": measured,
            "ollama": ollama_stats,
        }
    else:
        second = _synthesize(tts)
        result = {
            "label": label,
            "batch": batch,
            "ubatch": ubatch,
            "load_seconds": round(load_seconds, 3),
            "first": warm_take,
            "second": second,
        }
    print("DEXTER_BATCH_PERF_RESULT=" + json.dumps(result), flush=True)
    return 0


def _run_variant(label: str, batch: int, ubatch: int, concurrent: bool) -> dict[str, object]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--label", label,
        "--batch", str(batch),
        "--ubatch", str(ubatch),
    ]
    if concurrent:
        cmd.append("--concurrent")

    creationflags = subprocess.ABOVE_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
    completed = subprocess.run(
        cmd,
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
        raise SystemExit(f"{label} failed with exit code {completed.returncode}")
    marker = "DEXTER_BATCH_PERF_RESULT="
    line = next((line for line in completed.stdout.splitlines() if line.startswith(marker)), None)
    if line is None:
        print(completed.stdout, end="")
        raise SystemExit(f"{label} returned no result marker")
    return json.loads(line[len(marker):])


def _parent(concurrent: bool) -> int:
    if concurrent:
        print("Dexter concurrent Gemma + TTS batch A/B")
        print("Quit Dexter, but leave Ollama running. Comparing only 512 vs 128.\n")
        variants = [("batch512", 512, 512), ("batch128", 128, 128)]
    else:
        print("Dexter isolated TTS batch performance A/B")
        print("Quit Dexter before running this probe. Each row is a fresh Python process.\n")
        variants = [
            ("batch512", 512, 512),
            ("batch256", 256, 256),
            ("batch128", 128, 128),
        ]

    results = []
    for label, batch, ubatch in variants:
        print(f"Running {label} ...", flush=True)
        result = _run_variant(label, batch, ubatch, concurrent)
        results.append(result)
        if concurrent:
            take = result["concurrent"]
            print(
                f"  load={result['load_seconds']:>6.2f}s  "
                f"RTF={take['rtf']:.3f} TTFA={take['first_audio_seconds']:.3f}s  "
                f"Gemma stream lines={take['ollama_lines_during_window']}"
            )
        else:
            first = result["first"]
            second = result["second"]
            print(
                f"  load={result['load_seconds']:>6.2f}s  "
                f"first RTF={first['rtf']:.3f} TTFA={first['first_audio_seconds']:.3f}s  "
                f"second RTF={second['rtf']:.3f} TTFA={second['first_audio_seconds']:.3f}s"
            )

    base = results[0]
    print("\nRelative to 512/512 (lower is better):")
    for result in results[1:]:
        if concurrent:
            current = result["concurrent"]
            baseline = base["concurrent"]
        else:
            current = result["second"]
            baseline = base["second"]
        rtf_delta = current["rtf"] - baseline["rtf"]
        ttfa_delta = current["first_audio_seconds"] - baseline["first_audio_seconds"]
        print(
            f"  {result['label']:<10} dRTF={rtf_delta:+.3f}  dTTFA={ttfa_delta:+.3f}s"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--concurrent", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    args = parser.parse_args()
    if args.child:
        return _child(args.label, args.batch, args.ubatch, args.concurrent)
    return _parent(args.concurrent)


if __name__ == "__main__":
    raise SystemExit(main())
