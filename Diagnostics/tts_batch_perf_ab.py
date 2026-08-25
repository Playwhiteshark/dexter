"""Isolated NeuTTS batch/ubatch performance A/B for Dexter.

Each configuration runs in a fresh Python process and mirrors the validated
production voice path: Q4 GGUF on Intel Arc/Vulkan0, ONNX codec on CPU with
ORT 2/1 threads, Perth watermarking, and Dexter's 40/50 streaming pipeline.
Only llama.cpp n_batch/n_ubatch changes.

Run with Dexter fully quit:

    uv run python Diagnostics/tts_batch_perf_ab.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = REPO_ROOT / "Runtime" / "Voice"
MODEL_REPO = "neuphonic/neutts-2e-q4-gguf"
CODEC_REPO = "neuphonic/neucodec-onnx-decoder-int8"
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


def _child(label: str, batch: int, ubatch: int) -> int:
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

    first = _synthesize(tts)
    second = _synthesize(tts)
    result = {
        "label": label,
        "batch": batch,
        "ubatch": ubatch,
        "load_seconds": round(load_seconds, 3),
        "first": first,
        "second": second,
    }
    print("DEXTER_BATCH_PERF_RESULT=" + json.dumps(result), flush=True)
    return 0


def _parent() -> int:
    print("Dexter isolated TTS batch performance A/B")
    print("Quit Dexter before running this probe. Each row is a fresh Python process.\n")

    variants = [
        ("batch512", 512, 512),
        ("batch256", 256, 256),
        ("batch128", 128, 128),
    ]
    results = []
    marker = "DEXTER_BATCH_PERF_RESULT="

    for label, batch, ubatch in variants:
        print(f"Running {label} ...", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--label", label,
                "--batch", str(batch),
                "--ubatch", str(ubatch),
            ],
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
            check=False,
        )
        if completed.returncode != 0:
            print(completed.stdout, end="")
            raise SystemExit(f"{label} failed with exit code {completed.returncode}")
        line = next((line for line in completed.stdout.splitlines() if line.startswith(marker)), None)
        if line is None:
            print(completed.stdout, end="")
            raise SystemExit(f"{label} returned no result marker")
        result = json.loads(line[len(marker):])
        results.append(result)
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
        rtf_delta = result["second"]["rtf"] - base["second"]["rtf"]
        ttfa_delta = (
            result["second"]["first_audio_seconds"]
            - base["second"]["first_audio_seconds"]
        )
        print(
            f"  {result['label']:<10} second-run dRTF={rtf_delta:+.3f}  "
            f"dTTFA={ttfa_delta:+.3f}s"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    args = parser.parse_args()
    if args.child:
        return _child(args.label, args.batch, args.ubatch)
    return _parent()


if __name__ == "__main__":
    raise SystemExit(main())
