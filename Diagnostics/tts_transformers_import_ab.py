"""Fresh-process stock-vs-lean Transformers A/B for Dexter TTS.

Both variants mirror current production: Q4 NeuTTS-2E on Intel Arc/Vulkan0,
llama.cpp 128/128, lean ONNX-only NeuCodec, ORT 2/1, Perth watermarking and
40/50 streaming windows. The only difference is whether NeuTTS's unconditional
``from transformers import AutoTokenizer, AutoModelForCausalLM`` loads the full
Transformers package or Dexter's guarded GGUF-only compatibility surface.

Run with Dexter fully quit:

    uv run python Diagnostics/tts_transformers_import_ab.py
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = REPO_ROOT / "Runtime" / "Voice"
MODEL_REPO = "neuphonic/neutts-2e-q4-gguf"
CODEC_REPO = "neuphonic/neucodec-onnx-decoder-int8"
SAMPLE_RATE = 24_000
TEST_TEXT = (
    "Dexter is checking whether an unused Transformers import can be removed "
    "from the quantized speech path without changing synthesis."
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


def _mb(value: int | float) -> float:
    return round(float(value) / (1024.0 * 1024.0), 1)


def _memory() -> dict[str, float]:
    if os.name != "nt":
        raise RuntimeError("This probe currently targets Dexter's Windows runtime")

    from ctypes import wintypes

    class PMCEX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PMCEX),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PMCEX()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return {
        "working_set_mb": _mb(counters.WorkingSetSize),
        "private_mb": _mb(counters.PrivateUsage),
    }


def _delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return round(after[key] - before[key], 1)


def _install_batch128() -> None:
    import llama_cpp

    original_init = llama_cpp.Llama.__init__

    def tuned_init(self, *args: Any, **kwargs: Any):
        kwargs["n_batch"] = 128
        kwargs["n_ubatch"] = 128
        return original_init(self, *args, **kwargs)

    llama_cpp.Llama.__init__ = tuned_init


def _synthesize(tts) -> dict[str, Any]:
    import numpy as np

    started = time.perf_counter()
    first_audio: float | None = None
    samples = 0
    chunks = 0
    digest = hashlib.sha256()
    for wav in tts.infer_stream(
        TEST_TEXT,
        speaker="paul",
        emotion="neutral",
        temperature=1.0,
        top_k=50,
    ):
        array = np.asarray(wav, dtype=np.float32).reshape(-1)
        if array.size == 0:
            continue
        if first_audio is None:
            first_audio = time.perf_counter() - started
        digest.update(array.tobytes())
        samples += int(array.size)
        chunks += 1
    elapsed = time.perf_counter() - started
    audio_seconds = samples / SAMPLE_RATE
    return {
        "rtf": round(elapsed / audio_seconds, 4) if audio_seconds else 0.0,
        "ttfa": round(first_audio or 0.0, 4),
        "generation_seconds": round(elapsed, 4),
        "audio_seconds": round(audio_seconds, 4),
        "chunks": chunks,
        "sha256": digest.hexdigest(),
    }


def _child(mode: str) -> int:
    _configure_environment()
    baseline = _memory()

    # Current production already uses the ONNX-only NeuCodec compatibility path.
    from Runtime.Voice.neucodec_onnx_compat import install_as_neucodec

    install_as_neucodec()
    if mode == "lean":
        from Runtime.Voice.transformers_quantized_compat import install_as_transformers

        install_as_transformers()

    _install_batch128()
    before_neutts = _memory()

    from Runtime.Voice.neutts2e_pipelined_stream import install

    install()
    from neutts import NeuTTS2E

    after_neutts = _memory()
    import_root = sys.modules.get("transformers")
    transformers_path = getattr(import_root, "__file__", None)

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
    after_load = _memory()

    take = _synthesize(tts)
    time.sleep(0.15)
    final = _memory()

    result = {
        "mode": mode,
        "transformers_path": transformers_path,
        "baseline": baseline,
        "before_neutts": before_neutts,
        "after_neutts": after_neutts,
        "after_load": after_load,
        "final": final,
        "neutts_import_ws_mb": _delta(after_neutts, before_neutts, "working_set_mb"),
        "neutts_import_private_mb": _delta(after_neutts, before_neutts, "private_mb"),
        "total_ws_mb": _delta(final, baseline, "working_set_mb"),
        "total_private_mb": _delta(final, baseline, "private_mb"),
        "load_seconds": round(load_seconds, 3),
        "take": take,
    }
    print("DEXTER_TRANSFORMERS_AB_RESULT=" + json.dumps(result), flush=True)
    return 0


def _run(mode: str) -> dict[str, Any]:
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
    marker = "DEXTER_TRANSFORMERS_AB_RESULT="
    line = next((line for line in completed.stdout.splitlines() if line.startswith(marker)), None)
    if line is None:
        print(completed.stdout, end="")
        raise SystemExit(f"{mode} returned no result marker")
    return json.loads(line[len(marker):])


def _parent() -> int:
    print("Dexter stock-vs-lean Transformers import A/B")
    print(
        "Quit Dexter before running this probe. Both variants use the current lean "
        "NeuCodec path and production 128/128 backbone.\n"
    )
    results = []
    for mode in ("stock", "lean"):
        print(f"Running {mode} ...", flush=True)
        result = _run(mode)
        results.append(result)
        take = result["take"]
        print(
            f"  neuttsImportWS={result['neutts_import_ws_mb']:+7.1f} MB  "
            f"totalWS={result['total_ws_mb']:+7.1f} MB  "
            f"totalPriv={result['total_private_mb']:+7.1f} MB  "
            f"RTF={take['rtf']:.3f} TTFA={take['ttfa']:.3f}s"
        )
        print(f"  transformers={result['transformers_path']}")

    stock, lean = results
    print("\nLean relative to stock:")
    print(f"  working set saved: {stock['total_ws_mb'] - lean['total_ws_mb']:.1f} MB")
    print(f"  private saved:     {stock['total_private_mb'] - lean['total_private_mb']:.1f} MB")
    print(f"  dRTF:              {lean['take']['rtf'] - stock['take']['rtf']:+.3f}")
    print(f"  dTTFA:             {lean['take']['ttfa'] - stock['take']['ttfa']:+.3f}s")
    print(f"  exact PCM match:   {lean['take']['sha256'] == stock['take']['sha256']}")
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
