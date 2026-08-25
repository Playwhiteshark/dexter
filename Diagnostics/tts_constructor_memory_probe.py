"""Second-stage memory probe for Dexter's NeuTTS constructor.

This complements tts_memory_probe.py after the broad probe showed that most of
Dexter's TTS memory jump occurs inside NeuTTS2E(...) construction. It instruments
NeuTTS's own loader boundaries so one run attributes the constructor to:

    Q4 llama.cpp backbone -> ONNX codec -> Perth watermarker -> constructor tail

Then it measures warmup and one normal Paul synthesis. Run with Dexter fully
quit so the production worker is not resident at the same time.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = REPO_ROOT / "Runtime" / "Voice"

# Mirror the production worker's module resolution. Runtime/Voice must be first
# so NeuTTS/Perth see Dexter's small SciPy-free librosa compatibility module.
for value in (str(REPO_ROOT), str(VOICE_DIR)):
    if value in sys.path:
        sys.path.remove(value)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VOICE_DIR))

# Mirror the production machine-local model/auth cache. This is intentionally
# forced rather than setdefault: NeuTTS2EBackend.runtime_env() also overrides
# these locations for the child worker.
runtime_root = Path(
    os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(REPO_ROOT)
) / "Dexter" / "Voice" / "neutts2e"
cache = runtime_root / "cache"
hf_home = cache / "huggingface"
torch_home = cache / "torch"
xdg_home = cache / "xdg"
for path in (hf_home, torch_home, xdg_home):
    path.mkdir(parents=True, exist_ok=True)

os.environ.update(
    {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "TORCH_HOME": str(torch_home),
        "XDG_CACHE_HOME": str(xdg_home),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "GGML_VK_VISIBLE_DEVICES": "0",
        "ORT_INTRA_OP_NUM_THREADS": "2",
        "ORT_INTER_OP_NUM_THREADS": "1",
        "DEXTER_TTS_FIRST_STREAM_FRAMES": "40",
        "DEXTER_TTS_STEADY_STREAM_FRAMES": "50",
    }
)

# Reuse the already-tested Windows process-memory accounting and table format.
from Diagnostics import tts_memory_probe as mem

MODEL_REPO = "neuphonic/neutts-2e-q4-gguf"
CODEC_REPO = "neuphonic/neucodec-onnx-decoder-int8"


def _consume_stream(tts: Any) -> int:
    chunks = 0
    for wav in tts.infer_stream(
        "Dexter constructor memory probe. The sky is blue and the system is running normally.",
        speaker="paul",
        emotion="neutral",
        temperature=1.0,
        top_k=50,
    ):
        if wav is not None:
            chunks += 1
    return chunks


def main() -> int:
    if os.name != "nt":
        raise SystemExit("This probe targets Dexter's Windows runtime")

    print("Dexter TTS constructor memory probe")
    print(f"PID: {os.getpid()}  Python: {sys.version.split()[0]}")
    print("Quit Dexter before running this probe.\n")

    mem.record("python baseline")
    mem.import_stage("numpy", "import numpy")
    mem.import_stage("llama_cpp", "import llama_cpp")
    mem.import_stage("onnxruntime", "import onnxruntime")
    mem.import_stage("torch", "import torch")
    mem.import_stage("torchaudio", "import torchaudio")
    mem.import_stage("transformers", "import transformers")

    librosa_module = mem.import_stage("librosa", "import librosa shim")
    librosa_path = Path(getattr(librosa_module, "__file__", "")).resolve()
    expected_librosa = (VOICE_DIR / "librosa.py").resolve()
    if librosa_path != expected_librosa:
        raise RuntimeError(
            f"Probe imported unexpected librosa at {librosa_path}; expected {expected_librosa}"
        )

    mem.import_stage("perth", "import perth")
    mem.import_stage("neucodec", "import neucodec")
    neutts_core = mem.import_stage("neutts.neutts", "import neutts core")

    def install_pipeline() -> None:
        from Runtime.Voice.neutts2e_pipelined_stream import install

        install()

    mem.run_stage("install Dexter pipeline", install_pipeline)

    original_backbone = neutts_core.NeuTTS._load_backbone
    original_codec = neutts_core.NeuTTS._load_codec
    original_watermarker = neutts_core._load_watermarker

    def measured_backbone(self, *args, **kwargs):
        started = time.perf_counter()
        result = original_backbone(self, *args, **kwargs)
        model_path = getattr(getattr(self, "backbone", None), "model_path", None)
        note = f"model_path={model_path}" if model_path else ""
        mem.record("constructor: Q4 backbone", time.perf_counter() - started, note=note)
        return result

    def measured_codec(self, *args, **kwargs):
        started = time.perf_counter()
        result = original_codec(self, *args, **kwargs)
        mem.record("constructor: ONNX codec", time.perf_counter() - started)
        return result

    def measured_watermarker():
        started = time.perf_counter()
        result = original_watermarker()
        mem.record(
            "constructor: Perth model",
            time.perf_counter() - started,
            note=f"loaded={result is not None}",
        )
        return result

    neutts_core.NeuTTS._load_backbone = measured_backbone
    neutts_core.NeuTTS._load_codec = measured_codec
    neutts_core._load_watermarker = measured_watermarker

    try:
        from neutts import NeuTTS2E

        started = time.perf_counter()
        tts = NeuTTS2E(
            backbone_repo=MODEL_REPO,
            backbone_device="gpu",
            codec_repo=CODEC_REPO,
            codec_device="cpu",
            seed=42,
        )
        mem.record("constructor: tail", time.perf_counter() - started)
    finally:
        neutts_core.NeuTTS._load_backbone = original_backbone
        neutts_core.NeuTTS._load_codec = original_codec
        neutts_core._load_watermarker = original_watermarker

    mem.run_stage("NeuTTS warmup", tts.warmup)
    chunks = mem.run_stage("one synthesis", lambda: _consume_stream(tts))
    mem.rows[-1]["synthesis_chunks"] = chunks

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = REPO_ROOT / "Diagnostics" / f"tts-constructor-memory-probe_{stamp}.json"
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "pid": os.getpid(),
        "model_repo": MODEL_REPO,
        "codec_repo": CODEC_REPO,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "HF_HOME",
                "HF_HUB_CACHE",
                "GGML_VK_VISIBLE_DEVICES",
                "ORT_INTRA_OP_NUM_THREADS",
                "ORT_INTER_OP_NUM_THREADS",
                "DEXTER_TTS_FIRST_STREAM_FRAMES",
                "DEXTER_TTS_STEADY_STREAM_FRAMES",
            )
        },
        "rows": mem.rows,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved: {output_path}")
    if mem.rows:
        first = mem.rows[0]
        last = mem.rows[-1]
        print(
            "Total working-set growth: "
            f"{float(last['working_set_mb']) - float(first['working_set_mb']):.1f} MB"
        )
        print(
            "Total private-memory growth: "
            f"{float(last['private_mb']) - float(first['private_mb']):.1f} MB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
