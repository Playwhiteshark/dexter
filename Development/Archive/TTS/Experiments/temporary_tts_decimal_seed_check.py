"""Temporary listening check for NeuTTS-2E decimal pronunciation.

This intentionally bypasses Dexter text normalization, sentence buffering, and
live playback. It warms the exact production Paul/Q8/Arc pipeline once, then
renders the same already-verbalized decimal corpus with several sampling seeds.
Delete this diagnostic after the production seed policy is decided.
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Runtime.Voice.neutts2e_runtime import RUNTIME_ROOT, runtime_env

SAMPLE_RATE = 24_000
SEEDS = (42, 1234, 4321, 2026)
TEXT = (
    "Zero point two. Ten point zero. Forty two point zero. "
    "Eighty seven point zero. Ten point one."
)


def _apply_production_environment() -> None:
    env = runtime_env("q8")
    for key in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_DISABLE_SYMLINKS_WARNING",
        "TOKENIZERS_PARALLELISM",
        "DEXTER_TTS_VARIANT",
        "DEXTER_TTS_BACKBONE_DEVICE",
        "DEXTER_TTS_VULKAN_DEVICE",
        "DEXTER_TTS_PIPELINE_WATERMARK",
        "GGML_VK_VISIBLE_DEVICES",
    ):
        if key in env:
            os.environ[key] = env[key]


def _write_wav(path: Path, samples: np.ndarray) -> None:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm16.tobytes())


def main() -> int:
    _apply_production_environment()

    # Import native/model modules only after the production Vulkan/cache
    # environment is installed.
    from Runtime.Voice.neutts2e_pipelined_stream import install

    install()
    from neutts import NeuTTS2E

    print("Loading the production Paul/Q8/Arc voice once...")
    tts = NeuTTS2E(
        backbone_repo="neuphonic/neutts-2e-q8-gguf",
        backbone_device="gpu",
        codec_repo="neuphonic/neucodec-onnx-decoder-int8",
        codec_device="cpu",
        seed=42,
    )
    tts.warmup()

    out_dir = RUNTIME_ROOT / "diagnostics" / "decimal-seeds"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Text: {TEXT}")
    print(f"Output: {out_dir}")

    for seed in SEEDS:
        # NeuTTS calls _call_seed() for each streaming request. Changing only
        # this field isolates sampling seed without reloading the model.
        tts._seed = seed
        chunks = [
            np.asarray(chunk, dtype=np.float32).reshape(-1)
            for chunk in tts.infer_stream(
                TEXT,
                speaker="paul",
                emotion="neutral",
                temperature=1.0,
                top_k=50,
            )
        ]
        if not chunks:
            raise RuntimeError(f"seed {seed} produced no audio")
        samples = np.concatenate(chunks)
        path = out_dir / f"paul-decimals-seed-{seed}.wav"
        _write_wav(path, samples)
        print(f"seed {seed}: {path.name}")

    manifest = out_dir / "manifest.txt"
    manifest.write_text(
        "NeuTTS-2E decimal seed check\n"
        f"Text: {TEXT}\n"
        f"Seeds: {', '.join(str(seed) for seed in SEEDS)}\n",
        encoding="utf-8",
    )
    print("Done. Compare the four WAVs and note which seeds pronounce every decimal correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
