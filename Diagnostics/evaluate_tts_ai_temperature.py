r"""Test raw ``AI`` pronunciation across NeuTTS sampling temperatures.

This is diagnostic-only. It bypasses Dexter's text normalization and keeps the
production Arc + pipelined-Perth runtime, voice, seed, and hybrid stream windows.
Paul is loaded once, then the exact same three sentence contexts are rendered at
several temperatures while top_k remains at NeuTTS's production default of 50.

Run:
    uv run python Diagnostics\evaluate_tts_ai_temperature.py

Outputs:
    Runtime\Voice\debug\ai_temperature\
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Apply the exact production worker environment before importing NeuTTS/llama.cpp.
from Runtime.Voice.neutts2e_backend import NeuTTS2EBackend

_runtime_env = NeuTTS2EBackend.runtime_env()
for _name in (
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
    if _name in _runtime_env:
        os.environ[_name] = _runtime_env[_name]

import numpy as np
import neutts.neutts as neutts_core

from Runtime.Voice.neutts2e_pipelined_stream import install as install_pipeline

install_pipeline()
_ORIGINAL_PIPELINED_STREAM = neutts_core.NeuTTS._infer_stream_ggml
_ACTIVE_TEMPERATURE = 1.0


def _temperature_stream(
    self,
    ref_codes,
    ref_text: str,
    input_text: str,
    emotion: str | None = None,
    temperature: float = 1.0,
    top_k: int = 50,
):
    # Force only the diagnostic temperature. Everything else remains whatever
    # NeuTTS2E.infer_stream asks for, including top_k=50 and the fixed seed.
    return _ORIGINAL_PIPELINED_STREAM(
        self,
        ref_codes,
        ref_text,
        input_text,
        emotion,
        temperature=_ACTIVE_TEMPERATURE,
        top_k=top_k,
    )


neutts_core.NeuTTS._infer_stream_ggml = _temperature_stream

from neutts import NeuTTS2E

OUTPUT_DIR = REPO_ROOT / "Runtime" / "Voice" / "debug" / "ai_temperature"
TEMPERATURES = (1.0, 0.8, 0.6, 0.4)
CONTEXTS = {
    "initial": "AI is running locally.",
    "noun_modifier": "The AI system is running locally.",
    "conjunction": "The GPU is ready, and AI is ready too.",
}


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


def _render(tts: NeuTTS2E, text: str, temperature: float) -> tuple[np.ndarray, int]:
    global _ACTIVE_TEMPERATURE
    _ACTIVE_TEMPERATURE = temperature
    chunks: list[np.ndarray] = []
    for wav in tts.infer_stream(
        text,
        speaker="paul",
        emotion="neutral",
        temperature=temperature,
        top_k=50,
    ):
        chunk = np.asarray(wav, dtype=np.float32).reshape(-1)
        if chunk.size:
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError(f"No audio generated for {text!r} at temperature={temperature}")
    return np.concatenate(chunks), 24_000


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Raw-AI temperature matrix — live Dexter is unchanged.")
    print("Q8 / Arc Vulkan0 / pipelined Perth / Paul / seed 42 / top_k 50")
    print("Loading Paul once...\n")

    tts = NeuTTS2E(
        backbone_repo="neuphonic/neutts-2e-q8-gguf",
        backbone_device="gpu",
        codec_repo="neuphonic/neucodec-onnx-decoder-int8",
        codec_device="cpu",
        seed=42,
    )
    tts.warmup()

    manifest: list[str] = []
    index = 1
    for temperature in TEMPERATURES:
        for context_name, text in CONTEXTS.items():
            filename = f"{index:02d}_temp{temperature:.1f}_{context_name}.wav"
            path = OUTPUT_DIR / filename
            samples, sample_rate = _render(tts, text, temperature)
            _write_wav(path, samples, sample_rate)
            line = f"{filename}: temperature={temperature:.1f} | {text}"
            manifest.append(line)
            print(line)
            index += 1

    manifest_path = OUTPUT_DIR / "manifest.txt"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print("\nGoal: find ONE temperature where raw 'AI' is pronounced A-I in all 3 contexts.")
    print("Also note if lower temperatures noticeably flatten or degrade Paul's delivery.")
    print(f"\nOutput: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
