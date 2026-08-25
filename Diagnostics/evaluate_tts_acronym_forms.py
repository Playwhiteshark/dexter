r"""Generate a controlled NeuTTS-2E acronym-pronunciation matrix.

This diagnostic bypasses Dexter's text normalizer on purpose. It tests only how
Paul/NeuTTS-2E interprets several written forms of the same acronym across
multiple sentence contexts. The production backend is loaded once and reused
for every sample; live Dexter behavior is unchanged.

Run:
    uv run python Diagnostics\evaluate_tts_acronym_forms.py

Outputs are written under Runtime\Voice\debug\acronym_matrix\.
"""

from __future__ import annotations

import sys
import threading
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Runtime.Voice.neutts2e_backend import NeuTTS2EBackend

OUTPUT_DIR = REPO_ROOT / "Runtime" / "Voice" / "debug" / "acronym_matrix"

FORMS = {
    "raw": "AI",
    "spaced": "A I",
    "dotted": "A.I.",
    "dotted_spaced": "A. I.",
}

CONTEXTS = {
    "initial": "{token} is running locally.",
    "noun_modifier": "The {token} system is running locally.",
    "conjunction": "The GPU is ready, and {token} is ready too.",
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


def _render(backend: NeuTTS2EBackend, text: str) -> tuple[np.ndarray, int]:
    cancel = threading.Event()
    chunks: list[np.ndarray] = []
    sample_rate = 24_000
    for chunk in backend.synthesize(text, cancel):
        sample_rate = chunk.sample_rate
        chunks.append(np.asarray(chunk.samples, dtype=np.float32).reshape(-1))
    if not chunks:
        raise RuntimeError(f"No audio generated for {text!r}")
    return np.concatenate(chunks), sample_rate


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    backend = NeuTTS2EBackend()

    print("NeuTTS acronym-form matrix — production Dexter is unchanged.")
    print("Loading Paul once, then rendering every form/context combination...\n")

    manifest: list[str] = []
    backend.start()
    try:
        index = 1
        for context_name, template in CONTEXTS.items():
            for form_name, token in FORMS.items():
                text = template.format(token=token)
                filename = f"{index:02d}_{context_name}_{form_name}.wav"
                path = OUTPUT_DIR / filename
                samples, sample_rate = _render(backend, text)
                _write_wav(path, samples, sample_rate)
                line = f"{filename}: {text}"
                manifest.append(line)
                print(line)
                index += 1
    finally:
        backend.stop()

    manifest_path = OUTPUT_DIR / "manifest.txt"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")

    print("\nListen for the form that says the letter names correctly in ALL three contexts.")
    print("A single lucky sample is not enough; NeuTTS-2E is context-sensitive.")
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
