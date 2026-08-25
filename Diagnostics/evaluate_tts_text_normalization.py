r"""Evaluate a lightweight text-normalization frontend before production wiring.

Live Dexter is unchanged. Utterwise comes from the evaluation-only uv group;
Dexter contributes only conservative cleanup for single-underscore emphasis,
snake_case, and word-style hyphens.

Text-only:
    uv run --group tts-normalization-eval python Diagnostics\evaluate_tts_text_normalization.py

With raw/normalized NeuTTS WAV comparison:
    uv run --group tts-normalization-eval python Diagnostics\evaluate_tts_text_normalization.py --audio
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
import threading
import time
import tracemalloc
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CASES = [
    "AI is running locally.",
    "JSON output is valid.",
    "The CPU and GPU are busy.",
    "The LLM, TTS, and STT are active.",
    "NASA uses AI.",
    "_important_ text should sound normal.",
    "__bold_name__ should not confuse speech.",
    "snake_case should become ordinary words.",
    "real-time streaming should stay natural.",
    "The RTX-5050 is active.",
    "Python 3.13 is installed.",
    "Dexter v0.3.0 is running.",
    "Memory use is 5.57 GB.",
    "GPU usage is 81%.",
    "Open https://openai.com for details.",
    "Email hello@example.com.",
    "Call 911 immediately.",
    "Flight 911 departs at 6.",
    "The price is $12.50.",
    "The temperature is 25°C.",
    "Run the SQL query through the API.",
    "Keep x_1 intact as math-like text.",
    "The date is 2026-04-03.",
]

AUDIO_TEXT = (
    "AI and JSON are running in real-time. "
    "The GPU reads snake_case and _important_ text. "
    "Python 3.13 is using 81% of the CPU."
)

# Utterwise already handles normal Markdown such as headings, **, __, `, and ~~.
# These fill only the gaps relevant to Dexter output while leaving x_1, -3,
# phone-number hyphens, and ISO dates available to the normalizer unchanged.
_WHITESPACE_RE = re.compile(r"\s+")
_SINGLE_UNDERSCORE_EMPHASIS_RE = re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)")
_IDENTIFIER_UNDERSCORE_RE = re.compile(r"(?<=[A-Za-z])_(?=[A-Za-z])")
_WORD_HYPHEN_RE = re.compile(
    r"(?<=[A-Za-z])-(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])-(?=[A-Za-z])"
)


def dexter_preclean(text: str) -> str:
    value = _SINGLE_UNDERSCORE_EMPHASIS_RE.sub(r"\1", text)
    value = _IDENTIFIER_UNDERSCORE_RE.sub(" ", value)
    value = _WORD_HYPHEN_RE.sub(" ", value)
    value = value.replace("—", ", ").replace("–", ", ")
    return _WHITESPACE_RE.sub(" ", value).strip()


def load_utterwise():
    before_modules = set(sys.modules)
    tracemalloc.start()
    started = time.perf_counter()
    from utterwise import normalize

    import_seconds = time.perf_counter() - started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    module_count = len(set(sys.modules) - before_modules)
    return normalize, import_seconds, current_bytes, peak_bytes, module_count


def distribution_size_bytes() -> int:
    try:
        dist = importlib.metadata.distribution("utterwise")
    except importlib.metadata.PackageNotFoundError:
        return 0
    total = 0
    for entry in dist.files or ():
        try:
            path = Path(dist.locate_file(entry))
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            pass
    return total


def critical_checks(outputs: dict[str, str]) -> list[tuple[str, bool, str]]:
    ai = outputs["AI is running locally."]
    json_text = outputs["JSON output is valid."]
    technical = outputs["The CPU and GPU are busy."]
    markup = outputs["_important_ text should sound normal."]
    snake = outputs["snake_case should become ordinary words."]
    hyphen = outputs["real-time streaming should stay natural."]
    iso_date = outputs["The date is 2026-04-03."]
    return [
        ("AI is expanded", "A I" in ai.upper(), ai),
        ("JSON is expanded", "J S O N" in json_text.upper(), json_text),
        (
            "CPU/GPU are expanded",
            "C P U" in technical.upper() and "G P U" in technical.upper(),
            technical,
        ),
        ("single-underscore emphasis is gone", "_" not in markup, markup),
        (
            "snake_case becomes words",
            "_" not in snake and "snake case" in snake.lower(),
            snake,
        ),
        ("word hyphen becomes a boundary", "real-time" not in hyphen.lower(), hyphen),
        ("ISO date remains semantic", "april" in iso_date.lower(), iso_date),
    ]


def write_wav(path: Path, samples, sample_rate: int) -> None:
    import numpy as np

    pcm = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


def synthesize_pair(raw_text: str, normalized_text: str, output_dir: Path) -> tuple[Path, Path]:
    import numpy as np

    from Runtime.Voice.neutts2e_backend import NeuTTS2EBackend

    backend = NeuTTS2EBackend()
    cancel = threading.Event()

    def render(text: str):
        chunks = []
        sample_rate = 24_000
        for chunk in backend.synthesize(text, cancel):
            sample_rate = chunk.sample_rate
            chunks.append(chunk.samples)
        if not chunks:
            raise RuntimeError(f"No audio generated for {text!r}")
        return np.concatenate(chunks), sample_rate

    backend.start()
    try:
        raw_samples, raw_rate = render(raw_text)
        normalized_samples, normalized_rate = render(normalized_text)
    finally:
        backend.stop()

    raw_path = output_dir / "normalization_raw.wav"
    normalized_path = output_dir / "normalization_utterwise.wav"
    write_wav(raw_path, raw_samples, raw_rate)
    write_wav(normalized_path, normalized_samples, normalized_rate)
    return raw_path, normalized_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", action="store_true")
    args = parser.parse_args()

    normalize, import_s, current_b, peak_b, module_count = load_utterwise()
    print("Utterwise evaluation only — live Dexter TTS is unchanged.\n")
    print("FOOTPRINT")
    print(f"  distribution files: {distribution_size_bytes() / 1024:.1f} KiB")
    print(f"  import time:        {import_s * 1000:.2f} ms")
    print(f"  Python alloc now:   {current_b / 1024:.1f} KiB")
    print(f"  Python alloc peak:  {peak_b / 1024:.1f} KiB")
    print(f"  modules imported:   {module_count}")

    outputs: dict[str, str] = {}
    print("\nNORMALIZATION CORPUS")
    for index, raw in enumerate(CASES, start=1):
        cleaned = dexter_preclean(raw)
        spoken = str(normalize(cleaned))
        outputs[raw] = spoken
        print(f"\n[{index:02d}] raw:        {raw}")
        if cleaned != raw:
            print(f"     preclean:   {cleaned}")
        print(f"     normalized: {spoken}")

    failures = 0
    print("\nCRITICAL CHECKS")
    for label, ok, detail in critical_checks(outputs):
        marker = "PASS" if ok else "FAIL"
        failures += 0 if ok else 1
        print(f"  {marker:4s}  {label}: {detail}")

    benchmark_text = " ".join(CASES)
    iterations = 200
    started = time.perf_counter()
    for _ in range(iterations):
        normalize(dexter_preclean(benchmark_text))
    elapsed = time.perf_counter() - started
    print("\nTHROUGHPUT")
    print(f"  {iterations} passes: {elapsed:.3f}s")
    print(f"  per pass:         {(elapsed / iterations) * 1000:.3f} ms")
    print(f"  chars/pass:       {len(benchmark_text)}")

    normalized_audio_text = str(normalize(dexter_preclean(AUDIO_TEXT)))
    print("\nAUDIO SAMPLE TEXT")
    print(f"  raw:        {AUDIO_TEXT}")
    print(f"  normalized: {normalized_audio_text}")

    if args.audio:
        print("\nGenerating both WAVs with one NeuTTS worker...")
        raw_path, normalized_path = synthesize_pair(
            AUDIO_TEXT,
            normalized_audio_text,
            REPO_ROOT / "Runtime" / "Voice" / "debug",
        )
        print("\nAUDIO OUTPUT")
        print(f"  raw:        {raw_path}")
        print(f"  normalized: {normalized_path}")

    if failures:
        print(f"\n{failures} critical normalization check(s) failed.")
        return 2
    print("\nAll critical normalization checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
