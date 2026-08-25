r"""Evaluate a lightweight text-normalization frontend before production wiring.

This diagnostic intentionally keeps Dexter's live TTS path unchanged. It uses
Utterwise from the evaluation-only uv dependency group, applies only a tiny
Dexter-specific markup/identifier cleanup, and reports:

* raw -> pre-cleaned -> normalized text for realistic assistant output;
* critical acronym/markup sanity checks;
* import footprint and normalization throughput; and
* optional before/after WAVs through Dexter's existing NeuTTS backend.

Run text-only evaluation:
    uv run --group tts-normalization-eval python Diagnostics\evaluate_tts_text_normalization.py

Generate raw vs normalized audio samples too:
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

# Utterwise already owns ordinary Markdown stripping (headings, **, __, `, ~~)
# and speech normalization. Dexter only fills two assistant-output gaps:
# single-underscore emphasis / snake_case, and word-style hyphens. Keep these
# rules conservative so math such as x_1, negative numbers, phone numbers, and
# ISO dates still reach the expert normalizer unchanged.
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


def _load_utterwise():
    before_modules = set(sys.modules)
    tracemalloc.start()
    started = time.perf_counter()
    from utterwise import normalize

    import_seconds = time.perf_counter() - started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    imported_modules = sorted(set(sys.modules) - before_modules)
    return normalize, import_seconds, current_bytes, peak_bytes, imported_modules


def _distribution_size_bytes() -> int:
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


def _critical_checks(outputs: dict[str, str]) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def add(label: str, ok: bool, detail: str) -> None:
        checks.append((label, ok, detail))

    ai = outputs["AI is running locally."]
    json_text = outputs["JSON output is valid."]
    technical = outputs["The CPU and GPU are busy."]
    markup = outputs["_important_ text should sound normal."]
    snake = outputs["snake_case should become ordinary words."]
    hyphen = outputs["real-time streaming should stay natural."]
    mathish = outputs["Keep x_1 intact as math-like text."]
    iso_date = outputs["The date is 2026-04-03."]

    add("AI is not collapsed to a word", "A I" in ai.upper(), ai)
    add("JSON is expanded/spelled", "J S O N" in json_text.upper(), json_text)
    add(
        "CPU/GPU are expanded/spelled",
        "C P U" in technical.upper() and "G P U" in technical.upper(),
        technical,
    )
    add("Markdown underscores are gone", "_" not in markup, markup)
    add("snake_case becomes words", "_" not in snake and "snake case" in snake.lower(), snake)
    add("word hyphen becomes natural boundary", "real-time" not in hyphen.lower(), hyphen)
    add("math-like underscore is preserved for normalizer", "x" in mathish.lower(), mathish)
    add("ISO date still normalizes as a date", "april" in iso_date.lower(), iso_date)
    return checks


def _write_wav(path: Path, samples, sample_rate: int) -> None:
    import numpy as np

    pcm = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


def _synthesize_to_wav(text: str, path: Path) -> None:
    import numpy as np

    from Runtime.Voice.neutts2e_backend import NeuTTS2EBackend

    backend = NeuTTS2EBackend()
    cancel = threading.Event()
    chunks = []
    sample_rate = 24_000
    backend.start()
    try:
        for chunk in backend.synthesize(text, cancel):
            sample_rate = chunk.sample_rate
            chunks.append(chunk.samples)
    finally:
        backend.stop()

    if not chunks:
        raise RuntimeError(f"No audio generated for {text!r}")
    _write_wav(path, np.concatenate(chunks), sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        action="store_true",
        help="also generate raw and normalized comparison WAVs through NeuTTS",
    )
    args = parser.parse_args()

    normalize, import_seconds, current_bytes, peak_bytes, imported_modules = _load_utterwise()

    print("Utterwise evaluation only — live Dexter TTS is unchanged.\n")
    print("FOOTPRINT")
    print(f"  distribution files: {_distribution_size_bytes() / 1024:.1f} KiB")
    print(f"  import time:        {import_seconds * 1000:.2f} ms")
    print(f"  Python alloc now:   {current_bytes / 1024:.1f} KiB")
    print(f"  Python alloc peak:  {peak_bytes / 1024:.1f} KiB")
    print(f"  modules imported:   {len(imported_modules)}")

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

    print("\nCRITICAL CHECKS")
    failures = 0
    for label, ok, detail in _critical_checks(outputs):
        marker = "PASS" if ok else "FAIL"
        failures += 0 if ok else 1
        print(f"  {marker:4s}  {label}: {detail}")

    # Warmed throughput: representative text repeated enough to make timer
    # overhead irrelevant, but still tiny compared with TTS inference.
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
        output_dir = REPO_ROOT / "Runtime" / "Voice" / "debug"
        raw_path = output_dir / "normalization_raw.wav"
        normalized_path = output_dir / "normalization_utterwise.wav"
        print("\nGenerating raw comparison...")
        _synthesize_to_wav(AUDIO_TEXT, raw_path)
        print("Generating normalized comparison...")
        _synthesize_to_wav(normalized_audio_text, normalized_path)
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
