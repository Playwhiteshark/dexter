r"""Evaluate a lightweight text-normalization frontend before production wiring.

Live Dexter is unchanged. Utterwise comes from the evaluation-only uv group;
Dexter contributes only conservative cleanup for single-underscore emphasis,
snake_case, word-style hyphens, punctuation-preserving rendering, and a tiny
application pronunciation-override map.

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
from typing import Any

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
    "Wait; AI is ready, right? Yes: it is.",
    "It is -3°C; that's cold.",
]

AUDIO_TEXT = (
    "AI and JSON are running in real-time; SQL is ready too. "
    "The GPU reads snake_case and _important_ text. "
    "Python 3.13 is using 81% of the CPU, which is fairly high."
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

# Keep overrides extremely small. Utterwise's generic spaced-letter rendering
# is useful for technical initialisms such as SQL/CPU/GPU. The exceptions below
# are cases where NeuTTS itself has demonstrated ambiguity or where the desired
# spoken form is intentionally word-like.
PRONUNCIATION_OVERRIDES = {
    # A bare "A" can be read as the English article ("uh") rather than the
    # letter name, so spell AI with unambiguous ordinary words.
    "AI": "ay eye",
    # User preference and common technical pronunciation.
    "JSON": "jay son",
}

# Utterwise's plain formatter intentionally discards punctuation. NeuTTS-2E
# benefits from normal written punctuation as prosody cues, so retain the basic
# punctuation that carries sentence structure while still verbalizing semantic
# tokens such as -3, dates, percentages, and currency through Utterwise.
_PRESERVED_PUNCTUATION = {
    ".",
    ",",
    "!",
    "?",
    ":",
    ";",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    '"',
    "'",
}
_CLOSE_PUNCTUATION = re.compile(r"\s+([,.;:!?\)\]\}])")
_OPEN_PUNCTUATION = re.compile(r"([\(\[\{])\s+")


def dexter_preclean(text: str) -> str:
    value = _SINGLE_UNDERSCORE_EMPHASIS_RE.sub(r"\1", text)
    value = _IDENTIFIER_UNDERSCORE_RE.sub(" ", value)
    value = _WORD_HYPHEN_RE.sub(" ", value)
    # NeuTTS has been unreliable with dash glyphs in assistant prose. Preserve
    # their pause semantics with a comma; semantic minus signs are untouched.
    value = value.replace("—", ", ").replace("–", ", ")
    return _WHITESPACE_RE.sub(" ", value).strip()


def apply_pronunciation_overrides(text: str) -> str:
    value = text
    for written, spoken in PRONUNCIATION_OVERRIDES.items():
        value = re.sub(rf"\b{re.escape(written)}\b", spoken, value)
    return value


def load_utterwise():
    before_modules = set(sys.modules)
    tracemalloc.start()
    started = time.perf_counter()
    from utterwise.normalizers import Normalizer

    import_seconds = time.perf_counter() - started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    module_count = len(set(sys.modules) - before_modules)
    return Normalizer(), import_seconds, current_bytes, peak_bytes, module_count


def normalize_for_neutts(normalizer: Any, text: str) -> str:
    """Use Utterwise's semantics while retaining NeuTTS-useful punctuation."""
    prepared = apply_pronunciation_overrides(dexter_preclean(text))
    result = normalizer.normalize(prepared, output="plain", policy="assistant")

    # Token spans refer to result.cleaned_text after Utterwise's own Markdown
    # preprocessing. Rebuild from those spans so semantic replacements happen
    # without throwing away ordinary punctuation or the whitespace between it.
    cleaned = result.cleaned_text
    parts: list[str] = []
    cursor = 0
    for token, segment in zip(result.tokens, result.segments):
        start = int(token.start)
        end = int(token.end)
        if start > cursor:
            parts.append(cleaned[cursor:start])

        if token.type == "PUNCTUATION" and token.value in _PRESERVED_PUNCTUATION:
            parts.append(token.value)
        elif segment.text:
            parts.append(segment.text)
        cursor = end

    if cursor < len(cleaned):
        parts.append(cleaned[cursor:])

    rendered = "".join(parts)
    rendered = _WHITESPACE_RE.sub(" ", rendered)
    rendered = _CLOSE_PUNCTUATION.sub(r"\1", rendered)
    rendered = _OPEN_PUNCTUATION.sub(r"\1", rendered)
    return rendered.strip()


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
    sql_text = outputs["Run the SQL query through the API."]
    iso_date = outputs["The date is 2026-04-03."]
    punctuation = outputs["Wait; AI is ready, right? Yes: it is."]
    negative = outputs["It is -3°C; that's cold."]
    return [
        ("AI uses unambiguous letter names", "ay eye" in ai.lower(), ai),
        ("JSON uses preferred pronunciation", "jay son" in json_text.lower(), json_text),
        (
            "CPU/GPU are expanded",
            "C P U" in technical.upper() and "G P U" in technical.upper(),
            technical,
        ),
        # Do not override SQL. The earlier Utterwise form S Q L sounded correct,
        # while the custom 'ess cue ell' spelling introduced a NeuTTS artifact.
        ("SQL remains generic spaced letters", "S Q L" in sql_text.upper(), sql_text),
        ("single-underscore emphasis is gone", "_" not in markup, markup),
        (
            "snake_case becomes words",
            "_" not in snake and "snake case" in snake.lower(),
            snake,
        ),
        ("word hyphen becomes a boundary", "real-time" not in hyphen.lower(), hyphen),
        ("ISO date remains semantic", "april" in iso_date.lower(), iso_date),
        (
            "basic punctuation survives",
            all(mark in punctuation for mark in (";", ",", "?", ":", ".")),
            punctuation,
        ),
        (
            "negative number remains semantic",
            "minus three" in negative.lower() and ";" in negative,
            negative,
        ),
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

    normalizer, import_s, current_b, peak_b, module_count = load_utterwise()
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
        prepared = apply_pronunciation_overrides(dexter_preclean(raw))
        spoken = normalize_for_neutts(normalizer, raw)
        outputs[raw] = spoken
        print(f"\n[{index:02d}] raw:        {raw}")
        if prepared != raw:
            print(f"     prepared:   {prepared}")
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
        normalize_for_neutts(normalizer, benchmark_text)
    elapsed = time.perf_counter() - started
    print("\nTHROUGHPUT")
    print(f"  {iterations} passes: {elapsed:.3f}s")
    print(f"  per pass:         {(elapsed / iterations) * 1000:.3f} ms")
    print(f"  chars/pass:       {len(benchmark_text)}")

    normalized_audio_text = normalize_for_neutts(normalizer, AUDIO_TEXT)
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
