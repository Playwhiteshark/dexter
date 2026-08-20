# Dexter TTS Bench

Small, disposable CPU-only listening/latency harness for conversational TTS candidates.

It keeps each engine's Python environment **and model caches** under
`Development/TTSBench/.runtime/`, so losing engines can be deleted cleanly.
Generated WAVs live under `results/` and are kept when an engine is cleaned.

## Shared reference voice

Put these beside `bench.py` if you want to test reference/cloned voices:

- `reference.wav` — one clean natural speech clip. About 8-10 seconds is a good common target.
- `reference.txt` — the exact transcript of `reference.wav`.

Behavior when they are absent:

- **Chatterbox Nano** always generates its built-in voice. If `reference.wav` exists,
  it also prepares that voice once and generates a second `reference` set.
- **NeuTTS Nano** requires both `reference.wav` and `reference.txt`; if either is
  absent, the harness skips it cleanly.
- **NeuTTS-2E** ignores the reference files and tests every bundled speaker in
  `neutral` mode.

`reference.wav`, `reference.txt`, generated results, and temporary runtimes are
gitignored.

## Commands

From the Dexter repository root:

```bat
python Development\TTSBench\bench.py status
```

Install only the engine family you are about to test:

```bat
python Development\TTSBench\bench.py setup chatterbox
python Development\TTSBench\bench.py run chatterbox
```

NeuTTS Nano and NeuTTS-2E share one package environment:

```bat
python Development\TTSBench\bench.py setup neutts
python Development\TTSBench\bench.py run neutts-nano
python Development\TTSBench\bench.py run neutts-2e
```

Listen to all generated WAVs:

```bat
python Development\TTSBench\bench.py listen
```

Delete an engine family and all of its downloaded model/cache files while
keeping the generated WAVs:

```bat
python Development\TTSBench\bench.py clean chatterbox
python Development\TTSBench\bench.py clean neutts
```

Or remove every temporary engine:

```bat
python Development\TTSBench\bench.py clean all
```

## Python versions

The controller itself uses only the standard library.

- Chatterbox Nano supports Python 3.10+ and can use Python 3.14.
- NeuTTS 1.4.1 requires Python 3.10-3.13. The harness searches for a compatible
  interpreter without changing your global Python, including Dexter's existing
  tools venv and Windows `py` launcher installations.

If no compatible NeuTTS interpreter exists, `setup neutts` stops with a clear
message rather than installing anything into another environment.

## Metrics

Each run writes `metrics.json` beside its WAVs. It records:

- model load time;
- one-time reference/voice preparation time where applicable;
- generation time for every phrase;
- generated audio duration;
- realtime factor (`generation_seconds / audio_seconds`).

A realtime factor below `1.0` means the model generated the complete audio faster
than it would take to play. This benchmark measures complete-utterance generation;
streaming first-audio latency is a separate test for the eventual Dexter backend.
