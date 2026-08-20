from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"
PHRASES_FILE = ROOT / "phrases.txt"

WINDOWS_RUNTIME_ROOT = Path(
    os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(ROOT)
) / "DexterTTSBench" / "neutts"


def _runtime_python() -> Path:
    if os.name == "nt":
        return WINDOWS_RUNTIME_ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".runtime" / "neutts" / ".venv" / "bin" / "python"


def _runtime_env() -> dict[str, str]:
    runtime = WINDOWS_RUNTIME_ROOT if os.name == "nt" else ROOT / ".runtime" / "neutts"
    cache = runtime / "cache"
    hf_home = cache / "huggingface"
    torch_home = cache / "torch"
    xdg_home = cache / "xdg"
    for path in (hf_home, torch_home, xdg_home):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.pop("PIP_CONSTRAINT", None)
    env.update(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "TORCH_HOME": str(torch_home),
            "XDG_CACHE_HOME": str(xdg_home),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _load_phrases() -> list[str]:
    if not PHRASES_FILE.exists():
        raise SystemExit(f"Missing phrases file: {PHRASES_FILE}")
    phrases = [
        line.strip()
        for line in PHRASES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not phrases:
        raise SystemExit("phrases.txt contains no test phrases")
    return phrases


def _slug(text: str) -> str:
    value = "".join(char.lower() if char.isalnum() else "-" for char in text)
    return "-".join(part for part in value.split("-") if part)[:42] or "sample"


def _ensure_onnxruntime(python: Path, env: dict[str, str]) -> None:
    probe = subprocess.run(
        [str(python), "-c", "import onnxruntime"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    if probe.returncode == 0:
        return

    print("Installing onnxruntime into the existing disposable NeuTTS environment...")
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "onnxruntime",
        ],
        check=True,
        env=env,
    )


def _controller() -> int:
    python = _runtime_python()
    if not python.exists():
        raise SystemExit(
            "NeuTTS environment is missing. Run:\n"
            "  python3.13 Development\\TTSBench\\bench.py setup neutts"
        )

    env = _runtime_env()
    _ensure_onnxruntime(python, env)
    env["DEXTER_TTSBENCH_ONNX_WORKER"] = "1"
    return subprocess.run([str(python), str(Path(__file__).resolve())], env=env).returncode


def _worker() -> int:
    import soundfile as sf
    from neutts import NeuTTS2E

    phrases = _load_phrases()
    output_dir = RESULTS_ROOT / "neutts-2e"
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    codec_repo = "neuphonic/neucodec-onnx-decoder-int8"
    print("Loading NeuTTS-2E on CPU...")
    print(f"Decoder: {codec_repo}")
    started = time.perf_counter()
    tts = NeuTTS2E(
        backbone_repo="neuphonic/neutts-2e",
        backbone_device="cpu",
        codec_repo=codec_repo,
        codec_device="cpu",
        seed=42,
    )
    model_load_seconds = time.perf_counter() - started
    print(f"Model load: {model_load_seconds:.3f}s")

    speakers = sorted(NeuTTS2E.SPEAKERS)
    print(f"Speakers: {', '.join(speakers)}")
    sample_rate = 24_000
    results: list[dict[str, object]] = []

    for speaker in speakers:
        for index, text in enumerate(phrases, start=1):
            started_phrase = time.perf_counter()
            wav = tts.infer(
                text,
                speaker=speaker,
                emotion="neutral",
                temperature=1.0,
                top_k=50,
            )
            generation_seconds = time.perf_counter() - started_phrase
            audio_seconds = len(wav) / sample_rate
            rtf = generation_seconds / audio_seconds if audio_seconds else None
            output = output_dir / f"{speaker}_{index:02d}_{_slug(text)}.wav"
            sf.write(output, wav, sample_rate)
            results.append(
                {
                    "variant": speaker,
                    "emotion": "neutral",
                    "phrase_index": index,
                    "text": text,
                    "generation_seconds": generation_seconds,
                    "audio_seconds": audio_seconds,
                    "rtf": rtf,
                    "file": output.name,
                }
            )
            print(
                f"{speaker:8} #{index}: generate={generation_seconds:.3f}s "
                f"audio={audio_seconds:.3f}s rtf={rtf:.3f}"
            )

    metrics = {
        "engine": "neutts-2e",
        "device": "cpu",
        "decoder": codec_repo,
        "model_load_seconds": model_load_seconds,
        "voice_prepare_seconds": None,
        "reference_used": False,
        "speakers": speakers,
        "emotion": "neutral",
        "results": results,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Results: {output_dir}")
    return 0


def main() -> int:
    if os.environ.get("DEXTER_TTSBENCH_ONNX_WORKER") == "1":
        return _worker()
    return _controller()


if __name__ == "__main__":
    raise SystemExit(main())
