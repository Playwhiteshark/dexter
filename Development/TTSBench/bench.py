from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
RUNTIME_ROOT = ROOT / ".runtime"
RESULTS_ROOT = ROOT / "results"
REFERENCE_WAV = ROOT / "reference.wav"
REFERENCE_TXT = ROOT / "reference.txt"
PHRASES_FILE = ROOT / "phrases.txt"

ENGINE_FAMILY = {
    "chatterbox": "chatterbox",
    "neutts-nano": "neutts",
    "neutts-2e": "neutts",
}
PACKAGE_SPEC = {
    "chatterbox": "chatterbox-tts==0.1.7",
    "neutts": "neutts==1.4.1",
}
ENGINES = tuple(ENGINE_FAMILY)


def _python_version(executable: Path) -> tuple[int, int] | None:
    try:
        completed = subprocess.run(
            [
                str(executable),
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        major, minor = completed.stdout.strip().split(".", 1)
        return int(major), int(minor)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _supports(family: str, version: tuple[int, int] | None) -> bool:
    if version is None:
        return False
    if family == "chatterbox":
        return version >= (3, 10)
    if family == "neutts":
        return (3, 10) <= version < (3, 14)
    return False


def _launcher_python(version: str) -> Path | None:
    if os.name != "nt":
        return None
    try:
        completed = subprocess.run(
            ["py", f"-{version}", "-c", "import sys; print(sys.executable)"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return Path(value) if value else None


def _find_base_python(family: str) -> Path | None:
    candidates: list[Path] = [Path(sys.executable)]

    dexter_tools_python = (
        REPO_ROOT
        / "Runtime"
        / "Tools"
        / "dexter-tools"
        / ".venv"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    if dexter_tools_python.exists():
        candidates.append(dexter_tools_python)

    versions = (
        ("3.13", "3.12", "3.11", "3.10")
        if family == "neutts"
        else ("3.14", "3.13", "3.12", "3.11", "3.10")
    )
    for version in versions:
        candidate = _launcher_python(version)
        if candidate is not None:
            candidates.append(candidate)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if _supports(family, _python_version(candidate)):
            return candidate
    return None


def _family_dir(family: str) -> Path:
    return RUNTIME_ROOT / family


def _venv_python(family: str) -> Path:
    suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return _family_dir(family) / ".venv" / suffix


def _cache_env(family: str) -> dict[str, str]:
    cache = _family_dir(family) / "cache"
    hf_home = cache / "huggingface"
    torch_home = cache / "torch"
    xdg_home = cache / "xdg"
    for path in (hf_home, torch_home, xdg_home):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "TORCH_HOME": str(torch_home),
            "XDG_CACHE_HOME": str(xdg_home),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024.0 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def _load_phrases(path: Path = PHRASES_FILE) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Missing phrases file: {path}")
    phrases = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not phrases:
        raise SystemExit(f"{path.name} contains no test phrases")
    return phrases


def _slug(text: str) -> str:
    value = "".join(char.lower() if char.isalnum() else "-" for char in text)
    return "-".join(part for part in value.split("-") if part)[:42] or "sample"


def _resolve_family(name: str) -> str:
    if name in ENGINE_FAMILY:
        return ENGINE_FAMILY[name]
    if name in PACKAGE_SPEC:
        return name
    raise SystemExit(f"Unknown engine/family: {name}")


def setup(name: str) -> None:
    family = _resolve_family(name)
    python = _venv_python(family)
    if python.exists():
        print(f"{family}: environment already exists at {python.parent.parent}")
        return

    base_python = _find_base_python(family)
    if base_python is None:
        if family == "neutts":
            raise SystemExit(
                "NeuTTS requires Python 3.10-3.13. No compatible interpreter was found. "
                "Python 3.14 can still run this benchmark controller, but not NeuTTS 1.4.1."
            )
        raise SystemExit("No compatible Python interpreter found for Chatterbox.")

    family_dir = _family_dir(family)
    family_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating isolated {family} environment with {base_python}")
    subprocess.run(
        [str(base_python), "-m", "venv", str(family_dir / ".venv")],
        check=True,
    )

    python = _venv_python(family)
    print(f"Installing {PACKAGE_SPEC[family]} (pip cache disabled)")
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            PACKAGE_SPEC[family],
        ],
        check=True,
        env=_cache_env(family),
    )
    print(f"{family}: ready")


def run_engine(engine: str) -> None:
    if engine not in ENGINES:
        raise SystemExit(f"Unknown engine: {engine}")

    family = ENGINE_FAMILY[engine]
    python = _venv_python(family)
    if not python.exists():
        raise SystemExit(
            f"{family} is not installed. Run:\n"
            f"  python Development\\TTSBench\\bench.py setup {engine}"
        )

    _load_phrases()

    if engine == "neutts-nano":
        missing: list[str] = []
        if not REFERENCE_WAV.exists():
            missing.append("reference.wav")
        if not REFERENCE_TXT.exists():
            missing.append("reference.txt")
        if missing:
            print(
                "NeuTTS Nano skipped: it needs the shared reference audio and its exact "
                f"transcript. Missing: {', '.join(missing)}"
            )
            print(f"Put them in: {ROOT}")
            return

    output_dir = RESULTS_ROOT / engine
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(python),
        str(Path(__file__).resolve()),
        "_worker",
        engine,
        "--phrases",
        str(PHRASES_FILE),
        "--output-dir",
        str(output_dir),
    ]
    if REFERENCE_WAV.exists():
        command += ["--reference-wav", str(REFERENCE_WAV)]
    if REFERENCE_TXT.exists():
        command += ["--reference-text", str(REFERENCE_TXT)]

    print(f"Running {engine} on CPU...")
    subprocess.run(command, check=True, env=_cache_env(family))
    print(f"Results: {output_dir}")


def clean(name: str) -> None:
    targets = list(PACKAGE_SPEC) if name == "all" else [_resolve_family(name)]
    for family in targets:
        path = _family_dir(family)
        if not path.exists():
            print(f"{family}: nothing to delete")
            continue
        print(f"Deleting {family} environment + model/cache files: {path}")
        shutil.rmtree(path)
    print("Generated WAV results and reference files were kept.")


def status() -> None:
    phrases = _load_phrases()
    print(f"Benchmark root:  {ROOT}")
    print(f"Reference WAV:   {'present' if REFERENCE_WAV.exists() else 'missing'}")
    print(f"Reference text:  {'present' if REFERENCE_TXT.exists() else 'missing'}")
    print(f"Test phrases:    {len(phrases)}")
    print()
    for family in PACKAGE_SPEC:
        python = _venv_python(family)
        state = "installed" if python.exists() else "not installed"
        print(
            f"{family:11} {state:13} "
            f"runtime+cache={_human_bytes(_dir_size(_family_dir(family)))}"
        )
    print()
    for engine in ENGINES:
        result_dir = RESULTS_ROOT / engine
        wavs = len(list(result_dir.glob("*.wav"))) if result_dir.exists() else 0
        print(f"{engine:13} results={wavs} wav(s)")


def listen() -> None:
    wavs = sorted(RESULTS_ROOT.glob("*/*.wav"))
    if not wavs:
        print("No generated WAVs yet.")
        return
    if os.name != "nt":
        print("Interactive playback is currently implemented with Windows winsound.")
        for wav in wavs:
            print(wav)
        return

    import winsound

    while True:
        print()
        for index, wav in enumerate(wavs, start=1):
            print(f"{index:2}. {wav.relative_to(RESULTS_ROOT)}")
        choice = input("Play number, or q to quit: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return
        try:
            wav = wavs[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid selection.")
            continue
        print(f"Playing {wav.relative_to(RESULTS_ROOT)}")
        winsound.PlaySound(str(wav), winsound.SND_FILENAME)


def _worker_chatterbox(args: argparse.Namespace) -> int:
    import torchaudio as ta
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    phrases = _load_phrases(args.phrases)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained(device="cpu", nano=True)
    model_load_seconds = time.perf_counter() - started
    print(f"Model load: {model_load_seconds:.3f}s")

    results: list[dict[str, object]] = []

    def generate_variant(variant: str) -> None:
        for index, text in enumerate(phrases, start=1):
            started_phrase = time.perf_counter()
            wav = model.generate(text)
            generation_seconds = time.perf_counter() - started_phrase
            audio_seconds = float(wav.shape[-1]) / float(model.sr)
            rtf = generation_seconds / audio_seconds if audio_seconds else None
            output = args.output_dir / f"{variant}_{index:02d}_{_slug(text)}.wav"
            ta.save(str(output), wav, model.sr)
            results.append(
                {
                    "variant": variant,
                    "phrase_index": index,
                    "text": text,
                    "generation_seconds": generation_seconds,
                    "audio_seconds": audio_seconds,
                    "rtf": rtf,
                    "file": output.name,
                }
            )
            print(
                f"{variant:9} #{index}: generate={generation_seconds:.3f}s "
                f"audio={audio_seconds:.3f}s rtf={rtf:.3f}"
            )

    print("Generating built-in Chatterbox Nano voice...")
    generate_variant("builtin")

    voice_prepare_seconds = None
    if args.reference_wav and args.reference_wav.exists():
        print(f"Preparing shared reference voice: {args.reference_wav}")
        started = time.perf_counter()
        model.prepare_conditionals(str(args.reference_wav))
        voice_prepare_seconds = time.perf_counter() - started
        print(f"Reference preparation: {voice_prepare_seconds:.3f}s")
        generate_variant("reference")
    else:
        print("No reference.wav present; cloned Chatterbox variant skipped.")

    metrics = {
        "engine": "chatterbox-nano",
        "device": "cpu",
        "model_load_seconds": model_load_seconds,
        "voice_prepare_seconds": voice_prepare_seconds,
        "reference_used": bool(args.reference_wav and args.reference_wav.exists()),
        "results": results,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return 0


def _worker_neutts_nano(args: argparse.Namespace) -> int:
    import soundfile as sf
    from neutts import NeuTTS

    if not args.reference_wav or not args.reference_text:
        raise SystemExit("NeuTTS Nano worker requires reference.wav and reference.txt")

    phrases = _load_phrases(args.phrases)
    reference_text = args.reference_text.read_text(encoding="utf-8").strip()
    if not reference_text:
        raise SystemExit("reference.txt is empty; it must exactly transcribe reference.wav")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tts = NeuTTS(
        backbone_repo="neuphonic/neutts-nano",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
        seed=42,
    )
    model_load_seconds = time.perf_counter() - started
    print(f"Model load: {model_load_seconds:.3f}s")

    started = time.perf_counter()
    ref_codes = tts.encode_reference(str(args.reference_wav))
    voice_prepare_seconds = time.perf_counter() - started
    print(f"Reference preparation: {voice_prepare_seconds:.3f}s")

    sample_rate = 24_000
    results: list[dict[str, object]] = []
    for index, text in enumerate(phrases, start=1):
        started_phrase = time.perf_counter()
        wav = tts.infer(text, ref_codes, reference_text, temperature=1.0, top_k=50)
        generation_seconds = time.perf_counter() - started_phrase
        audio_seconds = len(wav) / sample_rate
        rtf = generation_seconds / audio_seconds if audio_seconds else None
        output = args.output_dir / f"reference_{index:02d}_{_slug(text)}.wav"
        sf.write(output, wav, sample_rate)
        results.append(
            {
                "variant": "reference",
                "phrase_index": index,
                "text": text,
                "generation_seconds": generation_seconds,
                "audio_seconds": audio_seconds,
                "rtf": rtf,
                "file": output.name,
            }
        )
        print(
            f"reference #{index}: generate={generation_seconds:.3f}s "
            f"audio={audio_seconds:.3f}s rtf={rtf:.3f}"
        )

    metrics = {
        "engine": "neutts-nano",
        "device": "cpu",
        "model_load_seconds": model_load_seconds,
        "voice_prepare_seconds": voice_prepare_seconds,
        "reference_used": True,
        "results": results,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return 0


def _worker_neutts_2e(args: argparse.Namespace) -> int:
    import soundfile as sf
    from neutts import NeuTTS2E

    phrases = _load_phrases(args.phrases)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    tts = NeuTTS2E(
        backbone_repo="neuphonic/neutts-2e",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
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
            output = args.output_dir / f"{speaker}_{index:02d}_{_slug(text)}.wav"
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
        "model_load_seconds": model_load_seconds,
        "voice_prepare_seconds": None,
        "reference_used": False,
        "speakers": speakers,
        "emotion": "neutral",
        "results": results,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return 0


def _run_worker(args: argparse.Namespace) -> int:
    if args.engine == "chatterbox":
        return _worker_chatterbox(args)
    if args.engine == "neutts-nano":
        return _worker_neutts_nano(args)
    if args.engine == "neutts-2e":
        return _worker_neutts_2e(args)
    raise SystemExit(f"Unknown worker engine: {args.engine}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Disposable CPU TTS benchmark harness for Dexter."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    setup_parser = sub.add_parser("setup", help="Create one isolated engine environment.")
    setup_parser.add_argument("name", choices=(*ENGINES, *PACKAGE_SPEC.keys()))

    run_parser = sub.add_parser("run", help="Generate the standard test samples.")
    run_parser.add_argument("engine", choices=ENGINES)

    clean_parser = sub.add_parser(
        "clean", help="Delete engine environments/caches but keep WAV results."
    )
    clean_parser.add_argument("name", choices=(*ENGINES, *PACKAGE_SPEC.keys(), "all"))

    sub.add_parser("status", help="Show references, installs, disk use, and results.")
    sub.add_parser("listen", help="Interactively play generated WAV results on Windows.")

    worker_parser = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("engine", choices=ENGINES)
    worker_parser.add_argument("--phrases", type=Path, required=True)
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    worker_parser.add_argument("--reference-wav", type=Path)
    worker_parser.add_argument("--reference-text", type=Path)

    args = parser.parse_args()
    try:
        if args.command == "setup":
            setup(args.name)
        elif args.command == "run":
            run_engine(args.engine)
        elif args.command == "clean":
            clean(args.name)
        elif args.command == "status":
            status()
        elif args.command == "listen":
            listen()
        elif args.command == "_worker":
            return _run_worker(args)
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
