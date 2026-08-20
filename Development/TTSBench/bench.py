from __future__ import annotations

import argparse
import inspect
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
WINDOWS_SHORT_RUNTIME_ROOT = Path(
    os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(ROOT)
) / "DexterTTSBench"
RESULTS_ROOT = ROOT / "results"
REFERENCE_WAV = ROOT / "reference.wav"
REFERENCE_TXT = ROOT / "reference.txt"
PHRASES_FILE = ROOT / "phrases.txt"

CHATTERBOX_NANO_COMMIT = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"

ENGINE_FAMILY = {
    "chatterbox": "chatterbox",
    "neutts-nano": "neutts",
    "neutts-2e": "neutts",
}
PACKAGE_SPEC = {
    # Nano was added upstream after the PyPI 0.1.7 build without a version bump,
    # so pin the exact upstream commit that introduced the nano=True API.
    "chatterbox": (
        "git+https://github.com/resemble-ai/chatterbox.git@"
        + CHATTERBOX_NANO_COMMIT
    ),
    "neutts": "neutts==1.4.1",
}
PACKAGE_LABEL = {
    "chatterbox": f"Chatterbox Nano upstream@{CHATTERBOX_NANO_COMMIT[:8]}",
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
    if family == "neutts":
        return (3, 10) <= version < (3, 14)
    if family == "chatterbox":
        # Upstream advertises 3.14, but on Windows spacy-pkuseg currently falls
        # back to a C++ source build. Keep the benchmark compiler-free.
        if os.name == "nt":
            return (3, 10) <= version < (3, 14)
        return version >= (3, 10)
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
    candidates: list[Path] = []

    # Prefer a 3.13-3.10 interpreter on Windows for both current engine stacks.
    if os.name == "nt":
        for version in ("3.13", "3.12", "3.11", "3.10"):
            candidate = _launcher_python(version)
            if candidate is not None:
                candidates.append(candidate)

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

    candidates.append(Path(sys.executable))

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
    # Keep the already-working Chatterbox runtime where it is. NeuTTS pulls in
    # PyTorch trees deep enough to trip WinError 206 from this repository path,
    # so on Windows give only NeuTTS a deliberately short disposable root.
    if os.name == "nt" and family == "neutts":
        return WINDOWS_SHORT_RUNTIME_ROOT / family
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
    # Do not inherit an externally-set PIP_CONSTRAINT. On Windows, pip parses
    # that environment variable as a whitespace-separated value, so a repo path
    # containing spaces can be truncated. The harness passes constraints via
    # explicit argv instead.
    env.pop("PIP_CONSTRAINT", None)
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


def _family_probe_code(family: str) -> str:
    if family == "chatterbox":
        return (
            "import inspect; "
            "from chatterbox.tts_turbo import ChatterboxTurboTTS; "
            "assert 'nano' in inspect.signature(ChatterboxTurboTTS.from_pretrained).parameters"
        )
    if family == "neutts":
        return "from neutts import NeuTTS, NeuTTS2E"
    raise ValueError(f"Unknown family: {family}")


def _family_ready(family: str, *, verbose: bool = False) -> bool:
    python = _venv_python(family)
    if not python.exists():
        return False
    try:
        kwargs = {
            "check": True,
            "env": _cache_env(family),
        }
        if not verbose:
            kwargs.update(
                {
                    "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL,
                }
            )
        subprocess.run([str(python), "-c", _family_probe_code(family)], **kwargs)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


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
    family_dir = _family_dir(family)

    if _family_ready(family):
        print(f"{family}: environment already ready at {_venv_python(family).parent.parent}")
        return

    if family_dir.exists():
        print(f"Removing incomplete/outdated {family} environment: {family_dir}")
        shutil.rmtree(family_dir)

    base_python = _find_base_python(family)
    if base_python is None:
        raise SystemExit(
            f"{family} needs Python 3.10-3.13 for this Windows benchmark. "
            "No compatible interpreter was found."
        )

    family_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Creating isolated {family} environment with {base_python}")
        subprocess.run(
            [str(base_python), "-m", "venv", str(family_dir / ".venv")],
            check=True,
        )

        python = _venv_python(family)
        print(f"Installing {PACKAGE_LABEL[family]} (pip cache disabled)")
        pip_command = [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
        ]
        if family == "neutts":
            constraints = ROOT / "neutts-constraints.txt"
            if not constraints.exists():
                raise RuntimeError(f"Missing NeuTTS constraints file: {constraints}")
            pip_command += ["--constraint", str(constraints)]
        pip_command.append(PACKAGE_SPEC[family])

        subprocess.run(
            pip_command,
            check=True,
            env=_cache_env(family),
        )

        if not _family_ready(family):
            print(f"{family} import/API probe failed; traceback follows:")
            _family_ready(family, verbose=True)
            raise RuntimeError(f"{family} installed but failed its import/API probe")
    except Exception:
        if family_dir.exists():
            print(f"Setup failed; deleting incomplete environment: {family_dir}")
            shutil.rmtree(family_dir, ignore_errors=True)
        raise

    print(f"{family}: ready")


def run_engine(engine: str) -> None:
    if engine not in ENGINES:
        raise SystemExit(f"Unknown engine: {engine}")

    family = ENGINE_FAMILY[engine]
    if not _family_ready(family):
        raise SystemExit(
            f"{family} is missing or outdated. Run:\n"
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
        str(_venv_python(family)),
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
        path = _family_dir(family)
        if _family_ready(family):
            state = "ready"
        elif path.exists():
            state = "broken/stale"
        else:
            state = "not installed"
        print(
            f"{family:11} {state:13} "
            f"runtime+cache={_human_bytes(_dir_size(path))} "
            f"path={path}"
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


def _write_metrics(output_dir: Path, metrics: dict[str, object]) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def _worker_chatterbox(args: argparse.Namespace) -> int:
    import torchaudio as ta
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    signature = inspect.signature(ChatterboxTurboTTS.from_pretrained)
    if "nano" not in signature.parameters:
        raise SystemExit(
            "Installed Chatterbox does not contain the Nano API. Re-run setup chatterbox."
        )

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

    _write_metrics(
        args.output_dir,
        {
            "engine": "chatterbox-nano",
            "upstream_commit": CHATTERBOX_NANO_COMMIT,
            "device": "cpu",
            "model_load_seconds": model_load_seconds,
            "voice_prepare_seconds": voice_prepare_seconds,
            "reference_used": bool(args.reference_wav and args.reference_wav.exists()),
            "results": results,
        },
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

    _write_metrics(
        args.output_dir,
        {
            "engine": "neutts-nano",
            "device": "cpu",
            "model_load_seconds": model_load_seconds,
            "voice_prepare_seconds": voice_prepare_seconds,
            "reference_used": True,
            "results": results,
        },
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

    _write_metrics(
        args.output_dir,
        {
            "engine": "neutts-2e",
            "device": "cpu",
            "model_load_seconds": model_load_seconds,
            "voice_prepare_seconds": None,
            "reference_used": False,
            "speakers": speakers,
            "emotion": "neutral",
            "results": results,
        },
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
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
