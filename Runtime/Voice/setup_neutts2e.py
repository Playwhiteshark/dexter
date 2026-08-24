"""Install/manage Dexter's isolated NeuTTS-2E Paul runtime.

Usage from the repository root:
    python3.13 Runtime\Voice\setup_neutts2e.py setup
    python3.13 Runtime\Voice\setup_neutts2e.py auth
    python3.13 Runtime\Voice\setup_neutts2e.py status
    python3.13 Runtime\Voice\setup_neutts2e.py smoke

The heavy ML stack lives under %LOCALAPPDATA%\Dexter\Voice\neutts2e on Windows,
not in Dexter's Host environment.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
CONSTRAINTS = ROOT / "neutts2e-constraints.txt"
WORKER = ROOT / "neutts2e_worker.py"
RUNTIME_ROOT = Path(
    os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(REPO_ROOT)
) / "Dexter" / "Voice" / "neutts2e"
OLD_BENCHMARK_RUNTIME = Path(
    os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(REPO_ROOT)
) / "DexterTTSBench" / "neutts"

PACKAGE_ARGS = (
    "neutts==1.4.1",
    "onnxruntime==1.29.0",
)


def _venv_python() -> Path:
    suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return RUNTIME_ROOT / ".venv" / suffix


def _hf_executable() -> Path:
    suffix = "Scripts/hf.exe" if os.name == "nt" else "bin/hf"
    return RUNTIME_ROOT / ".venv" / suffix


def _runtime_env() -> dict[str, str]:
    cache = RUNTIME_ROOT / "cache"
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


def _version(executable: Path) -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            [str(executable), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            check=True,
            capture_output=True,
            text=True,
        )
        major, minor = result.stdout.strip().split(".", 1)
        return int(major), int(minor)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _find_python313() -> Path:
    candidates: list[Path] = []
    direct = shutil.which("python3.13")
    if direct:
        candidates.append(Path(direct))

    if os.name == "nt":
        try:
            result = subprocess.run(
                ["py", "-3.13", "-c", "import sys; print(sys.executable)"],
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                candidates.append(Path(result.stdout.strip()))
        except (OSError, subprocess.CalledProcessError):
            pass

    candidates.append(Path(sys.executable))
    for candidate in candidates:
        if _version(candidate) == (3, 13):
            return candidate
    raise SystemExit(
        "NeuTTS-2E production runtime needs Python 3.13. "
        "Run this script with python3.13."
    )


def _migrate_benchmark_auth() -> bool:
    old_hf = OLD_BENCHMARK_RUNTIME / "cache" / "huggingface"
    new_hf = RUNTIME_ROOT / "cache" / "huggingface"
    if (new_hf / "token").exists() or not (old_hf / "token").exists():
        return False

    new_hf.mkdir(parents=True, exist_ok=True)
    copied = False
    for name in ("token", "stored_tokens"):
        source = old_hf / name
        if source.exists():
            shutil.copy2(source, new_hf / name)
            copied = True
    return copied


def setup() -> None:
    python = _venv_python()
    if python.exists():
        probe = subprocess.run(
            [str(python), "-c", "import neutts, onnxruntime"],
            env=_runtime_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            if _migrate_benchmark_auth():
                print("Migrated the existing benchmark Hugging Face login.")
            print(f"NeuTTS-2E runtime already ready: {RUNTIME_ROOT}")
            return
        print(f"Removing incomplete/outdated runtime: {RUNTIME_ROOT}")
        shutil.rmtree(RUNTIME_ROOT)

    base_python = _find_python313()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Creating production voice runtime with {base_python}")
    subprocess.run(
        [str(base_python), "-m", "venv", str(RUNTIME_ROOT / ".venv")],
        check=True,
    )

    if not CONSTRAINTS.exists():
        raise SystemExit(f"Missing constraints file: {CONSTRAINTS}")

    python = _venv_python()
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--constraint",
        str(CONSTRAINTS),
        *PACKAGE_ARGS,
    ]
    try:
        print("Installing pinned NeuTTS-2E CPU stack (pip cache disabled)...")
        subprocess.run(command, check=True, env=_runtime_env())
        subprocess.run(
            [str(python), "-c", "import neutts, onnxruntime"],
            check=True,
            env=_runtime_env(),
        )
    except Exception:
        print(f"Setup failed; deleting incomplete runtime: {RUNTIME_ROOT}")
        shutil.rmtree(RUNTIME_ROOT, ignore_errors=True)
        raise

    if _migrate_benchmark_auth():
        print("Migrated the existing benchmark Hugging Face login.")
    print(f"NeuTTS-2E production runtime ready: {RUNTIME_ROOT}")


def auth() -> None:
    executable = _hf_executable()
    if not executable.exists():
        raise SystemExit("Runtime is not installed. Run setup first.")
    subprocess.run([str(executable), "auth", "login"], check=True, env=_runtime_env())


def status() -> None:
    python = _venv_python()
    print(f"Runtime: {RUNTIME_ROOT}")
    print(f"Python:  {python if python.exists() else 'missing'}")
    token = RUNTIME_ROOT / "cache" / "huggingface" / "token"
    print(f"HF auth: {'present' if token.exists() else 'missing'}")
    if not python.exists():
        print("Stack:    not installed")
        return
    probe = subprocess.run(
        [str(python), "-c", "import neutts, onnxruntime"],
        env=_runtime_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Stack:    {'ready' if probe.returncode == 0 else 'broken'}")


def smoke() -> None:
    python = _venv_python()
    if not python.exists():
        raise SystemExit("Runtime is not installed. Run setup first.")
    if not WORKER.exists():
        raise SystemExit(f"Missing worker: {WORKER}")

    process = subprocess.Popen(
        [str(python), "-u", str(WORKER)],
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=_runtime_env(),
    )
    assert process.stdin is not None
    assert process.stdout is not None

    try:
        ready_line = process.stdout.readline()
        if not ready_line:
            raise SystemExit(f"Worker exited before startup completed (exit={process.poll()})")
        ready = json.loads(ready_line)
        if ready.get("type") != "ready":
            raise SystemExit(f"Worker failed to start: {ready}")
        print(
            f"Ready: Paul / PyTorch in {float(ready['load_seconds']):.2f}s. "
            "Measuring sentence synthesis latency..."
        )

        request_id = 1
        text = "Right, I've found the problem. It was using the old configuration file."
        started = time.perf_counter()
        process.stdin.write(json.dumps({"cmd": "synthesize", "id": request_id, "text": text}) + "\n")
        process.stdin.flush()

        first_audio: float | None = None
        audio_bytes = 0
        chunks = 0
        while True:
            line = process.stdout.readline()
            if not line:
                raise SystemExit(f"Worker exited during smoke test (exit={process.poll()})")
            message = json.loads(line)
            if message.get("type") == "audio":
                if first_audio is None:
                    first_audio = time.perf_counter() - started
                chunks += 1
                audio_bytes += len(base64.b64decode(message["pcm_f32le"]))
            elif message.get("type") == "done":
                total = time.perf_counter() - started
                audio_seconds = audio_bytes / 4 / 24_000
                rtf = total / audio_seconds if audio_seconds else float("inf")
                print(f"First audio: {first_audio:.3f}s" if first_audio is not None else "First audio: none")
                print(f"Generated:   {audio_seconds:.3f}s audio in {total:.3f}s ({chunks} chunk(s), RTF {rtf:.3f})")
                return
            elif message.get("type") in {"error", "fatal"}:
                raise SystemExit(f"Worker error: {message.get('message')}")
    finally:
        if process.poll() is None:
            try:
                process.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=5)
            except Exception:
                process.terminate()


def clean() -> None:
    if not RUNTIME_ROOT.exists():
        print("NeuTTS-2E production runtime is already absent.")
        return
    print(f"Deleting production voice runtime: {RUNTIME_ROOT}")
    shutil.rmtree(RUNTIME_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Dexter's NeuTTS-2E Paul runtime.")
    parser.add_argument("command", choices=("setup", "auth", "status", "smoke", "clean"))
    args = parser.parse_args()

    if args.command == "setup":
        setup()
    elif args.command == "auth":
        auth()
    elif args.command == "status":
        status()
    elif args.command == "smoke":
        smoke()
    elif args.command == "clean":
        clean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
