"""Validate and manage Dexter's NeuTTS-2E / Paul runtime.

Environment creation is intentionally not handled here. Dexter's root
pyproject.toml + uv.lock own the Python environment. This script manages voice
model authentication/cache and exercises the child-process protocol.
"""

from __future__ import annotations

import argparse
import base64
import importlib.metadata
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
WORKER = ROOT / "neutts2e_worker.py"
RUNTIME_ROOT = Path(
    os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(REPO_ROOT)
) / "Dexter" / "Voice" / "neutts2e"
OLD_BENCHMARK_RUNTIME = Path(
    os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(REPO_ROOT)
) / "DexterTTSBench" / "neutts"

EXPECTED_PACKAGES = (
    "agent-client-protocol",
    "moonshine-voice",
    "PySide6-Essentials",
    "neutts",
    "onnxruntime",
    "llama-cpp-python",
    "torch",
    "torchaudio",
)

MODEL_VARIANTS = {
    "q8": (
        "neuphonic/neutts-2e-q8-gguf",
        "neutts-2e-Q8_0.gguf",
        "https://huggingface.co/neuphonic/neutts-2e-q8-gguf",
    ),
    "q4": (
        "neuphonic/neutts-2e-q4-gguf",
        "neutts-2e-Q4_0.gguf",
        "https://huggingface.co/neuphonic/neutts-2e-q4-gguf",
    ),
}
CODEC_ACCESS = (
    "neuphonic/neucodec-onnx-decoder-int8",
    "model.onnx",
    "https://huggingface.co/neuphonic/neucodec-onnx-decoder-int8",
)
SAMPLE_RATE = 24_000


def runtime_env(variant: str | None = None) -> dict[str, str]:
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
    if variant is not None:
        env["DEXTER_TTS_VARIANT"] = variant
    return env


def _project_python_ok() -> bool:
    return sys.version_info[:2] == (3, 13)


def _hf_token_path() -> Path:
    return RUNTIME_ROOT / "cache" / "huggingface" / "token"


def _check_model_access(failures: list[str], variant: str) -> None:
    """Verify gated model access without downloading model payloads."""
    token_path = _hf_token_path()
    if not token_path.exists():
        failures.append(
            "Hugging Face login is missing. Run `uv run python "
            "Runtime\\Voice\\neutts2e_runtime.py auth`."
        )
        return

    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        failures.append("Hugging Face token file is empty.")
        return

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except Exception as exc:
        failures.append(f"Could not import Hugging Face access checker: {exc}")
        return

    cache_dir = RUNTIME_ROOT / "cache" / "huggingface" / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for repo_id, filename, access_url in (MODEL_VARIANTS[variant], CODEC_ACCESS):
        try:
            info = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                cache_dir=cache_dir,
                dry_run=True,
            )
        except GatedRepoError:
            failures.append(
                f"No Hugging Face file access for {repo_id}. "
                f"Accept the model terms at {access_url}"
            )
        except HfHubHTTPError as exc:
            failures.append(
                f"Hugging Face access check failed for {repo_id}: "
                f"HTTP {getattr(exc.response, 'status_code', '?')}"
            )
        except Exception as exc:
            failures.append(
                f"Hugging Face access check failed for {repo_id}: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            size = getattr(info, "file_size", None)
            suffix = f" ({size / (1024 * 1024):.1f} MB)" if size else ""
            print(f"HF model access:         OK  {repo_id}{suffix}")


def doctor(variant: str = "q8") -> None:
    failures: list[str] = []

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"TTS variant: {variant.upper()}")
    if not _project_python_ok():
        failures.append("Dexter must run from the uv-managed Python 3.13 environment.")

    expected_venv = (REPO_ROOT / ".venv").resolve()
    executable = Path(sys.executable).resolve()
    try:
        executable.relative_to(expected_venv)
    except ValueError:
        failures.append(
            f"Interpreter is not inside Dexter's root .venv ({expected_venv}). "
            "Use `uv run` after `uv sync`."
        )

    for package in EXPECTED_PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"Missing package: {package}")
        else:
            print(f"{package:23} {version}")

    if not failures:
        try:
            import llama_cpp  # noqa: F401
            import neutts  # noqa: F401
            import onnxruntime  # noqa: F401
        except Exception as exc:
            failures.append(f"Native/runtime import check failed: {exc}")

    token = _hf_token_path()
    print(f"HF auth: {'present' if token.exists() else 'missing'}")
    print(f"Voice data: {RUNTIME_ROOT}")

    if not failures:
        _check_model_access(failures, variant)

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("Runtime doctor: OK")


def migrate_auth() -> None:
    old_hf = OLD_BENCHMARK_RUNTIME / "cache" / "huggingface"
    new_hf = RUNTIME_ROOT / "cache" / "huggingface"
    if (new_hf / "token").exists():
        print("Production Hugging Face login already exists.")
        return
    if not (old_hf / "token").exists():
        print("No old benchmark Hugging Face login found.")
        return

    new_hf.mkdir(parents=True, exist_ok=True)
    for name in ("token", "stored_tokens"):
        source = old_hf / name
        if source.exists():
            shutil.copy2(source, new_hf / name)
    print("Migrated benchmark Hugging Face login into Dexter voice data.")


def auth() -> None:
    suffix = "hf.exe" if os.name == "nt" else "hf"
    executable = Path(sys.executable).with_name(suffix)
    if not executable.exists():
        raise SystemExit("The Hugging Face CLI is not installed in Dexter's environment.")
    subprocess.run([str(executable), "auth", "login"], check=True, env=runtime_env())


class _ProtocolReader:
    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        self.messages: queue.Queue[dict[str, object] | BaseException | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    self.messages.put(RuntimeError(f"Invalid worker JSON: {line!r}"))
                    return
                if not isinstance(value, dict):
                    self.messages.put(RuntimeError(f"Invalid worker payload: {value!r}"))
                    return
                self.messages.put(value)
        except BaseException as exc:
            self.messages.put(exc)
        finally:
            self.messages.put(None)

    def get(self, timeout: float) -> dict[str, object]:
        try:
            value = self.messages.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("Timed out waiting for NeuTTS-2E worker.") from exc
        if value is None:
            raise RuntimeError(
                f"NeuTTS-2E worker closed stdout (exit={self.process.poll()})."
            )
        if isinstance(value, BaseException):
            raise value
        return value


def _start_worker(variant: str) -> tuple[subprocess.Popen[str], _ProtocolReader]:
    if not WORKER.exists():
        raise SystemExit(f"Missing worker: {WORKER}")
    process = subprocess.Popen(
        [sys.executable, "-u", str(WORKER)],
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=runtime_env(variant),
    )
    return process, _ProtocolReader(process)


def _send(process: subprocess.Popen[str], payload: dict[str, object]) -> None:
    assert process.stdin is not None
    process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
    process.stdin.flush()


def _shutdown(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        _send(process, {"cmd": "shutdown"})
        process.wait(timeout=5)
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


def _read_stream_request(
    process: subprocess.Popen[str],
    reader: _ProtocolReader,
    *,
    request_id: int,
    text: str,
    timeout: float = 120,
    print_chunks: bool = False,
) -> tuple[list[np.ndarray], float | None, float, list[dict[str, float]]]:
    started = time.perf_counter()
    last_arrival = started
    first_audio: float | None = None
    chunks: list[np.ndarray] = []
    timings: list[dict[str, float]] = []

    _send(process, {"cmd": "synthesize", "id": request_id, "text": text})

    while True:
        message = reader.get(timeout=timeout)
        if message.get("id") not in (None, request_id):
            raise SystemExit(f"Protocol desync: {message}")
        kind = message.get("type")
        if kind == "audio":
            now = time.perf_counter()
            arrival_delta = now - last_arrival
            if first_audio is None:
                first_audio = now - started
            raw = base64.b64decode(str(message["pcm_f32le"]))
            samples = np.frombuffer(raw, dtype="<f4").copy()
            duration = samples.size / SAMPLE_RATE
            chunk_number = len(chunks) + 1
            timings.append(
                {
                    "chunk": float(chunk_number),
                    "arrival_delta_seconds": arrival_delta,
                    "audio_seconds": duration,
                    "arrival_minus_audio_seconds": arrival_delta - duration,
                }
            )
            chunks.append(samples)
            if print_chunks:
                label = "TTFA" if chunk_number == 1 else "gap"
                print(
                    f"Chunk {chunk_number:2d}: {label}={arrival_delta:6.3f}s  "
                    f"audio={duration:6.3f}s  "
                    f"arrival-audio={arrival_delta - duration:+6.3f}s"
                )
            last_arrival = now
        elif kind == "done":
            total = time.perf_counter() - started
            return chunks, first_audio, total, timings
        elif kind in {"error", "fatal"}:
            raise SystemExit(f"Worker error: {message.get('message')}")


def smoke(variant: str = "q8") -> None:
    doctor(variant)
    process, reader = _start_worker(variant)
    try:
        ready = reader.get(timeout=180)
        if ready.get("type") != "ready":
            raise SystemExit(f"Worker failed to start: {ready}")

        print(
            f"Ready: Paul / {variant.upper()} GGUF in "
            f"{float(ready['load_seconds']):.2f}s. Measuring streamed synthesis..."
        )

        text = "Right, I've found the problem. It was using the old configuration file."
        chunks, first_audio, total, _ = _read_stream_request(
            process,
            reader,
            request_id=1,
            text=text,
        )
        audio_samples = sum(chunk.size for chunk in chunks)
        audio_seconds = audio_samples / SAMPLE_RATE
        rtf = total / audio_seconds if audio_seconds else float("inf")
        print(
            f"First audio: {first_audio:.3f}s"
            if first_audio is not None
            else "First audio: none"
        )
        print(
            f"Generated:   {audio_seconds:.3f}s audio in {total:.3f}s "
            f"({len(chunks)} chunk(s), RTF {rtf:.3f})"
        )
        if first_audio is None or len(chunks) < 2:
            raise SystemExit("Streaming check failed: expected multiple audio chunks.")
        if rtf >= 1.0:
            print(
                "WARNING: RTF is >= 1.0; continuous live playback may starve "
                "without additional buffering or a faster backend."
            )

        request_id = 2
        _send(
            process,
            {
                "cmd": "synthesize",
                "id": request_id,
                "text": (
                    "This is a deliberately longer cancellation test sentence, "
                    "so Dexter can prove that speech can be interrupted without "
                    "waiting for the complete utterance to finish."
                ),
            },
        )
        sent_cancel = False
        while True:
            message = reader.get(timeout=90)
            if message.get("id") not in (None, request_id):
                raise SystemExit(f"Protocol desync: {message}")
            kind = message.get("type")
            if kind == "audio" and not sent_cancel:
                _send(process, {"cmd": "cancel", "id": request_id})
                sent_cancel = True
            elif kind == "done":
                if not sent_cancel or not bool(message.get("cancelled")):
                    raise SystemExit(f"Cancellation check failed: {message}")
                print("Cancellation: OK")
                break
            elif kind in {"error", "fatal"}:
                raise SystemExit(f"Worker error: {message.get('message')}")

        print(f"NeuTTS-2E {variant.upper()} smoke test: OK")
    finally:
        _shutdown(process)


def capture(variant: str = "q8") -> None:
    """Capture streamed PCM without real-time playback to isolate audio defects."""
    doctor(variant)
    process, reader = _start_worker(variant)
    diagnostics = RUNTIME_ROOT / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    wav_path = diagnostics / f"paul-{variant}-stream-{stamp}.wav"
    json_path = diagnostics / f"paul-{variant}-stream-{stamp}.json"

    text = (
        "Right, I've found the problem. It was using the old configuration file. "
        "The important question now is whether these pauses are present in the "
        "generated waveform itself, or only appear while Dexter plays the stream "
        "in real time. This diagnostic records the exact streamed audio without "
        "playing it, so we can separate synthesis artifacts from playback starvation."
    )

    try:
        ready = reader.get(timeout=180)
        if ready.get("type") != "ready":
            raise SystemExit(f"Worker failed to start: {ready}")
        print(
            f"Ready: Paul / {variant.upper()} GGUF in "
            f"{float(ready['load_seconds']):.2f}s. Capturing raw stream..."
        )

        chunks, first_audio, total, timings = _read_stream_request(
            process,
            reader,
            request_id=1,
            text=text,
            timeout=180,
            print_chunks=True,
        )
        if not chunks:
            raise SystemExit("Capture produced no audio.")

        combined = np.concatenate(chunks).astype(np.float32, copy=False)
        clipped = np.clip(combined, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype("<i2")
        with wave.open(str(wav_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(pcm16.tobytes())

        audio_seconds = combined.size / SAMPLE_RATE
        rtf = total / audio_seconds if audio_seconds else float("inf")
        result = {
            "variant": variant,
            "speaker": "paul",
            "sample_rate": SAMPLE_RATE,
            "text": text,
            "load_seconds": float(ready["load_seconds"]),
            "first_audio_seconds": first_audio,
            "generation_seconds": total,
            "audio_seconds": audio_seconds,
            "rtf": rtf,
            "chunks": len(chunks),
            "chunk_timings": timings,
            "wav": str(wav_path),
        }
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        worst = max(
            (item["arrival_minus_audio_seconds"] for item in timings[1:]),
            default=0.0,
        )
        print("")
        print(f"Captured: {audio_seconds:.3f}s audio in {total:.3f}s (RTF {rtf:.3f})")
        print(f"Worst post-first-chunk arrival deficit: {worst:+.3f}s")
        print(f"WAV:    {wav_path}")
        print(f"Timing: {json_path}")
        print("")
        print(
            "Listen to the WAV. If it is choppy too, the defect is already in "
            "NeuTTS streaming/codec output. If the WAV is smooth, Dexter's live "
            "playback path is introducing the choppiness."
        )
    finally:
        _shutdown(process)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Dexter's NeuTTS-2E / Paul runtime."
    )
    parser.add_argument(
        "command",
        choices=("doctor", "migrate-auth", "auth", "smoke", "capture"),
    )
    parser.add_argument(
        "--variant",
        choices=tuple(MODEL_VARIANTS),
        default="q8",
        help="NeuTTS-2E GGUF quantization to validate or test.",
    )
    args = parser.parse_args()

    if args.command == "doctor":
        doctor(args.variant)
    elif args.command == "migrate-auth":
        migrate_auth()
    elif args.command == "auth":
        auth()
    elif args.command == "smoke":
        smoke(args.variant)
    elif args.command == "capture":
        capture(args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
