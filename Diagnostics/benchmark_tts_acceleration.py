"""Compare NeuTTS-2E Q8 execution paths.

This benchmark keeps the voice, Q8 model, text, ONNX NeuCodec decoder, Perth
watermarking, and worker protocol fixed. GPU cases use a diagnostic-only worker
that forces llama.cpp verbosity, then verify the native model-load log itself.
A result is accepted as GPU-backed only when llama.cpp names the expected
accelerator and reports model layers actually offloaded to GPU.

The ``arc-pipelined`` case changes only scheduling: Arc token generation runs in
a producer thread while the main thread performs the same ONNX decode, Perth
watermarking, crop, and overlap-add operations in order.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_text = str(REPO_ROOT)
if repo_root_text not in sys.path:
    sys.path.insert(0, repo_root_text)

from Runtime.Voice import neutts2e_runtime as runtime

VERBOSE_WORKER = REPO_ROOT / "Diagnostics" / "neutts2e_verbose_worker.py"


@dataclass(frozen=True)
class Case:
    key: str
    label: str
    device: str
    threads: int | None = None
    vulkan_device: str | None = None
    expected_device_text: str | None = None
    forbidden_device_text: str | None = None
    pipeline_watermark: bool = False


TEXT = (
    "Right, I've found the problem. It was using the old configuration file. "
    "This benchmark keeps the speech model, speaker, codec, watermark, and text "
    "identical while changing only where work runs or whether independent stages "
    "are scheduled concurrently."
)

CASES = (
    Case("cpu-default", "CPU: NeuTTS default thread heuristic", "cpu"),
    Case("cpu16", "CPU: 16 generation threads", "cpu", threads=16),
    Case(
        "arc",
        "Intel Arc: Vulkan device 0 + serial Perth",
        "gpu",
        vulkan_device="0",
        expected_device_text="arc",
        forbidden_device_text="5050",
    ),
    Case(
        "arc-pipelined",
        "Intel Arc: Vulkan device 0 + pipelined Perth",
        "gpu",
        vulkan_device="0",
        expected_device_text="arc",
        forbidden_device_text="5050",
        pipeline_watermark=True,
    ),
    Case(
        "rtx",
        "RTX control: Vulkan device 1",
        "gpu",
        vulkan_device="1",
        expected_device_text="5050",
    ),
)


class _StderrCapture:
    """Tee a child process's native stderr to our stderr while retaining it."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            with self._lock:
                self._lines.append(line)
            sys.stderr.write(line)
            sys.stderr.flush()

    def text(self) -> str:
        with self._lock:
            return "".join(self._lines)

    def join(self, timeout: float = 1.0) -> None:
        self._thread.join(timeout=timeout)


def _start(case: Case):
    env = runtime.runtime_env("q8")
    env["DEXTER_TTS_BACKBONE_DEVICE"] = case.device
    if case.threads is not None:
        env["DEXTER_TTS_THREADS"] = str(case.threads)
    else:
        env.pop("DEXTER_TTS_THREADS", None)

    if case.vulkan_device is not None:
        env["DEXTER_TTS_VULKAN_DEVICE"] = case.vulkan_device
        env["GGML_VK_VISIBLE_DEVICES"] = case.vulkan_device
    else:
        env.pop("DEXTER_TTS_VULKAN_DEVICE", None)
        env.pop("GGML_VK_VISIBLE_DEVICES", None)

    if case.pipeline_watermark:
        env["DEXTER_TTS_PIPELINE_WATERMARK"] = "1"
    else:
        env.pop("DEXTER_TTS_PIPELINE_WATERMARK", None)

    worker = VERBOSE_WORKER if case.device == "gpu" else runtime.WORKER
    process = subprocess.Popen(
        [sys.executable, "-u", str(worker)],
        cwd=str(runtime.REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    return process, runtime._ProtocolReader(process), _StderrCapture(process)


def _offload_evidence(stderr_text: str) -> list[str]:
    evidence = []
    for line in stderr_text.splitlines():
        low = line.lower()
        if (
            "vulkan" in low
            or "using device" in low
            or "offload" in low
            or "gpu model buffer" in low
            or "model buffer size" in low and "vulkan" in low
        ):
            evidence.append(line.strip())
    return evidence


def _verify_gpu_ready(
    case: Case,
    ready: dict[str, object],
    stderr_text: str,
) -> None:
    if case.device != "gpu":
        return

    if not bool(ready.get("gpu_offload_supported")):
        raise RuntimeError(
            "This installed llama-cpp-python library reports GPU offload support = false. "
            "The GPU timing would actually be a CPU fallback."
        )

    requested_layers = int(ready.get("requested_gpu_layers") or 0)
    if requested_layers <= 0:
        raise RuntimeError(
            f"NeuTTS did not request GPU layer offload (n_gpu_layers={requested_layers})."
        )

    low = stderr_text.lower()
    if case.expected_device_text and case.expected_device_text.lower() not in low:
        raise RuntimeError(
            f"llama.cpp did not name the expected accelerator containing "
            f"{case.expected_device_text!r}. Native load evidence:\n"
            + "\n".join(_offload_evidence(stderr_text))
        )
    if case.forbidden_device_text and case.forbidden_device_text.lower() in low:
        raise RuntimeError(
            f"llama.cpp named forbidden accelerator text {case.forbidden_device_text!r}; "
            "do not benchmark TTS against the RTX used by Ollama."
        )

    if "vulkan" not in low:
        raise RuntimeError(
            "GPU support is present, but the verbose native load log contains no Vulkan "
            "device/backend evidence."
        )

    offload_match = re.search(
        r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+gpu",
        low,
    )
    if offload_match is None:
        raise RuntimeError(
            "llama.cpp named the accelerator but did not report model layers actually "
            "offloaded to GPU. Native load evidence:\n"
            + "\n".join(_offload_evidence(stderr_text))
        )

    offloaded = int(offload_match.group(1))
    total = int(offload_match.group(2))
    if offloaded <= 0 or total <= 0:
        raise RuntimeError(
            f"Invalid llama.cpp offload report: {offloaded}/{total} layers."
        )

    print("Verified native llama.cpp offload evidence:")
    for line in _offload_evidence(stderr_text):
        print(f"  {line}")


def _run(case: Case) -> dict[str, float | str | int | None | bool]:
    print("")
    print("=" * 72)
    print(case.label)
    print("=" * 72)
    process, reader, stderr_capture = _start(case)
    try:
        ready = reader.get(timeout=180)
        if ready.get("type") != "ready":
            raise RuntimeError(f"Worker failed to start: {ready}")

        # All model-load logs precede the ready protocol message. Give the tee
        # thread one scheduling turn to consume any final buffered stderr line.
        time.sleep(0.05)

        print(
            f"Ready in {float(ready['load_seconds']):.2f}s; "
            f"backbone={ready.get('backbone_device')}; "
            f"vulkan_device={ready.get('vulkan_device')}; "
            f"threads={ready.get('actual_threads')}; "
            f"Perth=on; pipeline={'on' if case.pipeline_watermark else 'off'}; "
            f"gpu_offload_supported={ready.get('gpu_offload_supported')}; "
            f"requested_gpu_layers={ready.get('requested_gpu_layers')}"
        )

        _verify_gpu_ready(case, ready, stderr_capture.text())

        chunks, first_audio, total, timings = runtime._read_stream_request(
            process,
            reader,
            request_id=1,
            text=TEXT,
            timeout=180,
            print_chunks=False,
        )
        audio_samples = sum(chunk.size for chunk in chunks)
        audio_seconds = audio_samples / runtime.SAMPLE_RATE
        rtf = total / audio_seconds if audio_seconds else float("inf")
        deficits = [item["arrival_minus_audio_seconds"] for item in timings[1:]]
        average_deficit = sum(deficits) / len(deficits) if deficits else 0.0
        worst_deficit = max(deficits, default=0.0)

        print(f"First audio:           {first_audio:.3f}s")
        print(f"Generated audio:       {audio_seconds:.3f}s")
        print(f"Total synthesis:       {total:.3f}s")
        print(f"RTF:                   {rtf:.3f}")
        print(f"Average chunk deficit: {average_deficit:+.3f}s")
        print(f"Worst chunk deficit:   {worst_deficit:+.3f}s")

        return {
            "key": case.key,
            "label": case.label,
            "pipeline_watermark": case.pipeline_watermark,
            "load_seconds": float(ready["load_seconds"]),
            "first_audio_seconds": first_audio,
            "audio_seconds": audio_seconds,
            "total_seconds": total,
            "rtf": rtf,
            "average_chunk_deficit": average_deficit,
            "worst_chunk_deficit": worst_deficit,
            "chunks": len(chunks),
        }
    finally:
        runtime._shutdown(process)
        stderr_capture.join()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("all", "cpu-default", "cpu16", "arc", "arc-pipelined", "rtx"),
        default="all",
        help="Run one execution path. RTX remains opt-in.",
    )
    args = parser.parse_args()

    runtime.doctor("q8")
    print("")
    print(
        "GPU timings are accepted only after native llama.cpp load logs prove the "
        "expected device and actual layer offload. Perth remains enabled."
    )

    if args.case == "all":
        selected = tuple(case for case in CASES if case.key != "rtx")
    else:
        selected = tuple(case for case in CASES if case.key == args.case)

    results = [_run(case) for case in selected]

    print("")
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for item in results:
        marker = "REAL-TIME" if float(item["rtf"]) < 1.0 else "TOO SLOW"
        print(
            f"{item['label']}: RTF {float(item['rtf']):.3f}, "
            f"first audio {float(item['first_audio_seconds']):.3f}s — {marker}"
        )

    best = min(results, key=lambda item: float(item["rtf"]))
    print("")
    print(f"Fastest measured path: {best['label']} (RTF {float(best['rtf']):.3f})")
    if float(best["rtf"]) >= 1.0:
        print(
            "No selected path is real-time yet. Do not solve that with a larger "
            "permanent playback buffer; compare the next measured optimization."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
