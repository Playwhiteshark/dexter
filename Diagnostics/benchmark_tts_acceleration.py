"""Compare NeuTTS-2E Q8 backbone execution on CPU and Intel Arc Vulkan.

This benchmark keeps the voice, Q8 model, text, ONNX NeuCodec decoder, and
worker protocol fixed. Only the GGUF backbone execution changes.

Cases:
- CPU / NeuTTS default thread heuristic (8 threads on the Core Ultra 7 255H)
- CPU / explicit 16 generation threads
- GPU / Vulkan device 0 (expected Intel Arc 140T on this machine)

The Vulkan worker writes llama.cpp device discovery to stderr. Verify that the
GPU case names Intel Arc 140T; abort the interpretation if it names the RTX.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# When invoked as `python Diagnostics\\benchmark_tts_acceleration.py`, Python
# puts Diagnostics/ rather than the repository root on sys.path. Add the repo
# root explicitly before importing Dexter modules so this diagnostic works from
# the documented command regardless of the caller's PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_text = str(REPO_ROOT)
if repo_root_text not in sys.path:
    sys.path.insert(0, repo_root_text)

from Runtime.Voice import neutts2e_runtime as runtime


@dataclass(frozen=True)
class Case:
    label: str
    device: str
    threads: int | None = None
    vulkan_device: str | None = None


TEXT = (
    "Right, I've found the problem. It was using the old configuration file. "
    "This benchmark keeps the speech model, speaker, codec, and text identical "
    "while changing only where the language model backbone performs its work. "
    "The result will tell us whether Dexter should use more CPU cores or move "
    "Paul's backbone onto the Intel Arc graphics processor."
)


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

    process = subprocess.Popen(
        [sys.executable, "-u", str(runtime.WORKER)],
        cwd=str(runtime.REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    return process, runtime._ProtocolReader(process)


def _run(case: Case) -> dict[str, float | str | int | None]:
    print("")
    print("=" * 72)
    print(case.label)
    print("=" * 72)
    process, reader = _start(case)
    try:
        ready = reader.get(timeout=180)
        if ready.get("type") != "ready":
            raise RuntimeError(f"Worker failed to start: {ready}")

        print(
            f"Ready in {float(ready['load_seconds']):.2f}s; "
            f"backbone={ready.get('backbone_device')}; "
            f"vulkan_device={ready.get('vulkan_device')}; "
            f"threads={ready.get('threads')}"
        )

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
        deficits = [
            item["arrival_minus_audio_seconds"]
            for item in timings[1:]
        ]
        average_deficit = sum(deficits) / len(deficits) if deficits else 0.0
        worst_deficit = max(deficits, default=0.0)

        print(f"First audio:           {first_audio:.3f}s")
        print(f"Generated audio:       {audio_seconds:.3f}s")
        print(f"Total synthesis:       {total:.3f}s")
        print(f"RTF:                   {rtf:.3f}")
        print(f"Average chunk deficit: {average_deficit:+.3f}s")
        print(f"Worst chunk deficit:   {worst_deficit:+.3f}s")

        return {
            "label": case.label,
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


def main() -> int:
    runtime.doctor("q8")
    print("")
    print("IMPORTANT: During the GPU case, llama.cpp should print an Intel Arc")
    print("140T Vulkan device to stderr. If it names the RTX 5050 instead, stop")
    print("the benchmark with Ctrl+C and do not use that result.")

    cases = (
        Case("CPU: NeuTTS default thread heuristic", "cpu"),
        Case("CPU: 16 generation threads", "cpu", threads=16),
        Case("Intel Arc candidate: Vulkan device 0", "gpu", vulkan_device="0"),
    )

    results = []
    for case in cases:
        results.append(_run(case))

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
            "No tested path is real-time yet. Do not solve that with a larger "
            "permanent playback buffer; test the next acceleration backend."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
