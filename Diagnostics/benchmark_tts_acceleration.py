"""Compare NeuTTS-2E Q8 backbone execution paths.

This benchmark keeps the voice, Q8 model, text, ONNX NeuCodec decoder, and
worker protocol fixed. Only the GGUF backbone execution changes.

Cases:
- CPU / NeuTTS default thread heuristic
- CPU / explicit 16 generation threads
- Intel Arc candidate / Vulkan visible device 0
- Optional RTX control / Vulkan visible device 1

GPU cases refuse to report a timing as an accelerator result unless the loaded
llama.cpp library reports GPU-offload support and the ggml backend registry
shows the expected accelerator device.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_text = str(REPO_ROOT)
if repo_root_text not in sys.path:
    sys.path.insert(0, repo_root_text)

from Runtime.Voice import neutts2e_runtime as runtime


@dataclass(frozen=True)
class Case:
    key: str
    label: str
    device: str
    threads: int | None = None
    vulkan_device: str | None = None
    expected_device_text: str | None = None
    forbidden_device_text: str | None = None


TEXT = (
    "Right, I've found the problem. It was using the old configuration file. "
    "This benchmark keeps the speech model, speaker, codec, and text identical "
    "while changing only where the language model backbone performs its work. "
    "The result will tell us whether Dexter should use more CPU cores or move "
    "Paul's backbone onto an available graphics processor."
)

CASES = (
    Case("cpu-default", "CPU: NeuTTS default thread heuristic", "cpu"),
    Case("cpu16", "CPU: 16 generation threads", "cpu", threads=16),
    Case(
        "arc",
        "Intel Arc candidate: Vulkan device 0",
        "gpu",
        vulkan_device="0",
        expected_device_text="arc",
        forbidden_device_text="5050",
    ),
    Case(
        "rtx",
        "RTX control: Vulkan device 1",
        "gpu",
        vulkan_device="1",
        expected_device_text="5050",
    ),
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


def _device_summary(ready: dict[str, object]) -> str:
    devices = ready.get("backend_devices")
    if not isinstance(devices, list):
        return "none reported"
    lines = []
    for item in devices:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"[{item.get('index')}] {item.get('backend')}/{item.get('type')}: "
            f"{item.get('name')} — {item.get('description')}"
        )
    return "\n".join(lines) if lines else "none reported"


def _verify_gpu_ready(case: Case, ready: dict[str, object]) -> None:
    if case.device != "gpu":
        return

    supported = bool(ready.get("gpu_offload_supported"))
    requested_layers = int(ready.get("requested_gpu_layers") or 0)
    devices = ready.get("backend_devices")
    device_error = ready.get("backend_device_error")

    if not supported:
        raise RuntimeError(
            "This installed llama-cpp-python library reports GPU offload support = false. "
            "The GPU timing would actually be a CPU fallback."
        )
    if requested_layers <= 0:
        raise RuntimeError(
            f"NeuTTS did not request GPU layer offload (n_gpu_layers={requested_layers})."
        )
    if device_error:
        raise RuntimeError(
            "Could not enumerate ggml backend devices from this llama.cpp build: "
            f"{device_error}"
        )
    if not isinstance(devices, list) or not devices:
        raise RuntimeError("llama.cpp reported GPU support but no backend devices were enumerated.")

    accelerator_texts: list[str] = []
    for item in devices:
        if not isinstance(item, dict):
            continue
        device_type = str(item.get("type") or "").lower()
        if device_type not in {"gpu", "igpu"}:
            continue
        accelerator_texts.append(
            " ".join(
                str(item.get(field) or "")
                for field in ("backend", "name", "description")
            ).lower()
        )

    if not accelerator_texts:
        raise RuntimeError(
            "GPU offload is supported, but the ggml registry exposes no GPU/iGPU device.\n"
            + _device_summary(ready)
        )

    combined = "\n".join(accelerator_texts)
    if case.expected_device_text and case.expected_device_text.lower() not in combined:
        raise RuntimeError(
            f"Expected accelerator containing {case.expected_device_text!r}, but ggml exposed:\n"
            + _device_summary(ready)
        )
    if case.forbidden_device_text and case.forbidden_device_text.lower() in combined:
        raise RuntimeError(
            f"Forbidden accelerator containing {case.forbidden_device_text!r} is visible; "
            "do not benchmark TTS against the RTX used by Ollama.\n"
            + _device_summary(ready)
        )


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
            f"threads={ready.get('actual_threads')}; "
            f"gpu_offload_supported={ready.get('gpu_offload_supported')}; "
            f"requested_gpu_layers={ready.get('requested_gpu_layers')}"
        )
        print("ggml backend devices:")
        print(_device_summary(ready))
        if ready.get("backend_device_error"):
            print(f"backend enumeration error: {ready.get('backend_device_error')}")
        print(f"llama.cpp feature info: {ready.get('llama_system_info')}")

        _verify_gpu_ready(case, ready)

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("all", "cpu-default", "cpu16", "arc", "rtx"),
        default="all",
        help="Run only one execution path. RTX is opt-in unless selected explicitly.",
    )
    args = parser.parse_args()

    runtime.doctor("q8")
    print("")
    print(
        "GPU results are accepted only if llama.cpp reports offload support and "
        "the ggml registry exposes the expected accelerator."
    )

    if args.case == "all":
        selected = tuple(case for case in CASES if case.key != "rtx")
    else:
        selected = tuple(case for case in CASES if case.key == args.case)

    results = []
    for case in selected:
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
            "No selected path is real-time yet. Do not solve that with a larger "
            "permanent playback buffer; verify/compare the next acceleration backend."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
