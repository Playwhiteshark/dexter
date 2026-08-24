"""Profile Dexter NeuTTS streaming stages without live playback.

The diagnostic can run the production CPU path or the verified Intel Arc
Vulkan path. It separates codec decode time from llama-cpp-python hot-loop
work: native eval, sampling, tokenization/detokenization, and remaining Python /
streaming orchestration.
"""

from __future__ import annotations

import argparse
import base64
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Runtime.Voice import neutts2e_runtime as runtime


def _start(device: str):
    env = runtime.runtime_env("q8")
    env["DEXTER_TTS_PROFILE_STAGES"] = "1"
    env["DEXTER_TTS_PROFILE_LLAMA_STAGES"] = "1"
    if device == "arc":
        env["DEXTER_TTS_BACKBONE_DEVICE"] = "gpu"
        env["DEXTER_TTS_VULKAN_DEVICE"] = "0"
        env["GGML_VK_VISIBLE_DEVICES"] = "0"
    else:
        env["DEXTER_TTS_BACKBONE_DEVICE"] = "cpu"
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


def _seconds(message: dict[str, object], name: str) -> float:
    return float(message.get(name) or 0.0)


def _calls(message: dict[str, object], name: str) -> int:
    return int(message.get(name) or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "arc"),
        default="cpu",
        help="Backbone execution path to profile. Arc uses Vulkan visible device 0.",
    )
    args = parser.parse_args()

    runtime.doctor("q8")
    process, reader = _start(args.device)
    try:
        ready = reader.get(timeout=180)
        if ready.get("type") != "ready":
            raise SystemExit(f"Worker failed to start: {ready}")

        print(
            f"Ready: Paul / Q8 in {float(ready['load_seconds']):.2f}s; "
            f"backbone={ready.get('backbone_device')}; "
            f"threads={ready.get('actual_threads')}; "
            f"gpu_offload_supported={ready.get('gpu_offload_supported')}."
        )

        text = (
            "Right, I've found the problem. It was using the old configuration file. "
            "This profile measures the model evaluation, token sampling, Python "
            "streaming loop, and audio reconstruction separately so Dexter can "
            "optimize the part that is actually limiting real-time speech."
        )
        request_id = 1
        started = time.perf_counter()
        last_arrival = started
        audio_samples = 0
        chunk_count = 0
        first_audio = None

        runtime._send(
            process,
            {"cmd": "synthesize", "id": request_id, "text": text},
        )

        while True:
            message = reader.get(timeout=180)
            if message.get("id") not in (None, request_id):
                raise SystemExit(f"Protocol desync: {message}")

            kind = message.get("type")
            if kind == "audio":
                now = time.perf_counter()
                raw = base64.b64decode(str(message["pcm_f32le"]))
                samples = len(raw) // 4
                audio_seconds = samples / runtime.SAMPLE_RATE
                arrival = now - last_arrival
                chunk_count += 1
                if first_audio is None:
                    first_audio = now - started
                print(
                    f"Chunk {chunk_count:2d}: arrived in {arrival:6.3f}s; "
                    f"contains {audio_seconds:6.3f}s audio; "
                    f"deficit {arrival - audio_seconds:+6.3f}s"
                )
                audio_samples += samples
                last_arrival = now
                continue

            if kind in {"error", "fatal"}:
                raise SystemExit(f"Worker error: {message.get('message')}")
            if kind != "done":
                continue

            total = time.perf_counter() - started
            audio_seconds = audio_samples / runtime.SAMPLE_RATE
            rtf = total / audio_seconds if audio_seconds else float("inf")

            decode = _seconds(message, "decode_seconds")
            eval_seconds = _seconds(message, "eval_seconds")
            sample = _seconds(message, "sample_seconds")
            tokenize = _seconds(message, "tokenize_seconds")
            detokenize = _seconds(message, "detokenize_seconds")
            measured = decode + eval_seconds + sample + tokenize + detokenize
            remainder = max(0.0, total - measured)

            def pct(value: float) -> float:
                return value / total * 100.0 if total else 0.0

            print("")
            print(f"First audio:                  {first_audio:.3f}s")
            print(f"Generated audio:              {audio_seconds:.3f}s")
            print(f"Total synthesis:              {total:.3f}s  (RTF {rtf:.3f})")
            print("")
            print(
                f"llama eval:                   {eval_seconds:.3f}s  "
                f"({pct(eval_seconds):5.1f}%, {_calls(message, 'eval_calls')} calls)"
            )
            print(
                f"llama sampling:               {sample:.3f}s  "
                f"({pct(sample):5.1f}%, {_calls(message, 'sample_calls')} calls)"
            )
            print(
                f"llama tokenize:               {tokenize:.3f}s  "
                f"({pct(tokenize):5.1f}%, {_calls(message, 'tokenize_calls')} calls)"
            )
            print(
                f"llama detokenize:             {detokenize:.3f}s  "
                f"({pct(detokenize):5.1f}%, {_calls(message, 'detokenize_calls')} calls)"
            )
            print(
                f"NeuCodec decode:              {decode:.3f}s  "
                f"({pct(decode):5.1f}%, {_calls(message, 'decode_calls')} calls)"
            )
            print(
                f"Python/streaming remainder:   {remainder:.3f}s  "
                f"({pct(remainder):5.1f}%)"
            )
            print("")
            print(
                "The remainder includes NeuTTS chunk bookkeeping, overlap-add / "
                "watermark work, llama-cpp-python completion formatting, protocol "
                "serialization, and timing overhead not inside the wrapped calls."
            )
            return 0
    finally:
        runtime._shutdown(process)


if __name__ == "__main__":
    raise SystemExit(main())
