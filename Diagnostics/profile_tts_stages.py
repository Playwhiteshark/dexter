"""Profile Dexter NeuTTS streaming stages without live playback.

This diagnostic uses the production worker/protocol but never opens an audio
output device. The worker reports cumulative NeuCodec decode time separately
from total synthesis time so we can estimate how much latency is attributable
to the GGUF/token-generation side versus audio reconstruction and other work.
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Runtime.Voice import neutts2e_runtime as runtime


def main() -> int:
    variant = "q8"
    runtime.doctor(variant)
    process, reader = runtime._start_worker(variant)
    try:
        ready = reader.get(timeout=180)
        if ready.get("type") != "ready":
            raise SystemExit(f"Worker failed to start: {ready}")

        print(
            f"Ready: Paul / Q8 in {float(ready['load_seconds']):.2f}s. "
            "Profiling synthesis stages..."
        )

        text = (
            "Right, I've found the problem. It was using the old configuration file. "
            "This profile measures how much time Dexter spends generating speech "
            "tokens compared with reconstructing the final audio."
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
            decode_seconds = float(message.get("decode_seconds") or 0.0)
            decode_calls = int(message.get("decode_calls") or 0)
            non_decode = max(0.0, total - decode_seconds)
            decode_share = (decode_seconds / total * 100.0) if total else 0.0
            non_decode_share = (non_decode / total * 100.0) if total else 0.0

            print("")
            print(f"First audio:                  {first_audio:.3f}s")
            print(f"Generated audio:              {audio_seconds:.3f}s")
            print(f"Total synthesis:              {total:.3f}s  (RTF {rtf:.3f})")
            print(
                f"NeuCodec decode:              {decode_seconds:.3f}s  "
                f"({decode_share:.1f}%, {decode_calls} calls)"
            )
            print(
                f"Token generation + other:     {non_decode:.3f}s  "
                f"({non_decode_share:.1f}%)"
            )
            print("")
            print(
                "Interpretation: if NeuCodec is a small share, accelerating the "
                "GGUF backbone is the highest-leverage next step. If NeuCodec is "
                "a large share, the ONNX decoder also needs acceleration/tuning."
            )
            return 0
    finally:
        runtime._shutdown(process)


if __name__ == "__main__":
    raise SystemExit(main())
