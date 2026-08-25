"""Benchmark larger NeuTTS stream windows for sustained real-time headroom.

This does not change production. It measures the current Arc Vulkan + pipelined
Perth path at the production 25-frame window and two larger windows. Larger
windows pay codec/watermark overhead less often, so they may improve sustained
RTF at the cost of first-audio latency. If a larger window wins decisively, the
production follow-up can keep the first window at 25 frames and enlarge only
subsequent windows.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Runtime.Voice import neutts2e_runtime as runtime

WORKER = REPO_ROOT / "Diagnostics" / "neutts2e_latency_worker.py"
TEXT = (
    "Dexter should keep speaking continuously even while the rest of the local "
    "assistant is busy. This benchmark is intentionally long enough to expose "
    "steady-state synthesis performance and repeated codec and watermark costs. "
    "The first-audio number is secondary here; the main goal is to find whether "
    "larger post-start streaming windows create enough real-time headroom to "
    "eliminate the small live playback jitter without adding a large buffer."
)


def run_case(frames: int) -> dict[str, float]:
    env = runtime.runtime_env("q8")
    env["DEXTER_TTS_DIAGNOSTIC_STREAM_FRAMES"] = str(frames)

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
        env=env,
    )
    reader = runtime._ProtocolReader(process)
    try:
        ready = reader.get(timeout=180)
        if ready.get("type") != "ready":
            raise RuntimeError(f"Worker failed to start: {ready}")
        if ready.get("backbone_device") != "gpu":
            raise RuntimeError(f"Expected GPU backbone, got {ready}")
        if not bool(ready.get("pipeline_watermark")):
            raise RuntimeError("Production Perth pipeline is not active")

        chunks, first_audio, total, timings = runtime._read_stream_request(
            process,
            reader,
            request_id=1,
            text=TEXT,
            timeout=180,
            print_chunks=False,
        )
        if first_audio is None or not chunks:
            raise RuntimeError("No streamed audio was produced")

        audio_seconds = sum(chunk.size for chunk in chunks) / runtime.SAMPLE_RATE
        rtf = total / audio_seconds
        post_first = [item["arrival_minus_audio_seconds"] for item in timings[1:]]
        average_deficit = sum(post_first) / len(post_first) if post_first else 0.0
        worst_deficit = max(post_first, default=0.0)
        first_chunk_seconds = timings[0]["audio_seconds"] if timings else 0.0

        print("")
        print(f"{frames:2d} frames ({first_chunk_seconds:.3f}s audio/chunk)")
        print(f"  First audio:           {first_audio:.3f}s")
        print(f"  Generated audio:       {audio_seconds:.3f}s")
        print(f"  Total synthesis:       {total:.3f}s")
        print(f"  RTF:                   {rtf:.3f}")
        print(f"  Average chunk deficit: {average_deficit:+.3f}s")
        print(f"  Worst chunk deficit:   {worst_deficit:+.3f}s")

        return {
            "frames": float(frames),
            "first_audio": first_audio,
            "rtf": rtf,
            "average_deficit": average_deficit,
            "worst_deficit": worst_deficit,
        }
    finally:
        runtime._shutdown(process)


def main() -> int:
    runtime.doctor("q8")
    print("")
    print("Production remains at 25 frames; testing 25 / 40 / 50 in isolated workers.")
    results = [run_case(frames) for frames in (25, 40, 50)]

    print("")
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    baseline = results[0]
    for result in results:
        first_delta_ms = (result["first_audio"] - baseline["first_audio"]) * 1000.0
        throughput_gain = baseline["rtf"] / result["rtf"] - 1.0
        print(
            f"{int(result['frames']):2d} frames: RTF {result['rtf']:.3f} "
            f"({throughput_gain:+.1%} throughput vs 25), first {result['first_audio']:.3f}s "
            f"({first_delta_ms:+.0f}ms)"
        )

    best = min(results, key=lambda item: item["rtf"])
    print(
        f"\nBest sustained window: {int(best['frames'])} frames "
        f"(RTF {best['rtf']:.3f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
