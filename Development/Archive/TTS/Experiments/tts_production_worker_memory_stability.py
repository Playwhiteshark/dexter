"""Check whether production TTS memory plateaus across repeated utterances.

The complete production worker has a small Python launcher plus a heavy child
interpreter. The first real synthesis was observed to add ~52 MB working set and
~95 MB private commit after warmup. That may be normal lazy first-use allocation
(Torch FFT plans/workspaces, ORT arenas, llama.cpp buffers, etc.), or it could be
unbounded per-request growth.

This diagnostic launches the exact production NeuTTS backend once, measures the
entire process tree after warmup, then sends five ordinary utterances through the
same persistent worker. It records tree working set/private commit after every
request without restarting the worker or forcing garbage collection. A healthy
result should rise on the first one or two requests and then broadly plateau.

Run with Dexter fully quit:

    uv run python Diagnostics/tts_production_worker_memory_stability.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Diagnostics.tts_production_worker_tree_memory import _tree_snapshot


TEST_TEXTS = [
    (
        "Dexter is checking whether the speech worker allocates one-time runtime "
        "buffers on its first real response and then settles into a stable footprint."
    ),
    (
        "A second ordinary answer now passes through exactly the same persistent voice "
        "worker so we can distinguish startup caches from memory that grows per request."
    ),
    (
        "This third response uses similar length and punctuation. If memory remains near "
        "the previous level, the earlier growth was probably normal lazy initialization."
    ),
    (
        "The fourth synthesis is another independent request. We are deliberately not "
        "restarting the model, trimming its working set, or forcing garbage collection."
    ),
    (
        "This fifth and final response gives the worker several chances to reveal any "
        "systematic accumulation before we call the TTS memory cleanup stable."
    ),
]


def _fmt_delta(value: float) -> str:
    return f"{value:+.1f}"


def main() -> int:
    if os.name != "nt":
        raise SystemExit("This diagnostic currently targets Windows")

    from Runtime.Voice.neutts2e_backend import NeuTTS2EBackend

    print("Dexter production TTS repeated-request memory stability check")
    print(
        "Quit Dexter before running this. One real production worker is kept alive for "
        "five sequential responses; the complete launcher+worker process tree is measured "
        "after warmup and after every response.\n"
    )

    backend = NeuTTS2EBackend()
    try:
        startup_started = time.perf_counter()
        backend.start()
        startup_seconds = time.perf_counter() - startup_started

        process = backend._process  # diagnostic access to backend-owned child
        if process is None or process.poll() is not None:
            raise RuntimeError("Production TTS worker is not alive after backend.start()")
        root_pid = int(process.pid)

        time.sleep(0.5)
        baseline = _tree_snapshot(root_pid)
        previous = baseline

        info = backend.ready_info or {}
        print(f"Backend root PID: {root_pid}")
        print(f"Startup + warmup: {startup_seconds:.2f} s")
        print(
            "Production settings: "
            f"variant={info.get('variant')} backbone={info.get('backbone_device')} "
            f"batch={info.get('backbone_batch')}/{info.get('backbone_ubatch')} "
            f"ORT={info.get('codec_ort_intra_threads')}/{info.get('codec_ort_inter_threads')} "
            f"stream={info.get('first_stream_frames')}/{info.get('steady_stream_frames')}"
        )
        print()
        print("Stage                 WS MB   Private MB   dWS req   dPriv req   dWS total   dPriv total   RTF   TTFA")
        print("-------------------  -------  ----------  --------  ----------  ---------  -----------  -----  -----")
        print(
            f"ready / warmed       {float(baseline['working_set_mb']):7.1f}  "
            f"{float(baseline['private_mb']):10.1f}  "
            f"{'-':>8}  {'-':>10}  {'+0.0':>9}  {'+0.0':>11}  {'-':>5}  {'-':>5}"
        )

        request_rows: list[dict[str, float]] = []
        for index, text in enumerate(TEST_TEXTS, start=1):
            cancel_event = threading.Event()
            samples = 0
            started = time.perf_counter()
            for chunk in backend.synthesize(text, cancel_event):
                samples += int(chunk.samples.size)
            elapsed = time.perf_counter() - started
            stats = backend.last_synthesis_stats or {}
            audio_seconds = samples / 24_000.0
            rtf = elapsed / audio_seconds if audio_seconds else 0.0
            ttfa = float(stats.get("first_audio_seconds") or 0.0)

            # Do not trim the working set or run GC in the worker. Let native
            # runtimes settle naturally, then sample the complete process tree.
            time.sleep(0.5)
            current = _tree_snapshot(root_pid)

            ws = float(current["working_set_mb"])
            private = float(current["private_mb"])
            prev_ws = float(previous["working_set_mb"])
            prev_private = float(previous["private_mb"])
            base_ws = float(baseline["working_set_mb"])
            base_private = float(baseline["private_mb"])

            row = {
                "ws": ws,
                "private": private,
                "delta_ws_request": ws - prev_ws,
                "delta_private_request": private - prev_private,
                "delta_ws_total": ws - base_ws,
                "delta_private_total": private - base_private,
                "rtf": rtf,
                "ttfa": ttfa,
            }
            request_rows.append(row)
            print(
                f"after request {index:<5}  {ws:7.1f}  {private:10.1f}  "
                f"{_fmt_delta(row['delta_ws_request']):>8}  "
                f"{_fmt_delta(row['delta_private_request']):>10}  "
                f"{_fmt_delta(row['delta_ws_total']):>9}  "
                f"{_fmt_delta(row['delta_private_total']):>11}  "
                f"{rtf:5.3f}  {ttfa:5.3f}"
            )
            previous = current

        final = request_rows[-1]
        late = request_rows[1:]
        late_ws_span = max(row["ws"] for row in late) - min(row["ws"] for row in late)
        late_private_span = max(row["private"] for row in late) - min(row["private"] for row in late)
        late_net_ws = request_rows[-1]["ws"] - request_rows[0]["ws"]
        late_net_private = request_rows[-1]["private"] - request_rows[0]["private"]

        print("\nSummary")
        print(
            f"  Warmup -> request 1: dWS={request_rows[0]['delta_ws_total']:+.1f} MB, "
            f"dPrivate={request_rows[0]['delta_private_total']:+.1f} MB"
        )
        print(
            f"  Request 1 -> request 5: dWS={late_net_ws:+.1f} MB, "
            f"dPrivate={late_net_private:+.1f} MB"
        )
        print(
            f"  Requests 2-5 observed span: WS={late_ws_span:.1f} MB, "
            f"Private={late_private_span:.1f} MB"
        )
        print(
            f"  Final footprint: WS={final['ws']:.1f} MB, Private={final['private']:.1f} MB"
        )
        print(
            "Interpretation: one-time growth followed by a narrow plateau is expected. "
            "Consistent positive per-request growth would justify a leak/cache investigation."
        )
        return 0
    finally:
        backend.stop()


if __name__ == "__main__":
    raise SystemExit(main())
