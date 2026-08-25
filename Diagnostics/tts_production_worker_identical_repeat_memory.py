"""Distinguish TTS leaks from shape-dependent native allocator/cache growth.

The earlier repeated-request production check used five different utterances and
showed ~30-43 MB of working-set/private growth per request after first use. That
could be a true leak, or bounded native caches/allocators retaining buffers for
new tensor/audio shapes. This probe removes shape variation: one production
worker synthesizes the exact same deterministic text ten times with the same
fixed NeuTTS seed, while the complete launcher+worker tree is measured after
every request.

Expected interpretation:
- identical PCM/shape + continuing positive memory staircase => leak/unbounded
  request accumulation is likely;
- identical PCM/shape + early plateau => previous growth is more likely bounded
  shape-dependent allocator/runtime caching.

Run with Dexter fully quit:

    uv run python Diagnostics/tts_production_worker_identical_repeat_memory.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Diagnostics.tts_production_worker_tree_memory import _tree_snapshot

TEST_TEXT = (
    "Dexter is repeating exactly the same speech request so we can tell a true "
    "memory leak from native caches that only grow when tensor or audio shapes change."
)
REQUESTS = 10


def main() -> int:
    if os.name != "nt":
        raise SystemExit("This diagnostic currently targets Windows")

    from Runtime.Voice.neutts2e_backend import NeuTTS2EBackend

    print("Dexter production TTS identical-request memory check")
    print(
        f"Quit Dexter before running this. One production worker synthesizes the exact "
        f"same deterministic utterance {REQUESTS} times; process-tree memory and PCM "
        "identity are checked after every request.\n"
    )

    backend = NeuTTS2EBackend()
    try:
        started = time.perf_counter()
        backend.start()
        startup_seconds = time.perf_counter() - started

        process = backend._process
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
        print(
            "Stage                 WS MB   Private MB   dWS req   dPriv req   samples   chunks   RTF   hash"
        )
        print(
            "-------------------  -------  ----------  --------  ----------  --------  -------  -----  ------------"
        )
        print(
            f"ready / warmed       {float(baseline['working_set_mb']):7.1f}  "
            f"{float(baseline['private_mb']):10.1f}  {'-':>8}  {'-':>10}  "
            f"{'-':>8}  {'-':>7}  {'-':>5}  {'-':>12}"
        )

        first_hash: str | None = None
        first_samples: int | None = None
        first_chunks: int | None = None
        rows: list[dict[str, float | int | str]] = []

        for index in range(1, REQUESTS + 1):
            cancel_event = threading.Event()
            digest = hashlib.sha256()
            samples = 0
            started = time.perf_counter()

            for chunk in backend.synthesize(TEST_TEXT, cancel_event):
                raw = chunk.samples.astype("<f4", copy=False).tobytes()
                digest.update(raw)
                samples += int(chunk.samples.size)

            elapsed = time.perf_counter() - started
            stats = backend.last_synthesis_stats or {}
            chunks = int(stats.get("chunks") or 0)
            audio_seconds = samples / 24_000.0
            rtf = elapsed / audio_seconds if audio_seconds else 0.0
            pcm_hash = digest.hexdigest()

            if first_hash is None:
                first_hash = pcm_hash
                first_samples = samples
                first_chunks = chunks
            else:
                if pcm_hash != first_hash:
                    raise RuntimeError(
                        f"Request {index} PCM differs from request 1; workload is not deterministic"
                    )
                if samples != first_samples or chunks != first_chunks:
                    raise RuntimeError(
                        f"Request {index} shape differs from request 1: "
                        f"samples={samples}/{first_samples}, chunks={chunks}/{first_chunks}"
                    )

            time.sleep(0.5)
            current = _tree_snapshot(root_pid)
            ws = float(current["working_set_mb"])
            private = float(current["private_mb"])
            prev_ws = float(previous["working_set_mb"])
            prev_private = float(previous["private_mb"])

            row = {
                "ws": ws,
                "private": private,
                "dws": ws - prev_ws,
                "dprivate": private - prev_private,
                "samples": samples,
                "chunks": chunks,
                "rtf": rtf,
                "hash": pcm_hash,
            }
            rows.append(row)
            print(
                f"after request {index:<5}  {ws:7.1f}  {private:10.1f}  "
                f"{float(row['dws']):+8.1f}  {float(row['dprivate']):+10.1f}  "
                f"{samples:8d}  {chunks:7d}  {rtf:5.3f}  {pcm_hash[:12]}"
            )
            previous = current

        base_ws = float(baseline["working_set_mb"])
        base_private = float(baseline["private_mb"])
        final = rows[-1]
        late_rows = rows[2:]  # requests 3-10, after two opportunities to warm caches
        late_net_ws = float(late_rows[-1]["ws"]) - float(late_rows[0]["ws"])
        late_net_private = float(late_rows[-1]["private"]) - float(late_rows[0]["private"])
        late_positive_dws = [float(r["dws"]) for r in late_rows if float(r["dws"]) > 0]
        late_positive_dprivate = [
            float(r["dprivate"]) for r in late_rows if float(r["dprivate"]) > 0
        ]

        print("\nSummary")
        print(f"  Deterministic PCM/shape across all requests: yes ({first_hash[:16]}...)")
        print(
            f"  Warmup -> request {REQUESTS}: "
            f"dWS={float(final['ws']) - base_ws:+.1f} MB, "
            f"dPrivate={float(final['private']) - base_private:+.1f} MB"
        )
        print(
            "  Request 3 -> request 10: "
            f"dWS={late_net_ws:+.1f} MB, dPrivate={late_net_private:+.1f} MB"
        )
        if late_positive_dws:
            print(
                f"  Late positive per-request WS deltas: "
                f"min={min(late_positive_dws):.1f}, max={max(late_positive_dws):.1f} MB"
            )
        if late_positive_dprivate:
            print(
                f"  Late positive per-request private deltas: "
                f"min={min(late_positive_dprivate):.1f}, max={max(late_positive_dprivate):.1f} MB"
            )
        print(
            "Interpretation: continuing growth with identical PCM and chunk geometry points "
            "toward request-lifetime accumulation/leak. A clear plateau instead points toward "
            "bounded shape-dependent native allocator/cache behavior in the earlier test."
        )
        return 0
    finally:
        backend.stop()


if __name__ == "__main__":
    raise SystemExit(main())
