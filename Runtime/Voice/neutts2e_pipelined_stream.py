"""Pipelined NeuTTS-2E GGUF streaming for Dexter.

NeuTTS 1.4.1 normally performs GGUF token generation, ONNX decode, Perth
watermarking, and overlap-add serially. Dexter's selected runtime places the
GGUF backbone on Intel Arc while codec/watermark work remains on CPU, so these
independent stages can overlap without changing the generated token sequence,
watermarking, or output ordering.

Dexter normally keeps NeuTTS's 25-frame first window for low startup latency,
then uses a larger steady-state window (50 frames by default) so codec/Perth
overhead is paid less often once speech has begun. ``DEXTER_TTS_FIRST_STREAM_FRAMES``
can override the first window for live diagnostics, while
``DEXTER_TTS_STEADY_STREAM_FRAMES`` overrides the steady-state size. Setting
both to 50 produces uniform 50-frame / 1.0-second windows.

``DEXTER_TTS_BACKEND_SAMPLING=1`` is an opt-in experiment that asks llama.cpp
to run the same sampling chain on the model backend. Production keeps this off
until the Vulkan path has been validated under concurrent LLM load.

This module mirrors the pinned NeuTTS 1.4.1 private streaming method. Keep the
NeuTTS dependency pinned while this integration exists and revalidate the
pipeline before upgrading NeuTTS.
"""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass

import numpy as np
import neutts.neutts as neutts_core


@dataclass(frozen=True)
class _DecodeJob:
    codes: tuple[str, ...]
    sample_start: int
    sample_end: int | None
    stride_samples: int
    final: bool


@dataclass(frozen=True)
class _ProducerError:
    error: BaseException


_SENTINEL = object()
_BACKEND_SAMPLER_ATTR = "_dexter_backend_sampler"


def _backend_sampling_requested() -> bool:
    return os.environ.get("DEXTER_TTS_BACKEND_SAMPLING", "0").strip() == "1"


def _attach_backend_sampler(tts) -> None:
    """Attach a llama.cpp backend sampler matching NeuTTS's generation settings."""
    if not _backend_sampling_requested():
        return

    import llama_cpp

    backbone = tts.backbone
    if not hasattr(backbone, "_init_sampler") or not hasattr(backbone, "ctx"):
        raise RuntimeError(
            "Dexter backend sampling requires llama-cpp-python's Llama sampler API"
        )

    # NeuTTS/Dexter currently call the backbone with temperature=1.0 and top_k=50.
    # Leaving the other arguments at llama-cpp-python defaults mirrors the same
    # high-level sampler chain used by Llama.__call__ for this request.
    sampler = backbone._init_sampler(
        top_k=50,
        temp=1.0,
    )

    try:
        active = bool(llama_cpp.llama_set_sampler(backbone.ctx, 0, sampler.sampler))
    except Exception:
        sampler.close()
        raise

    if not active:
        sampler.close()
        raise RuntimeError(
            "llama.cpp rejected Dexter backend sampling for the active Vulkan context"
        )

    setattr(tts, _BACKEND_SAMPLER_ATTR, sampler)


def _reset_backend_sampler(tts) -> None:
    sampler = getattr(tts, _BACKEND_SAMPLER_ATTR, None)
    if sampler is not None:
        sampler.reset()


def _frame_weight(frame: np.ndarray) -> np.ndarray:
    """Return NeuTTS's exact linear overlap-add weight for one 1-D frame."""
    length = int(frame.shape[-1])
    t = np.linspace(0, 1, length + 2, dtype=frame.dtype)[1:-1]
    return 0.5 - np.abs(t - 0.5)


def _first_stream_frames(default_frames: int) -> int:
    raw = os.environ.get(
        "DEXTER_TTS_FIRST_STREAM_FRAMES",
        str(default_frames),
    ).strip()
    try:
        frames = int(raw)
    except ValueError as exc:
        raise RuntimeError("DEXTER_TTS_FIRST_STREAM_FRAMES must be an integer") from exc
    if frames < 10 or frames > 100:
        raise RuntimeError("DEXTER_TTS_FIRST_STREAM_FRAMES must be between 10 and 100")
    return frames


def _steady_stream_frames(first_frames: int) -> int:
    raw = os.environ.get("DEXTER_TTS_STEADY_STREAM_FRAMES", "50").strip()
    try:
        frames = int(raw)
    except ValueError as exc:
        raise RuntimeError("DEXTER_TTS_STEADY_STREAM_FRAMES must be an integer") from exc
    if frames < 10 or frames > 100:
        raise RuntimeError("DEXTER_TTS_STEADY_STREAM_FRAMES must be between 10 and 100")
    return frames


def _incremental_overlap_segment(
    previous: np.ndarray | None,
    current: np.ndarray,
    *,
    previous_stride: int | None,
    emit_samples: int,
    final: bool,
) -> np.ndarray:
    """Produce only the not-yet-emitted overlap-add segment.

    ``previous_stride`` is the distance from the previous frame's origin to the
    current frame's origin. ``emit_samples`` is the distance from the current
    frame's origin to the next regular frame's origin. Keeping those separate is
    what lets Dexter use different first and steady-state window sizes without
    changing the overlap math.

    With NeuTTS's current window geometry, the not-yet-emitted region can depend
    only on the current frame and the immediately preceding frame. This is
    mathematically identical to recomputing the complete overlap-add history,
    while remaining O(n) over a streamed utterance.
    """
    current = np.asarray(current).reshape(-1)
    if previous is not None:
        previous = np.asarray(previous).reshape(-1)

    if emit_samples <= 0:
        raise RuntimeError("NeuTTS emit stride must be positive")
    if previous is not None:
        if previous_stride is None or previous_stride <= 0:
            raise RuntimeError("NeuTTS previous stride must be positive")
        previous_tail = max(0, int(previous.shape[-1]) - previous_stride)
    else:
        previous_tail = 0

    end = max(int(current.shape[-1]), previous_tail) if final else emit_samples
    if end <= 0:
        return np.empty(0, dtype=current.dtype)

    out = np.zeros(end, dtype=current.dtype)
    sum_weight = np.zeros(end, dtype=current.dtype)

    if previous is not None:
        assert previous_stride is not None
        prev_weight = _frame_weight(previous)
        prev_local_start = previous_stride
        prev_local_end = min(int(previous.shape[-1]), previous_stride + end)
        if prev_local_end > prev_local_start:
            count = prev_local_end - prev_local_start
            out[:count] += previous[prev_local_start:prev_local_end] * prev_weight[
                prev_local_start:prev_local_end
            ]
            sum_weight[:count] += prev_weight[prev_local_start:prev_local_end]

    curr_weight = _frame_weight(current)
    curr_local_end = min(int(current.shape[-1]), end)
    if curr_local_end > 0:
        out[:curr_local_end] += current[:curr_local_end] * curr_weight[:curr_local_end]
        sum_weight[:curr_local_end] += curr_weight[:curr_local_end]

    if not np.all(sum_weight > 0):
        raise RuntimeError("NeuTTS incremental overlap-add produced an uncovered sample")
    return out / sum_weight


def _pipelined_infer_stream_ggml(
    self,
    ref_codes,
    ref_text: str,
    input_text: str,
    emotion: str | None = None,
    temperature: float = 1.0,
    top_k: int = 50,
):
    """Overlap GGUF token generation with CPU decode + Perth watermarking."""
    if self.watermarker is None:
        raise RuntimeError("Dexter's pipelined NeuTTS path requires Perth watermarking")

    prompt = self._ggml_prompt(ref_codes, ref_text, input_text, emotion)
    _reset_backend_sampler(self)
    self.backbone.reset()

    first_frames = _first_stream_frames(int(self.streaming_frames_per_chunk))
    steady_frames = _steady_stream_frames(first_frames)

    # Two pending windows are enough to keep Arc generation ahead without
    # letting cancellation accumulate a large amount of useless work.
    jobs: queue.Queue[_DecodeJob | _ProducerError | object] = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    def put(value: _DecodeJob | _ProducerError | object) -> bool:
        while not stop_event.is_set():
            try:
                jobs.put(value, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def producer() -> None:
        token_cache: list[str] = [f"<|speech_{idx}|>" for idx in ref_codes]
        n_decoded_tokens = len(ref_codes)
        regular_windows = 0
        try:
            for item in self.backbone(
                prompt,
                max_tokens=self.max_context,
                temperature=temperature,
                top_k=top_k,
                stop=["<|SPEECH_GENERATION_END|>"],
                stream=True,
                seed=self._call_seed(),
            ):
                if stop_event.is_set():
                    return

                token_cache.append(item["choices"][0]["text"])
                frames = first_frames if regular_windows == 0 else steady_frames
                if (
                    len(token_cache) - n_decoded_tokens
                    < frames + self.streaming_lookforward
                ):
                    continue

                tokens_start = max(
                    n_decoded_tokens
                    - self.streaming_lookback
                    - self.streaming_overlap_frames,
                    0,
                )
                tokens_end = (
                    n_decoded_tokens
                    + frames
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = (
                    sample_start
                    + (frames + 2 * self.streaming_overlap_frames) * self.hop_length
                )
                stride_samples = frames * self.hop_length
                if not put(
                    _DecodeJob(
                        codes=tuple(token_cache[tokens_start:tokens_end]),
                        sample_start=sample_start,
                        sample_end=sample_end,
                        stride_samples=stride_samples,
                        final=False,
                    )
                ):
                    return
                n_decoded_tokens += frames
                regular_windows += 1

            remaining_tokens = len(token_cache) - n_decoded_tokens
            if remaining_tokens > 0:
                tokens_start = max(
                    len(token_cache)
                    - (
                        self.streaming_lookback
                        + self.streaming_overlap_frames
                        + remaining_tokens
                    ),
                    0,
                )
                sample_start = (
                    len(token_cache)
                    - tokens_start
                    - remaining_tokens
                    - self.streaming_overlap_frames
                ) * self.hop_length
                if not put(
                    _DecodeJob(
                        codes=tuple(token_cache[tokens_start:]),
                        sample_start=sample_start,
                        sample_end=None,
                        stride_samples=max(1, remaining_tokens * self.hop_length),
                        final=True,
                    )
                ):
                    return
        except BaseException as exc:
            put(_ProducerError(exc))
        finally:
            put(_SENTINEL)

    producer_thread = threading.Thread(
        target=producer,
        name="DexterTTSArcProducer",
        daemon=True,
    )
    producer_thread.start()

    previous_recon: np.ndarray | None = None
    previous_stride: int | None = None

    try:
        while True:
            value = jobs.get()
            if value is _SENTINEL:
                break
            if isinstance(value, _ProducerError):
                raise value.error
            assert isinstance(value, _DecodeJob)

            # Preserve NeuTTS 1.4.1's post-processing order exactly: decode the
            # context window, watermark the reconstruction, crop the requested
            # window, then overlap-add the already-watermarked windows.
            recon = self._decode("".join(value.codes))
            recon = self.watermarker.apply_watermark(recon, sample_rate=24_000)
            if value.sample_end is None:
                recon = recon[value.sample_start :]
            else:
                recon = recon[value.sample_start : value.sample_end]

            processed_recon = _incremental_overlap_segment(
                previous_recon,
                recon,
                previous_stride=previous_stride,
                emit_samples=value.stride_samples,
                final=value.final,
            )
            previous_recon = recon
            previous_stride = value.stride_samples
            yield processed_recon
    finally:
        stop_event.set()
        producer_thread.join(timeout=2.0)


def install() -> None:
    """Install Dexter's validated pipeline and optional backend-sampling probe."""
    if _backend_sampling_requested():
        original_load_backbone = neutts_core.NeuTTS._load_backbone

        def load_backbone_with_backend_sampling(self, backbone_repo, backbone_device):
            original_load_backbone(self, backbone_repo, backbone_device)
            if str(backbone_device).strip().lower() == "cpu":
                raise RuntimeError(
                    "DEXTER_TTS_BACKEND_SAMPLING requires the GPU/Vulkan backbone"
                )
            _attach_backend_sampler(self)

        neutts_core.NeuTTS._load_backbone = load_backbone_with_backend_sampling

    neutts_core.NeuTTS._infer_stream_ggml = _pipelined_infer_stream_ggml
