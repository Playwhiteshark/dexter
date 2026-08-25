"""Pipelined NeuTTS-2E GGUF streaming for Dexter.

NeuTTS 1.4.1 normally performs GGUF token generation, ONNX decode, Perth
watermarking, and overlap-add serially. Dexter's selected runtime places the
GGUF backbone on Intel Arc while codec/watermark work remains on CPU, so these
independent stages can overlap without changing the generated token sequence,
watermarking, or output ordering.

This module mirrors the pinned NeuTTS 1.4.1 private streaming method. Keep the
NeuTTS dependency pinned while this integration exists and revalidate the
pipeline before upgrading NeuTTS.
"""

from __future__ import annotations

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
    final: bool


@dataclass(frozen=True)
class _ProducerError:
    error: BaseException


_SENTINEL = object()


def _frame_weight(frame: np.ndarray) -> np.ndarray:
    """Return NeuTTS's exact linear overlap-add weight for one 1-D frame."""
    length = int(frame.shape[-1])
    t = np.linspace(0, 1, length + 2, dtype=frame.dtype)[1:-1]
    return 0.5 - np.abs(t - 0.5)


def _incremental_overlap_segment(
    previous: np.ndarray | None,
    current: np.ndarray,
    *,
    stride: int,
    final: bool,
) -> np.ndarray:
    """Produce only the not-yet-emitted overlap-add segment.

    NeuTTS's upstream implementation recomputes ``_linear_overlap_add`` over
    every decoded window after each new chunk, then slices away all audio that
    was already emitted. With Dexter's window geometry each frame is shorter
    than two strides, so the new segment can only depend on the current frame
    and the immediately preceding frame. Computing those two contributions is
    mathematically identical for the emitted segment while making streaming
    overlap-add O(n) instead of O(n^2) over an utterance.
    """
    if current.ndim != 1:
        current = np.asarray(current).reshape(-1)
    if previous is not None and previous.ndim != 1:
        previous = np.asarray(previous).reshape(-1)

    if previous is not None and int(previous.shape[-1]) >= 2 * stride:
        raise RuntimeError("NeuTTS overlap window unexpectedly spans more than two strides")
    if int(current.shape[-1]) >= 2 * stride:
        raise RuntimeError("NeuTTS overlap window unexpectedly spans more than two strides")

    # Local coordinates: current starts at 0; previous starts at -stride.
    start = 0
    if final:
        previous_end = int(previous.shape[-1]) - stride if previous is not None else 0
        end = max(int(current.shape[-1]), previous_end)
    else:
        end = stride

    if end <= start:
        return np.empty(0, dtype=current.dtype)

    out = np.zeros(end - start, dtype=current.dtype)
    sum_weight = np.zeros(end - start, dtype=current.dtype)

    if previous is not None:
        prev_weight = _frame_weight(previous)
        prev_local_start = stride
        prev_local_end = min(int(previous.shape[-1]), stride + end)
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
    self.backbone.reset()

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
                if (
                    len(token_cache[n_decoded_tokens:])
                    < self.streaming_frames_per_chunk + self.streaming_lookforward
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
                    + self.streaming_frames_per_chunk
                    + self.streaming_lookforward
                    + self.streaming_overlap_frames
                )
                sample_start = (n_decoded_tokens - tokens_start) * self.hop_length
                sample_end = (
                    sample_start
                    + (self.streaming_frames_per_chunk + 2 * self.streaming_overlap_frames)
                    * self.hop_length
                )
                if not put(
                    _DecodeJob(
                        codes=tuple(token_cache[tokens_start:tokens_end]),
                        sample_start=sample_start,
                        sample_end=sample_end,
                        final=False,
                    )
                ):
                    return
                n_decoded_tokens += self.streaming_frames_per_chunk

            remaining_tokens = len(token_cache) - n_decoded_tokens
            if len(token_cache) > n_decoded_tokens:
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
                stride=self.streaming_stride_samples,
                final=value.final,
            )
            previous_recon = recon
            yield processed_recon
    finally:
        stop_event.set()
        producer_thread.join(timeout=2.0)


def install() -> None:
    """Install the validated pipeline into the pinned NeuTTS base class."""
    neutts_core.NeuTTS._infer_stream_ggml = _pipelined_infer_stream_ggml
