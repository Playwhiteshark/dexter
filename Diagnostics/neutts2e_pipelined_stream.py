"""Diagnostic NeuTTS streaming pipeline that preserves Perth watermarking.

Upstream NeuTTS 1.4.1 performs GGUF token generation, ONNX decode, Perth
watermarking and overlap-add serially. On Dexter's target machine the backbone
runs on Intel Arc while codec/watermark work runs on CPU, so those stages can be
pipelined without changing the generated token sequence or removing the
watermark.

This module intentionally mirrors the pinned NeuTTS 1.4.1 private streaming
method. It is diagnostic-only until output equivalence, cancellation and live
latency have been validated.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

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


def _pipelined_infer_stream_ggml(
    self,
    ref_codes,
    ref_text: str,
    input_text: str,
    emotion: str | None = None,
    temperature: float = 1.0,
    top_k: int = 50,
):
    """Run Arc token generation concurrently with CPU decode/watermark work."""
    if self.watermarker is None:
        raise RuntimeError("Pipelined diagnostic requires Perth watermarking to be active")

    prompt = self._ggml_prompt(ref_codes, ref_text, input_text, emotion)
    self.backbone.reset()

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
                job = _DecodeJob(
                    codes=tuple(token_cache[tokens_start:tokens_end]),
                    sample_start=sample_start,
                    sample_end=sample_end,
                    final=False,
                )
                if not put(job):
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

    audio_cache: list[np.ndarray] = []
    n_decoded_samples = 0

    try:
        while True:
            value = jobs.get()
            if value is _SENTINEL:
                break
            if isinstance(value, _ProducerError):
                raise value.error
            assert isinstance(value, _DecodeJob)

            # Keep the exact upstream 1.4.1 post-processing order: decode the
            # context window, watermark that reconstruction, crop the requested
            # window, then overlap-add it with prior watermarked windows.
            recon = self._decode("".join(value.codes))
            recon = self.watermarker.apply_watermark(recon, sample_rate=24_000)
            if value.sample_end is None:
                recon = recon[value.sample_start :]
            else:
                recon = recon[value.sample_start : value.sample_end]
            audio_cache.append(recon)

            processed_recon = neutts_core._linear_overlap_add(
                audio_cache,
                stride=self.streaming_stride_samples,
            )
            if value.final:
                yield processed_recon[n_decoded_samples:]
            else:
                new_samples_end = len(audio_cache) * self.streaming_stride_samples
                yield processed_recon[n_decoded_samples:new_samples_end]
                n_decoded_samples = new_samples_end
    finally:
        stop_event.set()
        producer_thread.join(timeout=2.0)


def install() -> None:
    """Patch the pinned NeuTTS base class for one diagnostic worker process."""
    neutts_core.NeuTTS._infer_stream_ggml = _pipelined_infer_stream_ggml
