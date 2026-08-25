"""Lightweight written-text -> NeuTTS speech-text normalization.

Utterwise owns semantic text normalization (acronyms, numbers, dates, versions,
currency, and similar written forms). Dexter adds only the small amount of
assistant-output policy that the production voice actually needs:

* single-underscore emphasis and snake_case become ordinary word boundaries;
* word-style hyphens become spaces, while negatives/dates/phone numbers remain
  intact for semantic normalization;
* AI is rendered as ``A I`` everywhere, based on the production voice tests;
* JSON is rendered as the preferred spoken form ``jay son``; and
* source punctuation is retained because NeuTTS uses it for pauses/prosody.

No model, phonemizer, background process, or heavy NLP dependency lives here.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from typing import Any

from utterwise.normalizers import Normalizer


_WHITESPACE_RE = re.compile(r"\s+")
_SINGLE_UNDERSCORE_EMPHASIS_RE = re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)")
_IDENTIFIER_UNDERSCORE_RE = re.compile(r"(?<=[A-Za-z])_(?=[A-Za-z])")
_WORD_HYPHEN_RE = re.compile(
    r"(?<=[A-Za-z])-(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])-(?=[A-Za-z])"
)
_AI_RE = re.compile(r"\bAI\b")
_JSON_RE = re.compile(r"\bJSON\b")


class SpeechTextNormalizer:
    """Normalize already-cleaned assistant prose for NeuTTS."""

    def __init__(self) -> None:
        self._normalizer = Normalizer()

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        prepared = self._prepare(text)
        result = self._normalizer.normalize(
            prepared,
            output="plain",
            policy="assistant",
        )

        # Utterwise's plain renderer intentionally drops punctuation. NeuTTS
        # benefits from the punctuation already present in normal prose, so use
        # Utterwise's source-aligned semantic replacements while keeping source
        # punctuation and whitespace between tokens.
        cleaned = result.cleaned_text
        parts: list[str] = []
        cursor = 0
        for token, segment in zip(result.tokens, result.segments):
            start = int(token.start)
            end = int(token.end)
            if start > cursor:
                parts.append(cleaned[cursor:start])

            if token.type == "PUNCTUATION":
                parts.append(token.value)
            elif segment.text:
                parts.append(segment.text)
            else:
                # Be conservative if a future Utterwise rule yields an empty
                # non-punctuation segment: retain the original token rather
                # than silently deleting visible content.
                parts.append(cleaned[start:end])
            cursor = end

        if cursor < len(cleaned):
            parts.append(cleaned[cursor:])

        return _WHITESPACE_RE.sub(" ", "".join(parts)).strip()

    @staticmethod
    def _prepare(text: str) -> str:
        value = _SINGLE_UNDERSCORE_EMPHASIS_RE.sub(r"\1", text)
        value = _IDENTIFIER_UNDERSCORE_RE.sub(" ", value)
        value = _WORD_HYPHEN_RE.sub(" ", value)

        # Dash glyphs have been unreliable in NeuTTS prose. A comma retains
        # their pause semantics; ASCII minus signs are deliberately untouched.
        value = value.replace("—", ", ").replace("–", ", ")

        # These are measured application policies, not a general pronunciation
        # dictionary. Utterwise continues to own ordinary acronym handling.
        value = _AI_RE.sub("A I", value)
        value = _JSON_RE.sub("jay son", value)
        return _WHITESPACE_RE.sub(" ", value).strip()


class NormalizingSpeechBackend:
    """Transparent SpeechBackend wrapper that normalizes text before synthesis."""

    def __init__(self, inner: Any, *, normalizer: SpeechTextNormalizer | None = None) -> None:
        self.inner = inner
        self.normalizer = normalizer or SpeechTextNormalizer()

    @property
    def ready_info(self) -> Any:
        return getattr(self.inner, "ready_info", None)

    def start(self) -> None:
        self.inner.start()

    def synthesize(
        self,
        text: str,
        cancel_event: threading.Event,
    ) -> Iterable[Any]:
        return self.inner.synthesize(self.normalizer.normalize(text), cancel_event)

    def stop(self) -> None:
        self.inner.stop()
