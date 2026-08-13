from __future__ import annotations

import pyperclip

from config import MAX_CLIPBOARD_CHARS


def get_clipboard_text(max_chars: int = 50_000) -> dict:
    if max_chars < 1 or max_chars > MAX_CLIPBOARD_CHARS:
        raise ValueError(
            f"max_chars must be between 1 and {MAX_CLIPBOARD_CHARS}."
        )

    text = pyperclip.paste()
    if not isinstance(text, str):
        text = str(text)

    truncated = len(text) > max_chars

    return {
        "status": "ok",
        "operation": "get_clipboard",
        "text": text[:max_chars],
        "length": len(text),
        "truncated": truncated,
    }


def set_clipboard_text(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("Clipboard text must be a string.")
    if len(text) > MAX_CLIPBOARD_CHARS:
        raise ValueError(
            f"Clipboard text exceeds the {MAX_CLIPBOARD_CHARS}-character limit."
        )

    pyperclip.copy(text)
    observed = pyperclip.paste()
    verified = observed == text

    if not verified:
        raise RuntimeError("Clipboard write returned but verification failed.")

    return {
        "status": "ok",
        "operation": "set_clipboard",
        "length": len(text),
        "verified": True,
    }
