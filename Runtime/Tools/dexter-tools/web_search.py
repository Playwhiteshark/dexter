"""Bounded text-only web search for small current-information lookups."""

from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import urlsplit

try:
    from ddgs import DDGS
except ImportError:  # Keeps schema/tests importable before requirements install.
    DDGS = None


MAX_QUERY_CHARS = 500
MAX_RESULTS = 5
MAX_TITLE_CHARS = 200
MAX_SNIPPET_CHARS = 500
SEARCH_TIMEOUT_SECONDS = 8
SEARCH_BACKENDS = ("duckduckgo", "brave", "bing")


def _compact_text(value: Any, limit: int) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _safe_url(value: Any) -> str | None:
    url = str(value or "").strip()
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return url


def search_web(query: str) -> dict:
    """Return at most five normalized title/URL/snippet records."""
    requested = str(query or "").strip()
    if not requested:
        raise ValueError("query must not be empty.")
    if len(requested) > MAX_QUERY_CHARS:
        raise ValueError(f"query must be at most {MAX_QUERY_CHARS} characters.")
    if DDGS is None:
        raise RuntimeError(
            "Web search dependency is missing. Run install_requirements.ps1."
        )

    failures: list[str] = []
    raw_results: list[dict[str, Any]] = []
    for backend in SEARCH_BACKENDS:
        try:
            found = DDGS(timeout=SEARCH_TIMEOUT_SECONDS).text(
                requested,
                safesearch="moderate",
                max_results=MAX_RESULTS,
                backend=backend,
            )
            raw_results = list(found or [])
            if raw_results:
                break
        except Exception as exc:
            failures.append(f"{backend}: {type(exc).__name__}")

    normalized: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = _safe_url(item.get("href") or item.get("url"))
        if not url or url in seen_urls:
            continue
        title = _compact_text(item.get("title"), MAX_TITLE_CHARS)
        snippet = _compact_text(
            item.get("body") or item.get("snippet") or item.get("description"),
            MAX_SNIPPET_CHARS,
        )
        if not title:
            continue
        seen_urls.add(url)
        normalized.append({"title": title, "url": url, "snippet": snippet})
        if len(normalized) >= MAX_RESULTS:
            break

    if not normalized:
        detail = f" ({'; '.join(failures)})" if failures else ""
        raise RuntimeError(f"Web search returned no usable results{detail}.")

    return {
        "status": "ok",
        "operation": "web_search",
        "results": normalized,
        "count": len(normalized),
    }
