"""Deterministic spoken-name resolution for saved operational records."""

from __future__ import annotations

import re
from typing import Iterable


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_GENERIC_REQUEST_WORDS = frozenset(
    {
        "a",
        "an",
        "app",
        "application",
        "folder",
        "launch",
        "my",
        "open",
        "project",
        "projects",
        "resource",
        "resources",
        "saved",
        "start",
        "the",
        "workspace",
        "workspaces",
    }
)


def normalize_label(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(str(value or "").casefold()))


def _tokens(value: str, *, remove_generic: bool) -> tuple[str, ...]:
    tokens = tuple(_TOKEN_RE.findall(str(value or "").casefold()))
    if not remove_generic:
        return tokens
    meaningful = tuple(token for token in tokens if token not in _GENERIC_REQUEST_WORDS)
    return meaningful or tokens


def _identity_score(query: str, identity: str, identity_priority: int) -> tuple[int, int, int] | None:
    query_normalized = normalize_label(query)
    identity_normalized = normalize_label(identity)
    if not query_normalized or not identity_normalized:
        return None
    if query_normalized == identity_normalized:
        return (0, 0, identity_priority)

    query_tokens = _tokens(query, remove_generic=False)
    identity_tokens = _tokens(identity, remove_generic=False)
    query_core = _tokens(query, remove_generic=True)
    identity_core = _tokens(identity, remove_generic=True)
    if query_core == identity_core:
        # Prefer the real saved name that preserves more of the user's phrase.
        # Thus "open my AI workspace" selects a stored "AI Workspace" over a
        # stored "AI" when both exist, while either one works independently.
        return (1, abs(len(query_tokens) - len(identity_tokens)), identity_priority)

    # Speech-to-text commonly separates product/project compounds: a saved
    # ``FriendBox`` should resolve from spoken ``friend box`` (and vice versa).
    if "".join(query_core) == "".join(identity_core):
        return (2, abs(len(query_tokens) - len(identity_tokens)), identity_priority)

    query_set = set(query_core)
    identity_set = set(identity_core)
    if query_set and identity_set and (
        query_set.issubset(identity_set) or identity_set.issubset(query_set)
    ):
        return (3, abs(len(query_set) - len(identity_set)), identity_priority)

    # Retain the useful partial-name behavior for inputs such as "friend" ->
    # "FriendBox", but only after exact and token-aware matches have failed.
    if len(query_normalized) >= 3 and (
        query_normalized in identity_normalized
        or identity_normalized in query_normalized
    ):
        return (
            4,
            abs(len(query_normalized) - len(identity_normalized)),
            identity_priority,
        )
    return None


def rank_named_records(
    query: str,
    records: Iterable[dict],
) -> list[tuple[tuple[int, int, int], dict]]:
    """Rank saved records by names/aliases without inventing a target."""
    ranked: list[tuple[tuple[int, int, int], str, dict]] = []
    for record in records:
        identities = [record.get("name"), *(record.get("aliases") or [])]
        scores = [
            score
            for index, identity in enumerate(identities)
            if isinstance(identity, str)
            for score in [_identity_score(query, identity, 0 if index == 0 else 1)]
            if score is not None
        ]
        if scores:
            ranked.append((min(scores), normalize_label(record.get("name", "")), record))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [(score, record) for score, _name, record in ranked]


def resolve_named_record(
    query: str,
    records: Iterable[dict],
    *,
    record_label: str,
) -> dict:
    """Resolve exactly one best stored record or report grounded candidates."""
    ranked = rank_named_records(query, records)
    if not ranked:
        raise LookupError(f"No saved {record_label} matches {query!r}.")

    best_score = ranked[0][0]
    best = [record for score, record in ranked if score == best_score]
    if len(best) != 1:
        names = ", ".join(sorted(str(record.get("name")) for record in best))
        raise LookupError(
            f"Saved {record_label} name is ambiguous: {query!r}. Matches: {names}"
        )
    return best[0]
