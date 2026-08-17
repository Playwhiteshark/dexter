"""Chromium-family workspace state through Dexter's browser bridge."""

from __future__ import annotations

import time
from urllib.parse import urlparse

from browser_bridge import (
    browser_kind_for_process,
    capture_browser_windows,
    restore_browser_window,
)


def _norm(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _browser_suffixes(browser_kind: str) -> tuple[str, ...]:
    if browser_kind == "brave":
        return (" - brave", " - brave private")
    if browser_kind == "edge":
        return (" - microsoft edge", " - inprivate - microsoft edge")
    return (" - google chrome", " - chrome")


def _title_without_browser_suffix(title: str, browser_kind: str) -> str:
    value = _norm(title)
    for suffix in _browser_suffixes(browser_kind):
        if value.endswith(suffix):
            return value[: -len(suffix)].strip()
    return value


def _active_tab(window: dict) -> dict | None:
    tabs = window.get("tabs") or []
    for tab in tabs:
        if isinstance(tab, dict) and tab.get("active"):
            return tab
    return tabs[0] if tabs and isinstance(tabs[0], dict) else None


def _geometry_distance(os_window: dict, browser_window: dict) -> int:
    bounds = os_window.get("visible_bounds") or os_window.get("bounds") or {}
    if not all(key in bounds for key in ("left", "top", "right", "bottom")):
        return 10**9
    if not all(key in browser_window for key in ("left", "top", "width", "height")):
        return 10**9
    browser_right = int(browser_window["left"]) + int(browser_window["width"])
    browser_bottom = int(browser_window["top"]) + int(browser_window["height"])
    return (
        abs(int(bounds["left"]) - int(browser_window["left"]))
        + abs(int(bounds["top"]) - int(browser_window["top"]))
        + abs(int(bounds["right"]) - browser_right)
        + abs(int(bounds["bottom"]) - browser_bottom)
    )


def match_browser_window(os_window: dict, browser_windows: list[dict], browser_kind: str) -> dict | None:
    if not browser_windows:
        return None

    os_title = _title_without_browser_suffix(os_window.get("title") or "", browser_kind)
    title_matches: list[dict] = []
    for candidate in browser_windows:
        tab = _active_tab(candidate)
        tab_title = _norm(tab.get("title") if tab else "")
        if tab_title and (tab_title == os_title or tab_title in os_title or os_title in tab_title):
            title_matches.append(candidate)

    if len(title_matches) == 1:
        return title_matches[0]

    pool = title_matches or browser_windows
    ranked = sorted(pool, key=lambda candidate: _geometry_distance(os_window, candidate))
    if len(ranked) == 1:
        return ranked[0]

    best = _geometry_distance(os_window, ranked[0])
    second = _geometry_distance(os_window, ranked[1])
    # Require a meaningful geometry lead; extension-window dimensions can differ
    # slightly from DWM-visible bounds because Chromium reports its full frame.
    if best < 400 and second - best >= 40:
        return ranked[0]
    return None


def _restorable_url(url: str | None) -> bool:
    if not isinstance(url, str) or not url.strip():
        return False
    parsed = urlparse(url)
    scheme = parsed.scheme.casefold()
    # Browser-internal, extension/devtools, and about:blank pages are not
    # useful workspace navigation targets. Persist normal page/file URLs only.
    return scheme in {"http", "https", "file"}


def capture_browser_state(os_window: dict) -> dict:
    browser_kind = browser_kind_for_process(os_window.get("process_name"))
    if not browser_kind:
        raise LookupError("Not a supported Chromium browser window.")

    browser_windows = capture_browser_windows(browser_kind)
    matched = match_browser_window(os_window, browser_windows, browser_kind)
    if matched is None:
        raise LookupError(
            f"Browser bridge is connected for {browser_kind}, but the OS browser window could not be matched."
        )

    tabs = []
    active_index = 0
    skipped_tab_count = 0
    for index, tab in enumerate(matched.get("tabs") or []):
        if not isinstance(tab, dict) or not _restorable_url(tab.get("url")):
            skipped_tab_count += 1
            continue
        saved = {
            "url": tab["url"],
            "title": tab.get("title") or "",
            "pinned": bool(tab.get("pinned")),
            "active": bool(tab.get("active")),
            "group_key": tab.get("group_key"),
        }
        if saved["active"]:
            active_index = len(tabs)
        tabs.append(saved)

    if not tabs:
        raise RuntimeError("Browser window had no restorable tabs.")

    groups = [
        {
            "key": group.get("key"),
            "title": group.get("title") or "",
            "color": group.get("color"),
            "collapsed": bool(group.get("collapsed")),
        }
        for group in (matched.get("groups") or [])
        if isinstance(group, dict) and group.get("key")
    ]

    return {
        "kind": "chromium_browser",
        "browser_kind": browser_kind,
        "tabs": tabs,
        "groups": groups,
        "active_index": active_index,
        "captured_tab_count": len(tabs),
        "skipped_tab_count": skipped_tab_count,
    }


def restore_browser_state(saved: dict, current_window: dict) -> dict:
    state = saved.get("app_state") or {}
    browser_kind = state.get("browser_kind") or browser_kind_for_process(saved.get("process_name"))
    if not browser_kind:
        raise LookupError("Saved browser state does not identify the browser kind.")

    # A freshly launched Chromium window can become a stable Win32 HWND before
    # its MV3 extension/native-messaging bridge has reconnected. Give that
    # internal provider a short bounded grace period rather than racing it.
    deadline = time.monotonic() + 4.0
    matched = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            browser_windows = capture_browser_windows(browser_kind)
            matched = match_browser_window(current_window, browser_windows, browser_kind)
        except Exception as exc:
            last_error = exc
        if matched is not None:
            break
        time.sleep(0.10)
    if matched is None:
        detail = f" Last bridge error: {last_error}" if last_error else ""
        raise LookupError(
            "Browser bridge could not match the restored browser HWND to an extension window." + detail
        )

    session_id = matched.get("_bridge_session")
    window_id = matched.get("id")
    if not isinstance(session_id, str) or not isinstance(window_id, int):
        raise RuntimeError("Browser bridge match lacked a session/window identifier.")

    # Sanitize old snapshots too. v6 briefly allowed about:blank and could
    # therefore faithfully restore a white tab captured during navigation.
    clean_tabs = [
        tab for tab in (state.get("tabs") or [])
        if isinstance(tab, dict) and _restorable_url(tab.get("url"))
    ]
    if not clean_tabs:
        raise RuntimeError("Saved browser state has no restorable http/https/file tabs.")

    clean_state = dict(state)
    clean_state["tabs"] = clean_tabs
    active_from_flag = next(
        (index for index, tab in enumerate(clean_tabs) if tab.get("active")),
        None,
    )
    clean_state["active_index"] = (
        active_from_flag
        if active_from_flag is not None
        else max(0, min(int(state.get("active_index", 0) or 0), len(clean_tabs) - 1))
    )
    result = restore_browser_window(browser_kind, session_id, window_id, clean_state)
    return {
        "status": "ok",
        "kind": "chromium_browser",
        "tab_count": int(result.get("tab_count", len(clean_tabs))),
        "browser_kind": browser_kind,
        "reused_tab_count": int(result.get("reused_tab_count", 0)),
        "created_tab_count": int(result.get("created_tab_count", 0)),
        "removed_extra_tab_count": int(result.get("removed_extra_tab_count", 0)),
        "active_tab_ready": bool(result.get("active_tab_ready", True)),
        "active_tab_observed_url": result.get("active_tab_observed_url") or "",
        "failed_tab_count": int(result.get("failed_count", 0)),
    }
