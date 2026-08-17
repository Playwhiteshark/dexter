"""Internal per-application workspace state capture/restore.

This module deliberately stays behind save_workspace/restore_workspace. It does
not add model-facing tools. Unsupported or unavailable integrations degrade to a
warning instead of making the whole workspace unusable.
"""

from __future__ import annotations

from browser_state import capture_browser_state, restore_browser_state
from explorer_state import capture_explorer_state, restore_explorer_state
from terminal_state import capture_terminal_state
from vscode_state import capture_vscode_state, restore_vscode_state
from workspace_providers import provider_for_process_name, provider_for_saved_window


def capture_window_app_state(window: dict) -> tuple[dict | None, str | None]:
    provider = provider_for_process_name(window.get("process_name"))
    try:
        if provider == "chromium_browser":
            state = capture_browser_state(window)
            skipped = int(state.get("skipped_tab_count", 0))
            warning = (
                f"browser state skipped {skipped} non-restorable internal/devtools tab(s)"
                if skipped else None
            )
            return state, warning
        if provider == "vscode":
            return capture_vscode_state(window), None
        if provider == "explorer":
            return capture_explorer_state(window), None
        if provider == "windows_terminal":
            return capture_terminal_state(window), None
    except Exception as exc:
        return None, f"{provider or 'app'} state capture unavailable: {type(exc).__name__}: {exc}"
    return None, None


def restore_window_app_state(saved: dict, current_window: dict) -> tuple[dict | None, str | None]:
    state = saved.get("app_state")
    if not isinstance(state, dict):
        return None, None

    provider = provider_for_saved_window(saved)
    try:
        if provider == "chromium_browser" and state.get("kind") == "chromium_browser":
            result = restore_browser_state(saved, current_window)
            failed = int(result.get("failed_tab_count", 0))
            warnings = []
            if failed:
                warnings.append(f"browser restore could not recreate {failed} saved tab/group item(s)")
            if result.get("active_tab_ready") is False:
                warnings.append("browser active tab had not committed a real page before the restore timeout")
            return result, "; ".join(warnings) if warnings else None
        if provider == "vscode" and state.get("kind") == "vscode":
            return restore_vscode_state(saved, current_window), None
        if provider == "explorer" and state.get("kind") == "explorer":
            return restore_explorer_state(saved, current_window), None
        # Windows Terminal state is consumed at launch time by the provider.
    except Exception as exc:
        return None, f"{provider or 'app'} state restore unavailable: {type(exc).__name__}: {exc}"
    return None, None
