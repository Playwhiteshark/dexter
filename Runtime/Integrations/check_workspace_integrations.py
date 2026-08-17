from __future__ import annotations

import json
import os
import time
from pathlib import Path

HOST_NAME = "com.dexter.workspace_bridge"
EXTENSION_ID = "bchcobokpbkipfppdhkboghbpaglgngb"
FRESH_SECONDS = 20.0


def fresh_sessions(path: Path) -> list[dict]:
    now = time.time()
    result = []
    if not path.is_dir():
        return result
    for file in path.glob("*.json"):
        try:
            item = json.loads(file.read_text(encoding="utf-8"))
            updated = float(item.get("updated_unix", 0))
        except Exception:
            continue
        if now - updated <= FRESH_SECONDS:
            result.append(item)
    return result


def main() -> int:
    local = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Dexter"
    browser_root = local / "browser_bridge"
    vscode_root = local / "vscode_bridge"
    browser_sessions = fresh_sessions(browser_root / "sessions")
    vscode_sessions = fresh_sessions(vscode_root / "sessions")

    print("Dexter workspace integration status")
    print(f"Browser native host exe: {(browser_root / 'dexter_browser_bridge_host.exe').is_file()}")
    print(f"Browser native manifest: {(browser_root / (HOST_NAME + '.json')).is_file()}")
    print(f"Expected browser extension ID: {EXTENSION_ID}")
    print(f"Active browser bridge sessions: {len(browser_sessions)}")
    for item in browser_sessions:
        print(f"  - {item.get('browser_kind')} session={item.get('session_id')} pid={item.get('pid')}")
    print(f"Active VS Code bridge sessions: {len(vscode_sessions)}")
    for item in vscode_sessions:
        print(
            f"  - {item.get('workspace_name') or '<no workspace>'} "
            f"session={item.get('session_id')} pid={item.get('pid')} "
            f"focused={bool(item.get('focused'))} last_focused={item.get('last_focused_unix') or 0}"
        )

    ready = bool(browser_sessions) and bool(vscode_sessions)
    if not browser_sessions:
        print("Browser bridge is not active. Load/reload the Dexter Workspace Bridge extension in the browser.")
    if not vscode_sessions:
        print(
            "VS Code bridge is not active. Install/update Dexter Workspace State, "
            "then restart or reload the VS Code window."
        )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
