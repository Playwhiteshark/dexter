"""Safe, automated Goose/Gemma selection benchmark for Dexter tools v0.2.

The runner starts Goose non-interactively with an isolated config containing the
benchmark Dexter MCP extension. The benchmark server reuses production schemas,
simulates effectful tools, and records a startup marker plus every selected
Dexter tool/argument pair to JSONL.

Goose may expose its own tools with overlapping names (notably read_image), so
cases that validate a Dexter implementation name the dexter_tools extension
explicitly. A missing benchmark-server startup marker is reported as a harness
error rather than a model failure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_CASE_TIMEOUT_SECONDS = 300
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "gemma4-jarvis:12b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11435"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class ExpectedCall:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    expected_calls: tuple[ExpectedCall, ...]
    expected_text: str | None = None
    forbidden_calls: tuple[str, ...] = ()


def _root() -> Path:
    root = Path(__file__).resolve().parent / "benchmark_runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fixtures(root: Path) -> list[Case]:
    fixtures = root / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    good = fixtures / "good.txt"
    good.write_text("benchmark recovery file\n", encoding="utf-8")
    red = fixtures / "red.webp"
    Image.new("RGB", (32, 32), "red").save(red, format="WEBP")
    corrupt = fixtures / "corrupt.webp"
    corrupt.write_bytes(b"not a real image")

    return [
        Case(
            "file_hash",
            f"Use file_sha256 from the dexter_tools extension on {good}. Then briefly report the hash.",
            (ExpectedCall("file_sha256", {"path": str(good)}),),
        ),
        Case(
            "foreground_window",
            "Use get_foreground_window from the dexter_tools extension and tell me the current foreground "
            "window and process. Do not change anything.",
            (ExpectedCall("get_foreground_window"),),
        ),
        Case(
            "close_named_app_windows",
            "Close every open VS Code window on the current desktop. There may be "
            "several Code.exe processes. Use the exact process name Code.exe and do "
            "not close any other application.",
            (ExpectedCall("close_all_windows_for_app", {"process_name": "Code.exe"}),),
            forbidden_calls=("close_every_app_on_active_desktop",),
        ),
        Case(
            "remember_current_project",
            "Remember the currently active VS Code project as FriendBox. Do not "
            "guess or ask for its path; use the tool that reads the VS Code bridge.",
            (ExpectedCall("remember_current_project", {"name": "FriendBox"}),),
            forbidden_calls=("remember_resource", "search_saved"),
        ),
        Case(
            "open_saved_project_directly",
            "Open the previously saved project whose name is exactly FriendBox. "
            "Use the action that resolves saved resource names internally; do not "
            "call search_saved first.",
            (ExpectedCall("open_resource", {"name": "FriendBox"}),),
            forbidden_calls=("search_saved",),
        ),
        Case(
            "restore_workspace_directly",
            "Restore my saved AI workspace. Call restore_workspace directly with "
            "the name 'AI workspace'; do not call search_saved first.",
            (ExpectedCall("restore_workspace", {"name": "AI workspace"}),),
            forbidden_calls=("search_saved",),
        ),
        Case(
            "webp_vision",
            f"Use read_image from the dexter_tools extension on {red}. If you can see that it is red, "
            "reply with exactly RED on its own line.",
            (ExpectedCall("read_image", {"path": str(red)}),),
            "RED",
        ),
        Case(
            "bad_image_recovery",
            f"First use read_image from the dexter_tools extension on {corrupt}. It should fail. "
            f"After that failure, use file_sha256 from dexter_tools on {good}. If you successfully continue "
            "after the image error, reply with exactly RECOVERED on its own line.",
            (
                ExpectedCall("read_image", {"path": str(corrupt)}),
                ExpectedCall("file_sha256", {"path": str(good)}),
            ),
            "RECOVERED",
        ),
    ]


def _prepare_isolated_goose_root(root: Path) -> Path:
    """Create a minimal isolated Goose root containing only the benchmark MCP config."""
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = Path(os.environ.get("PYTHON", shutil.which("python") or "python"))
    benchmark_server = Path(__file__).resolve().parent / "benchmark_server.py"

    def yaml_quote(value: Path) -> str:
        # YAML single quotes keep Windows backslashes literal.
        return "'" + str(value).replace("'", "''") + "'"

    config = (
        "extensions:\n"
        "  dexter_tools:\n"
        "    enabled: true\n"
        "    type: stdio\n"
        "    name: dexter_tools\n"
        f"    cmd: {yaml_quote(python)}\n"
        "    args:\n"
        f"      - {yaml_quote(benchmark_server)}\n"
        "    timeout: 300\n"
    )
    (config_dir / "config.yaml").write_text(config, encoding="utf-8")
    return root


def _run_goose(
    prompt: str,
    timeout: int,
    goose_root: Path,
    trace_path: Path,
    fixture_root: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GOOSE_MODE"] = "auto"
    env["GOOSE_DISABLE_SESSION_NAMING"] = "true"
    env["GOOSE_RANDOM_THINKING_MESSAGES"] = "false"
    env["DEXTER_BENCHMARK_TRACE"] = str(trace_path)
    env["DEXTER_BENCHMARK_FIXTURE_ROOT"] = str(fixture_root)
    env["GOOSE_PATH_ROOT"] = str(goose_root)

    # Defaults mirror Dexter's pinned real stack but can be overridden without
    # editing source. Environment values take precedence over Goose config.
    env.setdefault("GOOSE_PROVIDER", os.environ.get("DEXTER_BENCHMARK_PROVIDER", DEFAULT_PROVIDER))
    env.setdefault("GOOSE_MODEL", os.environ.get("DEXTER_BENCHMARK_MODEL", DEFAULT_MODEL))
    env.setdefault("OLLAMA_HOST", os.environ.get("DEXTER_BENCHMARK_OLLAMA_HOST", DEFAULT_OLLAMA_HOST))

    return subprocess.run(
        ["goose", "run", "-t", prompt],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        env=env,
    )


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _server_started(trace: list[dict[str, Any]]) -> bool:
    return any(item.get("event") == "server_started" for item in trace)


def _args_match(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if str(observed.get(key)) != str(value):
            return False
    return True


def _calls_match(trace: list[dict[str, Any]], case: Case) -> tuple[bool, list[dict[str, Any]]]:
    # Grade tool selection from entry records. A tool can fail before producing a
    # normal result (the corrupt-image case is intentional), so post-call records
    # alone are not a reliable record of what the model selected.
    calls = [item for item in trace if item.get("event") == "tool_started" and item.get("tool")]
    if any(item.get("tool") in case.forbidden_calls for item in calls):
        return False, calls
    cursor = 0
    matched: list[dict[str, Any]] = []
    for expected in case.expected_calls:
        found = None
        while cursor < len(calls):
            item = calls[cursor]
            cursor += 1
            if item.get("tool") == expected.tool and _args_match(item.get("arguments") or {}, expected.arguments):
                found = item
                break
        if found is None:
            return False, calls
        matched.append(found)
    return True, calls


def _has_exact_answer_line(output: str, expected: str) -> bool:
    """Match a standalone answer, not incidental text such as red.webp in a path."""
    clean = _ANSI_ESCAPE.sub("", output)
    target = expected.strip()
    return any(line.strip() == target for line in clean.splitlines())


def _context_probe() -> int:
    """Print current production schema metrics; real prompt tokens come from Gateway tracing."""
    here = Path(__file__).resolve().parent
    return subprocess.call([str(here / ".venv" / "Scripts" / "python.exe"), str(here / "schema_audit.py")])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run safe real-Goose integration cases against Dexter.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_CASE_TIMEOUT_SECONDS)
    parser.add_argument("--case", help="Run only one named case.")
    parser.add_argument("--context", action="store_true", help="Run the live schema-size audit instead of model cases.")
    args = parser.parse_args()

    if args.context:
        return _context_probe()
    if shutil.which("goose") is None:
        raise RuntimeError("goose is not on PATH.")

    root = _root()
    cases = _fixtures(root)
    if args.case:
        cases = [case for case in cases if case.name == args.case]
        if not cases:
            raise ValueError(f"Unknown case: {args.case}")

    run_dir = root / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    summary: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="dexter-goose-benchmark-") as temp:
        goose_root = _prepare_isolated_goose_root(Path(temp))
        fixture_root = root / "fixtures"
        for case in cases:
            trace_path = run_dir / f"{case.name}.tools.jsonl"
            started = time.time()
            try:
                proc = _run_goose(case.prompt, args.timeout, goose_root, trace_path, fixture_root)
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                proc = subprocess.CompletedProcess(exc.cmd, 124, exc.stdout or "", exc.stderr or "timeout")
                timed_out = True

            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            trace = _read_trace(trace_path)
            harness_ok = _server_started(trace)
            calls_ok, observed_calls = _calls_match(trace, case) if harness_ok else (False, [])
            text_ok = case.expected_text is None or _has_exact_answer_line(output, case.expected_text)
            passed = proc.returncode == 0 and not timed_out and harness_ok and calls_ok and text_ok
            status = "PASS" if passed else "HARNESS" if not harness_ok else "FAIL"

            record = {
                "case": case.name,
                "prompt": case.prompt,
                "status": status,
                "passed": passed,
                "harness_ok": harness_ok,
                "returncode": proc.returncode,
                "seconds": round(time.time() - started, 3),
                "expected_calls": [
                    {"tool": item.tool, "arguments": item.arguments} for item in case.expected_calls
                ],
                "forbidden_calls": list(case.forbidden_calls),
                "observed_calls": [
                    {"tool": item.get("tool"), "arguments": item.get("arguments"), "error": item.get("error")}
                    for item in observed_calls
                ],
                "calls_ok": calls_ok,
                "expected_text": case.expected_text,
                "text_seen": text_ok,
            }
            summary.append(record)
            (run_dir / f"{case.name}.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
            (run_dir / f"{case.name}.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
            (run_dir / f"{case.name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            if status == "HARNESS":
                print(f"[HARNESS ERROR] {case.name}: benchmark MCP server did not start ({record['seconds']:.1f}s)")
            else:
                print(f"[{'PASS' if passed else 'FAIL'}] {case.name} ({record['seconds']:.1f}s)")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    passed_count = sum(1 for item in summary if item["passed"])
    harness_errors = sum(1 for item in summary if not item["harness_ok"])
    print(f"\nResult: {passed_count}/{len(summary)} passed")
    if harness_errors:
        print(f"Harness errors: {harness_errors} (these are not model/tool failures)")
    print(f"Run logs + tool-call traces: {run_dir}")
    return 0 if passed_count == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
