from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pyperclip
from PIL import Image

import config
import computer_tools
import filesystem
import jobs
import process_tools
import screen_tools


PASS = "PASS"
FAIL = "FAIL"


class TestFailure(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise TestFailure(message)


def test_root() -> Path:
    root = Path(config.ALLOWED_ROOTS[0]) / "_dexter_tool_tests"
    root.mkdir(parents=True, exist_ok=True)
    return root


def reset_test_root() -> Path:
    root = test_root()
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return root


def run_case(name: str, fn) -> bool:
    started = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - started
        print(f"[{PASS}] {name} ({elapsed:.2f}s)")
        return True
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"[{FAIL}] {name} ({elapsed:.2f}s): {type(exc).__name__}: {exc}")
        return False


def batch1() -> list[bool]:
    root = reset_test_root()

    def filesystem_chain():
        source = root / "alpha.txt"
        source.write_text("alpha\nLEGACY_KEYS = {'old': 'new'}\nomega\n", encoding="utf-8")

        created = filesystem.create_directory_verified(str(root / "Review"))
        check(created["verified"], "directory creation was not verified")

        listing = filesystem.list_directory_entries(str(root))
        check(any(item["name"] == "alpha.txt" for item in listing["entries"]), "list missed alpha.txt")

        names = filesystem.search_paths(str(root), "alpha")
        check(names["count"] == 1, "name search did not find exactly one alpha file")

        content = filesystem.search_text_contents(str(root), "LEGACY_KEYS")
        check(content["count"] == 1, "content search did not find LEGACY_KEYS")
        check(content["matches"][0]["line_number"] == 2, "content search line number is wrong")

        read = filesystem.read_text_file(str(source), start_line=2, max_lines=1)
        check("LEGACY_KEYS" in read["content"], "bounded read returned wrong line")

        copied_path = root / "Review" / "alpha-copy.txt"
        copied = filesystem.copy_file_verified(str(source), str(copied_path))
        check(copied["verified"] and copied["hashes_match"], "copy verification failed")

        renamed = filesystem.rename_file_verified(str(copied_path), "alpha-renamed.txt")
        renamed_path = Path(renamed["destination"])
        check(renamed_path.is_file(), "rename destination missing")

        moved_path = root / "alpha-moved.txt"
        moved = filesystem.move_file_verified(str(renamed_path), str(moved_path))
        check(moved["verified"] and moved_path.is_file(), "move verification failed")

        info = filesystem.get_path_info(str(moved_path))
        check(info["type"] == "file", "file_info returned wrong type")

        digest = filesystem.get_file_sha256(str(moved_path))
        check(len(digest["sha256"]) == 64, "sha256 has wrong length")

    def boundary_and_overwrite_guards():
        inside = root / "guard.txt"
        inside.write_text("guard", encoding="utf-8")

        try:
            filesystem.resolve_allowed_path(r"C:\Windows\System32\notepad.exe")
        except ValueError:
            pass
        else:
            raise TestFailure("outside-root path was not rejected")

        destination = root / "already.txt"
        destination.write_text("existing", encoding="utf-8")
        try:
            filesystem.copy_file_verified(str(inside), str(destination))
        except FileExistsError:
            pass
        else:
            raise TestFailure("copy overwrite was not rejected")

    def launcher_policy_guards():
        import launcher

        dangerous = root / "do-not-run.cmd"
        dangerous.write_text("@echo off\necho should-not-run\n", encoding="utf-8")

        try:
            launcher.open_allowed_file(str(dangerous))
        except PermissionError:
            pass
        else:
            raise TestFailure("open_file allowed an executable/script type")

        try:
            launcher.launch_app("notepad", arguments=["C:\\Windows\\win.ini"])
        except PermissionError:
            pass
        else:
            raise TestFailure("configured app accepted disallowed arbitrary arguments")

    def mcp_registration_smoke():
        import asyncio
        import server

        registered = asyncio.run(server.mcp.list_tools())
        names = {tool.name for tool in registered}
        required = {
            "search_files",
            "read_file",
            "open_app",
            "get_clipboard",
            "get_foreground_window",
            "computer_action",
            "start_click_job",
            "wait_for_image",
        }
        missing = sorted(required - names)
        check(not missing, f"MCP registration is missing tools: {missing}")
        check(len(names) >= 30, f"unexpectedly few MCP tools registered: {len(names)}")

    return [
        run_case("Batch 1 filesystem chain", filesystem_chain),
        run_case("Batch 1 boundary/overwrite guards", boundary_and_overwrite_guards),
        run_case("Batch 1 launcher policy guards", launcher_policy_guards),
        run_case("MCP tool registration/schema smoke", mcp_registration_smoke),
    ]


def batch2() -> list[bool]:
    def clipboard_roundtrip_restore():
        original = pyperclip.paste()
        marker = f"dexter-test-{time.time_ns()}"
        try:
            from clipboard_tools import set_clipboard_text, get_clipboard_text
            result = set_clipboard_text(marker)
            check(result["verified"], "clipboard set was not verified")
            observed = get_clipboard_text()
            check(observed["text"] == marker, "clipboard roundtrip mismatch")
        finally:
            pyperclip.copy(original)

    def process_wrapper():
        result = process_tools.process_running(pid=os.getpid())
        check(result["running"], "current test process was not found")
        listing = process_tools.list_processes_info(query="python", max_results=100)
        check(any(item["pid"] == os.getpid() for item in listing["processes"]), "process list missed current python")

    return [
        run_case("Batch 2 clipboard roundtrip + restore", clipboard_roundtrip_restore),
        run_case("Batch 2 process wrapper", process_wrapper),
    ]


def batch3() -> list[bool]:
    root = reset_test_root()

    def screenshot_and_template_match():
        screenshot_path = root / "screen.png"
        capture = screen_tools.capture_screenshot(str(screenshot_path))
        check(capture["verified"], "screenshot was not verified")

        image = Image.open(screenshot_path)
        # Crop a distinctive-ish region near the center. This is only a smoke
        # test of our wrapper/coordinate path, not a proof of OpenCV itself.
        cx = max(0, image.width // 2 - 40)
        cy = max(0, image.height // 2 - 40)
        crop = image.crop((cx, cy, min(cx + 80, image.width), min(cy + 80, image.height)))
        template_path = root / "template.png"
        crop.save(template_path)

        found = screen_tools.find_image_on_screen(
            str(template_path),
            confidence=0.99,
        )
        check(found["found"], "template cropped from current screen was not found")

    def job_scheduler_without_real_clicks():
        calls = []

        def fake_click_mouse(**kwargs):
            calls.append(kwargs)
            return {"status": "ok"}

        with patch.object(jobs, "click_mouse", side_effect=fake_click_mouse):
            started = jobs.start_click_interval_job(
                interval_seconds=0.10,
                position_mode="current",
                max_runs=3,
            )
            job_id = started["job"]["job_id"]

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                state = jobs.get_job_status(job_id)["job"]
                if state["state"] != "running":
                    break
                time.sleep(0.03)

            final = jobs.get_job_status(job_id)["job"]
            check(final["state"] == "completed", f"job ended as {final['state']}")
            check(final["runs"] == 3, f"expected 3 runs, got {final['runs']}")
            check(len(calls) == 3, f"fake click called {len(calls)} times")

            # Also prove that an unbounded job can be cancelled quickly.
            started2 = jobs.start_click_interval_job(
                interval_seconds=0.10,
                position_mode="current",
            )
            job_id2 = started2["job"]["job_id"]
            time.sleep(0.15)
            cancelled = jobs.cancel_background_job(job_id2, wait_seconds=1.0)["job"]
            check(cancelled["state"] == "cancelled", f"cancel ended as {cancelled['state']}")

    def computer_readonly_smoke():
        size = computer_tools.perform_computer_action("screen_size")
        pos = computer_tools.perform_computer_action("cursor_position")
        check(size["width"] > 0 and size["height"] > 0, "invalid screen size")
        check(isinstance(pos["x"], int) and isinstance(pos["y"], int), "invalid cursor position")

    return [
        run_case("Batch 3 screenshot + template matching", screenshot_and_template_match),
        run_case("Batch 3 background scheduler (mocked clicks)", job_scheduler_without_real_clicks),
        run_case("Batch 3 screen/cursor smoke", computer_readonly_smoke),
    ]


def chains() -> list[bool]:
    root = reset_test_root()

    def chain_find_copy_move_verify():
        project = root / "Project"
        review = root / "Review"
        project.mkdir()
        review.mkdir()
        target = project / "schema.py"
        target.write_text(
            "SCHEMA_VERSION = 4\nLEGACY_KEYS = {'raw': 'raw_lyrics'}\n",
            encoding="utf-8",
        )

        hits = filesystem.search_text_contents(str(project), "LEGACY_KEYS", extensions=[".py"])
        check(hits["count"] == 1, "chain could not resolve defining file")
        source = hits["matches"][0]["path"]

        copied = review / "schema-copy.py"
        filesystem.copy_file_verified(source, str(copied))
        final = review / "schema-final.py"
        filesystem.move_file_verified(str(copied), str(final))

        check(target.is_file(), "chain modified original")
        check(final.is_file(), "chain final file missing")
        check(
            filesystem.sha256_file(target) == filesystem.sha256_file(final),
            "chain final hash differs from original",
        )

    def chain_click_until_visual_condition_mocked():
        click_count = {"value": 0}

        def fake_click_mouse(**kwargs):
            click_count["value"] += 1
            return {"status": "ok"}

        def fake_find_image(**kwargs):
            return {
                "status": "ok",
                "found": click_count["value"] >= 2,
                "match": {"center_x": 10, "center_y": 10} if click_count["value"] >= 2 else None,
            }

        # The path only needs to satisfy the job API here because the screen
        # matcher itself is replaced at the tester layer; production code has
        # no test-only branch.
        fake_template = root / "stop.png"
        Image.new("RGB", (8, 8), "white").save(fake_template)

        with (
            patch.object(jobs, "click_mouse", side_effect=fake_click_mouse),
            patch.object(jobs, "find_image_on_screen", side_effect=fake_find_image),
        ):
            started = jobs.start_click_interval_job(
                interval_seconds=0.10,
                position_mode="current",
                stop_image_path=str(fake_template),
                max_seconds=2.0,
            )
            job_id = started["job"]["job_id"]

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                state = jobs.get_job_status(job_id)["job"]
                if state["state"] != "running":
                    break
                time.sleep(0.03)

            final = jobs.get_job_status(job_id)["job"]
            check(final["state"] == "completed", f"visual chain ended as {final['state']}")
            check(final["stop_reason"] == "stop_image_found", f"wrong stop reason: {final['stop_reason']}")
            check(final["runs"] == 2, f"expected 2 clicks before visual stop, got {final['runs']}")

    return [
        run_case("Chain: search -> copy -> move -> hash verify", chain_find_copy_move_verify),
        run_case("Chain: click loop -> visual stop condition (mocked)", chain_click_until_visual_condition_mocked),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Thin batch tester for Dexter tools. It does not alter server code."
    )
    parser.add_argument(
        "--batch",
        choices=["1", "2", "3", "chains", "all"],
        default="all",
        help="Run one implementation batch, chained tests, or everything.",
    )
    args = parser.parse_args()

    print(f"Dexter tool tests using: {test_root()}")
    print("Visible desktop actions are deliberately not auto-clicked or auto-typed.\n")

    results: list[bool] = []

    if args.batch in {"1", "all"}:
        results += batch1()
    if args.batch in {"2", "all"}:
        results += batch2()
    if args.batch in {"3", "all"}:
        results += batch3()
    if args.batch in {"chains", "all"}:
        results += chains()

    passed = sum(results)
    total = len(results)
    print(f"\nResult: {passed}/{total} passed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
