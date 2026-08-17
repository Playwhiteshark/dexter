"""Deterministic validation for Dexter tools v0.2.

No real mouse clicks or keyboard typing are performed. Desktop side effects are
mocked at the test layer; production modules contain no test-mode branches.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from PIL import Image, TiffImagePlugin


TRACE_FILE_NAME = "mutation_trace.jsonl"
_TEST_ROOT: Path | None = None


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_case(name: str, fn) -> bool:
    try:
        fn()
    except Exception as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        return False
    print(f"[PASS] {name}")
    return True


def test_root() -> Path:
    global _TEST_ROOT
    if _TEST_ROOT is None:
        _TEST_ROOT = Path(tempfile.gettempdir()) / "_dexter_tool_tests"
    return _TEST_ROOT


def reset_test_root() -> Path:
    import shutil

    root = test_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def reset_trace() -> None:
    path = test_root() / TRACE_FILE_NAME
    if path.exists():
        path.unlink()


def trace_mutation(operation: str, **details) -> None:
    path = test_root() / TRACE_FILE_NAME
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"operation": operation, **details}, ensure_ascii=False) + "\n")


def _configure_root(root: Path) -> None:
    import config

    config.ALLOWED_ROOTS[:] = [root.resolve()]


def environment_suite() -> list[bool]:
    def import_smoke():
        # This catches the exact class of startup failure previously caused by
        # corrupted NumPy source files.
        for module in ("numpy", "cv2", "pyautogui", "mcp", "PIL"):
            importlib.import_module(module)
        importlib.import_module("server")

    return [run_case("Environment: dependency/import smoke", import_smoke)]


def filesystem_suite() -> list[bool]:
    root = reset_test_root()
    _configure_root(root)
    import filesystem
    import server

    def boundary_and_reads():
        inside = root / "notes.txt"
        inside.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        result = server.read_file(str(inside))
        check(result["content"].startswith("alpha"), "read_file content missing")
        try:
            filesystem.resolve_allowed_path(str(root.parent / "outside.txt"))
        except ValueError:
            pass
        else:
            raise AssertionError("outside-root path was accepted")

    def compact_defaults():
        check(server.list_directory.__defaults__[0] == 50, "list_directory default changed")
        check(server.search_files.__defaults__[-1] == 25, "search_files default changed")
        check(server.read_file.__defaults__[1:] == (160, 12_000), "read_file defaults changed")
        check(server.write_file.__defaults__ == (False,), "write_file overwrite default changed")

    def verified_file_operations():
        review = root / "Review"
        created = server.create_directory(str(review))
        check(created["created"] and review.is_dir(), "create_directory postcondition failed")

        source = root / "source.txt"
        source.write_text("verified", encoding="utf-8")
        copied = review / "copied.txt"
        renamed = review / "renamed.txt"
        moved = root / "moved.txt"

        result = server.copy_file(str(source), str(copied))
        check(result == {"destination": str(copied.resolve())}, f"copy result too noisy: {result}")
        trace_mutation("copy_file", source=str(source), destination=str(copied))

        result = server.rename_file(str(copied), renamed.name)
        check(result == {"destination": str(renamed.resolve())}, f"rename result too noisy: {result}")
        check(renamed.is_file() and not copied.exists(), "rename postconditions failed")
        trace_mutation("rename_file", source=str(copied), destination=str(renamed))

        result = server.move_file(str(renamed), str(moved))
        check(result == {"destination": str(moved.resolve())}, f"move result too noisy: {result}")
        trace_mutation("move_file", source=str(renamed), destination=str(moved))
        check(source.exists() and moved.exists() and not renamed.exists(), "copy/move postconditions failed")

    def verified_text_write():
        target = root / "approval_text.txt"

        result = server.write_file(str(target), "original")
        check(
            result == {
                "path": str(target.resolve()),
                "created": True,
                "overwritten": False,
                "size_bytes": len(b"original"),
            },
            f"write result too noisy or incorrect: {result}",
        )
        check(target.read_bytes() == b"original", "write_file bytes differ from requested content")
        trace_mutation("write_file", path=str(target), overwrite=False)

        try:
            server.write_file(str(target), "must-not-replace")
        except FileExistsError as exc:
            check("already exists" in str(exc).lower(), "overwrite refusal is not actionable")
        else:
            raise AssertionError("write_file overwrote an existing file by default")

        replacement = "changed\nπ"
        result = server.write_file(str(target), replacement, overwrite=True)
        check(
            result == {
                "path": str(target.resolve()),
                "created": False,
                "overwritten": True,
                "size_bytes": len(replacement.encode("utf-8")),
            },
            f"overwrite result too noisy or incorrect: {result}",
        )
        check(
            target.read_bytes() == replacement.encode("utf-8"),
            "overwrite did not preserve exact UTF-8 bytes",
        )
        trace_mutation("write_file", path=str(target), overwrite=True)

        extensionless = root / "Dockerfile"
        server.write_file(str(extensionless), "FROM scratch\n")
        check(extensionless.read_bytes() == b"FROM scratch\n", "extensionless file write failed")

        directory_target = root / "not-a-file.txt"
        directory_target.mkdir()
        try:
            server.write_file(str(directory_target), "no")
        except IsADirectoryError:
            pass
        else:
            raise AssertionError("write_file accepted a directory as the target file")

        missing_parent = root / "missing-parent" / "file.txt"
        try:
            server.write_file(str(missing_parent), "no")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("write_file implicitly created a missing parent directory")

        outside = root.parent / "dexter-outside-write.txt"
        try:
            server.write_file(str(outside), "no")
        except ValueError:
            pass
        else:
            raise AssertionError("write_file accepted an outside-root path")
        finally:
            outside.unlink(missing_ok=True)

    def undecodable_text_errors_cleanly():
        path = root / "undefined-cp1252.txt"
        path.write_bytes(b"\x8f")
        try:
            server.read_file(str(path))
        except ValueError as exc:
            check("could not be decoded" in str(exc).lower(), "decode failure is not actionable")
        else:
            raise AssertionError("undefined cp1252 byte was accepted as text")

    def overwrite_refused():
        source = root / "a.txt"
        dest = root / "b.txt"
        source.write_text("a", encoding="utf-8")
        dest.write_text("b", encoding="utf-8")
        try:
            server.copy_file(str(source), str(dest))
        except FileExistsError as exc:
            check("already exists" in str(exc).lower(), "failure is not actionable")
        else:
            raise AssertionError("copy overwrote existing destination")

    return [
        run_case("Filesystem: root boundary + read", boundary_and_reads),
        run_case("Filesystem: compact defaults", compact_defaults),
        run_case("Filesystem: verified create/copy/rename/move + compact results", verified_file_operations),
        run_case("Filesystem: verified UTF-8 write + overwrite/boundary guards", verified_text_write),
        run_case("Filesystem: undecodable text fails cleanly", undecodable_text_errors_cleanly),
        run_case("Filesystem: overwrite refusal", overwrite_refused),
    ]


def image_suite() -> list[bool]:
    root = reset_test_root()
    _configure_root(root)
    import image_tools

    def assert_png(path: Path, expected_size: tuple[int, int] | None = None):
        result = image_tools.load_image_for_model(str(path))
        check(result.data.startswith(b"\x89PNG\r\n\x1a\n"), f"not normalized PNG: {path}")
        if expected_size is not None:
            check((result.width, result.height) == expected_size, f"wrong size: {(result.width, result.height)}")

    def common_formats_and_mismatch():
        png = root / "a.png"
        jpg = root / "b.jpg"
        webp = root / "c.webp"
        fake_ext = root / "actually-webp.bin"
        Image.new("RGB", (17, 13), "red").save(png)
        Image.new("RGB", (19, 11), "green").save(jpg)
        Image.new("RGB", (23, 9), "blue").save(webp, format="WEBP")
        fake_ext.write_bytes(webp.read_bytes())
        assert_png(png, (17, 13))
        assert_png(jpg, (19, 11))
        assert_png(webp, (23, 9))
        assert_png(fake_ext, (23, 9))

    def animation_first_frame():
        path = root / "animated.gif"
        frames = [Image.new("RGB", (12, 8), "red"), Image.new("RGB", (12, 8), "blue")]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)
        assert_png(path, (12, 8))

    def exif_orientation():
        path = root / "oriented.jpg"
        image = Image.new("RGB", (10, 20), "white")
        exif = image.getexif()
        exif[274] = 6  # rotate 90 degrees clockwise when displayed
        image.save(path, exif=exif)
        assert_png(path, (20, 10))

    def corrupt_rejected_before_media():
        path = root / "bad.webp"
        path.write_bytes(b"not an image")
        try:
            image_tools.load_image_for_model(str(path))
        except ValueError as exc:
            check("decode" in str(exc).lower(), "corrupt-image error is not clear")
        else:
            raise AssertionError("corrupt image was accepted")

    def outside_root_rejected():
        outside = root.parent / "dexter-outside-image.png"
        Image.new("RGB", (4, 4), "red").save(outside)
        try:
            image_tools.load_image_for_model(str(outside))
        except ValueError:
            pass
        else:
            raise AssertionError("outside-root image was accepted")
        finally:
            outside.unlink(missing_ok=True)

    def decompression_guard():
        path = root / "guard.png"
        Image.new("RGB", (20, 20), "red").save(path)
        old = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = 250  # 400 px -> Pillow bomb warning
            try:
                image_tools.load_image_for_model(str(path))
            except ValueError as exc:
                check("too large" in str(exc).lower(), "bomb warning not surfaced safely")
            else:
                raise AssertionError("decompression-bomb warning was ignored")
        finally:
            Image.MAX_IMAGE_PIXELS = old

    return [
        run_case("Images: PNG/JPEG/WebP + content/extension mismatch", common_formats_and_mismatch),
        run_case("Images: animation first frame", animation_first_frame),
        run_case("Images: EXIF orientation", exif_orientation),
        run_case("Images: corrupt input rejected", corrupt_rejected_before_media),
        run_case("Images: allowed-root boundary", outside_root_rejected),
        run_case("Images: Pillow decompression safeguard", decompression_guard),
    ]


def system_suite() -> list[bool]:
    root = reset_test_root()
    _configure_root(root)
    import launcher
    import process_tools
    import server

    def compact_clipboard_contract():
        with patch("server.get_clipboard_text", return_value={"status": "ok", "text": "abc", "length": 3, "truncated": False}):
            check(server.get_clipboard() == {"text": "abc", "length": 3, "truncated": False}, "clipboard contract")

    def foreground_contract():
        raw = {
            "status": "ok",
            "operation": "get_foreground_window",
            "window": {
                "hwnd": 7,
                "title": "Editor",
                "pid": 99,
                "process_name": "editor.exe",
                "bounds": {"left": 1},
            },
        }
        with patch("server.get_foreground_window_info", return_value=raw):
            result = server.get_foreground_window()
        check(result["title"] == "Editor" and "status" not in result, "foreground result not compact")

        with patch(
            "server.get_foreground_window_info",
            return_value={"status": "ok", "operation": "get_foreground_window", "window": None},
        ):
            check(server.get_foreground_window() == {"window": None}, "no-foreground-window contract")

    def launcher_policy_guards():
        dangerous = root / "do-not-run.cmd"
        dangerous.write_text("@echo off\necho should-not-run\n", encoding="utf-8")

        try:
            launcher.open_allowed_file(str(dangerous))
        except PermissionError:
            pass
        else:
            raise AssertionError("open_file allowed an executable/script type")

        try:
            launcher.launch_app("__dexter_unconfigured_test_app__")
        except ValueError:
            pass
        else:
            raise AssertionError("open_app accepted an unconfigured alias")

        alias = "__dexter_no_args_test__"
        previous_alias = launcher.APP_ALIASES.get(alias)
        launcher.APP_ALIASES[alias] = {
            "command": [r"C:\does-not-need-to-exist.exe"],
            "allow_arguments": False,
        }
        try:
            try:
                launcher.launch_app(alias, arguments=["unexpected"])
            except PermissionError:
                pass
            else:
                raise AssertionError("open_app accepted arguments for a no-arguments alias")
        finally:
            if previous_alias is None:
                launcher.APP_ALIASES.pop(alias, None)
            else:
                launcher.APP_ALIASES[alias] = previous_alias

        try:
            launcher.open_http_url("file:///C:/Windows/win.ini")
        except ValueError:
            pass
        else:
            raise AssertionError("open_url accepted a non-HTTP URL")

    def process_wrapper():
        result = process_tools.process_running(pid=os.getpid())
        check(result["running"], "current test process was not found")

    def process_name_exe_suffix_optional():
        class FakeProcess:
            def __init__(self, pid: int, name: str):
                self.info = {"pid": pid, "name": name}

        fake_pid = 4242
        fake_name = "ollama.exe"

        # Name-only lookups: .exe must be optional, but matching must remain
        # exact apart from that suffix.
        with patch.object(
            process_tools.psutil,
            "process_iter",
            return_value=[FakeProcess(fake_pid, fake_name)],
        ):
            without_exe = process_tools.process_running(name="ollama")
        check(without_exe["running"], "name without .exe did not match ollama.exe")
        check(
            without_exe["matches"] == [{"pid": fake_pid, "name": fake_name}],
            f"unexpected name-only match result: {without_exe}",
        )

        with patch.object(
            process_tools.psutil,
            "process_iter",
            return_value=[FakeProcess(fake_pid, fake_name)],
        ):
            with_exe = process_tools.process_running(name="ollama.exe")
        check(with_exe["running"], "name with .exe did not match ollama.exe")

        with patch.object(
            process_tools.psutil,
            "process_iter",
            return_value=[FakeProcess(fake_pid, fake_name)],
        ):
            partial = process_tools.process_running(name="oll")
        check(not partial["running"], "partial process name matched unexpectedly")

        with patch.object(
            process_tools.psutil,
            "process_iter",
            return_value=[FakeProcess(fake_pid, "chromedriver.exe")],
        ):
            similar = process_tools.process_running(name="chrome")
        check(not similar["running"], "chrome incorrectly matched chromedriver.exe")

        # PID + name lookups must use the same normalization.
        class FakePidProcess:
            def name(self):
                return fake_name

        with (
            patch.object(process_tools.psutil, "pid_exists", return_value=True),
            patch.object(process_tools.psutil, "Process", return_value=FakePidProcess()),
        ):
            check(
                process_tools.process_running(pid=fake_pid, name="ollama")["running"],
                "PID + name without .exe did not match ollama.exe",
            )
            check(
                process_tools.process_running(pid=fake_pid, name="ollama.exe")["running"],
                "PID + name with .exe did not match ollama.exe",
            )
            check(
                not process_tools.process_running(pid=fake_pid, name="oll")["running"],
                "PID + partial process name matched unexpectedly",
            )

    def open_url_result_contract():
        with patch(
            "server.open_http_url",
            return_value={"dispatched": True, "browser_accepted_request": False},
        ):
            check(server.open_url("https://example.com") == {"opened": False}, "open_url overstated success")
        with patch(
            "server.open_http_url",
            return_value={"dispatched": True, "browser_accepted_request": True},
        ):
            check(server.open_url("https://example.com") == {"opened": True}, "open_url lost successful result")

    return [
        run_case("System: compact clipboard result", compact_clipboard_contract),
        run_case("System: compact foreground-window result", foreground_contract),
        run_case("System: launcher policy guards", launcher_policy_guards),
        run_case("System: process wrapper", process_wrapper),
        run_case("System: optional Windows .exe process suffix", process_name_exe_suffix_optional),
        run_case("System: open_url success contract", open_url_result_contract),
    ]


def desktop_suite() -> list[bool]:
    root = reset_test_root()
    _configure_root(root)
    import jobs
    import server

    def screenshot_and_template_match():
        screenshot_path = root / "screen.png"
        capture = server.screenshot(str(screenshot_path))
        check(capture["width"] > 0 and capture["height"] > 0, "screenshot dimensions are invalid")

        with Image.open(screenshot_path) as image:
            cx = max(0, image.width // 2 - 40)
            cy = max(0, image.height // 2 - 40)
            crop = image.crop((cx, cy, min(cx + 80, image.width), min(cy + 80, image.height)))
            template_path = root / "template.png"
            crop.save(template_path)

        found = server.find_image(str(template_path), confidence=0.99)
        check(found["found"], "template cropped from current screen was not found")

    def screen_cursor_smoke():
        size = server.computer_action("screen_size")
        pos = server.computer_action("cursor_position")
        check(size["width"] > 0 and size["height"] > 0, "invalid screen size")
        check(isinstance(pos["x"], int) and isinstance(pos["y"], int), "invalid cursor position")

    def scheduler_mocked():
        clicks = {"count": 0}

        def fake_click_mouse(**kwargs):
            clicks["count"] += 1
            return {"status": "ok"}

        with patch.object(jobs, "click_mouse", side_effect=fake_click_mouse):
            started = jobs.start_click_interval_job(interval_seconds=0.10, position_mode="current", max_runs=2)
            job_id = started["job"]["job_id"]
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                state = jobs.get_job_status(job_id)["job"]
                if state["state"] != "running":
                    break
                time.sleep(0.03)
            final = jobs.get_job_status(job_id)["job"]
            check(final["runs"] == 2, f"expected 2 mocked clicks, got {final['runs']}")

            started2 = jobs.start_click_interval_job(interval_seconds=0.10, position_mode="current")
            job_id2 = started2["job"]["job_id"]
            time.sleep(0.15)
            cancelled = jobs.cancel_background_job(job_id2, wait_seconds=1.0)["job"]
            check(cancelled["state"] == "cancelled", f"cancel ended as {cancelled['state']}")

    def visual_stop_condition_mocked():
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
        run_case("Desktop: screenshot + template matching", screenshot_and_template_match),
        run_case("Desktop: screen/cursor smoke", screen_cursor_smoke),
        run_case("Desktop: background click scheduler (mocked)", scheduler_mocked),
        run_case("Desktop: visual stop condition (mocked)", visual_stop_condition_mocked),
    ]


def mcp_suite() -> list[bool]:
    root = reset_test_root()
    _configure_root(root)
    import server

    def registration_and_annotations():
        manager = server.mcp._tool_manager
        tools = manager.list_tools()
        names = [tool.name for tool in tools]
        check(len(names) == 37, f"expected 37 model-facing tools, got {len(names)}")
        check("read_image" in names, "read_image not registered")
        check("write_file" in names, "write_file not registered")
        check("dexter_ping" not in names, "dexter_ping should be replaced by read_image")
        for tool in tools:
            check(tool.annotations is not None, f"missing annotations: {tool.name}")

    def image_failure_does_not_return_image_content():
        bad = root / "broken.bin"
        bad.write_bytes(b"broken")
        try:
            server.read_image(str(bad))
        except Exception:
            pass
        else:
            raise AssertionError("corrupt image unexpectedly returned MCP image content")
        good = root / "good.txt"
        good.write_text("still works", encoding="utf-8")
        check(len(server.file_sha256(str(good))["sha256"]) == 64, "follow-up tool failed after image error")

    return [
        run_case("MCP: registration + standard annotations", registration_and_annotations),
        run_case("MCP: corrupt-image recovery contract", image_failure_does_not_return_image_content),
    ]


def benchmark_harness_suite() -> list[bool]:
    import benchmark_goose

    def exact_answer_grading():
        check(
            not benchmark_goose._has_exact_answer_line("loaded C:/tmp/red.webp", "RED"),
            "incidental path text was mistaken for the final answer",
        )
        check(
            benchmark_goose._has_exact_answer_line("status\nRED\n", "RED"),
            "standalone answer line was not recognized",
        )

    def isolated_config_points_to_benchmark_server():
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = benchmark_goose._prepare_isolated_goose_root(Path(temp).resolve())
            text = (root / "config" / "config.yaml").read_text(encoding="utf-8")
            check("dexter_tools:" in text, "isolated Goose config is missing dexter_tools")
            check("benchmark_server.py" in text, "isolated Goose config does not target benchmark_server.py")
            check("enabled: true" in text, "benchmark extension is not enabled")

    def fixture_root_replaces_normal_roots():
        import benchmark_server
        import config
        import filesystem

        previous = list(config.ALLOWED_ROOTS)
        try:
            with tempfile.TemporaryDirectory() as temp:
                fixture = Path(temp).resolve()
                config.ALLOWED_ROOTS[:] = [Path(r"C:\Users\example\private")]
                with patch.dict(os.environ, {benchmark_server.BENCHMARK_FIXTURE_ROOT_ENV: str(fixture)}):
                    benchmark_server._allow_fixture_root()
                check(config.ALLOWED_ROOTS == [fixture], "benchmark inherited normal filesystem roots")
                check(filesystem._resolved_roots() == [fixture], "filesystem did not see benchmark-only root")
        finally:
            config.ALLOWED_ROOTS[:] = previous

    def tool_selection_uses_start_records():
        case = benchmark_goose.Case(
            "recovery",
            "unused",
            (
                benchmark_goose.ExpectedCall("read_image", {"path": "bad.webp"}),
                benchmark_goose.ExpectedCall("file_sha256", {"path": "good.txt"}),
            ),
        )
        trace = [
            {"event": "server_started"},
            {"event": "tool_started", "tool": "read_image", "arguments": {"path": "bad.webp"}},
            # A failed tool may not have a usable result record. The selection
            # must still count because the benchmark saw the invocation begin.
            {"event": "tool_started", "tool": "file_sha256", "arguments": {"path": "good.txt"}},
            {"event": "tool_finished", "tool": "file_sha256", "arguments": {"path": "good.txt"}, "result": {"sha256": "abc"}},
        ]
        matched, observed = benchmark_goose._calls_match(trace, case)
        check(matched, "tool-start records did not satisfy expected call sequence")
        check(len(observed) == 2, "completion records leaked into selection grading")

    return [
        run_case("Benchmark: exact-answer grading", exact_answer_grading),
        run_case("Benchmark: isolated Goose config", isolated_config_points_to_benchmark_server),
        run_case("Benchmark: fixture root isolation", fixture_root_replaces_normal_roots),
        run_case("Benchmark: tool-start selection trace", tool_selection_uses_start_records),
    ]


def chains_suite() -> list[bool]:
    root = reset_test_root()
    _configure_root(root)
    import filesystem
    import server

    def find_copy_move_verify():
        project = root / "Project"
        review = root / "Review"
        project.mkdir()
        review.mkdir()
        target = project / "schema.py"
        target.write_text("SCHEMA_VERSION = 4\nLEGACY_KEYS = {'raw': 'raw_lyrics'}\n", encoding="utf-8")
        hits = server.search_file_contents(str(project), "LEGACY_KEYS", extensions=[".py"])
        check(len(hits["matches"]) == 1, "search did not resolve defining file")
        source = hits["matches"][0]["path"]
        copied = review / "schema-copy.py"
        server.copy_file(source, str(copied))
        final = review / "schema-final.py"
        server.move_file(str(copied), str(final))
        check(target.is_file() and final.is_file(), "chain files missing")
        check(filesystem.sha256_file(target) == filesystem.sha256_file(final), "chain hash mismatch")

    return [run_case("Chain: search -> copy -> move -> hash verify", find_copy_move_verify)]


SUITES = {
    "environment": environment_suite,
    "filesystem": filesystem_suite,
    "system": system_suite,
    "images": image_suite,
    "desktop": desktop_suite,
    "mcp": mcp_suite,
    "benchmark": benchmark_harness_suite,
    "chains": chains_suite,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Dexter tool validation runner.")
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    args = parser.parse_args()

    reset_trace()
    print(f"Dexter tool tests using: {test_root()}")
    print("Automatic tests do not perform real mouse clicks or keyboard typing.\n")
    selected = list(SUITES) if args.suite == "all" else [args.suite]
    results: list[bool] = []
    for name in selected:
        results += SUITES[name]()
    passed = sum(results)
    total = len(results)
    print(f"\nResult: {passed}/{total} passed.")
    trace = test_root() / TRACE_FILE_NAME
    if trace.exists():
        print(f"Mutation trace: {trace}")
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
