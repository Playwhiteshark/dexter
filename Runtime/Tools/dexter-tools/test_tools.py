"""Deterministic validation for Dexter tools v0.2.

No real mouse clicks or keyboard typing are performed. Desktop side effects are
mocked at the test layer; production modules contain no test-mode branches.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
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

    def verified_copy_move():
        source = root / "source.txt"
        source.write_text("verified", encoding="utf-8")
        copied = root / "copied.txt"
        moved = root / "moved.txt"
        result = server.copy_file(str(source), str(copied))
        check(result == {"destination": str(copied.resolve())}, f"copy result too noisy: {result}")
        trace_mutation("copy_file", source=str(source), destination=str(copied))
        result = server.move_file(str(copied), str(moved))
        check(result == {"destination": str(moved.resolve())}, f"move result too noisy: {result}")
        trace_mutation("move_file", source=str(copied), destination=str(moved))
        check(source.exists() and moved.exists() and not copied.exists(), "copy/move postconditions failed")

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
        run_case("Filesystem: verified copy/move + compact results", verified_copy_move),
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

    return [
        run_case("System: compact clipboard result", compact_clipboard_contract),
        run_case("System: compact foreground-window result", foreground_contract),
    ]


def desktop_suite() -> list[bool]:
    root = reset_test_root()
    _configure_root(root)
    import jobs

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

    return [run_case("Desktop: background click scheduler (mocked)", scheduler_mocked)]


def mcp_suite() -> list[bool]:
    root = reset_test_root()
    _configure_root(root)
    import server

    def registration_and_annotations():
        manager = server.mcp._tool_manager
        tools = manager.list_tools()
        names = [tool.name for tool in tools]
        check(len(names) == 36, f"expected 36 model-facing tools, got {len(names)}")
        check("read_image" in names, "read_image not registered")
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
