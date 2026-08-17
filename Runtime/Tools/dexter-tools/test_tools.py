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
import sys
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
        for module in ("numpy", "cv2", "pyautogui", "mcp", "PIL", "ddgs"):
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
    import app_discovery
    import launcher
    import process_tools
    import server
    import windows_tools

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

        candidate = app_discovery.AppCandidate(
            name="OMEN Gaming Hub",
            source="start_apps",
            launch_target="AD2F1837.OMENCommandCenter_abc!App",
        )
        with (
            patch("launcher.resolve_installed_app", return_value=candidate) as resolve_mock,
            patch(
                "launcher.launch_installed_app",
                return_value={
                    "status": "ok",
                    "operation": "open_app",
                    "app": "OMEN Gaming Hub",
                    "source": "start_apps",
                    "pid": 123,
                    "dispatched": True,
                },
            ) as launch_mock,
        ):
            result = launcher.launch_app("omen gaming hub")
        check(result["app"] == "OMEN Gaming Hub", "installed app name was not preserved")
        resolve_mock.assert_called_once_with("omen gaming hub")
        launch_mock.assert_called_once_with(candidate)

        try:
            launcher.launch_app("ordinary discovered app", arguments=["--unexpected"])
        except PermissionError:
            pass
        else:
            raise AssertionError("open_app accepted arguments for a dynamically discovered app")

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

    def installed_app_name_matching():
        candidates = (
            app_discovery.AppCandidate(
                name="OMEN Gaming Hub",
                source="start_apps",
                launch_target="omen-hub",
            ),
            app_discovery.AppCandidate(
                name="OMEN Light Studio",
                source="start_apps",
                launch_target="omen-light",
            ),
            app_discovery.AppCandidate(
                name="Google Chrome",
                source="start_apps",
                launch_target="chrome",
            ),
        )

        with patch("app_discovery.discover_installed_apps", return_value=candidates):
            exact = app_discovery.resolve_installed_app("OMEN Gaming Hub")
            check(exact.name == "OMEN Gaming Hub", "exact installed-app match failed")

            token = app_discovery.resolve_installed_app("chrome")
            check(token.name == "Google Chrome", "distinctive token match failed")

            typo = app_discovery.resolve_installed_app("omen gaming hup")
            check(typo.name == "OMEN Gaming Hub", "strong typo match failed")

            try:
                app_discovery.resolve_installed_app("omen")
            except ValueError as exc:
                message = str(exc)
                check(
                    "OMEN Gaming Hub" in message and "OMEN Light Studio" in message,
                    "ambiguous app error did not surface candidates",
                )
            else:
                raise AssertionError("ambiguous installed-app name was guessed")

    def configured_alias_precedes_discovery():
        alias = "__dexter_alias_precedence__"
        previous_alias = launcher.APP_ALIASES.get(alias)
        launcher.APP_ALIASES[alias] = {
            "command": [sys.executable],
            "allow_arguments": False,
        }
        try:
            with (
                patch("launcher.subprocess.Popen") as popen_mock,
                patch("launcher.resolve_installed_app") as discovery_mock,
            ):
                process = popen_mock.return_value
                process.pid = 444
                process.poll.return_value = None
                with patch("launcher.psutil.pid_exists", return_value=True):
                    result = launcher.launch_app(alias)
            check(result["source"] == "configured_alias", "configured alias source missing")
            check(not discovery_mock.called, "installed-app discovery overrode explicit alias")
        finally:
            if previous_alias is None:
                launcher.APP_ALIASES.pop(alias, None)
            else:
                launcher.APP_ALIASES[alias] = previous_alias

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

    def visible_frame_geometry_translation():
        outer = {
            "left": -8, "top": -8, "right": 1008, "bottom": 1008,
            "width": 1016, "height": 1016,
        }
        visible = {
            "left": 0, "top": 0, "right": 1000, "bottom": 1000,
            "width": 1000, "height": 1000,
        }
        target = {
            "left": 1000, "top": 0, "right": 2000, "bottom": 1000,
            "width": 1000, "height": 1000,
        }
        translated = windows_tools._outer_bounds_for_visible_target(target, outer, visible)
        check(
            translated == {
                "left": 992, "top": -8, "right": 2008, "bottom": 1008,
                "width": 1016, "height": 1016,
            },
            f"visible-frame translation was wrong: {translated}",
        )

    def window_state_control():
        class FakeUser32:
            def __init__(self):
                self.state = "normal"
                self.commands: list[int] = []

            def IsWindow(self, hwnd):
                return hwnd == 7

            def ShowWindow(self, hwnd, command):
                self.commands.append(command)
                if command == windows_tools.SW_MINIMIZE:
                    self.state = "minimized"
                elif command == windows_tools.SW_SHOWMAXIMIZED:
                    self.state = "maximized"
                elif command == windows_tools.SW_RESTORE:
                    self.state = "normal"
                return 1

            def IsIconic(self, hwnd):
                return self.state == "minimized"

            def IsZoomed(self, hwnd):
                return self.state == "maximized"

        fake = FakeUser32()
        with (
            patch.object(windows_tools, "user32", fake),
            patch.object(windows_tools, "_require_windows", return_value=None),
            patch.object(windows_tools.time, "sleep", return_value=None),
        ):
            minimized = windows_tools.set_window_state_verified(7, "minimized")
            check(minimized["state"] == "minimized", "window did not minimize")

            restored = windows_tools.set_window_state_verified(7, "normal")
            check(restored["state"] == "normal", "window did not restore")

            maximized = windows_tools.set_window_state_verified(7, "maximized")
            check(maximized["state"] == "maximized", "window did not maximize")

            try:
                windows_tools.set_window_state_verified(7, "closed")
            except ValueError:
                pass
            else:
                raise AssertionError("invalid window state was accepted")

        check(
            fake.commands == [
                windows_tools.SW_MINIMIZE,
                windows_tools.SW_RESTORE,
                windows_tools.SW_SHOWMAXIMIZED,
            ],
            f"wrong ShowWindow commands: {fake.commands}",
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
        run_case("System: installed app name matching", installed_app_name_matching),
        run_case("System: configured app alias precedes discovery", configured_alias_precedes_discovery),
        run_case("System: process wrapper", process_wrapper),
        run_case("System: optional Windows .exe process suffix", process_name_exe_suffix_optional),
        run_case("System: DWM-visible frame geometry translation", visible_frame_geometry_translation),
        run_case("System: window minimize/maximize/restore", window_state_control),
        run_case("System: open_url success contract", open_url_result_contract),
    ]


def memory_suite() -> list[bool]:
    root = reset_test_root()
    _configure_root(root)
    import browser_state
    import resources
    import snap_layout
    import state_store
    import terminal_state
    import vscode_state
    import workspace_providers
    import workspace_tools

    state_store.COMPUTER_MEMORY_PATH = root / "computer_memory.json"

    def resource_round_trip_and_aliases():
        saved = resources.remember_resource(
            "Dexter GitHub",
            "url",
            "https://github.com/example/dexter",
            aliases=["repo", "Dexter repo"],
        )
        check(saved["created"], "resource was not created")
        found = resources.search_resources("repo")
        check(len(found["resources"]) == 1, "resource alias search failed")
        check(found["resources"][0]["name"] == "Dexter GitHub", "wrong resource returned")

        with patch(
            "resources.open_http_url",
            return_value={"browser_accepted_request": True},
        ) as mocked:
            opened = resources.open_saved_resource("Dexter repo")
        check(opened["opened"], "saved URL did not open")
        mocked.assert_called_once_with("https://github.com/example/dexter")

        try:
            resources.remember_resource(
                "Dexter GitHub",
                "url",
                "https://github.com/example/other",
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("resource overwrite was allowed by default")

    def typed_project_validation():
        vscode = root / "CodeProject"
        vscode.mkdir()
        vs = root / "Solution.sln"
        vs.write_text("Microsoft Visual Studio Solution File", encoding="utf-8")
        unity = root / "UnityProject"
        (unity / "ProjectSettings").mkdir(parents=True)
        (unity / "ProjectSettings" / "ProjectVersion.txt").write_text(
            "m_EditorVersion: 6000.0.0f1\n",
            encoding="utf-8",
        )

        resources.remember_resource("Code project", "vscode", str(vscode))
        resources.remember_resource("VS solution", "visual_studio", str(vs))
        resources.remember_resource("Unity project", "unity", str(unity))
        found = resources.search_resources()
        kinds = {item["kind"] for item in found["resources"]}
        check({"vscode", "visual_studio", "unity"} <= kinds, "typed project resources missing")

    def typed_project_open_dispatch():
        # Reuse records created by the previous case. Opening a saved project
        # must use fixed, validated app-specific arguments rather than exposing
        # arbitrary command-line execution to the model.
        with patch("resources.launch_app_with_trusted_arguments") as launch:
            resources.open_saved_resource("Code project")
            launch.assert_called_once_with(
                "vscode",
                ["--new-window", str((root / "CodeProject").resolve())],
            )

        with patch("resources.launch_app_with_trusted_arguments") as launch:
            resources.open_saved_resource("VS solution")
            launch.assert_called_once_with(
                "visual studio",
                [str((root / "Solution.sln").resolve())],
            )

        with patch("resources.launch_app_with_trusted_arguments") as launch:
            resources.open_saved_resource("Unity project")
            launch.assert_called_once_with(
                "unity",
                ["-projectPath", str((root / "UnityProject").resolve())],
            )

    def natural_saved_name_resolution_is_internal():
        friendbox = root / "FriendBox"
        friendbox.mkdir()
        resources.remember_resource("FriendBox", "vscode", str(friendbox))

        with patch("resources.launch_app_with_trusted_arguments") as launch:
            opened = resources.open_saved_resource(
                "open my FriendBox VS Code project"
            )
        check(opened["name"] == "FriendBox", f"natural project name resolved incorrectly: {opened}")
        launch.assert_called_once_with(
            "vscode",
            ["--new-window", str(friendbox.resolve())],
        )

        def add_workspace(state):
            state["workspaces"]["ws_ai"] = {
                "id": "ws_ai",
                "name": "AI",
                "windows": [],
                "created_utc": "2026-01-01T00:00:00+00:00",
                "updated_utc": "2026-01-01T00:00:00+00:00",
            }

        state_store.mutate_state(add_workspace)
        resolved = workspace_tools._resolve_workspace("open my AI workspace")
        check(resolved["name"] == "AI", f"generic workspace words blocked resolution: {resolved}")
        found = workspace_tools.search_workspaces("the AI workspace")
        check(found["workspaces"][0]["name"] == "AI", f"workspace search did not use shared resolver: {found}")

        def add_more_specific_workspace(state):
            state["workspaces"]["ws_ai_full"] = {
                "id": "ws_ai_full",
                "name": "AI Workspace",
                "windows": [],
                "created_utc": "2026-01-01T00:00:00+00:00",
                "updated_utc": "2026-01-01T00:00:00+00:00",
            }

        state_store.mutate_state(add_more_specific_workspace)
        specific = workspace_tools._resolve_workspace("open my AI workspace")
        check(
            specific["name"] == "AI Workspace",
            f"more specific stored workspace name did not win: {specific}",
        )
        workspace_tools.forget_workspace("AI Workspace")
        workspace_tools.forget_workspace("AI workspace")
        resources.forget_resource("FriendBox project")

    def remember_current_vscode_project_from_bridge():
        with tempfile.TemporaryDirectory(prefix="dexter-external-vscode-") as external:
            project = Path(external) / "Dexter"
            project.mkdir()
            check(
                not project.resolve().is_relative_to(root.resolve()),
                "bridge exception test project accidentally used an allowed root",
            )

            try:
                resources.remember_resource("Manual outside project", "vscode", str(project))
            except ValueError as exc:
                check("outside" in str(exc).casefold(), f"wrong manual boundary error: {exc}")
            else:
                raise AssertionError("manual project path bypassed ALLOWED_ROOTS")

            session = {
                "session_id": "dexter-live",
                "workspace_name": "Dexter",
                "workspace_file": None,
                "workspace_folders": [str(project)],
                "active_file": str(project / "README.md"),
                "focused": True,
            }
            with patch("resources.current_vscode_session", return_value=session):
                saved = resources.remember_current_vscode_project(
                    aliases=["local assistant"]
                )

            record = saved["resource"]
            check(record["name"] == "Dexter", f"project name was not inferred: {record}")
            check(record["target"] == str(project.resolve()), f"project target was guessed: {record}")
            check("local assistant" in record["aliases"], f"explicit alias was lost: {record}")

            with patch("resources.launch_app_with_trusted_arguments") as launch:
                opened = resources.open_saved_resource("the Dexter project")
            check(opened["name"] == "Dexter", f"remembered current project did not reopen: {opened}")
            launch.assert_called_once_with(
                "vscode",
                ["--new-window", str(project.resolve())],
            )
            resources.forget_resource("Dexter VS Code project")

    def current_vscode_project_uses_last_real_focus():
        older = {
            "session_id": "older",
            "workspace_name": "Older",
            "focused": False,
            "last_focused_unix": 10.0,
        }
        intended = {
            "session_id": "intended",
            "workspace_name": "Intended",
            "focused": False,
            "last_focused_unix": 20.0,
        }
        with patch.object(vscode_state, "_sessions", return_value=[older, intended]):
            selected = vscode_state.current_vscode_session()
        check(selected["session_id"] == "intended", f"last focused project was not selected: {selected}")

        ambiguous = [{**older, "last_focused_unix": 0}, {**intended, "last_focused_unix": 0}]
        with patch.object(vscode_state, "_sessions", return_value=ambiguous):
            try:
                vscode_state.current_vscode_session()
            except LookupError as exc:
                check("Focus the intended" in str(exc), f"ambiguous focus error was not actionable: {exc}")
            else:
                raise AssertionError("ambiguous live VS Code projects were guessed")

    def workspace_app_state_capture_is_persisted():
        placement = {
            "show_state": "normal",
            "show_cmd": 1,
            "visible_bounds": {"left": 0, "top": 0, "right": 1200, "bottom": 800, "width": 1200, "height": 800},
            "current_bounds": {"left": 0, "top": 0, "right": 1200, "bottom": 800, "width": 1200, "height": 800},
            "normal_bounds": {"left": 0, "top": 0, "right": 1200, "bottom": 800, "width": 1200, "height": 800},
        }
        captured = {
            "status": "ok",
            "windows": [{
                "hwnd": 810,
                "title": "Dexter - Brave",
                "pid": 81,
                "process_name": "brave.exe",
                "exe": r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
                "placement": placement,
                "z_index": 0,
                "resizable": True,
            }],
        }
        app_state = {
            "kind": "chromium_browser",
            "browser_kind": "brave",
            "tabs": [
                {"url": "https://example.com", "title": "Example", "pinned": False, "active": True, "group_key": None}
            ],
            "groups": [],
            "active_index": 0,
            "captured_tab_count": 1,
        }
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=captured),
            patch("workspace_tools.capture_window_app_state", return_value=(app_state, None)),
            patch("workspace_tools.configured_app_alias_for_executable", return_value=None),
            patch("workspace_tools.resource_launch_candidates_for_window", return_value=[]),
        ):
            saved = workspace_tools.save_workspace("App State Capture")

        check(saved["app_state_window_count"] == 1, f"app state count was not grounded: {saved}")
        check(not saved["state_warnings"], f"unexpected app-state warning: {saved}")
        record = workspace_tools._resolve_workspace("App State Capture")
        stored = record["windows"][0].get("app_state")
        check(stored == app_state, f"app state was not persisted in workspace memory: {stored}")
        workspace_tools.forget_workspace("App State Capture")

    def workspace_app_state_restore_runs_before_geometry():
        placement = {
            "show_state": "normal",
            "show_cmd": 1,
            "visible_bounds": {"left": 10, "top": 10, "right": 1010, "bottom": 710, "width": 1000, "height": 700},
            "current_bounds": {"left": 10, "top": 10, "right": 1010, "bottom": 710, "width": 1000, "height": 700},
            "normal_bounds": {"left": 10, "top": 10, "right": 1010, "bottom": 710, "width": 1000, "height": 700},
        }
        captured = {"status": "ok", "windows": [{
            "hwnd": 820, "title": "Project - Visual Studio Code", "pid": 82,
            "process_name": "Code.exe", "exe": r"C:\Apps\Code.exe",
            "placement": placement, "z_index": 0, "resizable": True,
        }]}
        saved_state = {"kind": "vscode", "workspace_folders": [r"C:\Work\Jarvis"], "tabs": []}
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=captured),
            patch("workspace_tools.capture_window_app_state", return_value=(saved_state, None)),
            patch("workspace_tools.configured_app_alias_for_executable", return_value=None),
            patch("workspace_tools.resource_launch_candidates_for_window", return_value=[]),
        ):
            workspace_tools.save_workspace("App State Restore")

        current = {"status": "ok", "windows": [{**captured["windows"][0], "hwnd": 821}]}
        order = []
        def restore_state(saved, window):
            order.append("state")
            check(saved.get("app_state") == saved_state, "saved app state was not supplied to restorer")
            check(window["hwnd"] == 821, "current HWND was not supplied to app-state restorer")
            return ({"status": "ok", "kind": "vscode", "tab_count": 3}, None)
        def restore_geometry(items):
            order.append("geometry")
            return {"restored": [{"hwnd": 821, "state": "normal", "verified": True, "warning": None}], "failed": []}

        try:
            with (
                patch("workspace_tools.capture_workspace_windows", return_value=current),
                patch("workspace_tools.vscode_window_matches_saved_state", return_value=None),
                patch("workspace_tools.restore_window_app_state", side_effect=restore_state),
                patch("workspace_tools.restore_window_placements_batch", side_effect=restore_geometry),
            ):
                result = workspace_tools.restore_workspace("App State Restore", launch_missing=False)

            check(order == ["state", "geometry"], f"app state did not restore before geometry: {order}")
            check(result["complete"], f"successful app-state restore was reported incomplete: {result}")
            check(result["app_state_restored_count"] == 1, f"app state success missing: {result}")
        finally:
            try:
                workspace_tools.forget_workspace("App State Restore")
            except LookupError:
                pass

    def workspace_app_state_warning_marks_restore_degraded():
        placement = {
            "show_state": "normal", "show_cmd": 1,
            "visible_bounds": {"left": 0, "top": 0, "right": 800, "bottom": 600, "width": 800, "height": 600},
            "current_bounds": {"left": 0, "top": 0, "right": 800, "bottom": 600, "width": 800, "height": 600},
            "normal_bounds": {"left": 0, "top": 0, "right": 800, "bottom": 600, "width": 800, "height": 600},
        }
        captured = {"status": "ok", "windows": [{
            "hwnd": 830, "title": "Browser", "pid": 83,
            "process_name": "brave.exe", "exe": r"C:\Apps\brave.exe",
            "placement": placement, "z_index": 0, "resizable": True,
        }]}
        app_state = {"kind": "chromium_browser", "browser_kind": "brave", "tabs": [{"url": "https://example.com"}]}
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=captured),
            patch("workspace_tools.capture_window_app_state", return_value=(app_state, None)),
            patch("workspace_tools.configured_app_alias_for_executable", return_value=None),
            patch("workspace_tools.resource_launch_candidates_for_window", return_value=[]),
        ):
            workspace_tools.save_workspace("App State Warning")

        current = {"status": "ok", "windows": [{**captured["windows"][0], "hwnd": 831}]}
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=current),
            patch("workspace_tools.restore_window_app_state", return_value=(None, "browser bridge unavailable")),
            patch("workspace_tools.restore_window_placements_batch", return_value={
                "restored": [{"hwnd": 831, "state": "normal", "verified": True, "warning": None}], "failed": []
            }),
        ):
            result = workspace_tools.restore_workspace("App State Warning", launch_missing=False)

        check(result["restored_count"] == 1, f"geometry repair was lost: {result}")
        check(result["degraded"] is True and not result["complete"], f"app-state failure overstated completion: {result}")
        check("browser bridge unavailable" in result["warnings"][0]["warning"], f"app-state warning was lost: {result}")
        workspace_tools.forget_workspace("App State Warning")

    def browser_window_match_prefers_active_tab_title():
        os_window = {
            "title": "Local AI - Brave",
            "visible_bounds": {"left": 0, "top": 0, "right": 1200, "bottom": 900},
        }
        windows = [
            {"id": 1, "left": 0, "top": 0, "width": 1200, "height": 900, "tabs": [{"title": "Other", "active": True}]},
            {"id": 2, "left": 100, "top": 100, "width": 900, "height": 700, "tabs": [{"title": "Local AI", "active": True}]},
        ]
        matched = browser_state.match_browser_window(os_window, windows, "brave")
        check(matched and matched["id"] == 2, f"active browser tab title did not win matching: {matched}")

    def browser_restore_sanitizes_blank_tabs():
        saved = {
            "process_name": "brave.exe",
            "app_state": {
                "kind": "chromium_browser",
                "browser_kind": "brave",
                "tabs": [
                    {"url": "about:blank", "active": False},
                    {"url": "https://chatgpt.com/c/example", "active": True},
                ],
                "groups": [],
                "active_index": 1,
            },
        }
        current = {"title": "ChatGPT - Brave"}
        bridge_window = {"id": 9, "_bridge_session": "browser-session", "tabs": []}
        observed = {}

        def fake_restore(kind, session_id, window_id, payload):
            observed["kind"] = kind
            observed["session_id"] = session_id
            observed["window_id"] = window_id
            observed["payload"] = payload
            return {
                "tab_count": len(payload["tabs"]),
                "reused_tab_count": 1,
                "created_tab_count": 0,
                "removed_extra_tab_count": 0,
                "active_tab_ready": True,
                "failed_count": 0,
            }

        with (
            patch.object(browser_state, "capture_browser_windows", return_value=[bridge_window]),
            patch.object(browser_state, "match_browser_window", return_value=bridge_window),
            patch.object(browser_state, "restore_browser_window", side_effect=fake_restore),
        ):
            result = browser_state.restore_browser_state(saved, current)

        clean = observed["payload"]["tabs"]
        check(len(clean) == 1 and clean[0]["url"].startswith("https://chatgpt.com/"), f"blank/internal tab was not sanitized: {clean}")
        check(observed["payload"]["active_index"] == 0, f"active index was not recomputed after sanitizing: {observed}")
        check(result["reused_tab_count"] == 1 and result["active_tab_ready"], f"browser reconcile result was lost: {result}")
        check(not browser_state._restorable_url("about:blank"), "about:blank must never be persisted as a workspace page")

    def vscode_restore_uses_native_editor_session():
        saved = {
            "app_state": {
                "kind": "vscode",
                "workspace_file": None,
                "workspace_folders": [r"C:\\Work\\Jarvis"],
                "active_file": r"C:\\Work\\Jarvis\\main.py",
                # Old v6 snapshots can contain these. v6.1 must ignore them;
                # VS Code owns dirty/untitled/editor restoration itself.
                "tabs": [
                    {"kind": "text", "uri": "untitled:runtime.md", "active": True},
                    {"kind": "text", "uri": "file:///C:/Work/Jarvis/main.py"},
                ],
            }
        }
        current_window = {"title": "main.py - Jarvis - Visual Studio Code"}
        session = {
            "session_id": "jarvis",
            "workspace_file": None,
            "workspace_folders": [r"C:\\Work\\Jarvis"],
            "workspace_name": "Jarvis",
            "active_file": r"C:\\Work\\Jarvis\\main.py",
        }

        with (
            patch.object(vscode_state, "match_vscode_session", return_value=session),
            patch.object(vscode_state, "_wait_for_saved_workspace_session") as wait,
            patch.object(vscode_state.time, "sleep", return_value=None),
        ):
            result = vscode_state.restore_vscode_state(saved, current_window)

        check(result["editor_restore"] == "vscode_native_session", f"VS Code did not defer editor restore to VS Code: {result}")
        check(result["workspace_switched"] is False, f"VS Code was unnecessarily switched: {result}")
        wait.assert_not_called()

    def vscode_focus_is_not_window_identity():
        fake_window = {"title": "Project - Visual Studio Code"}
        unrelated = {
            "session_id": "real-live-session",
            "workspace_name": "Jarvis",
            "workspace_file": None,
            "workspace_folders": [r"C:\Users\ahmad\Jarvis"],
            "active_file": r"C:\Users\ahmad\Jarvis\runtime.md",
            "focused": True,
            "updated_unix": time.time(),
        }
        with patch("vscode_state._sessions", return_value=[unrelated]):
            matched = vscode_state.match_vscode_session(fake_window)
        check(
            matched is None,
            f"focused unrelated VS Code session incorrectly claimed the window: {matched}",
        )

    def vscode_wrong_project_is_not_hijacked():
        saved = {
            "title": "Jarvis - Visual Studio Code",
            "process_name": "Code.exe",
            "exe": r"C:\\Apps\\Code.exe",
            "workspace_provider": "vscode",
            "app_state": {
                "kind": "vscode",
                "workspace_file": None,
                "workspace_folders": [r"C:\\Work\\Jarvis"],
            },
        }
        other = {
            "hwnd": 901,
            "title": "Other - Visual Studio Code",
            "process_name": "Code.exe",
            "exe": r"C:\\Apps\\Code.exe",
            "resizable": True,
        }
        with patch("workspace_tools.vscode_window_matches_saved_state", return_value=False):
            candidates = workspace_tools._candidate_windows(saved, [other], set())
        check(candidates == [], f"Dexter would hijack a different dirty VS Code project: {candidates}")


    def terminal_restore_never_replays_saved_commands():
        state = {
            "kind": "windows_terminal",
            "cwd": str(root),
            "shell_name": "powershell.exe",
            "commandline": [r"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe", "-Command", "Remove-Item important.txt"],
        }
        command = terminal_state.terminal_launch_command(state)
        check(command[:4] == ["wt.exe", "-w", "-1", "new-tab"], f"wrong Terminal launch prefix: {command}")
        check("--startingDirectory" in command and str(root) in command, f"Terminal cwd missing: {command}")
        check("-Command" not in command and "Remove-Item important.txt" not in command, f"Terminal replayed saved command arguments: {command}")
        check(command[-1].casefold().endswith("powershell.exe"), f"Terminal did not preserve safe interactive shell identity: {command}")

    def app_state_provider_launch_uses_saved_context():
        saved_vscode = {
            "title": "Jarvis - Visual Studio Code",
            "process_name": "Code.exe",
            "exe": r"C:\Apps\Code.exe",
            "workspace_provider": "vscode",
            "app_state": {"kind": "vscode", "workspace_folders": [r"C:\Work\Jarvis"]},
        }
        with (
            patch.object(workspace_providers.os, "name", "nt"),
            patch.object(workspace_providers, "_validated_saved_executable", return_value=r"C:\Apps\Code.exe"),
            patch.object(workspace_providers, "vscode_open_target", return_value=r"C:\Work\Jarvis"),
            patch.object(workspace_providers, "_spawn", return_value={"provider": "vscode"}) as spawn,
        ):
            workspace_providers.launch_workspace_provider(saved_vscode)
        spawn.assert_called_once_with([r"C:\Apps\Code.exe", "-n", r"C:\Work\Jarvis"], "vscode")

        saved_terminal = {
            "title": "Terminal", "process_name": "WindowsTerminal.exe",
            "workspace_provider": "windows_terminal",
            "app_state": {"kind": "windows_terminal", "cwd": r"C:\Work\Jarvis"},
        }
        with (
            patch.object(workspace_providers.os, "name", "nt"),
            patch.object(workspace_providers, "terminal_launch_command", return_value=["wt.exe", "-w", "-1", "new-tab", "--startingDirectory", r"C:\Work\Jarvis"]),
            patch.object(workspace_providers, "_spawn", return_value={"provider": "windows_terminal"}) as spawn,
        ):
            workspace_providers.launch_workspace_provider(saved_terminal)
        spawn.assert_called_once()
        check(r"C:\Work\Jarvis" in spawn.call_args.args[0], "Terminal saved cwd was not consumed at launch")


    def workspace_capture_restore_and_forget():
        fake_placement = {
            "show_state": "minimized",
            "show_cmd": 2,
            "normal_bounds": {
                "left": 10,
                "top": 20,
                "right": 810,
                "bottom": 620,
                "width": 800,
                "height": 600,
            },
        }
        captured = {
            "status": "ok",
            "windows": [
                {
                    "hwnd": 101,
                    "title": "Notes",
                    "pid": 1,
                    "process_name": "notepad.exe",
                    "exe": r"C:\Windows\System32\notepad.exe",
                    "placement": fake_placement,
                    "z_index": 0,
                }
            ],
        }

        with (
            patch("workspace_tools.capture_workspace_windows", return_value=captured),
            patch("workspace_tools.configured_app_alias_for_executable", return_value="notepad"),
        ):
            saved = workspace_tools.save_workspace("Focus")
        check(saved["window_count"] == 1, "workspace did not capture one window")
        check(
            saved.get("captured_windows") == [
                {"title": "Notes", "process_name": "notepad.exe", "provider": None}
            ],
            f"workspace save did not return grounded captured-window summary: {saved}",
        )

        current = {
            "status": "ok",
            "windows": [
                {
                    "hwnd": 202,
                    "title": "Notes",
                    "pid": 2,
                    "process_name": "notepad.exe",
                    "exe": r"C:\Windows\System32\notepad.exe",
                    "placement": fake_placement,
                    "z_index": 0,
                }
            ],
        }
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=current),
            patch(
                "workspace_tools.restore_window_placements_batch",
                return_value={
                    "restored": [
                        {
                            "hwnd": 202,
                            "state": "minimized",
                            "verified": True,
                            "warning": None,
                        }
                    ],
                    "failed": [],
                },
            ) as restore_batch,
        ):
            result = workspace_tools.restore_workspace("Focus", launch_missing=False)
        check(result["complete"], f"workspace restore incomplete: {result}")
        check(result["restored_count"] == 1, "workspace did not restore one window")
        restore_batch.assert_called_once_with(
            [{"hwnd": 202, "placement": fake_placement}]
        )

        deleted = workspace_tools.forget_workspace("Focus")
        check(deleted["deleted"], "workspace was not forgotten")
        check(
            not workspace_tools.search_workspaces("Focus")["workspaces"],
            "forgotten workspace remained",
        )

    def native_snap_plan_for_three_window_layout():
        def item(hwnd, left, top, right, bottom):
            return {
                "hwnd": hwnd,
                "placement": {
                    "show_state": "arranged",
                    "visible_bounds": {
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "width": right - left,
                        "height": bottom - top,
                    },
                },
            }

        plan = snap_layout.infer_snap_plan(
            [
                item(1, 0, 0, 1219, 1704),
                item(2, 1219, 0, 2880, 1272),
                item(3, 1219, 1272, 2880, 1704),
            ]
        )
        zones = {window.hwnd: window.zone for window in plan.windows}
        check(zones == {1: "left", 2: "top-right", 3: "bottom-right"}, f"wrong Snap zones: {zones}")
        vertical = [d for d in plan.dividers if d.axis == "vertical"]
        horizontal = [d for d in plan.dividers if d.axis == "horizontal"]
        check(len(vertical) == 1 and vertical[0].target == 1219, f"wrong vertical divider: {vertical}")
        check(len(horizontal) == 1 and horizontal[0].target == 1272, f"wrong horizontal divider: {horizontal}")
        check(horizontal[0].segment_start == 1219 and horizontal[0].segment_end == 2880, "wrong right-side divider span")

    def workspace_restore_prefers_native_snap_for_arranged_group():
        def placement(left, top, right, bottom):
            return {
                "show_state": "arranged",
                "show_cmd": 1,
                "visible_bounds": {
                    "left": left, "top": top, "right": right, "bottom": bottom,
                    "width": right - left, "height": bottom - top,
                },
                "current_bounds": {
                    "left": left, "top": top, "right": right, "bottom": bottom,
                    "width": right - left, "height": bottom - top,
                },
                "normal_bounds": {
                    "left": 100, "top": 100, "right": 900, "bottom": 700,
                    "width": 800, "height": 600,
                },
            }

        saved_windows = [
            {"hwnd": 1, "title": "Left", "pid": 1, "process_name": "left.exe", "exe": r"C:\\Apps\\left.exe", "placement": placement(0, 0, 1219, 1704), "z_index": 0},
            {"hwnd": 2, "title": "Top Right", "pid": 2, "process_name": "top.exe", "exe": r"C:\\Apps\\top.exe", "placement": placement(1219, 0, 2880, 1272), "z_index": 1},
            {"hwnd": 3, "title": "Bottom Right", "pid": 3, "process_name": "bottom.exe", "exe": r"C:\\Apps\\bottom.exe", "placement": placement(1219, 1272, 2880, 1704), "z_index": 2},
        ]
        captured_save = {"status": "ok", "windows": saved_windows}
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=captured_save),
            patch("workspace_tools.configured_app_alias_for_executable", return_value=None),
            patch("workspace_tools.resource_launch_candidates_for_window", return_value=[]),
        ):
            workspace_tools.save_workspace("Native Snap Test")

        current = {"status": "ok", "windows": [{**item, "hwnd": item["hwnd"] + 100} for item in saved_windows]}
        native_restored = [
            {"hwnd": 101, "state": "arranged", "verified": True},
            {"hwnd": 102, "state": "arranged", "verified": True},
            {"hwnd": 103, "state": "arranged", "verified": True},
        ]
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=current),
            patch(
                "workspace_tools.restore_native_snap_layout",
                return_value={
                    "attempted": True,
                    "supported": True,
                    "complete": True,
                    "restored": native_restored,
                    "failed": [],
                },
            ) as native,
            patch("workspace_tools.restore_window_placements_batch") as generic,
        ):
            result = workspace_tools.restore_workspace("Native Snap Test", launch_missing=False)

        check(result["complete"], f"native Snap workspace restore incomplete: {result}")
        check(result["restored_count"] == 3, f"wrong native Snap restore count: {result}")
        native.assert_called_once()
        passed = native.call_args.args[0]
        check([item["hwnd"] for item in passed] == [101, 102, 103], f"wrong arranged HWNDs: {passed}")
        generic.assert_not_called()

    def workspace_native_snap_failure_falls_back_to_exact_geometry():
        def placement(left, top, right, bottom):
            return {
                "show_state": "arranged",
                "show_cmd": 1,
                "visible_bounds": {
                    "left": left, "top": top, "right": right, "bottom": bottom,
                    "width": right - left, "height": bottom - top,
                },
                "current_bounds": {
                    "left": left, "top": top, "right": right, "bottom": bottom,
                    "width": right - left, "height": bottom - top,
                },
                "normal_bounds": {
                    "left": 100, "top": 100, "right": 900, "bottom": 700,
                    "width": 800, "height": 600,
                },
            }

        saved_windows = [
            {"hwnd": 1, "title": "Left Fallback", "pid": 1, "process_name": "left-fallback.exe", "exe": r"C:\\Apps\\left-fallback.exe", "placement": placement(0, 0, 1440, 1704), "z_index": 0},
            {"hwnd": 2, "title": "Right Fallback", "pid": 2, "process_name": "right-fallback.exe", "exe": r"C:\\Apps\\right-fallback.exe", "placement": placement(1440, 0, 2880, 1704), "z_index": 1},
        ]
        with (
            patch("workspace_tools.capture_workspace_windows", return_value={"status": "ok", "windows": saved_windows}),
            patch("workspace_tools.configured_app_alias_for_executable", return_value=None),
            patch("workspace_tools.resource_launch_candidates_for_window", return_value=[]),
        ):
            workspace_tools.save_workspace("Snap Fallback")

        current = {"status": "ok", "windows": [{**item, "hwnd": item["hwnd"] + 200} for item in saved_windows]}
        fallback_result = {
            "restored": [
                {"hwnd": 201, "state": "arranged", "verified": True, "warning": None},
                {"hwnd": 202, "state": "arranged", "verified": True, "warning": None},
            ],
            "failed": [],
        }
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=current),
            patch(
                "workspace_tools.restore_native_snap_layout",
                return_value={
                    "attempted": True,
                    "supported": True,
                    "complete": False,
                    "restored": [],
                    "failed": [{"reason": "synthetic input blocked"}],
                },
            ),
            patch("workspace_tools.restore_window_placements_batch", return_value=fallback_result) as generic,
        ):
            result = workspace_tools.restore_workspace("Snap Fallback", launch_missing=False)

        check(not result["complete"], f"degraded Snap fallback must not claim complete: {result}")
        check(result.get("degraded") is True, f"degraded flag missing: {result}")
        check(result["restored_count"] == 2, f"fallback restore count wrong: {result}")
        check(result["warnings"], "native Snap failure did not surface a warning")
        check("synthetic input blocked" in result["warnings"][0]["warning"], f"native failure reason lost: {result}")
        generic.assert_called_once()

    def snap_corner_sequence_is_paced():
        send_calls = []
        sleep_calls = []

        def capture_send(events):
            send_calls.append([int(event.ki.wVk) for event in events])

        with (
            patch("snap_layout._focus", return_value=None),
            patch("snap_layout._send_inputs", side_effect=capture_send),
            patch("snap_layout._press_key", return_value=None),
            patch("snap_layout.time.sleep", side_effect=lambda value: sleep_calls.append(value)),
            patch("snap_layout.windows_tools._is_window_arranged", return_value=True),
        ):
            snap_layout._send_native_snap(1234, "top-right")

        check(
            send_calls == [
                [snap_layout.VK_LWIN],
                [snap_layout.VK_RIGHT, snap_layout.VK_RIGHT],
                [snap_layout.VK_UP, snap_layout.VK_UP],
                [snap_layout.VK_LWIN],
            ],
            f"corner Snap was not split into paced Win+arrow input: {send_calls}",
        )
        check(0.20 in sleep_calls, f"corner Snap did not pause between side and vertical arrows: {sleep_calls}")

    def snap_focus_uses_attached_input_fallback():
        class FakeKernel32:
            @staticmethod
            def GetCurrentThreadId():
                return 30

        class FakeUser32:
            def __init__(self):
                self.foreground = 900
                self.attach_calls = []

            def GetForegroundWindow(self):
                return self.foreground

            def GetWindowThreadProcessId(self, hwnd, _pid):
                if int(hwnd) == 900:
                    return 10
                if int(hwnd) == 1234:
                    return 20
                return 0

            def AttachThreadInput(self, current, other, attach):
                self.attach_calls.append((int(current), int(other), bool(attach)))
                return True

            def BringWindowToTop(self, _hwnd):
                return True

            def SetForegroundWindow(self, hwnd):
                self.foreground = int(hwnd)
                return True

            def SetFocus(self, _hwnd):
                return 0

        fake_user32 = FakeUser32()
        with (
            patch("snap_layout.windows_tools.focus_window_verified", side_effect=RuntimeError("foreground lock")),
            patch.object(snap_layout, "user32", fake_user32),
            patch.object(snap_layout, "kernel32", FakeKernel32()),
            patch("snap_layout.time.sleep", return_value=None),
            patch("snap_layout._press_key", return_value=None),
        ):
            snap_layout._focus(1234)

        check(fake_user32.foreground == 1234, "attached-input fallback did not activate target")
        check(
            (30, 10, True) in fake_user32.attach_calls and (30, 20, True) in fake_user32.attach_calls,
            f"foreground/target input queues were not attached: {fake_user32.attach_calls}",
        )
        check(
            (30, 10, False) in fake_user32.attach_calls and (30, 20, False) in fake_user32.attach_calls,
            f"attached input queues were not detached: {fake_user32.attach_calls}",
        )

    def snap_divider_uses_fast_path_then_conservative_fallback():
        divider = type("Divider", (), {"axis": "vertical", "target": 1219})()

        fast_calls = []
        fast_verify = iter([False, True])
        with (
            patch(
                "snap_layout._divider_target_verified",
                side_effect=lambda _plan, _divider: next(fast_verify),
            ),
            patch(
                "snap_layout._drag_divider_once",
                side_effect=lambda _plan, _divider, conservative=False: fast_calls.append(bool(conservative)),
            ),
        ):
            snap_layout._drag_plan_divider(object(), divider)

        check(fast_calls == [False], f"successful fast Snap drag did not stop immediately: {fast_calls}")

        fallback_calls = []
        fallback_verify = iter([False, False, True])
        with (
            patch(
                "snap_layout._divider_target_verified",
                side_effect=lambda _plan, _divider: next(fallback_verify),
            ),
            patch(
                "snap_layout._drag_divider_once",
                side_effect=lambda _plan, _divider, conservative=False: fallback_calls.append(bool(conservative)),
            ),
        ):
            snap_layout._drag_plan_divider(object(), divider)

        check(
            fallback_calls == [False, True],
            f"failed fast Snap drag did not fall back to conservative sequence: {fallback_calls}",
        )
        check(
            snap_layout.SNAP_DIVIDER_FAST_HOVER_SECONDS < snap_layout.SNAP_DIVIDER_SLOW_HOVER_SECONDS,
            "fast Snap profile is not actually faster than conservative hover timing",
        )

    def workspace_restore_reports_one_failure_without_losing_successes():
        placement = {
            "show_state": "normal",
            "show_cmd": 1,
            "current_bounds": {
                "left": 0, "top": 0, "right": 800, "bottom": 600,
                "width": 800, "height": 600,
            },
            "visible_bounds": {
                "left": 0, "top": 0, "right": 800, "bottom": 600,
                "width": 800, "height": 600,
            },
            "normal_bounds": {
                "left": 0, "top": 0, "right": 800, "bottom": 600,
                "width": 800, "height": 600,
            },
        }
        captured = {
            "status": "ok",
            "windows": [
                {
                    "hwnd": 10, "title": "One", "pid": 1,
                    "process_name": "one.exe", "exe": r"C:\Apps\one.exe",
                    "placement": placement, "z_index": 0,
                },
                {
                    "hwnd": 11, "title": "Two", "pid": 2,
                    "process_name": "two.exe", "exe": r"C:\Apps\two.exe",
                    "placement": placement, "z_index": 1,
                },
            ],
        }
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=captured),
            patch("workspace_tools.configured_app_alias_for_executable", return_value=None),
            patch("workspace_tools.resource_launch_candidates_for_window", return_value=[]),
        ):
            workspace_tools.save_workspace("Partial Restore")

        current = {
            "status": "ok",
            "windows": [
                {**captured["windows"][0], "hwnd": 20},
                {**captured["windows"][1], "hwnd": 21},
            ],
        }
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=current),
            patch(
                "workspace_tools.restore_window_placements_batch",
                return_value={
                    "restored": [
                        {"hwnd": 20, "state": "normal", "verified": True, "warning": None}
                    ],
                    "failed": [
                        {
                            "hwnd": 21,
                            "error": "window did not settle at the requested bounds",
                            "coordinate_space": "visible",
                            "requested": placement["visible_bounds"],
                            "observed": {**placement["visible_bounds"], "left": 4},
                        }
                    ],
                },
            ),
        ):
            result = workspace_tools.restore_workspace("Partial Restore", launch_missing=False)

        check(result["restored_count"] == 1, f"successful window was lost: {result}")
        check(len(result["failed"]) == 1, f"failed window was not reported: {result}")
        check(not result["complete"], "partial restore incorrectly reported complete")
        check(result["failed"][0]["title"] == "Two", "failure lost window identity")
        workspace_tools.forget_workspace("Partial Restore")

    def workspace_reopen_waits_past_splash_for_stable_main_window():
        placement = {
            "show_state": "normal",
            "show_cmd": 1,
            "current_bounds": {
                "left": 100, "top": 100, "right": 900, "bottom": 700,
                "width": 800, "height": 600,
            },
            "visible_bounds": {
                "left": 110, "top": 100, "right": 890, "bottom": 690,
                "width": 780, "height": 590,
            },
            "normal_bounds": {
                "left": 100, "top": 100, "right": 900, "bottom": 700,
                "width": 800, "height": 600,
            },
            "geometry_space": "dwm_extended_frame",
        }
        saved_capture = {
            "status": "ok",
            "windows": [
                {
                    "hwnd": 300,
                    "title": "scene.blend - Blender",
                    "pid": 30,
                    "process_name": "blender.exe",
                    "exe": r"C:\Program Files\Blender Foundation\Blender\blender.exe",
                    "resizable": True,
                    "placement": placement,
                    "z_index": 0,
                }
            ],
        }
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=saved_capture),
            patch("workspace_tools.configured_app_alias_for_executable", return_value=None),
            patch("workspace_tools.resource_launch_candidates_for_window", return_value=[]),
        ):
            workspace_tools.save_workspace("Splash Restore")

        splash = {
            **saved_capture["windows"][0],
            "hwnd": 301,
            "title": "Blender Loading",
            "resizable": False,
        }
        main = {
            **saved_capture["windows"][0],
            "hwnd": 302,
            "title": "scene.blend - Blender",
            "resizable": True,
        }
        calls = {"value": 0}

        def capture_sequence():
            calls["value"] += 1
            if calls["value"] == 1:
                return {"status": "ok", "windows": []}
            if calls["value"] == 2:
                return {"status": "ok", "windows": [splash]}
            return {"status": "ok", "windows": [main]}

        with (
            patch("workspace_tools.capture_workspace_windows", side_effect=capture_sequence),
            patch("workspace_tools.launch_workspace_provider", return_value={"provider": "blender"}) as launch,
            patch("workspace_tools.time.sleep", return_value=None),
            patch(
                "workspace_tools.restore_window_placements_batch",
                return_value={
                    "restored": [
                        {"hwnd": 302, "state": "normal", "verified": True, "warning": None}
                    ],
                    "failed": [],
                },
            ) as restore_batch,
        ):
            result = workspace_tools.restore_workspace("Splash Restore", timeout_seconds=1.0)

        check(result["complete"], f"reopened workspace was incomplete: {result}")
        check(result["launched"] == ["provider:blender"], f"unexpected launch record: {result}")
        check(not result["pending"], f"splash incorrectly remained pending: {result}")
        launch.assert_called_once()
        restore_batch.assert_called_once_with([{"hwnd": 302, "placement": placement}])
        workspace_tools.forget_workspace("Splash Restore")

    def workspace_reopen_leaves_transient_splash_pending():
        placement = {
            "show_state": "normal",
            "show_cmd": 1,
            "current_bounds": {
                "left": 10, "top": 10, "right": 810, "bottom": 610,
                "width": 800, "height": 600,
            },
            "visible_bounds": {
                "left": 20, "top": 10, "right": 800, "bottom": 600,
                "width": 780, "height": 590,
            },
            "normal_bounds": {
                "left": 10, "top": 10, "right": 810, "bottom": 610,
                "width": 800, "height": 600,
            },
            "geometry_space": "dwm_extended_frame",
        }
        saved_capture = {
            "status": "ok",
            "windows": [
                {
                    "hwnd": 400,
                    "title": "Blender",
                    "pid": 40,
                    "process_name": "blender.exe",
                    "exe": r"C:\Program Files\Blender Foundation\Blender\blender.exe",
                    "resizable": True,
                    "placement": placement,
                    "z_index": 0,
                }
            ],
        }
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=saved_capture),
            patch("workspace_tools.configured_app_alias_for_executable", return_value=None),
            patch("workspace_tools.resource_launch_candidates_for_window", return_value=[]),
        ):
            workspace_tools.save_workspace("Pending Splash")

        splash = {
            **saved_capture["windows"][0],
            "hwnd": 401,
            "title": "Blender Loading",
            "resizable": False,
        }
        calls = {"value": 0}

        def capture_sequence():
            calls["value"] += 1
            if calls["value"] == 1:
                return {"status": "ok", "windows": []}
            return {"status": "ok", "windows": [splash]}

        with (
            patch("workspace_tools.capture_workspace_windows", side_effect=capture_sequence),
            patch("workspace_tools.launch_workspace_provider", return_value={"provider": "blender"}),
            patch("workspace_tools.time.sleep", return_value=None),
            patch(
                "workspace_tools.restore_window_placements_batch",
                return_value={"restored": [], "failed": []},
            ) as restore_batch,
        ):
            result = workspace_tools.restore_workspace("Pending Splash", timeout_seconds=0.01)

        check(not result["complete"], "transient-only restore incorrectly reported complete")
        check(len(result["pending"]) == 1, f"pending startup was not reported: {result}")
        check(
            "Blender Loading" in result["pending"][0].get("transient_titles", []),
            f"transient window identity was lost: {result}",
        )
        restore_batch.assert_not_called()
        workspace_tools.forget_workspace("Pending Splash")

    def workspace_launch_failure_does_not_abort_other_windows():
        placement = {
            "show_state": "normal",
            "show_cmd": 1,
            "current_bounds": {
                "left": 0, "top": 0, "right": 800, "bottom": 600,
                "width": 800, "height": 600,
            },
            "visible_bounds": {
                "left": 0, "top": 0, "right": 800, "bottom": 600,
                "width": 800, "height": 600,
            },
            "normal_bounds": {
                "left": 0, "top": 0, "right": 800, "bottom": 600,
                "width": 800, "height": 600,
            },
            "geometry_space": "dwm_extended_frame",
        }
        saved_capture = {
            "status": "ok",
            "windows": [
                {
                    "hwnd": 500, "title": "Blender", "pid": 50,
                    "process_name": "blender.exe",
                    "exe": r"C:\Apps\blender.exe", "resizable": True,
                    "placement": placement, "z_index": 0,
                },
                {
                    "hwnd": 501, "title": "Notes", "pid": 51,
                    "process_name": "notepad.exe",
                    "exe": r"C:\Windows\System32\notepad.exe", "resizable": True,
                    "placement": placement, "z_index": 1,
                },
            ],
        }
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=saved_capture),
            patch("workspace_tools.configured_app_alias_for_executable", return_value=None),
            patch("workspace_tools.resource_launch_candidates_for_window", return_value=[]),
        ):
            workspace_tools.save_workspace("Launch Isolation")

        current = {
            "status": "ok",
            "windows": [{**saved_capture["windows"][1], "hwnd": 601}],
        }
        with (
            patch("workspace_tools.capture_workspace_windows", return_value=current),
            patch("workspace_tools.launch_workspace_provider", side_effect=RuntimeError("launcher broke")),
            patch(
                "workspace_tools.restore_window_placements_batch",
                return_value={
                    "restored": [
                        {"hwnd": 601, "state": "normal", "verified": True, "warning": None}
                    ],
                    "failed": [],
                },
            ),
        ):
            result = workspace_tools.restore_workspace("Launch Isolation", timeout_seconds=0.01)

        check(result["restored_count"] == 1, f"other window was not restored: {result}")
        check(len(result["failed"]) == 1, f"launch failure was not isolated: {result}")
        check("launcher broke" in result["failed"][0]["reason"], f"launch error was lost: {result}")
        workspace_tools.forget_workspace("Launch Isolation")

    def steam_provider_reopens_main_client_without_webhelper():
        import workspace_providers

        saved = {
            "title": "Steam",
            "process_name": "steamwebhelper.exe",
            "exe": r"C:\Program Files (x86)\Steam\bin\cef\cef.win7x64\steamwebhelper.exe",
            "workspace_provider": "steam",
        }

        # Preferred path: the registered Steam URI reopens the client whether
        # Steam is fully closed or merely sitting in the tray.
        with (
            patch.object(workspace_providers.os, "name", "nt"),
            patch.object(workspace_providers.os, "startfile", create=True) as startfile,
            patch.object(workspace_providers, "_spawn") as spawn,
        ):
            result = workspace_providers.launch_workspace_provider(saved)
        check(result["provider"] == "steam", f"Steam provider was not selected: {result}")
        check(result.get("transport") == "registered_protocol", f"Steam URI path was not used: {result}")
        startfile.assert_called_once_with("steam://open/main")
        spawn.assert_not_called()

        # Fallback path: if the protocol association is unavailable, resolve
        # the real Steam client and never dispatch steamwebhelper.exe itself.
        with (
            patch.object(workspace_providers.os, "name", "nt"),
            patch.object(
                workspace_providers.os,
                "startfile",
                side_effect=OSError("protocol unavailable"),
                create=True,
            ),
            patch.object(
                workspace_providers,
                "_find_steam_executable",
                return_value=r"C:\Program Files (x86)\Steam\steam.exe",
            ) as find_steam,
            patch.object(
                workspace_providers,
                "_spawn",
                return_value={"provider": "steam", "dispatched": True},
            ) as spawn,
        ):
            fallback = workspace_providers.launch_workspace_provider(saved)

        check(fallback["provider"] == "steam", f"Steam fallback failed: {fallback}")
        find_steam.assert_called_once_with(saved)
        spawn.assert_called_once_with([r"C:\Program Files (x86)\Steam\steam.exe"], "steam")


    return [
        run_case("Memory: named resource aliases + overwrite guard", resource_round_trip_and_aliases),
        run_case("Memory: typed development project validation", typed_project_validation),
        run_case("Memory: typed project open dispatch", typed_project_open_dispatch),
        run_case("Memory: natural names resolve inside action tools", natural_saved_name_resolution_is_internal),
        run_case("Memory: remember active VS Code project", remember_current_vscode_project_from_bridge),
        run_case("Memory: active VS Code project uses real focus history", current_vscode_project_uses_last_real_focus),
        run_case("Workspace: persist per-app internal state", workspace_app_state_capture_is_persisted),
        run_case("Workspace: restore app state before geometry", workspace_app_state_restore_runs_before_geometry),
        run_case("Workspace: app-state failure is honestly degraded", workspace_app_state_warning_marks_restore_degraded),
        run_case("Workspace: browser window matches active saved tab", browser_window_match_prefers_active_tab_title),
        run_case("Workspace: browser restore sanitizes blank tabs", browser_restore_sanitizes_blank_tabs),
        run_case("Workspace: VS Code defers editor restore to native session", vscode_restore_uses_native_editor_session),
        run_case("Workspace: VS Code focus is not window identity", vscode_focus_is_not_window_identity),
        run_case("Workspace: VS Code does not hijack another project", vscode_wrong_project_is_not_hijacked),
        run_case("Workspace: Terminal restore never replays old commands", terminal_restore_never_replays_saved_commands),
        run_case("Workspace: app providers launch saved context", app_state_provider_launch_uses_saved_context),
        run_case("Workspace: capture/restore placement + forget", workspace_capture_restore_and_forget),
        run_case("Workspace: infer native three-window Snap topology", native_snap_plan_for_three_window_layout),
        run_case("Workspace: prefer native Snap restore for arranged group", workspace_restore_prefers_native_snap_for_arranged_group),
        run_case("Workspace: native Snap failure is reported as degraded", workspace_native_snap_failure_falls_back_to_exact_geometry),
        run_case("Workspace: paced native corner Snap input", snap_corner_sequence_is_paced),
        run_case("Workspace: attached-input focus fallback", snap_focus_uses_attached_input_fallback),
        run_case("Workspace: fast Snap divider with conservative fallback", snap_divider_uses_fast_path_then_conservative_fallback),
        run_case("Workspace: isolate/report per-window restore failure", workspace_restore_reports_one_failure_without_losing_successes),
        run_case("Workspace: reopen waits past splash for stable main window", workspace_reopen_waits_past_splash_for_stable_main_window),
        run_case("Workspace: transient splash remains pending without resize", workspace_reopen_leaves_transient_splash_pending),
        run_case("Workspace: launch failure does not abort other windows", workspace_launch_failure_does_not_abort_other_windows),
        run_case("Workspace: Steam reopens main client without steamwebhelper", steam_provider_reopens_main_client_without_webhelper),
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
    import web_search as web_search_module
    import windows_tools

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Runtime.Host.permission_policy import (
        AUTO_APPROVED_TOOLS,
        FILE_WRITE_TARGETS,
    )

    def registration_and_annotations():
        manager = server.mcp._tool_manager
        tools = manager.list_tools()
        names = [tool.name for tool in tools]
        check(len(names) == 48, f"expected 48 model-facing tools, got {len(names)}")
        check("read_image" in names, "read_image not registered")
        check("write_file" in names, "write_file not registered")
        for expected in (
            "remember_resource",
            "remember_current_project",
            "search_saved",
            "open_resource",
            "save_workspace",
            "restore_workspace",
            "forget_saved",
            "set_window_state",
            "close_all_windows_for_app",
            "close_every_app_on_active_desktop",
            "web_search",
        ):
            check(expected in names, f"{expected} not registered")
        check("dexter_ping" not in names, "dexter_ping should be replaced by read_image")
        for tool in tools:
            check(tool.annotations is not None, f"missing annotations: {tool.name}")
        classified = AUTO_APPROVED_TOOLS | FILE_WRITE_TARGETS.keys()
        check(
            set(names) == classified,
            f"permission policy classification differs from live tools: "
            f"missing={set(names) - classified}, stale={classified - set(names)}",
        )

    def close_all_wrapper_contract():
        with patch(
            "server.close_all_apps_on_active_desktop",
            return_value={
                "scope": "active_virtual_desktop",
                "desktop_filter": "ivirtualdesktopmanager",
                "desktop_filter_warning": None,
                "closed_count": 2,
                "remaining_windows": [],
                "request_failures": [],
                "skipped_desktop_check": [],
                "note": None,
            },
        ) as mocked:
            result = server.close_every_app_on_active_desktop(3.0)
        check(result["scope"] == "active_virtual_desktop", f"wrong close-all scope: {result}")
        check(result["closed_count"] == 2, f"close-all count was dropped: {result}")
        mocked.assert_called_once_with(3.0)

    def remember_current_project_wrapper_contract():
        with patch(
            "server.remember_current_vscode_project",
            return_value={
                "resource": {
                    "name": "FriendBox",
                    "kind": "vscode",
                    "target": r"C:\Work\FriendBox",
                    "aliases": ["friend terminal"],
                },
                "created": True,
                "updated": False,
            },
        ) as mocked:
            result = server.remember_current_project(
                "FriendBox",
                ["friend terminal"],
            )
        check(result["name"] == "FriendBox" and result["created"], f"wrong project memory result: {result}")
        mocked.assert_called_once_with("FriendBox", ["friend terminal"], False)

    def close_named_app_wrapper_contract():
        with patch(
            "server.close_app_windows_on_active_desktop",
            return_value={
                "scope": "one_process_name_on_active_virtual_desktop",
                "process_name": "Code.exe",
                "closed_count": 2,
                "remaining_windows": [],
                "request_failures": [],
                "skipped_desktop_check": [],
                "note": None,
            },
        ) as mocked:
            result = server.close_all_windows_for_app("Code.exe", 3.0)
        check(
            result["scope"] == "one_process_name_on_active_virtual_desktop",
            f"wrong named-app scope: {result}",
        )
        check(result["closed_count"] == 2, f"named close count was dropped: {result}")
        mocked.assert_called_once_with("Code.exe", 3.0)

    def close_all_filters_other_virtual_desktops():
        class FakeUser32:
            def __init__(self):
                self.alive = {101, 102}
                self.posted = []

            @staticmethod
            def GetShellWindow():
                return 0

            @staticmethod
            def IsWindowVisible(_hwnd):
                return True

            def EnumWindows(self, callback, _lparam):
                callback(101, 0)
                callback(102, 0)
                return True

            def IsWindow(self, hwnd):
                return int(hwnd) in self.alive

            def PostMessageW(self, hwnd, _message, _wparam, _lparam):
                handle = int(hwnd)
                self.posted.append(handle)
                self.alive.discard(handle)
                return True

        class FakeDesktopManager:
            def __init__(self):
                self.checked = []
                self.closed = False

            def is_window_on_current_desktop(self, hwnd):
                self.checked.append(hwnd)
                return hwnd == 101

            def close(self):
                self.closed = True

        fake_user32 = FakeUser32()
        manager = FakeDesktopManager()
        with (
            patch("windows_tools._require_windows", return_value=None),
            patch.object(windows_tools, "user32", fake_user32),
            patch("windows_tools.VirtualDesktopManager", return_value=manager),
            patch("windows_tools._is_window_cloaked", return_value=False),
            patch("windows_tools._window_class_name", return_value="TestWindow"),
            patch("windows_tools._window_title", side_effect=lambda hwnd: f"Window {int(hwnd)}"),
            patch("windows_tools._window_pid", side_effect=lambda hwnd: 10_000 + int(hwnd)),
            patch("windows_tools._window_bounds", return_value={"width": 800, "height": 600}),
            patch("windows_tools._process_name", return_value="test.exe"),
        ):
            result = windows_tools.close_all_apps_on_active_desktop(0)

        check(manager.checked == [101, 102], f"desktop membership was not checked: {manager.checked}")
        check(fake_user32.posted == [101], f"inactive desktop window was closed: {fake_user32.posted}")
        check(result["closed_count"] == 1 and result["verified"], f"unexpected close result: {result}")
        check(manager.closed, "virtual desktop COM wrapper was not released")

    def close_named_app_filters_process_and_desktop():
        class FakeUser32:
            def __init__(self):
                self.alive = {101, 102, 103}
                self.posted = []

            @staticmethod
            def GetShellWindow():
                return 0

            @staticmethod
            def IsWindowVisible(_hwnd):
                return True

            def EnumWindows(self, callback, _lparam):
                callback(101, 0)
                callback(102, 0)
                callback(103, 0)
                return True

            def IsWindow(self, hwnd):
                return int(hwnd) in self.alive

            def PostMessageW(self, hwnd, _message, _wparam, _lparam):
                handle = int(hwnd)
                self.posted.append(handle)
                self.alive.discard(handle)
                return True

        class FakeDesktopManager:
            def __init__(self):
                self.checked = []
                self.closed = False

            def is_window_on_current_desktop(self, hwnd):
                self.checked.append(hwnd)
                return hwnd == 101

            def close(self):
                self.closed = True

        fake_user32 = FakeUser32()
        manager = FakeDesktopManager()
        process_names = {
            10_101: "Code.exe",
            10_102: "CODE.EXE",
            10_103: "notepad.exe",
        }
        with (
            patch("windows_tools._require_windows", return_value=None),
            patch.object(windows_tools, "user32", fake_user32),
            patch("windows_tools.VirtualDesktopManager", return_value=manager),
            patch("windows_tools._is_window_cloaked", return_value=False),
            patch("windows_tools._window_class_name", return_value="TestWindow"),
            patch("windows_tools._window_title", side_effect=lambda hwnd: f"Window {int(hwnd)}"),
            patch("windows_tools._window_pid", side_effect=lambda hwnd: 10_000 + int(hwnd)),
            patch("windows_tools._window_bounds", return_value={"width": 800, "height": 600}),
            patch("windows_tools._process_name", side_effect=lambda pid: process_names[int(pid)]),
        ):
            result = windows_tools.close_app_windows_on_active_desktop("code", 0)

        check(manager.checked == [101, 102], f"non-target app reached desktop check: {manager.checked}")
        check(fake_user32.posted == [101], f"wrong app/desktop window was closed: {fake_user32.posted}")
        check(result["closed_count"] == 1 and result["verified"], f"unexpected named close: {result}")
        check(manager.closed, "virtual desktop COM wrapper was not released")

    def bounded_web_search_contract():
        class FakeDDGS:
            def __init__(self, timeout):
                check(timeout == web_search_module.SEARCH_TIMEOUT_SECONDS, "wrong search timeout")

            def text(self, query, **kwargs):
                check(query == "Dexter release", "query changed")
                check(kwargs["max_results"] == 5, "search result bound changed")
                return [
                    {
                        "title": "  Dexter   Release  ",
                        "href": "https://example.com/release",
                        "body": " compact   snippet ",
                    },
                    {
                        "title": "Unsafe",
                        "href": "javascript:alert(1)",
                        "body": "must be removed",
                    },
                ]

        with patch.object(web_search_module, "DDGS", FakeDDGS):
            result = web_search_module.search_web("Dexter release")
        check(
            result["results"] == [
                {
                    "title": "Dexter Release",
                    "url": "https://example.com/release",
                    "snippet": "compact snippet",
                }
            ],
            f"web results were not compact/filtered: {result}",
        )

    def workspace_delete_wrapper_contract():
        with patch(
            "server.forget_workspace",
            return_value={
                "status": "ok",
                "operation": "forget_workspace",
                "id": "ws_test",
                "name": "Delete Me",
                "deleted": True,
            },
        ) as mocked:
            result = server.forget_saved("workspace", "Delete Me")
        check(result == {"name": "Delete Me", "deleted": True}, f"unexpected workspace delete wrapper result: {result}")
        mocked.assert_called_once_with("Delete Me")

    def workspace_app_state_wrapper_contract():
        with patch(
            "server.save_current_workspace",
            return_value={
                "name": "Coding",
                "window_count": 2,
                "captured_windows": [],
                "created": True,
                "updated": False,
                "app_state_window_count": 2,
                "state_warnings": [{"warning": "example"}],
            },
        ):
            saved = server.save_workspace("Coding")
        check(saved.get("app_state_window_count") == 2, f"save wrapper dropped app-state count: {saved}")
        check(saved.get("state_warnings") == [{"warning": "example"}], f"save wrapper dropped app-state warnings: {saved}")

        with patch(
            "server.restore_saved_workspace",
            return_value={
                "name": "Coding",
                "restored_count": 2,
                "launched": [],
                "missing": [],
                "pending": [],
                "ambiguous": [],
                "failed": [],
                "warnings": [],
                "app_state_restored": [{"kind": "vscode"}, {"kind": "chromium_browser"}],
                "app_state_restored_count": 2,
                "degraded": False,
                "complete": True,
            },
        ):
            restored = server.restore_workspace("Coding")
        check(restored.get("app_state_restored_count") == 2, f"restore wrapper dropped app-state count: {restored}")
        check(len(restored.get("app_state_restored") or []) == 2, f"restore wrapper dropped app-state details: {restored}")

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
        run_case("MCP: remember-current-project wrapper", remember_current_project_wrapper_contract),
        run_case("MCP: named-app close wrapper", close_named_app_wrapper_contract),
        run_case("MCP: active-desktop close-all wrapper", close_all_wrapper_contract),
        run_case("MCP: close-all filters inactive desktops", close_all_filters_other_virtual_desktops),
        run_case("MCP: named close filters app + inactive desktop", close_named_app_filters_process_and_desktop),
        run_case("MCP: bounded web-search contract", bounded_web_search_contract),
        run_case("MCP: delete-workspace wrapper contract", workspace_delete_wrapper_contract),
        run_case("MCP: workspace app-state wrapper contract", workspace_app_state_wrapper_contract),
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
    "memory": memory_suite,
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
