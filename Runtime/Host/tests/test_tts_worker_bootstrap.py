from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKER = REPO_ROOT / "Runtime" / "Voice" / "neutts2e_worker.py"


def test_worker_file_launch_bootstraps_repo_root() -> None:
    """The worker must resolve Runtime.* imports when launched by file path."""
    probe = (
        "import importlib.util, runpy; "
        f"runpy.run_path({str(WORKER)!r}, run_name='dexter_worker_import_probe'); "
        "assert importlib.util.find_spec('Runtime.Voice.neutts2e_pipelined_stream') is not None"
    )

    # Run outside the repository and without PYTHONPATH so the worker itself is
    # solely responsible for making the Runtime package importable.
    with tempfile.TemporaryDirectory() as cwd:
        result = subprocess.run(
            [sys.executable, "-I", "-c", probe],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )

    assert result.returncode == 0, result.stderr or result.stdout
