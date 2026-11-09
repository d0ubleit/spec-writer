"""S1.0 setup test: pyright strict mode must pass on the package skeleton."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_pyright_clean() -> None:
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "pyright", "src/"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"pyright failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
