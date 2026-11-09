"""S1.2: atomic_write + atomic_read_modify_write."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from spec_writer.engine.atomic_io import atomic_read_modify_write, atomic_write


def test_atomic_write_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    atomic_write(target, "hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"


def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    atomic_write(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_write_empty_content(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    atomic_write(target, "")
    assert target.exists()
    assert target.read_text(encoding="utf-8") == ""


def test_atomic_write_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "out.txt"
    atomic_write(target, "x")
    assert target.read_text(encoding="utf-8") == "x"


def test_atomic_write_no_tmp_leftover(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    atomic_write(target, "x")
    tmp = target.with_suffix(target.suffix + ".tmp")
    assert not tmp.exists()


def test_atomic_write_retries_permission_error(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    original_replace = Path.replace
    attempts: list[int] = []

    def flaky_replace(self: Path, dst: Any) -> Path:
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("av holds the file")
        return original_replace(self, dst)

    with patch.object(Path, "replace", flaky_replace):
        atomic_write(target, "ok")
    assert target.read_text(encoding="utf-8") == "ok"
    assert len(attempts) == 3


def test_atomic_write_raises_after_max_retries(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    def always_fail(self: Path, dst: Any) -> Path:
        raise PermissionError("never frees")

    with patch.object(Path, "replace", always_fail):
        with pytest.raises(PermissionError):
            atomic_write(target, "x")


def test_atomic_write_preserves_destination_on_mid_write_crash(tmp_path: Path) -> None:
    """If fsync fails after writing tmp, destination must remain untouched."""
    target = tmp_path / "out.txt"
    target.write_text("untouched", encoding="utf-8")

    with patch("spec_writer.engine.atomic_io._fsync_file", side_effect=OSError("disk")):
        with pytest.raises(OSError):
            atomic_write(target, "new")

    assert target.read_text(encoding="utf-8") == "untouched"


def test_atomic_read_modify_write(tmp_path: Path) -> None:
    target = tmp_path / "data.txt"
    target.write_text("v=1\n", encoding="utf-8")
    atomic_read_modify_write(target, lambda s: s.replace("v=1", "v=2"))
    assert target.read_text(encoding="utf-8") == "v=2\n"


def test_atomic_read_modify_write_creates_when_missing(tmp_path: Path) -> None:
    target = tmp_path / "data.txt"
    atomic_read_modify_write(target, lambda s: s + "first\n")
    assert target.read_text(encoding="utf-8") == "first\n"
