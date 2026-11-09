"""S1.2: EngineLock — concurrent-process protection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from spec_writer.engine.lock import EngineLock, LockState


def test_acquire_on_clean(tmp_path: Path) -> None:
    lock = EngineLock(tmp_path)
    result = lock.acquire()
    assert result.state is LockState.ACQUIRED
    lock_file = tmp_path / ".engine.lock"
    assert lock_file.exists()
    data = json.loads(lock_file.read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    assert "started_at" in data
    assert "host" in data


def test_acquire_busy_when_alive_pid(tmp_path: Path) -> None:
    (tmp_path / ".engine.lock").write_text(
        json.dumps({"pid": os.getpid(), "started_at": "2099-01-01T00:00:00Z", "host": "h"}),
        encoding="utf-8",
    )
    lock = EngineLock(tmp_path)
    result = lock.acquire()
    assert result.state is LockState.BUSY
    assert result.info is not None
    assert result.info.pid == os.getpid()


def test_acquire_stale_when_dead_pid(tmp_path: Path) -> None:
    dead_pid = 999_999
    (tmp_path / ".engine.lock").write_text(
        json.dumps({"pid": dead_pid, "started_at": "2099-01-01T00:00:00Z", "host": "h"}),
        encoding="utf-8",
    )
    lock = EngineLock(tmp_path)
    with patch("spec_writer.engine.lock._pid_alive", return_value=False):
        result = lock.acquire()
    assert result.state is LockState.STALE
    assert result.info is not None
    assert result.info.pid == dead_pid


def test_release_removes_lock_file(tmp_path: Path) -> None:
    lock = EngineLock(tmp_path)
    lock.acquire()
    lock.release()
    assert not (tmp_path / ".engine.lock").exists()


def test_release_idempotent(tmp_path: Path) -> None:
    lock = EngineLock(tmp_path)
    lock.release()  # nothing to remove → should not raise
    assert not (tmp_path / ".engine.lock").exists()


def test_context_manager_acquires_and_releases(tmp_path: Path) -> None:
    lock = EngineLock(tmp_path)
    with lock as result:
        assert result.state is LockState.ACQUIRED
        assert (tmp_path / ".engine.lock").exists()
    assert not (tmp_path / ".engine.lock").exists()


def test_context_manager_raises_when_busy(tmp_path: Path) -> None:
    (tmp_path / ".engine.lock").write_text(
        json.dumps({"pid": os.getpid(), "started_at": "x", "host": "h"}),
        encoding="utf-8",
    )
    lock = EngineLock(tmp_path)
    with pytest.raises(RuntimeError, match="lock"):
        with lock:
            pass
