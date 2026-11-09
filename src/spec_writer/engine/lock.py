"""EngineLock — prevents concurrent engine processes on the same project.

The lock is a JSON file at ``<meta_dir>/.engine.lock`` containing
``{pid, started_at, host}``. ``acquire()`` returns one of:

- ACQUIRED — lock taken, we own it.
- BUSY     — lock exists and the holding PID is still alive.
- STALE    — lock exists but the holding PID is dead (crashed run).

The caller decides what to do with STALE (recovery options surface to
the user — see docs/architecture.md). The lock is **not** taken on STALE
to avoid silent clobbering of partial state.

See docs/architecture.md.
"""

from __future__ import annotations

import json
import os
import socket
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import cast

from spec_writer.engine.atomic_io import atomic_write

_LOCK_FILENAME = ".engine.lock"


class LockState(str, Enum):
    ACQUIRED = "acquired"
    BUSY = "busy"
    STALE = "stale"


@dataclass(frozen=True)
class LockInfo:
    pid: int
    started_at: str
    host: str


@dataclass(frozen=True)
class LockResult:
    state: LockState
    info: LockInfo | None = None


def _pid_alive(pid: int) -> bool:
    """POSIX-only: signal 0 raises ProcessLookupError if the pid is gone.

    On Windows we treat any reachable pid as alive; CI runs on Linux so
    this is exercised end-to-end there.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user — still alive.
        return True
    return True


def _read_lock_info(path: Path) -> LockInfo | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    data = cast(dict[str, object], parsed)
    pid = data.get("pid")
    started_at = data.get("started_at")
    host = data.get("host")
    if not isinstance(pid, int) or not isinstance(started_at, str) or not isinstance(host, str):
        return None
    return LockInfo(pid=pid, started_at=started_at, host=host)


class EngineLock(AbstractContextManager["LockResult"]):
    """Filesystem-backed mutex for a single ``_meta/`` directory."""

    def __init__(self, meta_dir: Path) -> None:
        self._meta_dir = meta_dir
        self._lock_file = meta_dir / _LOCK_FILENAME
        self._held = False

    def acquire(self) -> LockResult:
        if self._lock_file.exists():
            info = _read_lock_info(self._lock_file)
            if info is None:
                # corrupt file — treat as stale so the caller can clean up
                return LockResult(state=LockState.STALE, info=None)
            if _pid_alive(info.pid):
                return LockResult(state=LockState.BUSY, info=info)
            return LockResult(state=LockState.STALE, info=info)

        self._meta_dir.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        host = socket.gethostname()
        atomic_write(
            self._lock_file,
            json.dumps({"pid": pid, "started_at": started_at, "host": host}),
        )
        self._held = True
        return LockResult(
            state=LockState.ACQUIRED,
            info=LockInfo(pid=pid, started_at=started_at, host=host),
        )

    def release(self) -> None:
        if not self._lock_file.exists():
            self._held = False
            return
        try:
            self._lock_file.unlink()
        except FileNotFoundError:
            pass
        self._held = False

    # Context manager — only enters the body if the lock was acquired.
    def __enter__(self) -> LockResult:
        result = self.acquire()
        if result.state is not LockState.ACQUIRED:
            raise RuntimeError(
                f"could not acquire engine lock at {self._lock_file}: state={result.state.value}"
            )
        return result

    def __exit__(self, *exc: object) -> None:
        if self._held:
            self.release()
