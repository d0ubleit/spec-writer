"""Atomic write helpers for spec files and _meta JSON state.

The write protocol:
  1. write content into ``<target>.tmp``
  2. fsync(tmp)
  3. replace(tmp → target)  — retry on PermissionError (Windows AV)

If any step before the rename fails, the destination is untouched and
the .tmp file is removed. ``Path.replace`` is the same call on POSIX
and Windows — Python abstracts the platform difference.

See docs/architecture.md.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

_RETRY_DELAYS_SEC = (0.05, 0.1, 0.2)


def _fsync_file(path: Path) -> None:
    """Open and fsync the file. Extracted so tests can mock it cleanly."""
    with path.open("rb") as f:
        os.fsync(f.fileno())


def _replace_with_retry(src: Path, dst: Path) -> None:
    last_exc: PermissionError | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS_SEC):
        try:
            src.replace(dst)
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt < len(_RETRY_DELAYS_SEC) - 1:
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically.

    Creates parent directories if they do not exist. On any failure
    before the rename, the destination remains unchanged.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        _fsync_file(tmp)
        _replace_with_retry(tmp, path)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def atomic_read_modify_write(path: Path, fn: Callable[[str], str]) -> None:
    """Read, transform, write back atomically.

    Missing source file is treated as empty string — the transform is
    called with ``""``.
    """
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    new_content = fn(current)
    atomic_write(path, new_content)
