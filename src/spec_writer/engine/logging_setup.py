"""Stdlib-logging configuration plus a daily-rotated JSONL file handler.

Two output sinks:

- Console (stderr): human-readable ``LEVEL name message``. Verbosity
  flags (``-v`` / ``-vv``) widen the visible level.
- ``_meta/logs/<YYYY-MM-DD>.jsonl``: one structured record per line
  (ts/level/component/event). The handler picks the date from each
  record's ``created`` timestamp, so processes that run across
  midnight roll over without explicit prompting.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

_REQUIRED_FIELDS = ("ts", "level", "component", "event")


class JsonlFileHandler(logging.Handler):
    """Write each log record as one JSON line under ``log_dir/<date>.jsonl``."""

    def __init__(self, log_dir: Path) -> None:
        super().__init__()
        self._dir = log_dir
        self._current_date: str | None = None
        self._stream: IO[str] | None = None

    def _rotate_if_needed(self, record: logging.LogRecord) -> None:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
        date_str = ts.strftime("%Y-%m-%d")
        if date_str == self._current_date and self._stream is not None:
            return
        if self._stream is not None:
            self._stream.close()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._stream = (self._dir / f"{date_str}.jsonl").open("a", encoding="utf-8")
        self._current_date = date_str

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._rotate_if_needed(record)
            ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
            payload = {
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "level": record.levelname,
                "component": record.name,
                "event": record.getMessage(),
            }
            assert self._stream is not None
            self._stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        finally:
            super().close()


def configure_logging(
    verbose: int = 0,
    log_dir: Path | None = None,
    stream: IO[str] | None = None,
) -> None:
    """Wire up console + optional JSONL logging on the root logger.

    Idempotent — re-running clears existing handlers first.
    Levels: 0 → WARNING, 1 → INFO, 2+ → DEBUG.
    """
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(stream if stream is not None else sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root.addHandler(console)

    if log_dir is not None:
        jsonl = JsonlFileHandler(log_dir)
        jsonl.setLevel(logging.DEBUG)
        root.addHandler(jsonl)


__all__ = ["JsonlFileHandler", "configure_logging"]
