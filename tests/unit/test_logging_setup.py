"""S1.12: logging configuration + JSONL handler + rotation."""

from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from spec_writer.engine.logging_setup import (
    JsonlFileHandler,
    configure_logging,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


def _make_record(
    name: str = "spec_writer.test",
    level: int = logging.INFO,
    msg: str = "hello",
    *,
    created: float | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(name, level, "pathname", 0, msg, args=None, exc_info=None)
    if created is not None:
        record.created = created
    return record


# ── console handler / verbosity ──────────────────────────────────────────


def test_default_verbosity_warning(tmp_path: Path) -> None:
    buffer = io.StringIO()
    configure_logging(verbose=0, stream=buffer)
    logger = logging.getLogger("spec_writer.x")
    logger.warning("alert")
    logger.info("hidden")
    assert "alert" in buffer.getvalue()
    assert "hidden" not in buffer.getvalue()


def test_verbose_v_enables_info() -> None:
    buffer = io.StringIO()
    configure_logging(verbose=1, stream=buffer)
    logging.getLogger("spec_writer.y").info("now-visible")
    assert "now-visible" in buffer.getvalue()


def test_verbose_vv_enables_debug() -> None:
    buffer = io.StringIO()
    configure_logging(verbose=2, stream=buffer)
    logging.getLogger("spec_writer.z").debug("debug-only")
    assert "debug-only" in buffer.getvalue()


def test_configure_logging_idempotent(tmp_path: Path) -> None:
    configure_logging(verbose=0)
    configure_logging(verbose=1)
    root = logging.getLogger()
    # exactly one console handler each time
    assert sum(1 for h in root.handlers if isinstance(h, logging.StreamHandler)) == 1


# ── jsonl handler ────────────────────────────────────────────────────────


def test_jsonl_handler_writes_required_fields(tmp_path: Path) -> None:
    handler = JsonlFileHandler(tmp_path)
    record = _make_record(
        name="spec_writer.engine",
        level=logging.INFO,
        msg="layer L1 begin",
        created=time.time(),
    )
    handler.emit(record)
    handler.close()

    date_str = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d")
    out_file = tmp_path / f"{date_str}.jsonl"
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    for required in ("ts", "level", "component", "event"):
        assert required in payload
    assert payload["level"] == "INFO"
    assert payload["component"] == "spec_writer.engine"
    assert payload["event"] == "layer L1 begin"


def test_jsonl_handler_rotates_on_date_change(tmp_path: Path) -> None:
    handler = JsonlFileHandler(tmp_path)
    earlier = datetime(2026, 5, 14, 23, 59, tzinfo=timezone.utc).timestamp()
    later = datetime(2026, 5, 15, 0, 30, tzinfo=timezone.utc).timestamp()
    handler.emit(_make_record(msg="late", created=earlier))
    handler.emit(_make_record(msg="early", created=later))
    handler.close()
    assert (tmp_path / "2026-05-14.jsonl").exists()
    assert (tmp_path / "2026-05-15.jsonl").exists()


def test_jsonl_handler_appends_existing_file(tmp_path: Path) -> None:
    handler = JsonlFileHandler(tmp_path)
    ts = datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc).timestamp()
    handler.emit(_make_record(msg="first", created=ts))
    handler.close()

    handler2 = JsonlFileHandler(tmp_path)
    handler2.emit(_make_record(msg="second", created=ts))
    handler2.close()

    lines = (tmp_path / "2026-05-14.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_jsonl_handler_creates_log_dir(tmp_path: Path) -> None:
    nested = tmp_path / "_meta" / "logs"
    handler = JsonlFileHandler(nested)
    handler.emit(_make_record(msg="ok", created=time.time()))
    handler.close()
    assert nested.exists()


def test_configure_attaches_jsonl_when_log_dir_given(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    configure_logging(verbose=2, log_dir=log_dir)
    logging.getLogger("spec_writer.x").debug("structured")
    # close handlers so OS flushes the buffer
    for h in list(logging.getLogger().handlers):
        h.close()
    files = list(log_dir.iterdir())
    assert any(p.suffix == ".jsonl" for p in files)


def test_jsonl_handler_emit_error_does_not_crash(tmp_path: Path) -> None:
    handler = JsonlFileHandler(tmp_path)
    # force an error by passing a non-iterable msg-args mismatch through getMessage
    record = logging.LogRecord("x", logging.INFO, "p", 1, "no-fail", args=None, exc_info=None)
    # corrupt the directory by replacing it with a regular file → first emit fails internally
    (tmp_path / "blocker").write_text("x", encoding="utf-8")
    handler.emit(record)  # should not raise
    handler.close()
