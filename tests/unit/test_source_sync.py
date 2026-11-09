"""S1.11: source parser + sources-raw → sources sync."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pymupdf  # type: ignore[import-untyped]

from spec_writer.io.parser import ParseResult, ParseSkipped, parse_source
from spec_writer.io.source_sync import sync_sources


def _make_pdf(path: Path, text: str) -> None:
    doc = pymupdf.Document()
    page = doc.new_page()
    page.insert_text((50, 100), text)
    doc.save(str(path))
    doc.close()


# ── parser ───────────────────────────────────────────────────────────────


def test_parse_md_passthrough(tmp_path: Path) -> None:
    src = tmp_path / "doc.md"
    src.write_text("# Hello\n", encoding="utf-8")
    outcome = parse_source(src)
    assert isinstance(outcome, ParseResult)
    assert outcome.content == "# Hello\n"


def test_parse_markdown_extension(tmp_path: Path) -> None:
    src = tmp_path / "doc.markdown"
    src.write_text("hi", encoding="utf-8")
    outcome = parse_source(src)
    assert isinstance(outcome, ParseResult)
    assert outcome.content == "hi"


def test_parse_txt_passthrough(tmp_path: Path) -> None:
    src = tmp_path / "doc.txt"
    src.write_text("plain text", encoding="utf-8")
    outcome = parse_source(src)
    assert isinstance(outcome, ParseResult)
    assert outcome.content == "plain text"


def test_parse_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, "Hello PDF\nSecond line")
    outcome = parse_source(pdf)
    assert isinstance(outcome, ParseResult)
    assert "Hello PDF" in outcome.content


def test_parse_docx_skipped(tmp_path: Path) -> None:
    src = tmp_path / "doc.docx"
    src.write_bytes(b"")  # contents irrelevant — we skip on extension
    outcome = parse_source(src)
    assert isinstance(outcome, ParseSkipped)
    assert "docx" in outcome.reason


def test_parse_unknown_extension_skipped(tmp_path: Path) -> None:
    src = tmp_path / "thing.xyz"
    src.write_text("?", encoding="utf-8")
    outcome = parse_source(src)
    assert isinstance(outcome, ParseSkipped)
    assert "unsupported" in outcome.reason


# ── sync_sources ─────────────────────────────────────────────────────────


def test_sync_empty_returns_empty_report(tmp_path: Path) -> None:
    report = sync_sources(tmp_path / "raw", tmp_path / "parsed")
    assert report.processed == []
    assert report.skipped_unchanged == []
    assert report.skipped_unsupported == []


def test_sync_initial_run_processes_all(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    parsed = tmp_path / "parsed"
    (raw / "a.md").write_text("alpha", encoding="utf-8")
    (raw / "b.txt").write_text("beta", encoding="utf-8")

    report = sync_sources(raw, parsed)
    assert len(report.processed) == 2
    assert (parsed / "a.md").read_text(encoding="utf-8") == "alpha"
    assert (parsed / "b.md").read_text(encoding="utf-8") == "beta"


def test_sync_pdf_writes_pdf_md_target(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    parsed = tmp_path / "parsed"
    pdf_path = raw / "biz.pdf"
    _make_pdf(pdf_path, "biz content")

    sync_sources(raw, parsed)
    assert (parsed / "biz.pdf.md").exists()
    assert "biz content" in (parsed / "biz.pdf.md").read_text(encoding="utf-8")


def test_sync_idempotent_second_run(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    parsed = tmp_path / "parsed"
    (raw / "doc.md").write_text("once", encoding="utf-8")
    sync_sources(raw, parsed)

    # ensure target mtime is at or after source mtime
    target = parsed / "doc.md"
    os.utime(target, (time.time() + 5, time.time() + 5))

    report = sync_sources(raw, parsed)
    assert report.processed == []
    assert len(report.skipped_unchanged) == 1


def test_sync_re_runs_when_source_newer(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    parsed = tmp_path / "parsed"
    src = raw / "doc.md"
    src.write_text("v1", encoding="utf-8")
    sync_sources(raw, parsed)

    src.write_text("v2", encoding="utf-8")
    # bump mtime forward so it's strictly newer than parsed target
    future = time.time() + 10
    os.utime(src, (future, future))

    report = sync_sources(raw, parsed)
    assert len(report.processed) == 1
    assert (parsed / "doc.md").read_text(encoding="utf-8") == "v2"


def test_sync_skips_docx_with_message(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    parsed = tmp_path / "parsed"
    (raw / "report.docx").write_bytes(b"")
    report = sync_sources(raw, parsed)
    assert not report.processed
    assert any("docx" in entry for entry in report.skipped_unsupported)


def test_sync_handles_nested_directories(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    parsed = tmp_path / "parsed"
    (raw / "section").mkdir(parents=True)
    (raw / "section" / "child.md").write_text("nested", encoding="utf-8")
    sync_sources(raw, parsed)
    assert (parsed / "section" / "child.md").read_text(encoding="utf-8") == "nested"


def test_sync_missing_raw_dir_returns_empty(tmp_path: Path) -> None:
    report = sync_sources(tmp_path / "absent", tmp_path / "parsed")
    assert report.processed == []
    assert report.skipped_unchanged == []


def test_sync_report_total_written(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    parsed = tmp_path / "parsed"
    (raw / "a.md").write_text("x", encoding="utf-8")
    (raw / "b.md").write_text("y", encoding="utf-8")
    report = sync_sources(raw, parsed)
    assert report.total_written == 2
