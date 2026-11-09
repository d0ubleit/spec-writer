"""Source-document parsers — convert raw inputs to UTF-8 markdown.

Phase 1 supports the formats v1 already handled in production:

- ``.md``, ``.markdown``, ``.txt`` — passthrough
- ``.pdf`` — extracted via PyMuPDF (text layer; OCR is out of scope)
- ``.docx`` — skipped with a structured ParseSkipped result

Unknown extensions also return ParseSkipped so callers can log /
surface to the user without crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pymupdf  # type: ignore[import-untyped]

_PASSTHROUGH_SUFFIXES = frozenset({".md", ".markdown", ".txt"})


@dataclass(frozen=True)
class ParseResult:
    content: str
    """The parsed text."""


@dataclass(frozen=True)
class ParseSkipped:
    """Returned when an input cannot be parsed by the current toolset."""

    reason: str


ParseOutcome = ParseResult | ParseSkipped


def parse_source(path: Path) -> ParseOutcome:
    """Dispatch on extension; return parsed text or a skip reason."""
    suffix = path.suffix.lower()
    if suffix in _PASSTHROUGH_SUFFIXES:
        return ParseResult(content=path.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        return ParseResult(content=_parse_pdf(path))
    if suffix == ".docx":
        return ParseSkipped(
            reason=(
                "docx is not supported in V2 launch; convert to markdown or PDF first "
                "(BACKLOG: structured-doc importer)"
            )
        )
    return ParseSkipped(reason=f"unsupported extension {suffix!r}")


def _parse_pdf(path: Path) -> str:
    pymupdf_any = cast(Any, pymupdf)
    doc = pymupdf_any.open(path)
    try:
        pages: list[str] = []
        for page in doc:
            pages.append(cast(str, page.get_text()))
        return "\n".join(pages)
    finally:
        doc.close()


__all__ = ["ParseOutcome", "ParseResult", "ParseSkipped", "parse_source"]
