"""Sync ``sources-raw/`` (originals) → ``sources/`` (parsed markdown).

Per docs/architecture.md, the engine never reads raw originals
directly. ``sync_sources`` is the one-way pipeline: for each file under
``sources-raw/``, parse it (or skip with reason) and write a ``.md``
mirror under ``sources/`` if the source has changed (mtime newer than
the parsed copy). Existing parsed files that already match are
left alone — idempotent re-runs are cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from spec_writer.engine.atomic_io import atomic_write
from spec_writer.io.parser import ParseResult, ParseSkipped, parse_source


def _empty_list() -> list[Path]:
    return []


def _empty_str_list() -> list[str]:
    return []


@dataclass(frozen=True)
class SyncReport:
    processed: list[Path] = field(default_factory=_empty_list)
    skipped_unchanged: list[Path] = field(default_factory=_empty_list)
    skipped_unsupported: list[str] = field(default_factory=_empty_str_list)

    @property
    def total_written(self) -> int:
        return len(self.processed)


def _target_for(source: Path, sources_raw: Path, sources_dir: Path) -> Path:
    relative = source.relative_to(sources_raw)
    if source.suffix.lower() == ".pdf":
        relative = relative.with_suffix(".pdf.md")
    elif source.suffix.lower() in (".md", ".markdown"):
        relative = relative.with_suffix(".md")
    elif source.suffix.lower() == ".txt":
        relative = relative.with_suffix(".md")
    return sources_dir / relative


def sync_sources(sources_raw: Path, sources_dir: Path) -> SyncReport:
    """Mirror ``sources-raw/`` → ``sources/`` with up-to-date parsing.

    The sync is mtime-driven: if the target is newer than the source,
    we assume it's already up to date and skip.
    """
    processed: list[Path] = []
    unchanged: list[Path] = []
    unsupported: list[str] = []

    if not sources_raw.exists():
        return SyncReport(processed, unchanged, unsupported)

    for source in sorted(sources_raw.rglob("*")):
        if not source.is_file():
            continue
        target = _target_for(source, sources_raw, sources_dir)
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            unchanged.append(source)
            continue

        outcome = parse_source(source)
        if isinstance(outcome, ParseSkipped):
            unsupported.append(f"{source.relative_to(sources_raw)}: {outcome.reason}")
            continue
        assert isinstance(outcome, ParseResult)
        atomic_write(target, outcome.content)
        processed.append(source)

    return SyncReport(processed, unchanged, unsupported)


__all__ = ["SyncReport", "sync_sources"]
