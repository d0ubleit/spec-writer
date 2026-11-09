"""Coverage ledger — passive log of REV-mode code touches.

The ledger records two facts about every code file in scope:

- ``visited`` — the agent read or grepped it (via ``Read`` / ``Grep``
  through the SDK; recorded by a thin middleware around tool calls).
- ``covered`` — at least one spec references it via ``implemented_by``
  (recorded by :mod:`agent.tools.record_spec`).

The ledger is **passive**: nothing here knows about layers or
pipelines. The agent reads its state through the ``coverage_report``
MCP tool ([S4.5]). Persisted as JSON under
``_meta/code_visited_ledger.json``; loaded lazily on first access.

Path normalisation: every key is stored as a POSIX-style relative
path from the project root. Absolute paths and ``./`` prefixes are
collapsed at write time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from spec_writer.engine.atomic_io import atomic_write

_LEDGER_FILENAME = "code_visited_ledger.json"


def _empty_visit_log() -> dict[str, list[list[int]]]:
    return {}


def _empty_str_set() -> set[str]:
    return set()


@dataclass
class CoverageLedger:
    """In-memory representation of the ledger.

    ``visited[path]`` = list of (start_line, end_line) pairs the agent
    has read; the same path may appear multiple times via partial
    reads at different ranges. ``covered`` = set of paths referenced
    by at least one spec's ``implemented_by``.
    """

    visited: dict[str, list[list[int]]] = field(default_factory=_empty_visit_log)
    covered: set[str] = field(default_factory=_empty_str_set)

    # ── update primitives ────────────────────────────────────────────

    def record_read(self, path: str, start_line: int = 0, end_line: int = -1) -> None:
        norm = _normalise(path)
        if not norm:
            return
        self.visited.setdefault(norm, []).append([start_line, end_line])

    def record_grep(self, path: str, matched_lines: list[int] | None = None) -> None:
        norm = _normalise(path)
        if not norm:
            return
        for line in matched_lines or [0]:
            self.visited.setdefault(norm, []).append([line, line])

    def record_implementation(self, references: list[str]) -> None:
        """Mark every file path referenced in ``implemented_by`` as covered."""
        for ref in references:
            path = ref.split(":", 1)[0]
            norm = _normalise(path)
            if norm:
                self.covered.add(norm)

    # ── queries ──────────────────────────────────────────────────────

    def is_visited(self, path: str) -> bool:
        return _normalise(path) in self.visited

    def is_covered(self, path: str) -> bool:
        return _normalise(path) in self.covered

    def visited_paths(self) -> set[str]:
        return set(self.visited)

    def covered_paths(self) -> set[str]:
        return set(self.covered)

    def uncovered_against(self, all_paths: list[str]) -> list[str]:
        """Paths in ``all_paths`` not yet covered."""
        covered = self.covered
        seen: set[str] = set()
        out: list[str] = []
        for raw in all_paths:
            norm = _normalise(raw)
            if not norm or norm in covered or norm in seen:
                continue
            seen.add(norm)
            out.append(norm)
        return out


# ── persistence ──────────────────────────────────────────────────────────


def ledger_path(meta_dir: Path) -> Path:
    return meta_dir / _LEDGER_FILENAME


def load(meta_dir: Path) -> CoverageLedger:
    """Read the ledger from disk; return an empty one if absent / unreadable."""
    path = ledger_path(meta_dir)
    if not path.exists():
        return CoverageLedger()
    try:
        raw_any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CoverageLedger()
    if not isinstance(raw_any, dict):
        return CoverageLedger()
    raw = cast(dict[str, Any], raw_any)
    return CoverageLedger(
        visited=_coerce_visited(raw.get("visited")),
        covered=_coerce_covered(raw.get("covered")),
    )


def save(ledger: CoverageLedger, meta_dir: Path) -> None:
    """Atomically write the ledger to disk."""
    payload = {
        "visited": ledger.visited,
        "covered": sorted(ledger.covered),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    atomic_write(ledger_path(meta_dir), rendered)


# ── helpers ──────────────────────────────────────────────────────────────


def _normalise(path: str | Path) -> str:
    if not path:
        return ""
    text = str(path).replace("\\", "/")
    if text.startswith("./"):
        text = text[2:]
    text = text.strip()
    return text


def _coerce_visited(value: object) -> dict[str, list[list[int]]]:
    if not isinstance(value, dict):
        return {}
    typed = cast(dict[str, Any], value)
    out: dict[str, list[list[int]]] = {}
    for key, entries in typed.items():
        if not isinstance(entries, list):
            continue
        entries_typed = cast(list[Any], entries)
        ranges: list[list[int]] = []
        for entry in entries_typed:
            if isinstance(entry, list) and len(cast(list[Any], entry)) == 2:
                pair = cast(list[Any], entry)
                try:
                    ranges.append([int(pair[0]), int(pair[1])])
                except (TypeError, ValueError):
                    continue
        if ranges:
            out[key] = ranges
    return out


def _coerce_covered(value: object) -> set[str]:
    if isinstance(value, list):
        return {item for item in cast(list[Any], value) if isinstance(item, str)}
    return set()


__all__ = ["CoverageLedger", "ledger_path", "load", "save"]
