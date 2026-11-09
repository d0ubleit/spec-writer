"""LanguageAnalyzer Protocol + FilesystemOnlyAnalyzer fallback.

A :class:`LanguageAnalyzer` extracts a best-effort list of
:class:`Symbol`-s from a single source file. Analyzers are
**stateless** and **synchronous** — analyzing one file shouldn't fan
out into I/O for other files.

The Protocol shape mirrors the v1 ``recovery/symbol_index.py`` symbol
contract but trimmed: we don't need imports/references in V2 at the
analyzer layer — the agent can grep for those on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

SymbolKind = Literal[
    "file",  # FilesystemOnlyAnalyzer placeholder
    "function",
    "method",
    "class",
    "struct",
    "interface",
    "enum",
    "trait",
    "type",
    "const",
    "route",
]


Visibility = Literal["public", "internal", "private", "protected", "unknown"]


def _empty_str_list() -> list[str]:
    return []


@dataclass(frozen=True)
class Symbol:
    """A best-effort hint about something defined in a source file.

    ``end_line`` may be ``-1`` when the analyzer could not determine
    the end (e.g. ``FilesystemOnlyAnalyzer`` uses the whole file as a
    single symbol). Coverage logic must treat ``-1`` as "rest of file".
    """

    name: str
    kind: SymbolKind
    visibility: Visibility
    start_line: int  # 1-based; 0 means "file-level / no line info"
    end_line: int  # 1-based inclusive; -1 means "rest of file"
    signature: str = ""
    # Tags let mixins / analyzers attach hints ("deprecated", "test-only").
    tags: list[str] = field(default_factory=_empty_str_list)


@runtime_checkable
class LanguageAnalyzer(Protocol):
    """A language analyzer recognises a file and lists its public surface.

    Implementations live in :mod:`agent.code_analyzers` and register
    themselves via :func:`register_analyzer` at import time. The
    ``discover_code_inventory`` MCP tool selects an analyzer per file
    by calling :func:`analyzer_for`.
    """

    @property
    def language(self) -> str:
        """Short identifier — ``"python"``, ``"php"``, ``"typescript"``, …"""
        ...

    @property
    def supported_extensions(self) -> list[str]:
        """File suffixes (with leading dot) this analyzer recognises."""
        ...

    def discover_symbols(self, path: Path, source: str) -> list[Symbol]:
        """Return symbols from one file.

        ``source`` is the file content as a string (loaded by the
        caller — analyzers don't do I/O). Empty list on parser failure
        — never raise.
        """
        ...


# ── Fallback ──────────────────────────────────────────────────────────────


class FilesystemOnlyAnalyzer:
    """Default analyzer for unrecognised extensions.

    Returns one ``file``-kind :class:`Symbol` covering the entire file;
    coverage logic falls back to file granularity on these files.
    Always active — never the "no analyzer" branch.
    """

    language: str = "filesystem"
    supported_extensions: list[str] = []  # noqa: RUF012 — fallback never mutated

    def discover_symbols(self, path: Path, source: str) -> list[Symbol]:
        del source  # not inspected
        line_count = 0
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                line_count = sum(1 for _ in fh)
        except OSError:
            line_count = 0
        return [
            Symbol(
                name=path.stem or path.name,
                kind="file",
                visibility="public",
                start_line=1 if line_count else 0,
                end_line=line_count if line_count else -1,
                signature=str(path.name),
            )
        ]


__all__ = [
    "FilesystemOnlyAnalyzer",
    "LanguageAnalyzer",
    "Symbol",
    "SymbolKind",
    "Visibility",
]
