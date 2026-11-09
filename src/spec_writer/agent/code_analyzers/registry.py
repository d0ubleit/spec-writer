"""Registry of available :class:`LanguageAnalyzer`-s.

Built up at import time — each analyzer module calls
:func:`register_analyzer` on import. :func:`analyzer_for` resolves a
file path to the right analyzer by extension, falling back to
:class:`FilesystemOnlyAnalyzer` when nothing matches.

There is **no** ``code_scope.languages`` config in methodology.yaml —
analyzers just exist in ``agent/code_analyzers/`` and self-register.
A user restricts the scope via ``code_scope.excludes`` glob patterns.
"""

from __future__ import annotations

from pathlib import Path

from spec_writer.agent.code_analyzers.protocol import (
    FilesystemOnlyAnalyzer,
    LanguageAnalyzer,
)

_REGISTRY: dict[str, LanguageAnalyzer] = {}
"""Keyed by lowercase extension (with leading dot)."""


def register_analyzer(analyzer: LanguageAnalyzer) -> None:
    """Bind ``analyzer`` to every extension it claims to support.

    A later registration for the same extension **overrides** earlier
    ones — useful for tests that want to install a stub analyzer.
    """
    for ext in analyzer.supported_extensions:
        _REGISTRY[ext.lower()] = analyzer


def analyzer_for(path: Path) -> LanguageAnalyzer:
    """Return the analyzer that covers ``path`` (suffix lookup).

    Always returns *something* — falls back to
    :class:`FilesystemOnlyAnalyzer` for unknown extensions.
    """
    return _REGISTRY.get(path.suffix.lower(), _FILESYSTEM_FALLBACK)


def registered_analyzers() -> dict[str, LanguageAnalyzer]:
    """Snapshot the current registry (useful for tests / introspection)."""
    return dict(_REGISTRY)


_FILESYSTEM_FALLBACK: LanguageAnalyzer = FilesystemOnlyAnalyzer()


__all__ = [
    "analyzer_for",
    "register_analyzer",
    "registered_analyzers",
]
