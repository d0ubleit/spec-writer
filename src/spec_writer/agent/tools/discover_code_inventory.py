"""``discover_code_inventory`` MCP tool — REV-mode code surface listing.

Walks the project tree under ``code_scope.includes`` (default
``["src/**", "app/**", "lib/**"]``), excludes ``code_scope.excludes``
(default vendor / node_modules / build artefacts / spec-editor), and
hands each file to the matching :class:`LanguageAnalyzer`. Returns a
per-file inventory of symbols.

**Best-effort, not source-of-truth.** Analyzers may miss things; the
agent uses this as a hint and reads code directly when it needs
authoritative answers.

Args (dict, all optional):
- ``includes`` — glob patterns (relative to project_root) to walk.
- ``excludes`` — glob patterns to exclude.
- ``max_files`` — cap; default 500. Beyond this, agent should narrow
  ``includes``/``excludes`` and re-call.
- ``visibility`` — ``"public"`` (default) / ``"all"``. ``public``
  filters symbols to ``visibility in {public, protected}``.

Returns: ``{"ok": True, "files": [...], "truncated": bool,
"summary": {...}}``.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, cast

from spec_writer.agent.code_analyzers import Symbol, analyzer_for
from spec_writer.agent.tool_registry import ToolContext, ToolRegistry

_DEFAULT_INCLUDES = ["src/**", "app/**", "lib/**"]
_DEFAULT_EXCLUDES = [
    "**/node_modules/**",
    "**/.git/**",
    "**/.venv/**",
    "**/vendor/**",
    "**/dist/**",
    "**/build/**",
    "**/__pycache__/**",
    "**/.next/**",
    "**/.svelte-kit/**",
    "spec-editor/**",
]
_MAX_FILES_DEFAULT = 500


class DiscoverCodeInventoryError(RuntimeError):
    pass


def _matches_any(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


def _walk(
    root: Path, includes: list[str], excludes: list[str], max_files: int
) -> tuple[list[Path], bool]:
    out: list[Path] = []
    if not root.exists():
        return out, False
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel_str = str(path.relative_to(root))
        except ValueError:
            continue
        # POSIX-style for matching
        rel_posix = rel_str.replace("\\", "/")
        if not _matches_any(rel_posix, includes):
            continue
        if _matches_any(rel_posix, excludes):
            continue
        out.append(path)
        if len(out) >= max_files:
            return out, True
    return out, False


def _filter_symbols(symbols: list[Symbol], visibility: str) -> list[Symbol]:
    if visibility == "all":
        return symbols
    keep = {"public", "protected"}
    return [s for s in symbols if s.visibility in keep]


async def discover_code_inventory(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    includes_raw = args.get("includes")
    excludes_raw = args.get("excludes")
    if includes_raw is None:
        includes = list(_DEFAULT_INCLUDES)
    elif isinstance(includes_raw, list):
        includes = [p for p in cast(list[Any], includes_raw) if isinstance(p, str)]
    else:
        raise DiscoverCodeInventoryError("'includes' must be a list of glob strings")
    if excludes_raw is None:
        excludes = list(_DEFAULT_EXCLUDES)
    elif isinstance(excludes_raw, list):
        excludes = [p for p in cast(list[Any], excludes_raw) if isinstance(p, str)]
    else:
        raise DiscoverCodeInventoryError("'excludes' must be a list of glob strings")

    max_files_raw = args.get("max_files", _MAX_FILES_DEFAULT)
    try:
        max_files = max(1, int(max_files_raw))
    except (TypeError, ValueError) as exc:
        raise DiscoverCodeInventoryError("'max_files' must be int") from exc

    visibility = str(args.get("visibility", "public")).lower()
    if visibility not in ("public", "all"):
        raise DiscoverCodeInventoryError("'visibility' must be 'public' or 'all'")

    root = ctx.project_ctx.root
    files, truncated = _walk(root, includes, excludes, max_files)

    inventory: list[dict[str, Any]] = []
    languages_seen: dict[str, int] = {}
    total_symbols = 0

    for path in files:
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            source = ""
        analyzer = analyzer_for(path)
        symbols = _filter_symbols(analyzer.discover_symbols(path, source), visibility)
        if not symbols and visibility == "public":
            # Filesystem fallback emits a public "file"-kind symbol; still useful.
            continue
        rel = path.relative_to(root)
        inventory.append(
            {
                "path": str(rel),
                "language": analyzer.language,
                "symbols": [
                    {
                        "name": s.name,
                        "kind": s.kind,
                        "visibility": s.visibility,
                        "start_line": s.start_line,
                        "end_line": s.end_line,
                        "signature": s.signature,
                    }
                    for s in symbols
                ],
            }
        )
        languages_seen[analyzer.language] = languages_seen.get(analyzer.language, 0) + 1
        total_symbols += len(symbols)

    return {
        "ok": True,
        "files": inventory,
        "truncated": truncated,
        "summary": {
            "files_scanned": len(files),
            "files_with_symbols": len(inventory),
            "languages": languages_seen,
            "total_symbols": total_symbols,
            "visibility_filter": visibility,
            "includes": includes,
            "excludes": excludes,
        },
    }


def register(registry: ToolRegistry) -> None:
    registry.register("discover_code_inventory", discover_code_inventory)


__all__ = ["DiscoverCodeInventoryError", "discover_code_inventory", "register"]
