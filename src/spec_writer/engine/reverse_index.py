"""Reverse index — code file → list of specs that reference it.

Built **on-the-fly** by walking the corpus and reading each spec's
``implemented_by`` field. No persistence — the corpus IS the source
of truth. Used by sync flows: when the agent learns a code file
changed, the reverse index tells it which specs need a refine pass.

Phase-4-minimum: linear scan; corpus sizes are O(thousands) at most.
If profiling shows this is a bottleneck, we cache per-mtime.
"""

from __future__ import annotations

from pathlib import Path

from spec_writer.io.spec_io import read_spec


def build(specs_root: Path) -> dict[str, list[str]]:
    """Return ``{code_file_path: [spec_id, ...]}`` for every spec with
    a non-empty ``implemented_by``.

    Path normalisation: drops trailing ``:start-end`` line ranges so a
    single file maps to all specs that touch any range in it. Same
    POSIX-style normalisation as :mod:`engine.coverage_ledger`.
    """
    index: dict[str, list[str]] = {}
    if not specs_root.exists():
        return index

    for path in sorted(specs_root.rglob("*.md")):
        try:
            spec = read_spec(path)
        except Exception:
            continue
        for ref in spec.implemented_by:
            file_part = ref.split(":", 1)[0].replace("\\", "/").strip()
            if file_part.startswith("./"):
                file_part = file_part[2:]
            if not file_part:
                continue
            index.setdefault(file_part, []).append(spec.id)
    # Deduplicate spec ids per file while preserving first-seen order.
    return {file: list(dict.fromkeys(ids)) for file, ids in index.items()}


__all__ = ["build"]
