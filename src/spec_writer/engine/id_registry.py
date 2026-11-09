"""IdRegistry — central allocator + collision detector for spec IDs.

ID format: ``L<n>-<PREFIX>-<NNN>``
  - ``L<n>`` — layer indicator (L1, L2, …)
  - ``<PREFIX>`` — type prefix from methodology (VIS, CAP, BHV, …)
  - ``<NNN>`` — zero-padded 3+ digit sequence, gap-filling allocation

The registry is rebuilt from disk at the start of every engine run —
stateless between sessions. See docs/architecture.md and the
the layer-prefixed ID format.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from spec_writer.types import Spec

_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(L\d+)-([A-Z][A-Z0-9]*)-(\d{3,})$")
_FRONTMATTER_ID_LINE: Final[re.Pattern[str]] = re.compile(r"^id:\s*([^\s#]+)\s*$", re.MULTILINE)
_FRONTMATTER_BLOCK: Final[re.Pattern[str]] = re.compile(r"\A---\n(.*?)\n---", re.DOTALL)


class MalformedIdError(ValueError):
    """Raised when an ID does not match the canonical format."""


def parse_id(spec_id: str) -> tuple[str, str, int] | None:
    """Return ``(layer, prefix, number)`` or ``None`` if malformed."""
    m = _ID_PATTERN.match(spec_id)
    if m is None:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def _first_gap(used: set[int]) -> int:
    """Return the smallest positive integer not in ``used``."""
    if not used:
        return 1
    n = 1
    while n in used:
        n += 1
    return n


def _extract_id_from_file(path: Path) -> str | None:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    block = _FRONTMATTER_BLOCK.search(content)
    if block is None:
        return None
    line = _FRONTMATTER_ID_LINE.search(block.group(1))
    if line is None:
        return None
    return line.group(1).strip("'\"")


class IdRegistry:
    """Track allocated IDs per ``(layer, prefix)`` bucket."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str], set[int]] = {}
        self._all_ids: set[str] = set()

    @classmethod
    def from_disk(cls, specs_dir: Path) -> IdRegistry:
        """Scan ``specs_dir`` recursively and register every well-formed ID found."""
        reg = cls()
        if not specs_dir.exists():
            return reg
        for spec_file in sorted(specs_dir.rglob("*.md")):
            spec_id = _extract_id_from_file(spec_file)
            if spec_id is None or parse_id(spec_id) is None:
                continue
            reg.register(spec_id)
        return reg

    @classmethod
    def find_duplicates(cls, specs_dir: Path) -> dict[str, list[Path]]:
        """Return ``{spec_id: [path, …]}`` for IDs that appear in more than one file."""
        seen: dict[str, list[Path]] = {}
        if not specs_dir.exists():
            return {}
        for spec_file in sorted(specs_dir.rglob("*.md")):
            spec_id = _extract_id_from_file(spec_file)
            if spec_id is None or parse_id(spec_id) is None:
                continue
            seen.setdefault(spec_id, []).append(spec_file)
        return {sid: paths for sid, paths in seen.items() if len(paths) > 1}

    def register(self, spec_id: str) -> None:
        """Mark ``spec_id`` as taken. No-op if already registered."""
        parsed = parse_id(spec_id)
        if parsed is None:
            raise MalformedIdError(f"cannot register malformed id {spec_id!r}")
        layer, prefix, n = parsed
        self._buckets.setdefault((layer, prefix), set()).add(n)
        self._all_ids.add(spec_id)

    def allocate(self, layer: str, prefix: str) -> str:
        """Return the next free ID in the ``(layer, prefix)`` bucket."""
        used = self._buckets.setdefault((layer, prefix), set())
        n = _first_gap(used)
        used.add(n)
        spec_id = f"{layer}-{prefix}-{n:03d}"
        self._all_ids.add(spec_id)
        return spec_id

    def is_taken(self, spec_id: str) -> bool:
        return spec_id in self._all_ids

    def release(self, spec_id: str) -> bool:
        """Mark ``spec_id`` as free again (used by ``delete_spec``).

        Returns True if the id was previously taken. Malformed ids
        return False without raising — callers may pass arbitrary
        strings from corpus state.
        """
        parsed = parse_id(spec_id)
        if parsed is None or spec_id not in self._all_ids:
            return False
        layer, prefix, n = parsed
        bucket = self._buckets.get((layer, prefix))
        if bucket is not None:
            bucket.discard(n)
        self._all_ids.discard(spec_id)
        return True

    def fix_duplicates(self, drafts: list[Spec]) -> int:
        """Renumber drafts whose IDs collide with existing registrations.

        Two-pass to keep clean drafts stable:
          1. Register every draft whose ID is currently free (so later
             allocate() calls avoid colliding with them).
          2. Re-allocate the remaining collision-carriers.

        Each draft must have a well-formed ID so the (layer, prefix) bucket
        can be inferred. Returns the count of drafts that were renumbered.
        """
        colliders: list[Spec] = []
        for draft in drafts:
            parsed = parse_id(draft.id)
            if parsed is None:
                raise MalformedIdError(f"draft has malformed id {draft.id!r}")
            if self.is_taken(draft.id):
                colliders.append(draft)
            else:
                self.register(draft.id)

        for draft in colliders:
            parsed = parse_id(draft.id)
            assert parsed is not None  # validated in pass 1
            layer, prefix, _ = parsed
            draft.id = self.allocate(layer, prefix)
        return len(colliders)


__all__ = ["IdRegistry", "MalformedIdError", "parse_id"]
