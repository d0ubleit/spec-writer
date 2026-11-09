"""ProjectContext — single-project root resolution + path safety.

Resolution order:
  1. Explicit ``--project-root`` (handled in CLI layer, not here).
  2. Walk up from given/current working directory until finding a
     ``spec-editor/`` subdirectory. The owning directory is the root.
  3. If none found → ``NoProjectRootError``.

After resolution, the context object is immutable. All file operations
performed by tools route through ``safe_path()`` which confines paths
to the project root (after symlink resolution).

See docs/architecture.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_SPEC_EDITOR_DIRNAME = "spec-editor"


class NoProjectRootError(RuntimeError):
    """Raised when no ``spec-editor/`` directory is found while walking up."""


class PathTraversalError(RuntimeError):
    """Raised when a path resolves outside the project root."""


def _walk_up_for_root(start: Path) -> Path | None:
    cursor = start.resolve()
    if not cursor.is_dir():
        cursor = cursor.parent
    while True:
        if (cursor / _SPEC_EDITOR_DIRNAME).is_dir():
            return cursor
        if cursor.parent == cursor:
            return None
        cursor = cursor.parent


@dataclass(frozen=True)
class ProjectContext:
    """Root + canonical subpaths of a single spec-writer project."""

    root: Path
    spec_editor: Path
    sources_raw: Path
    sources: Path
    specs: Path
    meta: Path
    methodology_yaml: Path

    @classmethod
    def from_cwd(cls, cwd: Path | None = None) -> ProjectContext:
        start = (cwd if cwd is not None else Path.cwd()).resolve()
        root = _walk_up_for_root(start)
        if root is None:
            raise NoProjectRootError(f"no {_SPEC_EDITOR_DIRNAME}/ found in {start} or any ancestor")
        spec_editor = root / _SPEC_EDITOR_DIRNAME
        return cls(
            root=root,
            spec_editor=spec_editor,
            sources_raw=spec_editor / "sources-raw",
            sources=spec_editor / "sources",
            specs=spec_editor / "specs",
            meta=spec_editor / "_meta",
            methodology_yaml=spec_editor / "methodology.yaml",
        )

    def safe_path(self, user_path: str | Path) -> Path:
        """Resolve ``user_path`` relative to the project root and verify it stays inside.

        - Absolute paths must point inside the root.
        - ``..`` segments that climb above the root → ``PathTraversalError``.
        - Symlinks are followed (``.resolve()``) before the containment check.
        """
        raw = Path(user_path)
        candidate = raw if raw.is_absolute() else (self.root / raw)
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise PathTraversalError(
                f"path escapes project root: {user_path!r} -> {resolved}"
            ) from exc
        return resolved
