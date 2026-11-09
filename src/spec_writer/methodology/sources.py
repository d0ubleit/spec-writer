"""MixinSource protocol — abstracts discovery of mixin YAML files.

Phase 1 ships a single implementation, ``FilesystemMixinSource``, which
globs ``*.yaml`` directly under a given root. Mixins with Python
sidecars (directory + ``<name>.yaml`` + ``<name>.py``) discover the
yaml entry the same way — the sidecar is loaded lazily by the
validator runner (S1.8) and not by the source.

``EntryPointMixinSource`` for pip-installed plugins is a known
limitation (not implemented yet).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class MixinSource(Protocol):
    def discover(self) -> list[Path]: ...


class FilesystemMixinSource:
    """Discover ``*.yaml`` files under ``root``, recursing one level for
    directory-style mixins (``<root>/<name>/<name>.yaml``).
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def discover(self) -> list[Path]:
        if not self._root.exists():
            return []
        results: list[Path] = []
        for entry in sorted(self._root.iterdir()):
            if entry.is_file() and entry.suffix == ".yaml":
                results.append(entry)
            elif entry.is_dir():
                candidate = entry / f"{entry.name}.yaml"
                if candidate.exists():
                    results.append(candidate)
        return results


__all__ = ["FilesystemMixinSource", "MixinSource"]
