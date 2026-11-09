"""On-disk spec I/O — canonical write/read backed by atomic_write.

``write_spec`` computes the canonical file path from the spec's id +
title (slug + type prefix + 3-digit number), packages the frontmatter
in the methodology-declared field order, and writes atomically. The
inverse, ``read_spec``, parses a markdown file back into a ``Spec``.

Layer directory comes from ``methodology.layers[].name``. Mixin-
specific anchor subdirs (e.g. ``api/`` for L4 api-contracts) come from
the spec type's ``file_pattern`` if present.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from spec_writer.engine.atomic_io import atomic_write
from spec_writer.engine.id_registry import parse_id
from spec_writer.engine.project_context import ProjectContext
from spec_writer.io.frontmatter import parse_frontmatter, write_canonical
from spec_writer.io.slug import make_slug
from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.types import Spec


class SpecIoError(RuntimeError):
    pass


def _layer_dir(methodology: ResolvedMethodology, layer_id: str) -> str:
    for layer in methodology.layers:
        if layer.id == layer_id:
            return layer.name
    raise SpecIoError(f"unknown layer id {layer_id!r}")


_ANCHOR_RE = re.compile(r"^([^/{]+)/")


def _anchor_subdir(file_pattern: str | None) -> str | None:
    if not file_pattern:
        return None
    m = _ANCHOR_RE.match(file_pattern)
    if m is None:
        return None
    return m.group(1)


def compute_spec_path(
    spec: Spec, methodology: ResolvedMethodology, project_ctx: ProjectContext
) -> Path:
    """Resolve the canonical on-disk path for ``spec``."""
    layer_dir = _layer_dir(methodology, spec.layer)
    parsed = parse_id(spec.id)
    if parsed is None:
        raise SpecIoError(f"spec id {spec.id!r} is malformed")
    _, prefix, number = parsed

    spec_type = methodology.spec_types.get(spec.type)
    anchor = _anchor_subdir(spec_type.file_pattern) if spec_type else None

    slug = make_slug(spec.title)
    filename = f"{slug}-{prefix}-{number:03d}.md"
    base = project_ctx.specs / layer_dir
    if anchor:
        base = base / anchor
    return base / filename


def _frontmatter_dict(spec: Spec) -> dict[str, Any]:
    payload: dict[str, Any] = spec.model_dump(mode="python", exclude={"body"})
    if spec.model_extra:
        payload.update(spec.model_extra)
    # Pydantic dumps optional None fields; drop them unless they have meaning.
    for drop_if_none in ("provenance", "status"):
        if payload.get(drop_if_none) is None:
            payload.pop(drop_if_none, None)
    return payload


def write_spec(spec: Spec, methodology: ResolvedMethodology, project_ctx: ProjectContext) -> Path:
    """Write ``spec`` atomically and return the final path."""
    path = compute_spec_path(spec, methodology, project_ctx)

    spec_type = methodology.spec_types.get(spec.type)
    methodology_order: list[str] = []
    if spec_type is not None:
        methodology_order.extend(f.name for f in spec_type.required_frontmatter)
        methodology_order.extend(
            f.name for f in spec_type.optional_frontmatter if f.name not in methodology_order
        )

    fm = _frontmatter_dict(spec)
    rendered = write_canonical(fm, body=spec.body, methodology_required_order=methodology_order)
    atomic_write(path, rendered)
    return path


def read_spec(path: Path) -> Spec:
    """Parse a single markdown file into a ``Spec``."""
    content = path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(content)
    return Spec(**fm, body=body)


def find_spec_by_id(specs_root: Path, spec_id: str) -> Path | None:
    """Scan ``specs_root`` recursively for the file containing ``spec_id``."""
    if not specs_root.exists():
        return None
    for path in specs_root.rglob("*.md"):
        try:
            spec = read_spec(path)
        except Exception:
            continue
        if spec.id == spec_id:
            return path
    return None


__all__ = ["SpecIoError", "compute_spec_path", "find_spec_by_id", "read_spec", "write_spec"]
