"""Bidirectional cross-spec edge symmetrization.

Per docs/methodology.md and principle 2: within-corpus references are
symmetric — ``A.up_to=[B]`` implies ``B.down_to=[A]``. The engine
keeps both edges in sync via two complementary functions:

- :func:`symmetrize_refs` — in-memory batch pass; mutates a full
  corpus list (used after producer batches, before validation).
- :func:`symmetrise_neighbors_on_disk` — surgical disk-bound pass;
  reads each affected neighbor, repairs reciprocal edges, writes back.
  Used by ``record_spec`` / ``patch_spec`` on every single-spec write
  so individual operations stay self-symmetric.

Outbound references to specs not present in the corpus are left
untouched — :mod:`.validators` reports those as dangling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spec_writer.io.spec_io import find_spec_by_id, read_spec, write_spec
from spec_writer.types import RelatesToEntry, RelatesToKind, Spec

if TYPE_CHECKING:  # pragma: no cover
    from spec_writer.agent.tool_registry import ToolContext


def _append_unique(target: list[str], value: str) -> None:
    if value not in target:
        target.append(value)


def _append_unique_relation(target: list[RelatesToEntry], entry: RelatesToEntry) -> None:
    if not any(existing.id == entry.id and existing.kind == entry.kind for existing in target):
        target.append(entry)


_PAIRED_KINDS: dict[RelatesToKind, RelatesToKind] = {
    "relates_to": "relates_to",
    "depends_on": "used_by",
    "used_by": "depends_on",
}


def symmetrize_refs(specs: list[Spec]) -> None:
    """Mutate ``specs`` so vertical and horizontal pairs are symmetric.

    Vertical: ``up_to`` ↔ ``down_to``.
    Horizontal (``relates_to`` entries): ``relates_to`` symmetric;
    ``depends_on`` ↔ ``used_by``.
    """
    by_id = {spec.id: spec for spec in specs}

    for spec in specs:
        for parent_id in spec.up_to:
            parent = by_id.get(parent_id)
            if parent is not None:
                _append_unique(parent.down_to, spec.id)
        for child_id in spec.down_to:
            child = by_id.get(child_id)
            if child is not None:
                _append_unique(child.up_to, spec.id)
        for entry in spec.relates_to:
            other = by_id.get(entry.id)
            if other is None:
                continue
            reverse_kind = _PAIRED_KINDS.get(entry.kind)
            if reverse_kind is None:
                continue
            _append_unique_relation(
                other.relates_to,
                RelatesToEntry(id=spec.id, kind=reverse_kind, reason=entry.reason),
            )


def symmetrise_neighbors_on_disk(
    spec: Spec,
    ctx: ToolContext,
    *,
    prior_up_to: list[str] | None = None,
    prior_down_to: list[str] | None = None,
    prior_relates_to: list[RelatesToEntry] | None = None,
) -> None:
    """Repair reciprocal edges on every neighbor referenced by ``spec``.

    Two passes (two repair passes):

    1. **Removal** — when priors are provided (i.e. called from
       ``patch_spec``), find edges present in priors but missing from
       current. For each such removed edge, find the previously-
       referenced neighbor on disk and strip the now-stale reciprocal
       edge. No-op if priors are ``None`` (i.e. called from
       ``record_spec`` on a brand-new spec).
    2. **Addition** — add reciprocal edges to current neighbors;
       idempotent (skips edges already present).
    """
    if prior_up_to is not None:
        removed_parents = [pid for pid in prior_up_to if pid not in spec.up_to]
        for parent_id in removed_parents:
            _strip_edge(ctx, parent_id, "down_to", spec.id)
    if prior_down_to is not None:
        removed_children = [cid for cid in prior_down_to if cid not in spec.down_to]
        for child_id in removed_children:
            _strip_edge(ctx, child_id, "up_to", spec.id)
    if prior_relates_to is not None:
        new_keys = {(e.id, e.kind) for e in spec.relates_to}
        for entry in prior_relates_to:
            if (entry.id, entry.kind) in new_keys:
                continue
            reverse_kind = _PAIRED_KINDS.get(entry.kind)
            if reverse_kind is None:
                continue
            _strip_relation(ctx, entry.id, spec.id, reverse_kind)

    # Addition pass — same shape as the in-memory symmetrize_refs but
    # neighbor-scoped via disk reads.
    for parent_id in spec.up_to:
        _ensure_edge(ctx, parent_id, "down_to", spec.id)
    for child_id in spec.down_to:
        _ensure_edge(ctx, child_id, "up_to", spec.id)
    for entry in spec.relates_to:
        reverse_kind = _PAIRED_KINDS.get(entry.kind)
        if reverse_kind is None:
            continue
        _ensure_relation(
            ctx,
            entry.id,
            RelatesToEntry(
                id=spec.id,
                kind=reverse_kind,
                reason=entry.reason,
            ),
        )


def _ensure_edge(ctx: ToolContext, neighbor_id: str, field: str, self_id: str) -> None:
    neighbor_path = find_spec_by_id(ctx.project_ctx.specs, neighbor_id)
    if neighbor_path is None:
        return  # dangling — validator surfaces this separately
    neighbor = read_spec(neighbor_path)
    edge_list: list[str] = getattr(neighbor, field)
    if self_id in edge_list:
        return
    edge_list.append(self_id)
    write_spec(neighbor, ctx.methodology, ctx.project_ctx)


def _strip_edge(ctx: ToolContext, neighbor_id: str, field: str, self_id: str) -> None:
    neighbor_path = find_spec_by_id(ctx.project_ctx.specs, neighbor_id)
    if neighbor_path is None:
        return
    neighbor = read_spec(neighbor_path)
    edge_list: list[str] = getattr(neighbor, field)
    if self_id not in edge_list:
        return
    edge_list.remove(self_id)
    write_spec(neighbor, ctx.methodology, ctx.project_ctx)


def _ensure_relation(ctx: ToolContext, neighbor_id: str, entry: RelatesToEntry) -> None:
    neighbor_path = find_spec_by_id(ctx.project_ctx.specs, neighbor_id)
    if neighbor_path is None:
        return
    neighbor = read_spec(neighbor_path)
    for existing in neighbor.relates_to:
        if existing.id == entry.id and existing.kind == entry.kind:
            return
    neighbor.relates_to.append(entry)
    write_spec(neighbor, ctx.methodology, ctx.project_ctx)


def _strip_relation(ctx: ToolContext, neighbor_id: str, self_id: str, reverse_kind: str) -> None:
    neighbor_path = find_spec_by_id(ctx.project_ctx.specs, neighbor_id)
    if neighbor_path is None:
        return
    neighbor = read_spec(neighbor_path)
    before = len(neighbor.relates_to)
    neighbor.relates_to[:] = [
        e for e in neighbor.relates_to if not (e.id == self_id and e.kind == reverse_kind)
    ]
    if len(neighbor.relates_to) != before:
        write_spec(neighbor, ctx.methodology, ctx.project_ctx)


__all__ = ["symmetrise_neighbors_on_disk", "symmetrize_refs"]
