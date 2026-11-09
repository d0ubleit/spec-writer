"""Cascade helpers — compute the blast radius of a spec edit.

When the user edits one spec, the engine often needs to revisit
neighbouring specs (parents, children, horizontal relations) to keep
the corpus consistent. This module computes the set of related ids
without doing any patching — that's the agent's job in cascade-fix
or targeted-edit-cascade mode.

The BFS walks `up_to` / `down_to` / `relates_to.id` edges. It stops at
``depth`` and never visits the same spec twice. Missing edges (ids
that don't resolve in the corpus) are recorded so the agent knows the
chain has a hole.
"""

from __future__ import annotations

from collections import deque

from spec_writer.types import Spec


def cascade_neighbours(
    spec_id: str,
    corpus: dict[str, Spec],
    depth: int = 2,
) -> list[str]:
    """Return all ids reachable from ``spec_id`` within ``depth`` edges (inclusive).

    Always includes ``spec_id`` itself even when missing from the corpus.
    Order is BFS so callers can show "directly adjacent first".
    """
    if depth < 1:
        depth = 1
    visited: set[str] = {spec_id}
    ordered: list[str] = [spec_id]
    queue: deque[tuple[str, int]] = deque([(spec_id, 0)])

    while queue:
        current_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue
        current = corpus.get(current_id)
        if current is None:
            continue
        for neighbour_id in _adjacent_ids(current):
            if neighbour_id in visited:
                continue
            visited.add(neighbour_id)
            ordered.append(neighbour_id)
            queue.append((neighbour_id, current_depth + 1))
    return ordered


def _adjacent_ids(spec: Spec) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for sid in spec.up_to:
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    for sid in spec.down_to:
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    for entry in spec.relates_to:
        if entry.id not in seen:
            seen.add(entry.id)
            out.append(entry.id)
    return out


__all__ = ["cascade_neighbours"]
