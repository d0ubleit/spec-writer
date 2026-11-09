"""``query_traceability`` tool — traverse the corpus's trace edges.

Given a spec id and a direction (``up`` / ``down`` / ``both``), walk
``up_to`` / ``down_to`` / ``relates_to`` edges by reading the on-disk
corpus and return the related specs as a list of ``{id, type, layer,
title, kind}`` dicts. ``depth`` controls how far the BFS expands.

The agent calls this when methodology requires it to fix vertical /
horizontal traceability: e.g. "this acceptance criterion references
behavior X — show me the parent capability so I can fix the chain".
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.io.spec_io import find_spec_by_id, read_spec
from spec_writer.types import Spec


class QueryTraceabilityError(RuntimeError):
    pass


def _load_corpus(specs_root: Path) -> dict[str, Spec]:
    corpus: dict[str, Spec] = {}
    if not specs_root.exists():
        return corpus
    for path in sorted(specs_root.rglob("*.md")):
        try:
            spec = read_spec(path)
        except Exception:
            continue
        corpus[spec.id] = spec
    return corpus


def _neighbours(spec: Spec, direction: str) -> list[tuple[str, str]]:
    """Return (neighbour_id, edge_kind) pairs for one node."""
    edges: list[tuple[str, str]] = []
    if direction in ("up", "both"):
        for parent in spec.up_to:
            edges.append((parent, "up_to"))
    if direction in ("down", "both"):
        for child in spec.down_to:
            edges.append((child, "down_to"))
    if direction == "both":
        for entry in spec.relates_to:
            edges.append((entry.id, entry.kind))
    return edges


async def query_traceability(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    spec_id = args.get("spec_id")
    direction = args.get("direction", "both")
    depth_raw = args.get("depth", 1)

    if not isinstance(spec_id, str) or not spec_id:
        raise QueryTraceabilityError("query_traceability requires string 'spec_id'")
    if direction not in ("up", "down", "both"):
        raise QueryTraceabilityError(
            f"direction must be 'up' / 'down' / 'both' (got {direction!r})"
        )
    try:
        depth = int(depth_raw)
    except (TypeError, ValueError) as exc:
        raise QueryTraceabilityError(f"'depth' must be int (got {depth_raw!r})") from exc
    if depth < 1:
        depth = 1

    path = find_spec_by_id(ctx.project_ctx.specs, spec_id)
    if path is None:
        return {"ok": False, "error": f"spec {spec_id!r} not found"}

    corpus = _load_corpus(ctx.project_ctx.specs)
    if spec_id not in corpus:
        return {"ok": False, "error": f"spec {spec_id!r} unreadable"}

    visited: set[str] = {spec_id}
    queue: deque[tuple[str, int]] = deque([(spec_id, 0)])
    results: list[dict[str, Any]] = []

    while queue:
        current_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue
        current_spec = corpus.get(current_id)
        if current_spec is None:
            continue
        for neighbour_id, edge_kind in _neighbours(current_spec, direction):
            if neighbour_id in visited:
                continue
            visited.add(neighbour_id)
            neighbour = corpus.get(neighbour_id)
            if neighbour is None:
                # dangling — record as such; the agent often needs to know
                results.append(
                    {
                        "id": neighbour_id,
                        "kind": edge_kind,
                        "from": current_id,
                        "missing": True,
                    }
                )
                continue
            results.append(
                {
                    "id": neighbour.id,
                    "type": neighbour.type,
                    "layer": neighbour.layer,
                    "title": neighbour.title,
                    "kind": edge_kind,
                    "from": current_id,
                }
            )
            queue.append((neighbour_id, current_depth + 1))

    return {"ok": True, "spec_id": spec_id, "direction": direction, "neighbours": results}


def register(registry: ToolRegistry) -> None:
    registry.register("query_traceability", query_traceability)


__all__ = ["QueryTraceabilityError", "query_traceability", "register"]
