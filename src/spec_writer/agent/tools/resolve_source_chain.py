"""``resolve_source_chain`` tool — walk ``up_to`` to the source-boundary.

Given a spec id, walk ``up_to`` recursively and return every ancestor
spec up to (and including) the layer marked ``source-boundary`` in the
methodology. The agent uses this when a middle-layer spec needs to be
re-evaluated against the original source documents — typically during
``refine`` and ``cascade-fix`` flows.

Returns ``{"ok": True, "chain": [...specs from leaf to root], "sources": [...]}``.
``sources`` lists the unique source-boundary ancestors at the top of
each path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.io.spec_io import find_spec_by_id, read_spec
from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.types import Spec


class ResolveSourceChainError(RuntimeError):
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


def _is_source_boundary(methodology: ResolvedMethodology, layer_id: str) -> bool:
    for layer in methodology.layers:
        if layer.id == layer_id:
            return layer.role == "source-boundary"
    return False


def _spec_dict(spec: Spec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "type": spec.type,
        "layer": spec.layer,
        "title": spec.title,
        "up_to": list(spec.up_to),
    }


async def resolve_source_chain(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    spec_id = args.get("spec_id")
    max_depth_raw = args.get("max_depth", 8)
    if not isinstance(spec_id, str) or not spec_id:
        raise ResolveSourceChainError("resolve_source_chain requires string 'spec_id'")
    try:
        max_depth = int(max_depth_raw)
    except (TypeError, ValueError) as exc:
        raise ResolveSourceChainError(f"'max_depth' must be int (got {max_depth_raw!r})") from exc
    if max_depth < 1:
        max_depth = 1

    if find_spec_by_id(ctx.project_ctx.specs, spec_id) is None:
        return {"ok": False, "error": f"spec {spec_id!r} not found"}

    corpus = _load_corpus(ctx.project_ctx.specs)
    start = corpus.get(spec_id)
    if start is None:
        return {"ok": False, "error": f"spec {spec_id!r} unreadable"}

    chain: list[dict[str, Any]] = [_spec_dict(start)]
    sources: list[dict[str, Any]] = []
    visited: set[str] = {spec_id}
    frontier: list[Spec] = [start]
    truncated = False

    for _depth in range(max_depth):
        next_frontier: list[Spec] = []
        for current in frontier:
            if _is_source_boundary(ctx.methodology, current.layer):
                if current.id not in {s["id"] for s in sources}:
                    sources.append(_spec_dict(current))
                continue
            for parent_id in current.up_to:
                if parent_id in visited:
                    continue
                visited.add(parent_id)
                parent = corpus.get(parent_id)
                if parent is None:
                    chain.append({"id": parent_id, "missing": True})
                    continue
                chain.append(_spec_dict(parent))
                if _is_source_boundary(ctx.methodology, parent.layer):
                    sources.append(_spec_dict(parent))
                else:
                    next_frontier.append(parent)
        if not next_frontier:
            break
        frontier = next_frontier
    else:
        truncated = True

    return {
        "ok": True,
        "spec_id": spec_id,
        "chain": chain,
        "sources": sources,
        "truncated": truncated,
    }


def register(registry: ToolRegistry) -> None:
    registry.register("resolve_source_chain", resolve_source_chain)


__all__ = ["ResolveSourceChainError", "register", "resolve_source_chain"]
