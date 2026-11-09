"""Seed builder — deterministic input package for the agent.

The engine does no clustering or chunking — it just collects the
relevant filesystem paths and a one-line sibling summary and hands
them to the draft producer. See docs/architecture.md

Phase 1 implements the FWD seed contract for source-boundary and
middle layers + the trivial REV cases. REV code-boundary returns an
empty path list (the agent walks code via ``discover_code_inventory``
in Phase 4); REV middle uses the layer immediately below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from spec_writer.engine.project_context import ProjectContext
from spec_writer.engine.scope import spec_matches_scope
from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.types import Direction, Layer, Mode, Scope, Spec


def _empty_paths() -> list[Path]:
    return []


def _empty_summary() -> dict[str, str]:
    return {}


def _empty_notes() -> list[str]:
    return []


@dataclass(frozen=True)
class Seed:
    """Read-only data package for one layer x direction agent call."""

    layer: Layer
    direction: Direction
    mode: Mode = "generate"
    paths: list[Path] = field(default_factory=_empty_paths)
    existing_corpus_summary: dict[str, str] = field(default_factory=_empty_summary)
    notes: list[str] = field(default_factory=_empty_notes)


def _layer_index(methodology: ResolvedMethodology, layer_id: str) -> int | None:
    for idx, layer in enumerate(methodology.layers):
        if layer.id == layer_id:
            return idx
    return None


def _adjacent_higher(methodology: ResolvedMethodology, layer_id: str) -> Layer | None:
    idx = _layer_index(methodology, layer_id)
    if idx is None or idx == 0:
        return None
    return methodology.layers[idx - 1]


def _adjacent_lower(methodology: ResolvedMethodology, layer_id: str) -> Layer | None:
    idx = _layer_index(methodology, layer_id)
    if idx is None or idx >= len(methodology.layers) - 1:
        return None
    return methodology.layers[idx + 1]


def _md_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.md"))


def build_seed(
    layer: Layer,
    direction: Direction,
    methodology: ResolvedMethodology,
    project_ctx: ProjectContext,
    corpus: dict[str, Spec],
    mode: Mode = "generate",
    instruction: str | None = None,
    scope: Scope | None = None,
) -> Seed:
    """Assemble the seed for the given layer-direction combination."""
    paths: list[Path] = []
    notes: list[str] = []
    if instruction:
        notes.append(f"User instruction: {instruction}")

    if direction == "fwd":
        if layer.role == "source-boundary":
            paths = _md_files(project_ctx.sources)
        else:
            parent = _adjacent_higher(methodology, layer.id)
            if parent is not None:
                paths = _md_files(project_ctx.specs / parent.name)
    else:  # rev
        if layer.role == "code-boundary":
            notes.append("REV code-boundary seed is empty — agent walks code directly.")
        else:
            below = _adjacent_lower(methodology, layer.id)
            if below is not None:
                paths = _md_files(project_ctx.specs / below.name)

    # Existing specs at this layer (refine/cascade reads), optionally
    # narrowed by scope. The narrowing only affects the agent's *visibility*
    # window — the engine still validates / metrics the full corpus.
    in_scope = corpus
    if scope is not None:
        in_scope = {
            sid: spec
            for sid, spec in corpus.items()
            if spec_matches_scope(spec, scope, methodology)
        }
        notes.append(
            f"Scope filter active: {scope.model_dump(exclude_none=True)} "
            f"→ {len(in_scope)} of {len(corpus)} spec(s) visible."
        )

    summary = {spec.id: spec.title for spec in in_scope.values() if spec.layer == layer.id}
    return Seed(
        layer=layer,
        direction=direction,
        mode=mode,
        paths=paths,
        existing_corpus_summary=summary,
        notes=notes,
    )


__all__ = ["Seed", "build_seed"]
