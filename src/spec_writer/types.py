"""Core Pydantic data structures for spec-writer V2.

These types are methodology-agnostic — the engine reads them, the
methodology registry decorates them with type-specific validation.
Methodology-defined frontmatter fields (e.g. `method`, `path` for an
api-contract) ride in :pyattr:`Spec.model_extra`.

See docs:
- docs/methodology.md — frontmatter contract
- docs/architecture.md — Scenario, Scope
- ID format decision
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Literal aliases ──────────────────────────────────────────────────────

Direction = Literal["fwd", "rev"]

Mode = Literal[
    "generate",
    "refine",
    "incremental-add",
    "cascade-fix",
    "targeted-edit-cascade",
    "sync",
]

LayerRole = Literal["source-boundary", "middle", "code-boundary"]

ProvenanceKind = Literal["doc", "code", "inferred"]

RelatesToKind = Literal["relates_to", "depends_on", "used_by"]

SectionFormat = Literal["markdown", "yaml-block", "json-block"]


# ── Value objects (frozen) ───────────────────────────────────────────────


class Provenance(BaseModel):
    """Origin information for a spec.

    `doc` — FWD, source document at `ref`.
    `code` — REV, trace via `implemented_by`; `ref` typically null.
    `inferred` — synth (chat instruction, bubble-up cascade); `ref` opt.
    """

    model_config = ConfigDict(frozen=True)

    kind: ProvenanceKind
    ref: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class RelatesToEntry(BaseModel):
    """Horizontal cross-spec link entry."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: RelatesToKind
    reason: str = ""


class TokenUsage(BaseModel):
    """Per-call or aggregated LLM token usage.

    ``input`` + ``output`` are the SDK's basic ``input_tokens`` and
    ``output_tokens``. ``cache_creation_input`` and ``cache_read_input``
    measure prompt-caching effectiveness (see the design rationale):
    cache_creation is the one-time write at ~1.25x normal cost; each
    subsequent turn's cache_read replaces equivalent input at ~0.1x.
    A high cache_read / (cache_read + input) ratio means caching is
    earning its keep.
    """

    model_config = ConfigDict(frozen=True)

    input: int = 0
    output: int = 0
    total: int = 0
    cache_creation_input: int = 0
    cache_read_input: int = 0


class FrontmatterField(BaseModel):
    """Methodology-declared frontmatter field metadata.

    Both `required_frontmatter` and `optional_frontmatter` entries
    in a mixin resolve to this shape.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    llm_guidance: str = ""
    validation: dict[str, Any] | None = None
    surface_when_missing: bool = False
    """Optional-fields only: when true, validators emit a WARNING for
    every spec lacking this field. Use for fields that are
    "near-required" in the project's culture (priority, owner,
    confidence) — so quiet skips don't accumulate as hidden debt.
    See docs/methodology.md."""


class Section(BaseModel):
    """Methodology-declared section inside a spec body."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    required: bool = False
    format: SectionFormat = "markdown"
    position: str = "end"
    guidance: str = ""
    schema_ref: str | None = None


# ── Aggregate / mutable models ───────────────────────────────────────────


class Layer(BaseModel):
    """Pyramid layer descriptor as loaded from the resolved methodology."""

    id: str
    role: LayerRole
    name: str
    purpose: str = ""
    is_final: bool = False
    guidance: dict[str, str] = Field(default_factory=dict)


def _empty_sections() -> list[Section]:
    return []


def _empty_fields() -> list[FrontmatterField]:
    return []


def _empty_relates_to() -> list[RelatesToEntry]:
    return []


class SpecType(BaseModel):
    """Methodology-defined spec type bound to a single layer."""

    id: str
    layer_id: str
    prefix: str
    sections: list[Section] = Field(default_factory=_empty_sections)
    required_frontmatter: list[FrontmatterField] = Field(default_factory=_empty_fields)
    optional_frontmatter: list[FrontmatterField] = Field(default_factory=_empty_fields)
    file_pattern: str | None = None
    up_to_types: list[str] = Field(default_factory=list)
    cross_cutting: bool = False
    """When true, specs of this type are exempt from pyramid-adjacency.
    Their ``up_to`` may reference any upper layer, not just the
    immediate one. Use for cross-cutting types (conventions, archviews)
    whose semantics genuinely apply to many lower-layer specs
    uniformly. See docs/methodology.md"""


class Spec(BaseModel):
    """Single specification document.

    Engine-mandatory frontmatter fields are explicit attributes.
    Methodology-defined extras (e.g. `method`/`path` for api-contract)
    fall into `model_extra` via ``extra="allow"`` so the type itself
    remains methodology-agnostic.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    layer: str
    title: str
    up_to: list[str] = Field(default_factory=list)
    down_to: list[str] = Field(default_factory=list)
    implemented_by: list[str] = Field(default_factory=list)
    status: str | None = None
    tags: list[str] = Field(default_factory=list)
    provenance: Provenance | None = None
    relates_to: list[RelatesToEntry] = Field(default_factory=_empty_relates_to)
    body: str = ""


# ── Scenario / Scope ─────────────────────────────────────────────────────


class Scope(BaseModel):
    """Declarative filter applied to the corpus.

    All filters combine with AND semantics; `None` means "no constraint
    on this dimension".
    """

    layers: list[str] | None = None
    types: list[str] | None = None
    tags: list[str] | None = None
    folder_pattern: str | None = None
    spec_ids: list[str] | None = None
    from_finding: list[str] | None = None


class Scenario(BaseModel):
    """Single engine-run configuration; one Scenario → one loop."""

    direction: Direction
    layers_in_scope: list[str]
    mode: Mode
    scope: Scope | None = None
    max_iterations: int = Field(default=5, ge=1)
    target_metrics: dict[str, float] = Field(default_factory=dict)
    auto_apply: bool = False


__all__ = [
    "Direction",
    "FrontmatterField",
    "Layer",
    "LayerRole",
    "Mode",
    "Provenance",
    "ProvenanceKind",
    "RelatesToEntry",
    "RelatesToKind",
    "Scenario",
    "Scope",
    "Section",
    "SectionFormat",
    "Spec",
    "SpecType",
    "TokenUsage",
]
