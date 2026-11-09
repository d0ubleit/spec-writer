"""Built-in metrics — registered at import time.

These cover the basic quality gates every methodology benefits from:

- ``metrics.spec_count_at_layer`` — gauge of corpus density.
- ``metrics.up_to_completeness_ratio`` — fraction of non-source specs
  with a non-empty ``up_to``. Should be 1.0 in a converged corpus.
- ``metrics.down_to_completeness_ratio`` — fraction of non-leaf-layer
  specs with at least one ``down_to`` child.
- ``metrics.required_frontmatter_ratio`` — fraction of specs that
  populate every required-frontmatter field declared by their type.
"""

from __future__ import annotations

from typing import cast

from spec_writer.methodology.metrics import (
    MetricContext,
    MetricValue,
    register_metric,
)
from spec_writer.types import Spec


def _specs_at_layer(specs: list[Spec], layer_id: str) -> list[Spec]:
    return [s for s in specs if s.layer == layer_id]


@register_metric("metrics.spec_count_at_layer")
def _spec_count(specs: list[Spec], ctx: MetricContext) -> MetricValue:
    n = len(_specs_at_layer(specs, ctx.layer_id))
    return MetricValue(value=float(n), kind="count", details={"layer": ctx.layer_id, "n": n})


def _is_source_boundary(ctx: MetricContext, layer_id: str) -> bool:
    for layer in ctx.methodology.layers:
        if layer.id == layer_id:
            return layer.role == "source-boundary"
    return False


def _is_code_boundary(ctx: MetricContext, layer_id: str) -> bool:
    for layer in ctx.methodology.layers:
        if layer.id == layer_id:
            return layer.role == "code-boundary"
    return False


@register_metric("metrics.up_to_completeness_ratio")
def _up_to_completeness(specs: list[Spec], ctx: MetricContext) -> MetricValue:
    """Fraction of layer specs (excluding source-boundary) with non-empty up_to."""
    if _is_source_boundary(ctx, ctx.layer_id):
        # vacuously true — source-boundary specs are not required to have up_to
        return MetricValue(
            value=1.0, kind="ratio", details={"layer": ctx.layer_id, "note": "source-boundary"}
        )
    layer_specs = _specs_at_layer(specs, ctx.layer_id)
    if not layer_specs:
        return MetricValue(value=0.0, kind="ratio", details={"layer": ctx.layer_id, "n": 0})
    with_up = sum(1 for s in layer_specs if s.up_to)
    return MetricValue(
        value=with_up / len(layer_specs),
        kind="ratio",
        details={"with_up_to": with_up, "total": len(layer_specs)},
    )


@register_metric("metrics.down_to_completeness_ratio")
def _down_to_completeness(specs: list[Spec], ctx: MetricContext) -> MetricValue:
    """Fraction of layer specs (excluding code-boundary) with non-empty down_to."""
    if _is_code_boundary(ctx, ctx.layer_id):
        return MetricValue(
            value=1.0, kind="ratio", details={"layer": ctx.layer_id, "note": "code-boundary"}
        )
    layer_specs = _specs_at_layer(specs, ctx.layer_id)
    if not layer_specs:
        return MetricValue(value=0.0, kind="ratio", details={"layer": ctx.layer_id, "n": 0})
    with_down = sum(1 for s in layer_specs if s.down_to)
    return MetricValue(
        value=with_down / len(layer_specs),
        kind="ratio",
        details={"with_down_to": with_down, "total": len(layer_specs)},
    )


@register_metric("metrics.required_frontmatter_ratio")
def _required_fm_ratio(specs: list[Spec], ctx: MetricContext) -> MetricValue:
    """Fraction of layer specs that have populated every required-fm field declared by their type."""
    layer_specs = _specs_at_layer(specs, ctx.layer_id)
    if not layer_specs:
        return MetricValue(value=0.0, kind="ratio", details={"layer": ctx.layer_id, "n": 0})

    complete = 0
    for spec in layer_specs:
        spec_type = ctx.methodology.spec_types.get(spec.type)
        if spec_type is None or not spec_type.required_frontmatter:
            complete += 1
            continue
        extras = cast(dict[str, object], spec.model_extra or {})
        engine_fields = {
            "id",
            "type",
            "layer",
            "title",
            "up_to",
            "down_to",
            "implemented_by",
            "status",
            "tags",
            "provenance",
            "relates_to",
        }
        missing = [
            f.name
            for f in spec_type.required_frontmatter
            if (f.name in engine_fields and not getattr(spec, f.name, None))
            or (f.name not in engine_fields and extras.get(f.name) in (None, "", [], {}))
        ]
        if not missing:
            complete += 1
    return MetricValue(
        value=complete / len(layer_specs),
        kind="ratio",
        details={"complete": complete, "total": len(layer_specs)},
    )


__all__ = [
    "_down_to_completeness",
    "_required_fm_ratio",
    "_spec_count",
    "_up_to_completeness",
]
