"""Metric framework — decorator-based registry mirroring validators.

A metric is a pure function that consumes the corpus + per-config
parameters and returns a numeric :class:`MetricValue`. The framework
also tracks ``target`` / ``warn_below`` / ``critical`` thresholds so
the engine can:

- Block convergence on any critical metric below target.
- Surface warnings without blocking.
- Inform the metric-aware ``progressing()`` check in the engine loop.

Metric config shape inside a mixin YAML:

```yaml
metrics:
  - id: up_to_completeness
    layer: L2
    func_ref: metrics.up_to_completeness_ratio
    target: 1.0
    warn_below: 0.8
    critical: false        # true → blocks convergence
    params:
      target_role: middle  # arbitrary; metric impl interprets
```
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.types import Spec

MetricKind = Literal["count", "ratio", "score"]


class MetricValue(BaseModel):
    """Raw output of a metric function — value + optional debug fields."""

    model_config = ConfigDict(frozen=True)

    value: float
    kind: MetricKind = "score"
    """Hint for downstream rendering only."""
    details: dict[str, Any] | None = None
    applicable: bool = True
    """When false, the metric returned a value (typically 0.0) but the
    denominator-source is structurally empty — e.g. a coverage metric
    that depends on L4 specs that haven't been generated yet. The
    runner respects this by setting ``MetricResult.target_met = None``
    and ``warning = False``, preserving the value as a planning signal
    without raising false alarms. See the design rationale."""


class MetricResult(BaseModel):
    """One metric evaluated against the corpus, with verdict.

    ``value`` is :data:`None` when the metric raised or had insufficient
    data (e.g. divide-by-zero on an empty denominator). ``target_met``
    is :data:`None` in that case too — the engine treats it as "not
    blocking" unless ``critical`` is set, in which case absence equals
    failure.

    ``applicable=False`` from the metric impl produces a result whose
    ``value`` is shown for diagnostics but whose ``target_met`` is
    :data:`None` and ``warning`` is :data:`False`.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    layer_id: str
    value: float | None = None
    target: float | None = None
    warn_below: float | None = None
    critical: bool = False
    target_met: bool | None = None
    warning: bool = False
    error: str | None = None
    kind: MetricKind = "score"
    details: dict[str, Any] | None = None
    applicable: bool = True


@dataclass(frozen=True)
class MetricContext:
    """Shared state passed to every metric on a given run."""

    corpus: dict[str, Spec]
    methodology: ResolvedMethodology
    layer_id: str
    params: dict[str, Any]


MetricImpl = Callable[[list[Spec], MetricContext], MetricValue]


# ── registry ─────────────────────────────────────────────────────────────


_REGISTRY: dict[str, MetricImpl] = {}


def register_metric(name: str) -> Callable[[MetricImpl], MetricImpl]:
    """Decorator that records a metric implementation by ``func_ref`` name."""

    def deco(func: MetricImpl) -> MetricImpl:
        if name in _REGISTRY:
            raise ValueError(f"metric {name!r} already registered")
        _REGISTRY[name] = func
        return func

    return deco


def get_metric(name: str) -> MetricImpl | None:
    return _REGISTRY.get(name)


def registered_names() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "MetricContext",
    "MetricImpl",
    "MetricKind",
    "MetricResult",
    "MetricValue",
    "get_metric",
    "register_metric",
    "registered_names",
]
