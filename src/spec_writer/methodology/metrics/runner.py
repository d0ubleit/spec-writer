"""Metric runner — compute every metric configured for the layers in scope.

Reads ``methodology.metrics_by_layer``, looks up each impl via
``func_ref``, executes it in a forgiving wrapper, and returns a
:class:`MetricResult` per config entry. The engine consults the
results to gate convergence (``critical=True`` + target unmet → block)
and the CLI ``metrics`` subcommand renders them.

Metric implementations are pure: they receive a snapshot of the
corpus and a ``MetricContext`` and return a :class:`MetricValue`. They
must not raise; if they do, the runner converts the exception to an
``error`` on the result (and surfaces it as a non-blocking issue
unless the metric is critical).
"""

from __future__ import annotations

import logging
from typing import Any, cast

from spec_writer.methodology.metrics import (
    MetricContext,
    MetricResult,
    get_metric,
)
from spec_writer.methodology.metrics import builtin as _builtin
from spec_writer.methodology.metrics import mixin_metrics as _mixin_metrics
from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.types import Spec

# Re-bind for side effect: importing modules triggers @register_metric.
del _builtin, _mixin_metrics

_LOG = logging.getLogger(__name__)


def run_metrics(
    specs: list[Spec],
    methodology: ResolvedMethodology,
    layer_ids: list[str] | None = None,
) -> list[MetricResult]:
    """Compute every methodology-declared metric whose layer is in scope.

    ``layer_ids`` filters which layers to evaluate; ``None`` means all.
    """
    corpus = {spec.id: spec for spec in specs}
    results: list[MetricResult] = []
    target_layers = (
        list(methodology.metrics_by_layer)
        if layer_ids is None
        else [lid for lid in layer_ids if lid in methodology.metrics_by_layer]
    )

    for layer_id in target_layers:
        for config in methodology.metrics_for(layer_id):
            results.append(_run_one(specs, methodology, corpus, layer_id, config))
    return results


def _run_one(
    specs: list[Spec],
    methodology: ResolvedMethodology,
    corpus: dict[str, Spec],
    layer_id: str,
    config: dict[str, Any],
) -> MetricResult:
    metric_id = str(config.get("id") or config.get("func_ref") or "?")
    func_ref = config.get("func_ref")
    target = _coerce_float(config.get("target"))
    warn_below = _coerce_float(config.get("warn_below"))
    critical = bool(config.get("critical", False))
    raw_params = config.get("params")
    params: dict[str, Any] = {}
    if isinstance(raw_params, dict):
        typed_params = cast(dict[str, Any], raw_params)
        params.update(typed_params)

    if not isinstance(func_ref, str):
        return MetricResult(
            id=metric_id,
            layer_id=layer_id,
            target=target,
            warn_below=warn_below,
            critical=critical,
            target_met=False if critical else None,
            error="metric config missing string 'func_ref'",
        )

    impl = get_metric(func_ref)
    if impl is None:
        return MetricResult(
            id=metric_id,
            layer_id=layer_id,
            target=target,
            warn_below=warn_below,
            critical=critical,
            target_met=False if critical else None,
            error=f"metric {func_ref!r} declared but not registered",
        )

    ctx = MetricContext(corpus=corpus, methodology=methodology, layer_id=layer_id, params=params)

    try:
        value = impl(specs, ctx)
    except Exception as exc:
        _LOG.exception("metric %s raised", metric_id)
        return MetricResult(
            id=metric_id,
            layer_id=layer_id,
            target=target,
            warn_below=warn_below,
            critical=critical,
            target_met=False if critical else None,
            error=f"{type(exc).__name__}: {exc}",
        )

    target_met: bool | None = None
    warning = False
    if value.applicable:
        if target is not None:
            target_met = value.value >= target
        warning = warn_below is not None and value.value < warn_below
    # else: applicable=False — value still shown for diagnostics but
    # target_met stays None and warning stays False. See the design rationale.

    return MetricResult(
        id=metric_id,
        layer_id=layer_id,
        value=value.value,
        target=target,
        warn_below=warn_below,
        critical=critical,
        target_met=target_met,
        warning=warning,
        kind=value.kind,
        details=value.details,
        applicable=value.applicable,
    )


def _coerce_float(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def critical_failures(results: list[MetricResult]) -> list[MetricResult]:
    """Metrics that block convergence — critical + target unmet (or errored)."""
    return [r for r in results if r.critical and r.target_met is False]


__all__ = ["critical_failures", "run_metrics"]
