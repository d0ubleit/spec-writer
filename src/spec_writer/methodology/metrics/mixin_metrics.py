"""Metrics for built-in mixins shipped beyond core-traceability.

Project-agnostic. Each mixin under ``methodology/_builtin/mixins/``
that declares a custom metric references its impl here via the
``metrics.<name>`` ``func_ref`` strings. Registered at import time.

Mixin → metric mapping:

- user-scenarios     → scenario_step_linkage_ratio (L3)
- rbac-matrix        → rbac_coverage_ratio (L4)
- requirements-traceability → priority_tagged_ratio (per-layer)
- test-cases         → behavior_test_coverage_ratio (L3),
                       scenario_test_coverage_ratio (L3)
"""

from __future__ import annotations

import re
from typing import Any, cast

import yaml

from spec_writer.methodology.metrics import (
    MetricContext,
    MetricValue,
    register_metric,
)
from spec_writer.types import Spec

_RBAC_AWARE_TYPES = frozenset({"api-contract", "ui-screen", "async-job", "integration-contract"})


def _extras(spec: Spec) -> dict[str, Any]:
    return spec.model_extra or {}


def _extract_steps(body: str) -> Any | None:
    pattern = re.compile(
        r"^#+\s*Steps\s*$\n(.*?)(?=^#+\s|\Z)",
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(body)
    if match is None:
        return None
    section = match.group(1).strip()
    fenced = re.search(r"```(?:ya?ml)?\s*\n(.*?)\n```", section, re.DOTALL)
    payload = fenced.group(1) if fenced else section
    try:
        return yaml.safe_load(payload)
    except yaml.YAMLError:
        return None


# ── user-scenarios ───────────────────────────────────────────────────────


@register_metric("metrics.scenario_step_linkage_ratio")
def _scenario_step_linkage(specs: list[Spec], ctx: MetricContext) -> MetricValue:
    """Fraction of user-scenarios whose every step references a real ui-screen."""
    del ctx
    scenarios = [s for s in specs if s.type == "user-scenario"]
    if not scenarios:
        return MetricValue(value=0.0, kind="ratio", details={"n": 0})
    screens = {s.id for s in specs if s.type == "ui-screen"}
    # If scenarios exist but ui-screens haven't been generated yet, the
    # metric is structurally 0.0 — surface value as planning signal but
    # set applicable=False so it doesn't fire warning-noise. See
    # the design rationale.
    if not screens:
        return MetricValue(
            value=0.0,
            kind="ratio",
            details={
                "linked": 0,
                "total": len(scenarios),
                "reason": "no ui-screen specs in corpus (L4 not generated)",
            },
            applicable=False,
        )
    good = 0
    for sc in scenarios:
        parsed = _extract_steps(sc.body)
        if not isinstance(parsed, list):
            continue
        items = cast(list[Any], parsed)
        if not items:
            continue
        ok = True
        for raw_step in items:
            if not isinstance(raw_step, dict):
                ok = False
                break
            step = cast(dict[str, Any], raw_step)
            screen = step.get("screen")
            if not isinstance(screen, str) or screen not in screens:
                ok = False
                break
        if ok:
            good += 1
    return MetricValue(
        value=good / len(scenarios),
        kind="ratio",
        details={"linked": good, "total": len(scenarios)},
    )


# ── rbac-matrix ──────────────────────────────────────────────────────────


@register_metric("metrics.rbac_coverage_ratio")
def _rbac_coverage(specs: list[Spec], ctx: MetricContext) -> MetricValue:
    """Fraction of rbac-aware code-boundary specs with a non-empty roles field.

    Counts only specs whose type belongs to the rbac-aware set
    (api-contract, ui-screen, async-job, integration-contract).
    """
    del ctx
    targeted = [s for s in specs if s.type in _RBAC_AWARE_TYPES]
    if not targeted:
        return MetricValue(value=1.0, kind="ratio", details={"n": 0})
    covered = 0
    for spec in targeted:
        roles = _extras(spec).get("roles")
        if isinstance(roles, list) and roles:
            covered += 1
        elif isinstance(roles, str) and roles:
            covered += 1
    return MetricValue(
        value=covered / len(targeted),
        kind="ratio",
        details={"covered": covered, "total": len(targeted)},
    )


# ── requirements-traceability ────────────────────────────────────────────


def _has_priority(spec: Spec) -> bool:
    value = _extras(spec).get("priority")
    return isinstance(value, str) and bool(value)


@register_metric("metrics.priority_tagged_ratio")
def _priority_tagged(specs: list[Spec], ctx: MetricContext) -> MetricValue:
    """Fraction of specs on the configured layer that declare a `priority` field.

    Helps catch the case where new specs land without MoSCoW
    classification, which makes phase-cut decisions impossible.
    """
    targeted = [s for s in specs if s.layer == ctx.layer_id]
    if not targeted:
        return MetricValue(value=0.0, kind="ratio", details={"layer": ctx.layer_id, "n": 0})
    tagged = sum(1 for s in targeted if _has_priority(s))
    return MetricValue(
        value=tagged / len(targeted),
        kind="ratio",
        details={"tagged": tagged, "total": len(targeted)},
    )


# ── test-cases ───────────────────────────────────────────────────────────


def _test_targets_for(specs: list[Spec], parent_type: str) -> dict[str, list[Spec]]:
    """spec_id (of given type) → list of test-case specs that up_to it."""
    parents = {s.id for s in specs if s.type == parent_type}
    out: dict[str, list[Spec]] = {pid: [] for pid in parents}
    for s in specs:
        if s.type != "test-case":
            continue
        for parent_id in s.up_to:
            if parent_id in out:
                out[parent_id].append(s)
    return out


@register_metric("metrics.behavior_test_coverage_ratio")
def _behavior_test_coverage(specs: list[Spec], ctx: MetricContext) -> MetricValue:
    """Fraction of behavior specs with at least one linked test-case."""
    del ctx
    by_parent = _test_targets_for(specs, "behavior")
    if not by_parent:
        return MetricValue(value=0.0, kind="ratio", details={"n": 0})
    test_cases = [s for s in specs if s.type == "test-case"]
    if not test_cases:
        return MetricValue(
            value=0.0,
            kind="ratio",
            details={
                "covered": 0,
                "total": len(by_parent),
                "reason": "no test-case specs in corpus (L4 not generated)",
            },
            applicable=False,
        )
    covered = sum(1 for tests in by_parent.values() if tests)
    return MetricValue(
        value=covered / len(by_parent),
        kind="ratio",
        details={"covered": covered, "total": len(by_parent)},
    )


@register_metric("metrics.scenario_test_coverage_ratio")
def _scenario_test_coverage(specs: list[Spec], ctx: MetricContext) -> MetricValue:
    """Fraction of user-scenario specs with at least one e2e/acceptance test-case."""
    del ctx
    by_parent = _test_targets_for(specs, "user-scenario")
    if not by_parent:
        return MetricValue(value=0.0, kind="ratio", details={"n": 0})
    test_cases = [s for s in specs if s.type == "test-case"]
    if not test_cases:
        return MetricValue(
            value=0.0,
            kind="ratio",
            details={
                "covered": 0,
                "total": len(by_parent),
                "reason": "no test-case specs in corpus (L4 not generated)",
            },
            applicable=False,
        )
    e2e_kinds = {"e2e", "acceptance"}
    covered = 0
    for tests in by_parent.values():
        for t in tests:
            kind = _extras(t).get("kind")
            if isinstance(kind, str) and kind in e2e_kinds:
                covered += 1
                break
    return MetricValue(
        value=covered / len(by_parent),
        kind="ratio",
        details={"covered": covered, "total": len(by_parent)},
    )


__all__ = [
    "_behavior_test_coverage",
    "_priority_tagged",
    "_rbac_coverage",
    "_scenario_step_linkage",
    "_scenario_test_coverage",
]
