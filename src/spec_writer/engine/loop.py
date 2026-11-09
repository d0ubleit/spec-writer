"""Engine loop — single core orchestration for every scenario.

The pipeline picks the layers in scope (in direction order), and for
each layer runs an iterative loop:

  build_seed → producer(seed) → symmetrize → validate → evaluate

The :class:`LoopState` is a 4-value summary per the design rationale:

- CONVERGED      — no error issues, gates passed; accept drafts.
- CONTINUE       — errors remain, still making progress; retry.
- STOPPED_SHORT  — iteration cap hit before convergence.
- FAILED         — no progress between iterations; abort.

Phase 1 keeps progress detection trivial (issue count must strictly
decrease); richer per-metric scoring lands with S3.5.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum

from spec_writer.agent.events import (
    EventBus,
    LayerCompleteEvent,
    SpecCreatedEvent,
    StatusEvent,
)
from spec_writer.engine.project_context import ProjectContext
from spec_writer.engine.seed import Seed, build_seed
from spec_writer.engine.traceability import symmetrize_refs
from spec_writer.io.spec_io import write_spec as default_write_spec
from spec_writer.methodology.metrics import MetricResult
from spec_writer.methodology.metrics.runner import critical_failures, run_metrics
from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.methodology.validators import ValidationIssue
from spec_writer.methodology.validators.runner import run_validators
from spec_writer.types import Layer, Scenario, Spec

_LOG = logging.getLogger(__name__)


# ── public types ─────────────────────────────────────────────────────────


class LoopState(str, Enum):
    CONVERGED = "converged"
    CONTINUE = "continue"
    STOPPED_SHORT = "stopped_short"
    FAILED = "failed"


def _empty_issues() -> list[ValidationIssue]:
    return []


def _empty_results() -> list[LayerResult]:
    return []


def _empty_metric_results() -> list[MetricResult]:
    return []


@dataclass(frozen=True)
class LayerResult:
    state: LoopState
    layer_id: str
    iterations: int
    accepted_specs: list[str]
    issues: list[ValidationIssue] = field(default_factory=_empty_issues)
    metrics: list[MetricResult] = field(default_factory=_empty_metric_results)


@dataclass(frozen=True)
class PipelineResult:
    succeeded: bool
    layer_results: list[LayerResult] = field(default_factory=_empty_results)
    reason: str | None = None


DraftProducer = Callable[
    [Layer, Seed, dict[str, Spec], ResolvedMethodology],
    Awaitable[list[Spec]],
]


SpecWriter = Callable[[Spec, ResolvedMethodology, ProjectContext], object]
"""Disk-write hook. Default is :func:`spec_writer.io.spec_io.write_spec`;
tests can pass a stub to suppress filesystem writes."""


# ── progress detection ───────────────────────────────────────────────────


def progressing(
    prev_error_count: int | None,
    current_error_count: int,
    *,
    prev_critical_metric_value_sum: float | None = None,
    current_critical_metric_value_sum: float | None = None,
) -> bool:
    """Strict decrease in error count OR strict increase in summed critical metric.

    Phase 1 only counted errors; with S3.5 a second axis arrives — if the
    error count holds steady but the sum of critical metric values rose,
    we still treat the loop as progressing because the agent is closing
    the metric gap. Conversely, both metrics flat AND errors flat = no
    progress.
    """
    if prev_error_count is None:
        return True
    if current_error_count < prev_error_count:
        return True
    if (
        prev_critical_metric_value_sum is not None
        and current_critical_metric_value_sum is not None
        and current_critical_metric_value_sum > prev_critical_metric_value_sum
    ):
        return True
    return False


@dataclass(frozen=True)
class IterationEvaluation:
    """Outcome of validating + scoring one produce-iteration."""

    state: LoopState
    issues: list[ValidationIssue]
    metrics: list[MetricResult]
    error_count: int
    critical_metric_failures: int
    critical_sum: float


def evaluate_iteration(
    *,
    layer: Layer,
    methodology: ResolvedMethodology,
    corpus: dict[str, Spec],
    drafts: list[Spec],
    iteration: int,
    max_iterations: int,
    prev_error_count: int | None,
    prev_critical_sum: float | None,
) -> IterationEvaluation:
    """Validate and score a provisional corpus built from ``corpus + drafts``.

    Shared by the hand-rolled :func:`run_layer_loop` and by the LangGraph
    orchestrator (:mod:`spec_writer.interface.batch.graph`) so both
    engines apply byte-for-byte the same acceptance rules.
    """
    candidates = {**corpus}
    for draft in drafts:
        candidates[draft.id] = draft

    symmetrize_refs(list(candidates.values()))

    issues = run_validators(list(candidates.values()), methodology)
    errors = [issue for issue in issues if issue.severity == "error"]

    metric_results = run_metrics(
        list(candidates.values()),
        methodology,
        layer_ids=[layer.id],
    )
    critical_unmet = critical_failures(metric_results)
    critical_sum = sum((result.value or 0.0) for result in metric_results if result.critical)

    state = evaluate_loop_state(
        error_count=len(errors),
        iteration=iteration,
        max_iterations=max_iterations,
        is_progressing=progressing(
            prev_error_count,
            len(errors),
            prev_critical_metric_value_sum=prev_critical_sum,
            current_critical_metric_value_sum=critical_sum,
        ),
        critical_metric_failures=len(critical_unmet),
    )

    return IterationEvaluation(
        state=state,
        issues=issues,
        metrics=metric_results,
        error_count=len(errors),
        critical_metric_failures=len(critical_unmet),
        critical_sum=critical_sum,
    )


def evaluate_loop_state(
    error_count: int,
    iteration: int,
    max_iterations: int,
    is_progressing: bool,
    critical_metric_failures: int = 0,
) -> LoopState:
    """Map (errors, iteration, progress, critical-metric-failures) → LoopState.

    Convergence requires both zero errors AND zero critical-metric failures.
    Iteration-cap takes priority over "failing" so the agent gets one more
    chance on subsequent runs.
    """
    if error_count == 0 and critical_metric_failures == 0:
        return LoopState.CONVERGED
    if iteration >= max_iterations:
        return LoopState.STOPPED_SHORT
    if not is_progressing:
        return LoopState.FAILED
    return LoopState.CONTINUE


# ── pipeline orchestration ───────────────────────────────────────────────


async def run_layered_pipeline(
    scenario: Scenario,
    methodology: ResolvedMethodology,
    project_ctx: ProjectContext,
    producer: DraftProducer,
    existing_corpus: dict[str, Spec] | None = None,
    event_bus: EventBus | None = None,
    spec_writer: SpecWriter | None = None,
    instruction: str | None = None,
) -> PipelineResult:
    """Run the full pipeline for one scenario.

    Layers are iterated in ``scenario.direction`` order. The first layer
    that fails short-circuits the pipeline and returns a failed result;
    successful layers are accepted into the corpus before moving on.
    """
    corpus = dict(existing_corpus or {})
    layers = methodology.layers_in_order(scenario.direction)
    layers_in_scope = [layer for layer in layers if layer.id in scenario.layers_in_scope]
    layer_results: list[LayerResult] = []

    if event_bus:
        event_bus.publish(
            StatusEvent(
                phase="pipeline:start",
                details={
                    "direction": scenario.direction,
                    "mode": scenario.mode,
                    "layers": [layer.id for layer in layers_in_scope],
                },
            )
        )

    writer: SpecWriter = spec_writer if spec_writer is not None else default_write_spec
    for layer in layers_in_scope:
        result = await run_layer_loop(
            layer,
            scenario,
            methodology,
            project_ctx,
            producer,
            corpus,
            event_bus,
            writer,
            instruction=instruction,
        )
        layer_results.append(result)
        if result.state in (LoopState.FAILED, LoopState.STOPPED_SHORT):
            return PipelineResult(
                succeeded=False,
                layer_results=layer_results,
                reason=f"layer {layer.id} ended in state {result.state.value}",
            )

    if event_bus:
        event_bus.publish(StatusEvent(phase="pipeline:done", details={}))

    return PipelineResult(succeeded=True, layer_results=layer_results)


async def run_layer_loop(
    layer: Layer,
    scenario: Scenario,
    methodology: ResolvedMethodology,
    project_ctx: ProjectContext,
    producer: DraftProducer,
    corpus: dict[str, Spec],
    event_bus: EventBus | None,
    spec_writer: SpecWriter = default_write_spec,
    instruction: str | None = None,
) -> LayerResult:
    """Iterate one layer until convergence or cap."""
    if event_bus:
        event_bus.publish(StatusEvent(phase="layer:begin", details={"layer": layer.id}))

    prev_errors: int | None = None
    prev_critical_sum: float | None = None
    last_issues: list[ValidationIssue] = []
    last_metrics: list[MetricResult] = []
    last_accepted: list[str] = []
    iteration = 0

    for iteration in range(1, scenario.max_iterations + 1):
        seed = build_seed(
            layer,
            scenario.direction,
            methodology,
            project_ctx,
            corpus,
            mode=scenario.mode,
            instruction=instruction,
            scope=scenario.scope,
        )
        drafts = await producer(layer, seed, corpus, methodology)

        evaluation = evaluate_iteration(
            layer=layer,
            methodology=methodology,
            corpus=corpus,
            drafts=drafts,
            iteration=iteration,
            max_iterations=scenario.max_iterations,
            prev_error_count=prev_errors,
            prev_critical_sum=prev_critical_sum,
        )
        issues = evaluation.issues
        metric_results = evaluation.metrics
        state = evaluation.state
        last_issues = issues
        last_metrics = metric_results

        if state is LoopState.CONVERGED:
            for draft in drafts:
                corpus[draft.id] = draft
                last_accepted.append(draft.id)
                spec_writer(draft, methodology, project_ctx)
                if event_bus:
                    event_bus.publish(SpecCreatedEvent(spec_id=draft.id, diff=f"+ {draft.title}"))
            if not last_accepted:
                # Producer returned nothing — validators trivially pass on
                # the unchanged corpus, but the layer hasn't actually been
                # generated. Treat this as FAILED so callers don't see a
                # silent no-op success (principle 9 fail-loudly).
                empty_issue = ValidationIssue(
                    severity="error",
                    code="empty_layer",
                    message=(
                        f"layer {layer.id} produced no drafts — the agent did not "
                        f"call record_spec (likely tool wiring or prompt issue)."
                    ),
                )
                return LayerResult(
                    state=LoopState.FAILED,
                    layer_id=layer.id,
                    iterations=iteration,
                    accepted_specs=[],
                    issues=[*issues, empty_issue],
                    metrics=metric_results,
                )
            if event_bus:
                event_bus.publish(
                    LayerCompleteEvent(layer=layer.id, metrics={"iterations": iteration})
                )
            return LayerResult(
                state=LoopState.CONVERGED,
                layer_id=layer.id,
                iterations=iteration,
                accepted_specs=last_accepted,
                issues=issues,
                metrics=metric_results,
            )

        if state in (LoopState.STOPPED_SHORT, LoopState.FAILED):
            _LOG.warning(
                "layer %s ended in state %s with %d errors / %d critical metric failures",
                layer.id,
                state.value,
                evaluation.error_count,
                evaluation.critical_metric_failures,
            )
            return LayerResult(
                state=state,
                layer_id=layer.id,
                iterations=iteration,
                accepted_specs=[],
                issues=issues,
                metrics=metric_results,
            )

        prev_errors = evaluation.error_count
        prev_critical_sum = evaluation.critical_sum

    return LayerResult(
        state=LoopState.STOPPED_SHORT,
        layer_id=layer.id,
        iterations=iteration,
        accepted_specs=[],
        issues=last_issues,
        metrics=last_metrics,
    )


__all__ = [
    "DraftProducer",
    "IterationEvaluation",
    "LayerResult",
    "LoopState",
    "PipelineResult",
    "evaluate_iteration",
    "evaluate_loop_state",
    "progressing",
    "run_layer_loop",
    "run_layered_pipeline",
]
