"""LangGraph orchestrator for the batch pipeline.

The interactive chat path stays agent-owned — one context per session,
the engine is a library of primitives (principle 5a). This module is the
deliberate exception the design allows: the *batch escape hatch*, where
there is no human to converse with and an explicit orchestrator is
appropriate.

It implements exactly the same ``seed → produce → evaluate → (retry |
next layer)`` semantics as :func:`spec_writer.engine.loop.run_layered_pipeline`,
but expresses them as a compiled LangGraph ``StateGraph``:

* typed state with JSON-checkpointable payloads,
* conditional edges for the convergence / retry decision,
* a checkpointer, so a run can be inspected and resumed,
* ``interrupt()``-based human approval between layers (opt-in).

Both engines share :func:`spec_writer.engine.loop.evaluate_iteration`, so
acceptance rules cannot drift apart.

Usage::

    runner = GraphPipelineRun(
        scenario=scenario,
        methodology=methodology,
        project_ctx=project_ctx,
        producer=producer,
        existing_corpus=corpus,
        require_approval=True,
    )
    result = await runner.run()
    while runner.interrupted:
        result = await runner.resume(approved=True)
"""

# LangGraph ships inline types for most modules but not all of them in
# this version; keep the local gate focused on our own code.
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from spec_writer.agent.events import (
    EventBus,
    LayerCompleteEvent,
    SpecCreatedEvent,
    StatusEvent,
)
from spec_writer.engine.loop import (
    DraftProducer,
    LayerResult,
    LoopState,
    PipelineResult,
    SpecWriter,
    evaluate_iteration,
)
from spec_writer.engine.project_context import ProjectContext
from spec_writer.engine.seed import build_seed
from spec_writer.io.spec_io import write_spec as default_write_spec
from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.methodology.validators import ValidationIssue
from spec_writer.types import Layer, Scenario, Spec

# ── state + runtime context ──────────────────────────────────────────────


class PipelineState(TypedDict, total=False):
    """Checkpointable slice of a run.

    Only JSON-friendly values live here. Heavy objects (corpus, resolved
    methodology, producer, results) ride in :class:`GraphContext`, which
    is not serialised — a fresh process resumes by rebuilding it from
    disk (see ``spec-writer generate --engine graph``).
    """

    layer_index: int
    iteration: int
    prev_error_count: int | None
    prev_critical_sum: float | None
    drafts: list[dict[str, Any]]
    phase: str
    reason: str | None


@dataclass
class GraphContext:
    """Non-serialised collaborators handed to every node."""

    scenario: Scenario
    methodology: ResolvedMethodology
    project_ctx: ProjectContext
    producer: DraftProducer
    layers: list[Layer]
    corpus: dict[str, Spec] = field(default_factory=dict)
    layer_results: list[LayerResult] = field(default_factory=list)
    event_bus: EventBus | None = None
    writer: SpecWriter = default_write_spec
    instruction: str | None = None


# ── nodes ────────────────────────────────────────────────────────────────


async def _prepare(state: PipelineState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    if ctx.event_bus:
        ctx.event_bus.publish(
            StatusEvent(
                phase="pipeline:start",
                details={
                    "engine": "langgraph",
                    "direction": ctx.scenario.direction,
                    "mode": ctx.scenario.mode,
                    "layers": [layer.id for layer in ctx.layers],
                },
            )
        )
    return {
        "layer_index": 0,
        "iteration": 1,
        "prev_error_count": None,
        "prev_critical_sum": None,
        "drafts": [],
        "phase": "produce",
        "reason": None,
    }


async def _approve(state: PipelineState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    """Human-in-the-loop gate before generating a layer.

    ``interrupt()`` suspends the graph and surfaces the payload to the
    caller; on resume the node re-executes and ``interrupt()`` returns
    the resumed value. The node is side-effect free, so re-execution is
    safe.
    """
    ctx = runtime.context
    layer = ctx.layers[int(state.get("layer_index", 0))]
    decision = interrupt(
        {
            "type": "layer_approval",
            "layer": layer.id,
            "layer_name": layer.name,
            "position": int(state.get("layer_index", 0)) + 1,
            "total": len(ctx.layers),
        }
    )
    if decision is False or decision is None:
        return {
            "phase": "rejected",
            "reason": f"layer {layer.id} rejected by operator",
        }
    return {"phase": "produce"}


async def _produce(state: PipelineState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    index = int(state.get("layer_index", 0))
    layer = ctx.layers[index]
    seed = build_seed(
        layer,
        ctx.scenario.direction,
        ctx.methodology,
        ctx.project_ctx,
        ctx.corpus,
        mode=ctx.scenario.mode,
        instruction=ctx.instruction,
        scope=ctx.scenario.scope,
    )
    drafts = await ctx.producer(layer, seed, ctx.corpus, ctx.methodology)
    return {"drafts": [draft.model_dump(mode="json") for draft in drafts]}


async def _evaluate(state: PipelineState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    index = int(state.get("layer_index", 0))
    layer = ctx.layers[index]
    iteration = int(state.get("iteration", 1))
    drafts = [Spec.model_validate(payload) for payload in state.get("drafts", [])]

    evaluation = evaluate_iteration(
        layer=layer,
        methodology=ctx.methodology,
        corpus=ctx.corpus,
        drafts=drafts,
        iteration=iteration,
        max_iterations=ctx.scenario.max_iterations,
        prev_error_count=state.get("prev_error_count"),
        prev_critical_sum=state.get("prev_critical_sum"),
    )

    if evaluation.state is LoopState.CONVERGED:
        if not drafts:
            # Producer returned nothing — the layer has not actually been
            # generated. Fail loudly instead of reporting a silent no-op
            # success (principle 9).
            empty_issue = ValidationIssue(
                severity="error",
                code="empty_layer",
                message=(
                    f"layer {layer.id} produced no drafts — the agent did not "
                    f"call record_spec (likely tool wiring or prompt issue)."
                ),
            )
            ctx.layer_results.append(
                LayerResult(
                    state=LoopState.FAILED,
                    layer_id=layer.id,
                    iterations=iteration,
                    accepted_specs=[],
                    issues=[*evaluation.issues, empty_issue],
                    metrics=evaluation.metrics,
                )
            )
            return {
                "phase": "failed",
                "reason": f"layer {layer.id} produced no drafts",
                "drafts": [],
            }

        accepted: list[str] = []
        for draft in drafts:
            ctx.corpus[draft.id] = draft
            accepted.append(draft.id)
            ctx.writer(draft, ctx.methodology, ctx.project_ctx)
            if ctx.event_bus:
                ctx.event_bus.publish(SpecCreatedEvent(spec_id=draft.id, diff=f"+ {draft.title}"))

        if ctx.event_bus:
            ctx.event_bus.publish(
                LayerCompleteEvent(layer=layer.id, metrics={"iterations": iteration})
            )

        ctx.layer_results.append(
            LayerResult(
                state=LoopState.CONVERGED,
                layer_id=layer.id,
                iterations=iteration,
                accepted_specs=accepted,
                issues=evaluation.issues,
                metrics=evaluation.metrics,
            )
        )
        next_index = index + 1
        return {
            "layer_index": next_index,
            "iteration": 1,
            "prev_error_count": None,
            "prev_critical_sum": None,
            "drafts": [],
            "phase": "next_layer" if next_index < len(ctx.layers) else "done",
        }

    if evaluation.state in (LoopState.STOPPED_SHORT, LoopState.FAILED):
        ctx.layer_results.append(
            LayerResult(
                state=evaluation.state,
                layer_id=layer.id,
                iterations=iteration,
                accepted_specs=[],
                issues=evaluation.issues,
                metrics=evaluation.metrics,
            )
        )
        return {
            "phase": "failed",
            "reason": f"layer {layer.id} ended in state {evaluation.state.value}",
            "drafts": [],
        }

    return {
        "iteration": iteration + 1,
        "prev_error_count": evaluation.error_count,
        "prev_critical_sum": evaluation.critical_sum,
        "drafts": [],
        "phase": "retry",
    }


# ── graph assembly ───────────────────────────────────────────────────────


def build_pipeline_graph(
    ctx: GraphContext,
    *,
    require_approval: bool = False,
    checkpointer: Any | None = None,
) -> Any:
    """Compile the batch pipeline graph.

    ``checkpointer`` defaults to an in-memory saver; pass a persistent
    saver (e.g. ``SqliteSaver``) to survive process restarts.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    graph: Any = StateGraph(PipelineState, context_schema=GraphContext)
    graph.add_node("prepare", _prepare)
    graph.add_node("produce", _produce)
    graph.add_node("evaluate", _evaluate)
    graph.add_node("approve", _approve)

    graph.add_edge(START, "prepare")

    if require_approval:
        graph.add_edge("prepare", "approve")
    else:
        graph.add_edge("prepare", "produce")

    def after_approval(state: PipelineState) -> str:
        return "produce" if state.get("phase") == "produce" else END

    graph.add_conditional_edges("approve", after_approval, ["produce", END])

    graph.add_edge("produce", "evaluate")

    def after_evaluate(state: PipelineState) -> str:
        phase = state.get("phase")
        if phase == "retry":
            return "produce"
        if phase == "next_layer":
            return "approve" if require_approval else "produce"
        return END

    graph.add_conditional_edges("evaluate", after_evaluate, ["produce", "approve", END])

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


# ── runner ───────────────────────────────────────────────────────────────


class GraphPipelineRun:
    """Stateful wrapper that hides graph plumbing from callers."""

    def __init__(
        self,
        *,
        scenario: Scenario,
        methodology: ResolvedMethodology,
        project_ctx: ProjectContext,
        producer: DraftProducer,
        existing_corpus: dict[str, Spec] | None = None,
        event_bus: EventBus | None = None,
        spec_writer: SpecWriter | None = None,
        instruction: str | None = None,
        require_approval: bool = False,
        thread_id: str | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        layers_in_scope = {layer.id for layer in methodology.layers}
        ordered = [
            layer
            for layer in methodology.layers_in_order(scenario.direction)
            if layer.id in scenario.layers_in_scope and layer.id in layers_in_scope
        ]
        self._ctx = GraphContext(
            scenario=scenario,
            methodology=methodology,
            project_ctx=project_ctx,
            producer=producer,
            layers=ordered,
            corpus=dict(existing_corpus or {}),
            event_bus=event_bus,
            writer=spec_writer or default_write_spec,
            instruction=instruction,
        )
        self.thread_id = thread_id or uuid.uuid4().hex
        self._config: dict[str, Any] = {"configurable": {"thread_id": self.thread_id}}
        self._graph = build_pipeline_graph(
            self._ctx,
            require_approval=require_approval,
            checkpointer=checkpointer,
        )
        self.interrupted = False
        self.interrupt_payload: Any = None

    async def run(self) -> PipelineResult | None:
        """Start a fresh run. Returns ``None`` when it suspends on approval."""
        return await self._invoke({})

    async def resume(self, approved: bool) -> PipelineResult | None:
        """Continue after an ``interrupt``."""
        return await self._invoke(Command(resume=approved))

    async def _invoke(self, payload: Any) -> PipelineResult | None:
        values = await self._graph.ainvoke(payload, self._config, context=self._ctx)
        interrupts = values.get("__interrupt__") if isinstance(values, dict) else None
        if interrupts:
            self.interrupted = True
            first = interrupts[0]
            self.interrupt_payload = getattr(first, "value", first)
            return None
        self.interrupted = False
        self.interrupt_payload = None
        return self.result_from_state(values if isinstance(values, dict) else {})

    def result_from_state(self, values: dict[str, Any]) -> PipelineResult:
        if values.get("phase") == "done":
            return PipelineResult(succeeded=True, layer_results=list(self._ctx.layer_results))
        reason = values.get("reason") or "pipeline did not converge"
        return PipelineResult(
            succeeded=False,
            layer_results=list(self._ctx.layer_results),
            reason=reason,
        )


async def run_pipeline_graph(
    scenario: Scenario,
    methodology: ResolvedMethodology,
    project_ctx: ProjectContext,
    producer: DraftProducer,
    *,
    existing_corpus: dict[str, Spec] | None = None,
    event_bus: EventBus | None = None,
    spec_writer: SpecWriter | None = None,
    instruction: str | None = None,
    require_approval: bool = False,
    thread_id: str | None = None,
    checkpointer: Any | None = None,
) -> PipelineResult:
    """One-shot convenience wrapper (no human in the loop)."""
    runner = GraphPipelineRun(
        scenario=scenario,
        methodology=methodology,
        project_ctx=project_ctx,
        producer=producer,
        existing_corpus=existing_corpus,
        event_bus=event_bus,
        spec_writer=spec_writer,
        instruction=instruction,
        require_approval=require_approval,
        thread_id=thread_id,
        checkpointer=checkpointer,
    )
    result = await runner.run()
    if result is None:
        raise RuntimeError(
            "pipeline graph suspended for approval; use GraphPipelineRun.resume() instead"
        )
    return result


__all__ = [
    "GraphContext",
    "GraphPipelineRun",
    "PipelineState",
    "build_pipeline_graph",
    "run_pipeline_graph",
]
