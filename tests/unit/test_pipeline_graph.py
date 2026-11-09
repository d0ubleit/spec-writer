"""LangGraph batch orchestrator — parity + convergence + HITL tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_writer.engine.loop import LoopState, run_layered_pipeline
from spec_writer.engine.project_context import ProjectContext
from spec_writer.interface.batch.graph import (
    GraphPipelineRun,
    run_pipeline_graph,
)
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.sources import FilesystemMixinSource
from spec_writer.types import Layer, Scenario, Spec

# ── helpers (mirror tests/unit/test_engine_loop.py) ──────────────────────


def _bootstrap_project(tmp_path: Path) -> ProjectContext:
    (tmp_path / "spec-editor").mkdir()
    (tmp_path / "spec-editor" / "sources-raw").mkdir()
    (tmp_path / "spec-editor" / "sources").mkdir()
    (tmp_path / "spec-editor" / "specs").mkdir()
    (tmp_path / "spec-editor" / "_meta").mkdir()
    return ProjectContext.from_cwd(tmp_path)


def _methodology(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base.yaml"
    base.write_text(
        (
            "version: 1\n"
            "layers:\n"
            "  - id: L1\n    role: source-boundary\n    name: vision\n"
            "  - id: L2\n    role: middle\n    name: capabilities\n"
            "  - id: L3\n    role: middle\n    name: behaviors\n"
            "  - id: L4\n    role: code-boundary\n    name: technical\n    is_final: true\n"
        ),
        encoding="utf-8",
    )
    mixins = tmp_path / "mixins"
    mixins.mkdir()
    (mixins / "vision.yaml").write_text(
        "mixin: vision\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins / "cap.yaml").write_text(
        "mixin: cap\nadds_types:\n  - id: capability\n    layer_name: capabilities\n    prefix: CAP\n",
        encoding="utf-8",
    )
    (mixins / "core-traceability.yaml").write_text(
        (
            "mixin: core-traceability\n"
            "validators:\n"
            "  - id: pyramid-adjacency\n"
            "    scope: per-spec\n"
            "    func_ref: validators.check_pyramid_adjacency\n"
            "  - id: dangling-ref\n"
            "    scope: cross-spec\n"
            "    func_ref: validators.check_dangling_refs\n"
            "  - id: english-output\n"
            "    scope: per-spec\n"
            "    func_ref: validators.check_english_output\n"
        ),
        encoding="utf-8",
    )
    return load_methodology(
        base,
        [FilesystemMixinSource(mixins)],
        active_mixins=["vision", "cap", "core-traceability"],
    )


class _Recorder:
    def __init__(self) -> None:
        self.ids: list[str] = []

    def __call__(self, spec: Spec, methodology: object, project_ctx: object) -> None:
        self.ids.append(spec.id)


def _vision(spec_id: str = "L1-VIS-001", *, up_to: list[str] | None = None) -> Spec:
    return Spec(id=spec_id, type="vision", layer="L1", title="Vision", up_to=up_to or [])


def _capability(spec_id: str = "L2-CAP-001", *, up_to: list[str] | None = None) -> Spec:
    return Spec(
        id=spec_id,
        type="capability",
        layer="L2",
        title="Capability",
        up_to=up_to or ["L1-VIS-001"],
    )


def _producer_returning(by_layer: dict[str, list[Spec]]):
    async def producer(layer: Layer, seed, corpus, methodology) -> list[Spec]:
        # Deep-copy: symmetrize_refs mutates specs in place, and these
        # fixtures are shared between the loop and graph runs.
        return [spec.model_copy(deep=True) for spec in by_layer.get(layer.id, [])]

    return producer


# ── parity with the hand-rolled loop ────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_matches_loop_on_converged_run(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    rm = _methodology(tmp_path / "meta")
    scenario = Scenario(
        direction="fwd", layers_in_scope=["L1", "L2"], mode="generate", max_iterations=3
    )
    producer = _producer_returning({"L1": [_vision()], "L2": [_capability()]})

    loop_written = _Recorder()
    loop_result = await run_layered_pipeline(scenario, rm, ctx, producer, spec_writer=loop_written)

    graph_written = _Recorder()
    graph_result = await run_pipeline_graph(scenario, rm, ctx, producer, spec_writer=graph_written)

    assert loop_result.succeeded is True
    assert graph_result.succeeded is True
    assert [r.layer_id for r in graph_result.layer_results] == [
        r.layer_id for r in loop_result.layer_results
    ]
    assert [r.accepted_specs for r in graph_result.layer_results] == [
        r.accepted_specs for r in loop_result.layer_results
    ]
    assert graph_written.ids == loop_written.ids == ["L1-VIS-001", "L2-CAP-001"]


# ── convergence semantics ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_retries_then_converges(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    rm = _methodology(tmp_path / "meta")
    scenario = Scenario(direction="fwd", layers_in_scope=["L1"], mode="generate", max_iterations=3)

    calls = {"n": 0}

    async def flaky(layer: Layer, seed, corpus, methodology) -> list[Spec]:
        calls["n"] += 1
        if calls["n"] == 1:
            # source-boundary specs must not carry up_to → validator error
            return [_vision(up_to=["L1-VIS-999"])]
        return [_vision()]

    result = await run_pipeline_graph(scenario, rm, ctx, flaky, spec_writer=_Recorder())

    assert result.succeeded is True
    assert calls["n"] == 2
    assert result.layer_results[0].iterations == 2
    assert result.layer_results[0].state is LoopState.CONVERGED


@pytest.mark.asyncio
async def test_graph_fails_when_no_progress(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    rm = _methodology(tmp_path / "meta")
    scenario = Scenario(direction="fwd", layers_in_scope=["L1"], mode="generate", max_iterations=5)

    async def broken(layer: Layer, seed, corpus, methodology) -> list[Spec]:
        return [_vision(up_to=["L1-VIS-999"])]

    result = await run_pipeline_graph(scenario, rm, ctx, broken, spec_writer=_Recorder())

    assert result.succeeded is False
    assert result.layer_results[0].state is LoopState.FAILED
    assert "failed" in (result.reason or "")


@pytest.mark.asyncio
async def test_graph_empty_drafts_fail_loudly(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    rm = _methodology(tmp_path / "meta")
    scenario = Scenario(direction="fwd", layers_in_scope=["L1"], mode="generate", max_iterations=2)

    async def silent(layer: Layer, seed, corpus, methodology) -> list[Spec]:
        return []

    result = await run_pipeline_graph(scenario, rm, ctx, silent, spec_writer=_Recorder())

    assert result.succeeded is False
    assert any(i.code == "empty_layer" for i in result.layer_results[0].issues)


# ── human-in-the-loop ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_interrupts_before_each_layer_and_resumes(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    rm = _methodology(tmp_path / "meta")
    scenario = Scenario(
        direction="fwd", layers_in_scope=["L1", "L2"], mode="generate", max_iterations=3
    )
    producer = _producer_returning({"L1": [_vision()], "L2": [_capability()]})

    runner = GraphPipelineRun(
        scenario=scenario,
        methodology=rm,
        project_ctx=ctx,
        producer=producer,
        spec_writer=_Recorder(),
        require_approval=True,
    )

    paused = await runner.run()
    assert paused is None
    assert runner.interrupted is True
    assert runner.interrupt_payload["layer"] == "L1"

    paused = await runner.resume(approved=True)
    assert paused is None
    assert runner.interrupt_payload["layer"] == "L2"

    result = await runner.resume(approved=True)
    assert result is not None
    assert result.succeeded is True
    assert [r.layer_id for r in result.layer_results] == ["L1", "L2"]


@pytest.mark.asyncio
async def test_graph_rejection_aborts_pipeline(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    rm = _methodology(tmp_path / "meta")
    scenario = Scenario(direction="fwd", layers_in_scope=["L1"], mode="generate", max_iterations=3)

    runner = GraphPipelineRun(
        scenario=scenario,
        methodology=rm,
        project_ctx=ctx,
        producer=_producer_returning({"L1": [_vision()]}),
        spec_writer=_Recorder(),
        require_approval=True,
    )

    assert await runner.run() is None
    result = await runner.resume(approved=False)
    assert result is not None
    assert result.succeeded is False
    assert "rejected" in (result.reason or "")
    assert result.layer_results == []
