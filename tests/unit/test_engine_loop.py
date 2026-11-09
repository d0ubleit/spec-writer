"""S1.9: engine loop — pipeline + per-layer iteration + 4-state evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_writer.engine.loop import (
    LoopState,
    evaluate_loop_state,
    progressing,
    run_layered_pipeline,
)
from spec_writer.engine.project_context import ProjectContext
from spec_writer.engine.seed import build_seed
from spec_writer.engine.traceability import symmetrize_refs
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.sources import FilesystemMixinSource
from spec_writer.types import Layer, RelatesToEntry, Scenario, Spec

# ── helpers ──────────────────────────────────────────────────────────────


def _bootstrap_project(tmp_path: Path) -> ProjectContext:
    (tmp_path / "spec-editor").mkdir()
    (tmp_path / "spec-editor" / "sources-raw").mkdir()
    (tmp_path / "spec-editor" / "sources").mkdir()
    (tmp_path / "spec-editor" / "specs").mkdir()
    (tmp_path / "spec-editor" / "_meta").mkdir()
    return ProjectContext.from_cwd(tmp_path)


def _methodology(tmp_path: Path, *, with_core_trace: bool = True):
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
    active = ["vision", "cap"]
    if with_core_trace:
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
        active.append("core-traceability")
    return load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=active)


# ── layers ordering ──────────────────────────────────────────────────────


def test_layers_in_order_fwd(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    assert [layer.id for layer in rm.layers_in_order("fwd")] == ["L1", "L2", "L3", "L4"]


def test_layers_in_order_rev(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    assert [layer.id for layer in rm.layers_in_order("rev")] == ["L4", "L3", "L2", "L1"]


# ── seed builder ─────────────────────────────────────────────────────────


def test_seed_fwd_source_boundary_reads_sources(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    (ctx.sources / "biz.md").write_text("source doc", encoding="utf-8")
    rm = _methodology(tmp_path / "meta")
    layer = next(layer for layer in rm.layers if layer.id == "L1")
    seed = build_seed(layer, "fwd", rm, ctx, corpus={})
    assert any(p.name == "biz.md" for p in seed.paths)


def test_seed_fwd_middle_reads_parent_layer(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    parent_dir = ctx.specs / "vision"
    parent_dir.mkdir()
    (parent_dir / "v.md").write_text("L1", encoding="utf-8")
    rm = _methodology(tmp_path / "meta")
    layer = next(layer for layer in rm.layers if layer.id == "L2")
    seed = build_seed(layer, "fwd", rm, ctx, corpus={})
    assert any(p.name == "v.md" for p in seed.paths)


def test_seed_rev_code_boundary_empty(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    rm = _methodology(tmp_path / "meta")
    layer = next(layer for layer in rm.layers if layer.id == "L4")
    seed = build_seed(layer, "rev", rm, ctx, corpus={})
    assert seed.paths == []
    assert any("REV" in note for note in seed.notes)


def test_seed_rev_middle_reads_lower_layer(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    lower = ctx.specs / "technical"
    lower.mkdir()
    (lower / "api.md").write_text("L4", encoding="utf-8")
    rm = _methodology(tmp_path / "meta")
    layer = next(layer for layer in rm.layers if layer.id == "L3")
    seed = build_seed(layer, "rev", rm, ctx, corpus={})
    assert any(p.name == "api.md" for p in seed.paths)


def test_seed_includes_sibling_summary(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    rm = _methodology(tmp_path / "meta")
    layer = next(layer for layer in rm.layers if layer.id == "L1")
    existing = Spec(id="L1-VIS-001", type="vision", layer="L1", title="Existing Vision", up_to=[])
    seed = build_seed(layer, "fwd", rm, ctx, corpus={existing.id: existing})
    assert seed.existing_corpus_summary == {"L1-VIS-001": "Existing Vision"}


# ── traceability symmetrize ──────────────────────────────────────────────


def test_symmetrize_adds_down_to() -> None:
    parent = Spec(id="L1-VIS-001", type="vision", layer="L1", title="P", up_to=[])
    child = Spec(
        id="L2-CAP-001",
        type="capability",
        layer="L2",
        title="C",
        up_to=["L1-VIS-001"],
    )
    symmetrize_refs([parent, child])
    assert parent.down_to == ["L2-CAP-001"]


def test_symmetrize_adds_up_to_from_down_to() -> None:
    parent = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="P",
        up_to=[],
        down_to=["L2-CAP-001"],
    )
    child = Spec(id="L2-CAP-001", type="capability", layer="L2", title="C", up_to=[])
    symmetrize_refs([parent, child])
    assert child.up_to == ["L1-VIS-001"]


def test_symmetrize_idempotent() -> None:
    parent = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="P",
        up_to=[],
        down_to=["L2-CAP-001"],
    )
    child = Spec(
        id="L2-CAP-001",
        type="capability",
        layer="L2",
        title="C",
        up_to=["L1-VIS-001"],
    )
    symmetrize_refs([parent, child])
    symmetrize_refs([parent, child])
    assert parent.down_to == ["L2-CAP-001"]
    assert child.up_to == ["L1-VIS-001"]


def test_symmetrize_paired_relates_to() -> None:
    a = Spec(
        id="L2-CAP-001",
        type="capability",
        layer="L2",
        title="A",
        up_to=[],
        relates_to=[RelatesToEntry(id="L2-CAP-002", kind="depends_on", reason="why")],
    )
    b = Spec(id="L2-CAP-002", type="capability", layer="L2", title="B", up_to=[])
    symmetrize_refs([a, b])
    assert any(r.id == "L2-CAP-001" and r.kind == "used_by" for r in b.relates_to)


def test_symmetrize_skips_unknown_ids() -> None:
    spec = Spec(
        id="L2-CAP-001",
        type="capability",
        layer="L2",
        title="A",
        up_to=["L1-VIS-999"],
    )
    symmetrize_refs([spec])
    # nothing to do; should not crash
    assert spec.up_to == ["L1-VIS-999"]


# ── 4-state evaluation ───────────────────────────────────────────────────


def test_evaluate_converged_when_no_errors() -> None:
    assert (
        evaluate_loop_state(error_count=0, iteration=1, max_iterations=5, is_progressing=True)
        is LoopState.CONVERGED
    )


def test_evaluate_continue_when_progressing() -> None:
    assert (
        evaluate_loop_state(error_count=2, iteration=2, max_iterations=5, is_progressing=True)
        is LoopState.CONTINUE
    )


def test_evaluate_failed_when_not_progressing() -> None:
    assert (
        evaluate_loop_state(error_count=2, iteration=2, max_iterations=5, is_progressing=False)
        is LoopState.FAILED
    )


def test_evaluate_stopped_short_at_cap() -> None:
    assert (
        evaluate_loop_state(error_count=2, iteration=5, max_iterations=5, is_progressing=True)
        is LoopState.STOPPED_SHORT
    )


def test_progressing_first_iteration_true() -> None:
    assert progressing(None, 3) is True


def test_progressing_strict_decrease() -> None:
    assert progressing(5, 3) is True
    assert progressing(3, 3) is False
    assert progressing(3, 5) is False


# ── end-to-end pipeline ──────────────────────────────────────────────────


async def _producer_l1(layer: Layer, seed, corpus, methodology) -> list[Spec]:
    if layer.id == "L1":
        return [
            Spec(
                id="L1-VIS-001",
                type="vision",
                layer="L1",
                title="Generated Vision",
                up_to=[],
            )
        ]
    return []


@pytest.mark.asyncio
async def test_pipeline_single_layer_converges(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    rm = _methodology(tmp_path / "meta")
    scenario = Scenario(direction="fwd", layers_in_scope=["L1"], mode="generate", max_iterations=3)
    result = await run_layered_pipeline(scenario, rm, ctx, _producer_l1)
    assert result.succeeded is True
    assert len(result.layer_results) == 1
    assert result.layer_results[0].state is LoopState.CONVERGED
    assert result.layer_results[0].accepted_specs == ["L1-VIS-001"]


async def _producer_two_layers(layer: Layer, seed, corpus, methodology) -> list[Spec]:
    if layer.id == "L1":
        return [
            Spec(
                id="L1-VIS-001",
                type="vision",
                layer="L1",
                title="Vision",
                up_to=[],
            )
        ]
    if layer.id == "L2":
        return [
            Spec(
                id="L2-CAP-001",
                type="capability",
                layer="L2",
                title="Cap",
                up_to=["L1-VIS-001"],
            )
        ]
    return []


@pytest.mark.asyncio
async def test_pipeline_two_layers_compose(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    rm = _methodology(tmp_path / "meta")
    scenario = Scenario(
        direction="fwd", layers_in_scope=["L1", "L2"], mode="generate", max_iterations=3
    )
    result = await run_layered_pipeline(scenario, rm, ctx, _producer_two_layers)
    assert result.succeeded is True
    states = [layer_result.state for layer_result in result.layer_results]
    assert states == [LoopState.CONVERGED, LoopState.CONVERGED]


async def _producer_bad_skip_level(layer: Layer, seed, corpus, methodology) -> list[Spec]:
    if layer.id == "L1":
        return [Spec(id="L1-VIS-001", type="vision", layer="L1", title="V", up_to=[])]
    if layer.id == "L2":
        # Invalid: empty up_to on a middle layer
        return [Spec(id="L2-CAP-001", type="capability", layer="L2", title="C", up_to=[])]
    return []


async def _producer_empty(layer: Layer, seed, corpus, methodology) -> list[Spec]:
    return []


@pytest.mark.asyncio
async def test_pipeline_empty_drafts_marks_layer_failed(tmp_path: Path) -> None:
    """A producer that returns no drafts must NOT silently 'converge'."""
    ctx = _bootstrap_project(tmp_path)
    rm = _methodology(tmp_path / "meta")
    scenario = Scenario(direction="fwd", layers_in_scope=["L1"], mode="generate", max_iterations=1)
    result = await run_layered_pipeline(scenario, rm, ctx, _producer_empty)
    assert result.succeeded is False
    assert result.layer_results[0].state is LoopState.FAILED
    assert any(i.code == "empty_layer" for i in result.layer_results[0].issues)


@pytest.mark.asyncio
async def test_pipeline_failed_layer_short_circuits(tmp_path: Path) -> None:
    ctx = _bootstrap_project(tmp_path)
    rm = _methodology(tmp_path / "meta")
    scenario = Scenario(
        direction="fwd", layers_in_scope=["L1", "L2"], mode="generate", max_iterations=2
    )
    result = await run_layered_pipeline(scenario, rm, ctx, _producer_bad_skip_level)
    assert result.succeeded is False
    assert result.layer_results[-1].state in (LoopState.FAILED, LoopState.STOPPED_SHORT)
