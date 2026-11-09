"""S3.5 — metric registry, runner, builtin metrics, engine integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_writer.engine.loop import evaluate_loop_state, progressing
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.metrics import (
    MetricContext,
    MetricValue,
    get_metric,
    register_metric,
    registered_names,
)
from spec_writer.methodology.metrics.runner import critical_failures, run_metrics
from spec_writer.methodology.sources import FilesystemMixinSource
from spec_writer.types import Spec


def _write_base(meta: Path) -> Path:
    (meta / "base.yaml").write_text(
        (
            "version: 1\n"
            "layers:\n"
            "  - id: L1\n    role: source-boundary\n    name: vision\n"
            "  - id: L2\n    role: middle\n    name: capabilities\n"
            "  - id: L3\n    role: code-boundary\n    name: contracts\n"
        ),
        encoding="utf-8",
    )
    return meta / "base.yaml"


def _write_mixins(meta: Path, *, with_metric: bool = True) -> Path:
    mixins = meta / "mixins"
    mixins.mkdir()
    (mixins / "vision.yaml").write_text(
        "mixin: vision\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins / "cap.yaml").write_text(
        "mixin: cap\nadds_types:\n  - id: capability\n    layer_name: capabilities\n    prefix: CAP\n    up_to_types: [vision]\n",
        encoding="utf-8",
    )
    metric_block = ""
    if with_metric:
        metric_block = (
            "metrics:\n"
            "  - id: cap_up_to\n"
            "    layer: L2\n"
            "    func_ref: metrics.up_to_completeness_ratio\n"
            "    target: 1.0\n"
            "    critical: true\n"
        )
    (mixins / "gate.yaml").write_text(
        "mixin: gate\n" + metric_block,
        encoding="utf-8",
    )
    return mixins


def _load_methodology_with_metric(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    base = _write_base(meta)
    mixins = _write_mixins(meta)
    return load_methodology(
        base, [FilesystemMixinSource(mixins)], active_mixins=["vision", "cap", "gate"]
    )


def _spec(spec_id: str, type_: str, layer: str, up_to: list[str] | None = None) -> Spec:
    return Spec(
        id=spec_id,
        type=type_,
        layer=layer,
        title=spec_id,
        up_to=up_to or [],
        body="",
    )


# ── registry ─────────────────────────────────────────────────────────────


def test_builtin_metrics_registered() -> None:
    names = registered_names()
    assert "metrics.up_to_completeness_ratio" in names
    assert "metrics.down_to_completeness_ratio" in names
    assert "metrics.spec_count_at_layer" in names
    assert "metrics.required_frontmatter_ratio" in names


def test_get_metric_missing() -> None:
    assert get_metric("metrics.does_not_exist") is None


def test_register_metric_duplicate_raises() -> None:
    @register_metric("test.dup_metric")
    def _impl1(_specs: list[Spec], _ctx: MetricContext) -> MetricValue:
        return MetricValue(value=1.0)

    with pytest.raises(ValueError, match="already registered"):

        @register_metric("test.dup_metric")
        def _impl2(_specs: list[Spec], _ctx: MetricContext) -> MetricValue:
            return MetricValue(value=2.0)


# ── builtin metrics ──────────────────────────────────────────────────────


def test_up_to_completeness_ratio_full(tmp_path: Path) -> None:
    methodology = _load_methodology_with_metric(tmp_path)
    specs = [
        _spec("L1-VIS-001", "vision", "L1"),
        _spec("L2-CAP-001", "capability", "L2", up_to=["L1-VIS-001"]),
        _spec("L2-CAP-002", "capability", "L2", up_to=["L1-VIS-001"]),
    ]
    results = run_metrics(specs, methodology, layer_ids=["L2"])
    assert len(results) == 1
    assert results[0].value == 1.0
    assert results[0].target_met is True


def test_up_to_completeness_ratio_partial(tmp_path: Path) -> None:
    methodology = _load_methodology_with_metric(tmp_path)
    specs = [
        _spec("L1-VIS-001", "vision", "L1"),
        _spec("L2-CAP-001", "capability", "L2", up_to=["L1-VIS-001"]),
        _spec("L2-CAP-002", "capability", "L2"),  # dangling
    ]
    results = run_metrics(specs, methodology, layer_ids=["L2"])
    assert results[0].value == pytest.approx(0.5)
    assert results[0].target_met is False
    assert results[0].critical is True


def test_up_to_completeness_source_boundary_is_vacuously_true(tmp_path: Path) -> None:
    methodology = _load_methodology_with_metric(tmp_path)
    # Adjust the metric to apply on L1 by re-loading with a different mixin set
    meta = tmp_path / "meta_alt"
    meta.mkdir()
    base = _write_base(meta)
    mixins = meta / "mixins"
    mixins.mkdir()
    (mixins / "v.yaml").write_text(
        "mixin: v\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins / "g.yaml").write_text(
        "mixin: g\nmetrics:\n  - id: vis_up_to\n    layer: L1\n    func_ref: metrics.up_to_completeness_ratio\n    target: 1.0\n",
        encoding="utf-8",
    )
    rm = load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["v", "g"])
    specs = [_spec("L1-VIS-001", "vision", "L1")]
    results = run_metrics(specs, rm, layer_ids=["L1"])
    assert results[0].value == 1.0
    _ = methodology  # silence ruff F841


def test_spec_count_at_layer(tmp_path: Path) -> None:
    methodology = _load_methodology_with_metric(tmp_path)
    # Override mixin to use spec_count metric for L2
    meta = tmp_path / "meta_count"
    meta.mkdir()
    base = _write_base(meta)
    mixins = meta / "mixins"
    mixins.mkdir()
    (mixins / "vision.yaml").write_text(
        "mixin: vision\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins / "cap.yaml").write_text(
        "mixin: cap\nadds_types:\n  - id: capability\n    layer_name: capabilities\n    prefix: CAP\n",
        encoding="utf-8",
    )
    (mixins / "gate.yaml").write_text(
        "mixin: gate\nmetrics:\n  - id: cap_count\n    layer: L2\n    func_ref: metrics.spec_count_at_layer\n",
        encoding="utf-8",
    )
    rm = load_methodology(
        base, [FilesystemMixinSource(mixins)], active_mixins=["vision", "cap", "gate"]
    )
    specs = [
        _spec("L2-CAP-001", "capability", "L2"),
        _spec("L2-CAP-002", "capability", "L2"),
        _spec("L2-CAP-003", "capability", "L2"),
    ]
    results = run_metrics(specs, rm, layer_ids=["L2"])
    assert results[0].value == 3.0
    _ = methodology


# ── runner edge cases ────────────────────────────────────────────────────


def test_metric_missing_func_ref_records_error(tmp_path: Path) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    base = _write_base(meta)
    mixins = meta / "mixins"
    mixins.mkdir()
    (mixins / "vision.yaml").write_text(
        "mixin: vision\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins / "gate.yaml").write_text(
        "mixin: gate\nmetrics:\n  - id: bad\n    layer: L1\n    func_ref: metrics.ghost\n    target: 0.5\n    critical: true\n",
        encoding="utf-8",
    )
    rm = load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["vision", "gate"])
    results = run_metrics([], rm, layer_ids=["L1"])
    assert results[0].error is not None
    assert "not registered" in results[0].error
    assert results[0].target_met is False  # critical → defaults to failed


def test_metric_raising_implementation_caught(tmp_path: Path) -> None:
    @register_metric("test.always_fails")
    def _boom(_specs: list[Spec], _ctx: MetricContext) -> MetricValue:
        raise ValueError("boom")

    meta = tmp_path / "meta"
    meta.mkdir()
    base = _write_base(meta)
    mixins = meta / "mixins"
    mixins.mkdir()
    (mixins / "v.yaml").write_text(
        "mixin: v\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins / "g.yaml").write_text(
        "mixin: g\nmetrics:\n  - id: t\n    layer: L1\n    func_ref: test.always_fails\n",
        encoding="utf-8",
    )
    rm = load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["v", "g"])
    results = run_metrics([], rm, layer_ids=["L1"])
    assert results[0].error is not None
    assert "boom" in results[0].error


def test_critical_failures_filter() -> None:
    from spec_writer.methodology.metrics import MetricResult

    results = [
        MetricResult(id="a", layer_id="L2", value=0.5, target=0.9, critical=True, target_met=False),
        MetricResult(
            id="b", layer_id="L2", value=0.5, target=0.9, critical=False, target_met=False
        ),
        MetricResult(id="c", layer_id="L2", value=1.0, target=0.9, critical=True, target_met=True),
    ]
    failed = critical_failures(results)
    assert [r.id for r in failed] == ["a"]


# ── engine integration ──────────────────────────────────────────────────


def test_evaluate_loop_state_blocks_on_critical_metric() -> None:
    state = evaluate_loop_state(
        error_count=0,
        iteration=1,
        max_iterations=5,
        is_progressing=True,
        critical_metric_failures=1,
    )
    from spec_writer.engine.loop import LoopState

    assert state is LoopState.CONTINUE


def test_evaluate_loop_state_converged_when_clean() -> None:
    from spec_writer.engine.loop import LoopState

    state = evaluate_loop_state(0, 1, 5, True, 0)
    assert state is LoopState.CONVERGED


def test_progressing_metric_aware_handles_flat_errors_rising_metric() -> None:
    # errors flat, but critical metric improved — counts as progress
    assert progressing(
        prev_error_count=3,
        current_error_count=3,
        prev_critical_metric_value_sum=0.4,
        current_critical_metric_value_sum=0.7,
    )


def test_progressing_metric_aware_no_progress() -> None:
    assert not progressing(
        prev_error_count=3,
        current_error_count=3,
        prev_critical_metric_value_sum=0.7,
        current_critical_metric_value_sum=0.7,
    )


def test_progressing_first_iteration() -> None:
    assert progressing(prev_error_count=None, current_error_count=5)
