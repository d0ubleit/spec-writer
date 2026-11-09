"""Tests for the five post-pilot engine fixes (traceability, validation, usage).

Covers:
- (A) `read_spec` MCP tool returns frontmatter + body for an id.
- (B) Scoped `run_validators` with `all_specs` keeps cross-layer
  references resolvable (no phantom `dangling_ref`).
- (C) `cross_cutting: true` exempts a spec type from
  `pyramid_adjacency`.
- (D) `MetricValue.applicable=False` propagates to MetricResult with
  `target_met=None` and `warning=False`.
- (E) `surface_when_missing: true` on optional fields produces a
  warning when the field is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spec_writer.agent.tool_registry import ToolContext
from spec_writer.agent.tools.read_spec import read_spec_tool
from spec_writer.engine.id_registry import IdRegistry
from spec_writer.engine.project_context import ProjectContext
from spec_writer.io.spec_io import write_spec
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.metrics import (
    MetricContext,
    MetricResult,
    MetricValue,
    register_metric,
)
from spec_writer.methodology.metrics.runner import run_metrics
from spec_writer.methodology.sources import FilesystemMixinSource
from spec_writer.methodology.validators import ValidatorContext
from spec_writer.methodology.validators.builtin import (
    check_pyramid_adjacency,
    check_surface_when_missing,
)
from spec_writer.methodology.validators.runner import run_validators
from spec_writer.types import Spec

# ── shared fixtures ──────────────────────────────────────────────────────


def _methodology(tmp_path: Path, *, cross_cutting: bool = False, surface: bool = False):
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
    cc_line = "    cross_cutting: true\n" if cross_cutting else ""
    surface_block = (
        "    optional_frontmatter:\n"
        "      - priority:\n"
        '          description: "MoSCoW"\n'
        "          surface_when_missing: true\n"
        if surface
        else ""
    )
    (mixins / "conv.yaml").write_text(
        "mixin: conv\nadds_types:\n  - id: convention\n    layer_role: code-boundary\n    prefix: CONV\n"
        + cc_line
        + surface_block,
        encoding="utf-8",
    )
    (mixins / "core.yaml").write_text(
        "mixin: core\n"
        "validators:\n"
        "  - id: dangling-ref\n    scope: cross-spec\n    func_ref: validators.check_dangling_refs\n"
        "  - id: pyramid-adjacency\n    scope: per-spec\n    func_ref: validators.check_pyramid_adjacency\n"
        "  - id: surface-when-missing\n    scope: per-spec\n    func_ref: validators.check_surface_when_missing\n",
        encoding="utf-8",
    )
    return load_methodology(
        base, [FilesystemMixinSource(mixins)], ["vision", "cap", "conv", "core"]
    )


def _spec(
    spec_id: str, type_: str, layer: str, *, up_to: list[str] | None = None, **extras: Any
) -> Spec:
    return Spec(id=spec_id, type=type_, layer=layer, title="x", up_to=up_to or [], **extras)


# ── (A) read_spec MCP tool ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_spec_returns_frontmatter_and_body(tmp_path: Path) -> None:
    methodology = _methodology(tmp_path / "m")
    (tmp_path / "spec-editor" / "sources-raw").mkdir(parents=True)
    (tmp_path / "spec-editor" / "sources").mkdir()
    (tmp_path / "spec-editor" / "specs").mkdir()
    (tmp_path / "spec-editor" / "_meta").mkdir()
    project = ProjectContext.from_cwd(tmp_path)

    spec = _spec("L1-VIS-001", "vision", "L1", body="## Purpose\n\nhello world\n")
    write_spec(spec, methodology, project)

    ctx = ToolContext(project_ctx=project, methodology=methodology, id_registry=IdRegistry())
    result = await read_spec_tool({"spec_id": "L1-VIS-001"}, ctx)

    assert result["ok"] is True
    assert result["spec_id"] == "L1-VIS-001"
    assert result["type"] == "vision"
    assert result["layer"] == "L1"
    assert "hello world" in result["body"]
    assert result["frontmatter"]["id"] == "L1-VIS-001"


@pytest.mark.asyncio
async def test_read_spec_missing_returns_error(tmp_path: Path) -> None:
    methodology = _methodology(tmp_path / "m")
    (tmp_path / "spec-editor" / "sources-raw").mkdir(parents=True)
    (tmp_path / "spec-editor" / "sources").mkdir()
    (tmp_path / "spec-editor" / "specs").mkdir()
    (tmp_path / "spec-editor" / "_meta").mkdir()
    project = ProjectContext.from_cwd(tmp_path)
    ctx = ToolContext(project_ctx=project, methodology=methodology, id_registry=IdRegistry())

    result = await read_spec_tool({"spec_id": "L1-VIS-999"}, ctx)
    assert result["ok"] is False
    assert "not found" in result["error"]


# ── () read_spec serializes date/datetime/UUID safely ────────────


@pytest.mark.asyncio
async def test_read_spec_serializes_date_typed_extras_as_iso_string(
    tmp_path: Path,
) -> None:
    """: YAML date/datetime extras come back as ISO-format strings."""
    import datetime
    from uuid import UUID

    methodology = _methodology(tmp_path / "m")
    (tmp_path / "spec-editor" / "sources-raw").mkdir(parents=True)
    (tmp_path / "spec-editor" / "sources").mkdir()
    (tmp_path / "spec-editor" / "specs").mkdir()
    (tmp_path / "spec-editor" / "_meta").mkdir()
    project = ProjectContext.from_cwd(tmp_path)

    # Write a spec carrying a date-typed extra (mimics ARCH-001 accepted_exceptions
    # case that broke JSON serialization in Phase 2).
    spec_path = tmp_path / "spec-editor" / "specs" / "vision" / "L1-VIS-001-x.md"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        (
            "---\n"
            "id: L1-VIS-001\n"
            "type: vision\n"
            "layer: L1\n"
            "title: X\n"
            "decision_record_date: 2026-05-22\n"
            "decision_record_time: 2026-05-22T14:30:00\n"
            "decision_uuid: 12345678-1234-5678-1234-567812345678\n"
            "---\n"
            "## Purpose\n\nhello\n"
        ),
        encoding="utf-8",
    )

    ctx = ToolContext(project_ctx=project, methodology=methodology, id_registry=IdRegistry())
    result = await read_spec_tool({"spec_id": "L1-VIS-001"}, ctx)

    # Must succeed (the bug was that this failed with serialization error).
    assert result["ok"] is True
    fm = result["frontmatter"]
    # date/datetime/UUID are now strings — JSON-safe for MCP wire.
    assert fm["decision_record_date"] == "2026-05-22"
    assert fm["decision_record_time"] == "2026-05-22T14:30:00"
    assert fm["decision_uuid"] == "12345678-1234-5678-1234-567812345678"
    # And the whole payload is round-tripable through json (the actual MCP
    # wire concern — would fail without _json_safe).
    import json

    json.dumps(result)  # must not raise

    # Sanity: prove a date value would have broken json without the fix.
    raw_date = datetime.date(2026, 5, 22)
    raw_uuid = UUID("12345678-1234-5678-1234-567812345678")
    with pytest.raises(TypeError):
        json.dumps({"d": raw_date})
    with pytest.raises(TypeError):
        json.dumps({"u": raw_uuid})


# ── (B) scoped validate with all_specs keeps refs resolvable ─────────────


def test_run_validators_scoped_without_all_specs_flags_dangling_cross_layer(tmp_path: Path) -> None:
    """Backward-compat: caller passes scoped subset only — cross-layer
    refs DO look dangling. This is the pre-fix behavior."""
    methodology = _methodology(tmp_path / "m")
    cap = _spec("L2-CAP-001", "capability", "L2", up_to=["L1-VIS-001"])

    # Validating only L2 — without all_specs, ctx.corpus = [cap], so
    # `up_to: [L1-VIS-001]` looks dangling.
    issues = run_validators([cap], methodology)
    dangling = [i for i in issues if i.code == "dangling_ref"]
    assert dangling, "expected phantom dangling_ref without all_specs (backward-compat)"


def test_run_validators_scoped_with_all_specs_resolves_cross_layer(tmp_path: Path) -> None:
    """Fix #2: pass full corpus as all_specs — cross-layer edges resolve."""
    methodology = _methodology(tmp_path / "m")
    vis = _spec("L1-VIS-001", "vision", "L1", up_to=[])
    cap = _spec("L2-CAP-001", "capability", "L2", up_to=["L1-VIS-001"])

    issues = run_validators([cap], methodology, all_specs=[vis, cap])
    dangling = [i for i in issues if i.code == "dangling_ref"]
    assert not dangling, f"got phantom dangling_refs even with all_specs: {dangling}"


# ── (C) cross_cutting exempts pyramid_adjacency ──────────────────────────


def test_pyramid_adjacency_normal_type_flags_skip(tmp_path: Path) -> None:
    methodology = _methodology(tmp_path / "m", cross_cutting=False)
    vis = _spec("L1-VIS-001", "vision", "L1")
    # convention is L4 (code-boundary); pointing up_to L1 vision skips L2 + L3.
    conv = _spec("L4-CONV-001", "convention", "L4", up_to=["L1-VIS-001"])

    ctx = ValidatorContext(corpus={vis.id: vis, conv.id: conv}, methodology=methodology)
    issues = check_pyramid_adjacency(conv, ctx)
    skip = [i for i in issues if i.code == "pyramid_skip_level"]
    assert skip, "expected pyramid_skip_level on a non-cross-cutting L4→L1 skip"


def test_pyramid_adjacency_cross_cutting_type_exempt(tmp_path: Path) -> None:
    methodology = _methodology(tmp_path / "m", cross_cutting=True)
    vis = _spec("L1-VIS-001", "vision", "L1")
    conv = _spec("L4-CONV-001", "convention", "L4", up_to=["L1-VIS-001"])

    ctx = ValidatorContext(corpus={vis.id: vis, conv.id: conv}, methodology=methodology)
    issues = check_pyramid_adjacency(conv, ctx)
    assert issues == [], f"cross_cutting type should be exempt, got: {issues}"


# ── () up_to_types is consulted by pyramid validator ───────────────


def _methodology_with_up_to_types(tmp_path: Path):
    """Methodology where 'convention' declares up_to_types: [capability, vision].

    Mirrors the tech-stack pattern where the L4 type explicitly opts in
    to anchoring at non-immediate-parent types.
    """
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
    (mixins / "bhv.yaml").write_text(
        "mixin: bhv\nadds_types:\n  - id: behavior\n    layer_name: behaviors\n    prefix: BHV\n",
        encoding="utf-8",
    )
    (mixins / "conv.yaml").write_text(
        "mixin: conv\n"
        "adds_types:\n"
        "  - id: convention\n"
        "    layer_role: code-boundary\n"
        "    prefix: CONV\n"
        "    up_to_types: [capability, vision]\n",
        encoding="utf-8",
    )
    (mixins / "core.yaml").write_text(
        "mixin: core\n"
        "validators:\n"
        "  - id: pyramid-adjacency\n    scope: per-spec\n    func_ref: validators.check_pyramid_adjacency\n",
        encoding="utf-8",
    )
    return load_methodology(
        base, [FilesystemMixinSource(mixins)], ["vision", "cap", "bhv", "conv", "core"]
    )


def test_pyramid_adjacency_up_to_types_allows_non_adjacent_parent(tmp_path: Path) -> None:
    """: type declares up_to_types: [capability, vision] → L4→L2 is OK."""
    methodology = _methodology_with_up_to_types(tmp_path / "m")
    cap = _spec("L2-CAP-001", "capability", "L2", up_to=[])
    # L4 → L2 (capability) — skips L3, but allowed because up_to_types names capability.
    conv = _spec("L4-CONV-001", "convention", "L4", up_to=["L2-CAP-001"])

    ctx = ValidatorContext(corpus={cap.id: cap, conv.id: conv}, methodology=methodology)
    issues = check_pyramid_adjacency(conv, ctx)
    assert issues == [], (
        f"up_to_types=[capability,vision] should allow L4→L2 (capability), got: {issues}"
    )


def test_pyramid_adjacency_up_to_types_rejects_wrong_type(tmp_path: Path) -> None:
    """: parent type NOT in up_to_types → pyramid_wrong_parent_type."""
    methodology = _methodology_with_up_to_types(tmp_path / "m")
    bhv = _spec("L3-BHV-001", "behavior", "L3", up_to=[])
    # L4 → L3 (behavior) — adjacent layer, BUT up_to_types only lists
    # [capability, vision], so this is wrong-type.
    conv = _spec("L4-CONV-001", "convention", "L4", up_to=["L3-BHV-001"])

    ctx = ValidatorContext(corpus={bhv.id: bhv, conv.id: conv}, methodology=methodology)
    issues = check_pyramid_adjacency(conv, ctx)
    wrong = [i for i in issues if i.code == "pyramid_wrong_parent_type"]
    assert wrong, (
        f"expected pyramid_wrong_parent_type when parent.type not in up_to_types; got: {issues}"
    )


def test_pyramid_adjacency_no_up_to_types_falls_back_to_adjacency(tmp_path: Path) -> None:
    """: when up_to_types is empty, fallback to immediate-prior-layer rule."""
    # Use the default helper (no up_to_types declared on convention).
    methodology = _methodology(tmp_path / "m", cross_cutting=False)
    vis = _spec("L1-VIS-001", "vision", "L1")
    # L4→L1 — should still fail with pyramid_skip_level under fallback.
    conv = _spec("L4-CONV-001", "convention", "L4", up_to=["L1-VIS-001"])

    ctx = ValidatorContext(corpus={vis.id: vis, conv.id: conv}, methodology=methodology)
    issues = check_pyramid_adjacency(conv, ctx)
    skip = [i for i in issues if i.code == "pyramid_skip_level"]
    assert skip, "fallback to pyramid_skip_level expected when up_to_types undeclared"


# ── (D) MetricValue.applicable propagates ────────────────────────────────


_APPLICABLE_FALSE_TARGET = 1.0
_APPLICABLE_FALSE_WARN = 0.8


@register_metric("metrics.test_applicable_false")
def _metric_inapplicable(specs: list[Spec], ctx: MetricContext) -> MetricValue:
    """Metric impl that signals applicable=False (denominator empty)."""
    del specs, ctx
    return MetricValue(
        value=0.0,
        kind="ratio",
        details={"n": 0, "reason": "denominator source missing"},
        applicable=False,
    )


def test_metric_applicable_false_suppresses_warning(tmp_path: Path) -> None:
    """Fix #4: applicable=False → target_met=None, warning=False, value preserved."""
    methodology = _methodology(tmp_path / "m")
    # Inject a config that uses our test metric
    methodology.metrics_by_layer["L3"] = [
        {
            "id": "test-applicable-false",
            "layer": "L3",
            "func_ref": "metrics.test_applicable_false",
            "target": _APPLICABLE_FALSE_TARGET,
            "warn_below": _APPLICABLE_FALSE_WARN,
        }
    ]
    results = run_metrics([], methodology, layer_ids=["L3"])
    assert len(results) == 1
    r = results[0]
    assert r.value == 0.0, "value should still be shown for diagnostics"
    assert r.target_met is None, "target_met must be None when applicable=False"
    assert r.warning is False, "warning must be suppressed when applicable=False"
    assert r.applicable is False
    assert r.details and r.details.get("n") == 0


@register_metric("metrics.test_applicable_true")
def _metric_applicable(specs: list[Spec], ctx: MetricContext) -> MetricValue:
    del specs, ctx
    return MetricValue(value=0.5, kind="ratio", details={"x": 1}, applicable=True)


def test_metric_applicable_true_normal_behavior(tmp_path: Path) -> None:
    methodology = _methodology(tmp_path / "m")
    methodology.metrics_by_layer["L3"] = [
        {
            "id": "test-applicable-true",
            "layer": "L3",
            "func_ref": "metrics.test_applicable_true",
            "target": 1.0,
            "warn_below": 0.8,
        }
    ]
    results = run_metrics([], methodology, layer_ids=["L3"])
    r = results[0]
    assert r.value == 0.5
    assert r.target_met is False  # 0.5 < 1.0
    assert r.warning is True  # 0.5 < 0.8
    assert r.applicable is True


# ── (E) surface_when_missing emits warning when optional field absent ────


def test_surface_when_missing_emits_warning(tmp_path: Path) -> None:
    methodology = _methodology(tmp_path / "m", surface=True)
    # convention without priority field
    conv = _spec("L4-CONV-001", "convention", "L4", up_to=[])

    ctx = ValidatorContext(corpus={conv.id: conv}, methodology=methodology)
    issues = check_surface_when_missing(conv, ctx)
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].code == "surfaced_missing_optional_frontmatter"
    assert issues[0].field == "priority"


def test_surface_when_missing_silent_when_present(tmp_path: Path) -> None:
    methodology = _methodology(tmp_path / "m", surface=True)
    conv = _spec("L4-CONV-001", "convention", "L4", up_to=[], priority="must")

    ctx = ValidatorContext(corpus={conv.id: conv}, methodology=methodology)
    issues = check_surface_when_missing(conv, ctx)
    assert issues == [], f"should be silent when field is present: {issues}"


def test_surface_when_missing_ignores_fields_without_flag(tmp_path: Path) -> None:
    methodology = _methodology(tmp_path / "m", surface=False)
    conv = _spec("L4-CONV-001", "convention", "L4", up_to=[])

    ctx = ValidatorContext(corpus={conv.id: conv}, methodology=methodology)
    issues = check_surface_when_missing(conv, ctx)
    assert issues == [], "no surface_when_missing → no warnings"


# keep type-checker happy
_ = MetricResult
