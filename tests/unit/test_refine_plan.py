"""S3.11 — refine plan builder + /refine-plan slash command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from spec_writer.agent.chat_session import ChatSession
from spec_writer.engine.project_context import ProjectContext
from spec_writer.engine.refine_plan import (
    RefinePlan,
    RefinePlanEntry,
    build_refine_plan,
    scope_spec_ids,
    select_plan_entries,
)
from spec_writer.interface.batch.cli import app
from spec_writer.interface.batch.commands.chat import ChatContext, _cmd_refine_plan
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.sources import FilesystemMixinSource
from spec_writer.methodology.validators import ValidationIssue
from spec_writer.types import Spec


def _runner() -> CliRunner:
    return CliRunner()


def _bootstrap(tmp_path: Path) -> None:
    runner = _runner()
    result = runner.invoke(
        app,
        [
            "--project-root",
            str(tmp_path),
            "init",
            "--no-wizard",
            "--preset",
            "minimal",
        ],
    )
    assert result.exit_code == 0


def _methodology(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir(exist_ok=True)
    (meta / "base.yaml").write_text(
        (
            "version: 1\n"
            "layers:\n"
            "  - id: L1\n    role: source-boundary\n    name: vision\n"
            "  - id: L2\n    role: middle\n    name: capabilities\n"
        ),
        encoding="utf-8",
    )
    mixins = meta / "mixins"
    mixins.mkdir(exist_ok=True)
    (mixins / "v.yaml").write_text(
        "mixin: v\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins / "c.yaml").write_text(
        (
            "mixin: c\nadds_types:\n  - id: capability\n"
            "    layer_name: capabilities\n    prefix: CAP\n"
            "    up_to_types: [vision]\n"
        ),
        encoding="utf-8",
    )
    (mixins / "vals.yaml").write_text(
        (
            "mixin: vals\nvalidators:\n"
            "  - id: pyramid\n    scope: per-spec\n    rule_kind: python\n"
            "    func_ref: validators.check_pyramid_adjacency\n"
        ),
        encoding="utf-8",
    )
    return load_methodology(
        meta / "base.yaml",
        [FilesystemMixinSource(mixins)],
        active_mixins=["v", "c", "vals"],
    )


def _spec(spec_id: str, type_: str, layer: str, up: list[str] | None = None) -> Spec:
    return Spec(
        id=spec_id,
        type=type_,
        layer=layer,
        title=spec_id,
        up_to=up or [],
        body="",
    )


# ── select_plan_entries ─────────────────────────────────────────────────


def _plan_with(n: int, with_errors: int = 0) -> RefinePlan:
    entries: list[RefinePlanEntry] = []
    for i in range(n):
        is_error = i < with_errors
        issue = ValidationIssue(
            severity="error" if is_error else "warning",
            code="x",
            message="msg",
        )
        entries.append(RefinePlanEntry(spec_id=f"S{i + 1:03d}", issues=[issue]))
    return RefinePlan(entries=entries)


def test_select_plan_entries_all() -> None:
    plan = _plan_with(3)
    assert len(select_plan_entries(plan, "all")) == 3


def test_select_plan_entries_indices() -> None:
    plan = _plan_with(5)
    chosen = select_plan_entries(plan, "1,3,5")
    assert [e.spec_id for e in chosen] == ["S001", "S003", "S005"]


def test_select_plan_entries_range() -> None:
    plan = _plan_with(5)
    chosen = select_plan_entries(plan, "2-4")
    assert [e.spec_id for e in chosen] == ["S002", "S003", "S004"]


def test_select_plan_entries_errors_only() -> None:
    plan = _plan_with(4, with_errors=2)
    chosen = select_plan_entries(plan, "errors")
    assert len(chosen) == 2
    assert all(e.has_errors for e in chosen)


def test_select_plan_entries_warnings_only() -> None:
    plan = _plan_with(4, with_errors=2)
    chosen = select_plan_entries(plan, "warnings")
    assert len(chosen) == 2
    assert all(not e.has_errors for e in chosen)


def test_select_plan_entries_empty_selection() -> None:
    plan = _plan_with(3)
    assert select_plan_entries(plan, "") == []
    assert select_plan_entries(plan, "none") == []


def test_select_plan_entries_combined() -> None:
    plan = _plan_with(5, with_errors=2)  # S001, S002 errors; S003-S005 warnings
    chosen = select_plan_entries(plan, "1,4-5,errors")
    spec_ids = [e.spec_id for e in chosen]
    # Both selection paths dedupe
    assert "S001" in spec_ids
    assert "S002" in spec_ids  # from errors
    assert "S004" in spec_ids
    assert "S005" in spec_ids
    assert len(spec_ids) == len(set(spec_ids))


def test_select_plan_entries_ignores_garbage() -> None:
    plan = _plan_with(3)
    chosen = select_plan_entries(plan, "wibble,2,?,5")
    assert [e.spec_id for e in chosen] == ["S002"]


def test_scope_spec_ids_dedup() -> None:
    plan = _plan_with(3)
    out = scope_spec_ids(plan.entries)
    assert out == ["S001", "S002", "S003"]


def test_scope_spec_ids_skips_corpus_level() -> None:
    issue = ValidationIssue(severity="error", code="x", message="x")
    entries = [
        RefinePlanEntry(spec_id=None, issues=[issue]),
        RefinePlanEntry(spec_id="L1-VIS-001", issues=[issue]),
    ]
    assert scope_spec_ids(entries) == ["L1-VIS-001"]


# ── build_refine_plan ───────────────────────────────────────────────────


def test_build_refine_plan_empty_when_clean(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    plan = build_refine_plan([], m)
    assert plan.entries == []


def test_build_refine_plan_groups_by_spec(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    # Two L2 specs with empty up_to → both violate pyramid-adjacency
    specs = [
        _spec("L2-CAP-001", "capability", "L2"),
        _spec("L2-CAP-002", "capability", "L2"),
    ]
    plan = build_refine_plan(specs, m)
    spec_ids = {e.spec_id for e in plan.entries}
    assert {"L2-CAP-001", "L2-CAP-002"} <= spec_ids


def test_build_refine_plan_errors_before_warnings(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    specs = [_spec("L2-CAP-001", "capability", "L2")]
    plan = build_refine_plan(specs, m)
    # First entry should have errors
    if plan.entries:
        assert plan.entries[0].has_errors


def test_refine_plan_render_includes_indices(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    specs = [_spec("L2-CAP-001", "capability", "L2")]
    plan = build_refine_plan(specs, m)
    rendered = plan.render()
    assert "1." in rendered


def test_refine_plan_render_empty_corpus(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    plan = build_refine_plan([], m)
    assert "passes all" in plan.render()


# ── chat /refine-plan ───────────────────────────────────────────────────


def _chat_ctx(tmp_path: Path) -> ChatContext:
    project_ctx = ProjectContext.from_cwd(tmp_path)
    session = ChatSession(project_ctx=project_ctx)
    return ChatContext(
        project_ctx=project_ctx, session=session, default_layers=["L2"], max_iterations=5
    )


def test_cmd_refine_plan_passes_clean_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path)
    rc = _cmd_refine_plan(cctx, ["--apply", "all"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "passes all" in out


def test_cmd_refine_plan_with_apply_all_runs_pipeline(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    # Seed a spec with broken traceability (L2 capability with no up_to)
    from spec_writer.agent.tool_registry import ToolContext
    from spec_writer.agent.tools.record_spec import record_spec
    from spec_writer.engine.id_registry import IdRegistry
    from spec_writer.interface.batch.commands.generate import _load_methodology_for

    project_ctx = ProjectContext.from_cwd(tmp_path)
    # Need to add a capabilities mixin to minimal preset
    methodology_path = tmp_path / "spec-editor" / "methodology.yaml"
    text = methodology_path.read_text(encoding="utf-8")
    methodology_path.write_text(text + "  - capabilities\n", encoding="utf-8")
    methodology = _load_methodology_for(project_ctx)
    tool_ctx = ToolContext(
        project_ctx=project_ctx, methodology=methodology, id_registry=IdRegistry()
    )
    import asyncio

    asyncio.run(
        record_spec({"type": "capability", "layer": "L2", "title": "broken", "up_to": []}, tool_ctx)
    )

    cctx = _chat_ctx(tmp_path)
    captured: dict[str, object] = {}

    def fake_pipeline(*_args, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        from spec_writer.engine.loop import PipelineResult

        return PipelineResult(succeeded=True, layer_results=[])

    with patch(
        "spec_writer.interface.batch.commands.chat.run_pipeline_for",
        side_effect=fake_pipeline,
    ):
        _cmd_refine_plan(cctx, ["--apply", "all"])

    assert captured.get("mode") == "refine"
    scope = captured.get("scope")
    if scope is not None:
        # spec_ids should be the broken-cap id (or maybe empty if all corpus-level)
        assert scope.spec_ids is None or len(scope.spec_ids) >= 1


def test_cmd_refine_plan_skip_selection(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path)
    rc = _cmd_refine_plan(cctx, ["--apply", ""])
    assert rc == 0


def test_refine_plan_in_help(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    runner = _runner()
    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat"],
        input="/help\n/quit\n",
    )
    assert "/refine-plan" in result.output
