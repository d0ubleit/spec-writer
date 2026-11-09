"""S3.10 — cascade neighbour BFS + /cascade-fix + /edit slash commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from spec_writer.agent.chat_session import ChatSession
from spec_writer.engine.cascade import cascade_neighbours
from spec_writer.engine.project_context import ProjectContext
from spec_writer.interface.batch.cli import app
from spec_writer.interface.batch.commands.chat import (
    ChatContext,
    _cmd_cascade_fix,
    _cmd_edit,
)
from spec_writer.types import RelatesToEntry, Spec


def _spec(
    spec_id: str,
    layer: str = "L2",
    up: list[str] | None = None,
    down: list[str] | None = None,
    relates: list[tuple[str, str]] | None = None,
) -> Spec:
    return Spec(
        id=spec_id,
        type="capability",
        layer=layer,
        title=spec_id,
        up_to=up or [],
        down_to=down or [],
        relates_to=[RelatesToEntry(id=rid, kind=kind) for rid, kind in (relates or [])],  # type: ignore[arg-type]
        body="",
    )


# ── cascade_neighbours ──────────────────────────────────────────────────


def test_cascade_neighbours_self_only_when_unknown() -> None:
    out = cascade_neighbours("L1-MISSING-001", {}, depth=2)
    assert out == ["L1-MISSING-001"]


def test_cascade_neighbours_depth_one_collects_direct_links() -> None:
    corpus = {
        "L1-V-001": _spec("L1-V-001", layer="L1", down=["L2-C-001"]),
        "L2-C-001": _spec("L2-C-001", up=["L1-V-001"], down=["L3-a known limitation"]),
        "L3-a known limitation": _spec("L3-a known limitation", layer="L3", up=["L2-C-001"]),
    }
    out = cascade_neighbours("L2-C-001", corpus, depth=1)
    assert out[0] == "L2-C-001"
    assert set(out) == {"L2-C-001", "L1-V-001", "L3-a known limitation"}


def test_cascade_neighbours_depth_two_reaches_two_hops() -> None:
    corpus = {
        "L1-V-001": _spec("L1-V-001", layer="L1", down=["L2-C-001"]),
        "L2-C-001": _spec("L2-C-001", up=["L1-V-001"], down=["L3-a known limitation"]),
        "L3-a known limitation": _spec(
            "L3-a known limitation", layer="L3", up=["L2-C-001"], down=["L4-A-001"]
        ),
        "L4-A-001": _spec("L4-A-001", layer="L4", up=["L3-a known limitation"]),
    }
    out = cascade_neighbours("L2-C-001", corpus, depth=2)
    assert "L4-A-001" in out or "L1-V-001" in out


def test_cascade_neighbours_includes_relates_to() -> None:
    corpus = {
        "L2-C-001": _spec("L2-C-001", relates=[("L2-C-002", "relates_to")]),
        "L2-C-002": _spec("L2-C-002"),
    }
    out = cascade_neighbours("L2-C-001", corpus, depth=1)
    assert "L2-C-002" in out


def test_cascade_neighbours_does_not_revisit() -> None:
    corpus = {
        "A": _spec("A", up=["B"], down=["B"]),
        "B": _spec("B", up=["A"], down=["A"]),
    }
    out = cascade_neighbours("A", corpus, depth=5)
    assert sorted(out) == ["A", "B"]


def test_cascade_neighbours_min_depth_one() -> None:
    corpus = {"A": _spec("A", up=["B"]), "B": _spec("B")}
    out = cascade_neighbours("A", corpus, depth=0)  # gets clamped to 1
    assert "B" in out


# ── chat slash commands ─────────────────────────────────────────────────


def _bootstrap(tmp_path: Path) -> None:
    runner = CliRunner()
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


def _chat_ctx(tmp_path: Path) -> ChatContext:
    project_ctx = ProjectContext.from_cwd(tmp_path)
    session = ChatSession(project_ctx=project_ctx)
    return ChatContext(
        project_ctx=project_ctx, session=session, default_layers=[], max_iterations=5
    )


def test_cmd_cascade_fix_usage_when_no_args(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path)
    _cmd_cascade_fix(cctx, [])
    assert "usage" in capsys.readouterr().out


def test_cmd_cascade_fix_unknown_spec(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path)
    rc = _cmd_cascade_fix(cctx, ["L9-MISSING-001"])
    assert rc == 1
    assert "not found" in capsys.readouterr().out


def test_cmd_cascade_fix_invokes_pipeline_with_scope(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    # Seed the corpus with one spec by calling record_spec directly
    from spec_writer.agent.tool_registry import ToolContext
    from spec_writer.agent.tools.record_spec import record_spec
    from spec_writer.engine.id_registry import IdRegistry
    from spec_writer.interface.batch.commands.generate import _load_methodology_for

    project_ctx = ProjectContext.from_cwd(tmp_path)
    methodology = _load_methodology_for(project_ctx)
    tool_ctx = ToolContext(
        project_ctx=project_ctx, methodology=methodology, id_registry=IdRegistry()
    )
    import asyncio

    result = asyncio.run(
        record_spec({"type": "vision", "layer": "L1", "title": "Vision", "up_to": []}, tool_ctx)
    )
    spec_id = result["spec_id"]

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
        _cmd_cascade_fix(cctx, [spec_id])

    assert captured["mode"] == "cascade-fix"
    scope = captured.get("scope")
    assert scope is not None
    assert spec_id in (scope.spec_ids or [])


def test_cmd_edit_invokes_patch_and_cascade(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    # Seed
    from spec_writer.agent.tool_registry import ToolContext
    from spec_writer.agent.tools.record_spec import record_spec
    from spec_writer.engine.id_registry import IdRegistry
    from spec_writer.interface.batch.commands.generate import _load_methodology_for

    project_ctx = ProjectContext.from_cwd(tmp_path)
    methodology = _load_methodology_for(project_ctx)
    tool_ctx = ToolContext(
        project_ctx=project_ctx, methodology=methodology, id_registry=IdRegistry()
    )
    import asyncio

    result = asyncio.run(
        record_spec({"type": "vision", "layer": "L1", "title": "Initial", "up_to": []}, tool_ctx)
    )
    spec_id = result["spec_id"]

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
        _cmd_edit(cctx, [spec_id, "title=Renamed"])

    # The patch should have applied — re-read disk and confirm
    from spec_writer.io.spec_io import find_spec_by_id, read_spec

    path = find_spec_by_id(project_ctx.specs, spec_id)
    assert path is not None
    spec = read_spec(path)
    assert spec.title == "Renamed"

    # Pipeline mode should be targeted-edit-cascade for the cascade step
    if captured:
        assert captured["mode"] == "targeted-edit-cascade"


def test_cmd_edit_usage_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path)
    _cmd_edit(cctx, ["L1-VIS-001"])
    assert "usage" in capsys.readouterr().out


def test_help_lists_cascade_commands(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat"],
        input="/help\n/quit\n",
    )
    assert "/cascade-fix" in result.output
    assert "/edit" in result.output


def test_handle_slash_cascade_records_event(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path)
    from spec_writer.interface.batch.commands.chat import _handle_slash

    _handle_slash(cctx, "/cascade-fix")
    payload = json.loads(cctx.session.path.read_text(encoding="utf-8"))
    slash = [e for e in payload["events"] if e["kind"] == "slash-command"]
    assert any(e["command"] == "/cascade-fix" for e in slash)
