"""S3.9 — /add-feature slash command (incremental-add mode)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from spec_writer.agent.chat_session import ChatSession
from spec_writer.engine.project_context import ProjectContext
from spec_writer.interface.batch.cli import app
from spec_writer.interface.batch.commands.chat import ChatContext, _cmd_add_feature


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


def _chat_ctx(tmp_path: Path, layers: list[str] | None = None) -> ChatContext:
    project_ctx = ProjectContext.from_cwd(tmp_path)
    session = ChatSession(project_ctx=project_ctx)
    return ChatContext(
        project_ctx=project_ctx,
        session=session,
        default_layers=layers or [],
        max_iterations=5,
    )


def test_add_feature_without_description_prints_usage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path, layers=["L2"])
    rc = _cmd_add_feature(cctx, ["--layer", "L2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "usage" in out


def test_add_feature_without_layers_prints_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path)
    rc = _cmd_add_feature(cctx, ["build", "an", "auth", "feature"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no layers" in out


def test_add_feature_invokes_pipeline_with_instruction(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path, layers=["L2"])

    captured: dict[str, object] = {}

    def fake_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        # Return an empty success-ish result to avoid downstream raise
        from spec_writer.engine.loop import PipelineResult

        return PipelineResult(succeeded=True, layer_results=[])

    with patch(
        "spec_writer.interface.batch.commands.chat.run_pipeline_for",
        side_effect=fake_pipeline,
    ):
        _cmd_add_feature(cctx, ["--layer", "L2", "add", "user", "profile", "page"])

    assert captured["mode"] == "incremental-add"
    assert captured["layers"] == ["L2"]
    assert captured["instruction"] == "add user profile page"


def test_add_feature_uses_default_layers_when_no_flag(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path, layers=["L2", "L3"])
    captured: dict[str, object] = {}

    def fake_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        from spec_writer.engine.loop import PipelineResult

        return PipelineResult(succeeded=True, layer_results=[])

    with patch(
        "spec_writer.interface.batch.commands.chat.run_pipeline_for",
        side_effect=fake_pipeline,
    ):
        _cmd_add_feature(cctx, ["sign", "out", "flow"])

    assert captured["layers"] == ["L2", "L3"]
    assert captured["instruction"] == "sign out flow"


def test_add_feature_logs_slash_event(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path, layers=["L2"])
    with patch(
        "spec_writer.interface.batch.commands.chat.run_pipeline_for",
        return_value=None,
    ):
        try:
            from spec_writer.engine.loop import PipelineResult

            with patch(
                "spec_writer.interface.batch.commands.chat.run_pipeline_for",
                return_value=PipelineResult(succeeded=True, layer_results=[]),
            ):
                # Use _handle_slash to exercise session logging path
                from spec_writer.interface.batch.commands.chat import _handle_slash

                _handle_slash(cctx, "/add-feature --layer L2 dark mode")
        finally:
            pass
    payload = json.loads(cctx.session.path.read_text(encoding="utf-8"))
    slashes = [e for e in payload["events"] if e["kind"] == "slash-command"]
    assert any(e["command"] == "/add-feature" for e in slashes)


def test_add_feature_in_help_output(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    runner = _runner()
    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat"],
        input="/help\n/quit\n",
    )
    assert "/add-feature" in result.output
