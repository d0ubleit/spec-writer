"""S2.9 — chat REPL slash command dispatch + free-form fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spec_writer.agent.chat_session import ChatSession
from spec_writer.engine.project_context import ProjectContext
from spec_writer.interface.batch.cli import app
from spec_writer.interface.batch.commands.chat import (
    ChatContext,
    _cmd_help,
    _cmd_metrics,
    _cmd_session,
    _cmd_status,
    _cmd_validate,
    _extract_layers,
    _handle_slash,
)


def _runner() -> CliRunner:
    return CliRunner()


def _project(tmp_path: Path) -> ProjectContext:
    (tmp_path / "spec-editor").mkdir(exist_ok=True)
    for sub in ("sources-raw", "sources", "specs", "_meta"):
        (tmp_path / "spec-editor" / sub).mkdir(exist_ok=True)
    return ProjectContext.from_cwd(tmp_path)


def _chat_ctx(tmp_path: Path, layers: list[str] | None = None) -> ChatContext:
    ctx = _project(tmp_path)
    session = ChatSession(project_ctx=ctx)
    return ChatContext(
        project_ctx=ctx,
        session=session,
        default_layers=layers or [],
        max_iterations=5,
    )


# ── argument extraction ─────────────────────────────────────────────────


def test_extract_layers_simple() -> None:
    assert _extract_layers(["--layer", "L1"]) == ["L1"]


def test_extract_layers_multiple() -> None:
    assert _extract_layers(["--layer", "L1", "--layer", "L2"]) == ["L1", "L2"]


def test_extract_layers_ignores_unknown_flags() -> None:
    assert _extract_layers(["--layer", "L1", "--other"]) == ["L1"]


# ── slash command dispatch ──────────────────────────────────────────────


def test_handle_slash_unknown_command_logs_and_returns_true(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cctx = _chat_ctx(tmp_path)
    keep_going = _handle_slash(cctx, "/wibble")
    assert keep_going is True
    captured = capsys.readouterr()
    assert "unknown command" in captured.out


def test_handle_slash_quit_returns_false(tmp_path: Path) -> None:
    cctx = _chat_ctx(tmp_path)
    assert _handle_slash(cctx, "/quit") is False


def test_handle_slash_exit_returns_false(tmp_path: Path) -> None:
    cctx = _chat_ctx(tmp_path)
    assert _handle_slash(cctx, "/exit") is False


def test_handle_slash_help(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cctx = _chat_ctx(tmp_path)
    _handle_slash(cctx, "/help")
    out = capsys.readouterr().out
    assert "/help" in out
    assert "/generate" in out
    assert "/quit" in out


def test_handle_slash_logs_to_session(tmp_path: Path) -> None:
    cctx = _chat_ctx(tmp_path)
    _handle_slash(cctx, "/help")
    payload = json.loads(cctx.session.path.read_text(encoding="utf-8"))
    kinds = [e["kind"] for e in payload["events"]]
    assert "slash-command" in kinds


# ── individual handlers ─────────────────────────────────────────────────


def test_cmd_status_reports_corpus_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cctx = _chat_ctx(tmp_path)
    _cmd_status(cctx, [])
    out = capsys.readouterr().out
    assert "corpus: 0 spec" in out


def test_cmd_session_prints_id_and_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cctx = _chat_ctx(tmp_path)
    _cmd_session(cctx, [])
    out = capsys.readouterr().out
    assert cctx.session.session_id in out


def test_cmd_help_lists_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cctx = _chat_ctx(tmp_path)
    _cmd_help(cctx, [])
    out = capsys.readouterr().out
    for cmd in ("/help", "/generate", "/refine", "/quit"):
        assert cmd in out


def _bootstrap_minimal_project(tmp_path: Path) -> None:
    """Write a minimal methodology.yaml so /validate / /metrics can load."""
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


def test_cmd_validate_runs_on_empty_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _bootstrap_minimal_project(tmp_path)
    cctx = _chat_ctx(tmp_path)
    # Replace project_ctx so it picks up the bootstrapped methodology
    cctx.project_ctx = ProjectContext.from_cwd(tmp_path)
    cctx.session = ChatSession(project_ctx=cctx.project_ctx)
    _cmd_validate(cctx, [])
    out = capsys.readouterr().out
    assert "error(s)" in out


def test_cmd_metrics_runs_on_empty_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _bootstrap_minimal_project(tmp_path)
    cctx = _chat_ctx(tmp_path)
    cctx.project_ctx = ProjectContext.from_cwd(tmp_path)
    cctx.session = ChatSession(project_ctx=cctx.project_ctx)
    _cmd_metrics(cctx, [])
    # minimal preset's core-traceability mixin declares completeness metrics
    out = capsys.readouterr().out
    assert "completeness" in out or "no metrics" in out


# ── full CLI smoke ──────────────────────────────────────────────────────


def test_chat_cli_quits_cleanly_on_quit_command(tmp_path: Path) -> None:
    _bootstrap_minimal_project(tmp_path)
    runner = _runner()
    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat"],
        input="/quit\n",
    )
    assert result.exit_code == 0
    assert "chat session" in result.output


def test_chat_cli_help_command(tmp_path: Path) -> None:
    _bootstrap_minimal_project(tmp_path)
    runner = _runner()
    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat"],
        input="/help\n/quit\n",
    )
    assert result.exit_code == 0
    assert "/generate" in result.output
    assert "/quit" in result.output


def test_chat_cli_writes_session_file(tmp_path: Path) -> None:
    _bootstrap_minimal_project(tmp_path)
    runner = _runner()
    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat"],
        input="/help\n/quit\n",
    )
    assert result.exit_code == 0
    sessions_dir = tmp_path / "spec-editor" / "_meta" / "sessions"
    assert sessions_dir.exists()
    files = list(sessions_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    kinds = [e["kind"] for e in payload["events"]]
    assert "slash-command" in kinds
