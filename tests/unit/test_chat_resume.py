"""S3.13 — `spec-writer chat --resume <id>` rehydrates history."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from spec_writer.agent.chat_session import ChatSession
from spec_writer.engine.project_context import ProjectContext
from spec_writer.interface.batch.cli import app


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


def test_replay_history_tail_filters_to_user_and_agent(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    project_ctx = ProjectContext.from_cwd(tmp_path)
    session = ChatSession(project_ctx=project_ctx)
    session.log_user("first user message")
    session.log_agent("first reply")
    session.log_tool_call("validate", {})
    session.log_tool_result("validate", {"ok": True, "issues": []})
    session.log_note("methodology_loaded")
    session.log_user("second user message")
    session.log_agent("second reply")

    from spec_writer.agent.chat_agent_factory import replay_history_tail

    tail = replay_history_tail(session)
    # Tool calls / results / notes are skipped; only user + agent remain
    assert [m.role for m in tail] == ["user", "assistant", "user", "assistant"]
    assert tail[0].content == "first user message"
    assert tail[2].content == "second user message"


def test_resume_unknown_session_id_exits_12(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    runner = _runner()
    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat", "--resume", "non-existent-id"],
        input="/quit\n",
    )
    assert result.exit_code == 12
    assert "not found" in result.output


def test_resume_existing_session_succeeds(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    project_ctx = ProjectContext.from_cwd(tmp_path)
    session = ChatSession(project_ctx=project_ctx)
    session.log_user("hi")
    session.log_agent("hello back")
    session_id = session.session_id

    runner = _runner()
    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat", "--resume", session_id],
        input="/quit\n",
    )
    assert result.exit_code == 0
    assert "resumed" in result.output
    assert session_id in result.output


def test_resume_chat_loads_methodology(tmp_path: Path) -> None:
    """Smoke: starting chat (resume) should bootstrap methodology without crash."""
    _bootstrap(tmp_path)
    project_ctx = ProjectContext.from_cwd(tmp_path)
    session = ChatSession(project_ctx=project_ctx)
    session.log_user("anything")
    runner = _runner()
    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat", "--resume", session.session_id],
        input="/status\n/quit\n",
    )
    # Even if methodology is loaded, /status uses corpus walking which
    # works on empty specs/.
    assert result.exit_code == 0


def test_resume_in_help(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    runner = _runner()
    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat", "--help"],
    )
    assert result.exit_code == 0
    assert "--resume" in result.output


def test_resume_argument_in_subcommand_signature() -> None:
    """Ensure `chat` accepts --resume via Typer (not just optional)."""
    # Inspect runtime by looking at typer option markers — quick & narrow.
    import inspect

    from spec_writer.interface.batch.commands.chat import run

    sig = inspect.signature(run)
    assert "resume" in sig.parameters


# ── streaming sanity (FakeLLM path) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_chat_agent_yields_events_in_order(tmp_path: Path) -> None:
    """`_stream_chat_agent` consumes the agent's stream and writes to stdout."""
    # Indirect — we exercise ChatAgent directly to verify event order;
    # the CLI side is hard to assert without monkey-patching asyncio.run.
    from spec_writer.agent.chat_agent import ChatAgent
    from spec_writer.agent.events import TextDeltaEvent, ToolCallEvent
    from spec_writer.agent.providers.fake import FakeLLMClient, ScriptedResponse
    from spec_writer.methodology.loader import load_methodology
    from spec_writer.methodology.sources import FilesystemMixinSource

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "base.yaml").write_text(
        "version: 1\nlayers:\n  - id: L1\n    role: source-boundary\n    name: vision\n",
        encoding="utf-8",
    )
    mixins = meta / "mixins"
    mixins.mkdir()
    (mixins / "v.yaml").write_text(
        "mixin: v\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    methodology = load_methodology(
        meta / "base.yaml", [FilesystemMixinSource(mixins)], active_mixins=["v"]
    )

    project_dir = tmp_path / "proj"
    (project_dir / "spec-editor").mkdir(parents=True)
    for sub in ("sources-raw", "sources", "specs", "_meta"):
        (project_dir / "spec-editor" / sub).mkdir()
    project = ProjectContext.from_cwd(project_dir)

    session = ChatSession(project_ctx=project)
    llm = FakeLLMClient(
        [
            ScriptedResponse(
                tool_calls=[("list_sources", {})],
                text="Looked.",
            )
        ]
    )
    agent = ChatAgent(llm, project, methodology, session)

    events = [event async for event in agent.handle_user("scan sources")]
    types = [type(e).__name__ for e in events]
    assert "ToolCallEvent" in types
    assert "TextDeltaEvent" in types
    # Order: tool call first, then text
    assert types.index("ToolCallEvent") < types.index("TextDeltaEvent")
    # Avoid unused-symbol warnings from importing for type-checking
    _ = ToolCallEvent, TextDeltaEvent
