"""S4.6 — REV-direction guidance in ChatAgent + scripted REV flow."""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_writer.agent.chat_agent import ChatAgent, build_chat_system_prompt
from spec_writer.agent.chat_session import ChatSession
from spec_writer.agent.events import TextDeltaEvent, ToolCallEvent
from spec_writer.agent.providers.fake import FakeLLMClient, ScriptedResponse
from spec_writer.engine.project_context import ProjectContext
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.sources import FilesystemMixinSource


def _project(tmp_path: Path) -> ProjectContext:
    (tmp_path / "spec-editor").mkdir(exist_ok=True)
    for sub in ("sources-raw", "sources", "specs", "_meta"):
        (tmp_path / "spec-editor" / sub).mkdir(exist_ok=True)
    return ProjectContext.from_cwd(tmp_path)


def _methodology(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir(exist_ok=True)
    (meta / "base.yaml").write_text(
        (
            "version: 1\n"
            "layers:\n"
            "  - id: L1\n    role: source-boundary\n    name: vision\n"
            "  - id: L4\n    role: code-boundary\n    name: contracts\n"
        ),
        encoding="utf-8",
    )
    mixins = meta / "mixins"
    mixins.mkdir(exist_ok=True)
    (mixins / "v.yaml").write_text(
        "mixin: v\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins / "api.yaml").write_text(
        "mixin: api\nadds_types:\n  - id: api-contract\n    layer_role: code-boundary\n    prefix: API\n",
        encoding="utf-8",
    )
    return load_methodology(
        meta / "base.yaml", [FilesystemMixinSource(mixins)], active_mixins=["v", "api"]
    )


# ── system prompt direction blocks ──────────────────────────────────────


def test_fwd_prompt_says_top_down(tmp_path: Path) -> None:
    prompt = build_chat_system_prompt(_methodology(tmp_path), _project(tmp_path), "fwd")
    assert "FWD" in prompt
    assert "top-down" in prompt
    assert "list_sources" in prompt
    assert "REV" not in prompt or "REV (code → sources)" not in prompt


def test_rev_prompt_says_bottom_up(tmp_path: Path) -> None:
    prompt = build_chat_system_prompt(_methodology(tmp_path), _project(tmp_path), "rev")
    assert "REV" in prompt
    assert "bottom-up" in prompt
    assert "discover_code_inventory" in prompt
    assert "implemented_by" in prompt
    assert "coverage_report" in prompt


def test_rev_prompt_mentions_warn_below_ask_user_rule(tmp_path: Path) -> None:
    prompt = build_chat_system_prompt(_methodology(tmp_path), _project(tmp_path), "rev")
    assert "warn_below" in prompt
    assert "ask_user" in prompt


def test_default_direction_is_fwd(tmp_path: Path) -> None:
    default_prompt = build_chat_system_prompt(_methodology(tmp_path), _project(tmp_path))
    assert "FWD" in default_prompt


# ── ChatAgent direction wiring ──────────────────────────────────────────


def test_chat_agent_default_direction_uses_fwd_block(tmp_path: Path) -> None:
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    session = ChatSession(project_ctx=project)
    llm = FakeLLMClient([])

    agent = ChatAgent(llm, project, methodology, session)
    assert "FWD" in agent.history[0].content


def test_chat_agent_rev_direction_uses_rev_block(tmp_path: Path) -> None:
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    session = ChatSession(project_ctx=project)
    llm = FakeLLMClient([])

    agent = ChatAgent(llm, project, methodology, session, direction="rev")
    assert "REV" in agent.history[0].content
    assert "bottom-up" in agent.history[0].content


# ── scripted REV flow ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rev_scripted_flow_inventory_then_record(tmp_path: Path) -> None:
    """Agent in REV mode discovers code, records an L4 spec, calls coverage_report."""
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    session = ChatSession(project_ctx=project)

    llm = FakeLLMClient(
        [
            ScriptedResponse(
                tool_calls=[
                    ("discover_code_inventory", {}),
                    (
                        "record_spec",
                        {
                            "type": "api-contract",
                            "layer": "L4",
                            "title": "Login Endpoint",
                            "implemented_by": ["src/auth/login.py:1-10"],
                        },
                    ),
                    ("coverage_report", {}),
                ],
                text="Discovered code, wrote L4-API-001, checked coverage.",
            )
        ]
    )

    agent = ChatAgent(llm, project, methodology, session, direction="rev")
    events = [event async for event in agent.handle_user("Start REV extraction")]

    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    tool_names = [tc.tool for tc in tool_calls]
    assert "discover_code_inventory" in tool_names
    assert "record_spec" in tool_names
    assert "coverage_report" in tool_names
    assert any(isinstance(e, TextDeltaEvent) for e in events)


# ── CLI flag plumbing ───────────────────────────────────────────────────


def test_chat_cli_rejects_invalid_direction(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from spec_writer.interface.batch.cli import app

    runner = CliRunner()
    runner.invoke(
        app,
        ["--project-root", str(tmp_path), "init", "--no-wizard", "--preset", "minimal"],
    )
    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat", "--direction", "sideways"],
        input="/quit\n",
    )
    assert result.exit_code == 2
    assert "fwd" in result.output and "rev" in result.output


def test_chat_cli_accepts_rev_direction(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from spec_writer.interface.batch.cli import app

    runner = CliRunner()
    runner.invoke(
        app,
        ["--project-root", str(tmp_path), "init", "--no-wizard", "--preset", "minimal"],
    )
    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat", "--direction", "rev"],
        input="/quit\n",
    )
    assert result.exit_code == 0


def test_chat_cli_help_lists_direction(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from spec_writer.interface.batch.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--direction" in result.output
