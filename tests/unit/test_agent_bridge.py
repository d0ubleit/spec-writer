"""Tests for the agent_bridge module — session state machine, MCP tools, manager.

The bridge cannot use real ``ChatAgent`` round-trips in unit tests
(those need Claude credentials and tokens), so we drive ``handle_user``
with custom :class:`FakeLLMClient` subclasses that synthesise the
ask_user / no-ask_user / error scenarios we care about.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from spec_writer.agent.chat_agent import ChatAgent
from spec_writer.agent.chat_session import ChatSession
from spec_writer.agent.events import EngineEvent, StatusEvent, TextDeltaEvent
from spec_writer.agent.llm_client import Message, ToolSpec
from spec_writer.agent.providers.fake import FakeLLMClient, ScriptedResponse
from spec_writer.agent.tools.ask_user import get_answerer, set_answerer
from spec_writer.engine.project_context import ProjectContext
from spec_writer.interface.agent_bridge.session import (
    ChatBridgeSession,
    SessionAlreadyAskingError,
    SessionBusyError,
    SessionClosedError,
    SessionNotAskingError,
)
from spec_writer.interface.agent_bridge.tools import (
    BridgeError,
    SessionManager,
    chat_answer,
    chat_close,
    chat_history,
    chat_list_sessions,
    chat_open,
    chat_send,
)
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.methodology.sources import FilesystemMixinSource

# ── fixtures ──────────────────────────────────────────────────────────────


def _make_minimal_methodology(tmp_path: Path) -> ResolvedMethodology:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base.yaml"
    base.write_text(
        "version: 1\n"
        "layers:\n"
        "  - id: L1\n    role: source-boundary\n    name: vision\n"
        "  - id: L4\n    role: code-boundary\n    name: tech\n    is_final: true\n",
        encoding="utf-8",
    )
    mixins_dir = tmp_path / "mixins"
    mixins_dir.mkdir()
    (mixins_dir / "vision.yaml").write_text(
        "mixin: vision\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    return load_methodology(base, [FilesystemMixinSource(mixins_dir)], ["vision"])


def _make_project(tmp_path: Path) -> ProjectContext:
    spec_editor = tmp_path / "spec-editor"
    for sub in ("sources-raw", "sources", "specs", "_meta", "_meta/sessions"):
        (spec_editor / sub).mkdir(parents=True)
    (spec_editor / "methodology.yaml").write_text("preset: minimal\n", encoding="utf-8")
    return ProjectContext.from_cwd(tmp_path)


def _build_bridge_session(
    tmp_path: Path,
    llm: Any,
    *,
    direction: str = "fwd",
) -> ChatBridgeSession:
    project_ctx = _make_project(tmp_path)
    methodology = _make_minimal_methodology(tmp_path / "mm")
    chat_session = ChatSession(project_ctx=project_ctx)
    agent = ChatAgent(
        llm=llm,
        project_ctx=project_ctx,
        methodology=methodology,
        session=chat_session,
        direction=direction,
    )
    return ChatBridgeSession(agent=agent, chat_session=chat_session)


@pytest.fixture(autouse=True)
def _clear_answerer() -> Any:
    """Make sure no answerer leaks across tests (global slot)."""
    set_answerer(None)
    yield
    set_answerer(None)


# ── 1. plain send / done path ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_returns_done_for_text_only_reply(tmp_path: Path) -> None:
    llm = FakeLLMClient([ScriptedResponse(text="hello world")])
    session = _build_bridge_session(tmp_path, llm)
    session.install_answerer()
    try:
        result = await session.send("hi there")
        assert result.status == "done"
        assert result.reply == "hello world"
        assert session.state == "idle"
    finally:
        session.uninstall_answerer()
        await session.close()


# ── 2. ask_user interception via answerer hook ───────────────────────────


class AskingFakeLLM:
    """LLM whose chat() yields prefix, then drives the global answerer,
    then yields a suffix containing the answer it received.
    Mimics what a real LLM does when it invokes the ask_user MCP tool."""

    name = "asking-fake"

    def __init__(self, prefix: str, question: str, suffix_template: str) -> None:
        self._prefix = prefix
        self._question = question
        self._suffix_template = suffix_template

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int = 4096,
        response_schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[EngineEvent]:
        del messages, tools, max_tokens, response_schema, system_prompt
        yield TextDeltaEvent(text=self._prefix)
        answerer = get_answerer()
        assert answerer is not None, "test invariant: bridge must install answerer"
        result = answerer(self._question)
        # ask_user is async-friendly; accept either sync or awaitable.
        if asyncio.iscoroutine(result):
            answer = await result
        else:
            answer = result  # type: ignore[assignment]
        yield TextDeltaEvent(text=self._suffix_template.format(answer=answer))


@pytest.mark.asyncio
async def test_ask_user_suspends_then_resumes(tmp_path: Path) -> None:
    llm = AskingFakeLLM(
        prefix="thinking… ",
        question="which CMS?",
        suffix_template="got: {answer}",
    )
    session = _build_bridge_session(tmp_path, llm)
    session.install_answerer()
    try:
        # First call: hits ask_user, returns asking
        first = await session.send("start")
        assert first.status == "asking"
        assert first.asked == "which CMS?"
        assert first.reply == "thinking… "
        assert session.state == "asking"

        # Supply answer; agent resumes and finishes
        second = await session.answer("Drupal")
        assert second.status == "done"
        assert "got: Drupal" in second.reply
        assert session.state == "idle"
    finally:
        session.uninstall_answerer()
        await session.close()


# ── 3. errors / wrong-state guards ───────────────────────────────────────


@pytest.mark.asyncio
async def test_send_during_asking_raises(tmp_path: Path) -> None:
    llm = AskingFakeLLM(prefix="x", question="q", suffix_template="y{answer}")
    session = _build_bridge_session(tmp_path, llm)
    session.install_answerer()
    try:
        await session.send("first")
        assert session.state == "asking"
        with pytest.raises(SessionAlreadyAskingError):
            await session.send("second")
    finally:
        session.uninstall_answerer()
        await session.close()


@pytest.mark.asyncio
async def test_answer_outside_asking_raises(tmp_path: Path) -> None:
    llm = FakeLLMClient([ScriptedResponse(text="ok")])
    session = _build_bridge_session(tmp_path, llm)
    session.install_answerer()
    try:
        with pytest.raises(SessionNotAskingError):
            await session.answer("nope")
    finally:
        session.uninstall_answerer()
        await session.close()


@pytest.mark.asyncio
async def test_handle_user_exception_yields_error(tmp_path: Path) -> None:
    llm = FakeLLMClient([ScriptedResponse(exception=RuntimeError("boom"))])
    session = _build_bridge_session(tmp_path, llm)
    session.install_answerer()
    try:
        result = await session.send("hello")
        # ChatAgent catches and emits StatusEvent(phase=error); bridge surfaces it.
        assert result.status == "error"
        assert result.error is not None and "boom" in result.error
        assert session.state == "idle"
    finally:
        session.uninstall_answerer()
        await session.close()


@pytest.mark.asyncio
async def test_close_releases_pending_question(tmp_path: Path) -> None:
    llm = AskingFakeLLM(prefix="x", question="q", suffix_template="y{answer}")
    session = _build_bridge_session(tmp_path, llm)
    session.install_answerer()
    try:
        await session.send("hi")
        assert session.state == "asking"
        session.uninstall_answerer()
        await session.close()
        assert session.state == "closed"
        with pytest.raises(SessionClosedError):
            await session.send("anything")
    finally:
        # idempotent
        session.uninstall_answerer()


# ── 4. SessionManager + MCP tool handlers ────────────────────────────────


def _make_project_via_init(tmp_path: Path) -> Path:
    """Build a real on-disk project with a minimal methodology.yaml the
    bridge's load_methodology_for_project can consume."""
    # Use the package's own built-in base + minimal preset (vision only).
    root = tmp_path / "project"
    (root / "spec-editor" / "sources-raw").mkdir(parents=True)
    (root / "spec-editor" / "sources").mkdir()
    (root / "spec-editor" / "specs").mkdir()
    (root / "spec-editor" / "_meta" / "sessions").mkdir(parents=True)
    # Use the built-in `minimal` preset shape
    (root / "spec-editor" / "methodology.yaml").write_text(
        "preset: minimal\nactive_mixins:\n  - core-traceability\n  - vision\n",
        encoding="utf-8",
    )
    # Stub agent.yaml so build_chat_agent does not blow up looking for it
    (root / "spec-editor" / "agent.yaml").write_text(
        "provider: claude-subscription\nmodel: claude-opus-4-7\n",
        encoding="utf-8",
    )
    return root


@pytest.mark.asyncio
async def test_manager_refuses_second_open(tmp_path: Path) -> None:
    """Manager enforces single-answerer invariant: cannot open two."""
    # We can't easily build a real ChatAgent here without subscription
    # plumbing, so we exercise the manager by mocking build_chat_agent.
    import spec_writer.interface.agent_bridge.tools as bridge_tools_module

    def _fake_build_chat_agent(
        project_ctx: Any,
        methodology: Any,
        session: Any,
        *,
        direction: str = "fwd",
        resume_from: bool = False,
        llm_override: Any = None,
    ) -> Any:
        return ChatAgent(
            llm=FakeLLMClient([ScriptedResponse(text="-")]),
            project_ctx=project_ctx,
            methodology=methodology,
            session=session,
            direction=direction,
        )

    original = bridge_tools_module.build_chat_agent
    bridge_tools_module.build_chat_agent = _fake_build_chat_agent  # type: ignore[assignment]
    try:
        manager = SessionManager()
        root = _make_project_via_init(tmp_path)
        first = await chat_open(manager, {"project_root": str(root)})
        assert first["ok"] is True
        sid = first["session_id"]
        # Second open should fail
        with pytest.raises(BridgeError, match="another session"):
            await chat_open(manager, {"project_root": str(root)})
        await chat_close(manager, {"session_id": sid})
        # After close, we can open again
        third = await chat_open(manager, {"project_root": str(root)})
        assert third["ok"] is True
        await chat_close(manager, {"session_id": third["session_id"]})
    finally:
        bridge_tools_module.build_chat_agent = original  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_tools_send_and_history_roundtrip(tmp_path: Path) -> None:
    import spec_writer.interface.agent_bridge.tools as bridge_tools_module

    def _fake_build(
        project_ctx: Any,
        methodology: Any,
        session: Any,
        *,
        direction: str = "fwd",
        resume_from: bool = False,
        llm_override: Any = None,
    ) -> Any:
        return ChatAgent(
            llm=FakeLLMClient([ScriptedResponse(text="acknowledged")]),
            project_ctx=project_ctx,
            methodology=methodology,
            session=session,
            direction=direction,
        )

    original = bridge_tools_module.build_chat_agent
    bridge_tools_module.build_chat_agent = _fake_build  # type: ignore[assignment]
    try:
        manager = SessionManager()
        root = _make_project_via_init(tmp_path)
        opened = await chat_open(manager, {"project_root": str(root)})
        sid = opened["session_id"]

        sent = await chat_send(manager, {"session_id": sid, "message": "hello"})
        assert sent["status"] == "done"
        assert sent["reply"] == "acknowledged"
        assert sent["ok"] is True

        listed = await chat_list_sessions(manager, {})
        assert any(s["session_id"] == sid for s in listed["sessions"])

        history = await chat_history(manager, {"session_id": sid, "format": "markdown"})
        assert "## " in history["transcript"] or "# Session" in history["transcript"]
        assert "hello" in history["transcript"]
        assert "acknowledged" in history["transcript"]

        history_json = await chat_history(manager, {"session_id": sid, "format": "json"})
        kinds = {e.get("kind") for e in history_json["events"]}
        assert {"user-utterance", "agent-message"} <= kinds

        await chat_close(manager, {"session_id": sid})
    finally:
        bridge_tools_module.build_chat_agent = original  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_tools_answer_resumes_paused_turn(tmp_path: Path) -> None:
    import spec_writer.interface.agent_bridge.tools as bridge_tools_module

    def _fake_build(
        project_ctx: Any,
        methodology: Any,
        session: Any,
        *,
        direction: str = "fwd",
        resume_from: bool = False,
        llm_override: Any = None,
    ) -> Any:
        return ChatAgent(
            llm=AskingFakeLLM(
                prefix="working… ",
                question="which db?",
                suffix_template="picked {answer}",
            ),
            project_ctx=project_ctx,
            methodology=methodology,
            session=session,
            direction=direction,
        )

    original = bridge_tools_module.build_chat_agent
    bridge_tools_module.build_chat_agent = _fake_build  # type: ignore[assignment]
    try:
        manager = SessionManager()
        root = _make_project_via_init(tmp_path)
        opened = await chat_open(manager, {"project_root": str(root)})
        sid = opened["session_id"]

        sent = await chat_send(manager, {"session_id": sid, "message": "go"})
        assert sent["status"] == "asking"
        assert sent["asked"] == "which db?"

        answered = await chat_answer(manager, {"session_id": sid, "answer": "postgres"})
        assert answered["status"] == "done"
        assert "picked postgres" in answered["reply"]
        await chat_close(manager, {"session_id": sid})
    finally:
        bridge_tools_module.build_chat_agent = original  # type: ignore[assignment]


# ── 5. MCP server tool registration smoke ────────────────────────────────


@pytest.mark.asyncio
async def test_server_registers_all_tools() -> None:
    from spec_writer.interface.agent_bridge.server import _TOOL_DEFINITIONS, build_server

    server, _manager = build_server()
    names = {t.name for t in _TOOL_DEFINITIONS}
    expected = {
        "chat_open",
        "chat_send",
        "chat_answer",
        "chat_cancel",
        "chat_resume",
        "chat_history",
        "chat_close",
        "chat_list_sessions",
    }
    assert expected <= names
    assert server.name == "spec-writer-bridge"


# keep imports alive
_ = StatusEvent
_ = SessionBusyError
_ = json
