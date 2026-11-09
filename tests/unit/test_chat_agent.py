"""S3.13 — ChatAgent session-level loop + streaming + compression gating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spec_writer.agent.chat_agent import ChatAgent, build_chat_system_prompt
from spec_writer.agent.chat_session import ChatSession
from spec_writer.agent.compression import CompressionPolicy, summarizer_for
from spec_writer.agent.events import (
    StatusEvent,
    TextDeltaEvent,
    ToolCallEvent,
)
from spec_writer.agent.llm_client import Message
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
    return load_methodology(
        meta / "base.yaml", [FilesystemMixinSource(mixins)], active_mixins=["v"]
    )


# ── system prompt ───────────────────────────────────────────────────────


def test_build_chat_system_prompt_mentions_methodology_layers(tmp_path: Path) -> None:
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    prompt = build_chat_system_prompt(methodology, project)
    assert "L1" in prompt
    assert "L2" in prompt
    assert "vision" in prompt


def test_build_chat_system_prompt_mentions_ownership(tmp_path: Path) -> None:
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    prompt = build_chat_system_prompt(methodology, project)
    # core ownership rules from the ownership principle
    assert "One conversation" in prompt or "one context" in prompt.lower()
    assert "ask_user" in prompt
    assert "validate" in prompt


def test_build_chat_system_prompt_lists_engine_primitives(tmp_path: Path) -> None:
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    prompt = build_chat_system_prompt(methodology, project)
    for tool in (
        "list_sources",
        "list_corpus",
        "record_spec",
        "patch_spec",
        "validate",
        "metrics",
        "query_traceability",
    ):
        assert f"mcp__spec-writer__{tool}" in prompt


# ── basic single-turn flow ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_user_appends_to_history(tmp_path: Path) -> None:
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    session = ChatSession(project_ctx=project)
    llm = FakeLLMClient([ScriptedResponse(text="Hello.")])

    agent = ChatAgent(llm, project, methodology, session)
    events = [event async for event in agent.handle_user("Hi there")]

    # System + user + assistant
    assert agent.history[0].role == "system"
    assert agent.history[1].role == "user"
    assert agent.history[1].content == "Hi there"
    assert agent.history[2].role == "assistant"
    assert agent.history[2].content == "Hello."
    assert any(isinstance(e, TextDeltaEvent) for e in events)


@pytest.mark.asyncio
async def test_handle_user_logs_to_session(tmp_path: Path) -> None:
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    session = ChatSession(project_ctx=project)
    llm = FakeLLMClient([ScriptedResponse(text="Acknowledged.")])

    agent = ChatAgent(llm, project, methodology, session)
    _ = [e async for e in agent.handle_user("List my sources please")]

    payload = json.loads(session.path.read_text(encoding="utf-8"))
    kinds = [e["kind"] for e in payload["events"]]
    assert "user-utterance" in kinds
    assert "agent-message" in kinds


@pytest.mark.asyncio
async def test_handle_user_streams_tool_call_then_text(tmp_path: Path) -> None:
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    session = ChatSession(project_ctx=project)
    llm = FakeLLMClient(
        [
            ScriptedResponse(
                tool_calls=[("list_sources", {})],
                text="I'll check sources first.",
            )
        ]
    )

    agent = ChatAgent(llm, project, methodology, session)
    events = [event async for event in agent.handle_user("what's in sources?")]

    tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
    text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert len(tool_call_events) == 1
    assert tool_call_events[0].tool == "list_sources"
    assert len(text_events) == 1

    # Session logged the tool call
    payload = json.loads(session.path.read_text(encoding="utf-8"))
    kinds = [e["kind"] for e in payload["events"]]
    assert "tool-call" in kinds


@pytest.mark.asyncio
async def test_handle_user_empty_text_is_noop(tmp_path: Path) -> None:
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    session = ChatSession(project_ctx=project)
    llm = FakeLLMClient([])  # would raise if called

    agent = ChatAgent(llm, project, methodology, session)
    events = [event async for event in agent.handle_user("   ")]
    assert events == []
    assert llm.remaining == 0


@pytest.mark.asyncio
async def test_handle_user_llm_error_surfaced_as_status(tmp_path: Path) -> None:
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    session = ChatSession(project_ctx=project)
    llm = FakeLLMClient([ScriptedResponse(exception=RuntimeError("boom"))])

    agent = ChatAgent(llm, project, methodology, session)
    events = [event async for event in agent.handle_user("hi")]
    statuses = [e for e in events if isinstance(e, StatusEvent)]
    assert len(statuses) == 1
    assert statuses[0].phase == "error"


# ── multi-turn history persistence ──────────────────────────────────────


@pytest.mark.asyncio
async def test_two_turns_history_grows(tmp_path: Path) -> None:
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    session = ChatSession(project_ctx=project)
    llm = FakeLLMClient(
        [
            ScriptedResponse(text="First reply."),
            ScriptedResponse(text="Second reply."),
        ]
    )

    agent = ChatAgent(llm, project, methodology, session)
    async for _ in agent.handle_user("first"):
        pass
    async for _ in agent.handle_user("second"):
        pass

    # System + (user + assistant) x 2 = 5 entries
    assert len(agent.history) == 5
    assert agent.history[1].content == "first"
    assert agent.history[2].content == "First reply."
    assert agent.history[3].content == "second"
    assert agent.history[4].content == "Second reply."

    # Each chat() saw the cumulative history
    assert len(llm.calls_made) == 2
    assert len(llm.calls_made[0].messages) == 2  # system + user
    assert len(llm.calls_made[1].messages) == 4  # system + user + assistant + user


# ── initial_history (resume path) ───────────────────────────────────────


@pytest.mark.asyncio
async def test_initial_history_used_verbatim(tmp_path: Path) -> None:
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    session = ChatSession(project_ctx=project)

    initial = [
        Message(role="system", content="custom-system"),
        Message(role="user", content="earlier user msg"),
        Message(role="assistant", content="earlier assistant reply"),
    ]
    llm = FakeLLMClient([ScriptedResponse(text="continued reply")])
    agent = ChatAgent(llm, project, methodology, session, initial_history=initial)
    async for _ in agent.handle_user("continue"):
        pass

    assert agent.history[0].content == "custom-system"
    assert agent.history[1].content == "earlier user msg"
    # The chat call should have seen the initial history + new user message
    sent = llm.calls_made[0].messages
    assert sent[0].content == "custom-system"
    assert sent[-1].content == "continue"


# ── summarize gating ────────────────────────────────────────────────────


def test_summarizer_for_returns_none_on_plain_llm() -> None:
    llm = FakeLLMClient([])
    assert summarizer_for(llm) is None


def test_summarizer_for_returns_method_on_provider_with_summarize() -> None:
    class FakeSummarizingLLM:
        name = "fake-summarize"

        async def chat(self, *a, **kw):  # type: ignore[no-untyped-def]
            if False:
                yield  # make it async iterator
            return

        async def summarize(self, messages: list[Message]) -> str:
            return "(summary)"

    llm = FakeSummarizingLLM()
    summarizer = summarizer_for(llm)
    assert summarizer is not None


@pytest.mark.asyncio
async def test_compression_policy_skips_claude_compact_when_no_summarizer(
    tmp_path: Path,
) -> None:
    """Smoke: ChatAgent doesn't crash when policy lists claude_compact + LLM lacks summarize."""
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    session = ChatSession(project_ctx=project)
    llm = FakeLLMClient([ScriptedResponse(text="ok")])

    policy = CompressionPolicy(
        trigger_chars=0,  # always trigger
        strategies=["claude_compact", "sliding_window"],
        window_keep=2,
    )
    agent = ChatAgent(llm, project, methodology, session, compression_policy=policy)
    events = [event async for event in agent.handle_user("hello")]
    assert any(isinstance(e, TextDeltaEvent) for e in events)


# ── allowed tools surface ───────────────────────────────────────────────


def test_chat_agent_advertises_engine_primitive_tools(tmp_path: Path) -> None:
    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    session = ChatSession(project_ctx=project)
    llm = FakeLLMClient([])
    agent = ChatAgent(llm, project, methodology, session)
    tools = agent.allowed_tools
    for expected in (
        "mcp__spec-writer__record_spec",
        "mcp__spec-writer__validate",
        "mcp__spec-writer__metrics",
        "mcp__spec-writer__list_sources",
        "mcp__spec-writer__list_corpus",
        "mcp__spec-writer__ask_user",
    ):
        assert expected in tools


# ── unused result variable safety ───────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_result_event_logged(tmp_path: Path) -> None:
    """ToolResultEvent forwarded by FakeLLM should reach session log."""
    from spec_writer.agent.events import ToolResultEvent as _TRE

    project = _project(tmp_path)
    methodology = _methodology(tmp_path)
    session = ChatSession(project_ctx=project)

    class _CustomFake:
        name = "custom"

        async def chat(self, *a, **kw):  # type: ignore[no-untyped-def]
            del a, kw
            yield ToolCallEvent(tool="validate", args={})
            yield _TRE(tool="validate", result={"ok": True, "issues": []})
            yield TextDeltaEvent(text="done")

    agent = ChatAgent(_CustomFake(), project, methodology, session)
    _ = [e async for e in agent.handle_user("check")]
    payload = json.loads(session.path.read_text(encoding="utf-8"))
    kinds = [e["kind"] for e in payload["events"]]
    assert "tool-result" in kinds
