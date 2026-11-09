"""S2.1: Claude subscription provider unit tests (mocked SDK)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from spec_writer.agent.events import (
    RateLimitEvent,
    TextDeltaEvent,
    ToolCallEvent,
    UsageEvent,
)
from spec_writer.agent.llm_client import Message
from spec_writer.agent.providers.claude_subscription import (
    ClaudeSubscriptionClient,
)


class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class ToolUseBlock:
    def __init__(self, name: str, args: dict[str, Any]) -> None:
        self.name = name
        self.input = args


class AssistantMessage:
    def __init__(self, blocks: list[Any]) -> None:
        self.content = blocks


class _RateLimitInfo:
    def __init__(self, status: str, resets_at: datetime | None) -> None:
        self.status = status
        self.resets_at = resets_at


class SdkRateLimitEvent:
    def __init__(self, status: str, resets_at: datetime | None) -> None:
        self.rate_limit_info = _RateLimitInfo(status, resets_at)


# Force the class name SDK code dispatches on.
SdkRateLimitEvent.__name__ = "RateLimitEvent"


def _text_block(text: str) -> Any:
    return TextBlock(text)


def _tool_block(name: str, args: dict[str, Any]) -> Any:
    return ToolUseBlock(name, args)


def _rate_limit(status: str, resets_at: datetime | None = None) -> Any:
    return SdkRateLimitEvent(status, resets_at)


def _assistant(blocks: list[Any]) -> Any:
    return AssistantMessage(blocks)


class SystemMessage:
    """Fake SDK init system message carrying a session_id."""

    def __init__(self, session_id: str) -> None:
        self.subtype = "init"
        self.data = {"session_id": session_id}


class ResultMessage:
    """Fake SDK end-of-turn message carrying token usage."""

    def __init__(self, usage: dict[str, Any] | None = None) -> None:
        self.usage = usage


@pytest.fixture
def patched_sdk(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {"calls": []}

    def make_query(*, prompt, options=None, transport=None):
        captured["prompt"] = prompt
        captured["options"] = options
        captured["calls"].append({"prompt": prompt, "options": options})

        async def gen():
            # Optionally seed a session_id once via `events_first_call`.
            if len(captured["calls"]) == 1 and "events_first_call" in captured:
                events = captured["events_first_call"]
            else:
                events = captured.get("events", [])
            for item in events:
                yield item

        return gen()

    class FakeOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "query", make_query)
    monkeypatch.setattr(claude_agent_sdk, "ClaudeAgentOptions", FakeOptions)
    return captured


@pytest.mark.asyncio
async def test_text_block_yields_text_delta(patched_sdk) -> None:
    patched_sdk["events"] = [_assistant([_text_block("hello world")])]
    client = ClaudeSubscriptionClient(model="claude-opus-4-7")
    events = []
    async for event in client.chat([Message(role="user", content="hi")], tools=[]):
        events.append(event)
    assert any(isinstance(e, TextDeltaEvent) and e.text == "hello world" for e in events)


@pytest.mark.asyncio
async def test_tool_block_yields_tool_call(patched_sdk) -> None:
    patched_sdk["events"] = [_assistant([_tool_block("record_spec", {"id": "L1-VIS-001"})])]
    client = ClaudeSubscriptionClient()
    events = []
    async for event in client.chat([Message(role="user", content="x")], tools=[]):
        events.append(event)
    assert any(isinstance(e, ToolCallEvent) and e.tool == "record_spec" for e in events)


@pytest.mark.asyncio
async def test_rate_limit_event_propagates(patched_sdk) -> None:
    when = datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)
    patched_sdk["events"] = [_rate_limit("rejected", when)]
    client = ClaudeSubscriptionClient()
    events = []
    async for event in client.chat([Message(role="user", content="x")], tools=[]):
        events.append(event)
    rate = next(e for e in events if isinstance(e, RateLimitEvent))
    assert rate.status == "rejected"
    assert rate.resets_at == when


@pytest.mark.asyncio
async def test_system_prompt_routed_via_options(patched_sdk) -> None:
    """: system_prompt rides ClaudeAgentOptions, not user message."""
    patched_sdk["events"] = []
    client = ClaudeSubscriptionClient(enable_session_resume=False)
    async for _ in client.chat(
        [Message(role="user", content="hi")],
        tools=[],
        system_prompt="be terse",
    ):
        pass
    options = patched_sdk["options"]
    # System prompt rides via the options kwarg, NOT embedded in the user message.
    assert options.kwargs.get("system_prompt") == "be terse"
    assert "[SYSTEM]" not in patched_sdk["prompt"]
    assert "[USER]" in patched_sdk["prompt"]
    assert "be terse" not in patched_sdk["prompt"]


@pytest.mark.asyncio
async def test_system_role_message_routed_via_options(patched_sdk) -> None:
    """: a Message(role='system') also routes through options."""
    patched_sdk["events"] = []
    client = ClaudeSubscriptionClient(enable_session_resume=False)
    async for _ in client.chat(
        [Message(role="system", content="STATIC"), Message(role="user", content="hi")],
        tools=[],
    ):
        pass
    options = patched_sdk["options"]
    assert options.kwargs.get("system_prompt") == "STATIC"
    # The user-prompt blob no longer contains the system content.
    assert "STATIC" not in patched_sdk["prompt"]


@pytest.mark.asyncio
async def test_options_carries_model(patched_sdk) -> None:
    patched_sdk["events"] = []
    client = ClaudeSubscriptionClient(model="claude-haiku-4-5-20251001")
    async for _ in client.chat([Message(role="user", content="hi")], tools=[]):
        pass
    options = patched_sdk["options"]
    assert options.kwargs["model"] == "claude-haiku-4-5-20251001"


def test_name_attribute() -> None:
    assert ClaudeSubscriptionClient.name == "claude-subscription"


# ── () session-resume ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_turn_captures_session_id(patched_sdk) -> None:
    """First turn: CLI emits SystemMessage(init, session_id) → client stores it."""
    patched_sdk["events_first_call"] = [
        SystemMessage("sess-abc-123"),
        _assistant([_text_block("hi back")]),
    ]
    client = ClaudeSubscriptionClient()
    async for _ in client.chat([Message(role="user", content="hi")], tools=[]):
        pass
    assert client._session_id == "sess-abc-123"


@pytest.mark.asyncio
async def test_second_turn_uses_resume_and_sends_delta_only(patched_sdk) -> None:
    """Second turn: options.resume=session_id; prompt = only the new message."""
    patched_sdk["events_first_call"] = [
        SystemMessage("sess-abc-123"),
        _assistant([_text_block("first reply")]),
    ]
    patched_sdk["events"] = [_assistant([_text_block("second reply")])]
    client = ClaudeSubscriptionClient()

    history = [Message(role="user", content="message one")]
    async for _ in client.chat(history, tools=[]):
        pass
    history.append(Message(role="assistant", content="first reply"))
    history.append(Message(role="user", content="message two"))
    async for _ in client.chat(history, tools=[]):
        pass

    calls = patched_sdk["calls"]
    assert len(calls) == 2
    # Turn 1 sent the full history; turn 2 sends only the delta.
    assert "message one" in calls[0]["prompt"]
    assert "message two" not in calls[0]["prompt"]
    assert "message one" not in calls[1]["prompt"]
    assert "message two" in calls[1]["prompt"]
    # Turn 2 uses resume; turn 1 didn't.
    assert "resume" not in calls[0]["options"].kwargs
    assert calls[1]["options"].kwargs.get("resume") == "sess-abc-123"


@pytest.mark.asyncio
async def test_resume_omits_system_prompt(patched_sdk) -> None:
    """On resume the CLI already has the system prompt — don't re-send it."""
    patched_sdk["events_first_call"] = [
        SystemMessage("sess-xyz"),
        _assistant([_text_block("ok")]),
    ]
    patched_sdk["events"] = []
    client = ClaudeSubscriptionClient()

    history = [Message(role="user", content="m1")]
    async for _ in client.chat(history, tools=[], system_prompt="STATIC SYSTEM"):
        pass
    history.append(Message(role="assistant", content="ok"))
    history.append(Message(role="user", content="m2"))
    async for _ in client.chat(history, tools=[], system_prompt="STATIC SYSTEM"):
        pass

    calls = patched_sdk["calls"]
    assert calls[0]["options"].kwargs.get("system_prompt") == "STATIC SYSTEM"
    assert "system_prompt" not in calls[1]["options"].kwargs


@pytest.mark.asyncio
async def test_history_drift_resets_session(patched_sdk) -> None:
    """If the caller's history no longer extends our submitted prefix → reset."""
    patched_sdk["events_first_call"] = [
        SystemMessage("sess-orig"),
        _assistant([_text_block("ok")]),
    ]
    patched_sdk["events"] = [
        SystemMessage("sess-new"),
        _assistant([_text_block("ok2")]),
    ]
    client = ClaudeSubscriptionClient()

    history = [
        Message(role="user", content="A"),
        Message(role="user", content="B"),
    ]
    async for _ in client.chat(history, tools=[]):
        pass
    assert client._session_id == "sess-orig"

    # Caller rewrites history (e.g. compression) — prior prefix differs.
    rewritten = [Message(role="user", content="SUMMARY of A+B")]
    # Drift detector fires before query — capture options to verify no resume.
    async for _ in client.chat(rewritten, tools=[]):
        pass
    calls = patched_sdk["calls"]
    assert "resume" not in calls[1]["options"].kwargs
    # The new session_id from the second call is now stored.
    assert client._session_id == "sess-new"


@pytest.mark.asyncio
async def test_reset_session_forces_fresh_call(patched_sdk) -> None:
    """Explicit reset_session() drops state and the next call starts fresh."""
    patched_sdk["events_first_call"] = [
        SystemMessage("sess-1"),
        _assistant([_text_block("hi")]),
    ]
    patched_sdk["events"] = [
        SystemMessage("sess-2"),
        _assistant([_text_block("again")]),
    ]
    client = ClaudeSubscriptionClient()
    async for _ in client.chat([Message(role="user", content="m1")], tools=[]):
        pass
    client.reset_session()
    async for _ in client.chat(
        [Message(role="user", content="m1"), Message(role="user", content="m2")],
        tools=[],
    ):
        pass
    calls = patched_sdk["calls"]
    assert "resume" not in calls[1]["options"].kwargs


@pytest.mark.asyncio
async def test_enable_session_resume_false_skips_state(patched_sdk) -> None:
    """Opt-out: legacy callers can disable session-resume entirely."""
    patched_sdk["events_first_call"] = [
        SystemMessage("sess-ignored"),
        _assistant([_text_block("ok")]),
    ]
    patched_sdk["events"] = [_assistant([_text_block("ok2")])]
    client = ClaudeSubscriptionClient(enable_session_resume=False)

    history = [Message(role="user", content="m1")]
    async for _ in client.chat(history, tools=[]):
        pass
    assert client._session_id is None
    history.append(Message(role="user", content="m2"))
    async for _ in client.chat(history, tools=[]):
        pass
    calls = patched_sdk["calls"]
    # Both turns send the full history with no resume.
    assert "resume" not in calls[1]["options"].kwargs
    assert "m1" in calls[1]["prompt"] and "m2" in calls[1]["prompt"]


# ──  UsageEvent extraction ────────────────────────────────


@pytest.mark.asyncio
async def test_result_message_emits_usage_event(patched_sdk) -> None:
    """: ResultMessage.usage → UsageEvent with cache stats."""
    patched_sdk["events"] = [
        _assistant([_text_block("hi")]),
        ResultMessage(
            usage={
                "input_tokens": 123,
                "output_tokens": 45,
                "cache_creation_input_tokens": 6789,
                "cache_read_input_tokens": 1011,
            }
        ),
    ]
    client = ClaudeSubscriptionClient(enable_session_resume=False)
    events: list[Any] = []
    async for event in client.chat([Message(role="user", content="hi")], tools=[]):
        events.append(event)

    usage_events = [e for e in events if isinstance(e, UsageEvent)]
    assert len(usage_events) == 1
    u = usage_events[0].usage
    assert u.input == 123
    assert u.output == 45
    assert u.total == 168
    assert u.cache_creation_input == 6789
    assert u.cache_read_input == 1011


@pytest.mark.asyncio
async def test_result_message_without_usage_skips_event(patched_sdk) -> None:
    """: ResultMessage with usage=None emits no UsageEvent."""
    patched_sdk["events"] = [
        _assistant([_text_block("hi")]),
        ResultMessage(usage=None),
    ]
    client = ClaudeSubscriptionClient(enable_session_resume=False)
    events: list[Any] = []
    async for event in client.chat([Message(role="user", content="hi")], tools=[]):
        events.append(event)
    assert not any(isinstance(e, UsageEvent) for e in events)


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_claude_subscription_smoke() -> None:
    """Real Claude call; opt-in via `pytest -m smoke`."""
    client = ClaudeSubscriptionClient()
    events = []
    async for event in client.chat(
        [Message(role="user", content="Reply with exactly the word PONG and nothing else.")],
        tools=[],
    ):
        events.append(event)
    text = "".join(e.text for e in events if isinstance(e, TextDeltaEvent))
    assert "PONG" in text.upper()
