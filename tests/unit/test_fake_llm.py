"""S1.7: FakeLLMClient — scripted responses, call recording."""

from __future__ import annotations

import pytest

from spec_writer.agent.events import TextDeltaEvent, ToolCallEvent
from spec_writer.agent.llm_client import Message, ToolSpec
from spec_writer.agent.providers.fake import (
    FakeLLMClient,
    ScriptedResponse,
)


def _msg(text: str) -> list[Message]:
    return [Message(role="user", content=text)]


async def _consume(client: FakeLLMClient, content: str = "hi"):
    events = []
    async for ev in client.chat(_msg(content), tools=[]):
        events.append(ev)
    return events


@pytest.mark.asyncio
async def test_fake_returns_scripted_sequence() -> None:
    client = FakeLLMClient(
        [
            ScriptedResponse(text="first"),
            ScriptedResponse(text="second"),
        ]
    )
    out1 = await _consume(client)
    out2 = await _consume(client)
    assert len(out1) == 1 and isinstance(out1[0], TextDeltaEvent) and out1[0].text == "first"
    assert len(out2) == 1 and isinstance(out2[0], TextDeltaEvent) and out2[0].text == "second"


@pytest.mark.asyncio
async def test_fake_emits_tool_calls_then_text() -> None:
    client = FakeLLMClient(
        [
            ScriptedResponse(
                tool_calls=[("record_spec", {"id": "L1-VIS-001"})],
                text="done",
            )
        ]
    )
    events = await _consume(client)
    assert isinstance(events[0], ToolCallEvent)
    assert events[0].tool == "record_spec"
    assert events[0].args == {"id": "L1-VIS-001"}
    assert isinstance(events[1], TextDeltaEvent)
    assert events[1].text == "done"


@pytest.mark.asyncio
async def test_fake_exhausted_raises_assertion() -> None:
    client = FakeLLMClient([ScriptedResponse(text="only")])
    await _consume(client)
    with pytest.raises(AssertionError, match="no more scripted"):
        await _consume(client)


@pytest.mark.asyncio
async def test_fake_records_calls() -> None:
    client = FakeLLMClient([ScriptedResponse(text="x")])
    tools = [ToolSpec(name="record_spec", description="...", input_schema={})]
    async for _ in client.chat(_msg("hello"), tools=tools, max_tokens=1000):
        pass
    [call] = client.calls_made
    assert call.messages[0].content == "hello"
    assert call.tools[0].name == "record_spec"
    assert call.max_tokens == 1000


@pytest.mark.asyncio
async def test_fake_raises_scripted_exception() -> None:
    client = FakeLLMClient([ScriptedResponse(exception=RuntimeError("boom"))])
    with pytest.raises(RuntimeError, match="boom"):
        await _consume(client)


@pytest.mark.asyncio
async def test_fake_remaining_decreases() -> None:
    client = FakeLLMClient(
        [ScriptedResponse(text="a"), ScriptedResponse(text="b"), ScriptedResponse(text="c")]
    )
    assert client.remaining == 3
    await _consume(client)
    assert client.remaining == 2


@pytest.mark.asyncio
async def test_fake_empty_response_no_events() -> None:
    client = FakeLLMClient([ScriptedResponse()])
    events = await _consume(client)
    assert events == []


def test_fake_implements_llmclient_protocol() -> None:
    from spec_writer.agent.llm_client import LLMClient

    client = FakeLLMClient([])
    # static type-narrowing — Protocol checks via isinstance need runtime_checkable;
    # we assert structurally that the name attribute and chat method exist.
    assert (
        isinstance(client, LLMClient.__class__) is False
    )  # Protocol classes themselves aren't instances
    assert hasattr(client, "name")
    assert hasattr(client, "chat")
    assert client.name == "fake"
