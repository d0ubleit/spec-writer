"""FakeLLMClient — deterministic scripted-response provider for tests.

Each ``ScriptedResponse`` is consumed by one ``chat()`` call; the
fake yields the tool calls and final text in the configured order
and records every call's inputs so assertions can verify the agent
passed the expected tools / messages.

If the script is exhausted, the next ``chat()`` call raises
``AssertionError`` rather than silently fabricating output — Phase 1
tests must script every LLM round-trip.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from spec_writer.agent.events import EngineEvent, TextDeltaEvent, ToolCallEvent
from spec_writer.agent.llm_client import Message, ToolSpec


def _empty_tool_calls() -> list[tuple[str, dict[str, Any]]]:
    return []


@dataclass(frozen=True)
class ScriptedResponse:
    """One response in the fake's queue.

    Order of emission: any ``tool_calls`` first (as ``ToolCallEvent``),
    then ``text`` (as a single ``TextDeltaEvent``). If ``exception``
    is set, it is raised before any events are yielded.
    """

    text: str | None = None
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=_empty_tool_calls)
    exception: Exception | None = None


@dataclass(frozen=True)
class CallRecord:
    """What the agent passed into a single chat() call."""

    messages: list[Message]
    tools: list[ToolSpec]
    max_tokens: int
    system_prompt: str | None
    response_schema: dict[str, Any] | None


class FakeLLMClient:
    """In-memory ``LLMClient`` returning scripted responses."""

    name = "fake"

    def __init__(self, responses: list[ScriptedResponse]) -> None:
        self._responses: list[ScriptedResponse] = list(responses)
        self._calls: list[CallRecord] = []

    @property
    def calls_made(self) -> list[CallRecord]:
        return list(self._calls)

    @property
    def remaining(self) -> int:
        return len(self._responses)

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int = 4096,
        response_schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[EngineEvent]:
        self._calls.append(
            CallRecord(
                messages=list(messages),
                tools=list(tools),
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                response_schema=response_schema,
            )
        )
        if not self._responses:
            raise AssertionError(
                f"FakeLLMClient: no more scripted responses (call #{len(self._calls)})"
            )
        scripted = self._responses.pop(0)
        if scripted.exception is not None:
            raise scripted.exception
        for tool_name, args in scripted.tool_calls:
            yield ToolCallEvent(tool=tool_name, args=args)
        if scripted.text is not None:
            yield TextDeltaEvent(text=scripted.text)


__all__ = ["CallRecord", "FakeLLMClient", "ScriptedResponse"]
