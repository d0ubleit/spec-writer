"""LLMClient Protocol — vendor-agnostic streaming interface.

Real provider implementations live in ``agent/providers/`` (Phase 2);
Phase 1 only ships the protocol and a deterministic FakeLLMClient
for unit tests. See docs/architecture.md.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from spec_writer.agent.events import EngineEvent


@dataclass(frozen=True)
class Message:
    """Single conversation message — Phase 1 keeps content as plain text."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True)
class ToolSpec:
    """Description of a callable tool advertised to the model."""

    name: str
    description: str
    input_schema: dict[str, Any]


class LLMClient(Protocol):
    """Vendor-portable streaming chat interface.

    Implementations yield ``EngineEvent`` instances (TextDeltaEvent,
    ToolCallEvent, RateLimitEvent, …) as the model produces output;
    callers iterate with ``async for``.
    """

    @property
    def name(self) -> str: ...

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int = 4096,
        response_schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[EngineEvent]: ...


@runtime_checkable
class SupportsSummarize(Protocol):
    """Optional provider capability — collapse older messages into a summary.

    Providers that implement this enable the ``claude_compact`` compression
    strategy (see :mod:`agent.compression`). Providers without it (e.g.
    FakeLLM, third-party adapters) cause the strategy to be silently
    skipped with a warning — model-agnostic strategies still run.
    """

    async def summarize(self, messages: list[Message]) -> str: ...


__all__ = ["LLMClient", "Message", "SupportsSummarize", "ToolSpec"]
