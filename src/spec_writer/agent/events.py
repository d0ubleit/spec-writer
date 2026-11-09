"""EngineEvent — vendor-agnostic streaming events the engine emits.

The engine and providers raise these events; UIs/CLI/tests subscribe
to a single normalized stream regardless of which LLM SDK produced
the underlying delta. See docs/architecture.md.

Phase 1 implements the minimal 9-event surface listed in the
IMPLEMENTATION-PLAN; vendor-specific adapters (S2.1) translate
SDK events into these.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypeAlias

from spec_writer.types import TokenUsage

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class StatusEvent:
    """Generic lifecycle marker — phase plus arbitrary details."""

    phase: str
    details: dict[str, Any]


@dataclass(frozen=True)
class TextDeltaEvent:
    """Partial text from the model (streaming)."""

    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    """The model invoked a tool."""

    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolResultEvent:
    """A tool call returned a result back to the model."""

    tool: str
    result: Any


@dataclass(frozen=True)
class SpecCreatedEvent:
    """A spec was written to disk (atomic) — diff vs previous content."""

    spec_id: str
    diff: str


@dataclass(frozen=True)
class LayerCompleteEvent:
    """A layer loop finished — converged or stopped, with final metrics."""

    layer: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ValidationFailedEvent:
    """One or more validators returned errors for a spec."""

    spec_id: str
    issues: list[dict[str, Any]]


@dataclass(frozen=True)
class RateLimitEvent:
    """The LLM provider rate-limit warning or rejection."""

    status: Literal["warning", "rejected"]
    resets_at: datetime | None


@dataclass(frozen=True)
class EngineErrorEvent:
    """A non-recoverable engine error surfaced for display / abort."""

    error: str
    details: dict[str, Any]


@dataclass(frozen=True)
class UsageEvent:
    """LLM token usage reported by the provider after each turn.

    Carries cache stats too so callers can compute hit-ratio. See
    the design rationale. Consumed by ChatAgent / runtime to feed
    :class:`TokenLedger`.
    """

    usage: TokenUsage


EngineEvent: TypeAlias = (
    StatusEvent
    | TextDeltaEvent
    | ToolCallEvent
    | ToolResultEvent
    | SpecCreatedEvent
    | LayerCompleteEvent
    | ValidationFailedEvent
    | RateLimitEvent
    | EngineErrorEvent
    | UsageEvent
)


# ── pub/sub bus ──────────────────────────────────────────────────────────


Handler: TypeAlias = Callable[[EngineEvent], None]


class EventBus:
    """Synchronous fan-out — handlers run in subscription order.

    Errors in one handler are logged and swallowed so subsequent
    handlers still receive the event (principle 9: fail loudly, but
    don't silently drop events from other subscribers).
    """

    def __init__(self) -> None:
        self._handlers: list[Handler] = []

    def subscribe(self, handler: Handler) -> None:
        self._handlers.append(handler)

    def unsubscribe(self, handler: Handler) -> None:
        if handler in self._handlers:
            self._handlers.remove(handler)

    def publish(self, event: EngineEvent) -> None:
        for handler in list(self._handlers):
            try:
                handler(event)
            except Exception:
                _LOG.exception("event handler %r raised on %r", handler, event)


__all__ = [
    "EngineErrorEvent",
    "EngineEvent",
    "EventBus",
    "Handler",
    "LayerCompleteEvent",
    "RateLimitEvent",
    "SpecCreatedEvent",
    "StatusEvent",
    "TextDeltaEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "ValidationFailedEvent",
]
