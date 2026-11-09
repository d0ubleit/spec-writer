"""Console renderer — collapse an EngineEvent into one human-readable line.

Lines have a fixed prefix and key=value tail so they grep cleanly:

    [status] generate layer=L1
    [tool]   record_spec id=L1-VIS-001
    [spec]   L1-VIS-001 created
    [layer]  L1 complete coverage=1.0
    [error]  lock busy pid=1234

Stable ordering is achieved by sorting detail keys alphabetically;
this keeps snapshot tests reproducible across Python versions.
"""

from __future__ import annotations

from typing import Any

from spec_writer.agent.events import (
    EngineEvent,
    LayerCompleteEvent,
    RateLimitEvent,
    SpecCreatedEvent,
    StatusEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    ValidationFailedEvent,
)


def _kv(d: dict[str, Any]) -> str:
    return " ".join(f"{k}={d[k]}" for k in sorted(d))


def render(event: EngineEvent) -> str:
    """Return a single-line representation of ``event``."""
    if isinstance(event, StatusEvent):
        tail = _kv(event.details)
        return f"[status] {event.phase} {tail}".rstrip()
    if isinstance(event, TextDeltaEvent):
        return f"[text] {event.text}"
    if isinstance(event, ToolCallEvent):
        return f"[tool] {event.tool} {_kv(event.args)}".rstrip()
    if isinstance(event, ToolResultEvent):
        return f"[tool-result] {event.tool} {event.result!r}"
    if isinstance(event, SpecCreatedEvent):
        return f"[spec] {event.spec_id} created"
    if isinstance(event, LayerCompleteEvent):
        return f"[layer] {event.layer} complete {_kv(event.metrics)}".rstrip()
    if isinstance(event, ValidationFailedEvent):
        codes = ",".join(str(i.get("code", "?")) for i in event.issues)
        return f"[validation-failed] {event.spec_id} {codes}"
    if isinstance(event, RateLimitEvent):
        when = event.resets_at.isoformat() if event.resets_at else "unknown"
        return f"[rate-limit] {event.status} resets_at={when}"
    if isinstance(event, UsageEvent):
        u = event.usage
        return (
            f"[usage] in={u.input} out={u.output} "
            f"cache_read={u.cache_read_input} cache_write={u.cache_creation_input}"
        )
    # Only EngineErrorEvent remains in the Union here.
    return f"[error] {event.error} {_kv(event.details)}".rstrip()


__all__ = ["render"]
