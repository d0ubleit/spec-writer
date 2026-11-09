"""S1.6: EngineEvent dataclasses + EventBus pub/sub."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from spec_writer.agent.events import (
    EngineErrorEvent,
    EngineEvent,
    EventBus,
    LayerCompleteEvent,
    RateLimitEvent,
    SpecCreatedEvent,
    StatusEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    ValidationFailedEvent,
)

# ── 1. event types instantiable ──────────────────────────────────────────


def test_status_event() -> None:
    ev = StatusEvent(phase="generate", details={"layer": "L1"})
    assert ev.phase == "generate"


def test_text_delta_event() -> None:
    ev = TextDeltaEvent(text="hello")
    assert ev.text == "hello"


def test_tool_call_event() -> None:
    ev = ToolCallEvent(tool="record_spec", args={"id": "L1-VIS-001"})
    assert ev.tool == "record_spec"
    assert ev.args["id"] == "L1-VIS-001"


def test_tool_result_event() -> None:
    ev = ToolResultEvent(tool="record_spec", result={"ok": True})
    assert ev.tool == "record_spec"


def test_spec_created_event() -> None:
    ev = SpecCreatedEvent(spec_id="L1-VIS-001", diff="+ created")
    assert ev.spec_id == "L1-VIS-001"


def test_layer_complete_event() -> None:
    ev = LayerCompleteEvent(layer="L1", metrics={"coverage": 0.9})
    assert ev.layer == "L1"
    assert ev.metrics["coverage"] == 0.9


def test_validation_failed_event() -> None:
    ev = ValidationFailedEvent(
        spec_id="L1-VIS-001",
        issues=[{"code": "missing_section", "field": "purpose"}],
    )
    assert ev.spec_id == "L1-VIS-001"
    assert ev.issues[0]["code"] == "missing_section"


def test_rate_limit_event() -> None:
    ev = RateLimitEvent(
        status="warning", resets_at=datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)
    )
    assert ev.status == "warning"


def test_engine_error_event() -> None:
    ev = EngineErrorEvent(error="lock busy", details={"pid": 1234})
    assert ev.error == "lock busy"


def test_events_are_frozen() -> None:
    from dataclasses import FrozenInstanceError

    ev = StatusEvent(phase="x", details={})
    with pytest.raises(FrozenInstanceError):
        ev.phase = "y"  # type: ignore[misc]


# ── 2. EventBus pub/sub ──────────────────────────────────────────────────


def test_bus_single_subscriber_receives_event() -> None:
    received: list[EngineEvent] = []
    bus = EventBus()
    bus.subscribe(received.append)
    bus.publish(StatusEvent(phase="start", details={}))
    assert len(received) == 1
    assert isinstance(received[0], StatusEvent)


def test_bus_multiple_subscribers_each_get_event() -> None:
    a: list[EngineEvent] = []
    b: list[EngineEvent] = []
    bus = EventBus()
    bus.subscribe(a.append)
    bus.subscribe(b.append)
    bus.publish(TextDeltaEvent(text="x"))
    assert len(a) == 1 and len(b) == 1


def test_bus_publishes_to_subscribers_in_order() -> None:
    order: list[str] = []
    bus = EventBus()
    bus.subscribe(lambda _ev: order.append("a"))
    bus.subscribe(lambda _ev: order.append("b"))
    bus.publish(StatusEvent(phase="x", details={}))
    assert order == ["a", "b"]


def test_bus_unsubscribe() -> None:
    received: list[EngineEvent] = []
    bus = EventBus()

    def handler(ev: EngineEvent) -> None:
        received.append(ev)

    bus.subscribe(handler)
    bus.unsubscribe(handler)
    bus.publish(StatusEvent(phase="x", details={}))
    assert received == []


def test_bus_handler_exception_does_not_block_others() -> None:
    received: list[EngineEvent] = []
    bus = EventBus()
    bus.subscribe(lambda _ev: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.subscribe(received.append)
    bus.publish(StatusEvent(phase="x", details={}))
    assert len(received) == 1


# ── 3. console renderer ──────────────────────────────────────────────────


def test_renderer_status() -> None:
    from spec_writer.interface.console_renderer import render

    out = render(StatusEvent(phase="generate", details={"layer": "L1"}))
    assert "generate" in out
    assert "L1" in out


def test_renderer_tool_call() -> None:
    from spec_writer.interface.console_renderer import render

    out = render(ToolCallEvent(tool="record_spec", args={"id": "L1-VIS-001"}))
    assert "record_spec" in out
    assert "L1-VIS-001" in out


def test_renderer_spec_created() -> None:
    from spec_writer.interface.console_renderer import render

    out = render(SpecCreatedEvent(spec_id="L1-VIS-001", diff="+ added"))
    assert "L1-VIS-001" in out


def test_renderer_layer_complete() -> None:
    from spec_writer.interface.console_renderer import render

    out = render(LayerCompleteEvent(layer="L2", metrics={"x": 0.9}))
    assert "L2" in out


def test_renderer_validation_failed() -> None:
    from spec_writer.interface.console_renderer import render

    out = render(
        ValidationFailedEvent(
            spec_id="L1-VIS-001",
            issues=[{"code": "missing_section", "field": "purpose"}],
        )
    )
    assert "L1-VIS-001" in out
    assert "missing_section" in out


def test_renderer_rate_limit() -> None:
    from spec_writer.interface.console_renderer import render

    out = render(
        RateLimitEvent(
            status="rejected", resets_at=datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)
        )
    )
    assert "rate" in out.lower()
    assert "2026" in out


def test_renderer_engine_error() -> None:
    from spec_writer.interface.console_renderer import render

    out = render(EngineErrorEvent(error="lock busy", details={"pid": 1234}))
    assert "lock busy" in out


def test_renderer_text_delta_concatenates() -> None:
    from spec_writer.interface.console_renderer import render

    out = render(TextDeltaEvent(text="streaming…"))
    assert "streaming…" in out


def test_renderer_tool_result() -> None:
    from spec_writer.interface.console_renderer import render

    out = render(ToolResultEvent(tool="record_spec", result={"ok": True}))
    assert "record_spec" in out


# ── 4. golden sequence ───────────────────────────────────────────────────


def test_golden_console_sequence() -> None:
    from spec_writer.interface.console_renderer import render

    sequence = [
        StatusEvent(phase="start", details={"direction": "fwd"}),
        StatusEvent(phase="layer:begin", details={"layer": "L1"}),
        ToolCallEvent(tool="record_spec", args={"id": "L1-VIS-001"}),
        SpecCreatedEvent(spec_id="L1-VIS-001", diff="+ vision"),
        LayerCompleteEvent(layer="L1", metrics={"coverage": 1.0}),
        StatusEvent(phase="done", details={}),
    ]
    output = "\n".join(render(ev) for ev in sequence)
    assert output == (
        "[status] start direction=fwd\n"
        "[status] layer:begin layer=L1\n"
        "[tool] record_spec id=L1-VIS-001\n"
        "[spec] L1-VIS-001 created\n"
        "[layer] L1 complete coverage=1.0\n"
        "[status] done"
    )
