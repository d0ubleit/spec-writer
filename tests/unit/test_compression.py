"""S2.7 — conversation-history compression policies."""

from __future__ import annotations

import pytest

from spec_writer.agent.compression import (
    CompressionPolicy,
    apply_policy,
    claude_compact,
    estimate_size,
    inject_spec_state,
    sliding_window,
    truncate_tool_results,
)
from spec_writer.agent.llm_client import Message
from spec_writer.types import Spec


def _msgs(*pairs: tuple[str, str]) -> list[Message]:
    return [Message(role=r, content=c) for r, c in pairs]  # type: ignore[arg-type]


def _spec(spec_id: str, layer: str, title: str) -> Spec:
    return Spec(id=spec_id, type="vision", layer=layer, title=title, body="")


# ── estimate_size ───────────────────────────────────────────────────────


def test_estimate_size_sums_content() -> None:
    ms = _msgs(("system", "abc"), ("user", "xy"))
    assert estimate_size(ms) == 5


# ── truncate_tool_results ───────────────────────────────────────────────


def test_truncate_tool_results_caps_oversized_tool_messages() -> None:
    ms = _msgs(
        ("system", "sys"),
        ("tool", "x" * 50),
        ("user", "u"),
    )
    out = truncate_tool_results(ms, budget=10)
    assert out[1].content.startswith("x" * 10)
    assert "truncated" in out[1].content
    # non-tool messages untouched
    assert out[0].content == "sys"
    assert out[2].content == "u"


def test_truncate_tool_results_skips_short_messages() -> None:
    ms = _msgs(("tool", "short"))
    out = truncate_tool_results(ms, budget=100)
    assert out[0].content == "short"


# ── sliding_window ──────────────────────────────────────────────────────


def test_sliding_window_keeps_system_and_recent() -> None:
    ms = _msgs(
        ("system", "S"),
        ("user", "u1"),
        ("assistant", "a1"),
        ("tool", "t1"),
        ("user", "u2"),
        ("assistant", "a2"),
        ("tool", "t2"),
    )
    out = sliding_window(ms, keep=3)
    # system kept, plus the last 3 non-system
    assert out[0].content == "S"
    assert [m.content for m in out[1:]] == ["u2", "a2", "t2"]


def test_sliding_window_passthrough_when_below_threshold() -> None:
    ms = _msgs(("system", "S"), ("user", "u"))
    assert sliding_window(ms, keep=5) == ms


def test_sliding_window_zero_keep_keeps_one() -> None:
    ms = _msgs(("system", "S"), ("user", "u1"), ("user", "u2"))
    out = sliding_window(ms, keep=0)
    # Defensive: min 1 message kept
    assert len(out) == 2  # system + 1


# ── inject_spec_state ───────────────────────────────────────────────────


def test_inject_spec_state_appends_user_message() -> None:
    ms = _msgs(("system", "S"))
    corpus = {"L1-VIS-001": _spec("L1-VIS-001", "L1", "Vision One")}
    out = inject_spec_state(ms, corpus)
    assert len(out) == 2
    assert out[1].role == "user"
    assert "L1-VIS-001" in out[1].content
    assert "Vision One" in out[1].content


def test_inject_spec_state_empty_corpus_noop() -> None:
    ms = _msgs(("system", "S"))
    assert inject_spec_state(ms, {}) == ms


# ── claude_compact ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_claude_compact_no_summarizer_noop() -> None:
    ms = _msgs(("system", "S"), ("user", "u"))
    out = await claude_compact(ms, summarizer=None)
    assert out == ms


@pytest.mark.asyncio
async def test_claude_compact_summarizes_older_turns() -> None:
    ms = _msgs(
        ("system", "S"),
        ("user", "u1"),
        ("assistant", "a1"),
        ("tool", "t1"),
        ("user", "u2"),
        ("assistant", "a2"),
        ("tool", "t2"),
        ("user", "u3"),
    )

    async def fake_summarizer(messages: list[Message]) -> str:
        return f"SUMMARY({len(messages)} msgs)"

    out = await claude_compact(ms, summarizer=fake_summarizer, keep_recent=3)
    # system preserved, summary inserted, last 3 kept
    assert out[0].role == "system"
    assert out[1].role == "user"
    assert "SUMMARY" in out[1].content
    assert [m.content for m in out[-3:]] == ["a2", "t2", "u3"]


@pytest.mark.asyncio
async def test_claude_compact_passthrough_below_keep() -> None:
    ms = _msgs(("system", "S"), ("user", "u"))

    async def fake_summarizer(_messages: list[Message]) -> str:
        return "X"

    out = await claude_compact(ms, summarizer=fake_summarizer, keep_recent=6)
    assert out == ms


# ── apply_policy ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_apply_policy_below_trigger_is_noop() -> None:
    ms = _msgs(("system", "S"), ("user", "u"))
    policy = CompressionPolicy(
        trigger_chars=10_000,
        strategies=["truncate_tool_results", "sliding_window"],
    )
    out = await apply_policy(ms, policy)
    assert out == ms


@pytest.mark.asyncio
async def test_apply_policy_runs_strategies_in_order() -> None:
    ms = _msgs(
        ("system", "S"),
        ("user", "u1"),
        ("tool", "x" * 5_000),
        ("user", "u2"),
        ("tool", "y" * 5_000),
        ("user", "u3"),
    )
    policy = CompressionPolicy(
        trigger_chars=100,
        tool_result_char_budget=100,
        window_keep=3,
        strategies=["truncate_tool_results", "sliding_window"],
    )
    out = await apply_policy(ms, policy)
    # tool messages got truncated then window trimmed to system + last 3
    assert out[0].role == "system"
    assert len(out) == 1 + 3
    for m in out:
        assert len(m.content) <= 200  # not 5000+


@pytest.mark.asyncio
async def test_apply_policy_inject_spec_state() -> None:
    ms = _msgs(("system", "S"), ("user", "u"))
    policy = CompressionPolicy(trigger_chars=0, strategies=["inject_spec_state"])
    corpus = {"L1-VIS-001": _spec("L1-VIS-001", "L1", "Vision")}
    out = await apply_policy(ms, policy, corpus=corpus)
    assert out[-1].role == "user"
    assert "L1-VIS-001" in out[-1].content
