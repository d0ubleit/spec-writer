"""Conversation-history compression strategies.

The agent runtime sometimes accumulates long traces (especially during
a feedback loop). Without compression, every turn re-sends the full
history and burns tokens linearly. This module ships four orthogonal
transformations that callers can chain via :func:`apply_policy`:

- :func:`truncate_tool_results` — collapse oversized tool result
  messages to a fixed character budget plus an elision marker.
- :func:`sliding_window` — keep the system prompt + the last ``keep``
  messages, dropping the middle.
- :func:`inject_spec_state` — append a compact "current corpus"
  snapshot as a fresh ``user`` message so the agent has up-to-date
  state without re-reading the whole history.
- :func:`claude_compact` — provider-specific: when the active
  :class:`LLMClient` implements :class:`SupportsSummarize`, summarize
  older history into one message. Otherwise the strategy is skipped
  with a warning and other model-agnostic strategies still run.

The runtime calls :func:`apply_policy` between LLM turns when the
estimated message-list size exceeds ``CompressionPolicy.trigger_chars``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from spec_writer.agent.llm_client import Message, SupportsSummarize
from spec_writer.types import Spec

_LOG = logging.getLogger(__name__)

SummarizerFn = Callable[[list[Message]], Awaitable[str]]


def _empty_strategies() -> list[CompressionStrategy]:
    return []


CompressionStrategy = Literal[
    "truncate_tool_results",
    "sliding_window",
    "inject_spec_state",
    "claude_compact",
]


@dataclass(frozen=True)
class CompressionPolicy:
    """Declarative knobs for the compression pipeline."""

    trigger_chars: int = 60_000
    """Skip compression entirely below this total message-content size."""

    tool_result_char_budget: int = 2_000
    """Per tool-result message ceiling for truncate_tool_results."""

    window_keep: int = 12
    """How many recent messages sliding_window retains (system always kept)."""

    strategies: list[CompressionStrategy] = field(default_factory=_empty_strategies)
    """Order matters — strategies are applied left to right."""


# ── individual transforms ───────────────────────────────────────────────


def estimate_size(messages: list[Message]) -> int:
    """Cheap byte-equivalent estimate (sum of len(content))."""
    return sum(len(m.content) for m in messages)


def truncate_tool_results(messages: list[Message], budget: int) -> list[Message]:
    """Cap ``role='tool'`` message contents to ``budget`` chars + elision."""
    out: list[Message] = []
    for msg in messages:
        if msg.role == "tool" and len(msg.content) > budget:
            head = msg.content[:budget].rstrip()
            elided = len(msg.content) - budget
            new_content = f"{head}\n... [truncated {elided} chars] ..."
            out.append(Message(role=msg.role, content=new_content))
        else:
            out.append(msg)
    return out


def sliding_window(messages: list[Message], keep: int) -> list[Message]:
    """Keep the system prompt(s) and the last ``keep`` non-system messages."""
    if keep <= 0:
        keep = 1
    systems = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    if len(rest) <= keep:
        return list(messages)
    kept_rest = rest[-keep:]
    return [*systems, *kept_rest]


def inject_spec_state(messages: list[Message], corpus: dict[str, Spec]) -> list[Message]:
    """Append a snapshot of the corpus as a fresh user message.

    Useful after we've truncated or windowed: the agent loses the
    accumulated tool results, so we hand it the current state in one
    compact block instead.
    """
    if not corpus:
        return list(messages)

    lines: list[str] = ["Current corpus state (id — layer — title):"]
    for spec_id in sorted(corpus):
        spec = corpus[spec_id]
        lines.append(f"  {spec.id}  {spec.layer}  {spec.title}")

    snapshot = "\n".join(lines)
    return [*messages, Message(role="user", content=snapshot)]


async def claude_compact(
    messages: list[Message],
    summarizer: SummarizerFn | None,
    keep_recent: int = 6,
) -> list[Message]:
    """Replace older messages with a Claude-generated summary.

    The summarizer is optional — without one this is a no-op (returns
    ``messages`` unchanged). When supplied, it receives the messages to
    be summarized and must return a plain-text summary string. The
    output preserves system prompts, then inserts a single ``user``
    message containing the summary, then keeps the last ``keep_recent``
    messages.
    """
    if summarizer is None:
        return list(messages)

    systems = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    if len(rest) <= keep_recent:
        return list(messages)

    to_summarize = rest[:-keep_recent]
    summary = await summarizer(to_summarize)
    return [
        *systems,
        Message(role="user", content=f"[Previous conversation summary]\n{summary}"),
        *rest[-keep_recent:],
    ]


# ── pipeline ────────────────────────────────────────────────────────────


def summarizer_for(llm: object) -> SummarizerFn | None:
    """Return ``llm.summarize`` bound method if the provider supports it.

    Used by :func:`apply_policy` to gate the ``claude_compact`` strategy
    on provider capability. Returning ``None`` causes the strategy to
    be skipped (with a logged warning, once per policy run).
    """
    if isinstance(llm, SupportsSummarize):
        return llm.summarize
    return None


async def apply_policy(
    messages: list[Message],
    policy: CompressionPolicy,
    *,
    corpus: dict[str, Spec] | None = None,
    summarizer: SummarizerFn | None = None,
) -> list[Message]:
    """Run the requested strategies in declaration order.

    Strategy order is intentional: ``truncate_tool_results`` keeps every
    message present (just smaller), ``sliding_window`` drops messages,
    ``inject_spec_state`` re-anchors the agent after either of the
    above, and ``claude_compact`` is the last resort.

    Compression is skipped entirely when ``estimate_size`` is below
    ``policy.trigger_chars``.
    """
    if estimate_size(messages) < policy.trigger_chars:
        return list(messages)

    out = list(messages)
    for strategy in policy.strategies:
        if strategy == "truncate_tool_results":
            out = truncate_tool_results(out, policy.tool_result_char_budget)
        elif strategy == "sliding_window":
            out = sliding_window(out, policy.window_keep)
        elif strategy == "inject_spec_state":
            out = inject_spec_state(out, corpus or {})
        elif strategy == "claude_compact":
            if summarizer is None:
                _LOG.warning(
                    "compression: claude_compact strategy skipped — "
                    "active LLM provider has no summarize() method"
                )
                continue
            out = await claude_compact(out, summarizer)
    return out


__all__ = [
    "CompressionPolicy",
    "CompressionStrategy",
    "SummarizerFn",
    "apply_policy",
    "claude_compact",
    "estimate_size",
    "inject_spec_state",
    "sliding_window",
    "summarizer_for",
    "truncate_tool_results",
]
