"""``claude-subscription`` LLMClient via ``claude-agent-sdk``.

Stateful adapter — the client remembers its CLI-side ``session_id``
after the first turn and resumes that session on subsequent calls
. The CLI maintains the conversation server-side
so each turn pays only for the delta + cached prefix, not the full
re-sent history.

Auto-detects history drift (compression strategy rewrote messages) by
fingerprinting submitted messages — a mismatch forces a fresh session.

The unit tests mock the SDK to keep the suite hermetic.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Literal, cast

from spec_writer.agent.events import (
    EngineEvent,
    RateLimitEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from spec_writer.agent.llm_client import Message, ToolSpec
from spec_writer.types import TokenUsage


def _flatten_messages(messages: list[Message]) -> str:
    parts: list[str] = []
    for msg in messages:
        prefix = msg.role.upper()
        parts.append(f"[{prefix}]\n{msg.content}")
    return "\n\n".join(parts)


def _extract_system_prompt(
    messages: list[Message], extra_system_prompt: str | None
) -> tuple[list[Message], str | None]:
    """Split system messages out so they can ride via ``ClaudeAgentOptions.system_prompt``.

    The Claude CLI applies a stable ``cache_control`` breakpoint to the
    ``--system-prompt`` argument; user-message-embedded system content
    invalidates that cache each turn. See the design rationale.
    """
    system_chunks: list[str] = [m.content for m in messages if m.role == "system"]
    if extra_system_prompt:
        system_chunks.append(extra_system_prompt)
    non_system = [m for m in messages if m.role != "system"]
    combined = "\n\n".join(c for c in system_chunks if c) or None
    return non_system, combined


def _fingerprint_message(m: Message) -> str:
    h = hashlib.sha256()
    h.update(m.role.encode())
    h.update(b"\0")
    h.update(m.content.encode())
    return h.hexdigest()


def _extract_session_id(message: Any) -> str | None:
    """Pull the CLI's ``session_id`` from an SDK message if present."""
    cls_name = type(message).__name__
    if cls_name == "SystemMessage":
        if getattr(message, "subtype", "") == "init":
            data: Any = getattr(message, "data", {})
            if isinstance(data, dict):
                sid: Any = cast(dict[str, Any], data).get("session_id")
                return sid if isinstance(sid, str) else None
    elif cls_name == "ResultMessage":
        result_sid: Any = getattr(message, "session_id", None)
        return result_sid if isinstance(result_sid, str) else None
    return None


def _extract_usage(message: Any) -> TokenUsage | None:
    """Pull token usage from a ``ResultMessage`` if present .

    SDK's ``ResultMessage.usage`` is a dict like::

        {
            "input_tokens": 123,
            "output_tokens": 45,
            "cache_creation_input_tokens": 678,
            "cache_read_input_tokens": 910,
        }

    Returns ``None`` for non-Result messages OR when ``usage`` is
    missing/malformed.
    """
    if type(message).__name__ != "ResultMessage":
        return None
    raw: Any = getattr(message, "usage", None)
    if not isinstance(raw, dict):
        return None
    raw_dict = cast(dict[str, Any], raw)

    def _int(key: str) -> int:
        value = raw_dict.get(key)
        return value if isinstance(value, int) else 0

    inp = _int("input_tokens")
    out = _int("output_tokens")
    return TokenUsage(
        input=inp,
        output=out,
        total=inp + out,
        cache_creation_input=_int("cache_creation_input_tokens"),
        cache_read_input=_int("cache_read_input_tokens"),
    )


def _translate_block(block: Any) -> EngineEvent | None:
    # claude_agent_sdk content blocks
    cls_name = type(block).__name__
    if cls_name == "TextBlock":
        text = getattr(block, "text", "") or ""
        return TextDeltaEvent(text=text)
    if cls_name == "ToolUseBlock":
        return ToolCallEvent(
            tool=str(getattr(block, "name", "")),
            args=cast(dict[str, Any], getattr(block, "input", {}) or {}),
        )
    if cls_name == "ToolResultBlock":
        return ToolResultEvent(
            tool=str(getattr(block, "tool_use_id", "")),
            result=getattr(block, "content", None),
        )
    return None


def _translate_message(message: Any) -> list[EngineEvent]:
    """Translate a single SDK message into zero or more EngineEvents."""
    cls_name = type(message).__name__
    events: list[EngineEvent] = []
    if cls_name == "AssistantMessage":
        for block in getattr(message, "content", []) or []:
            event = _translate_block(block)
            if event is not None:
                events.append(event)
    elif cls_name == "RateLimitEvent":
        info = getattr(message, "rate_limit_info", None)
        raw_status = getattr(info, "status", "warning") if info is not None else "warning"
        status: Literal["warning", "rejected"] = (
            "rejected" if raw_status == "rejected" else "warning"
        )
        resets_at = getattr(info, "resets_at", None) if info is not None else None
        events.append(
            RateLimitEvent(
                status=status,
                resets_at=resets_at if isinstance(resets_at, datetime) else None,
            )
        )
    elif cls_name == "ResultMessage":
        # End-of-turn message carries token usage . Session-id
        # extraction lives separately; here we surface usage for the
        # ledger via UsageEvent.
        usage = _extract_usage(message)
        if usage is not None and (
            usage.input or usage.output or usage.cache_creation_input or usage.cache_read_input
        ):
            events.append(UsageEvent(usage=usage))
    # SystemMessage / StreamEvent — caller handles session_id; they
    # produce no user-facing events.
    return events


class ClaudeSubscriptionClient:
    """LLMClient implementation backed by ``claude-agent-sdk``.

    Custom tools must be registered as an SDK MCP server (see
    :mod:`spec_writer.agent.mcp_tools`) and passed via ``mcp_servers`` +
    ``allowed_tools``; the SDK then handles the full tool-call round-trip
    internally and the runtime only observes events.

    Stateful: holds the CLI-side ``session_id`` between :meth:`chat`
    invocations so the conversation lives server-side. Pass
    ``enable_session_resume=False`` to opt out (e.g. for legacy tests).
    """

    name = "claude-subscription"

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        *,
        mcp_servers: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        enable_session_resume: bool = True,
    ) -> None:
        self._model = model
        self._mcp_servers = mcp_servers or {}
        self._allowed_tools = allowed_tools
        self._enable_resume = enable_session_resume
        self._session_id: str | None = None
        self._sent_fingerprints: list[str] = []

    def reset_session(self) -> None:
        """Forget the current CLI session — next ``chat`` starts a fresh one.

        Callers should invoke this when they intentionally rewrite the
        conversation in a way the auto-drift detector cannot catch.
        """
        self._session_id = None
        self._sent_fingerprints = []

    def _detect_history_drift(self, current_non_system: list[Message]) -> bool:
        """Return True if ``current`` does not extend our last submission."""
        if len(current_non_system) < len(self._sent_fingerprints):
            return True
        for i, prior_fp in enumerate(self._sent_fingerprints):
            if _fingerprint_message(current_non_system[i]) != prior_fp:
                return True
        return False

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int = 4096,
        response_schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[EngineEvent]:
        # Lazy import keeps the unit tests fast and lets test code
        # monkeypatch the symbol cleanly.
        from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore[import-untyped]

        non_system, sp_text = _extract_system_prompt(messages, system_prompt)

        # If the caller's history no longer matches what we last submitted
        # (e.g. compression rewrote it), the server-side conversation is
        # stale — restart with a fresh session.
        if (
            self._enable_resume
            and self._session_id is not None
            and self._detect_history_drift(non_system)
        ):
            self.reset_session()

        is_resume = self._enable_resume and self._session_id is not None
        if is_resume:
            delta = non_system[len(self._sent_fingerprints) :]
            if not delta:
                # Caller passed an unchanged history — nothing to send.
                del tools, max_tokens, response_schema
                return
            prompt = _flatten_messages(delta)
        else:
            prompt = _flatten_messages(non_system)

        options_kwargs: dict[str, Any] = {"model": self._model}
        if is_resume:
            options_kwargs["resume"] = self._session_id
            # Don't re-send system_prompt on resume — CLI has it stored.
        elif sp_text is not None:
            options_kwargs["system_prompt"] = sp_text
        if self._mcp_servers:
            options_kwargs["mcp_servers"] = self._mcp_servers
        if self._allowed_tools is not None:
            options_kwargs["allowed_tools"] = list(self._allowed_tools)

        options = ClaudeAgentOptions(**options_kwargs)

        async for message in query(prompt=prompt, options=options):
            if self._session_id is None and self._enable_resume:
                sid = _extract_session_id(message)
                if sid:
                    self._session_id = sid
            for event in _translate_message(message):
                yield event

        # Record what we submitted so we can detect drift / send delta next turn.
        if self._enable_resume:
            self._sent_fingerprints = [_fingerprint_message(m) for m in non_system]

        # Mark the tool/max_tokens/response_schema params as intentionally
        # consumed by the runtime layer (Phase 2 doesn't pipe them through).
        del tools, max_tokens, response_schema

    async def summarize(self, messages: list[Message]) -> str:
        """Compress older conversation into a single plain-text summary.

        Used by :func:`agent.compression.claude_compact` when the chat
        history outgrows the trigger threshold. Other providers without
        a ``summarize`` method cause the strategy to be skipped (see
        :class:`agent.llm_client.SupportsSummarize`).

        Summarisation is a fresh one-shot call — does NOT reuse the
        session-resume state (it would pollute the server-side conversation
        with the summariser's prompt). Callers responsible for invoking
        :meth:`reset_session` after compaction rewrote the history;
        :meth:`chat` also auto-detects drift via fingerprints.
        """
        from claude_agent_sdk import ClaudeAgentOptions, query  # type: ignore[import-untyped]

        body = _flatten_messages(messages)
        prompt = (
            "Summarize the following conversation between a software-spec agent and a user. "
            "Keep the facts the agent has established (chosen layers, spec ids written, "
            "user clarifications, open questions, validation errors still pending). "
            "Drop pleasantries and verbatim tool-call payloads. Output ONE plain-text "
            "paragraph, no headings, no markdown. Conversation:\n\n" + body
        )

        options = ClaudeAgentOptions(model=self._model)
        chunks: list[str] = []
        async for message in query(prompt=prompt, options=options):
            for event in _translate_message(message):
                if isinstance(event, TextDeltaEvent):
                    chunks.append(event.text)
        return "".join(chunks).strip()


__all__ = ["ClaudeSubscriptionClient"]
