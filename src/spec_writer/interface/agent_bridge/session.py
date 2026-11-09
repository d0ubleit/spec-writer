"""ChatBridgeSession — one bridge-owned chat session.

State machine
=============

A session is in one of three states::

    idle    — no turn in progress; ready for a new user message
    working — handle_user() task running; waiting on the LLM
    asking  — handle_user() task paused inside ask_user(); waiting on
              the human (i.e. the orchestrating agent) to answer

Transitions are driven by two entry points:

- ``send(message)``       — must be called in ``idle``; kicks off a new
                            turn and waits for the first interruption
                            point (``asking`` or ``done``).
- ``answer(text)``        — must be called in ``asking``; supplies the
                            answer to ``ask_user`` and waits for the
                            next interruption.

Both calls return the same shape (``TurnResult``) describing what just
happened so the caller can render text deltas, see which MCP tools
the agent invoked, and decide whether another ``send``/``answer`` is
needed.

``ask_user`` interception
=========================

The bridge installs a session-local :func:`set_answerer` callback that
publishes the agent's question to a queue and awaits the answer from a
companion queue. There is one global ``_answerer`` slot so only one
bridge session may be active at a time; the bridge ``open`` / ``close``
helpers track that invariant.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from spec_writer.agent.chat_agent import ChatAgent
from spec_writer.agent.chat_session import ChatSession
from spec_writer.agent.events import (
    EngineEvent,
    StatusEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from spec_writer.agent.tools.ask_user import set_answerer

_LOG = logging.getLogger(__name__)

SessionState = Literal["idle", "working", "asking", "closed"]


class SessionBusyError(RuntimeError):
    """Raised when ``send`` is called while a turn is still running or asking."""


class SessionAlreadyAskingError(RuntimeError):
    """Raised when ``send`` is called while the agent is asking — caller must ``answer`` first."""


class SessionNotAskingError(RuntimeError):
    """Raised when ``answer`` is called outside an ``asking`` state."""


class SessionClosedError(RuntimeError):
    """Raised when ``send``/``answer`` is called on a closed session."""


def _empty_tool_call_list() -> list[dict[str, Any]]:
    return []


@dataclass(frozen=True)
class TurnResult:
    """Return value of ``send`` / ``answer``.

    ``status``:
      - ``done``     — turn completed; agent finished speaking
      - ``asking``   — agent paused inside ``ask_user``; call ``answer``
      - ``error``    — turn failed (see ``error``); session usable again
      - ``thinking`` — timeout hit before any event; partial state in
                       ``reply`` and ``tool_calls``
    """

    status: Literal["done", "asking", "error", "thinking"]
    reply: str = ""
    asked: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=_empty_tool_call_list)
    error: str | None = None
    partial: bool = False


class ChatBridgeSession:
    """One bridge session.

    Holds the live :class:`ChatAgent` plus the queue plumbing that
    lets ``ask_user`` suspend the agent's turn and resume it later.
    """

    def __init__(self, *, agent: ChatAgent, chat_session: ChatSession) -> None:
        self._agent = agent
        self._chat = chat_session
        self._state: SessionState = "idle"
        self._lock = asyncio.Lock()
        # ask_user plumbing — single-shot per question
        self._pending_question: asyncio.Future[str] | None = None
        self._pending_answer: asyncio.Future[str] | None = None
        # Per-turn accumulators
        self._turn_text: list[str] = []
        self._turn_tool_calls: list[dict[str, Any]] = []
        self._turn_task: asyncio.Task[None] | None = None
        self._turn_done: asyncio.Event = asyncio.Event()
        self._turn_error: str | None = None

    # ── public state ────────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def session_id(self) -> str:
        return self._chat.session_id

    @property
    def chat_session(self) -> ChatSession:
        return self._chat

    # ── lifecycle ───────────────────────────────────────────────────

    def install_answerer(self) -> None:
        """Register the answerer hook in the global slot.

        Bridge surface code is expected to call this once after
        construction and clear via :meth:`uninstall_answerer` on close.
        """
        set_answerer(self._answerer)

    def uninstall_answerer(self) -> None:
        set_answerer(None)

    async def close(self) -> None:
        """Mark closed; cancel any in-flight turn task."""
        self._state = "closed"
        task = self._turn_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        # Resolve any waiting futures so coroutines don't leak.
        if self._pending_answer is not None and not self._pending_answer.done():
            self._pending_answer.set_exception(SessionClosedError("session closed"))
        if self._pending_question is not None and not self._pending_question.done():
            self._pending_question.set_exception(SessionClosedError("session closed"))

    # ── send / answer ───────────────────────────────────────────────

    async def send(self, message: str, *, timeout: float = 300.0) -> TurnResult:
        async with self._lock:
            if self._state == "closed":
                raise SessionClosedError("session closed")
            if self._state == "asking":
                raise SessionAlreadyAskingError("agent is waiting on ask_user; call answer() first")
            if self._state == "working":
                raise SessionBusyError("a turn is already in progress")
            # idle → kick off new turn
            self._reset_turn_accumulators()
            self._state = "working"
            self._turn_task = asyncio.create_task(self._run_turn(message))
            return await self._await_pause_or_done(timeout=timeout)

    async def answer(self, answer_text: str, *, timeout: float = 300.0) -> TurnResult:
        async with self._lock:
            if self._state == "closed":
                raise SessionClosedError("session closed")
            if self._state != "asking":
                raise SessionNotAskingError(
                    f"answer() requires state='asking', got {self._state!r}"
                )
            pending = self._pending_answer
            assert pending is not None, "asking state with no pending answer future"
            pending.set_result(answer_text)
            self._pending_answer = None
            self._pending_question = None
            self._state = "working"
            return await self._await_pause_or_done(timeout=timeout)

    async def cancel(self) -> None:
        """Abort the in-flight turn, if any."""
        async with self._lock:
            task = self._turn_task
            if task is None or task.done():
                self._state = "idle" if self._state != "closed" else "closed"
                return
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            self._state = "idle" if self._state != "closed" else "closed"

    # ── internals ───────────────────────────────────────────────────

    async def _await_pause_or_done(self, *, timeout: float) -> TurnResult:
        """Wait until the running turn either pauses on ask_user or finishes."""
        question_fut = self._pending_question
        done_fut: asyncio.Task[bool] = asyncio.ensure_future(self._turn_done.wait())
        waiters: list[asyncio.Future[Any]] = [done_fut]
        if question_fut is not None:
            waiters.append(question_fut)

        try:
            await asyncio.wait_for(
                asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            return TurnResult(
                status="thinking",
                reply="".join(self._turn_text),
                tool_calls=list(self._turn_tool_calls),
                partial=True,
            )

        # Check question first — if asked, the run task may still be
        # alive waiting on the answer future, which is fine.
        if question_fut is not None and question_fut.done() and not done_fut.done():
            asked = question_fut.result()
            self._state = "asking"
            return TurnResult(
                status="asking",
                reply="".join(self._turn_text),
                asked=asked,
                tool_calls=list(self._turn_tool_calls),
            )

        # Otherwise the turn finished.
        if self._turn_error is not None:
            self._state = "idle"
            return TurnResult(
                status="error",
                reply="".join(self._turn_text),
                tool_calls=list(self._turn_tool_calls),
                error=self._turn_error,
            )
        self._state = "idle"
        return TurnResult(
            status="done",
            reply="".join(self._turn_text),
            tool_calls=list(self._turn_tool_calls),
        )

    def _reset_turn_accumulators(self) -> None:
        self._turn_text = []
        self._turn_tool_calls = []
        self._turn_error = None
        self._turn_done = asyncio.Event()
        loop = asyncio.get_event_loop()
        self._pending_question = loop.create_future()
        self._pending_answer = loop.create_future()

    async def _answerer(self, question: str) -> str:
        """Bridge-side ``ask_user`` callback.

        Publishes the question to ``_pending_question``, then awaits
        the orchestrator-supplied answer via ``_pending_answer``.
        """
        question_fut = self._pending_question
        answer_fut = self._pending_answer
        if question_fut is None or answer_fut is None:
            # No turn in progress somehow — surface a stub so the
            # agent doesn't crash.
            return "(no answerer available — proceed with reasonable defaults)"
        if not question_fut.done():
            question_fut.set_result(question)
        return await answer_fut

    async def _run_turn(self, message: str) -> None:
        try:
            async for event in self._agent.handle_user(message):
                self._observe_event(event)
        except asyncio.CancelledError:
            self._turn_error = "cancelled"
            raise
        except Exception as exc:
            _LOG.exception("agent_bridge: turn failed")
            self._turn_error = f"{type(exc).__name__}: {exc}"
        finally:
            # If we exited without the question ever being resolved,
            # cancel the pending question future so waiters wake up.
            if self._pending_question is not None and not self._pending_question.done():
                self._pending_question.cancel()
            self._turn_done.set()

    def _observe_event(self, event: EngineEvent) -> None:
        if isinstance(event, TextDeltaEvent):
            self._turn_text.append(event.text)
        elif isinstance(event, ToolCallEvent):
            self._turn_tool_calls.append(
                {
                    "tool": event.tool,
                    "args": event.args,
                    "ok": None,
                    "result": None,
                }
            )
        elif isinstance(event, ToolResultEvent):
            # Pair with the most recent matching tool call.
            for entry in reversed(self._turn_tool_calls):
                if entry["tool"] == event.tool and entry["ok"] is None:
                    entry["ok"] = True
                    entry["result"] = event.result
                    break
            else:
                self._turn_tool_calls.append(
                    {
                        "tool": event.tool,
                        "args": {},
                        "ok": True,
                        "result": event.result,
                    }
                )
        elif isinstance(event, StatusEvent):
            if event.phase == "error":
                details = event.details or {}
                self._turn_error = str(details.get("reason", "unknown error"))


__all__ = [
    "ChatBridgeSession",
    "SessionAlreadyAskingError",
    "SessionBusyError",
    "SessionClosedError",
    "SessionNotAskingError",
    "SessionState",
    "TurnResult",
]
