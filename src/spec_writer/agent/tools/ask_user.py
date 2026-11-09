"""``ask_user`` tool — interactive clarification, used by ``dialog`` mode.

Phase 2 behaviour: in batch (non-chat) runs the tool errors out so the
agent's prompt can recover without blocking — the user instructed
"batch is non-interactive". A chat session installs an ``answerer``
callback via :func:`set_answerer`; when present, ``ask_user`` invokes
it (awaiting if needed) and returns the answer string.

The CLI ``spec-writer chat`` REPL (S2.9) registers a real answerer
that surfaces the question to the user and waits for input.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry

AnswerFn = Callable[[str], str | Awaitable[str]]


class AskUserError(RuntimeError):
    pass


_answerer: AnswerFn | None = None


def set_answerer(fn: AnswerFn | None) -> None:
    """Install (or clear) the chat-side answerer callback.

    The callback receives the agent's question string and returns the
    user's answer. The chat REPL installs one for the duration of a
    session; batch runs leave this unset and the tool errors out.
    """
    global _answerer
    _answerer = fn


def get_answerer() -> AnswerFn | None:
    return _answerer


async def ask_user(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    del ctx  # unused — ask_user is mode-not-context dependent
    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        raise AskUserError("ask_user requires non-empty string 'question'")

    answerer = _answerer
    if answerer is None:
        return {
            "ok": False,
            "error": (
                "ask_user is only available in dialog/chat mode; in batch runs the agent "
                "must either proceed with a reasonable default or defer the question "
                "to a refine pass with --instruction"
            ),
        }

    result = answerer(question)
    if inspect.isawaitable(result):
        answer = await result
    else:
        answer = result
    return {"ok": True, "answer": str(answer)}


def register(registry: ToolRegistry) -> None:
    registry.register("ask_user", ask_user)


__all__ = ["AnswerFn", "AskUserError", "ask_user", "get_answerer", "register", "set_answerer"]
