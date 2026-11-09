"""agent_bridge — expose the chat-mode ChatAgent over MCP (stdio).

Lets another agent (e.g. Claude Code) drive a spec-writer chat session
through the same user-facing surface a human would use: open a session,
send free-form messages, answer ``ask_user`` prompts, inspect tool
calls. The bridge holds at most one live session per ``session_id`` in
memory; sessions also persist to ``_meta/sessions/<id>.json`` so they
can be resumed after a bridge restart.
"""

from spec_writer.interface.agent_bridge.session import (
    ChatBridgeSession,
    SessionAlreadyAskingError,
    SessionBusyError,
    SessionNotAskingError,
)

__all__ = [
    "ChatBridgeSession",
    "SessionAlreadyAskingError",
    "SessionBusyError",
    "SessionNotAskingError",
]
