"""Stdio MCP server exposing the bridge tools.

Entry point: ``python -m spec_writer.interface.agent_bridge``.

Tools are advertised via :meth:`Server.list_tools` and dispatched
through :meth:`Server.call_tool`; the wire format is the standard MCP
JSON-RPC over stdio. Errors raised by handlers are returned as
tool-result text payloads (``{"ok": false, "error": ...}``) — the
server itself never crashes on a single tool failure.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.lowlevel import Server
from mcp.types import TextContent, Tool

from spec_writer.interface.agent_bridge import tools as bridge_tools
from spec_writer.interface.agent_bridge.tools import BridgeError, SessionManager

_LOG = logging.getLogger(__name__)

# Tool definitions — kept inline so the schema stays next to the handler.
_TOOL_DEFINITIONS: list[Tool] = [
    Tool(
        name="chat_open",
        description=(
            "Open a new spec-writer chat session against a project root. "
            "Returns the session_id; single-session: must chat_close the prior "
            "session before opening another."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_root": {
                    "type": "string",
                    "description": "Absolute path to the project root (must contain spec-editor/).",
                },
                "direction": {
                    "type": "string",
                    "enum": ["fwd", "rev"],
                    "description": "Workflow direction: fwd (sources→code, default) or rev (code→sources).",
                    "default": "fwd",
                },
            },
            "required": ["project_root"],
        },
    ),
    Tool(
        name="chat_send",
        description=(
            "Send a new user message to the spec-writer agent and wait for the "
            "first interruption (agent finishes speaking or hits ask_user). "
            "Returns {status, reply, asked?, tool_calls, partial}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "message": {"type": "string", "description": "Free-form user message."},
                "timeout": {
                    "type": "number",
                    "description": "Seconds to wait before returning status='thinking'.",
                    "default": 300,
                },
            },
            "required": ["session_id", "message"],
        },
    ),
    Tool(
        name="chat_answer",
        description=(
            "Supply the answer to a pending ask_user prompt. The agent resumes "
            "the current turn and continues until the next interruption."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "answer": {"type": "string", "description": "Answer text for ask_user."},
                "timeout": {"type": "number", "default": 300},
            },
            "required": ["session_id", "answer"],
        },
    ),
    Tool(
        name="chat_cancel",
        description="Abort the in-flight turn (does not close the session).",
        inputSchema={
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    ),
    Tool(
        name="chat_resume",
        description=(
            "Open a session by re-attaching to an existing on-disk session log. "
            "User and agent messages are replayed into history; tool results are not."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_root": {"type": "string"},
                "session_id": {
                    "type": "string",
                    "description": "Existing session_id under _meta/sessions/.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["fwd", "rev"],
                    "default": "fwd",
                },
            },
            "required": ["project_root", "session_id"],
        },
    ),
    Tool(
        name="chat_history",
        description="Render the session transcript (markdown by default; json for raw events).",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "format": {
                    "type": "string",
                    "enum": ["markdown", "json"],
                    "default": "markdown",
                },
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="chat_close",
        description="End a session: cancels any in-flight turn and releases the answerer slot.",
        inputSchema={
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    ),
    Tool(
        name="chat_list_sessions",
        description="List currently open bridge sessions with their state.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


_HANDLERS = {
    "chat_open": bridge_tools.chat_open,
    "chat_send": bridge_tools.chat_send,
    "chat_answer": bridge_tools.chat_answer,
    "chat_cancel": bridge_tools.chat_cancel,
    "chat_resume": bridge_tools.chat_resume,
    "chat_history": bridge_tools.chat_history,
    "chat_close": bridge_tools.chat_close,
    "chat_list_sessions": bridge_tools.chat_list_sessions,
}


def build_server() -> tuple[Server[Any], SessionManager]:
    """Construct the MCP Server with bridge tools wired in.

    Returns ``(server, manager)`` — the manager is exposed so tests
    can drive it directly without spinning up stdio transport.
    """
    server: Server[Any] = Server("spec-writer-bridge")
    manager = SessionManager()

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return list(_TOOL_DEFINITIONS)

    @server.call_tool()
    async def _call_tool(name: str, args: dict[str, Any]) -> list[TextContent]:
        handler = _HANDLERS.get(name)
        if handler is None:
            payload = {"ok": False, "error": f"unknown tool {name!r}"}
        else:
            try:
                payload = await handler(manager, args)
            except BridgeError as exc:
                payload = {"ok": False, "error": str(exc)}
            except Exception as exc:
                _LOG.exception("bridge tool %s raised", name)
                payload = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    return server, manager


async def run_stdio() -> None:
    """Start the stdio MCP server loop."""
    from mcp.server.stdio import stdio_server

    logging.basicConfig(level=logging.INFO)
    server, _ = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


__all__ = ["build_server", "run_stdio"]
