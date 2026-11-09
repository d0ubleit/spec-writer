"""Bridge tool handlers — one per MCP tool, plus the in-memory
:class:`SessionManager` that owns live :class:`ChatBridgeSession`-s.

Tools
=====

- ``chat_open(project_root, direction="fwd")`` — create a new session
- ``chat_send(session_id, message, timeout?)`` — send a new user
  message, wait for the agent to pause or finish
- ``chat_answer(session_id, answer, timeout?)`` — resume a paused
  ``ask_user`` with the answer text
- ``chat_resume(project_root, session_id, direction="fwd")`` — restore
  a prior session from its on-disk log
- ``chat_cancel(session_id)`` — abort the in-flight turn
- ``chat_history(session_id, format="markdown")`` — transcript
- ``chat_close(session_id)`` — end a session
- ``chat_list_sessions()`` — diagnostic

Single-session invariant
========================

``set_answerer`` is a process-wide singleton, so only one session can
be the active ``ask_user`` answerer at any moment. The manager will
refuse a second ``chat_open`` until the first is closed. Multi-session
support is a future refactor.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from spec_writer.agent.chat_agent_factory import (
    UnsupportedProviderError,
    build_chat_agent,
)
from spec_writer.agent.chat_session import ChatSession, load_session
from spec_writer.engine.project_context import NoProjectRootError, ProjectContext
from spec_writer.interface.agent_bridge.session import (
    ChatBridgeSession,
    SessionAlreadyAskingError,
    SessionBusyError,
    SessionClosedError,
    SessionNotAskingError,
    TurnResult,
)
from spec_writer.methodology.loader import (
    MethodologyLoadError,
    load_methodology_for_project,
)

_LOG = logging.getLogger(__name__)


class BridgeError(RuntimeError):
    """Surfaced via tool result instead of crashing the MCP server."""


# ── manager ─────────────────────────────────────────────────────────────


class SessionManager:
    """Owns the dict of live sessions + enforces the single-answerer invariant."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatBridgeSession] = {}
        self._active_answerer_id: str | None = None
        self._lock = asyncio.Lock()

    async def open_new(self, project_root: Path, direction: str) -> ChatBridgeSession:
        async with self._lock:
            if self._active_answerer_id is not None:
                raise BridgeError(
                    f"another session ({self._active_answerer_id!r}) is already active; "
                    f"call chat_close on it first (single-answerer invariant)"
                )
            session = self._build_session(project_root, direction, resume_id=None)
            self._sessions[session.session_id] = session
            session.install_answerer()
            self._active_answerer_id = session.session_id
            return session

    async def open_resume(
        self, project_root: Path, session_id: str, direction: str
    ) -> ChatBridgeSession:
        async with self._lock:
            if self._active_answerer_id is not None:
                raise BridgeError(
                    f"another session ({self._active_answerer_id!r}) is already active"
                )
            session = self._build_session(project_root, direction, resume_id=session_id)
            self._sessions[session.session_id] = session
            session.install_answerer()
            self._active_answerer_id = session.session_id
            return session

    def get(self, session_id: str) -> ChatBridgeSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise BridgeError(f"unknown session_id {session_id!r}") from exc

    async def close(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                raise BridgeError(f"unknown session_id {session_id!r}")
            session.uninstall_answerer()
            await session.close()
            if self._active_answerer_id == session_id:
                self._active_answerer_id = None

    def list_sessions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for sid, sess in self._sessions.items():
            out.append(
                {
                    "session_id": sid,
                    "state": sess.state,
                    "project_root": str(sess.chat_session.project_ctx.root),
                    "event_count": len(sess.chat_session.events),
                    "active_answerer": sid == self._active_answerer_id,
                }
            )
        return out

    def _build_session(
        self,
        project_root: Path,
        direction: str,
        *,
        resume_id: str | None,
    ) -> ChatBridgeSession:
        try:
            project_ctx = ProjectContext.from_cwd(project_root)
        except NoProjectRootError as exc:
            raise BridgeError(str(exc)) from exc

        try:
            methodology = load_methodology_for_project(project_ctx)
        except MethodologyLoadError as exc:
            raise BridgeError(str(exc)) from exc

        if resume_id is not None:
            try:
                chat_session = load_session(project_ctx, resume_id)
            except FileNotFoundError as exc:
                raise BridgeError(
                    f"session {resume_id!r} not found under {project_ctx.meta / 'sessions'}"
                ) from exc
        else:
            chat_session = ChatSession(project_ctx=project_ctx)
            chat_session.log_note("bridge_started", direction=direction)

        try:
            agent = build_chat_agent(
                project_ctx,
                methodology,
                chat_session,
                direction=direction,
                resume_from=resume_id is not None,
            )
        except UnsupportedProviderError as exc:
            raise BridgeError(str(exc)) from exc

        return ChatBridgeSession(agent=agent, chat_session=chat_session)


# ── tool handlers ───────────────────────────────────────────────────────


async def chat_open(manager: SessionManager, args: dict[str, Any]) -> dict[str, Any]:
    project_root_arg = args.get("project_root")
    if not isinstance(project_root_arg, str) or not project_root_arg:
        raise BridgeError("chat_open requires a non-empty 'project_root' string")
    direction = _normalise_direction(args.get("direction"))
    session = await manager.open_new(Path(project_root_arg), direction)
    return {
        "ok": True,
        "session_id": session.session_id,
        "project_root": str(session.chat_session.project_ctx.root),
        "direction": direction,
    }


async def chat_send(manager: SessionManager, args: dict[str, Any]) -> dict[str, Any]:
    session_id = _require_str(args, "session_id")
    message = _require_str(args, "message")
    timeout = _coerce_timeout(args.get("timeout"), default=300.0)
    session = manager.get(session_id)
    try:
        result = await session.send(message, timeout=timeout)
    except (SessionBusyError, SessionAlreadyAskingError, SessionClosedError) as exc:
        raise BridgeError(str(exc)) from exc
    return _turn_to_dict(result, session_id)


async def chat_answer(manager: SessionManager, args: dict[str, Any]) -> dict[str, Any]:
    session_id = _require_str(args, "session_id")
    answer = _require_str(args, "answer")
    timeout = _coerce_timeout(args.get("timeout"), default=300.0)
    session = manager.get(session_id)
    try:
        result = await session.answer(answer, timeout=timeout)
    except (SessionNotAskingError, SessionClosedError) as exc:
        raise BridgeError(str(exc)) from exc
    return _turn_to_dict(result, session_id)


async def chat_cancel(manager: SessionManager, args: dict[str, Any]) -> dict[str, Any]:
    session_id = _require_str(args, "session_id")
    session = manager.get(session_id)
    await session.cancel()
    return {"ok": True, "session_id": session_id, "state": session.state}


async def chat_resume(manager: SessionManager, args: dict[str, Any]) -> dict[str, Any]:
    project_root_arg = _require_str(args, "project_root")
    session_id = _require_str(args, "session_id")
    direction = _normalise_direction(args.get("direction"))
    session = await manager.open_resume(Path(project_root_arg), session_id, direction)
    return {
        "ok": True,
        "session_id": session.session_id,
        "project_root": str(session.chat_session.project_ctx.root),
        "direction": direction,
        "restored_events": len(session.chat_session.events),
    }


async def chat_history(manager: SessionManager, args: dict[str, Any]) -> dict[str, Any]:
    session_id = _require_str(args, "session_id")
    fmt = str(args.get("format") or "markdown").lower()
    session = manager.get(session_id)
    if fmt == "markdown":
        return {"ok": True, "format": "markdown", "transcript": _render_markdown(session)}
    if fmt == "json":
        return {
            "ok": True,
            "format": "json",
            "events": session.chat_session.events,
        }
    raise BridgeError(f"unsupported history format {fmt!r}; use 'markdown' or 'json'")


async def chat_close(manager: SessionManager, args: dict[str, Any]) -> dict[str, Any]:
    session_id = _require_str(args, "session_id")
    await manager.close(session_id)
    return {"ok": True, "session_id": session_id}


async def chat_list_sessions(manager: SessionManager, args: dict[str, Any]) -> dict[str, Any]:
    del args
    return {"ok": True, "sessions": manager.list_sessions()}


# ── helpers ─────────────────────────────────────────────────────────────


def _require_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise BridgeError(f"missing or empty {key!r} string argument")
    return value


def _normalise_direction(raw: Any) -> str:
    if raw is None:
        return "fwd"
    text = str(raw).lower()
    if text not in ("fwd", "rev"):
        raise BridgeError(f"direction must be 'fwd' or 'rev', got {raw!r}")
    return text


def _coerce_timeout(raw: Any, *, default: float) -> float:
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise BridgeError(f"timeout must be a number, got {raw!r}") from exc
    if value <= 0:
        raise BridgeError("timeout must be positive")
    return value


def _turn_to_dict(result: TurnResult, session_id: str) -> dict[str, Any]:
    payload = asdict(result)
    payload["session_id"] = session_id
    payload["ok"] = result.status != "error"
    return payload


def _render_markdown(session: ChatBridgeSession) -> str:
    """Render the on-disk session events as a human-readable transcript."""
    lines: list[str] = [
        f"# Session {session.session_id}",
        "",
        f"- project: `{session.chat_session.project_ctx.root}`",
        f"- started: {session.chat_session.started_at}",
        f"- state: {session.state}",
        "",
        "---",
        "",
    ]
    for event in session.chat_session.events:
        kind = event.get("kind")
        ts = event.get("ts", "")
        if kind == "user-utterance":
            lines.append(f"### User · {ts}")
            lines.append("")
            lines.append(str(event.get("text", "")))
        elif kind == "agent-message":
            lines.append(f"### Agent · {ts}")
            lines.append("")
            lines.append(str(event.get("text", "")))
        elif kind == "tool-call":
            tool = event.get("tool", "?")
            args = cast(dict[str, Any], event.get("args") or {})
            args_compact = json.dumps(args, ensure_ascii=False, default=str)
            lines.append(f"- 🛠 `{tool}` ← `{args_compact[:200]}`")
        elif kind == "tool-result":
            tool = event.get("tool", "?")
            result = event.get("result")
            result_compact = json.dumps(result, ensure_ascii=False, default=str)
            lines.append(f"  ↳ `{result_compact[:200]}`")
        elif kind == "slash-command":
            lines.append(f"### Slash · {ts}")
            lines.append("")
            lines.append(f"`{event.get('command')}` {event.get('args') or ''}")
        elif kind == "system-note":
            note = event.get("note") or event.get("text") or ""
            lines.append(f"_system: {note}_")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "BridgeError",
    "SessionManager",
    "chat_answer",
    "chat_cancel",
    "chat_close",
    "chat_history",
    "chat_list_sessions",
    "chat_open",
    "chat_resume",
    "chat_send",
]
