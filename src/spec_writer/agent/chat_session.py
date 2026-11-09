"""Chat session writer — persists one REPL run to ``_meta/sessions/<id>.json``.

Each ``spec-writer chat`` invocation gets its own session id (timestamp
prefix + short uuid). The file accumulates one append-only record per
event:

- user-utterance (free-form text the user typed)
- slash-command (slash-prefixed command + its parsed args)
- agent-message (assistant text delta, coalesced per turn)
- tool-call / tool-result (so we can replay what happened)
- system note (engine-level: methodology reload, errors, etc.)

Writes are best-effort and never propagate I/O errors — the chat
session continues even if the disk gets weird. We do warn-log so the
problem is visible. Snapshot on every append (no buffering) so a
crash mid-run still leaves a recoverable record.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from spec_writer.engine.atomic_io import atomic_write
from spec_writer.engine.project_context import ProjectContext

_LOG = logging.getLogger(__name__)


EventKind = Literal[
    "user-utterance",
    "slash-command",
    "agent-message",
    "tool-call",
    "tool-result",
    "system-note",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty_events() -> list[dict[str, Any]]:
    return []


def _generate_session_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    short = uuid.uuid4().hex[:8]
    return f"{stamp}-{short}"


@dataclass
class ChatSession:
    """Append-only log of one chat run, serialized to a single JSON file."""

    project_ctx: ProjectContext
    session_id: str = field(default_factory=_generate_session_id)
    started_at: str = field(default_factory=_now_iso)
    events: list[dict[str, Any]] = field(default_factory=_empty_events)

    @property
    def path(self) -> Path:
        return self.project_ctx.meta / "sessions" / f"{self.session_id}.json"

    # ── high-level event helpers ─────────────────────────────────────

    def log_user(self, text: str) -> None:
        self._append("user-utterance", {"text": text})

    def log_slash(self, command: str, args: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"command": command}
        if args:
            payload["args"] = args
        self._append("slash-command", payload)

    def log_agent(self, text: str) -> None:
        self._append("agent-message", {"text": text})

    def log_tool_call(self, tool: str, args: dict[str, Any]) -> None:
        self._append("tool-call", {"tool": tool, "args": args})

    def log_tool_result(self, tool: str, result: dict[str, Any]) -> None:
        self._append("tool-result", {"tool": tool, "result": result})

    def log_note(self, message: str, **extra: Any) -> None:
        payload: dict[str, Any] = {"message": message}
        if extra:
            payload["details"] = extra
        self._append("system-note", payload)

    # ── persistence ─────────────────────────────────────────────────

    def _append(self, kind: EventKind, payload: dict[str, Any]) -> None:
        entry: dict[str, Any] = {"at": _now_iso(), "kind": kind, **payload}
        self.events.append(entry)
        try:
            self._snapshot()
        except OSError as exc:
            _LOG.warning("chat session %s: could not persist event: %s", self.session_id, exc)

    def _snapshot(self) -> None:
        payload = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "events": self.events,
        }
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(self.path, rendered)


def load_session(project_ctx: ProjectContext, session_id: str) -> ChatSession:
    """Hydrate a previously-written session from disk."""
    path = project_ctx.meta / "sessions" / f"{session_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ChatSession(
        project_ctx=project_ctx,
        session_id=raw["session_id"],
        started_at=raw["started_at"],
        events=list(raw.get("events", [])),
    )


def list_sessions(project_ctx: ProjectContext) -> list[str]:
    sessions_dir = project_ctx.meta / "sessions"
    if not sessions_dir.exists():
        return []
    return sorted(p.stem for p in sessions_dir.glob("*.json"))


__all__ = [
    "ChatSession",
    "EventKind",
    "list_sessions",
    "load_session",
]
