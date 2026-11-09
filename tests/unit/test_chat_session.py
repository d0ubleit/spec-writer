"""S2.8 — chat session writer."""

from __future__ import annotations

import json
from pathlib import Path

from spec_writer.agent.chat_session import (
    ChatSession,
    list_sessions,
    load_session,
)
from spec_writer.engine.project_context import ProjectContext


def _project(tmp_path: Path) -> ProjectContext:
    (tmp_path / "spec-editor").mkdir()
    for sub in ("sources-raw", "sources", "specs", "_meta"):
        (tmp_path / "spec-editor" / sub).mkdir()
    return ProjectContext.from_cwd(tmp_path)


def test_chat_session_writes_file_on_first_event(tmp_path: Path) -> None:
    ctx = _project(tmp_path)
    session = ChatSession(project_ctx=ctx)
    assert not session.path.exists()
    session.log_user("Hello agent")
    assert session.path.exists()
    payload = json.loads(session.path.read_text(encoding="utf-8"))
    assert payload["session_id"] == session.session_id
    assert payload["events"][0]["kind"] == "user-utterance"
    assert payload["events"][0]["text"] == "Hello agent"


def test_chat_session_records_each_event_kind(tmp_path: Path) -> None:
    ctx = _project(tmp_path)
    session = ChatSession(project_ctx=ctx)
    session.log_user("free-form")
    session.log_slash("/refine-plan", args={"scope": "L2"})
    session.log_agent("Generating…")
    session.log_tool_call("record_spec", {"type": "vision", "title": "X"})
    session.log_tool_result("record_spec", {"ok": True, "spec_id": "L1-VIS-001"})
    session.log_note("methodology hot-reloaded", source="loader")

    payload = json.loads(session.path.read_text(encoding="utf-8"))
    kinds = [e["kind"] for e in payload["events"]]
    assert kinds == [
        "user-utterance",
        "slash-command",
        "agent-message",
        "tool-call",
        "tool-result",
        "system-note",
    ]
    slash = payload["events"][1]
    assert slash["command"] == "/refine-plan"
    assert slash["args"] == {"scope": "L2"}


def test_chat_session_path_uses_meta_sessions(tmp_path: Path) -> None:
    ctx = _project(tmp_path)
    session = ChatSession(project_ctx=ctx)
    assert session.path.parent == ctx.meta / "sessions"
    assert session.path.suffix == ".json"


def test_load_session_round_trip(tmp_path: Path) -> None:
    ctx = _project(tmp_path)
    session = ChatSession(project_ctx=ctx)
    session.log_user("first")
    session.log_agent("response")

    loaded = load_session(ctx, session.session_id)
    assert loaded.session_id == session.session_id
    assert len(loaded.events) == 2
    assert loaded.events[0]["text"] == "first"


def test_list_sessions_sorted(tmp_path: Path) -> None:
    ctx = _project(tmp_path)
    s1 = ChatSession(project_ctx=ctx, session_id="20260101T000000-aaaaaaaa")
    s2 = ChatSession(project_ctx=ctx, session_id="20260101T000005-bbbbbbbb")
    s1.log_user("a")
    s2.log_user("b")
    assert list_sessions(ctx) == [s1.session_id, s2.session_id]


def test_chat_session_atomic_overwrite(tmp_path: Path) -> None:
    ctx = _project(tmp_path)
    session = ChatSession(project_ctx=ctx)
    session.log_user("one")
    first_size = session.path.stat().st_size
    session.log_user("two")
    second_size = session.path.stat().st_size
    assert second_size > first_size
    payload = json.loads(session.path.read_text(encoding="utf-8"))
    assert [e["text"] for e in payload["events"]] == ["one", "two"]
