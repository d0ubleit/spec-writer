"""Sync state — durable record of "what state of the code did we last reconcile?".

Lives at ``_meta/sync_state.json``. Contains:

- ``last_sync_commit`` — git commit hash at the time of the last
  successful sync, or ``None`` if the project isn't a git repo.
- ``code_scope_hash`` — hash of the code surface tracked by the
  methodology's ``code_scope.includes / excludes`` (Phase 4 minimum:
  a sha256 of the sorted file paths + sizes). Lets us detect drift
  even on non-git projects.
- ``updated_at`` — ISO-8601 UTC timestamp.

The agent reads this via :func:`load` to know the previous baseline,
and writes via :func:`save` (or the ``record_sync_complete`` MCP
tool) when a sync run finishes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from spec_writer.engine.atomic_io import atomic_write

_FILENAME = "sync_state.json"


@dataclass(frozen=True)
class SyncState:
    last_sync_commit: str | None = None
    code_scope_hash: str | None = None
    updated_at: str | None = None


def state_path(meta_dir: Path) -> Path:
    return meta_dir / _FILENAME


def load(meta_dir: Path) -> SyncState:
    path = state_path(meta_dir)
    if not path.exists():
        return SyncState()
    try:
        raw_any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return SyncState()
    if not isinstance(raw_any, dict):
        return SyncState()
    raw = cast(dict[str, Any], raw_any)
    return SyncState(
        last_sync_commit=_str_or_none(raw.get("last_sync_commit")),
        code_scope_hash=_str_or_none(raw.get("code_scope_hash")),
        updated_at=_str_or_none(raw.get("updated_at")),
    )


def save(state: SyncState, meta_dir: Path) -> None:
    payload = {
        "last_sync_commit": state.last_sync_commit,
        "code_scope_hash": state.code_scope_hash,
        "updated_at": state.updated_at or _now_iso(),
    }
    atomic_write(
        state_path(meta_dir),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


__all__ = ["SyncState", "load", "save", "state_path"]
