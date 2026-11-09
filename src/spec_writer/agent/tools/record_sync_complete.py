"""``record_sync_complete`` MCP tool — finalise a sync run.

The agent calls this once it has finished reconciling the corpus with
the current code: updates ``_meta/sync_state.json`` with the new
baseline commit (or scope hash) so subsequent ``detect_code_changes``
calls compare against the post-sync state.

Args (dict, all optional):
- ``last_sync_commit`` — explicit commit hash; if omitted and the
  project is a git repo, current HEAD is captured automatically.
- ``code_scope_hash`` — explicit hash; if omitted and the project is
  NOT a git repo, a fresh hash is computed from the default code
  scope.

Returns: ``{"ok": True, "previous": SyncState_dict, "current": SyncState_dict}``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.engine import change_detector, sync_state

_DEFAULT_INCLUDES = ["src/**", "app/**", "lib/**"]
_DEFAULT_EXCLUDES = [
    "**/node_modules/**",
    "**/.git/**",
    "**/.venv/**",
    "**/vendor/**",
    "**/dist/**",
    "**/build/**",
    "**/__pycache__/**",
    "spec-editor/**",
]


async def record_sync_complete(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    project_root = ctx.project_ctx.root
    previous = sync_state.load(ctx.project_ctx.meta)

    last_commit_arg = args.get("last_sync_commit")
    last_commit: str | None = (
        last_commit_arg if isinstance(last_commit_arg, str) and last_commit_arg else None
    )
    if last_commit is None and change_detector.is_git_repo(project_root):
        result = change_detector.detect_via_git(project_root, None)
        last_commit = result.current_commit

    code_hash_arg = args.get("code_scope_hash")
    code_hash: str | None = (
        code_hash_arg if isinstance(code_hash_arg, str) and code_hash_arg else None
    )
    if code_hash is None and not change_detector.is_git_repo(project_root):
        code_hash = change_detector.compute_code_scope_hash(
            project_root, _DEFAULT_INCLUDES, _DEFAULT_EXCLUDES
        )

    new_state = sync_state.SyncState(
        last_sync_commit=last_commit,
        code_scope_hash=code_hash,
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    sync_state.save(new_state, ctx.project_ctx.meta)

    return {
        "ok": True,
        "previous": {
            "last_sync_commit": previous.last_sync_commit,
            "code_scope_hash": previous.code_scope_hash,
            "updated_at": previous.updated_at,
        },
        "current": {
            "last_sync_commit": new_state.last_sync_commit,
            "code_scope_hash": new_state.code_scope_hash,
            "updated_at": new_state.updated_at,
        },
    }


def register(registry: ToolRegistry) -> None:
    registry.register("record_sync_complete", record_sync_complete)


__all__ = ["record_sync_complete", "register"]
