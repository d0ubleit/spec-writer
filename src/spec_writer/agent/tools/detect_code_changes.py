"""``detect_code_changes`` MCP tool — agent-driven sync detection.

Reads ``_meta/sync_state.json`` for the baseline; runs the git-first
detector (or hash fallback for non-git projects); returns a
structured change list the agent can classify and act on. **Does
not modify** state — :func:`record_sync_complete` is a separate tool
the agent calls when it's satisfied with the sync.
"""

from __future__ import annotations

from typing import Any, cast

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


class DetectCodeChangesError(RuntimeError):
    pass


async def detect_code_changes(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_inc = args.get("includes")
    raw_exc = args.get("excludes")
    if raw_inc is None:
        includes = list(_DEFAULT_INCLUDES)
    elif isinstance(raw_inc, list):
        includes = [p for p in cast(list[Any], raw_inc) if isinstance(p, str)]
    else:
        raise DetectCodeChangesError("'includes' must be a list")
    if raw_exc is None:
        excludes = list(_DEFAULT_EXCLUDES)
    elif isinstance(raw_exc, list):
        excludes = [p for p in cast(list[Any], raw_exc) if isinstance(p, str)]
    else:
        raise DetectCodeChangesError("'excludes' must be a list")

    project_root = ctx.project_ctx.root
    state = sync_state.load(ctx.project_ctx.meta)

    if change_detector.is_git_repo(project_root):
        result = change_detector.detect_via_git(project_root, state.last_sync_commit)
    else:
        result = change_detector.detect_via_hash(
            project_root, includes, excludes, state.code_scope_hash
        )

    return {
        "ok": True,
        "method": result.method,
        "baseline": {
            "last_sync_commit": state.last_sync_commit,
            "code_scope_hash": state.code_scope_hash,
            "updated_at": state.updated_at,
        },
        "current": {
            "commit": result.current_commit,
            "hash": result.current_hash,
        },
        "baseline_lost": result.baseline_lost,
        "hash_changed": result.hash_changed,
        "added": result.added,
        "modified": result.modified,
        "deleted": result.deleted,
        "renamed": [{"from": old, "to": new} for old, new in result.renamed],
        "error": result.error,
    }


def register(registry: ToolRegistry) -> None:
    registry.register("detect_code_changes", detect_code_changes)


__all__ = ["DetectCodeChangesError", "detect_code_changes", "register"]
