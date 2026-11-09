"""``delete_spec`` MCP tool — remove a spec from the corpus.

Used in sync flows when the agent decides a spec no longer has a
real-world counterpart (e.g. its code was deleted). Performs:

1. Find the spec by id (via :func:`find_spec_by_id`).
2. Delete the file from disk.
3. Release the id back to :class:`IdRegistry` so it can be reused.
4. Do **not** patch siblings that reference this spec — that's the
   agent's next step (it should already know via
   ``query_traceability``).

Returns ``{"ok": True, "spec_id": ..., "path": ..., "released_id": True}``.

This is destructive. The agent should usually call ``ask_user`` for
confirmation before invoking it (system prompt enforces this for
non-orphan specs).
"""

from __future__ import annotations

from typing import Any

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.io.spec_io import find_spec_by_id


class DeleteSpecError(RuntimeError):
    pass


async def delete_spec(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    spec_id = args.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id:
        raise DeleteSpecError("delete_spec requires string 'spec_id'")

    path = find_spec_by_id(ctx.project_ctx.specs, spec_id)
    if path is None:
        return {"ok": False, "error": f"spec {spec_id!r} not found"}

    try:
        path.unlink()
    except OSError as exc:
        return {"ok": False, "error": f"could not delete {path}: {exc}"}

    released = ctx.id_registry.release(spec_id)
    return {
        "ok": True,
        "spec_id": spec_id,
        "path": str(path),
        "released_id": released,
    }


def register(registry: ToolRegistry) -> None:
    registry.register("delete_spec", delete_spec)


__all__ = ["DeleteSpecError", "delete_spec", "register"]
