"""``list_sources`` tool — enumerate files in ``spec-editor/sources/``.

The agent uses this to discover what input material is available
before deciding what to read with the SDK builtin ``Read``. Returns
file paths with sizes; **does not** read contents (that's the agent's
own decision).

Args: ``{}``.

Returns: ``{"ok": True, "files": [{"path", "size_bytes", "suffix"}, …]}``.
``path`` is relative to ``spec-editor/`` so the agent can pass it to
``Read`` directly.
"""

from __future__ import annotations

from typing import Any

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry


async def list_sources(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    del args
    root = ctx.project_ctx.sources
    if not root.exists():
        return {"ok": True, "files": []}

    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ctx.project_ctx.spec_editor)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        files.append(
            {
                "path": str(rel),
                "absolute_path": str(path),
                "size_bytes": size,
                "suffix": path.suffix.lstrip("."),
            }
        )
    return {"ok": True, "files": files}


def register(registry: ToolRegistry) -> None:
    registry.register("list_sources", list_sources)


__all__ = ["list_sources", "register"]
