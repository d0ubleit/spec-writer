"""``build_reverse_index`` MCP tool — code-file → [spec_id, …] map.

Used by sync flows: when the agent learns which code files changed,
it asks for the reverse index to know which specs need refining.
Built on-the-fly from the on-disk corpus; no persistence.
"""

from __future__ import annotations

from typing import Any

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.engine import reverse_index


async def build_reverse_index(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    del args  # Phase 4 minimum: no filters; the index is small
    index = reverse_index.build(ctx.project_ctx.specs)
    return {"ok": True, "index": index, "count": len(index)}


def register(registry: ToolRegistry) -> None:
    registry.register("build_reverse_index", build_reverse_index)


__all__ = ["build_reverse_index", "register"]
