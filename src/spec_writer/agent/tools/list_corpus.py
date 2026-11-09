"""``list_corpus`` tool — enumerate existing specs with frontmatter only.

The agent uses this to understand the current corpus before deciding
what to write / patch. **Does not return spec bodies** — agent reads
specific ones with the SDK builtin ``Read`` if needed.

Args (dict, all optional):
- ``layer`` — filter to one layer id (e.g. ``"L2"``).
- ``type`` — filter to one spec type id (e.g. ``"capability"``).
- ``tag`` — filter to specs carrying this tag.

Returns: ``{"ok": True, "count": N, "specs": [{...}, ...]}`` where each
entry has ``id``, ``type``, ``layer``, ``title``, ``up_to``,
``down_to``, ``tags``, ``status``, ``path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.io.spec_io import read_spec


async def list_corpus(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    layer = args.get("layer") if isinstance(args.get("layer"), str) else None
    spec_type = args.get("type") if isinstance(args.get("type"), str) else None
    tag = args.get("tag") if isinstance(args.get("tag"), str) else None

    specs_root = ctx.project_ctx.specs
    entries: list[dict[str, Any]] = []
    if not specs_root.exists():
        return {"ok": True, "count": 0, "specs": []}

    for path in sorted(specs_root.rglob("*.md")):
        try:
            spec = read_spec(path)
        except Exception:
            continue
        if layer is not None and spec.layer != layer:
            continue
        if spec_type is not None and spec.type != spec_type:
            continue
        if tag is not None and tag not in spec.tags:
            continue
        entries.append(
            {
                "id": spec.id,
                "type": spec.type,
                "layer": spec.layer,
                "title": spec.title,
                "up_to": list(spec.up_to),
                "down_to": list(spec.down_to),
                "tags": list(spec.tags),
                "status": spec.status,
                "path": str(_relative_to_project(path, ctx.project_ctx.spec_editor)),
            }
        )
    return {"ok": True, "count": len(entries), "specs": entries}


def _relative_to_project(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def register(registry: ToolRegistry) -> None:
    registry.register("list_corpus", list_corpus)


__all__ = ["list_corpus", "register"]
