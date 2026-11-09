"""``read_spec`` tool — return one spec's full frontmatter + body by id.

The chat-mode ChatAgent needs to inspect the current state of an
existing spec before patching it — otherwise patches risk overwriting
unrelated content. The SDK-builtin ``Read`` tool can do this when
the agent's sandbox grants ``spec-editor/specs/**`` read permission;
``read_spec`` is the server-side fallback that works regardless of the
sandbox configuration (the bridge process has unconditional disk
access).

Args (dict):
- ``spec_id`` (required) — canonical id (e.g. ``L2-CAP-007``).

Returns: ``{ok, spec_id, type, layer, title, frontmatter, body, path}``
on success; ``{ok: False, error}`` if id not found.

``frontmatter`` is a dict — engine-mandatory fields plus any
methodology-defined extras populated on the spec. ``body`` is the raw
markdown content (everything after the closing ``---``).
"""

from __future__ import annotations

import datetime
from typing import Any, cast
from uuid import UUID

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.io.spec_io import find_spec_by_id, read_spec
from spec_writer.types import Spec


class ReadSpecError(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    """Convert non-JSON-native frontmatter values into wire-serializable forms.

    YAML lets users embed ``date``/``datetime``/``UUID`` values directly
    (parsed via ``yaml.safe_load``). MCP serializes the tool result to
    JSON; those types break the encoder. Recurse so dict/list payloads
    are converted leaf-by-leaf. See the design rationale.
    """
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        items: dict[Any, Any] = cast(dict[Any, Any], value)
        return {str(k): _json_safe(v) for k, v in items.items()}
    if isinstance(value, (list, tuple)):
        seq: list[Any] | tuple[Any, ...] = cast("list[Any] | tuple[Any, ...]", value)
        return [_json_safe(v) for v in seq]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Unknown type — coerce to string so the wire never fails.
    return str(value)


def _frontmatter_dict(spec: Spec) -> dict[str, Any]:
    """Snapshot the engine-mandatory + methodology-extra frontmatter."""
    fm: dict[str, Any] = {
        "id": spec.id,
        "type": spec.type,
        "layer": spec.layer,
        "title": spec.title,
        "up_to": list(spec.up_to),
        "down_to": list(spec.down_to),
        "implemented_by": list(spec.implemented_by),
        "tags": list(spec.tags),
        "status": spec.status,
        "relates_to": [
            {"id": rel.id, "kind": rel.kind, "reason": rel.reason} for rel in spec.relates_to
        ],
    }
    if spec.provenance is not None:
        fm["provenance"] = {
            "kind": spec.provenance.kind,
            "ref": spec.provenance.ref,
            "confidence": spec.provenance.confidence,
        }
    extras = spec.model_extra or {}
    for key, value in extras.items():
        if key not in fm:
            fm[key] = value
    return {k: _json_safe(v) for k, v in fm.items()}


async def read_spec_tool(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    spec_id = args.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id:
        raise ReadSpecError("read_spec requires non-empty string 'spec_id'")

    specs_root = ctx.project_ctx.specs
    if not specs_root.exists():
        return {"ok": False, "error": f"specs directory does not exist: {specs_root}"}

    path = find_spec_by_id(specs_root, spec_id)
    if path is None:
        return {"ok": False, "error": f"spec_id {spec_id!r} not found under {specs_root}"}

    try:
        spec = read_spec(path)
    except Exception as exc:
        return {"ok": False, "error": f"failed to parse {path}: {exc}"}

    return {
        "ok": True,
        "spec_id": spec.id,
        "type": spec.type,
        "layer": spec.layer,
        "title": spec.title,
        "frontmatter": _frontmatter_dict(spec),
        "body": spec.body,
        "path": str(path),
    }


def register(registry: ToolRegistry) -> None:
    registry.register("read_spec", read_spec_tool)


__all__ = ["ReadSpecError", "read_spec_tool", "register"]
