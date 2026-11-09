"""``record_spec`` tool — writes a new spec to disk.

Args (dict):
- ``type`` (required) — methodology-defined spec type id (e.g. ``vision``)
- ``layer`` (required) — layer id (e.g. ``L1``)
- ``title`` (required) — human-readable title
- ``id`` (optional) — pre-allocated spec id; if missing, IdRegistry
  allocates a fresh one
- ``up_to`` (optional) — list of parent spec ids
- ``body`` (optional) — markdown body
- any other key — passed through into Spec.model_extra so methodology-
  defined fields (``method``, ``path``, etc.) reach the file

Returns: ``{"ok": True, "spec_id": ..., "path": ...}``.
"""

from __future__ import annotations

from typing import Any, cast

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.engine import coverage_ledger
from spec_writer.engine.traceability import symmetrise_neighbors_on_disk
from spec_writer.io.spec_io import write_spec
from spec_writer.types import Spec


class RecordSpecError(RuntimeError):
    pass


async def record_spec(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    type_id = args.get("type")
    layer = args.get("layer")
    title = args.get("title")
    if not isinstance(type_id, str) or not isinstance(layer, str) or not isinstance(title, str):
        raise RecordSpecError("record_spec requires string fields: type, layer, title")

    spec_type = ctx.methodology.spec_types.get(type_id)
    if spec_type is None:
        raise RecordSpecError(
            f"unknown spec type {type_id!r}; methodology declares: "
            f"{sorted(ctx.methodology.spec_types)}"
        )

    explicit_id = args.get("id")
    if isinstance(explicit_id, str) and explicit_id:
        spec_id = explicit_id
        if ctx.id_registry.is_taken(spec_id):
            raise RecordSpecError(f"spec id {spec_id!r} is already taken")
        ctx.id_registry.register(spec_id)
    else:
        spec_id = ctx.id_registry.allocate(layer, spec_type.prefix)

    spec_payload: dict[str, Any] = {k: v for k, v in args.items() if k not in {"id", "body"}}
    spec_payload["id"] = spec_id
    spec_payload["type"] = type_id
    spec_payload["layer"] = layer
    spec_payload["title"] = title
    body = cast(str, args.get("body") or "")

    try:
        spec = Spec(**spec_payload, body=body)
    except Exception as exc:
        raise RecordSpecError(f"draft spec failed validation: {exc}") from exc

    path = write_spec(spec, ctx.methodology, ctx.project_ctx)

    # Auto-symmetrise edges to existing neighbors on disk so callers
    # don't need a follow-up no-op patch_spec to fix the parent's
    # down_to / etc. (a design constraint). Brand-new
    # spec has no priors — pass None.
    if spec.up_to or spec.down_to or spec.relates_to:
        symmetrise_neighbors_on_disk(spec, ctx)

    # REV coverage: any implemented_by entries point at real code paths;
    # update the passive ledger so coverage_report reflects the new spec.
    implemented_by = args.get("implemented_by")
    if isinstance(implemented_by, list) and implemented_by:
        typed_refs = cast(list[Any], implemented_by)
        refs = [item for item in typed_refs if isinstance(item, str)]
        if refs:
            ledger = coverage_ledger.load(ctx.project_ctx.meta)
            ledger.record_implementation(refs)
            coverage_ledger.save(ledger, ctx.project_ctx.meta)

    return {"ok": True, "spec_id": spec_id, "path": str(path)}


def register(registry: ToolRegistry) -> None:
    registry.register("record_spec", record_spec)


__all__ = ["RecordSpecError", "record_spec", "register"]
