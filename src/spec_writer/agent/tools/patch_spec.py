"""``patch_spec`` tool — apply a partial update to an existing spec.

Args:
- ``spec_id`` (required) — id of the spec to update
- ``changes`` (required) — mapping of field → new value. Passing
  ``None`` for an extras-field deletes the key from frontmatter;
  core nullable fields like ``status`` / ``provenance`` accept
  ``None`` as a normal value.

Immutable fields (``id``, ``type``, ``layer``) are silently ignored —
changing them requires a full rewrite via ``record_spec``.

Edge fields (``up_to``, ``down_to``, ``relates_to``) trigger a
neighbor-scoped symmetrisation pass: every reciprocal edge is
repaired even when the patched value matches the prior value, and
removed edges leave no stale reciprocal on the previous neighbor.

Pydantic-typed list fields (``relates_to``) are revalidated through
the model so callers can pass plain dicts from the MCP wire.

Unknown keys in ``changes`` are rejected with ``PatchSpecError`` so
callers cannot silently mis-spell an op or invent one. The
``body_replace`` op provides surgical substring edits per SDK
Edit-tool semantics.
"""

from __future__ import annotations

from typing import Any, cast

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.engine.traceability import symmetrise_neighbors_on_disk
from spec_writer.io.spec_io import find_spec_by_id, read_spec, write_spec
from spec_writer.types import Provenance, RelatesToEntry, Spec

_IMMUTABLE = frozenset({"id", "type", "layer"})
_EDGE_FIELDS = frozenset({"up_to", "down_to", "relates_to"})
_RESERVED_OPS = frozenset({"body", "body_replace"})


class PatchSpecError(RuntimeError):
    pass


def _is_core_field(key: str) -> bool:
    """Whether ``key`` is a declared Spec field (vs. methodology extra)."""
    return key in Spec.model_fields


def _coerce_value(key: str, value: Any) -> Any:
    """Revalidate Pydantic-typed Spec fields when patched .

    The MCP wire delivers ``relates_to`` / ``provenance`` as plain
    dict/list payloads; ``setattr`` on a Pydantic-typed attribute does
    NOT revalidate (Pydantic v2 validates on construct, not on
    mutation), so the typed list silently stores raw dicts and breaks
    on the next access. Manually run the model validator here.
    """
    if key == "relates_to":
        if not isinstance(value, list):
            raise PatchSpecError(
                f"patch_spec: 'relates_to' must be a list of entries, got {type(value).__name__}"
            )
        raw_list = cast(list[Any], value)
        return [
            item if isinstance(item, RelatesToEntry) else RelatesToEntry.model_validate(item)
            for item in raw_list
        ]
    if key == "provenance":
        if value is None or isinstance(value, Provenance):
            return value
        return Provenance.model_validate(value)
    return value


def _allowed_extras_for(spec: Spec, ctx: ToolContext) -> set[str]:
    """Methodology-declared frontmatter fields for the spec's type.

    Both required + optional frontmatter declared on the spec_type are
    valid patch_spec extras keys. Anything else is rejected .
    """
    spec_type = ctx.methodology.spec_types.get(spec.type)
    if spec_type is None:
        return set()
    return {f.name for f in spec_type.required_frontmatter} | {
        f.name for f in spec_type.optional_frontmatter
    }


def _validate_changes_keys(changes_dict: dict[str, Any], spec: Spec, ctx: ToolContext) -> None:
    """Reject unknown keys in changes .

    Allowed:
    - core Spec fields,
    - reserved ops (``body``, ``body_replace``),
    - methodology-declared extras for this spec's type,
    - keys that already exist in ``spec.model_extra`` (so stale
      frontmatter pollution and dropped methodology fields can be
      cleaned up via ``patch_spec({key: None})``).

    Anything else is a typo or an invented operation — raise loudly so
    the caller can recover.
    """
    allowed_extras = _allowed_extras_for(spec, ctx)
    current_extras = set(spec.model_extra or {})
    for key in changes_dict:
        if key in _IMMUTABLE:
            continue  # silently skipped later, but known
        if key in _RESERVED_OPS:
            continue
        if _is_core_field(key):
            continue
        if key in allowed_extras:
            continue
        if key in current_extras:
            continue  # stale frontmatter — patching it (or clearing) is fine
        # Unknown key — surface a helpful diagnostic.
        known = sorted(_RESERVED_OPS | set(Spec.model_fields) | allowed_extras)
        raise PatchSpecError(
            f"patch_spec: unknown key {key!r} in changes for spec type "
            f"{spec.type!r}. Known operations + fields: {known}. "
            f"If you meant a surgical body edit, use 'body_replace': "
            f"[{{'old': ..., 'new': ...}}, ...]."
        )


def _apply_body_replace(spec: Spec, ops: Any) -> None:
    """Apply ordered substring substitutions to ``spec.body`` .

    Each op: ``{old: str, new: str}``. ``old`` must appear exactly
    once in the current body — 0 matches → ``not found``, >1 matches →
    ``ambiguous``. Both raise so silent mismatches can't recur.
    """
    if not isinstance(ops, list):
        raise PatchSpecError(
            f"patch_spec: 'body_replace' must be a list of "
            f"{{old, new}} ops, got {type(ops).__name__}"
        )
    for idx, raw_op in enumerate(cast(list[Any], ops)):
        if not isinstance(raw_op, dict):
            raise PatchSpecError(
                f"patch_spec: body_replace[{idx}] must be an object with "
                f"'old' and 'new' keys, got {type(raw_op).__name__}"
            )
        op = cast(dict[str, Any], raw_op)
        old = op.get("old")
        new = op.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            raise PatchSpecError(f"patch_spec: body_replace[{idx}] requires string 'old' and 'new'")
        count = spec.body.count(old)
        if count == 0:
            preview = old if len(old) <= 80 else old[:77] + "..."
            raise PatchSpecError(
                f"patch_spec: body_replace[{idx}].old not found in body: {preview!r}"
            )
        if count > 1:
            preview = old if len(old) <= 80 else old[:77] + "..."
            raise PatchSpecError(
                f"patch_spec: body_replace[{idx}].old matches {count} times in body — "
                f"provide more context to make it unique: {preview!r}"
            )
        spec.body = spec.body.replace(old, new, 1)


async def patch_spec(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    spec_id = args.get("spec_id")
    changes = args.get("changes")
    if not isinstance(spec_id, str):
        raise PatchSpecError("patch_spec requires string 'spec_id'")
    if not isinstance(changes, dict):
        raise PatchSpecError("patch_spec requires dict 'changes'")
    changes_dict = cast(dict[str, Any], changes)

    path = find_spec_by_id(ctx.project_ctx.specs, spec_id)
    if path is None:
        raise PatchSpecError(f"spec {spec_id!r} not found in {ctx.project_ctx.specs}")

    spec = read_spec(path)

    # a known limitation: reject unknown keys before any mutation.
    _validate_changes_keys(changes_dict, spec, ctx)

    # Capture priors so edge cleanup can detect removed edges .
    prior_up_to = list(spec.up_to)
    prior_down_to = list(spec.down_to)
    prior_relates_to = list(spec.relates_to)

    applied: list[str] = []
    touched_edges = False
    for key, value in changes_dict.items():
        if key in _IMMUTABLE:
            continue
        # a known limitation extras-cleanup path takes priority over reserved-op dispatch:
        # patch_spec({"body_replace": None}) on a spec polluted with a
        # stale `body_replace` frontmatter key () removes that
        # key rather than trying to apply a null body-replace op.
        if value is None and not _is_core_field(key):
            if spec.model_extra is not None and key in spec.model_extra:
                del spec.model_extra[key]
            applied.append(key)
            continue
        if key == "body":
            spec.body = str(value)
        elif key == "body_replace":
            _apply_body_replace(spec, value)
        else:
            setattr(spec, key, _coerce_value(key, value))
        applied.append(key)
        if key in _EDGE_FIELDS:
            touched_edges = True

    new_path = write_spec(spec, ctx.methodology, ctx.project_ctx)
    if new_path != path and path.exists():
        # title or other rename caused the file to move
        path.unlink(missing_ok=True)
    if touched_edges:
        symmetrise_neighbors_on_disk(
            spec,
            ctx,
            prior_up_to=prior_up_to,
            prior_down_to=prior_down_to,
            prior_relates_to=prior_relates_to,
        )
    return {"ok": True, "spec_id": spec_id, "path": str(new_path), "applied": sorted(applied)}


def register(registry: ToolRegistry) -> None:
    registry.register("patch_spec", patch_spec)


__all__ = ["PatchSpecError", "patch_spec", "register"]
