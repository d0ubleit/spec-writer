"""``resolve_scope`` tool — translate free-form text into a ``Scope`` filter.

Phase 2 ships a minimum-viable resolver: it lexes the input string for
recognisable tokens (layer ids, spec ids, methodology-declared spec
type ids, tag-like ``#word`` markers, ``from:`` finding ids) and
returns the merged ``Scope`` shape. Anything we can't parse falls into
a ``residual`` field so the agent can decide whether to ask the user
for clarification.

Examples
--------
``"capabilities on L2 from:FIN-12"``
  → ``Scope(layers=["L2"], types=["capability"], from_finding=["FIN-12"])``

``"#auth API contracts"``
  → ``Scope(types=["api-contract"], tags=["auth"])``

``"L1-FEAT-001 L1-FEAT-002"``
  → ``Scope(spec_ids=["L1-FEAT-001", "L1-FEAT-002"])``
"""

from __future__ import annotations

import re
from typing import Any

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.engine.id_registry import parse_id
from spec_writer.types import Scope

_LAYER_RE = re.compile(r"^L\d+$", re.IGNORECASE)
_TAG_RE = re.compile(r"^#([A-Za-z0-9_-]+)$")
_FROM_RE = re.compile(r"^from:(.+)$", re.IGNORECASE)
_TYPE_ALIAS: dict[str, str] = {
    "capabilities": "capability",
    "behaviors": "behavior",
    "behaviours": "behavior",
    "ac": "acceptance-criterion",
    "acs": "acceptance-criterion",
    "apis": "api-contract",
    "api": "api-contract",
    "api-contracts": "api-contract",
    "schemas": "entity-schema",
    "entities": "entity-schema",
    "visions": "vision",
}


class ResolveScopeError(RuntimeError):
    pass


def _normalize_token(token: str) -> str:
    return token.strip(" ,.;:").lower()


async def resolve_scope(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    text = args.get("text")
    if not isinstance(text, str):
        raise ResolveScopeError("resolve_scope requires string 'text'")

    layers: list[str] = []
    types: list[str] = []
    tags: list[str] = []
    spec_ids: list[str] = []
    from_finding: list[str] = []
    residual: list[str] = []

    known_types = set(ctx.methodology.spec_types)

    for raw in text.split():
        token = _normalize_token(raw)
        if not token:
            continue

        if parse_id(raw):
            spec_ids.append(raw)
            continue

        match = _FROM_RE.match(raw)
        if match:
            from_finding.append(match.group(1))
            continue

        match = _TAG_RE.match(raw)
        if match:
            tags.append(match.group(1))
            continue

        if _LAYER_RE.match(token):
            layers.append(token.upper())
            continue

        canonical = _TYPE_ALIAS.get(token, token)
        if canonical in known_types:
            types.append(canonical)
            continue

        residual.append(raw)

    scope = Scope(
        layers=sorted(set(layers)) or None,
        types=sorted(set(types)) or None,
        tags=sorted(set(tags)) or None,
        spec_ids=sorted(set(spec_ids)) or None,
        from_finding=sorted(set(from_finding)) or None,
    )

    return {
        "ok": True,
        "scope": scope.model_dump(exclude_none=True),
        "residual": residual,
    }


def register(registry: ToolRegistry) -> None:
    registry.register("resolve_scope", resolve_scope)


__all__ = ["ResolveScopeError", "register", "resolve_scope"]
