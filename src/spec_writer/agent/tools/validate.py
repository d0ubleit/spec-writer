"""``validate`` tool — run validators across the current corpus.

Returns a structured issue list the agent can read and act on. The
agent uses this whenever it wants to know if the corpus (or a
filtered slice of it) is clean.

Args (dict, all optional):
- ``scope`` — declarative filter; same shape as :class:`Scope`
  (layers / types / tags / spec_ids / folder_pattern). Narrows which
  specs the validators run against. ``None`` means the full corpus.

Returns: ``{"ok": True, "count": N, "issues": [issue_dict, ...]}``.
Each ``issue_dict`` mirrors :class:`ValidationIssue` (severity, code,
message, spec_id?, field?, fix_hint?).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.engine.scope import filter_corpus
from spec_writer.io.spec_io import read_spec
from spec_writer.methodology.validators.runner import run_validators
from spec_writer.types import Scope, Spec


class ValidateToolError(RuntimeError):
    pass


def _load_corpus(specs_root: Path) -> dict[str, Spec]:
    corpus: dict[str, Spec] = {}
    if not specs_root.exists():
        return corpus
    for path in sorted(specs_root.rglob("*.md")):
        try:
            spec = read_spec(path)
        except Exception:
            continue
        corpus[spec.id] = spec
    return corpus


def _parse_scope(raw: object) -> Scope | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValidateToolError("'scope' must be an object")
    try:
        return Scope(**cast(dict[str, Any], raw))
    except Exception as exc:
        raise ValidateToolError(f"invalid scope: {exc}") from exc


async def validate(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    scope = _parse_scope(args.get("scope"))

    corpus = _load_corpus(ctx.project_ctx.specs)
    in_scope = filter_corpus(corpus, scope, ctx.methodology)
    # Pass the *full* corpus as `all_specs` so cross-layer / cross-type
    # references in scoped specs still resolve. `specs` controls which
    # specs are validated; `all_specs` controls what validators see when
    # checking references (see docs/architecture.md).
    issues = run_validators(
        list(in_scope.values()),
        ctx.methodology,
        all_specs=list(corpus.values()),
    )

    return {
        "ok": True,
        "count": len(issues),
        "issues": [
            {
                "severity": i.severity,
                "code": i.code,
                "message": i.message,
                "spec_id": i.spec_id,
                "field": i.field,
                "fix_hint": i.fix_hint,
            }
            for i in issues
        ],
    }


def register(registry: ToolRegistry) -> None:
    registry.register("validate", validate)


__all__ = ["ValidateToolError", "register", "validate"]
