"""Validator runner — collect-all execution against a spec corpus.

Per docs/architecture.md: validators **always**
run to completion; no fail-fast. The runner gathers all issues and
returns them grouped so the agent can patch every error in one round.

Per-spec validators run once per (spec, validator) pair.
Cross-spec validators run once per unique validator id, with the
entire spec list as input.
"""

from __future__ import annotations

from typing import Any

from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.methodology.validators import (
    ValidationIssue,
    ValidatorContext,
    get_validator,
)
from spec_writer.methodology.validators import builtin as _builtin
from spec_writer.methodology.validators import mixin_validators as _mixin_validators
from spec_writer.types import Spec

# Importing the modules above triggers @register_validator decorators
# so all built-in rules are available without the caller importing
# them explicitly.
del _builtin, _mixin_validators


def run_validators(
    specs: list[Spec],
    methodology: ResolvedMethodology,
    all_specs: list[Spec] | None = None,
) -> list[ValidationIssue]:
    """Run every validator registered for any spec type in the corpus.

    ``specs`` controls **which** specs are validated (per-spec validators
    iterate this list; cross-spec validators receive it as input).

    ``all_specs`` (optional) controls **what the validators know about**
    when resolving references. If provided, ``ctx.corpus`` is built from
    it — a full corpus snapshot — so cross-layer / cross-type references
    resolve cleanly even when ``specs`` is a scoped subset. If omitted,
    ``ctx.corpus = specs`` (backward-compatible).

    The split matters for scoped validation: a caller invoking
    ``validate(scope={layer: 'L3'})`` wants to validate L3 specs against
    the **full** corpus (so a L3-→-L2 ``up_to`` edge resolves), not
    against the L3-only subset (which would falsely flag every cross-
    layer edge as ``dangling_ref``).

    The methodology stores validator configs per-type, plus a wildcard
    bucket (``"*"``) for cross-cutting rules. The runner deduplicates
    cross-spec validators (one execution per unique ``id``) and
    collects every issue.
    """
    issues: list[ValidationIssue] = []
    corpus_specs = all_specs if all_specs is not None else specs
    corpus = {spec.id: spec for spec in corpus_specs}
    ctx = ValidatorContext(corpus=corpus, methodology=methodology)

    cross_seen: set[str] = set()

    for spec in specs:
        for config in methodology.validators_for(spec.type):
            issues.extend(_dispatch(config, spec, specs, ctx, cross_seen))

    # If the corpus is empty but cross-spec validators are registered,
    # still execute them once (they typically just no-op on []).
    if not specs:
        for config in methodology.validators_for("*"):
            issues.extend(_dispatch(config, None, specs, ctx, cross_seen))

    return issues


def _dispatch(
    config: dict[str, Any],
    spec: Spec | None,
    all_specs: list[Spec],
    ctx: ValidatorContext,
    cross_seen: set[str],
) -> list[ValidationIssue]:
    func_ref = config.get("func_ref")
    scope = config.get("scope")
    validator_id = config.get("id", func_ref)
    if not isinstance(func_ref, str) or not isinstance(scope, str):
        return []

    impl = get_validator(func_ref)
    if impl is None:
        return [
            ValidationIssue(
                severity="warning",
                code="validator_not_registered",
                message=f"validator {func_ref!r} declared but not implemented",
            )
        ]
    func, declared_scope = impl
    effective_scope = scope or declared_scope

    if effective_scope == "per-spec":
        if spec is None:
            return []
        return list(func(spec, ctx))
    if effective_scope == "cross-spec":
        if validator_id in cross_seen:
            return []
        cross_seen.add(str(validator_id))
        return list(func(all_specs, ctx))
    return []


__all__ = ["run_validators"]
