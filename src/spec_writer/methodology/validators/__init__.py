"""Validator framework — Protocol, ValidationIssue, registry.

Validators are pure functions registered by ``func_ref`` strings that
mixin YAMLs reference (e.g. ``validators.check_pyramid_adjacency``).
Phase 1 ships all built-ins in :mod:`.builtin`; third-party mixin
sidecars  plug in via the same registry in Phase 3.

Each validator carries a ``scope``:
- ``per-spec`` — invoked once per spec with ``(spec, ctx)``.
- ``cross-spec`` — invoked once with all specs in scope as
  ``(specs, ctx)``.

Validators return zero or more :class:`ValidationIssue` records.
Severity ``error`` blocks gates; ``warning`` / ``info`` surface but
don't block.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.types import Spec


class ValidationIssue(BaseModel):
    """Single issue surfaced by a validator."""

    model_config = ConfigDict(frozen=True)

    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    spec_id: str | None = None
    field: str | None = None
    fix_hint: str | None = None


@dataclass(frozen=True)
class ValidatorContext:
    """Shared state passed to every validator on a given run."""

    corpus: dict[str, Spec]
    methodology: ResolvedMethodology


PerSpecValidator = Callable[[Spec, ValidatorContext], list[ValidationIssue]]
CrossSpecValidator = Callable[[list[Spec], ValidatorContext], list[ValidationIssue]]
ValidatorScope = Literal["per-spec", "cross-spec"]


# ── registry ─────────────────────────────────────────────────────────────


_REGISTRY: dict[str, tuple[Callable[..., list[ValidationIssue]], ValidatorScope]] = {}


def register_validator(
    name: str, scope: ValidatorScope
) -> Callable[[Callable[..., list[ValidationIssue]]], Callable[..., list[ValidationIssue]]]:
    """Decorator that records a validator implementation by ``func_ref`` name."""

    def deco(
        func: Callable[..., list[ValidationIssue]],
    ) -> Callable[..., list[ValidationIssue]]:
        _REGISTRY[name] = (func, scope)
        return func

    return deco


def get_validator(name: str) -> tuple[Callable[..., list[ValidationIssue]], ValidatorScope] | None:
    return _REGISTRY.get(name)


def registered_names() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "CrossSpecValidator",
    "PerSpecValidator",
    "ValidationIssue",
    "ValidatorContext",
    "ValidatorScope",
    "get_validator",
    "register_validator",
    "registered_names",
]
