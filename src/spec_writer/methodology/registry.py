"""ResolvedMethodology — the registry the engine consults at each step.

Engine code never works with raw YAML; it always asks the registry
("which types live on L2?", "what's the prefix for type X?",
"which validators are bound to api-contract?"). Built once at the
start of an engine run and treated as read-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from spec_writer.types import Direction, Layer, SpecType


def _empty_str_listdict() -> dict[str, list[dict[str, Any]]]:
    return {}


def _empty_hint_map() -> dict[tuple[str, str], str]:
    return {}


def _empty_enums() -> dict[str, list[str]]:
    return {}


def _empty_defaults() -> dict[str, Any]:
    return {}


@dataclass
class ResolvedMethodology:
    """Read-only view over a loaded base + mixin set."""

    layers: list[Layer]
    spec_types: dict[str, SpecType]
    types_per_layer: dict[str, list[str]]
    validators: dict[str, list[dict[str, Any]]]
    """Validator configs keyed by *spec type id*. Special key ``"*"``
    holds validators that apply across all types (e.g. core-traceability
    cross-spec rules)."""
    hooks_by_point: dict[str, list[dict[str, Any]]] = field(default_factory=_empty_str_listdict)
    metrics_by_layer: dict[str, list[dict[str, Any]]] = field(default_factory=_empty_str_listdict)
    tools_by_layer: dict[str, list[dict[str, Any]]] = field(default_factory=_empty_str_listdict)
    generator_hints_map: dict[tuple[str, str], str] = field(default_factory=_empty_hint_map)
    field_enums: dict[str, list[str]] = field(default_factory=_empty_enums)
    defaults: dict[str, Any] = field(default_factory=_empty_defaults)

    # ── API used by engine / validators / generators ─────────────────────

    def layers_in_order(self, direction: Direction) -> list[Layer]:
        return list(self.layers) if direction == "fwd" else list(reversed(self.layers))

    def types_at_layer(self, layer_id: str) -> list[SpecType]:
        ids = self.types_per_layer.get(layer_id, [])
        return [self.spec_types[i] for i in ids if i in self.spec_types]

    def prefix_for(self, spec_type_id: str) -> str:
        st = self.spec_types.get(spec_type_id)
        if st is None:
            raise KeyError(f"unknown spec type {spec_type_id!r}")
        return st.prefix

    def validators_for(self, spec_type_id: str) -> list[dict[str, Any]]:
        return list(self.validators.get(spec_type_id, [])) + list(self.validators.get("*", []))

    def metrics_for(self, layer_id: str) -> list[dict[str, Any]]:
        return list(self.metrics_by_layer.get(layer_id, []))

    def tools_for(self, layer_id: str) -> list[dict[str, Any]]:
        return list(self.tools_by_layer.get(layer_id, []))

    def hooks_for(
        self,
        point: Literal[
            "before_run",
            "after_run",
            "before_layer",
            "after_layer",
            "before_iteration",
            "after_iteration",
            "before_spec_write",
            "after_spec_write",
            "before_validate",
            "after_validate",
            "before_layer_complete",
            "on_validation_failure",
            "on_iteration_cap_reached",
        ],
    ) -> list[dict[str, Any]]:
        return list(self.hooks_by_point.get(point, []))

    def generator_hints(self, layer_id: str, spec_type_id: str) -> str:
        return self.generator_hints_map.get((layer_id, spec_type_id), "")

    def field_enum_values(self, field_name: str) -> list[str]:
        return list(self.field_enums.get(field_name, []))


__all__ = ["ResolvedMethodology"]
