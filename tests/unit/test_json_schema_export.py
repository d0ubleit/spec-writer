"""S1.1: every public type exports a JSON Schema (for frontend integration)."""

from __future__ import annotations

import pytest

from spec_writer.types import (
    FrontmatterField,
    Layer,
    Provenance,
    RelatesToEntry,
    Scenario,
    Scope,
    Section,
    Spec,
    SpecType,
    TokenUsage,
)

MODELS = [
    Spec,
    Scenario,
    Scope,
    Layer,
    SpecType,
    Section,
    FrontmatterField,
    Provenance,
    RelatesToEntry,
    TokenUsage,
]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
def test_model_exports_json_schema(model: type) -> None:
    schema = model.model_json_schema()  # type: ignore[attr-defined]
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"
    assert "properties" in schema
    assert schema.get("title") == model.__name__
