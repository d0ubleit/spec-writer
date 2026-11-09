"""S1.1: Spec model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spec_writer.types import Spec


def test_spec_minimal_valid() -> None:
    spec = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="Platform Vision",
        up_to=[],
    )
    assert spec.id == "L1-VIS-001"
    assert spec.up_to == []
    assert spec.down_to == []
    assert spec.implemented_by == []
    assert spec.tags == []
    assert spec.body == ""


def test_spec_full_frontmatter() -> None:
    spec = Spec(
        id="L4-API-012",
        type="api-contract",
        layer="L4",
        title="PATCH /api/v1/profiles/{id}",
        up_to=["L3-BHV-008"],
        implemented_by=["src/api/profile.py:42-87"],
        tags=["user-management", "http-api"],
        status="draft",
        body="## Purpose\n\nEdit profile.\n",
    )
    assert spec.layer == "L4"
    assert "src/api/profile.py:42-87" in spec.implemented_by


def test_spec_missing_id_fails() -> None:
    with pytest.raises(ValidationError):
        Spec(type="vision", layer="L1", title="X")  # type: ignore[call-arg]


def test_spec_missing_title_fails() -> None:
    with pytest.raises(ValidationError):
        Spec(id="L1-VIS-001", type="vision", layer="L1")  # type: ignore[call-arg]


def test_spec_methodology_extra_fields_allowed() -> None:
    """Methodology-defined frontmatter (method/path/auth/...) lives as extras."""
    spec = Spec(
        id="L4-API-012",
        type="api-contract",
        layer="L4",
        title="POST /users",
        up_to=["L3-BHV-001"],
        method="POST",
        path="/users",
        auth="authenticated",
    )
    extras = spec.model_extra or {}
    assert extras["method"] == "POST"
    assert extras["path"] == "/users"
    assert extras["auth"] == "authenticated"


def test_spec_relates_to_entries() -> None:
    spec = Spec(
        id="L2-CAP-007",
        type="capability",
        layer="L2",
        title="User Profile Management",
        up_to=["L1-VIS-001"],
        relates_to=[{"id": "L2-CAP-002", "kind": "depends_on", "reason": "needs session"}],
    )
    assert spec.relates_to[0].id == "L2-CAP-002"
    assert spec.relates_to[0].kind == "depends_on"


def test_spec_relates_to_unknown_kind_fails() -> None:
    with pytest.raises(ValidationError):
        Spec(
            id="L2-CAP-007",
            type="capability",
            layer="L2",
            title="X",
            up_to=[],
            relates_to=[{"id": "L2-CAP-002", "kind": "bogus"}],
        )


def test_spec_provenance_nested() -> None:
    spec = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="Vision",
        up_to=[],
        provenance={"kind": "doc", "ref": "biz-req.pdf.md", "confidence": 0.9},
    )
    assert spec.provenance is not None
    assert spec.provenance.kind == "doc"
