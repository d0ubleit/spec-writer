"""S1.1: Provenance kind literal validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spec_writer.types import Provenance


@pytest.mark.parametrize("kind", ["doc", "code", "inferred"])
def test_provenance_all_kinds(kind: str) -> None:
    p = Provenance(kind=kind)  # type: ignore[arg-type]
    assert p.kind == kind


def test_provenance_unknown_kind_fails() -> None:
    with pytest.raises(ValidationError):
        Provenance(kind="invented")  # type: ignore[arg-type]


def test_provenance_optional_fields() -> None:
    p = Provenance(kind="doc")
    assert p.ref is None
    assert p.confidence is None


def test_provenance_full() -> None:
    p = Provenance(kind="doc", ref="biz-req.pdf.md", confidence=0.95)
    assert p.ref == "biz-req.pdf.md"
    assert p.confidence == 0.95


def test_provenance_confidence_range() -> None:
    with pytest.raises(ValidationError):
        Provenance(kind="doc", confidence=1.5)
    with pytest.raises(ValidationError):
        Provenance(kind="doc", confidence=-0.1)
