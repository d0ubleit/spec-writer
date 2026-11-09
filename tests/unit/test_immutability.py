"""S1.1: value objects are frozen (immutable)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spec_writer.types import (
    FrontmatterField,
    Provenance,
    RelatesToEntry,
    Section,
    TokenUsage,
)


def test_provenance_frozen() -> None:
    p = Provenance(kind="doc")
    with pytest.raises(ValidationError):
        p.kind = "code"  # type: ignore[misc]


def test_relates_to_frozen() -> None:
    r = RelatesToEntry(id="L1-VIS-001", kind="relates_to")
    with pytest.raises(ValidationError):
        r.id = "L1-VIS-002"  # type: ignore[misc]


def test_token_usage_frozen() -> None:
    t = TokenUsage(input=10, output=20, total=30)
    with pytest.raises(ValidationError):
        t.total = 50  # type: ignore[misc]


def test_frontmatter_field_frozen() -> None:
    f = FrontmatterField(name="method")
    with pytest.raises(ValidationError):
        f.name = "renamed"  # type: ignore[misc]


def test_section_frozen() -> None:
    s = Section(id="purpose", title="Purpose")
    with pytest.raises(ValidationError):
        s.title = "Modified"  # type: ignore[misc]
