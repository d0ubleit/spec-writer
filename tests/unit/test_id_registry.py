"""S1.3: IdRegistry — sequential, gap-filling allocation per (layer, prefix)."""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_writer.engine.id_registry import (
    IdRegistry,
    MalformedIdError,
    parse_id,
)
from spec_writer.types import Spec

# ── parse_id ─────────────────────────────────────────────────────────────


def test_parse_id_valid() -> None:
    assert parse_id("L2-CAP-007") == ("L2", "CAP", 7)
    assert parse_id("L4-API-012") == ("L4", "API", 12)
    assert parse_id("L1-VIS-001") == ("L1", "VIS", 1)


def test_parse_id_unpadded_ok() -> None:
    assert parse_id("L4-API-0123") == ("L4", "API", 123)


def test_parse_id_invalid() -> None:
    assert parse_id("CAP-007") is None
    assert parse_id("L2-cap-007") is None
    assert parse_id("L2-CAP-7") is None  # NNN must be 3+ digits
    assert parse_id("") is None
    assert parse_id("garbage") is None


# ── allocate ─────────────────────────────────────────────────────────────


def test_allocate_starts_at_001() -> None:
    reg = IdRegistry()
    assert reg.allocate("L2", "CAP") == "L2-CAP-001"


def test_allocate_sequential() -> None:
    reg = IdRegistry()
    ids = [reg.allocate("L2", "CAP") for _ in range(3)]
    assert ids == ["L2-CAP-001", "L2-CAP-002", "L2-CAP-003"]


def test_allocate_gap_filling() -> None:
    reg = IdRegistry()
    reg.register("L2-CAP-001")
    reg.register("L2-CAP-003")
    assert reg.allocate("L2", "CAP") == "L2-CAP-002"
    assert reg.allocate("L2", "CAP") == "L2-CAP-004"


def test_allocate_independent_buckets() -> None:
    reg = IdRegistry()
    assert reg.allocate("L2", "CAP") == "L2-CAP-001"
    assert reg.allocate("L3", "BHV") == "L3-BHV-001"
    assert reg.allocate("L2", "ENT") == "L2-ENT-001"


# ── is_taken ─────────────────────────────────────────────────────────────


def test_is_taken_after_allocate() -> None:
    reg = IdRegistry()
    reg.allocate("L2", "CAP")
    assert reg.is_taken("L2-CAP-001")
    assert not reg.is_taken("L2-CAP-002")


def test_is_taken_after_register() -> None:
    reg = IdRegistry()
    reg.register("L4-API-099")
    assert reg.is_taken("L4-API-099")


# ── fix_duplicates ───────────────────────────────────────────────────────


def test_fix_duplicates_renumbers_collisions() -> None:
    reg = IdRegistry()
    reg.register("L2-CAP-001")

    drafts = [
        Spec(id="L2-CAP-001", type="capability", layer="L2", title="Existing collision"),
        Spec(id="L2-CAP-002", type="capability", layer="L2", title="New"),
    ]
    fixed = reg.fix_duplicates(drafts)
    assert fixed == 1
    assert drafts[0].id != "L2-CAP-001"
    assert drafts[0].id.startswith("L2-CAP-")
    assert drafts[1].id == "L2-CAP-002"


def test_fix_duplicates_returns_zero_when_clean() -> None:
    reg = IdRegistry()
    drafts = [
        Spec(id="L2-CAP-007", type="capability", layer="L2", title="A"),
        Spec(id="L2-CAP-008", type="capability", layer="L2", title="B"),
    ]
    fixed = reg.fix_duplicates(drafts)
    assert fixed == 0
    assert reg.is_taken("L2-CAP-007")
    assert reg.is_taken("L2-CAP-008")


def test_fix_duplicates_raises_on_malformed_id() -> None:
    reg = IdRegistry()
    bad = Spec(id="bogus", type="capability", layer="L2", title="X")
    with pytest.raises(MalformedIdError):
        reg.fix_duplicates([bad])


# ── from_disk ────────────────────────────────────────────────────────────


def _write_spec(path: Path, spec_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'---\nid: {spec_id}\ntype: capability\nlayer: L2\ntitle: "x"\n---\nbody\n',
        encoding="utf-8",
    )


def test_from_disk_empty_dir(tmp_path: Path) -> None:
    reg = IdRegistry.from_disk(tmp_path)
    assert not reg.is_taken("L2-CAP-001")


def test_from_disk_recursive_scan(tmp_path: Path) -> None:
    _write_spec(tmp_path / "capabilities" / "a.md", "L2-CAP-001")
    _write_spec(tmp_path / "capabilities" / "b.md", "L2-CAP-005")
    _write_spec(tmp_path / "vision" / "v.md", "L1-VIS-001")
    reg = IdRegistry.from_disk(tmp_path)
    assert reg.is_taken("L2-CAP-001")
    assert reg.is_taken("L2-CAP-005")
    assert reg.is_taken("L1-VIS-001")
    assert reg.allocate("L2", "CAP") == "L2-CAP-002"


def test_from_disk_skips_malformed_id(tmp_path: Path) -> None:
    _write_spec(tmp_path / "good.md", "L2-CAP-001")
    (tmp_path / "bad.md").write_text(
        "---\nid: not-valid\nlayer: L2\ntype: x\ntitle: y\n---\n", encoding="utf-8"
    )
    (tmp_path / "no_id.md").write_text("---\ntype: x\n---\n", encoding="utf-8")
    (tmp_path / "no_frontmatter.md").write_text("# just markdown\n", encoding="utf-8")
    reg = IdRegistry.from_disk(tmp_path)
    assert reg.is_taken("L2-CAP-001")


def test_from_disk_extracts_quoted_id(tmp_path: Path) -> None:
    (tmp_path / "q.md").write_text(
        '---\nid: "L2-CAP-042"\ntype: capability\nlayer: L2\ntitle: "X"\n---\nbody\n',
        encoding="utf-8",
    )
    reg = IdRegistry.from_disk(tmp_path)
    assert reg.is_taken("L2-CAP-042")


def test_from_disk_returns_empty_when_dir_missing(tmp_path: Path) -> None:
    reg = IdRegistry.from_disk(tmp_path / "does-not-exist")
    assert not reg.is_taken("L2-CAP-001")
    dups = IdRegistry.find_duplicates(tmp_path / "does-not-exist")
    assert dups == {}


def test_register_rejects_malformed() -> None:
    reg = IdRegistry()
    with pytest.raises(MalformedIdError):
        reg.register("not-an-id")


# ── find_duplicates ──────────────────────────────────────────────────────


def test_find_duplicates_none(tmp_path: Path) -> None:
    _write_spec(tmp_path / "a.md", "L2-CAP-001")
    _write_spec(tmp_path / "b.md", "L2-CAP-002")
    dups = IdRegistry.find_duplicates(tmp_path)
    assert dups == {}


def test_find_duplicates_reports_collisions(tmp_path: Path) -> None:
    _write_spec(tmp_path / "a.md", "L2-CAP-001")
    _write_spec(tmp_path / "nested" / "b.md", "L2-CAP-001")
    _write_spec(tmp_path / "c.md", "L2-CAP-002")
    dups = IdRegistry.find_duplicates(tmp_path)
    assert "L2-CAP-001" in dups
    assert len(dups["L2-CAP-001"]) == 2
    assert "L2-CAP-002" not in dups
