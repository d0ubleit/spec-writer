"""S1.8: core-traceability validators + runner."""

from __future__ import annotations

from pathlib import Path

from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.sources import FilesystemMixinSource
from spec_writer.methodology.validators import ValidatorContext
from spec_writer.methodology.validators.builtin import (
    check_dangling_refs,
    check_english_output,
    check_pyramid_adjacency,
    check_required_frontmatter,
    check_required_sections,
    check_vertical_symmetry,
)
from spec_writer.methodology.validators.runner import run_validators
from spec_writer.types import Spec

# ── fixtures ─────────────────────────────────────────────────────────────


def _methodology(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(
        (
            "version: 1\n"
            "layers:\n"
            "  - id: L1\n    role: source-boundary\n    name: vision\n"
            "  - id: L2\n    role: middle\n    name: capabilities\n"
            "  - id: L3\n    role: middle\n    name: behaviors\n"
            "  - id: L4\n    role: code-boundary\n    name: technical\n    is_final: true\n"
        ),
        encoding="utf-8",
    )
    mixins = tmp_path / "mixins"
    mixins.mkdir()
    (mixins / "vision.yaml").write_text(
        "mixin: vision\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins / "cap.yaml").write_text(
        "mixin: cap\nadds_types:\n  - id: capability\n    layer_name: capabilities\n    prefix: CAP\n",
        encoding="utf-8",
    )
    (mixins / "bhv.yaml").write_text(
        "mixin: bhv\nadds_types:\n  - id: behavior\n    layer_name: behaviors\n    prefix: BHV\n",
        encoding="utf-8",
    )
    (mixins / "core-traceability.yaml").write_text(
        (
            "mixin: core-traceability\n"
            "validators:\n"
            "  - id: pyramid-adjacency\n"
            "    scope: per-spec\n"
            "    func_ref: validators.check_pyramid_adjacency\n"
            "  - id: dangling-ref\n"
            "    scope: cross-spec\n"
            "    func_ref: validators.check_dangling_refs\n"
            "  - id: vertical-symmetry\n"
            "    scope: cross-spec\n"
            "    func_ref: validators.check_vertical_symmetry\n"
            "  - id: english-output\n"
            "    scope: per-spec\n"
            "    func_ref: validators.check_english_output\n"
        ),
        encoding="utf-8",
    )
    return load_methodology(
        base,
        [FilesystemMixinSource(mixins)],
        active_mixins=["vision", "cap", "bhv", "core-traceability"],
    )


def _ctx(specs: list[Spec], methodology) -> ValidatorContext:
    return ValidatorContext(corpus={s.id: s for s in specs}, methodology=methodology)


# ── pyramid adjacency ────────────────────────────────────────────────────


def test_adjacency_l1_no_up_to_ok(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    spec = Spec(id="L1-VIS-001", type="vision", layer="L1", title="X", up_to=[])
    issues = check_pyramid_adjacency(spec, _ctx([spec], rm))
    assert issues == []


def test_adjacency_l1_with_up_to_errors(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    spec = Spec(id="L1-VIS-001", type="vision", layer="L1", title="X", up_to=["whatever"])
    issues = check_pyramid_adjacency(spec, _ctx([spec], rm))
    assert any(i.code == "pyramid_source_has_up_to" for i in issues)


def test_adjacency_l2_missing_up_to_errors(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    spec = Spec(id="L2-CAP-001", type="capability", layer="L2", title="X", up_to=[])
    issues = check_pyramid_adjacency(spec, _ctx([spec], rm))
    assert any(i.code == "pyramid_missing_up_to" for i in issues)


def test_adjacency_l2_correct_parent_ok(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    parent = Spec(id="L1-VIS-001", type="vision", layer="L1", title="V", up_to=[])
    child = Spec(id="L2-CAP-001", type="capability", layer="L2", title="C", up_to=["L1-VIS-001"])
    issues = check_pyramid_adjacency(child, _ctx([parent, child], rm))
    assert issues == []


def test_adjacency_l3_skip_to_l1_errors(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    grandparent = Spec(id="L1-VIS-001", type="vision", layer="L1", title="V", up_to=[])
    skipping = Spec(id="L3-BHV-001", type="behavior", layer="L3", title="B", up_to=["L1-VIS-001"])
    issues = check_pyramid_adjacency(skipping, _ctx([grandparent, skipping], rm))
    assert any(i.code == "pyramid_skip_level" for i in issues)


# ── dangling refs ────────────────────────────────────────────────────────


def test_dangling_up_to_detected(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    spec = Spec(id="L2-CAP-001", type="capability", layer="L2", title="X", up_to=["L1-VIS-099"])
    issues = check_dangling_refs([spec], _ctx([spec], rm))
    assert any(i.code == "dangling_ref" and i.field == "up_to" for i in issues)


def test_dangling_relates_to_detected(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    spec = Spec(
        id="L2-CAP-001",
        type="capability",
        layer="L2",
        title="X",
        up_to=[],
        relates_to=[{"id": "L2-CAP-999", "kind": "relates_to"}],
    )
    issues = check_dangling_refs([spec], _ctx([spec], rm))
    assert any(i.field == "relates_to" for i in issues)


def test_dangling_no_issues_when_clean(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    parent = Spec(id="L1-VIS-001", type="vision", layer="L1", title="P", up_to=[])
    child = Spec(id="L2-CAP-001", type="capability", layer="L2", title="C", up_to=["L1-VIS-001"])
    issues = check_dangling_refs([parent, child], _ctx([parent, child], rm))
    assert issues == []


# ── vertical symmetry ───────────────────────────────────────────────────


def test_vertical_symmetric_pair_ok(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    parent = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="P",
        up_to=[],
        down_to=["L2-CAP-001"],
    )
    child = Spec(
        id="L2-CAP-001",
        type="capability",
        layer="L2",
        title="C",
        up_to=["L1-VIS-001"],
    )
    issues = check_vertical_symmetry([parent, child], _ctx([parent, child], rm))
    assert issues == []


def test_vertical_asymmetric_detected(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    parent = Spec(id="L1-VIS-001", type="vision", layer="L1", title="P", up_to=[])
    child = Spec(
        id="L2-CAP-001",
        type="capability",
        layer="L2",
        title="C",
        up_to=["L1-VIS-001"],
    )
    # parent.down_to missing
    issues = check_vertical_symmetry([parent, child], _ctx([parent, child], rm))
    assert any(i.code == "vertical_asymmetry" for i in issues)


# ── english output ───────────────────────────────────────────────────────


def test_english_title_ok(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    spec = Spec(id="L1-VIS-001", type="vision", layer="L1", title="User Profile", up_to=[])
    issues = check_english_output(spec, _ctx([spec], rm))
    assert issues == []


def test_extended_latin_allowed(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    spec = Spec(id="L1-VIS-001", type="vision", layer="L1", title="Café Strategy", up_to=[])
    issues = check_english_output(spec, _ctx([spec], rm))
    assert issues == []


def test_cyrillic_title_errors(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    spec = Spec(id="L1-VIS-001", type="vision", layer="L1", title="Профиль", up_to=[])
    issues = check_english_output(spec, _ctx([spec], rm))
    assert any(i.code == "non_english_title" for i in issues)


def test_source_gloss_marker_allowed(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    spec = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="[SOURCE] Профиль / English: User Profile",
        up_to=[],
    )
    issues = check_english_output(spec, _ctx([spec], rm))
    assert issues == []


def test_cyrillic_status_errors(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    spec = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="X",
        up_to=[],
        status="одобрено",
    )
    issues = check_english_output(spec, _ctx([spec], rm))
    assert any(i.code == "non_english_field" and i.field == "status" for i in issues)


def test_cyrillic_tag_errors(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    spec = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="X",
        up_to=[],
        tags=["профиль"],
    )
    issues = check_english_output(spec, _ctx([spec], rm))
    assert any(i.field == "tags" for i in issues)


# ── structural completeness ──────────────────────────────────────────────


def _methodology_with_sections(tmp_path: Path):
    """Methodology where 'vision' has a required 'Purpose' section and a
    required 'tags' frontmatter field."""
    base = tmp_path / "base.yaml"
    base.write_text(
        ("version: 1\nlayers:\n  - id: L1\n    role: source-boundary\n    name: vision\n"),
        encoding="utf-8",
    )
    mixins = tmp_path / "mixins"
    mixins.mkdir()
    (mixins / "vision.yaml").write_text(
        (
            "mixin: vision\n"
            "adds_types:\n"
            "  - id: vision\n"
            "    layer_role: source-boundary\n"
            "    prefix: VIS\n"
            "    sections:\n"
            "      - id: purpose\n"
            "        title: Purpose\n"
            "        required: true\n"
            "      - id: notes\n"
            "        title: Notes\n"
            "        required: false\n"
            "    required_frontmatter:\n"
            "      - tags:\n"
            "          description: 'topic tags'\n"
            "          llm_guidance: 'lowercase kebab-case'\n"
        ),
        encoding="utf-8",
    )
    return load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["vision"])


def test_required_section_present_ok(tmp_path: Path) -> None:
    rm = _methodology_with_sections(tmp_path)
    spec = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="X",
        up_to=[],
        body="## Purpose\n\nA platform.\n",
    )
    issues = check_required_sections(spec, _ctx([spec], rm))
    assert issues == []


def test_required_section_missing_errors(tmp_path: Path) -> None:
    rm = _methodology_with_sections(tmp_path)
    spec = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="X",
        up_to=[],
        body="# Some other heading\n\nNo Purpose here.\n",
    )
    issues = check_required_sections(spec, _ctx([spec], rm))
    assert any(i.code == "missing_required_section" for i in issues)


def test_required_section_case_insensitive(tmp_path: Path) -> None:
    rm = _methodology_with_sections(tmp_path)
    spec = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="X",
        up_to=[],
        body="## purpose\n\nlowercase title still ok.\n",
    )
    issues = check_required_sections(spec, _ctx([spec], rm))
    assert issues == []


def test_optional_section_missing_ok(tmp_path: Path) -> None:
    rm = _methodology_with_sections(tmp_path)
    spec = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="X",
        up_to=[],
        body="## Purpose\n\nbody\n",
    )
    # Notes section omitted but it's optional — no issue from that branch
    issues = check_required_sections(spec, _ctx([spec], rm))
    assert all(i.field != "body" or "Notes" not in i.message for i in issues)


def test_required_frontmatter_present_ok(tmp_path: Path) -> None:
    rm = _methodology_with_sections(tmp_path)
    spec = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="X",
        up_to=[],
        tags=["a"],
    )
    issues = check_required_frontmatter(spec, _ctx([spec], rm))
    assert issues == []


def test_required_frontmatter_missing_errors(tmp_path: Path) -> None:
    rm = _methodology_with_sections(tmp_path)
    spec = Spec(id="L1-VIS-001", type="vision", layer="L1", title="X", up_to=[])
    issues = check_required_frontmatter(spec, _ctx([spec], rm))
    assert any(i.code == "missing_required_frontmatter" and i.field == "tags" for i in issues)


def test_required_frontmatter_empty_list_errors(tmp_path: Path) -> None:
    """Empty `tags: []` should fail required check — empty defeats purpose."""
    rm = _methodology_with_sections(tmp_path)
    spec = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="X",
        up_to=[],
        tags=[],
    )
    issues = check_required_frontmatter(spec, _ctx([spec], rm))
    # tags is a known core field on Spec (not in extras), so extras lookup
    # returns None → also flagged
    assert any(i.code == "missing_required_frontmatter" for i in issues)


# ── runner ───────────────────────────────────────────────────────────────


def test_runner_collects_per_spec_and_cross(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    parent = Spec(id="L1-VIS-001", type="vision", layer="L1", title="P", up_to=[])
    child_skip = Spec(id="L3-BHV-001", type="behavior", layer="L3", title="C", up_to=["L1-VIS-001"])
    issues = run_validators([parent, child_skip], rm)
    codes = {i.code for i in issues}
    assert "pyramid_skip_level" in codes  # per-spec triggered


def test_runner_runs_cross_spec_once(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    a = Spec(id="L1-VIS-001", type="vision", layer="L1", title="A", up_to=[])
    b = Spec(id="L2-CAP-001", type="capability", layer="L2", title="B", up_to=["L1-VIS-999"])
    issues = run_validators([a, b], rm)
    dangling = [i for i in issues if i.code == "dangling_ref"]
    # Each missing reference creates one issue (not duplicated by per-type iteration)
    assert len(dangling) == 1


def test_runner_empty_corpus_no_crash(tmp_path: Path) -> None:
    rm = _methodology(tmp_path)
    issues = run_validators([], rm)
    assert issues == []


def test_runner_unregistered_validator_warns(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "version: 1\nlayers:\n  - id: L1\n    role: source-boundary\n    name: vision\n",
        encoding="utf-8",
    )
    mixins = tmp_path / "mixins"
    mixins.mkdir()
    (mixins / "v.yaml").write_text(
        "mixin: v\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins / "bogus.yaml").write_text(
        (
            "mixin: bogus\n"
            "validators:\n"
            "  - id: ghost\n"
            "    scope: per-spec\n"
            "    func_ref: validators.does_not_exist\n"
        ),
        encoding="utf-8",
    )
    rm = load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["v", "bogus"])
    spec = Spec(id="L1-VIS-001", type="vision", layer="L1", title="X", up_to=[])
    issues = run_validators([spec], rm)
    assert any(i.code == "validator_not_registered" for i in issues)
