"""S1.5: methodology loader — base + mixins → ResolvedMethodology."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from spec_writer.methodology.loader import (
    MethodologyLoadError,
    load_methodology,
)
from spec_writer.methodology.sources import FilesystemMixinSource

# ── helpers ──────────────────────────────────────────────────────────────


def _builtin_base() -> Path:
    base = resources.files("spec_writer.methodology._builtin").joinpath("base.yaml")
    return Path(str(base))


def _builtin_mixins_dir() -> Path:
    return Path(str(resources.files("spec_writer.methodology._builtin").joinpath("mixins")))


def _write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ── 1. load_base ─────────────────────────────────────────────────────────


def test_load_base_yields_four_layers() -> None:
    rm = load_methodology(_builtin_base(), [], active_mixins=[])
    layers = rm.layers_in_order("fwd")
    assert [layer.id for layer in layers] == ["L1", "L2", "L3", "L4"]
    assert layers[0].role == "source-boundary"
    assert layers[-1].role == "code-boundary"
    assert layers[-1].is_final is True


def test_load_base_rev_direction_reversed() -> None:
    rm = load_methodology(_builtin_base(), [], active_mixins=[])
    layers = rm.layers_in_order("rev")
    assert [layer.id for layer in layers] == ["L4", "L3", "L2", "L1"]


def test_load_base_no_types_yet() -> None:
    rm = load_methodology(_builtin_base(), [], active_mixins=[])
    assert rm.types_at_layer("L1") == []
    assert rm.types_at_layer("L4") == []


# ── 2. load_single_mixin ─────────────────────────────────────────────────


def test_vision_mixin_adds_l1_type() -> None:
    rm = load_methodology(
        _builtin_base(),
        [FilesystemMixinSource(_builtin_mixins_dir())],
        active_mixins=["vision"],
    )
    types = rm.types_at_layer("L1")
    assert len(types) == 1
    assert types[0].id == "vision"
    assert types[0].prefix == "VIS"
    assert rm.prefix_for("vision") == "VIS"


def test_vision_mixin_provides_generator_hint() -> None:
    rm = load_methodology(
        _builtin_base(),
        [FilesystemMixinSource(_builtin_mixins_dir())],
        active_mixins=["vision"],
    )
    hint = rm.generator_hints("L1", "vision")
    assert "L1 vision" in hint


# ── 3. multiple mixins ───────────────────────────────────────────────────


def test_multiple_mixins_compose() -> None:
    rm = load_methodology(
        _builtin_base(),
        [FilesystemMixinSource(_builtin_mixins_dir())],
        active_mixins=["vision", "capabilities", "core-traceability"],
    )
    l1_types = rm.types_at_layer("L1")
    l2_types = rm.types_at_layer("L2")
    assert [t.id for t in l1_types] == ["vision"]
    assert [t.id for t in l2_types] == ["capability"]
    # core-traceability validators registered globally
    validator_ids = {v["id"] for v in rm.validators_for("vision")}
    assert "pyramid-adjacency" in validator_ids
    assert "english-output" in validator_ids


# ── 4. mixin targeting ───────────────────────────────────────────────────


def test_layer_role_targeting(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "by_role.yaml",
        (
            "mixin: by_role\n"
            "adds_types:\n"
            "  - id: t-role\n"
            "    layer_role: code-boundary\n"
            "    prefix: TR\n"
        ),
    )
    rm = load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["by_role"])
    assert any(t.id == "t-role" for t in rm.types_at_layer("L4"))


def test_layer_name_targeting(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "by_name.yaml",
        (
            "mixin: by_name\n"
            "adds_types:\n"
            "  - id: t-name\n"
            "    layer_name: behaviors\n"
            "    prefix: TN\n"
        ),
    )
    rm = load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["by_name"])
    assert any(t.id == "t-name" for t in rm.types_at_layer("L3"))


def test_unknown_layer_role_error(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "bad.yaml",
        ("mixin: bad\nadds_types:\n  - id: t-bad\n    layer_name: nonexistent\n    prefix: BD\n"),
    )
    with pytest.raises(MethodologyLoadError, match="nonexistent"):
        load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["bad"])


# ── 5. conflict detection (collect-all) ──────────────────────────────────


def test_duplicate_type_id_collected(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "a.yaml",
        "mixin: a\nadds_types:\n  - id: dup\n    layer_role: source-boundary\n    prefix: AA\n",
    )
    _write_yaml(
        mixins / "b.yaml",
        "mixin: b\nadds_types:\n  - id: dup\n    layer_role: source-boundary\n    prefix: BB\n",
    )
    with pytest.raises(MethodologyLoadError) as ei:
        load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["a", "b"])
    assert "dup" in str(ei.value)


def test_duplicate_section_id_collected(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "owner.yaml",
        (
            "mixin: owner\n"
            "adds_types:\n"
            "  - id: cap\n"
            "    layer_name: capabilities\n"
            "    prefix: CA\n"
            "    sections:\n"
            "      - id: purpose\n"
            "        title: Purpose\n"
        ),
    )
    _write_yaml(
        mixins / "extender_a.yaml",
        (
            "mixin: extender_a\n"
            "extends_types:\n"
            "  - id: cap\n"
            "    add_sections:\n"
            "      - id: actors\n"
            "        title: Actors\n"
        ),
    )
    _write_yaml(
        mixins / "extender_b.yaml",
        (
            "mixin: extender_b\n"
            "extends_types:\n"
            "  - id: cap\n"
            "    add_sections:\n"
            "      - id: actors\n"
            "        title: Actors duplicate\n"
        ),
    )
    with pytest.raises(MethodologyLoadError) as ei:
        load_methodology(
            base,
            [FilesystemMixinSource(mixins)],
            active_mixins=["owner", "extender_a", "extender_b"],
        )
    assert "actors" in str(ei.value)


# ── 6. missing mixin with suggestion ─────────────────────────────────────


def test_missing_mixin_with_suggestion() -> None:
    with pytest.raises(MethodologyLoadError) as ei:
        load_methodology(
            _builtin_base(),
            [FilesystemMixinSource(_builtin_mixins_dir())],
            active_mixins=["visionz"],
        )
    msg = str(ei.value)
    assert "visionz" in msg
    assert "vision" in msg  # close match suggestion


# ── 7. resolved API surface ──────────────────────────────────────────────


def test_resolved_api_layers_and_types() -> None:
    rm = load_methodology(
        _builtin_base(),
        [FilesystemMixinSource(_builtin_mixins_dir())],
        active_mixins=["vision", "capabilities", "core-traceability"],
    )
    assert rm.prefix_for("vision") == "VIS"
    assert rm.prefix_for("capability") == "CAP"
    # generator hints fall back to "" for unknown
    assert rm.generator_hints("L3", "unknown") == ""
    # tools/hooks/metrics — empty lists for now (Phase 1 minimal)
    assert rm.tools_for("L1") == []
    assert rm.hooks_for("after_run") == []
    assert rm.metrics_for("L1") == []


# ── 8. extends_field ─────────────────────────────────────────────────────


def test_extends_field_enum(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "lifecycle.yaml",
        (
            "mixin: lifecycle\n"
            "extends_field:\n"
            "  - name: status\n"
            "    add_values: [draft, review, approved]\n"
        ),
    )
    _write_yaml(
        mixins / "compliance.yaml",
        (
            "mixin: compliance\n"
            "extends_field:\n"
            "  - name: status\n"
            "    add_values: [audited, certified]\n"
        ),
    )
    rm = load_methodology(
        base, [FilesystemMixinSource(mixins)], active_mixins=["lifecycle", "compliance"]
    )
    values = rm.field_enum_values("status")
    assert set(values) == {"draft", "review", "approved", "audited", "certified"}


# ── extra coverage: error paths + global features ───────────────────────


def test_adds_types_missing_id(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "bad.yaml",
        "mixin: bad\nadds_types:\n  - layer_role: source-boundary\n    prefix: XX\n",
    )
    with pytest.raises(MethodologyLoadError, match="missing string 'id'"):
        load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["bad"])


def test_adds_types_missing_prefix(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "bad.yaml",
        "mixin: bad\nadds_types:\n  - id: t\n    layer_role: source-boundary\n",
    )
    with pytest.raises(MethodologyLoadError, match="missing string 'prefix'"):
        load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["bad"])


def test_adds_types_missing_layer_target(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "bad.yaml",
        "mixin: bad\nadds_types:\n  - id: t\n    prefix: T\n",
    )
    with pytest.raises(MethodologyLoadError, match="missing layer"):
        load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["bad"])


def test_layer_id_targeting(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "by_id.yaml",
        "mixin: by_id\nadds_types:\n  - id: t-id\n    layer: L2\n    prefix: TI\n",
    )
    rm = load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["by_id"])
    assert any(t.id == "t-id" for t in rm.types_at_layer("L2"))


def test_layer_role_ambiguous(tmp_path: Path) -> None:
    """layer_role: middle is ambiguous — multiple matches."""
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "bad.yaml",
        "mixin: bad\nadds_types:\n  - id: t\n    layer_role: middle\n    prefix: T\n",
    )
    with pytest.raises(MethodologyLoadError, match="multiple layers"):
        load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["bad"])


def test_extends_types_unknown_target(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "bad.yaml",
        (
            "mixin: bad\n"
            "extends_types:\n"
            "  - id: nonexistent\n"
            "    add_sections:\n"
            "      - id: s\n"
            "        title: S\n"
        ),
    )
    with pytest.raises(MethodologyLoadError, match="unknown type"):
        load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["bad"])


def test_metrics_and_hooks_registered(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "instr.yaml",
        (
            "mixin: instr\n"
            "metrics:\n"
            "  - id: m1\n"
            "    layer: L2\n"
            "    target: 0.9\n"
            "hooks:\n"
            "  after_layer:\n"
            "    - tool: git.commit\n"
            "tools_by_layer:\n"
            "  L1:\n"
            "    - name: rag.search\n"
        ),
    )
    rm = load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["instr"])
    assert any(m["id"] == "m1" for m in rm.metrics_for("L2"))
    assert any(h["tool"] == "git.commit" for h in rm.hooks_for("after_layer"))
    assert any(t["name"] == "rag.search" for t in rm.tools_for("L1"))


def test_long_form_frontmatter(tmp_path: Path) -> None:
    """required_frontmatter supports long-form single-key map."""
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "api.yaml",
        (
            "mixin: api\n"
            "adds_types:\n"
            "  - id: api-contract\n"
            "    layer_role: code-boundary\n"
            "    prefix: API\n"
            "    required_frontmatter:\n"
            "      - method\n"
            "      - auth:\n"
            "          description: 'auth requirement'\n"
            "          llm_guidance: 'pick one of anon/authd/role'\n"
            "          validation:\n"
            "            kind: enum\n"
            "            values: [anon, authd, role]\n"
        ),
    )
    rm = load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["api"])
    api = rm.spec_types["api-contract"]
    method_field = next(f for f in api.required_frontmatter if f.name == "method")
    auth_field = next(f for f in api.required_frontmatter if f.name == "auth")
    assert method_field.description == ""
    assert auth_field.description == "auth requirement"
    assert auth_field.validation is not None
    assert auth_field.validation["kind"] == "enum"


def test_unknown_layer_id_error(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "bad.yaml",
        "mixin: bad\nadds_types:\n  - id: t\n    layer: L99\n    prefix: BD\n",
    )
    with pytest.raises(MethodologyLoadError, match="L99"):
        load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["bad"])


def test_unknown_layer_role_no_match(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "bad.yaml",
        "mixin: bad\nadds_types:\n  - id: t\n    layer_role: nonexistent\n    prefix: T\n",
    )
    with pytest.raises(MethodologyLoadError, match="no layer with role"):
        load_methodology(base, [FilesystemMixinSource(mixins)], active_mixins=["bad"])


def test_extends_types_added_sections(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "base_cap.yaml",
        (
            "mixin: base_cap\n"
            "adds_types:\n"
            "  - id: cap\n"
            "    layer_name: capabilities\n"
            "    prefix: CA\n"
            "    sections:\n"
            "      - id: purpose\n"
            "        title: Purpose\n"
        ),
    )
    _write_yaml(
        mixins / "actors.yaml",
        (
            "mixin: actors\n"
            "extends_types:\n"
            "  - id: cap\n"
            "    add_sections:\n"
            "      - id: actors\n"
            "        title: Actors\n"
            "    add_required_frontmatter:\n"
            "      - actor_count\n"
            "    add_optional_frontmatter:\n"
            "      - actor_notes\n"
        ),
    )
    rm = load_methodology(
        base, [FilesystemMixinSource(mixins)], active_mixins=["base_cap", "actors"]
    )
    cap = rm.spec_types["cap"]
    assert [s.id for s in cap.sections] == ["purpose", "actors"]
    assert any(f.name == "actor_count" for f in cap.required_frontmatter)
    assert any(f.name == "actor_notes" for f in cap.optional_frontmatter)


# ── 9. extends_all_types (cross-cutting required field) ──────────────────


def test_extends_all_types_applies_universally(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_base(), encoding="utf-8")
    mixins = tmp_path / "mixins"
    _write_yaml(
        mixins / "vision_local.yaml",
        (
            "mixin: vision_local\n"
            "adds_types:\n"
            "  - id: vis\n"
            "    layer_role: source-boundary\n"
            "    prefix: VI\n"
        ),
    )
    _write_yaml(
        mixins / "cap_local.yaml",
        (
            "mixin: cap_local\n"
            "adds_types:\n"
            "  - id: cap\n"
            "    layer_name: capabilities\n"
            "    prefix: CA\n"
        ),
    )
    _write_yaml(
        mixins / "tagging.yaml",
        ("mixin: tagging\nextends_all_types:\n  required_frontmatter:\n    - tags\n"),
    )
    rm = load_methodology(
        base,
        [FilesystemMixinSource(mixins)],
        active_mixins=["vision_local", "cap_local", "tagging"],
    )
    vis = next(t for t in rm.types_at_layer("L1") if t.id == "vis")
    cap = next(t for t in rm.types_at_layer("L2") if t.id == "cap")
    assert any(f.name == "tags" for f in vis.required_frontmatter)
    assert any(f.name == "tags" for f in cap.required_frontmatter)


# ── helpers (yaml templates) ─────────────────────────────────────────────


def _minimal_base() -> str:
    return (
        "version: 1\n"
        "layers:\n"
        "  - id: L1\n"
        "    role: source-boundary\n"
        "    name: vision\n"
        "  - id: L2\n"
        "    role: middle\n"
        "    name: capabilities\n"
        "  - id: L3\n"
        "    role: middle\n"
        "    name: behaviors\n"
        "  - id: L4\n"
        "    role: code-boundary\n"
        "    name: technical\n"
        "    is_final: true\n"
    )
