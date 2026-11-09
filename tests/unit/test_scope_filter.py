"""S3.6 — Scope filter applied to corpus + seed."""

from __future__ import annotations

from pathlib import Path

from spec_writer.engine.project_context import ProjectContext
from spec_writer.engine.scope import filter_corpus, spec_matches_scope
from spec_writer.engine.seed import build_seed
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.sources import FilesystemMixinSource
from spec_writer.types import Scope, Spec


def _spec(spec_id: str, type_: str, layer: str, tags: list[str] | None = None) -> Spec:
    return Spec(
        id=spec_id,
        type=type_,
        layer=layer,
        title=spec_id,
        tags=tags or [],
        body="",
    )


def _methodology(tmp_path: Path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "base.yaml").write_text(
        (
            "version: 1\n"
            "layers:\n"
            "  - id: L1\n    role: source-boundary\n    name: vision\n"
            "  - id: L2\n    role: middle\n    name: capabilities\n"
            "  - id: L3\n    role: middle\n    name: behaviors\n"
        ),
        encoding="utf-8",
    )
    mixins = meta / "mixins"
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
    return load_methodology(
        meta / "base.yaml",
        [FilesystemMixinSource(mixins)],
        active_mixins=["vision", "cap", "bhv"],
    )


# ── spec_matches_scope ──────────────────────────────────────────────────


def test_scope_none_filters_pass_through(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    spec = _spec("L1-VIS-001", "vision", "L1")
    assert spec_matches_scope(spec, Scope(), m)


def test_scope_layers_filter(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    spec = _spec("L1-VIS-001", "vision", "L1")
    assert spec_matches_scope(spec, Scope(layers=["L1"]), m)
    assert not spec_matches_scope(spec, Scope(layers=["L2"]), m)


def test_scope_types_filter(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    spec = _spec("L2-CAP-001", "capability", "L2")
    assert spec_matches_scope(spec, Scope(types=["capability"]), m)
    assert not spec_matches_scope(spec, Scope(types=["vision"]), m)


def test_scope_tags_filter_any_overlap(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    spec = _spec("L3-BHV-001", "behavior", "L3", tags=["auth", "core"])
    assert spec_matches_scope(spec, Scope(tags=["auth"]), m)
    assert spec_matches_scope(spec, Scope(tags=["auth", "missing"]), m)
    assert not spec_matches_scope(spec, Scope(tags=["unknown"]), m)


def test_scope_spec_ids_filter(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    spec = _spec("L2-CAP-001", "capability", "L2")
    assert spec_matches_scope(spec, Scope(spec_ids=["L2-CAP-001"]), m)
    assert not spec_matches_scope(spec, Scope(spec_ids=["L2-CAP-099"]), m)


def test_scope_folder_pattern_matches(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    spec = _spec("L2-CAP-001", "capability", "L2")
    # synthetic path is "capabilities/L2-CAP-001"
    assert spec_matches_scope(spec, Scope(folder_pattern="capabilities/*"), m)
    assert not spec_matches_scope(spec, Scope(folder_pattern="vision/*"), m)


def test_scope_multiple_dimensions_and_semantics(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    spec = _spec("L2-CAP-001", "capability", "L2", tags=["auth"])
    # Both must hold
    scope = Scope(layers=["L2"], types=["capability"], tags=["auth"])
    assert spec_matches_scope(spec, scope, m)
    scope_fail = Scope(layers=["L2"], types=["vision"])
    assert not spec_matches_scope(spec, scope_fail, m)


# ── filter_corpus ───────────────────────────────────────────────────────


def test_filter_corpus_none_returns_copy(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    corpus = {"L1-VIS-001": _spec("L1-VIS-001", "vision", "L1")}
    out = filter_corpus(corpus, None, m)
    assert out == corpus
    assert out is not corpus  # copy, not reference


def test_filter_corpus_narrows(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    corpus = {
        "L1-VIS-001": _spec("L1-VIS-001", "vision", "L1"),
        "L2-CAP-001": _spec("L2-CAP-001", "capability", "L2"),
        "L2-CAP-002": _spec("L2-CAP-002", "capability", "L2"),
        "L3-BHV-001": _spec("L3-BHV-001", "behavior", "L3"),
    }
    out = filter_corpus(corpus, Scope(layers=["L2"]), m)
    assert set(out) == {"L2-CAP-001", "L2-CAP-002"}


# ── build_seed with scope ───────────────────────────────────────────────


def _project(tmp_path: Path) -> ProjectContext:
    (tmp_path / "spec-editor").mkdir()
    for sub in ("sources-raw", "sources", "specs", "_meta"):
        (tmp_path / "spec-editor" / sub).mkdir()
    return ProjectContext.from_cwd(tmp_path)


def test_build_seed_with_scope_filters_summary(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    project_ctx = _project(tmp_path)
    corpus = {
        "L2-CAP-001": _spec("L2-CAP-001", "capability", "L2", tags=["auth"]),
        "L2-CAP-002": _spec("L2-CAP-002", "capability", "L2", tags=["payments"]),
    }
    layer = next(lay for lay in m.layers if lay.id == "L2")
    seed = build_seed(
        layer,
        "fwd",
        m,
        project_ctx,
        corpus,
        mode="refine",
        scope=Scope(tags=["auth"]),
    )
    # Summary should include only the auth-tagged spec.
    assert "L2-CAP-001" in seed.existing_corpus_summary
    assert "L2-CAP-002" not in seed.existing_corpus_summary
    # Note records the narrowing for the agent's benefit
    assert any("Scope filter active" in n for n in seed.notes)


def test_build_seed_without_scope_keeps_all(tmp_path: Path) -> None:
    m = _methodology(tmp_path)
    project_ctx = _project(tmp_path)
    corpus = {
        "L2-CAP-001": _spec("L2-CAP-001", "capability", "L2"),
        "L2-CAP-002": _spec("L2-CAP-002", "capability", "L2"),
    }
    layer = next(lay for lay in m.layers if lay.id == "L2")
    seed = build_seed(layer, "fwd", m, project_ctx, corpus, mode="refine")
    assert set(seed.existing_corpus_summary) == {"L2-CAP-001", "L2-CAP-002"}
    assert not any("Scope filter" in n for n in seed.notes)
