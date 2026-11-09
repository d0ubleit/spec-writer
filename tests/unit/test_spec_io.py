"""S1.10: spec_io write/read + path computation."""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_writer.engine.project_context import ProjectContext
from spec_writer.io.spec_io import (
    SpecIoError,
    compute_spec_path,
    find_spec_by_id,
    read_spec,
    write_spec,
)
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.sources import FilesystemMixinSource
from spec_writer.types import Spec


def _bootstrap(tmp_path: Path) -> tuple[ProjectContext, object]:
    (tmp_path / "spec-editor").mkdir()
    for sub in ("sources-raw", "sources", "specs", "_meta"):
        (tmp_path / "spec-editor" / sub).mkdir()
    ctx = ProjectContext.from_cwd(tmp_path)

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "base.yaml").write_text(
        (
            "version: 1\n"
            "layers:\n"
            "  - id: L1\n    role: source-boundary\n    name: vision\n"
            "  - id: L2\n    role: middle\n    name: capabilities\n"
            "  - id: L4\n    role: code-boundary\n    name: technical\n    is_final: true\n"
        ),
        encoding="utf-8",
    )
    mixins = meta / "mixins"
    mixins.mkdir()
    (mixins / "vision.yaml").write_text(
        "mixin: vision\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins / "api.yaml").write_text(
        (
            "mixin: api\n"
            "adds_types:\n"
            "  - id: api-contract\n"
            "    layer_role: code-boundary\n"
            "    prefix: API\n"
            "    file_pattern: 'api/{slug}-API-{number}.md'\n"
        ),
        encoding="utf-8",
    )
    rm = load_methodology(
        meta / "base.yaml",
        [FilesystemMixinSource(mixins)],
        active_mixins=["vision", "api"],
    )
    return ctx, rm


def test_compute_path_basic(tmp_path: Path) -> None:
    ctx, rm = _bootstrap(tmp_path)
    spec = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="Platform Vision",
        up_to=[],
    )
    path = compute_spec_path(spec, rm, ctx)
    assert path == ctx.specs / "vision" / "platform-vision-VIS-001.md"


def test_compute_path_with_anchor(tmp_path: Path) -> None:
    ctx, rm = _bootstrap(tmp_path)
    spec = Spec(
        id="L4-API-005",
        type="api-contract",
        layer="L4",
        title="PATCH /profiles",
        up_to=[],
    )
    path = compute_spec_path(spec, rm, ctx)
    assert path == ctx.specs / "technical" / "api" / "patch-profiles-API-005.md"


def test_compute_path_malformed_id(tmp_path: Path) -> None:
    ctx, rm = _bootstrap(tmp_path)
    spec = Spec(id="not-valid", type="vision", layer="L1", title="X", up_to=[])
    with pytest.raises(SpecIoError):
        compute_spec_path(spec, rm, ctx)


def test_compute_path_unknown_layer(tmp_path: Path) -> None:
    ctx, rm = _bootstrap(tmp_path)
    spec = Spec(id="L9-VIS-001", type="vision", layer="L9", title="X", up_to=[])
    with pytest.raises(SpecIoError):
        compute_spec_path(spec, rm, ctx)


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    ctx, rm = _bootstrap(tmp_path)
    spec = Spec(
        id="L1-VIS-001",
        type="vision",
        layer="L1",
        title="Platform Vision",
        up_to=[],
        tags=["platform"],
        body="## Purpose\n\nA platform.\n",
    )
    path = write_spec(spec, rm, ctx)
    assert path.exists()

    reloaded = read_spec(path)
    assert reloaded.id == "L1-VIS-001"
    assert reloaded.title == "Platform Vision"
    assert reloaded.tags == ["platform"]
    assert "A platform." in reloaded.body


def test_write_renders_canonical_frontmatter(tmp_path: Path) -> None:
    ctx, rm = _bootstrap(tmp_path)
    spec = Spec(id="L1-VIS-001", type="vision", layer="L1", title="X", up_to=[])
    path = write_spec(spec, rm, ctx)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\nid: L1-VIS-001\n")
    assert 'title: "X"\n' in text
    assert "up_to: []\n" in text


def test_find_spec_by_id(tmp_path: Path) -> None:
    ctx, rm = _bootstrap(tmp_path)
    spec = Spec(id="L1-VIS-001", type="vision", layer="L1", title="X", up_to=[])
    expected = write_spec(spec, rm, ctx)
    found = find_spec_by_id(ctx.specs, "L1-VIS-001")
    assert found == expected


def test_find_spec_by_id_missing(tmp_path: Path) -> None:
    ctx, _rm = _bootstrap(tmp_path)
    assert find_spec_by_id(ctx.specs, "L1-VIS-999") is None


def test_find_spec_by_id_no_specs_dir(tmp_path: Path) -> None:
    assert find_spec_by_id(tmp_path / "nope", "L1-VIS-001") is None
