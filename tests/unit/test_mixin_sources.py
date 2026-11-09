"""S1.5: MixinSource — FilesystemMixinSource discovery."""

from __future__ import annotations

from pathlib import Path

from spec_writer.methodology.sources import FilesystemMixinSource


def test_discover_empty_dir(tmp_path: Path) -> None:
    assert FilesystemMixinSource(tmp_path).discover() == []


def test_discover_missing_dir(tmp_path: Path) -> None:
    src = FilesystemMixinSource(tmp_path / "nope")
    assert src.discover() == []


def test_discover_flat_yaml_files(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("mixin: a\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("mixin: b\n", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("# not yaml", encoding="utf-8")

    paths = FilesystemMixinSource(tmp_path).discover()
    names = [p.name for p in paths]
    assert "a.yaml" in names
    assert "b.yaml" in names
    assert "ignore.md" not in names


def test_discover_directory_mixin_with_sidecar(tmp_path: Path) -> None:
    """A directory `<name>/` containing `<name>.yaml` (+ optional .py)
    counts as a single discoverable mixin entry."""
    side = tmp_path / "core-traceability"
    side.mkdir()
    (side / "core-traceability.yaml").write_text("mixin: core-traceability\n", encoding="utf-8")
    (side / "validators.py").write_text("# sidecar\n", encoding="utf-8")

    paths = FilesystemMixinSource(tmp_path).discover()
    assert any(p.name == "core-traceability.yaml" for p in paths)


def test_discover_directory_without_named_yaml_ignored(tmp_path: Path) -> None:
    side = tmp_path / "weird"
    side.mkdir()
    (side / "other.yaml").write_text("mixin: x\n", encoding="utf-8")
    paths = FilesystemMixinSource(tmp_path).discover()
    assert paths == []
