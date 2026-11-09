"""S1.2: ProjectContext + safe_path."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from spec_writer.engine.project_context import (
    NoProjectRootError,
    PathTraversalError,
    ProjectContext,
)


def _bootstrap(root: Path) -> Path:
    """Create a minimal spec-editor/ layout at `root`."""
    (root / "spec-editor").mkdir()
    (root / "spec-editor" / "sources-raw").mkdir()
    (root / "spec-editor" / "sources").mkdir()
    (root / "spec-editor" / "specs").mkdir()
    (root / "spec-editor" / "_meta").mkdir()
    return root


def test_from_cwd_finds_root_at_cwd(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    ctx = ProjectContext.from_cwd(tmp_path)
    assert ctx.root == tmp_path.resolve()
    assert ctx.spec_editor == (tmp_path / "spec-editor").resolve()


def test_from_cwd_walks_up_to_root(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    ctx = ProjectContext.from_cwd(nested)
    assert ctx.root == tmp_path.resolve()


def test_from_cwd_raises_when_no_project_root(tmp_path: Path) -> None:
    with pytest.raises(NoProjectRootError):
        ProjectContext.from_cwd(tmp_path)


def test_from_cwd_uses_current_directory_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bootstrap(tmp_path)
    monkeypatch.chdir(tmp_path)
    ctx = ProjectContext.from_cwd()
    assert ctx.root == tmp_path.resolve()


def test_safe_path_resolves_under_root(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    ctx = ProjectContext.from_cwd(tmp_path)
    p = ctx.safe_path("spec-editor/specs/foo.md")
    assert p == (tmp_path / "spec-editor" / "specs" / "foo.md").resolve()


def test_safe_path_accepts_already_absolute_inside_root(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    ctx = ProjectContext.from_cwd(tmp_path)
    inside = (tmp_path / "spec-editor" / "specs" / "bar.md").resolve()
    p = ctx.safe_path(inside)
    assert p == inside


def test_safe_path_blocks_dotdot_escape(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    ctx = ProjectContext.from_cwd(tmp_path)
    with pytest.raises(PathTraversalError):
        ctx.safe_path("../etc/passwd")


def test_safe_path_blocks_absolute_outside(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    ctx = ProjectContext.from_cwd(tmp_path)
    with pytest.raises(PathTraversalError):
        ctx.safe_path("/etc/passwd")


@pytest.mark.skipif(sys.platform == "win32", reason="symlink semantics differ on Windows")
def test_safe_path_blocks_symlink_escape(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    target_outside = tmp_path.parent / "outside.txt"
    target_outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "spec-editor" / "leak"
    os.symlink(target_outside, link)

    ctx = ProjectContext.from_cwd(tmp_path)
    with pytest.raises(PathTraversalError):
        ctx.safe_path("spec-editor/leak")
