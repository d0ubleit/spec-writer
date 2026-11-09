"""S4.7 — sync_state + change_detector + reverse_index + MCP tools."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from spec_writer.agent.tool_registry import ToolContext
from spec_writer.agent.tools.build_reverse_index import build_reverse_index
from spec_writer.agent.tools.detect_code_changes import detect_code_changes
from spec_writer.agent.tools.record_sync_complete import record_sync_complete
from spec_writer.engine import change_detector, reverse_index, sync_state
from spec_writer.engine.id_registry import IdRegistry
from spec_writer.engine.project_context import ProjectContext
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.sources import FilesystemMixinSource


def _ctx(tmp_path: Path) -> ToolContext:
    (tmp_path / "spec-editor").mkdir()
    for sub in ("sources-raw", "sources", "specs", "_meta"):
        (tmp_path / "spec-editor" / sub).mkdir()
    project_ctx = ProjectContext.from_cwd(tmp_path)

    meta_yaml = tmp_path / "meta"
    meta_yaml.mkdir()
    (meta_yaml / "base.yaml").write_text(
        "version: 1\nlayers:\n  - id: L4\n    role: code-boundary\n    name: contracts\n",
        encoding="utf-8",
    )
    mixins = meta_yaml / "mixins"
    mixins.mkdir()
    (mixins / "api.yaml").write_text(
        "mixin: api\nadds_types:\n  - id: api-contract\n    layer_role: code-boundary\n    prefix: API\n",
        encoding="utf-8",
    )
    methodology = load_methodology(
        meta_yaml / "base.yaml",
        [FilesystemMixinSource(mixins)],
        active_mixins=["api"],
    )
    return ToolContext(project_ctx=project_ctx, methodology=methodology, id_registry=IdRegistry())


def _init_git(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)


def _git_commit(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=root, check=True)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


# ── sync_state ──────────────────────────────────────────────────────────


def test_sync_state_load_missing_returns_defaults(tmp_path: Path) -> None:
    meta = tmp_path / "_meta"
    meta.mkdir()
    state = sync_state.load(meta)
    assert state.last_sync_commit is None
    assert state.code_scope_hash is None
    assert state.updated_at is None


def test_sync_state_save_load_round_trip(tmp_path: Path) -> None:
    meta = tmp_path / "_meta"
    meta.mkdir()
    sync_state.save(
        sync_state.SyncState(
            last_sync_commit="abc123",
            code_scope_hash="def456",
            updated_at="2026-05-15T00:00:00+00:00",
        ),
        meta,
    )
    loaded = sync_state.load(meta)
    assert loaded.last_sync_commit == "abc123"
    assert loaded.code_scope_hash == "def456"
    assert loaded.updated_at == "2026-05-15T00:00:00+00:00"


def test_sync_state_save_uses_now_when_updated_at_missing(tmp_path: Path) -> None:
    meta = tmp_path / "_meta"
    meta.mkdir()
    sync_state.save(sync_state.SyncState(last_sync_commit="x"), meta)
    loaded = sync_state.load(meta)
    assert loaded.updated_at is not None
    assert "T" in loaded.updated_at  # ISO format


def test_sync_state_corrupt_file_returns_empty(tmp_path: Path) -> None:
    meta = tmp_path / "_meta"
    meta.mkdir()
    (meta / "sync_state.json").write_text("not-json", encoding="utf-8")
    assert sync_state.load(meta) == sync_state.SyncState()


# ── change_detector — git route ─────────────────────────────────────────


def test_is_git_repo_false_for_plain_dir(tmp_path: Path) -> None:
    assert not change_detector.is_git_repo(tmp_path)


def test_is_git_repo_true_after_init(tmp_path: Path) -> None:
    _init_git(tmp_path)
    assert change_detector.is_git_repo(tmp_path)


def test_detect_via_git_first_run_no_baseline(tmp_path: Path) -> None:
    _init_git(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    commit = _git_commit(tmp_path, "initial")

    result = change_detector.detect_via_git(tmp_path, None)
    assert result.method == "git"
    assert result.current_commit == commit
    assert result.added == []
    assert result.modified == []


def test_detect_via_git_detects_added_modified_deleted(tmp_path: Path) -> None:
    _init_git(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    baseline = _git_commit(tmp_path, "initial")

    (tmp_path / "a.py").write_text("x = 10\n")  # modified
    (tmp_path / "b.py").unlink()  # deleted
    (tmp_path / "c.py").write_text("z = 3\n")  # added
    _git_commit(tmp_path, "second")

    result = change_detector.detect_via_git(tmp_path, baseline)
    assert "a.py" in result.modified
    assert "b.py" in result.deleted
    assert "c.py" in result.added
    assert result.baseline_lost is False


def test_detect_via_git_baseline_lost(tmp_path: Path) -> None:
    _init_git(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    _git_commit(tmp_path, "initial")

    result = change_detector.detect_via_git(tmp_path, "deadbeef1234567890")
    assert result.baseline_lost is True


# ── change_detector — hash route ────────────────────────────────────────


def test_compute_code_scope_hash_stable_on_same_state(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    h1 = change_detector.compute_code_scope_hash(tmp_path, ["src/**"], [])
    h2 = change_detector.compute_code_scope_hash(tmp_path, ["src/**"], [])
    assert h1 == h2


def test_compute_code_scope_hash_changes_on_file_edit(tmp_path: Path) -> None:
    import os
    import time

    (tmp_path / "src").mkdir()
    file = tmp_path / "src" / "a.py"
    file.write_text("x = 1\n")
    h1 = change_detector.compute_code_scope_hash(tmp_path, ["src/**"], [])
    time.sleep(0.01)
    file.write_text("x = 12345\n")
    # ensure mtime_ns actually advanced (Linux nanosecond resolution)
    os.utime(file, ns=(time.time_ns(), time.time_ns()))
    h2 = change_detector.compute_code_scope_hash(tmp_path, ["src/**"], [])
    assert h1 != h2


def test_detect_via_hash_no_baseline_returns_not_changed(tmp_path: Path) -> None:
    result = change_detector.detect_via_hash(tmp_path, ["**"], [], None)
    assert result.hash_changed is False
    assert result.method == "hash"
    assert result.current_hash is not None


def test_detect_via_hash_changed_flag(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    file = tmp_path / "src" / "a.py"
    file.write_text("x = 1\n")
    first = change_detector.compute_code_scope_hash(tmp_path, ["src/**"], [])
    # Add a NEW file so the path-set differs — guaranteed hash change
    (tmp_path / "src" / "b.py").write_text("y = 2\n")
    result = change_detector.detect_via_hash(tmp_path, ["src/**"], [], first)
    assert result.hash_changed is True


# ── reverse_index ───────────────────────────────────────────────────────


def test_reverse_index_empty_corpus(tmp_path: Path) -> None:
    specs = tmp_path / "specs"
    specs.mkdir()
    assert reverse_index.build(specs) == {}


def test_reverse_index_groups_specs_by_file(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    from spec_writer.agent.tool_registry import ToolRegistry
    from spec_writer.agent.tools.record_spec import register as register_record_spec

    reg = ToolRegistry()
    register_record_spec(reg)
    asyncio.run(
        reg.call(
            "record_spec",
            {
                "type": "api-contract",
                "layer": "L4",
                "title": "Login",
                "implemented_by": ["src/auth.py:1-10"],
            },
            ctx,
        )
    )
    asyncio.run(
        reg.call(
            "record_spec",
            {
                "type": "api-contract",
                "layer": "L4",
                "title": "Logout",
                "implemented_by": ["src/auth.py:20-30"],
            },
            ctx,
        )
    )
    index = reverse_index.build(ctx.project_ctx.specs)
    assert "src/auth.py" in index
    assert len(index["src/auth.py"]) == 2


# ── MCP-tool wrappers ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_detect_code_changes_hash_mode(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.py").write_text("y = 1\n")
    result = await detect_code_changes({}, ctx)
    assert result["ok"] is True
    assert result["method"] == "hash"
    assert result["current"]["hash"] is not None


@pytest.mark.asyncio
async def test_detect_code_changes_git_mode(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _init_git(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.py").write_text("y = 1\n")
    _git_commit(tmp_path, "initial")
    result = await detect_code_changes({}, ctx)
    assert result["method"] == "git"
    assert result["current"]["commit"] is not None


@pytest.mark.asyncio
async def test_build_reverse_index_via_tool(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    from spec_writer.agent.tool_registry import ToolRegistry
    from spec_writer.agent.tools.record_spec import register as register_record_spec

    reg = ToolRegistry()
    register_record_spec(reg)
    await reg.call(
        "record_spec",
        {
            "type": "api-contract",
            "layer": "L4",
            "title": "Login",
            "implemented_by": ["src/auth.py:1-10"],
        },
        ctx,
    )
    result = await build_reverse_index({}, ctx)
    assert result["ok"] is True
    assert "src/auth.py" in result["index"]
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_record_sync_complete_updates_baseline_hash(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    result = await record_sync_complete({}, ctx)
    assert result["ok"] is True
    assert result["current"]["code_scope_hash"] is not None
    # And the file persists
    loaded = sync_state.load(ctx.project_ctx.meta)
    assert loaded.code_scope_hash is not None


@pytest.mark.asyncio
async def test_record_sync_complete_uses_git_head_when_repo(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _init_git(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    commit = _git_commit(tmp_path, "initial")
    result = await record_sync_complete({}, ctx)
    assert result["current"]["last_sync_commit"] == commit


# ── MCP wiring ──────────────────────────────────────────────────────────


def test_default_allowed_tools_includes_sync_primitives() -> None:
    from spec_writer.agent.mcp_tools import default_allowed_tools

    names = default_allowed_tools()
    assert "mcp__spec-writer__detect_code_changes" in names
    assert "mcp__spec-writer__build_reverse_index" in names
    assert "mcp__spec-writer__record_sync_complete" in names
