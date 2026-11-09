"""S4.3 — discover_code_inventory MCP tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.agent.tools.discover_code_inventory import (
    DiscoverCodeInventoryError,
    discover_code_inventory,
)
from spec_writer.agent.tools.discover_code_inventory import (
    register as register_dci,
)
from spec_writer.engine.id_registry import IdRegistry
from spec_writer.engine.project_context import ProjectContext
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.sources import FilesystemMixinSource


def _ctx(tmp_path: Path) -> ToolContext:
    (tmp_path / "spec-editor").mkdir()
    for sub in ("sources-raw", "sources", "specs", "_meta"):
        (tmp_path / "spec-editor" / sub).mkdir()
    project_ctx = ProjectContext.from_cwd(tmp_path)

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "base.yaml").write_text(
        "version: 1\nlayers:\n  - id: L1\n    role: source-boundary\n    name: vision\n",
        encoding="utf-8",
    )
    mixins = meta / "mixins"
    mixins.mkdir()
    (mixins / "v.yaml").write_text(
        "mixin: v\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    methodology = load_methodology(
        meta / "base.yaml", [FilesystemMixinSource(mixins)], active_mixins=["v"]
    )
    return ToolContext(project_ctx=project_ctx, methodology=methodology, id_registry=IdRegistry())


def _seed_code(project_root: Path) -> None:
    src = project_root / "src"
    src.mkdir()
    (src / "auth.py").write_text(
        "def login(user, password):\n    return True\n\n"
        "class AuthService:\n    def logout(self):\n        return None\n"
    )
    (src / "User.php").write_text(
        "<?php\nclass User {\n  public function getName() { return $this->name; }\n}\n"
    )
    (src / "ui.tsx").write_text("export function App() { return null; }\n")
    # excluded files
    (project_root / "node_modules").mkdir()
    (project_root / "node_modules" / "lib.js").write_text("export function ignored() {}\n")


# ── walk + filter ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_discover_finds_python_php_typescript(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _seed_code(tmp_path)
    result = await discover_code_inventory({}, ctx)
    assert result["ok"] is True
    paths = {f["path"] for f in result["files"]}
    assert any(p.endswith("auth.py") for p in paths)
    assert any(p.endswith("User.php") for p in paths)
    assert any(p.endswith("ui.tsx") for p in paths)
    assert all("node_modules" not in p for p in paths)


@pytest.mark.asyncio
async def test_discover_respects_custom_excludes(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _seed_code(tmp_path)
    result = await discover_code_inventory(
        {"includes": ["src/**"], "excludes": ["src/auth.py"]}, ctx
    )
    paths = {f["path"] for f in result["files"]}
    assert not any(p.endswith("auth.py") for p in paths)
    assert any(p.endswith("User.php") for p in paths)


@pytest.mark.asyncio
async def test_discover_respects_custom_includes(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _seed_code(tmp_path)
    result = await discover_code_inventory({"includes": ["src/**.py"]}, ctx)
    paths = {f["path"] for f in result["files"]}
    assert paths == {str(Path("src") / "auth.py")}


@pytest.mark.asyncio
async def test_discover_summary_counts_languages(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _seed_code(tmp_path)
    result = await discover_code_inventory({}, ctx)
    languages = result["summary"]["languages"]
    assert languages.get("python", 0) >= 1
    assert languages.get("php", 0) >= 1
    assert languages.get("typescript", 0) >= 1
    assert result["summary"]["total_symbols"] >= 4


@pytest.mark.asyncio
async def test_discover_truncates_above_max_files(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    for i in range(15):
        (src / f"m{i}.py").write_text(f"def f{i}(): pass\n")
    result = await discover_code_inventory({"max_files": 5}, ctx)
    assert result["truncated"] is True
    assert len(result["files"]) == 5


@pytest.mark.asyncio
async def test_discover_visibility_filter(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    (src / "mix.py").write_text(
        "def public_fn():\n    pass\ndef _internal_fn():\n    pass\ndef __private_fn():\n    pass\n"
    )

    public = await discover_code_inventory({"visibility": "public"}, ctx)
    public_names = {
        s["name"] for f in public["files"] for s in f["symbols"] if f["path"].endswith("mix.py")
    }
    assert "public_fn" in public_names
    assert "_internal_fn" not in public_names
    assert "__private_fn" not in public_names

    all_syms = await discover_code_inventory({"visibility": "all"}, ctx)
    all_names = {
        s["name"] for f in all_syms["files"] for s in f["symbols"] if f["path"].endswith("mix.py")
    }
    assert {"public_fn", "_internal_fn", "__private_fn"} <= all_names


@pytest.mark.asyncio
async def test_discover_rejects_bad_includes_type(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    with pytest.raises(DiscoverCodeInventoryError):
        await discover_code_inventory({"includes": "not-a-list"}, ctx)


@pytest.mark.asyncio
async def test_discover_rejects_bad_visibility(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    with pytest.raises(DiscoverCodeInventoryError):
        await discover_code_inventory({"visibility": "private"}, ctx)


@pytest.mark.asyncio
async def test_discover_empty_project_returns_empty(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await discover_code_inventory({}, ctx)
    assert result["ok"] is True
    assert result["files"] == []
    assert result["summary"]["total_symbols"] == 0


# ── MCP wiring smoke ────────────────────────────────────────────────────


def test_default_allowed_tools_includes_inventory() -> None:
    from spec_writer.agent.mcp_tools import default_allowed_tools

    assert "mcp__spec-writer__discover_code_inventory" in default_allowed_tools()


def test_local_registry_dispatch(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _seed_code(tmp_path)
    reg = ToolRegistry()
    register_dci(reg)
    assert reg.is_registered("discover_code_inventory")

    import asyncio

    result = asyncio.run(reg.call("discover_code_inventory", {}, ctx))
    assert result["ok"] is True
