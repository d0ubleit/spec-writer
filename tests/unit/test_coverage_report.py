"""S4.5 — coverage_report MCP tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_writer.agent.tool_registry import ToolContext
from spec_writer.agent.tools.coverage_report import coverage_report
from spec_writer.engine import coverage_ledger
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


def _seed_src(root: Path, files: dict[str, str]) -> None:
    src = root / "src"
    src.mkdir(exist_ok=True)
    for rel, content in files.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


@pytest.mark.asyncio
async def test_coverage_report_empty_inventory(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await coverage_report({}, ctx)
    assert result["ok"] is True
    assert result["total_files"] == 0
    assert result["ratio_visited"] == 0.0
    assert result["ratio_covered"] == 0.0
    assert result["uncovered"] == []


@pytest.mark.asyncio
async def test_coverage_report_lists_uncovered(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _seed_src(tmp_path, {"auth.py": "def login(): ...\n", "db.py": "def query(): ...\n"})
    result = await coverage_report({}, ctx)
    paths = set(result["uncovered"])
    assert paths == {"src/auth.py", "src/db.py"}
    assert result["ratio_covered"] == 0.0


@pytest.mark.asyncio
async def test_coverage_report_reflects_visited_and_covered(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _seed_src(tmp_path, {"auth.py": "def login(): ...\n", "db.py": "def query(): ...\n"})

    ledger = coverage_ledger.CoverageLedger()
    ledger.record_read("src/auth.py", 1, 5)
    ledger.record_implementation(["src/db.py:1-10"])
    coverage_ledger.save(ledger, ctx.project_ctx.meta)

    result = await coverage_report({}, ctx)
    assert result["visited"] == ["src/auth.py"]
    assert result["covered"] == ["src/db.py"]
    assert result["uncovered"] == ["src/auth.py"]
    assert result["ratio_visited"] == pytest.approx(0.5)
    assert result["ratio_covered"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_coverage_report_separates_outside_inventory(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _seed_src(tmp_path, {"auth.py": "def login(): ...\n"})

    ledger = coverage_ledger.CoverageLedger()
    ledger.record_read("legacy/oauth.php", 1, 10)
    ledger.record_implementation(["deprecated/util.py:1-2"])
    coverage_ledger.save(ledger, ctx.project_ctx.meta)

    result = await coverage_report({}, ctx)
    assert result["visited_outside_inventory"] == ["legacy/oauth.php"]
    assert result["covered_outside_inventory"] == ["deprecated/util.py"]


@pytest.mark.asyncio
async def test_coverage_report_respects_includes(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _seed_src(
        tmp_path,
        {"auth.py": "def login(): ...\n", "extra/util.py": "def x(): ...\n"},
    )
    result = await coverage_report({"includes": ["src/auth.py"]}, ctx)
    assert result["total_files"] == 1
    assert "src/auth.py" in result["uncovered"]


def test_default_allowed_tools_includes_coverage_report() -> None:
    from spec_writer.agent.mcp_tools import default_allowed_tools

    assert "mcp__spec-writer__coverage_report" in default_allowed_tools()


def test_code_coverage_gate_mixin_yaml_exists() -> None:
    """Mixin file ships with the package."""
    from importlib import resources

    base = resources.files("spec_writer.methodology._builtin.mixins")
    yaml_path = Path(str(base.joinpath("code-coverage-gate.yaml")))
    assert yaml_path.exists()
    text = yaml_path.read_text(encoding="utf-8")
    assert "before_layer_complete" in text
    assert "target: 0.85" in text
    assert "warn_below: 0.60" in text
