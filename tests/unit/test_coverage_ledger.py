"""S4.4 — coverage ledger (passive, JSON-backed)."""

from __future__ import annotations

from pathlib import Path

from spec_writer.engine import coverage_ledger


def test_record_read_adds_to_visited(tmp_path: Path) -> None:
    ledger = coverage_ledger.CoverageLedger()
    ledger.record_read("src/auth.py", 10, 42)
    assert ledger.is_visited("src/auth.py")
    assert [10, 42] in ledger.visited["src/auth.py"]


def test_record_read_normalises_paths() -> None:
    ledger = coverage_ledger.CoverageLedger()
    ledger.record_read("./src/auth.py")
    ledger.record_read("src\\auth.py")  # windows-style
    assert ledger.is_visited("src/auth.py")
    # Both should land under the same key
    assert len(ledger.visited["src/auth.py"]) == 2


def test_record_read_ignores_blank() -> None:
    ledger = coverage_ledger.CoverageLedger()
    ledger.record_read("")
    assert ledger.visited == {}


def test_record_grep_default_line_zero() -> None:
    ledger = coverage_ledger.CoverageLedger()
    ledger.record_grep("src/x.py")
    assert ledger.is_visited("src/x.py")
    assert ledger.visited["src/x.py"] == [[0, 0]]


def test_record_implementation_marks_covered() -> None:
    ledger = coverage_ledger.CoverageLedger()
    ledger.record_implementation(["src/auth.py:10-42", "src/other.py"])
    assert ledger.is_covered("src/auth.py")
    assert ledger.is_covered("src/other.py")


def test_record_implementation_skips_blank() -> None:
    ledger = coverage_ledger.CoverageLedger()
    ledger.record_implementation(["", "  "])
    assert ledger.covered == set()


def test_visited_paths_and_covered_paths_returns_sets() -> None:
    ledger = coverage_ledger.CoverageLedger()
    ledger.record_read("src/a.py")
    ledger.record_implementation(["src/a.py:1-2", "src/b.py:3-4"])
    assert ledger.visited_paths() == {"src/a.py"}
    assert ledger.covered_paths() == {"src/a.py", "src/b.py"}


def test_uncovered_against_returns_subset_preserving_order() -> None:
    ledger = coverage_ledger.CoverageLedger()
    ledger.record_implementation(["src/a.py:1-2"])
    out = ledger.uncovered_against(["src/a.py", "src/b.py", "src/c.py", "src/b.py"])
    assert out == ["src/b.py", "src/c.py"]  # a covered, b duplicate dropped


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    meta = tmp_path / "_meta"
    meta.mkdir()
    ledger = coverage_ledger.CoverageLedger()
    ledger.record_read("src/auth.py", 1, 5)
    ledger.record_implementation(["src/auth.py", "src/db.py"])
    coverage_ledger.save(ledger, meta)

    loaded = coverage_ledger.load(meta)
    assert loaded.is_visited("src/auth.py")
    assert loaded.covered_paths() == {"src/auth.py", "src/db.py"}
    assert loaded.visited["src/auth.py"] == [[1, 5]]


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    meta = tmp_path / "_meta"
    meta.mkdir()
    ledger = coverage_ledger.load(meta)
    assert ledger.visited == {}
    assert ledger.covered == set()


def test_load_corrupt_file_returns_empty(tmp_path: Path) -> None:
    meta = tmp_path / "_meta"
    meta.mkdir()
    (meta / "code_visited_ledger.json").write_text("{not valid json", encoding="utf-8")
    ledger = coverage_ledger.load(meta)
    assert ledger.visited == {}
    assert ledger.covered == set()


def test_load_skips_malformed_entries(tmp_path: Path) -> None:
    meta = tmp_path / "_meta"
    meta.mkdir()
    (meta / "code_visited_ledger.json").write_text(
        '{"visited": {"good.py": [[1, 2]], "broken.py": [["x"]]}, "covered": ["good.py", 42]}',
        encoding="utf-8",
    )
    ledger = coverage_ledger.load(meta)
    assert ledger.is_visited("good.py")
    assert not ledger.is_visited("broken.py")
    assert ledger.covered == {"good.py"}


def test_save_uses_atomic_write(tmp_path: Path) -> None:
    """Repeated saves must not leave the file half-written."""
    meta = tmp_path / "_meta"
    meta.mkdir()
    ledger = coverage_ledger.CoverageLedger()
    ledger.record_read("a.py")
    coverage_ledger.save(ledger, meta)
    ledger.record_read("b.py")
    coverage_ledger.save(ledger, meta)
    loaded = coverage_ledger.load(meta)
    assert loaded.visited_paths() == {"a.py", "b.py"}


# ── record_spec → covered integration ────────────────────────────────────


def test_record_spec_updates_covered(tmp_path: Path) -> None:
    from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
    from spec_writer.agent.tools.record_spec import register as register_record_spec
    from spec_writer.engine.id_registry import IdRegistry
    from spec_writer.engine.project_context import ProjectContext
    from spec_writer.methodology.loader import load_methodology
    from spec_writer.methodology.sources import FilesystemMixinSource

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
    ctx = ToolContext(project_ctx=project_ctx, methodology=methodology, id_registry=IdRegistry())
    reg = ToolRegistry()
    register_record_spec(reg)

    import asyncio

    asyncio.run(
        reg.call(
            "record_spec",
            {
                "type": "api-contract",
                "layer": "L4",
                "title": "Login Endpoint",
                "up_to": [],
                "implemented_by": ["src/auth/login.php:42-58"],
            },
            ctx,
        )
    )

    loaded = coverage_ledger.load(project_ctx.meta)
    assert loaded.is_covered("src/auth/login.php")
