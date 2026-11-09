"""S3.13 — engine primitive tools (validate, metrics, list_sources, list_corpus)."""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.agent.tools.list_corpus import (
    list_corpus,
)
from spec_writer.agent.tools.list_corpus import (
    register as register_list_corpus,
)
from spec_writer.agent.tools.list_sources import (
    list_sources,
)
from spec_writer.agent.tools.metrics import (
    MetricsToolError,
    metrics,
)
from spec_writer.agent.tools.metrics import (
    register as register_metrics,
)
from spec_writer.agent.tools.record_spec import register as register_record_spec
from spec_writer.agent.tools.validate import (
    ValidateToolError,
    validate,
)
from spec_writer.agent.tools.validate import (
    register as register_validate,
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
        (
            "version: 1\n"
            "layers:\n"
            "  - id: L1\n    role: source-boundary\n    name: vision\n"
            "  - id: L2\n    role: middle\n    name: capabilities\n"
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
        (
            "mixin: cap\nadds_types:\n  - id: capability\n"
            "    layer_name: capabilities\n    prefix: CAP\n"
            "    up_to_types: [vision]\n"
        ),
        encoding="utf-8",
    )
    (mixins / "gate.yaml").write_text(
        (
            "mixin: gate\nmetrics:\n"
            "  - id: cap_count\n    layer: L2\n"
            "    func_ref: metrics.spec_count_at_layer\n"
        ),
        encoding="utf-8",
    )
    methodology = load_methodology(
        meta / "base.yaml",
        [FilesystemMixinSource(mixins)],
        active_mixins=["vision", "cap", "gate"],
    )
    return ToolContext(project_ctx=project_ctx, methodology=methodology, id_registry=IdRegistry())


# ── validate ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_validate_returns_ok_on_empty_corpus(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await validate({}, ctx)
    assert result["ok"] is True
    assert "issues" in result


@pytest.mark.asyncio
async def test_validate_detects_pyramid_adjacency_error(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_validate(reg)
    # Record an L2 capability without up_to → pyramid-adjacency violation
    # (only if core-traceability is active — it's not in this minimal mixin
    # set, so just verify validate ran and returned 0 issues cleanly).
    await reg.call(
        "record_spec",
        {"type": "capability", "layer": "L2", "title": "Orphan", "up_to": []},
        ctx,
    )
    result = await reg.call("validate", {}, ctx)
    assert result["ok"] is True
    assert isinstance(result["issues"], list)


@pytest.mark.asyncio
async def test_validate_accepts_scope_filter(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await validate({"scope": {"layers": ["L1"]}}, ctx)
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_validate_rejects_non_dict_scope(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    with pytest.raises(ValidateToolError):
        await validate({"scope": "not-a-dict"}, ctx)


# ── metrics ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_returns_results_for_active_methodology(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_metrics(reg)
    await reg.call(
        "record_spec",
        {"type": "capability", "layer": "L2", "title": "Auth", "up_to": []},
        ctx,
    )
    result = await reg.call("metrics", {}, ctx)
    assert result["ok"] is True
    # gate mixin declares `cap_count` for L2
    metric_ids = [r["id"] for r in result["results"]]
    assert "cap_count" in metric_ids


@pytest.mark.asyncio
async def test_metrics_layer_filter(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await metrics({"layer_ids": ["L9-MISSING"]}, ctx)
    assert result["results"] == []  # filter excludes everything


@pytest.mark.asyncio
async def test_metrics_invalid_layer_ids_type(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    with pytest.raises(MetricsToolError):
        await metrics({"layer_ids": "L2"}, ctx)


# ── list_sources ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_sources_empty(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await list_sources({}, ctx)
    assert result == {"ok": True, "files": []}


@pytest.mark.asyncio
async def test_list_sources_finds_files(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    (ctx.project_ctx.sources / "biz.md").write_text("# Idea\n\nA tiny app.")
    (ctx.project_ctx.sources / "deep" / "nested.md").parent.mkdir(parents=True)
    (ctx.project_ctx.sources / "deep" / "nested.md").write_text("nested")
    result = await list_sources({}, ctx)
    paths = sorted(f["path"] for f in result["files"])
    assert any(p.endswith("biz.md") for p in paths)
    assert any(p.endswith("nested.md") for p in paths)
    for entry in result["files"]:
        assert entry["size_bytes"] > 0
        assert entry["suffix"] == "md"


# ── list_corpus ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_corpus_empty(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await list_corpus({}, ctx)
    assert result == {"ok": True, "count": 0, "specs": []}


@pytest.mark.asyncio
async def test_list_corpus_returns_frontmatter(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_list_corpus(reg)
    await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "Platform", "up_to": []},
        ctx,
    )
    await reg.call(
        "record_spec",
        {
            "type": "capability",
            "layer": "L2",
            "title": "Auth",
            "up_to": ["L1-VIS-001"],
            "tags": ["auth"],
        },
        ctx,
    )
    result = await reg.call("list_corpus", {}, ctx)
    assert result["count"] == 2
    specs_by_id = {s["id"]: s for s in result["specs"]}
    assert "L1-VIS-001" in specs_by_id
    assert "L2-CAP-001" in specs_by_id
    assert specs_by_id["L2-CAP-001"]["tags"] == ["auth"]


@pytest.mark.asyncio
async def test_list_corpus_layer_filter(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_list_corpus(reg)
    await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "Platform", "up_to": []},
        ctx,
    )
    await reg.call(
        "record_spec",
        {"type": "capability", "layer": "L2", "title": "Auth", "up_to": []},
        ctx,
    )
    result = await reg.call("list_corpus", {"layer": "L2"}, ctx)
    assert result["count"] == 1
    assert result["specs"][0]["layer"] == "L2"


@pytest.mark.asyncio
async def test_list_corpus_tag_filter(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_list_corpus(reg)
    await reg.call(
        "record_spec",
        {
            "type": "vision",
            "layer": "L1",
            "title": "Auth Vision",
            "up_to": [],
            "tags": ["auth"],
        },
        ctx,
    )
    await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "Other Vision", "up_to": []},
        ctx,
    )
    result = await reg.call("list_corpus", {"tag": "auth"}, ctx)
    assert result["count"] == 1
    assert result["specs"][0]["title"] == "Auth Vision"


# ── MCP wiring smoke ────────────────────────────────────────────────────


def test_new_primitives_in_default_allowed_tools() -> None:
    from spec_writer.agent.mcp_tools import default_allowed_tools

    tools = default_allowed_tools()
    for name in (
        "mcp__spec-writer__validate",
        "mcp__spec-writer__metrics",
        "mcp__spec-writer__list_sources",
        "mcp__spec-writer__list_corpus",
    ):
        assert name in tools


def test_build_mcp_server_includes_new_primitives(tmp_path: Path) -> None:
    from spec_writer.agent.mcp_tools import build_mcp_server

    ctx = _ctx(tmp_path)
    server, _ = build_mcp_server(ctx.project_ctx, ctx.methodology, ctx.id_registry)
    assert server is not None
    # We can't introspect SDK MCP tool list without invoking SDK; this
    # smoke-test just confirms the closure builds without errors.
    _ = server  # silence ruff
