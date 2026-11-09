"""S2.5 — query_traceability / resolve_scope / resolve_source_chain / ask_user."""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.agent.tools.ask_user import (
    AskUserError,
    set_answerer,
)
from spec_writer.agent.tools.ask_user import (
    register as register_ask_user,
)
from spec_writer.agent.tools.query_traceability import (
    QueryTraceabilityError,
)
from spec_writer.agent.tools.query_traceability import (
    register as register_query_traceability,
)
from spec_writer.agent.tools.record_spec import register as register_record_spec
from spec_writer.agent.tools.resolve_scope import (
    ResolveScopeError,
)
from spec_writer.agent.tools.resolve_scope import (
    register as register_resolve_scope,
)
from spec_writer.agent.tools.resolve_source_chain import (
    ResolveSourceChainError,
)
from spec_writer.agent.tools.resolve_source_chain import (
    register as register_resolve_source_chain,
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
        (
            "mixin: cap\nadds_types:\n  - id: capability\n"
            "    layer_name: capabilities\n    prefix: CAP\n"
            "    up_to_types: [vision]\n"
        ),
        encoding="utf-8",
    )
    (mixins / "bhv.yaml").write_text(
        (
            "mixin: bhv\nadds_types:\n  - id: behavior\n"
            "    layer_name: behaviors\n    prefix: BHV\n"
            "    up_to_types: [capability]\n"
        ),
        encoding="utf-8",
    )
    rm = load_methodology(
        meta / "base.yaml",
        [FilesystemMixinSource(mixins)],
        active_mixins=["vision", "cap", "bhv"],
    )
    return ToolContext(project_ctx=project_ctx, methodology=rm, id_registry=IdRegistry())


async def _seed_three_layers(reg: ToolRegistry, ctx: ToolContext) -> dict[str, str]:
    """Write three connected specs: vision → capability → behavior."""
    v = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "Platform Vision", "up_to": []},
        ctx,
    )
    c = await reg.call(
        "record_spec",
        {
            "type": "capability",
            "layer": "L2",
            "title": "Auth Capability",
            "up_to": [v["spec_id"]],
        },
        ctx,
    )
    b = await reg.call(
        "record_spec",
        {
            "type": "behavior",
            "layer": "L3",
            "title": "Sign In Behavior",
            "up_to": [c["spec_id"]],
            "tags": ["auth"],
        },
        ctx,
    )
    return {"vision": v["spec_id"], "capability": c["spec_id"], "behavior": b["spec_id"]}


# ── query_traceability ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_traceability_walks_up(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_query_traceability(reg)
    ids = await _seed_three_layers(reg, ctx)

    result = await reg.call(
        "query_traceability",
        {"spec_id": ids["behavior"], "direction": "up", "depth": 1},
        ctx,
    )
    assert result["ok"] is True
    assert len(result["neighbours"]) == 1
    assert result["neighbours"][0]["id"] == ids["capability"]
    assert result["neighbours"][0]["kind"] == "up_to"


@pytest.mark.asyncio
async def test_query_traceability_depth_two_reaches_source(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_query_traceability(reg)
    ids = await _seed_three_layers(reg, ctx)

    result = await reg.call(
        "query_traceability",
        {"spec_id": ids["behavior"], "direction": "up", "depth": 2},
        ctx,
    )
    neighbour_ids = {n["id"] for n in result["neighbours"]}
    assert ids["capability"] in neighbour_ids
    assert ids["vision"] in neighbour_ids


@pytest.mark.asyncio
async def test_query_traceability_down(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_query_traceability(reg)
    ids = await _seed_three_layers(reg, ctx)

    # children are inferred via symmetrize, but freshly-written specs have empty
    # down_to until validators run. So we patch the parent spec to add a down_to.
    from spec_writer.io.spec_io import find_spec_by_id, read_spec, write_spec

    cap_path = find_spec_by_id(ctx.project_ctx.specs, ids["capability"])
    assert cap_path is not None
    spec = read_spec(cap_path)
    spec.down_to = [ids["behavior"]]
    write_spec(spec, ctx.methodology, ctx.project_ctx)

    result = await reg.call(
        "query_traceability",
        {"spec_id": ids["capability"], "direction": "down", "depth": 1},
        ctx,
    )
    assert result["ok"] is True
    assert any(n["id"] == ids["behavior"] for n in result["neighbours"])


@pytest.mark.asyncio
async def test_query_traceability_unknown_spec(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_query_traceability(reg)
    result = await reg.call(
        "query_traceability",
        {"spec_id": "L9-MISSING-001", "direction": "up"},
        ctx,
    )
    assert result["ok"] is False
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_query_traceability_bad_direction(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_query_traceability(reg)
    with pytest.raises(QueryTraceabilityError, match="direction"):
        await reg.call(
            "query_traceability",
            {"spec_id": "L1-VIS-001", "direction": "sideways"},
            ctx,
        )


@pytest.mark.asyncio
async def test_query_traceability_missing_spec_id(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_query_traceability(reg)
    with pytest.raises(QueryTraceabilityError, match="spec_id"):
        await reg.call("query_traceability", {}, ctx)


# ── resolve_scope ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_scope_layers(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_resolve_scope(reg)
    result = await reg.call("resolve_scope", {"text": "capabilities on L2"}, ctx)
    assert result["ok"] is True
    scope = result["scope"]
    assert scope.get("layers") == ["L2"]
    assert scope.get("types") == ["capability"]


@pytest.mark.asyncio
async def test_resolve_scope_tags_and_ids(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_resolve_scope(reg)
    result = await reg.call(
        "resolve_scope",
        {"text": "#auth L1-VIS-001 L1-VIS-002 behaviors"},
        ctx,
    )
    scope = result["scope"]
    assert scope["tags"] == ["auth"]
    assert scope["spec_ids"] == ["L1-VIS-001", "L1-VIS-002"]
    assert scope["types"] == ["behavior"]


@pytest.mark.asyncio
async def test_resolve_scope_from_finding(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_resolve_scope(reg)
    result = await reg.call(
        "resolve_scope",
        {"text": "from:FIN-12 from:FIN-13"},
        ctx,
    )
    assert result["scope"]["from_finding"] == ["FIN-12", "FIN-13"]


@pytest.mark.asyncio
async def test_resolve_scope_residual(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_resolve_scope(reg)
    result = await reg.call("resolve_scope", {"text": "wibble wobble L1"}, ctx)
    assert result["scope"].get("layers") == ["L1"]
    assert sorted(result["residual"]) == ["wibble", "wobble"]


@pytest.mark.asyncio
async def test_resolve_scope_empty(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_resolve_scope(reg)
    result = await reg.call("resolve_scope", {"text": ""}, ctx)
    assert result["ok"] is True
    assert result["scope"] == {}
    assert result["residual"] == []


@pytest.mark.asyncio
async def test_resolve_scope_requires_text(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_resolve_scope(reg)
    with pytest.raises(ResolveScopeError):
        await reg.call("resolve_scope", {}, ctx)


# ── resolve_source_chain ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_source_chain_reaches_source_boundary(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_resolve_source_chain(reg)
    ids = await _seed_three_layers(reg, ctx)

    result = await reg.call(
        "resolve_source_chain",
        {"spec_id": ids["behavior"]},
        ctx,
    )
    assert result["ok"] is True
    chain_ids = [c["id"] for c in result["chain"]]
    assert ids["behavior"] in chain_ids
    assert ids["capability"] in chain_ids
    assert ids["vision"] in chain_ids
    assert result["sources"][0]["id"] == ids["vision"]
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_resolve_source_chain_source_spec_self(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_resolve_source_chain(reg)
    ids = await _seed_three_layers(reg, ctx)

    result = await reg.call("resolve_source_chain", {"spec_id": ids["vision"]}, ctx)
    assert result["ok"] is True
    # When the start spec is itself a source-boundary, it is its own source.
    assert result["chain"][0]["id"] == ids["vision"]
    assert any(s["id"] == ids["vision"] for s in result["sources"])


@pytest.mark.asyncio
async def test_resolve_source_chain_unknown(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_resolve_source_chain(reg)
    result = await reg.call("resolve_source_chain", {"spec_id": "L9-X-999"}, ctx)
    assert result["ok"] is False


@pytest.mark.asyncio
async def test_resolve_source_chain_bad_depth(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_resolve_source_chain(reg)
    with pytest.raises(ResolveSourceChainError, match="max_depth"):
        await reg.call(
            "resolve_source_chain",
            {"spec_id": "L1-VIS-001", "max_depth": "deep"},
            ctx,
        )


# ── ask_user ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ask_user_batch_mode_errors(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_ask_user(reg)
    set_answerer(None)
    result = await reg.call("ask_user", {"question": "Need clarification?"}, ctx)
    assert result["ok"] is False
    assert "dialog/chat mode" in result["error"]


@pytest.mark.asyncio
async def test_ask_user_with_sync_answerer(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_ask_user(reg)
    try:
        set_answerer(lambda _q: "yes please")
        result = await reg.call("ask_user", {"question": "Continue?"}, ctx)
        assert result["ok"] is True
        assert result["answer"] == "yes please"
    finally:
        set_answerer(None)


@pytest.mark.asyncio
async def test_ask_user_with_async_answerer(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_ask_user(reg)

    async def answer(_q: str) -> str:
        return "async answer"

    try:
        set_answerer(answer)
        result = await reg.call("ask_user", {"question": "Q?"}, ctx)
        assert result["ok"] is True
        assert result["answer"] == "async answer"
    finally:
        set_answerer(None)


@pytest.mark.asyncio
async def test_ask_user_requires_question(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_ask_user(reg)
    set_answerer(None)
    with pytest.raises(AskUserError):
        await reg.call("ask_user", {"question": ""}, ctx)


# ── mcp_tools wiring smoke ───────────────────────────────────────────────


def test_mcp_default_allowed_tools_includes_new_tools() -> None:
    from spec_writer.agent.mcp_tools import default_allowed_tools

    names = default_allowed_tools()
    assert "mcp__spec-writer__record_spec" in names
    assert "mcp__spec-writer__query_traceability" in names
    assert "mcp__spec-writer__resolve_scope" in names
    assert "mcp__spec-writer__resolve_source_chain" in names
    assert "mcp__spec-writer__ask_user" in names


def test_mcp_build_server_does_not_raise(tmp_path: Path) -> None:
    from spec_writer.agent.mcp_tools import build_mcp_server

    ctx = _ctx(tmp_path)
    server, recorded_ids = build_mcp_server(ctx.project_ctx, ctx.methodology, ctx.id_registry)
    assert server is not None
    assert recorded_ids == []
