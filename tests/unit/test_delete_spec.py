"""S4.8 — delete_spec MCP tool + IdRegistry.release."""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.agent.tools.delete_spec import DeleteSpecError, delete_spec
from spec_writer.agent.tools.delete_spec import register as register_delete_spec
from spec_writer.agent.tools.record_spec import register as register_record_spec
from spec_writer.engine.id_registry import IdRegistry, MalformedIdError
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


# ── IdRegistry.release ──────────────────────────────────────────────────


def test_release_taken_id_returns_true() -> None:
    reg = IdRegistry()
    reg.register("L4-API-001")
    assert reg.release("L4-API-001") is True
    assert not reg.is_taken("L4-API-001")


def test_release_unknown_id_returns_false() -> None:
    reg = IdRegistry()
    assert reg.release("L4-API-999") is False


def test_release_then_allocate_reuses_id() -> None:
    reg = IdRegistry()
    reg.register("L4-API-001")
    reg.release("L4-API-001")
    # allocate should pick 001 again — first free slot
    new_id = reg.allocate("L4", "API")
    assert new_id == "L4-API-001"


def test_release_malformed_id_returns_false_no_raise() -> None:
    reg = IdRegistry()
    # release should be tolerant of malformed ids (corpus garbage)
    assert reg.release("not-an-id") is False
    # But register still rejects malformed ids
    with pytest.raises(MalformedIdError):
        reg.register("bogus")


# ── delete_spec tool ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_spec_removes_file_and_releases_id(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_delete_spec(reg)

    created = await reg.call(
        "record_spec",
        {"type": "api-contract", "layer": "L4", "title": "Login", "up_to": []},
        ctx,
    )
    spec_id = created["spec_id"]
    spec_path = Path(created["path"])
    assert spec_path.exists()

    result = await reg.call("delete_spec", {"spec_id": spec_id}, ctx)
    assert result["ok"] is True
    assert result["spec_id"] == spec_id
    assert result["released_id"] is True
    assert not spec_path.exists()
    assert not ctx.id_registry.is_taken(spec_id)


@pytest.mark.asyncio
async def test_delete_spec_unknown_id_returns_error(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    result = await delete_spec({"spec_id": "L9-MISSING-999"}, ctx)
    assert result["ok"] is False
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_delete_spec_missing_id_raises(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    with pytest.raises(DeleteSpecError):
        await delete_spec({}, ctx)


@pytest.mark.asyncio
async def test_delete_then_record_reuses_id(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_delete_spec(reg)

    first = await reg.call(
        "record_spec",
        {"type": "api-contract", "layer": "L4", "title": "Login", "up_to": []},
        ctx,
    )
    spec_id = first["spec_id"]
    await reg.call("delete_spec", {"spec_id": spec_id}, ctx)

    second = await reg.call(
        "record_spec",
        {"type": "api-contract", "layer": "L4", "title": "Login Again", "up_to": []},
        ctx,
    )
    # Same first-gap allocation → same id reused
    assert second["spec_id"] == spec_id


# ── MCP wiring smoke ────────────────────────────────────────────────────


def test_default_allowed_tools_includes_delete_spec() -> None:
    from spec_writer.agent.mcp_tools import default_allowed_tools

    assert "mcp__spec-writer__delete_spec" in default_allowed_tools()


# ── ChatAgent prompt covers orphan workflow ─────────────────────────────


def test_chat_system_prompt_mentions_orphan_workflow(tmp_path: Path) -> None:
    from spec_writer.agent.chat_agent import build_chat_system_prompt
    from spec_writer.engine.project_context import ProjectContext

    (tmp_path / "spec-editor").mkdir()
    for sub in ("sources-raw", "sources", "specs", "_meta"):
        (tmp_path / "spec-editor" / sub).mkdir()
    project = ProjectContext.from_cwd(tmp_path)

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

    prompt = build_chat_system_prompt(methodology, project)
    assert "orphan" in prompt
    assert "delete_spec" in prompt
    # ask_user confirmation expectation present
    assert "ask_user" in prompt
