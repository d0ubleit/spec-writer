"""S1.10: tool_registry + record_spec/patch_spec end-to-end."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spec_writer.agent.tool_registry import (
    ToolContext,
    ToolNotRegisteredError,
    ToolRegistry,
)
from spec_writer.agent.tools.patch_spec import PatchSpecError
from spec_writer.agent.tools.patch_spec import register as register_patch_spec
from spec_writer.agent.tools.record_spec import RecordSpecError
from spec_writer.agent.tools.record_spec import register as register_record_spec
from spec_writer.engine.id_registry import IdRegistry
from spec_writer.engine.project_context import ProjectContext
from spec_writer.io.spec_io import find_spec_by_id, read_spec
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
        "mixin: cap\nadds_types:\n  - id: capability\n    layer_name: capabilities\n    prefix: CAP\n",
        encoding="utf-8",
    )
    rm = load_methodology(
        meta / "base.yaml",
        [FilesystemMixinSource(mixins)],
        active_mixins=["vision", "cap"],
    )
    return ToolContext(project_ctx=project_ctx, methodology=rm, id_registry=IdRegistry())


# ── ToolRegistry ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_registry_register_and_call(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()

    async def echo(args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return {"echo": args}

    reg.register("echo", echo)
    assert reg.is_registered("echo")
    result = await reg.call("echo", {"k": 1}, ctx)
    assert result == {"echo": {"k": 1}}


@pytest.mark.asyncio
async def test_registry_unknown_tool_raises(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    with pytest.raises(ToolNotRegisteredError):
        await reg.call("ghost", {}, ctx)


def test_registry_duplicate_registration_raises() -> None:
    reg = ToolRegistry()

    async def h(_args: dict[str, Any], _ctx: ToolContext) -> dict[str, Any]:
        return {}

    reg.register("h", h)
    with pytest.raises(ValueError):
        reg.register("h", h)


def test_registry_names_sorted() -> None:
    reg = ToolRegistry()

    async def f(_a: dict[str, Any], _c: ToolContext) -> dict[str, Any]:
        return {}

    reg.register("b", f)
    reg.register("a", f)
    assert reg.names() == ["a", "b"]


# ── record_spec ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_spec_allocates_id_and_writes(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    result = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "Platform Vision", "up_to": []},
        ctx,
    )
    assert result["ok"] is True
    assert result["spec_id"] == "L1-VIS-001"
    assert Path(result["path"]).exists()
    reloaded = read_spec(Path(result["path"]))
    assert reloaded.title == "Platform Vision"


@pytest.mark.asyncio
async def test_record_spec_with_explicit_id(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    result = await reg.call(
        "record_spec",
        {
            "id": "L1-VIS-042",
            "type": "vision",
            "layer": "L1",
            "title": "Custom Id",
            "up_to": [],
        },
        ctx,
    )
    assert result["spec_id"] == "L1-VIS-042"


@pytest.mark.asyncio
async def test_record_spec_collision_rejected(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    await reg.call(
        "record_spec",
        {"id": "L1-VIS-001", "type": "vision", "layer": "L1", "title": "A", "up_to": []},
        ctx,
    )
    with pytest.raises(RecordSpecError, match="already taken"):
        await reg.call(
            "record_spec",
            {"id": "L1-VIS-001", "type": "vision", "layer": "L1", "title": "B", "up_to": []},
            ctx,
        )


@pytest.mark.asyncio
async def test_record_spec_unknown_type_rejected(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    with pytest.raises(RecordSpecError, match="unknown spec type"):
        await reg.call(
            "record_spec",
            {"type": "bogus", "layer": "L1", "title": "X", "up_to": []},
            ctx,
        )


@pytest.mark.asyncio
async def test_record_spec_missing_required_field(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    with pytest.raises(RecordSpecError, match="string fields"):
        await reg.call("record_spec", {"type": "vision", "layer": "L1"}, ctx)


@pytest.mark.asyncio
async def test_record_spec_passes_extras(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    result = await reg.call(
        "record_spec",
        {
            "type": "vision",
            "layer": "L1",
            "title": "X",
            "up_to": [],
            "tags": ["alpha", "beta"],
        },
        ctx,
    )
    reloaded = read_spec(Path(result["path"]))
    assert reloaded.tags == ["alpha", "beta"]


# ── patch_spec ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_spec_updates_fields(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    create = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "Original", "up_to": []},
        ctx,
    )

    patch = await reg.call(
        "patch_spec",
        {"spec_id": create["spec_id"], "changes": {"tags": ["new-tag"]}},
        ctx,
    )
    assert patch["ok"] is True
    assert patch["applied"] == ["tags"]

    reloaded = read_spec(Path(patch["path"]))
    assert reloaded.tags == ["new-tag"]


@pytest.mark.asyncio
async def test_patch_spec_immutable_fields_silently_ignored(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    create = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "X", "up_to": []},
        ctx,
    )
    patch = await reg.call(
        "patch_spec",
        {
            "spec_id": create["spec_id"],
            "changes": {"id": "L1-VIS-999", "type": "capability", "layer": "L2", "tags": ["x"]},
        },
        ctx,
    )
    assert patch["applied"] == ["tags"]
    reloaded = read_spec(Path(patch["path"]))
    assert reloaded.id == create["spec_id"]
    assert reloaded.type == "vision"
    assert reloaded.layer == "L1"


@pytest.mark.asyncio
async def test_patch_spec_not_found(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_patch_spec(reg)
    with pytest.raises(PatchSpecError, match="not found"):
        await reg.call("patch_spec", {"spec_id": "L1-VIS-999", "changes": {}}, ctx)


@pytest.mark.asyncio
async def test_patch_spec_body_update(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    create = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "X", "up_to": [], "body": "old"},
        ctx,
    )
    patch = await reg.call(
        "patch_spec",
        {"spec_id": create["spec_id"], "changes": {"body": "new content"}},
        ctx,
    )
    reloaded = read_spec(Path(patch["path"]))
    assert "new content" in reloaded.body


@pytest.mark.asyncio
async def test_patch_spec_renames_when_title_changes(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    create = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "Old Title", "up_to": []},
        ctx,
    )
    old_path = Path(create["path"])

    patch = await reg.call(
        "patch_spec",
        {"spec_id": create["spec_id"], "changes": {"title": "New Title"}},
        ctx,
    )
    new_path = Path(patch["path"])
    assert new_path != old_path
    assert new_path.exists()
    assert not old_path.exists()
    assert find_spec_by_id(ctx.project_ctx.specs, create["spec_id"]) == new_path


@pytest.mark.asyncio
async def test_patch_spec_requires_changes_dict(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_patch_spec(reg)
    with pytest.raises(PatchSpecError, match="dict 'changes'"):
        await reg.call("patch_spec", {"spec_id": "L1-VIS-001", "changes": "not a dict"}, ctx)


# ── patch_spec symmetrises edges ───────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_spec_up_to_symmetrises_parent_down_to(tmp_path: Path) -> None:
    """a known limitation: patching `up_to` adds the reciprocal `down_to` on the parent."""
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    parent = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "Parent", "up_to": []},
        ctx,
    )
    child = await reg.call(
        "record_spec",
        {"type": "capability", "layer": "L2", "title": "Child", "up_to": []},
        ctx,
    )

    await reg.call(
        "patch_spec",
        {"spec_id": child["spec_id"], "changes": {"up_to": [parent["spec_id"]]}},
        ctx,
    )

    reloaded_parent = read_spec(Path(parent["path"]))
    assert child["spec_id"] in reloaded_parent.down_to


@pytest.mark.asyncio
async def test_patch_spec_unchanged_up_to_still_repairs_drift(tmp_path: Path) -> None:
    """a known limitation: same-value patch still triggers symmetrisation (repair pass)."""
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    parent = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "Parent", "up_to": []},
        ctx,
    )
    child = await reg.call(
        "record_spec",
        {
            "type": "capability",
            "layer": "L2",
            "title": "Child",
            "up_to": [parent["spec_id"]],
        },
        ctx,
    )

    # Manually break the symmetry on disk to simulate drift.
    parent_spec = read_spec(Path(parent["path"]))
    parent_spec.down_to = []
    from spec_writer.io.spec_io import write_spec as _write_spec

    _write_spec(parent_spec, ctx.methodology, ctx.project_ctx)
    assert read_spec(Path(parent["path"])).down_to == []  # confirmed drift

    # Same-value re-patch — should still repair the parent.
    await reg.call(
        "patch_spec",
        {"spec_id": child["spec_id"], "changes": {"up_to": [parent["spec_id"]]}},
        ctx,
    )

    reloaded_parent = read_spec(Path(parent["path"]))
    assert child["spec_id"] in reloaded_parent.down_to


# ── patch_spec(field=None) removes extras keys ────────────────────────


@pytest.mark.asyncio
async def test_patch_spec_none_removes_extras_key(tmp_path: Path) -> None:
    """a known limitation: passing None for a methodology-extra deletes the key."""
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    create = await reg.call(
        "record_spec",
        {
            "type": "vision",
            "layer": "L1",
            "title": "X",
            "up_to": [],
            "track2_addendum": "some-temporary-value",
        },
        ctx,
    )
    # Confirm extra is on the spec.
    extras_before = read_spec(Path(create["path"])).model_extra or {}
    assert extras_before.get("track2_addendum") == "some-temporary-value"

    await reg.call(
        "patch_spec",
        {"spec_id": create["spec_id"], "changes": {"track2_addendum": None}},
        ctx,
    )

    reloaded = read_spec(Path(create["path"]))
    # Key fully removed, not stored as None.
    assert "track2_addendum" not in (reloaded.model_extra or {})
    # On-disk YAML doesn't contain `track2_addendum: null` either.
    rendered = Path(create["path"]).read_text(encoding="utf-8")
    assert "track2_addendum" not in rendered


@pytest.mark.asyncio
async def test_patch_spec_none_keeps_core_nullable_field(tmp_path: Path) -> None:
    """a known limitation corollary: None on a core nullable field (status) still works."""
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    create = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "X", "up_to": [], "status": "draft"},
        ctx,
    )

    await reg.call(
        "patch_spec",
        {"spec_id": create["spec_id"], "changes": {"status": None}},
        ctx,
    )
    reloaded = read_spec(Path(create["path"]))
    assert reloaded.status is None
