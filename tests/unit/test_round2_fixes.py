"""Tests for the round-2 engine fixes (symmetrisation, key validation, scoped validate).

Covers:
- (A) patch_spec revalidates Pydantic-typed fields
- (B) record_spec auto-symmetrises on initial write
- (D) patch_spec removes stale parent down_to on re-anchor
- (E) Cross-spec validators read full corpus
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.agent.tools.patch_spec import (
    PatchSpecError,
)
from spec_writer.agent.tools.patch_spec import (
    register as register_patch_spec,
)
from spec_writer.agent.tools.record_spec import register as register_record_spec
from spec_writer.engine.id_registry import IdRegistry
from spec_writer.engine.project_context import ProjectContext
from spec_writer.io.spec_io import read_spec
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.sources import FilesystemMixinSource
from spec_writer.methodology.validators import ValidatorContext
from spec_writer.methodology.validators.mixin_validators import (
    check_rbac_roles_resolved,
)
from spec_writer.types import RelatesToEntry, Spec

# ── shared fixture ───────────────────────────────────────────────────────


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
        "mixin: vision\nadds_types:\n  - id: vision\n"
        "    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins / "cap.yaml").write_text(
        "mixin: cap\nadds_types:\n  - id: capability\n"
        "    layer_name: capabilities\n    prefix: CAP\n",
        encoding="utf-8",
    )
    rm = load_methodology(
        meta / "base.yaml",
        [FilesystemMixinSource(mixins)],
        active_mixins=["vision", "cap"],
    )
    return ToolContext(project_ctx=project_ctx, methodology=rm, id_registry=IdRegistry())


# ── (A) a known limitation: patch_spec revalidates Pydantic-typed fields ───────────────


@pytest.mark.asyncio
async def test_patch_spec_revalidates_relates_to_dicts(tmp_path: Path) -> None:
    """: passing relates_to as list-of-dicts gets parsed to RelatesToEntry."""
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    a = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "A", "up_to": []},
        ctx,
    )
    b = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "B", "up_to": []},
        ctx,
    )

    # Pass raw dict-shaped relates_to (the MCP wire form) — previously
    # this broke with `'dict' object has no attribute 'kind'`.
    result = await reg.call(
        "patch_spec",
        {
            "spec_id": a["spec_id"],
            "changes": {
                "relates_to": [
                    {"id": b["spec_id"], "kind": "relates_to", "reason": "test"},
                ]
            },
        },
        ctx,
    )
    assert result["ok"] is True

    reloaded = read_spec(Path(result["path"]))
    assert len(reloaded.relates_to) == 1
    entry = reloaded.relates_to[0]
    # Must be a proper Pydantic model now.
    assert isinstance(entry, RelatesToEntry)
    assert entry.id == b["spec_id"]
    assert entry.kind == "relates_to"
    assert entry.reason == "test"


@pytest.mark.asyncio
async def test_patch_spec_rejects_relates_to_non_list(tmp_path: Path) -> None:
    """: passing a non-list to relates_to fails loudly."""
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    a = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "A", "up_to": []},
        ctx,
    )
    with pytest.raises(PatchSpecError, match="must be a list"):
        await reg.call(
            "patch_spec",
            {"spec_id": a["spec_id"], "changes": {"relates_to": "not-a-list"}},
            ctx,
        )


@pytest.mark.asyncio
async def test_patch_spec_revalidates_provenance_dict(tmp_path: Path) -> None:
    """: provenance dict gets parsed to Provenance model."""
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    a = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "A", "up_to": []},
        ctx,
    )
    result = await reg.call(
        "patch_spec",
        {
            "spec_id": a["spec_id"],
            "changes": {"provenance": {"kind": "doc", "ref": "spec.md", "confidence": 0.9}},
        },
        ctx,
    )
    assert result["ok"] is True
    reloaded = read_spec(Path(result["path"]))
    assert reloaded.provenance is not None
    assert reloaded.provenance.kind == "doc"
    assert reloaded.provenance.confidence == 0.9


# ── (B) a known limitation: record_spec auto-symmetrises on initial write ──────────────


@pytest.mark.asyncio
async def test_record_spec_symmetrises_parent_down_to(tmp_path: Path) -> None:
    """: brand-new spec's parent down_to is populated immediately
    without a follow-up patch_spec call."""
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
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
    reloaded_parent = read_spec(Path(parent["path"]))
    assert child["spec_id"] in reloaded_parent.down_to


# ── (D) a known limitation: patch_spec removes stale parent down_to on re-anchor ───────


@pytest.mark.asyncio
async def test_patch_spec_removes_stale_parent_on_reanchor(tmp_path: Path) -> None:
    """: changing up_to from [OLD] to [NEW] strips OLD.down_to."""
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    old_parent = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "Old", "up_to": []},
        ctx,
    )
    new_parent = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "New", "up_to": []},
        ctx,
    )
    child = await reg.call(
        "record_spec",
        {
            "type": "capability",
            "layer": "L2",
            "title": "Child",
            "up_to": [old_parent["spec_id"]],
        },
        ctx,
    )

    # Sanity: old parent now has child in down_to (a known limitation symmetrise).
    assert child["spec_id"] in read_spec(Path(old_parent["path"])).down_to

    # Re-anchor.
    await reg.call(
        "patch_spec",
        {"spec_id": child["spec_id"], "changes": {"up_to": [new_parent["spec_id"]]}},
        ctx,
    )

    # Old parent's down_to no longer references the child (a known limitation cleanup).
    assert child["spec_id"] not in read_spec(Path(old_parent["path"])).down_to
    # New parent's down_to does (a known limitation add).
    assert child["spec_id"] in read_spec(Path(new_parent["path"])).down_to


@pytest.mark.asyncio
async def test_patch_spec_removes_stale_relates_to(tmp_path: Path) -> None:
    """: removing a relates_to entry strips reciprocal from neighbor."""
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    a = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "A", "up_to": []},
        ctx,
    )
    b = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "B", "up_to": []},
        ctx,
    )
    # Set up a relates_to edge A → B.
    await reg.call(
        "patch_spec",
        {
            "spec_id": a["spec_id"],
            "changes": {"relates_to": [{"id": b["spec_id"], "kind": "relates_to", "reason": "x"}]},
        },
        ctx,
    )
    # Sanity: reciprocal is present.
    b_spec = read_spec(Path(b["path"]))
    assert any(e.id == a["spec_id"] for e in b_spec.relates_to)

    # Now remove the relates_to entirely on A.
    await reg.call(
        "patch_spec",
        {"spec_id": a["spec_id"], "changes": {"relates_to": []}},
        ctx,
    )

    # B's relates_to no longer points back at A.
    b_spec_after = read_spec(Path(b["path"]))
    assert not any(e.id == a["spec_id"] for e in b_spec_after.relates_to)


# ── (E) a known limitation: Cross-spec validator finds matrix out of scope ─────────────


def test_rbac_validator_finds_matrix_out_of_scope(tmp_path: Path) -> None:
    """: scoped validate of a role-using spec finds the matrix
    even when the matrix is not in `specs` (only in ctx.corpus)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base.yaml"
    base.write_text(
        (
            "version: 1\n"
            "layers:\n"
            "  - id: L1\n    role: source-boundary\n    name: vision\n"
            "  - id: L2\n    role: middle\n    name: capabilities\n"
            "  - id: L4\n    role: code-boundary\n    name: technical\n    is_final: true\n"
        ),
        encoding="utf-8",
    )
    mixins = tmp_path / "mixins"
    mixins.mkdir()
    (mixins / "vision.yaml").write_text(
        "mixin: vision\nadds_types:\n  - id: vision\n"
        "    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins / "cap.yaml").write_text(
        "mixin: cap\nadds_types:\n  - id: capability\n"
        "    layer_name: capabilities\n    prefix: CAP\n",
        encoding="utf-8",
    )
    (mixins / "rbac.yaml").write_text(
        "mixin: rbac\nadds_types:\n  - id: rbac-matrix\n"
        "    layer_role: code-boundary\n    prefix: RBAC\n",
        encoding="utf-8",
    )
    (mixins / "api.yaml").write_text(
        "mixin: api\nadds_types:\n  - id: api-contract\n"
        "    layer_role: code-boundary\n    prefix: API\n",
        encoding="utf-8",
    )
    methodology = load_methodology(
        base,
        [FilesystemMixinSource(mixins)],
        active_mixins=["vision", "cap", "rbac", "api"],
    )

    matrix_body = "## Roles\n\n```yaml\n- id: admin\n- id: operator\n```\n"
    matrix = Spec(
        id="L4-RBAC-001",
        type="rbac-matrix",
        layer="L4",
        title="RBAC",
        up_to=[],
        body=matrix_body,
    )
    api = Spec(
        id="L4-API-001",
        type="api-contract",
        layer="L4",
        title="api",
        up_to=[],
        roles=["admin"],  # type: ignore[call-arg]
    )
    ctx = ValidatorContext(corpus={matrix.id: matrix, api.id: api}, methodology=methodology)

    # Scope to just the api spec — matrix is OUT of scope.
    issues = check_rbac_roles_resolved([api], ctx)
    # 'admin' IS in the matrix (in corpus), so no rbac_unknown_role.
    unknown = [i for i in issues if i.code == "rbac_unknown_role"]
    assert unknown == [], f"expected matrix to be found via ctx.corpus, got: {unknown}"


def test_rbac_validator_flags_unknown_role_when_matrix_in_corpus(
    tmp_path: Path,
) -> None:
    """: when role really isn't in the matrix, still flag it."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base.yaml"
    base.write_text(
        (
            "version: 1\n"
            "layers:\n"
            "  - id: L1\n    role: source-boundary\n    name: vision\n"
            "  - id: L2\n    role: middle\n    name: capabilities\n"
            "  - id: L4\n    role: code-boundary\n    name: technical\n    is_final: true\n"
        ),
        encoding="utf-8",
    )
    mixins = tmp_path / "mixins"
    mixins.mkdir()
    (mixins / "rbac.yaml").write_text(
        "mixin: rbac\nadds_types:\n  - id: rbac-matrix\n"
        "    layer_role: code-boundary\n    prefix: RBAC\n",
        encoding="utf-8",
    )
    (mixins / "api.yaml").write_text(
        "mixin: api\nadds_types:\n  - id: api-contract\n"
        "    layer_role: code-boundary\n    prefix: API\n",
        encoding="utf-8",
    )
    methodology = load_methodology(
        base, [FilesystemMixinSource(mixins)], active_mixins=["rbac", "api"]
    )

    matrix = Spec(
        id="L4-RBAC-001",
        type="rbac-matrix",
        layer="L4",
        title="RBAC",
        up_to=[],
        body="## Roles\n\n```yaml\n- id: admin\n```\n",
    )
    api = Spec(
        id="L4-API-001",
        type="api-contract",
        layer="L4",
        title="api",
        up_to=[],
        roles=["bogus-role"],  # type: ignore[call-arg]
    )
    ctx = ValidatorContext(corpus={matrix.id: matrix, api.id: api}, methodology=methodology)
    issues = check_rbac_roles_resolved([api], ctx)
    unknown = [i for i in issues if i.code == "rbac_unknown_role"]
    assert unknown, "real unknown role must still be flagged"
    assert any("bogus-role" in i.message for i in unknown)


def _unused_helpers(_: Any) -> None:
    # Keep import alive for type-checker; not used at runtime.
    _ = check_rbac_roles_resolved


# ──  patch_spec rejects unknown keys ───────────────────────────────


@pytest.mark.asyncio
async def test_patch_spec_rejects_unknown_changes_key(tmp_path: Path) -> None:
    """: unknown keys raise so silent no-ops can't recur."""
    ctx = _ctx(tmp_path)
    reg = ToolRegistry()
    register_record_spec(reg)
    register_patch_spec(reg)
    create = await reg.call(
        "record_spec",
        {"type": "vision", "layer": "L1", "title": "X", "up_to": []},
        ctx,
    )
    with pytest.raises(PatchSpecError, match="unknown key 'body_replac'"):
        # Typo on body_replace — must not silently succeed
        await reg.call(
            "patch_spec",
            {
                "spec_id": create["spec_id"],
                "changes": {"body_replac": [{"old": "X", "new": "Y"}]},
            },
            ctx,
        )


@pytest.mark.asyncio
async def test_patch_spec_accepts_existing_extras_for_cleanup(tmp_path: Path) -> None:
    """: keys already in model_extra are patchable so stale
    frontmatter pollution can be cleaned via patch_spec({key: None})."""
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
            # Simulate : spec has a polluted body_replace
            # frontmatter key. record_spec accepts arbitrary args into
            # extras, so we use that to reproduce the polluted state.
            "body_replace": [{"old": "foo", "new": "bar"}],
        },
        ctx,
    )
    # Sanity: pollution is on the spec.
    polluted = read_spec(Path(create["path"]))
    assert "body_replace" in (polluted.model_extra or {})

    # Now clean it up via patch_spec({"body_replace": None}) — must NOT
    # be interpreted as "apply body_replace op with None ops".
    result = await reg.call(
        "patch_spec",
        {"spec_id": create["spec_id"], "changes": {"body_replace": None}},
        ctx,
    )
    assert result["ok"] is True

    cleaned = read_spec(Path(create["path"]))
    assert "body_replace" not in (cleaned.model_extra or {})


# ──  body_replace surgical edit op ─────────────────────────────────


@pytest.mark.asyncio
async def test_patch_spec_body_replace_applies_unique_substitution(
    tmp_path: Path,
) -> None:
    """: body_replace with a single op patches one substring."""
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
            "body": "Original purpose text.\n\nDetails follow here.",
        },
        ctx,
    )
    result = await reg.call(
        "patch_spec",
        {
            "spec_id": create["spec_id"],
            "changes": {
                "body_replace": [{"old": "Original purpose text.", "new": "Updated purpose text."}]
            },
        },
        ctx,
    )
    assert result["ok"] is True
    reloaded = read_spec(Path(result["path"]))
    assert "Updated purpose text." in reloaded.body
    assert "Original purpose text." not in reloaded.body


@pytest.mark.asyncio
async def test_patch_spec_body_replace_raises_on_not_found(tmp_path: Path) -> None:
    """: missing substring raises rather than silently failing."""
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
            "body": "Some body content.",
        },
        ctx,
    )
    with pytest.raises(PatchSpecError, match="not found"):
        await reg.call(
            "patch_spec",
            {
                "spec_id": create["spec_id"],
                "changes": {"body_replace": [{"old": "nonexistent string", "new": "x"}]},
            },
            ctx,
        )


@pytest.mark.asyncio
async def test_patch_spec_body_replace_raises_on_ambiguous(tmp_path: Path) -> None:
    """: multi-match substring raises (force unique context)."""
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
            "body": "foo bar foo baz foo",
        },
        ctx,
    )
    with pytest.raises(PatchSpecError, match="matches 3 times"):
        await reg.call(
            "patch_spec",
            {
                "spec_id": create["spec_id"],
                "changes": {"body_replace": [{"old": "foo", "new": "FOO"}]},
            },
            ctx,
        )


@pytest.mark.asyncio
async def test_patch_spec_body_replace_chains_in_order(tmp_path: Path) -> None:
    """: ordered ops cascade — second op sees first op's result."""
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
            "body": "alpha then beta then gamma",
        },
        ctx,
    )
    result = await reg.call(
        "patch_spec",
        {
            "spec_id": create["spec_id"],
            "changes": {
                "body_replace": [
                    {"old": "alpha", "new": "ALPHA"},
                    {"old": "ALPHA then beta", "new": "AB chain"},
                ]
            },
        },
        ctx,
    )
    assert result["ok"] is True
    reloaded = read_spec(Path(result["path"]))
    assert reloaded.body.strip() == "AB chain then gamma"
