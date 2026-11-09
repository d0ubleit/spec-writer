"""S2.3: AgentRuntime drives LLMClient ↔ ToolRegistry."""

from __future__ import annotations

from pathlib import Path

import pytest

from spec_writer.agent.feedback import format_validation_feedback
from spec_writer.agent.providers.fake import FakeLLMClient, ScriptedResponse
from spec_writer.agent.runtime import AgentRuntime
from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.agent.tools.patch_spec import register as register_patch_spec
from spec_writer.agent.tools.record_spec import register as register_record_spec
from spec_writer.engine.id_registry import IdRegistry
from spec_writer.engine.project_context import ProjectContext
from spec_writer.engine.seed import Seed
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.sources import FilesystemMixinSource
from spec_writer.methodology.validators import ValidationIssue


def _bootstrap(tmp_path: Path):
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
    (mixins / "core-traceability.yaml").write_text(
        (
            "mixin: core-traceability\n"
            "validators:\n"
            "  - id: pyramid-adjacency\n"
            "    scope: per-spec\n"
            "    func_ref: validators.check_pyramid_adjacency\n"
            "  - id: english-output\n"
            "    scope: per-spec\n"
            "    func_ref: validators.check_english_output\n"
        ),
        encoding="utf-8",
    )
    rm = load_methodology(
        meta / "base.yaml",
        [FilesystemMixinSource(mixins)],
        active_mixins=["vision", "core-traceability"],
    )

    registry = ToolRegistry()
    register_record_spec(registry)
    register_patch_spec(registry)
    ctx = ToolContext(project_ctx=project_ctx, methodology=rm, id_registry=IdRegistry())
    layer = next(layer for layer in rm.layers if layer.id == "L1")
    return ctx, registry, rm, layer


def _seed(layer):
    return Seed(layer=layer, direction="fwd")


# ── feedback formatter ───────────────────────────────────────────────────


def test_feedback_formats_errors_and_warnings() -> None:
    issues = [
        ValidationIssue(
            severity="error",
            code="missing_section",
            spec_id="L1-VIS-001",
            field="purpose",
            message="missing section",
            fix_hint="add Purpose section",
        ),
        ValidationIssue(
            severity="warning",
            code="short_spec",
            spec_id="L1-VIS-001",
            message="spec is short",
        ),
    ]
    feedback = format_validation_feedback(issues)
    assert "ERRORS" in feedback
    assert "missing_section" in feedback
    assert "patch_spec" in feedback
    assert "short_spec" in feedback


def test_feedback_no_issues_empty() -> None:
    assert format_validation_feedback([]) == ""


# ── runtime + FakeLLM ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_runtime_records_spec_via_tool_call(tmp_path: Path) -> None:
    ctx, registry, rm, layer = _bootstrap(tmp_path)
    fake = FakeLLMClient(
        [
            ScriptedResponse(
                tool_calls=[
                    (
                        "record_spec",
                        {
                            "type": "vision",
                            "layer": "L1",
                            "title": "Generated Vision",
                            "up_to": [],
                            "body": "## Purpose\n\nA vision.\n",
                        },
                    )
                ],
                text="done",
            ),
            ScriptedResponse(text="acknowledged"),
        ]
    )
    runtime = AgentRuntime(fake, registry, ctx)
    drafts = await runtime.produce_drafts(layer, _seed(layer), corpus={}, methodology=rm)
    assert len(drafts) == 1
    assert drafts[0].title == "Generated Vision"
    assert drafts[0].id == "L1-VIS-001"


@pytest.mark.asyncio
async def test_runtime_no_tool_calls_returns_empty(tmp_path: Path) -> None:
    ctx, registry, rm, layer = _bootstrap(tmp_path)
    fake = FakeLLMClient([ScriptedResponse(text="I cannot produce a spec.")])
    runtime = AgentRuntime(fake, registry, ctx)
    drafts = await runtime.produce_drafts(layer, _seed(layer), corpus={}, methodology=rm)
    assert drafts == []


@pytest.mark.asyncio
async def test_runtime_feedback_retry_on_validation_error(tmp_path: Path) -> None:
    """First attempt produces a non-English title (validator fails); the
    feedback message is appended to history and the second attempt
    produces a clean spec."""
    ctx, registry, rm, layer = _bootstrap(tmp_path)
    fake = FakeLLMClient(
        [
            # First attempt — non-English title triggers english-output validator
            ScriptedResponse(
                tool_calls=[
                    (
                        "record_spec",
                        {
                            "type": "vision",
                            "layer": "L1",
                            "title": "Платформа",
                            "up_to": [],
                        },
                    )
                ],
                text="initial",
            ),
            ScriptedResponse(text="initial done"),  # no more tool calls this turn
            # Second attempt — patch via patch_spec
            ScriptedResponse(
                tool_calls=[
                    (
                        "patch_spec",
                        {
                            "spec_id": "L1-VIS-001",
                            "changes": {"title": "Platform Vision"},
                        },
                    )
                ],
                text="fixing",
            ),
            ScriptedResponse(text="fixed"),
        ]
    )
    runtime = AgentRuntime(fake, registry, ctx, max_feedback_rounds=1)
    drafts = await runtime.produce_drafts(layer, _seed(layer), corpus={}, methodology=rm)
    assert len(drafts) == 1
    assert drafts[0].title == "Platform Vision"


@pytest.mark.asyncio
async def test_runtime_unknown_tool_recorded_as_error(tmp_path: Path) -> None:
    ctx, registry, rm, layer = _bootstrap(tmp_path)
    fake = FakeLLMClient(
        [
            ScriptedResponse(tool_calls=[("ghost", {})], text="trying"),
            ScriptedResponse(text="ok"),
        ]
    )
    runtime = AgentRuntime(fake, registry, ctx)
    drafts = await runtime.produce_drafts(layer, _seed(layer), corpus={}, methodology=rm)
    assert drafts == []


@pytest.mark.asyncio
async def test_as_producer_returns_callable(tmp_path: Path) -> None:
    ctx, registry, rm, layer = _bootstrap(tmp_path)
    fake = FakeLLMClient(
        [
            ScriptedResponse(
                tool_calls=[
                    (
                        "record_spec",
                        {
                            "type": "vision",
                            "layer": "L1",
                            "title": "Producer Test",
                            "up_to": [],
                        },
                    )
                ],
                text="done",
            ),
            ScriptedResponse(text="bye"),
        ]
    )
    runtime = AgentRuntime(fake, registry, ctx)
    producer = runtime.as_producer()
    drafts = await producer(layer, _seed(layer), {}, rm)
    assert len(drafts) == 1


@pytest.mark.asyncio
async def test_runtime_tool_exception_captured(tmp_path: Path) -> None:
    ctx, registry, rm, layer = _bootstrap(tmp_path)

    async def buggy(_args, _ctx):
        raise RuntimeError("kaboom")

    registry.register("buggy", buggy)
    fake = FakeLLMClient(
        [
            ScriptedResponse(tool_calls=[("buggy", {})], text="ouch"),
            ScriptedResponse(text="done"),
        ]
    )
    runtime = AgentRuntime(fake, registry, ctx)
    drafts = await runtime.produce_drafts(layer, _seed(layer), corpus={}, methodology=rm)
    assert drafts == []
