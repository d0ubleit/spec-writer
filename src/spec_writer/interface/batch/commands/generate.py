"""``spec-writer generate`` — run the engine pipeline for the given scope.

Phase 2 wires the CLI to the agent runtime: agent.yaml selects a
provider (default ``claude-subscription``), the runtime drives
record_spec/patch_spec tool calls, and the engine pipeline executes
the layered loop. Tests that don't want a real LLM round-trip
inject a different ``build_producer`` (e.g. a stub returning fixed
Spec objects) — Phase 1's hook still applies.
"""

from __future__ import annotations

import asyncio
from importlib import resources
from pathlib import Path
from typing import Any, cast

import typer
import yaml

from spec_writer.agent.mcp_tools import build_mcp_server, default_allowed_tools
from spec_writer.agent.runtime import AgentRuntime
from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.agent.tools.ask_user import register as register_ask_user
from spec_writer.agent.tools.build_reverse_index import register as register_build_reverse_index
from spec_writer.agent.tools.coverage_report import register as register_coverage_report
from spec_writer.agent.tools.delete_spec import register as register_delete_spec
from spec_writer.agent.tools.detect_code_changes import register as register_detect_code_changes
from spec_writer.agent.tools.discover_code_inventory import (
    register as register_discover_code_inventory,
)
from spec_writer.agent.tools.list_corpus import register as register_list_corpus
from spec_writer.agent.tools.list_sources import register as register_list_sources
from spec_writer.agent.tools.metrics import register as register_metrics
from spec_writer.agent.tools.patch_spec import register as register_patch_spec
from spec_writer.agent.tools.query_traceability import register as register_query_traceability
from spec_writer.agent.tools.record_spec import register as register_record_spec
from spec_writer.agent.tools.record_sync_complete import register as register_record_sync_complete
from spec_writer.agent.tools.resolve_scope import register as register_resolve_scope
from spec_writer.agent.tools.resolve_source_chain import register as register_resolve_source_chain
from spec_writer.agent.tools.validate import register as register_validate
from spec_writer.config import AgentConfig, load_agent_config
from spec_writer.engine.id_registry import IdRegistry
from spec_writer.engine.loop import DraftProducer, PipelineResult, run_layered_pipeline
from spec_writer.engine.project_context import NoProjectRootError, ProjectContext
from spec_writer.io.source_sync import sync_sources
from spec_writer.methodology.loader import (
    MethodologyLoadError,
    load_methodology,
)
from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.methodology.sources import FilesystemMixinSource
from spec_writer.types import Scenario, Scope, Spec


def _default_build_producer(
    project_ctx: ProjectContext,
    methodology: ResolvedMethodology,
    config: AgentConfig,
) -> DraftProducer:
    """Wire Claude provider (with MCP server) → AgentRuntime → producer adapter."""
    if config.provider != "claude-subscription":
        raise typer.BadParameter(
            f"provider {config.provider!r} not implemented in Phase 2 autonomous run"
        )

    id_registry = IdRegistry.from_disk(project_ctx.specs)
    mcp_server, recorded_ids = build_mcp_server(project_ctx, methodology, id_registry)

    from spec_writer.agent.providers.claude_subscription import (
        ClaudeSubscriptionClient,
    )

    llm = ClaudeSubscriptionClient(
        model=config.model,
        mcp_servers={"spec-writer": mcp_server},
        allowed_tools=default_allowed_tools(),
    )

    # Local ToolRegistry is unused in SDK-MCP mode but the AgentRuntime
    # still expects a real registry (e.g. for FakeLLM fallback). Wire the
    # same handlers so behaviour stays uniform if mode flips.
    registry = ToolRegistry()
    register_record_spec(registry)
    register_patch_spec(registry)
    register_query_traceability(registry)
    register_resolve_scope(registry)
    register_resolve_source_chain(registry)
    register_ask_user(registry)
    register_validate(registry)
    register_metrics(registry)
    register_list_sources(registry)
    register_list_corpus(registry)
    register_discover_code_inventory(registry)
    register_coverage_report(registry)
    register_detect_code_changes(registry)
    register_build_reverse_index(registry)
    register_record_sync_complete(registry)
    register_delete_spec(registry)
    ctx = ToolContext(project_ctx=project_ctx, methodology=methodology, id_registry=id_registry)
    runtime = AgentRuntime(llm, registry, ctx, recorded_ids_buffer=recorded_ids)
    return runtime.as_producer()


# Module-level hook — tests override this for FakeLLM / stub flows.
build_producer = _default_build_producer


def _load_methodology_for(project_ctx: ProjectContext) -> ResolvedMethodology:
    methodology_yaml = project_ctx.methodology_yaml
    if not methodology_yaml.exists():
        raise MethodologyLoadError(f"missing {methodology_yaml} — run `spec-writer init` first")
    config = yaml.safe_load(methodology_yaml.read_text(encoding="utf-8")) or {}
    active = list(config.get("active_mixins") or [])
    base_path = Path(str(resources.files("spec_writer.methodology._builtin").joinpath("base.yaml")))
    mixins_dir = Path(str(resources.files("spec_writer.methodology._builtin").joinpath("mixins")))
    return load_methodology(base_path, [FilesystemMixinSource(mixins_dir)], active)


def _load_existing_corpus(project_ctx: ProjectContext):
    """Read every .md spec file under specs/ into a {id: Spec} dict."""
    from spec_writer.io.spec_io import read_spec

    corpus: dict[str, Spec] = {}
    if not project_ctx.specs.exists():
        return corpus
    for path in sorted(project_ctx.specs.rglob("*.md")):
        try:
            spec = read_spec(path)
        except Exception:
            continue
        corpus[spec.id] = spec
    return corpus


def run_pipeline_for(
    project_ctx: ProjectContext,
    direction: str,
    layers: list[str],
    mode: str,
    max_iterations: int,
    instruction: str | None = None,
    scope: Scope | None = None,
    engine: str = "loop",
    approve_layers: bool = False,
) -> PipelineResult:
    """Shared engine invocation used by both generate and refine.

    ``engine`` picks the orchestrator: ``loop`` (default, hand-rolled
    convergence loop) or ``graph`` (LangGraph ``StateGraph`` with
    checkpointer + optional per-layer approval interrupts).
    """
    sync_sources(project_ctx.sources_raw, project_ctx.sources)
    methodology = _load_methodology_for(project_ctx)
    agent_cfg = load_agent_config(project_root=project_ctx.root)

    layers_in_scope = layers or [layer.id for layer in methodology.layers]
    scenario = Scenario(
        direction=cast(Any, direction),
        layers_in_scope=layers_in_scope,
        mode=cast(Any, mode),
        max_iterations=max_iterations,
        scope=scope,
    )

    corpus = _load_existing_corpus(project_ctx)
    producer = build_producer(project_ctx, methodology, agent_cfg)

    if engine == "graph":
        return asyncio.run(
            _run_graph_pipeline(
                scenario,
                methodology,
                project_ctx,
                producer,
                corpus,
                instruction=instruction,
                approve_layers=approve_layers,
            )
        )
    if engine != "loop":
        raise typer.BadParameter(f"unknown engine {engine!r} (expected 'loop' or 'graph')")

    return asyncio.run(
        run_layered_pipeline(
            scenario,
            methodology,
            project_ctx,
            producer,
            existing_corpus=corpus,
            instruction=instruction,
        )
    )


async def _run_graph_pipeline(
    scenario: Scenario,
    methodology: ResolvedMethodology,
    project_ctx: ProjectContext,
    producer: DraftProducer,
    corpus: dict[str, Spec],
    *,
    instruction: str | None,
    approve_layers: bool,
) -> PipelineResult:
    """Drive the LangGraph engine, prompting for approval when it pauses."""
    from spec_writer.interface.batch.graph import GraphPipelineRun

    runner = GraphPipelineRun(
        scenario=scenario,
        methodology=methodology,
        project_ctx=project_ctx,
        producer=producer,
        existing_corpus=corpus,
        instruction=instruction,
        require_approval=approve_layers,
    )
    result = await runner.run()
    while result is None:
        payload = cast(dict[str, Any], runner.interrupt_payload or {})
        layer = payload.get("layer", "?")
        name = payload.get("layer_name", "")
        position = payload.get("position", "?")
        total = payload.get("total", "?")
        approved = typer.confirm(
            f"Generate layer {layer} ({name}) — {position}/{total}?",
            default=True,
        )
        result = await runner.resume(approved=approved)
    return result


def run(
    ctx: typer.Context,
    direction: str = typer.Option("fwd", "--direction", help="fwd|rev"),
    layer: list[str] = typer.Option(
        [], "--layer", help="Layer ids in scope (repeat); default: all."
    ),
    mode: str = typer.Option("generate", "--mode"),
    max_iterations: int = typer.Option(5, "--max-iterations"),
    engine: str = typer.Option(
        "loop",
        "--engine",
        help="Orchestrator: 'loop' (default) or 'graph' (LangGraph).",
    ),
    approve_layers: bool = typer.Option(
        False,
        "--approve-layers/--no-approve-layers",
        help="Graph engine only: pause for approval before each layer.",
    ),
    instruction: str | None = typer.Option(
        None,
        "--instruction",
        help="Free-form note threaded into the agent's user message.",
    ),
) -> None:
    """Execute one engine pipeline run."""
    state = cast(dict[str, object], ctx.obj or {})
    explicit_root = cast(Path | None, state.get("project_root"))

    try:
        project_ctx = (
            ProjectContext.from_cwd(explicit_root) if explicit_root else ProjectContext.from_cwd()
        )
    except NoProjectRootError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=11) from exc

    try:
        result = run_pipeline_for(
            project_ctx,
            direction=direction,
            layers=layer,
            mode=mode,
            max_iterations=max_iterations,
            instruction=instruction,
            engine=engine,
            approve_layers=approve_layers,
        )
    except MethodologyLoadError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=10) from exc

    _report_result(result)


def _report_result(result: PipelineResult) -> None:
    if result.succeeded:
        accepted_total = sum(len(lr.accepted_specs) for lr in result.layer_results)
        typer.echo(
            f"OK — {len(result.layer_results)} layer(s) converged, {accepted_total} spec(s) written"
        )
        for layer_result in result.layer_results:
            if layer_result.accepted_specs:
                typer.echo(f"  {layer_result.layer_id}: {', '.join(layer_result.accepted_specs)}")
            _emit_metric_summary(layer_result)
        raise typer.Exit(code=0)

    typer.echo(f"FAIL — {result.reason}", err=True)
    for layer_result in result.layer_results:
        errors = [i for i in layer_result.issues if i.severity == "error"]
        if errors:
            typer.echo(
                f"\nLayer {layer_result.layer_id} ({layer_result.state.value}) — "
                f"{len(errors)} error(s):",
                err=True,
            )
            for issue in errors[:10]:
                loc = issue.spec_id or "?"
                if issue.field:
                    loc = f"{loc}.{issue.field}"
                typer.echo(f"  [{issue.code}] {loc}: {issue.message}", err=True)
                if issue.fix_hint:
                    typer.echo(f"    fix: {issue.fix_hint}", err=True)
            if len(errors) > 10:
                typer.echo(f"  ... and {len(errors) - 10} more", err=True)
        _emit_metric_summary(layer_result, err=True)
    raise typer.Exit(code=1)


def _emit_metric_summary(layer_result: object, err: bool = False) -> None:
    metrics = getattr(layer_result, "metrics", None) or []
    if not metrics:
        return
    failing = [r for r in metrics if r.target_met is False and r.critical]
    warning = [r for r in metrics if r.warning]
    if not (failing or warning):
        return
    if failing:
        typer.echo(
            f"  metrics: {len(failing)} critical gate(s) below target",
            err=err,
        )
        for r in failing[:5]:
            v = "n/a" if r.value is None else f"{r.value:.2f}"
            typer.echo(f"    [FAIL] {r.id}={v} (target {r.target})", err=err)
    if warning:
        typer.echo(f"  metrics: {len(warning)} warning(s)", err=err)
        for r in warning[:5]:
            v = "n/a" if r.value is None else f"{r.value:.2f}"
            typer.echo(f"    [WARN] {r.id}={v} (warn_below {r.warn_below})", err=err)


__all__ = ["_report_result", "build_producer", "run", "run_pipeline_for"]
