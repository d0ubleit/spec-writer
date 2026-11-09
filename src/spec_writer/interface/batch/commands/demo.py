"""``spec-writer demo`` — a self-contained, offline tour of the pipeline.

No API key, no subscription, no network. The "model" is a scripted
:class:`~spec_writer.agent.providers.fake.FakeLLMClient` that emits the
same ``record_spec`` tool calls a real agent would, the tool dispatch is
the production :class:`~spec_writer.agent.tool_registry.ToolRegistry`,
and the orchestration runs on the LangGraph engine.

What the demo shows, end to end:

1. ``init`` scaffold — a project with the ``default-backend`` methodology.
2. A source document is ingested (``sources-raw/`` → ``sources/``).
3. L1 vision + L2 capability are generated through tool calls.
4. Validators and metrics run over the resulting corpus.
5. The traceability tree is walked from the vision down to capabilities.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any, cast

import typer

from spec_writer.agent.events import EngineEvent, SpecCreatedEvent
from spec_writer.agent.providers.fake import FakeLLMClient, ScriptedResponse
from spec_writer.agent.runtime import AgentRuntime
from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.agent.tools.record_spec import register as register_record_spec
from spec_writer.engine.id_registry import IdRegistry
from spec_writer.engine.loop import PipelineResult
from spec_writer.engine.project_context import ProjectContext
from spec_writer.interface.batch.commands.generate import _load_methodology_for
from spec_writer.interface.batch.commands.init import bootstrap_project
from spec_writer.io.source_sync import sync_sources
from spec_writer.methodology.metrics.runner import run_metrics
from spec_writer.methodology.validators.runner import run_validators
from spec_writer.types import Scenario

_DEMO_SOURCE = """\
# Mini CRM — product brief

A small team needs a lightweight CRM to keep track of their customers.

- The system must let an operator create, edit and archive customer records.
- Customer records hold a name, email, phone number and a free-form note.
- Operators must be able to search customers by name or email.
"""

_VISION_BODY = """\
## Purpose

A lightweight CRM that helps a small team keep every customer record in one
place: create, edit, archive and search customers without spreadsheet
churn.

## Target users

- Operator — the team member who maintains customer records.
"""

_CAPABILITY_BODY = """\
## Purpose

Manage the customer lifecycle: create a customer, amend their details,
archive a customer who left, and find a customer quickly by name or email.

## Actors

- Operator
"""


def _scripted_agent() -> FakeLLMClient:
    """Two turns per layer: one tool call, then a one-line wrap-up."""
    return FakeLLMClient(
        [
            ScriptedResponse(
                tool_calls=[
                    (
                        "record_spec",
                        {
                            "type": "vision",
                            "layer": "L1",
                            "title": "Mini CRM — one place for every customer record",
                            "tags": ["demo", "crm"],
                            "body": _VISION_BODY,
                        },
                    )
                ]
            ),
            ScriptedResponse(text="L1 vision recorded."),
            ScriptedResponse(
                tool_calls=[
                    (
                        "record_spec",
                        {
                            "type": "capability",
                            "layer": "L2",
                            "title": "Manage the customer lifecycle",
                            "up_to": ["L1-VIS-001"],
                            "tags": ["demo", "crm"],
                            "body": _CAPABILITY_BODY,
                        },
                    )
                ]
            ),
            ScriptedResponse(text="L2 capability recorded."),
        ]
    )


def _make_producer(project_ctx: ProjectContext, methodology: Any):
    registry = ToolRegistry()
    register_record_spec(registry)
    ctx = ToolContext(
        project_ctx=project_ctx,
        methodology=methodology,
        id_registry=IdRegistry.from_disk(project_ctx.specs),
    )
    runtime = AgentRuntime(_scripted_agent(), registry, ctx)
    return runtime.as_producer()


def _prepare_root(root: Path, clean: bool) -> None:
    if root.exists() and clean:
        if any(root.iterdir()) and not (root / "spec-editor").exists():
            raise typer.BadParameter(
                f"refusing to delete {root} — it does not look like a spec-writer demo project"
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def _walk_traceability(corpus: dict[str, Any], spec_id: str, depth: int = 0) -> list[str]:
    spec = corpus.get(spec_id)
    if spec is None:
        return []
    lines = [f"{'  ' * depth}└─ {spec.id} — {spec.title}"]
    for child_id in sorted(spec.down_to):
        lines.extend(_walk_traceability(corpus, child_id, depth + 1))
    return lines


def _print_report(project_ctx: ProjectContext, result: PipelineResult) -> None:
    methodology = _load_methodology_for(project_ctx)
    from spec_writer.interface.batch.commands.generate import _load_existing_corpus

    corpus = _load_existing_corpus(project_ctx)
    specs = list(corpus.values())

    files = sorted(path.relative_to(project_ctx.specs) for path in project_ctx.specs.rglob("*.md"))
    typer.echo("\n[3/5] Files written")
    for path in files:
        typer.echo(f"      specs/{path}")

    issues = run_validators(specs, methodology)
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    typer.echo(
        f"\n[4/5] Validation — {len(specs)} spec(s), "
        f"{len(errors)} error(s), {len(warnings)} warning(s)"
    )
    for issue in (errors + warnings)[:5]:
        typer.echo(f"      [{issue.severity}] {issue.code}: {issue.message}")

    typer.echo("\n[5/5] Metrics")
    populated_layers = {spec.layer for spec in specs}
    for metric in run_metrics(specs, methodology):
        if metric.value is None or metric.layer_id not in populated_layers:
            continue
        typer.echo(f"      {metric.id} = {metric.value:.2f} (target {metric.target})")

    typer.echo("\nTraceability — one vision, one capability")
    for line in _walk_traceability(corpus, "L1-VIS-001"):
        typer.echo(f"      {line}")

    total = sum(len(layer.accepted_specs) for layer in result.layer_results)
    typer.echo(
        f"\nPipeline: {'OK' if result.succeeded else 'FAIL'} — "
        f"{len(result.layer_results)} layer(s) converged, {total} spec(s) written"
    )


def run(
    ctx: typer.Context,
    project: Path = typer.Option(
        Path("demo-output"),
        "--project",
        help="Where to scaffold the throwaway demo project.",
    ),
    clean: bool = typer.Option(
        True,
        "--clean/--keep",
        help="Remove a previous demo project before starting.",
    ),
    engine: str = typer.Option(
        "graph",
        "--engine",
        help="Orchestrator to demonstrate: 'graph' (LangGraph) or 'loop'.",
    ),
) -> None:
    """Run a deterministic demo of the full spec-writing pipeline (offline)."""
    state = cast(dict[str, object], ctx.obj or {})
    base = cast(Path | None, state.get("project_root")) or Path.cwd()
    root = (base / project).resolve() if not project.is_absolute() else project.resolve()

    typer.echo(f"[1/5] Scaffolding demo project at {root}")
    _prepare_root(root, clean)
    bootstrap_project(root, preset="default-backend")

    project_ctx = ProjectContext.from_cwd(root)
    (project_ctx.sources_raw / "mini-crm-brief.md").write_text(_DEMO_SOURCE, encoding="utf-8")
    sync_sources(project_ctx.sources_raw, project_ctx.sources)
    typer.echo("      source ingested: sources/mini-crm-brief.md")

    methodology = _load_methodology_for(project_ctx)
    producer = _make_producer(project_ctx, methodology)

    created: list[str] = []

    def _on_event(event: EngineEvent) -> None:
        if isinstance(event, SpecCreatedEvent):
            created.append(event.spec_id)
            typer.echo(f"      + {event.spec_id}")

    from spec_writer.agent.events import EventBus

    bus = EventBus()
    bus.subscribe(_on_event)

    scenario = Scenario(
        direction="fwd",
        layers_in_scope=["L1", "L2"],
        mode="generate",
        max_iterations=2,
    )

    typer.echo(f"\n[2/5] Running the {engine} pipeline against the scripted agent")
    if engine == "graph":
        from spec_writer.interface.batch.graph import run_pipeline_graph

        result = asyncio.run(
            run_pipeline_graph(
                scenario,
                methodology,
                project_ctx,
                producer,
                event_bus=bus,
            )
        )
    elif engine == "loop":
        from spec_writer.engine.loop import run_layered_pipeline

        result = asyncio.run(
            run_layered_pipeline(
                scenario,
                methodology,
                project_ctx,
                producer,
                event_bus=bus,
            )
        )
    else:
        raise typer.BadParameter(f"unknown engine {engine!r}")

    _print_report(project_ctx, result)

    typer.echo("\nNext:")
    typer.echo(f"  spec-writer chat --project-root {root}   # real agent (needs `claude` CLI auth)")
    typer.echo(f"  spec-writer validate --project-root {root}")
    if not result.succeeded:
        raise typer.Exit(code=1)


__all__ = ["run"]
