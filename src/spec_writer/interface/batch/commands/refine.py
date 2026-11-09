"""``spec-writer refine`` — iterate the existing corpus to fix issues.

Phase 2 minimum implementation: loads existing specs from disk,
re-runs the pipeline with ``mode='refine'``, and lets the agent
apply patches via the standard ``patch_spec`` MCP tool. The same
build_producer hook from :mod:`.generate` is used so tests can
inject stubs uniformly.

The engine's per-layer convergence loop already calls the validator
runner and feeds errors back to the agent (see AgentRuntime feedback
loop in S2.3). So a Phase 2 ``refine`` over existing corpus = "run
the same pipeline but in ``refine`` mode, with the disk corpus
pre-loaded so the agent sees its current state".
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import typer

from spec_writer.engine.project_context import NoProjectRootError, ProjectContext
from spec_writer.interface.batch.commands.generate import (
    _report_result,
    run_pipeline_for,
)
from spec_writer.methodology.loader import MethodologyLoadError
from spec_writer.types import Scope


def run(
    ctx: typer.Context,
    layer: list[str] = typer.Option(
        [], "--layer", help="Layer ids in scope (repeat); default: all."
    ),
    type_: list[str] = typer.Option(
        [], "--type", help="Only refine specs of this type (repeatable)."
    ),
    tag: list[str] = typer.Option(
        [], "--tag", help="Only refine specs carrying any of these tags (repeatable)."
    ),
    spec_id: list[str] = typer.Option(
        [], "--spec-id", help="Only refine specs with these ids (repeatable)."
    ),
    folder_pattern: str | None = typer.Option(
        None,
        "--folder-pattern",
        help="fnmatch pattern against '<layer-name>/<spec-id>' (e.g. 'api/*').",
    ),
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
        None, "--instruction", help="Free-form note threaded into the agent's prompt."
    ),
) -> None:
    """Refine specs at the given layer(s) against validators."""
    state = cast(dict[str, object], ctx.obj or {})
    explicit_root = cast(Path | None, state.get("project_root"))

    try:
        project_ctx = (
            ProjectContext.from_cwd(explicit_root) if explicit_root else ProjectContext.from_cwd()
        )
    except NoProjectRootError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=11) from exc

    scope = _build_scope(
        layers=layer,
        types=type_,
        tags=tag,
        spec_ids=spec_id,
        folder_pattern=folder_pattern,
    )

    try:
        result = run_pipeline_for(
            project_ctx,
            direction="fwd",
            layers=layer,
            mode="refine",
            max_iterations=max_iterations,
            instruction=instruction,
            scope=scope,
            engine=engine,
            approve_layers=approve_layers,
        )
    except MethodologyLoadError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=10) from exc

    _report_result(result)


def _build_scope(
    *,
    layers: list[str],
    types: list[str],
    tags: list[str],
    spec_ids: list[str],
    folder_pattern: str | None,
) -> Scope | None:
    """Materialise a Scope from CLI options, or None when no filter was given."""
    if not (layers or types or tags or spec_ids or folder_pattern):
        return None
    return Scope(
        layers=layers or None,
        types=types or None,
        tags=tags or None,
        spec_ids=spec_ids or None,
        folder_pattern=folder_pattern,
    )


__all__ = ["run"]
