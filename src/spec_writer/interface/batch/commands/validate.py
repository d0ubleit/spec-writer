"""``spec-writer validate`` — run validators on the corpus, no LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import typer

from spec_writer.engine.project_context import NoProjectRootError, ProjectContext
from spec_writer.interface.batch.commands.generate import _load_methodology_for
from spec_writer.io.spec_io import read_spec
from spec_writer.methodology.loader import MethodologyLoadError
from spec_writer.methodology.validators.runner import run_validators


def _load_corpus(project_ctx: ProjectContext):
    specs = []
    if project_ctx.specs.exists():
        for path in sorted(project_ctx.specs.rglob("*.md")):
            try:
                specs.append(read_spec(path))
            except Exception:
                continue
    return specs


def run(
    ctx: typer.Context,
    output_format: str = typer.Option("table", "--format", help="table|json|terse"),
) -> None:
    """Validate every spec in the corpus against the active methodology."""
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
        methodology = _load_methodology_for(project_ctx)
    except MethodologyLoadError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=10) from exc

    specs = _load_corpus(project_ctx)
    issues = run_validators(specs, methodology)
    errors = [i for i in issues if i.severity == "error"]

    if output_format == "json":
        typer.echo(json.dumps([i.model_dump() for i in issues], default=str, indent=2))
    elif output_format == "terse":
        typer.echo(
            f"{len(specs)} specs, {len(errors)} errors, {len(issues) - len(errors)} non-error"
        )
    else:
        if not specs:
            typer.echo("(empty corpus)")
        else:
            typer.echo(f"Validated {len(specs)} spec(s):")
        if not issues:
            typer.echo("  ✓ all clean")
        else:
            for issue in issues:
                head = f"  [{issue.severity}] {issue.code}"
                if issue.spec_id:
                    head += f"  {issue.spec_id}"
                    if issue.field:
                        head += f".{issue.field}"
                typer.echo(head)
                typer.echo(f"    {issue.message}")
                if issue.fix_hint:
                    typer.echo(f"    fix: {issue.fix_hint}")

    raise typer.Exit(code=1 if errors else 0)


__all__ = ["run"]
