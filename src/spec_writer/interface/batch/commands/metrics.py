"""``spec-writer metrics`` — compute methodology metrics over the on-disk corpus.

Read-only counterpart to ``validate``: loads the project's resolved
methodology, walks specs/ to build the corpus, then evaluates every
methodology-declared metric. Output is one line per metric:

  L2 up_to_completeness=1.00 (target 1.0) ok
  L3 sign_in_coverage=0.40 (target 0.8) WARN below 0.6
  L3 critical_acceptance_coverage=0.20 (target 0.9 critical) FAIL

Exit codes:
- 0 — every metric either meets target or is non-critical.
- 1 — at least one critical metric is below target.
- 10/11 — methodology / project-root errors (same as generate/validate).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import typer

from spec_writer.engine.project_context import NoProjectRootError, ProjectContext
from spec_writer.interface.batch.commands.generate import (
    _load_existing_corpus,
    _load_methodology_for,
)
from spec_writer.methodology.loader import MethodologyLoadError
from spec_writer.methodology.metrics import MetricResult
from spec_writer.methodology.metrics.runner import critical_failures, run_metrics


def run(
    ctx: typer.Context,
    layer: list[str] = typer.Option(
        [], "--layer", help="Filter to specific layer ids (repeatable)."
    ),
) -> None:
    """Compute and print every methodology-declared metric."""
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

    corpus = _load_existing_corpus(project_ctx)
    specs = list(corpus.values())
    results = run_metrics(specs, methodology, layer_ids=layer or None)

    if not results:
        typer.echo("(no metrics declared for the active methodology)")
        raise typer.Exit(code=0)

    for r in results:
        typer.echo(_format_result(r))

    failed = critical_failures(results)
    if failed:
        typer.echo(f"\n{len(failed)} critical metric(s) below target", err=True)
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


def _format_result(r: MetricResult) -> str:
    label = f"{r.layer_id} {r.id}"
    if r.error is not None:
        return f"{label} ERROR: {r.error}"
    value_str = "n/a" if r.value is None else f"{r.value:.2f}"
    parts = [f"{label}={value_str}"]
    if r.target is not None:
        parts.append(f"(target {r.target}{' critical' if r.critical else ''})")
    if r.target_met is False:
        parts.append("FAIL" if r.critical else "below target")
    elif r.warning:
        parts.append(f"WARN below {r.warn_below}")
    else:
        parts.append("ok")
    return " ".join(parts)


__all__ = ["run"]
