"""``spec-writer status`` — engine state summary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import cast

import typer

from spec_writer.engine.project_context import NoProjectRootError, ProjectContext
from spec_writer.interface.batch.commands.generate import _load_methodology_for
from spec_writer.io.spec_io import read_spec
from spec_writer.methodology.loader import MethodologyLoadError


def _count_specs_per_layer(project_ctx: ProjectContext) -> dict[str, int]:
    counter: Counter[str] = Counter()
    if not project_ctx.specs.exists():
        return {}
    for path in project_ctx.specs.rglob("*.md"):
        try:
            spec = read_spec(path)
        except Exception:
            continue
        counter[spec.layer] += 1
    return dict(counter)


def _lock_summary(project_ctx: ProjectContext) -> str:
    lock = project_ctx.meta / ".engine.lock"
    if not lock.exists():
        return "(clean)"
    try:
        data = json.loads(lock.read_text(encoding="utf-8"))
        return f"pid={data.get('pid')} started_at={data.get('started_at')} host={data.get('host')}"
    except Exception:
        return "(corrupt lock file)"


def run(ctx: typer.Context) -> None:
    """Print project root, methodology, spec counts, lock state."""
    state = cast(dict[str, object], ctx.obj or {})
    explicit_root = cast(Path | None, state.get("project_root"))

    try:
        project_ctx = (
            ProjectContext.from_cwd(explicit_root) if explicit_root else ProjectContext.from_cwd()
        )
    except NoProjectRootError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=11) from exc

    typer.echo(f"Project root: {project_ctx.root}")
    typer.echo(f"spec-editor:  {project_ctx.spec_editor}")

    try:
        methodology = _load_methodology_for(project_ctx)
        layer_summary = ", ".join(
            f"{layer.id} ({layer.role}, {layer.name})" for layer in methodology.layers
        )
        typer.echo(f"Layers:       {layer_summary}")
        typer.echo(f"Spec types:   {', '.join(sorted(methodology.spec_types)) or '(none)'}")
    except MethodologyLoadError as exc:
        typer.echo(f"Methodology:  ERROR {exc}", err=True)

    counts = _count_specs_per_layer(project_ctx)
    if counts:
        ordered = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        typer.echo(f"Specs:        {ordered} (total {sum(counts.values())})")
    else:
        typer.echo("Specs:        (none)")

    typer.echo(f"Engine lock:  {_lock_summary(project_ctx)}")


__all__ = ["run"]
