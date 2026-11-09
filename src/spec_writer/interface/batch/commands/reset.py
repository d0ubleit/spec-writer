"""``spec-writer reset`` — clean cache/sessions/locks under _meta/."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

import typer

from spec_writer.engine.project_context import NoProjectRootError, ProjectContext


def _remove(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def run(
    ctx: typer.Context,
    cache: bool = typer.Option(False, "--cache"),
    sessions: bool = typer.Option(False, "--sessions"),
    locks: bool = typer.Option(False, "--locks"),
    all_: bool = typer.Option(False, "--all", "-a"),
) -> None:
    """Delete designated _meta/ subdirectories. Spec/state files preserved."""
    state = cast(dict[str, object], ctx.obj or {})
    explicit_root = cast(Path | None, state.get("project_root"))

    try:
        project_ctx = (
            ProjectContext.from_cwd(explicit_root) if explicit_root else ProjectContext.from_cwd()
        )
    except NoProjectRootError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=11) from exc

    if not (cache or sessions or locks or all_):
        typer.echo("specify at least one of --cache / --sessions / --locks / --all", err=True)
        raise typer.Exit(code=2)

    targets: list[tuple[str, Path]] = []
    if all_ or cache:
        targets.append(
            ("cache (methodology.resolved.json)", project_ctx.meta / "methodology.resolved.json")
        )
        targets.append(("code_visited_ledger", project_ctx.meta / "code_visited_ledger.json"))
    if all_ or sessions:
        targets.append(("sessions/", project_ctx.meta / "sessions"))
    if all_ or locks:
        targets.append((".engine.lock", project_ctx.meta / ".engine.lock"))

    removed = 0
    for label, path in targets:
        if _remove(path):
            typer.echo(f"removed {label}")
            removed += 1
    if removed == 0:
        typer.echo("nothing to remove")
    else:
        typer.echo(f"done — {removed} target(s) removed")


__all__ = ["run"]
