"""``spec-writer prompt-debug`` — render the agent prompts without calling the LLM.

Loads the project methodology, builds the system + user prompts the
runtime would send for the chosen ``--layer`` and ``--mode``, and prints
them to stdout. Useful for verifying that a methodology change actually
reaches the model and for hand-eyeballing agent context before
spending real-Claude tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import typer

from spec_writer.agent.runtime import _build_system_prompt, _build_user_message
from spec_writer.engine.project_context import NoProjectRootError, ProjectContext
from spec_writer.engine.seed import build_seed
from spec_writer.interface.batch.commands.generate import (
    _load_existing_corpus,
    _load_methodology_for,
)
from spec_writer.methodology.loader import MethodologyLoadError


def run(
    ctx: typer.Context,
    layer: str = typer.Option(..., "--layer", help="Layer id (e.g. L1)."),
    mode: str = typer.Option("generate", "--mode", help="Scenario mode."),
    direction: str = typer.Option("fwd", "--direction", help="fwd|rev."),
) -> None:
    """Print the system + user prompts the agent would receive."""
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

    layer_obj = next((lay for lay in methodology.layers if lay.id == layer), None)
    if layer_obj is None:
        typer.echo(
            f"unknown layer {layer!r}; methodology declares: "
            f"{[lay.id for lay in methodology.layers]}",
            err=True,
        )
        raise typer.Exit(code=2)

    corpus = _load_existing_corpus(project_ctx)
    seed = build_seed(
        layer_obj,
        cast(Any, direction),
        methodology,
        project_ctx,
        corpus,
        mode=cast(Any, mode),
    )

    typer.echo("══════ SYSTEM PROMPT ══════")
    typer.echo(_build_system_prompt(layer_obj, methodology, direction=direction))
    typer.echo("\n══════ USER MESSAGE ══════")
    typer.echo(_build_user_message(layer_obj, seed, corpus))


__all__ = ["run"]
