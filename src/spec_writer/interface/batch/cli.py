"""Top-level typer app for ``spec-writer`` CLI.

Phase 1 wires the ``init`` and ``generate`` subcommands fully, plus
typer-registered placeholders for the remaining commands listed in
COMMANDS.md so ``spec-writer --help`` shows the full surface area
(per S1.13 acceptance). Placeholders exit with code 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from spec_writer.engine.logging_setup import configure_logging
from spec_writer.interface.batch.commands import chat as _chat
from spec_writer.interface.batch.commands import demo as _demo
from spec_writer.interface.batch.commands import generate as _generate
from spec_writer.interface.batch.commands import init as _init
from spec_writer.interface.batch.commands import metrics as _metrics
from spec_writer.interface.batch.commands import prompt_debug as _prompt_debug
from spec_writer.interface.batch.commands import refine as _refine
from spec_writer.interface.batch.commands import reset as _reset
from spec_writer.interface.batch.commands import status as _status
from spec_writer.interface.batch.commands import usage_report as _usage_report
from spec_writer.interface.batch.commands import validate as _validate

app = typer.Typer(
    name="spec-writer",
    help="Layered specification corpus agent (V2).",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _global_options(
    ctx: typer.Context,
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        help="Explicit project root (override walk-up from CWD).",
    ),
    config: Path | None = typer.Option(
        None, "--config", help="Override project agent.yaml location."
    ),
    user_config: Path | None = typer.Option(
        None, "--user-config", help="Override ~/.spec-writer/agent.yaml location."
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase log verbosity (repeatable)."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Errors only."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan actions without writing to disk."),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI colors."),
) -> None:
    """Global flags shared by every subcommand."""
    if quiet:
        verbose = -1
    configure_logging(verbose=verbose)
    state: dict[str, object] = {
        "project_root": project_root,
        "config": config,
        "user_config": user_config,
        "verbose": verbose,
        "quiet": quiet,
        "dry_run": dry_run,
        "no_color": no_color,
    }
    ctx.obj = state


# ── Phase-1 functional subcommands ───────────────────────────────────────

app.command(name="init", help="Bootstrap a new spec-writer project.")(_init.run)
app.command(name="generate", help="Generate specs for the configured scope.")(_generate.run)
app.command(name="refine", help="Refine existing specs against validators.")(_refine.run)
app.command(name="validate", help="Run validators on the corpus (no LLM).")(_validate.run)
app.command(name="status", help="Project state summary.")(_status.run)
app.command(name="reset", help="Clean _meta/ targets (cache/sessions/locks).")(_reset.run)
app.command(name="prompt-debug", help="Render agent prompts without calling LLM.")(
    _prompt_debug.run
)
app.command(name="metrics", help="Compute methodology metrics over the corpus.")(_metrics.run)
app.command(name="chat", help="Interactive REPL backed by the spec-writer agent.")(_chat.run)
app.command(name="demo", help="Offline end-to-end demo; no API key required.")(_demo.run)
app.command(
    name="usage-report",
    help="Aggregate LLM token usage from _meta/usage/ .",
)(_usage_report.run)


# ── Phase-1 placeholders (exit 2 with TODO message) ──────────────────────


def _make_placeholder(name: str, hint: str):
    def _impl() -> None:
        typer.echo(f"`spec-writer {name}` — {hint}", err=True)
        raise typer.Exit(code=2)

    _impl.__doc__ = f"Placeholder: {hint}"
    _impl.__name__ = f"_{name}_placeholder"
    return _impl


for _name, _hint in (("sync", "sync coordinator arrives in Phase 4 (S4.7)"),):
    app.command(name=_name, help=f"[placeholder] {_hint}")(_make_placeholder(_name, _hint))


def main(argv: list[str] | None = None) -> int:
    """Entry point usable from tests; returns the exit code."""
    try:
        app(argv, standalone_mode=False)
    except typer.Exit as exc:
        return int(exc.exit_code)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
        return code
    return 0


if __name__ == "__main__":  # pragma: no cover — invoked via console script
    sys.exit(main())


__all__ = ["app", "main"]
