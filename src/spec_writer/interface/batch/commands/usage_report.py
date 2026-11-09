"""``spec-writer usage-report`` — aggregate token usage from JSONL logs.

Reads every ``_meta/usage/*.jsonl`` file and prints daily totals plus
the cache-hit ratio (cache_read / (cache_read + input)).

The cost numbers are heuristic — actual SDK billing varies by model
and changes over time; the ratio is the reliable metric for tuning
caching strategy. See the design rationale.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import cast

import typer

from spec_writer.engine.project_context import NoProjectRootError, ProjectContext


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.1%}"


def _format_int(n: int) -> str:
    return f"{n:>12,}"


def run(ctx: typer.Context) -> None:
    """Print per-day usage totals + cache-hit ratio."""
    state = cast(dict[str, object], ctx.obj or {})
    explicit_root = cast(Path | None, state.get("project_root"))

    try:
        project_ctx = (
            ProjectContext.from_cwd(explicit_root) if explicit_root else ProjectContext.from_cwd()
        )
    except NoProjectRootError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    usage_dir = project_ctx.meta / "usage"
    if not usage_dir.exists() or not any(usage_dir.iterdir()):
        typer.echo(f"No usage data at {usage_dir}")
        typer.echo("(a known limitation: ChatAgent records usage automatically on every turn;")
        typer.echo(" if this directory is empty, no chat sessions have run since the fix shipped.)")
        return

    # Aggregate per-day, plus a corpus-wide grand total.
    per_day: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "input": 0,
            "output": 0,
            "cache_creation_input": 0,
            "cache_read_input": 0,
            "turns": 0,
        }
    )
    grand: dict[str, int] = defaultdict(int)
    layer_totals: dict[str, int] = defaultdict(int)

    for jsonl in sorted(usage_dir.glob("*.jsonl")):
        date = jsonl.stem
        for raw_line in jsonl.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            inp = int(record.get("input", 0))
            out = int(record.get("output", 0))
            cwrite = int(record.get("cache_creation_input", 0))
            cread = int(record.get("cache_read_input", 0))
            layer = str(record.get("layer", "unknown"))
            per_day[date]["input"] += inp
            per_day[date]["output"] += out
            per_day[date]["cache_creation_input"] += cwrite
            per_day[date]["cache_read_input"] += cread
            per_day[date]["turns"] += 1
            grand["input"] += inp
            grand["output"] += out
            grand["cache_creation_input"] += cwrite
            grand["cache_read_input"] += cread
            grand["turns"] += 1
            layer_totals[layer] += inp + out

    typer.echo(f"Usage report from {usage_dir}\n")
    typer.echo(
        f"{'Date':<12} {'input':>12} {'output':>12} {'cache_write':>12} "
        f"{'cache_read':>12} {'turns':>6}  hit-rate"
    )
    typer.echo("─" * 86)
    for date in sorted(per_day):
        d = per_day[date]
        hit = _format_ratio(d["cache_read_input"], d["cache_read_input"] + d["input"])
        typer.echo(
            f"{date:<12} {_format_int(d['input'])} {_format_int(d['output'])} "
            f"{_format_int(d['cache_creation_input'])} {_format_int(d['cache_read_input'])} "
            f"{d['turns']:>6}  {hit}"
        )

    typer.echo("─" * 86)
    grand_hit = _format_ratio(grand["cache_read_input"], grand["cache_read_input"] + grand["input"])
    typer.echo(
        f"{'TOTAL':<12} {_format_int(grand['input'])} {_format_int(grand['output'])} "
        f"{_format_int(grand['cache_creation_input'])} {_format_int(grand['cache_read_input'])} "
        f"{grand['turns']:>6}  {grand_hit}"
    )

    typer.echo("\nBy layer:")
    for layer, total in sorted(layer_totals.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {layer:<10} {_format_int(total)} (input + output)")
