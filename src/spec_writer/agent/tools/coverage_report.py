"""``coverage_report`` MCP tool — REV coverage snapshot.

Reads the passive :mod:`engine.coverage_ledger` and compares it
against the current code inventory (``discover_code_inventory`` walks
fresh — best-effort hint). Returns:

- ``visited`` — paths the agent read / grepped (set, JSON-serialised
  as a sorted list).
- ``covered`` — paths referenced by at least one spec's
  ``implemented_by``.
- ``uncovered`` — paths in inventory that are neither visited nor
  covered, ordered by inventory walk order.
- ``ratio_visited`` / ``ratio_covered`` — float in [0, 1].
- ``total_files`` — inventory size used for ratios.

Args (dict, all optional):
- ``includes`` / ``excludes`` / ``max_files`` — passed straight to
  :func:`discover_code_inventory` so the report matches the inventory
  the agent is working from.

The agent calls this after a generation pass to see where it stands;
the chat-system-prompt tells the agent to surface options via
``ask_user`` when ``ratio_covered`` falls below the ``warn_below``
threshold declared in :file:`code-coverage-gate` mixin (see S4.5
hook config).
"""

from __future__ import annotations

from typing import Any

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.agent.tools.discover_code_inventory import discover_code_inventory
from spec_writer.engine import coverage_ledger


class CoverageReportError(RuntimeError):
    pass


async def coverage_report(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    inventory_args: dict[str, Any] = {
        k: args[k] for k in ("includes", "excludes", "max_files") if k in args
    }
    inventory = await discover_code_inventory(inventory_args, ctx)
    if not inventory.get("ok"):
        return {
            "ok": False,
            "error": "coverage_report could not build inventory",
            "inventory_error": inventory.get("error"),
        }

    files = [f["path"] for f in inventory["files"]]
    ledger = coverage_ledger.load(ctx.project_ctx.meta)

    visited_against_inv = [p for p in files if ledger.is_visited(p)]
    covered_against_inv = [p for p in files if ledger.is_covered(p)]
    uncovered = ledger.uncovered_against(files)

    total = len(files)
    ratio_visited = len(visited_against_inv) / total if total else 0.0
    ratio_covered = len(covered_against_inv) / total if total else 0.0

    # Also report ledger entries that don't appear in the current inventory —
    # could be files outside includes/excludes or deleted since last walk.
    visited_outside = sorted(ledger.visited_paths() - set(files))
    covered_outside = sorted(ledger.covered_paths() - set(files))

    return {
        "ok": True,
        "total_files": total,
        "visited": sorted(visited_against_inv),
        "covered": sorted(covered_against_inv),
        "uncovered": uncovered,
        "visited_outside_inventory": visited_outside,
        "covered_outside_inventory": covered_outside,
        "ratio_visited": ratio_visited,
        "ratio_covered": ratio_covered,
    }


def register(registry: ToolRegistry) -> None:
    registry.register("coverage_report", coverage_report)


__all__ = ["CoverageReportError", "coverage_report", "register"]
