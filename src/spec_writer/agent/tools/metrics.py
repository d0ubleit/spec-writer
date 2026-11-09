"""``metrics`` tool — compute methodology-declared metrics over the corpus.

Args (dict, all optional):
- ``layer_ids`` — list of layer ids to filter; default = all layers
  that declare any metric.

Returns: ``{"ok": True, "results": [metric_dict, ...]}``. Each
``metric_dict`` mirrors :class:`MetricResult` (id, layer_id, value,
target, warn_below, critical, target_met, warning, error, kind,
details).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.io.spec_io import read_spec
from spec_writer.methodology.metrics.runner import run_metrics
from spec_writer.types import Spec


class MetricsToolError(RuntimeError):
    pass


def _load_corpus(specs_root: Path) -> list[Spec]:
    out: list[Spec] = []
    if not specs_root.exists():
        return out
    for path in sorted(specs_root.rglob("*.md")):
        try:
            out.append(read_spec(path))
        except Exception:
            continue
    return out


async def metrics(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    raw_layers = args.get("layer_ids")
    layer_ids: list[str] | None
    if raw_layers is None:
        layer_ids = None
    elif isinstance(raw_layers, list):
        layer_ids = [item for item in cast(list[Any], raw_layers) if isinstance(item, str)]
    else:
        raise MetricsToolError("'layer_ids' must be a list of strings")

    corpus = _load_corpus(ctx.project_ctx.specs)
    results = run_metrics(corpus, ctx.methodology, layer_ids=layer_ids)

    return {
        "ok": True,
        "results": [
            {
                "id": r.id,
                "layer_id": r.layer_id,
                "value": r.value,
                "target": r.target,
                "warn_below": r.warn_below,
                "critical": r.critical,
                "target_met": r.target_met,
                "warning": r.warning,
                "error": r.error,
                "kind": r.kind,
                "details": r.details,
            }
            for r in results
        ],
    }


def register(registry: ToolRegistry) -> None:
    registry.register("metrics", metrics)


__all__ = ["MetricsToolError", "metrics", "register"]
