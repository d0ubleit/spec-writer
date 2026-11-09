"""Token usage tracking with per-scope caps.

Records every LLM call to ``_meta/usage/<YYYY-MM-DD>.jsonl`` (one
JSON object per line) and maintains in-memory aggregates per layer
and per project. ``check_cap`` returns the first cap that would be
exceeded by the next call so the engine can abort cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from spec_writer.config import TokenLimits
from spec_writer.engine.atomic_io import atomic_write
from spec_writer.types import TokenUsage

CapName = Literal["per_session", "per_layer", "per_project"]


def _empty_layer_totals() -> dict[str, int]:
    return {}


@dataclass
class TokenLedger:
    """In-memory tally + JSONL durable log."""

    limits: TokenLimits
    usage_dir: Path | None = None
    per_session_total: int = 0
    per_project_total: int = 0
    per_layer_totals: dict[str, int] = field(default_factory=_empty_layer_totals)

    def record(self, layer_id: str, usage: TokenUsage, *, call_label: str = "") -> None:
        """Add ``usage`` to the running tallies and append a JSONL line."""
        total = usage.total or (usage.input + usage.output)
        self.per_session_total += total
        self.per_project_total += total
        self.per_layer_totals[layer_id] = self.per_layer_totals.get(layer_id, 0) + total
        self._write_jsonl(layer_id, usage, total, call_label)

    def check_cap(self, layer_id: str, prospective_tokens: int = 0) -> CapName | None:
        """Return the first cap that would be exceeded, or ``None`` if safe."""
        if self.limits.per_session is not None and (
            self.per_session_total + prospective_tokens > self.limits.per_session
        ):
            return "per_session"
        if self.limits.per_layer is not None:
            current_layer = self.per_layer_totals.get(layer_id, 0)
            if current_layer + prospective_tokens > self.limits.per_layer:
                return "per_layer"
        if self.limits.per_project is not None and (
            self.per_project_total + prospective_tokens > self.limits.per_project
        ):
            return "per_project"
        return None

    def _write_jsonl(self, layer_id: str, usage: TokenUsage, total: int, call_label: str) -> None:
        if self.usage_dir is None:
            return
        ts = datetime.now(timezone.utc)
        record = {
            "ts": ts.isoformat().replace("+00:00", "Z"),
            "layer": layer_id,
            "input": usage.input,
            "output": usage.output,
            "total": total,
            # Cache stats — zero unless provider populated them .
            "cache_creation_input": usage.cache_creation_input,
            "cache_read_input": usage.cache_read_input,
            "label": call_label,
        }
        import json

        line = json.dumps(record, ensure_ascii=False)
        path = self.usage_dir / f"{ts.strftime('%Y-%m-%d')}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        atomic_write(path, existing + line + "\n")


__all__ = ["CapName", "TokenLedger"]
