"""Hot-reload helper — diff two ResolvedMethodology snapshots.

The chat REPL keeps no persistent methodology reference (every
slash command re-loads), but the user wants to see *what changed*
when they edit ``methodology.yaml`` or a mixin file mid-session. This
module produces a compact :class:`MethodologyDiff` between two
snapshots so the REPL can print it.

We don't try to be exhaustive — only the surface attributes a user
typically twiddles: layers, spec types, validator names, metric ids,
hook points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spec_writer.methodology.registry import ResolvedMethodology


def _empty_str_list() -> list[str]:
    return []


@dataclass(frozen=True)
class MethodologyDiff:
    layers_added: list[str] = field(default_factory=_empty_str_list)
    layers_removed: list[str] = field(default_factory=_empty_str_list)
    types_added: list[str] = field(default_factory=_empty_str_list)
    types_removed: list[str] = field(default_factory=_empty_str_list)
    validator_ids_added: list[str] = field(default_factory=_empty_str_list)
    validator_ids_removed: list[str] = field(default_factory=_empty_str_list)
    metric_ids_added: list[str] = field(default_factory=_empty_str_list)
    metric_ids_removed: list[str] = field(default_factory=_empty_str_list)
    hook_points_added: list[str] = field(default_factory=_empty_str_list)
    hook_points_removed: list[str] = field(default_factory=_empty_str_list)

    @property
    def is_empty(self) -> bool:
        return not any(
            getattr(self, attr)
            for attr in (
                "layers_added",
                "layers_removed",
                "types_added",
                "types_removed",
                "validator_ids_added",
                "validator_ids_removed",
                "metric_ids_added",
                "metric_ids_removed",
                "hook_points_added",
                "hook_points_removed",
            )
        )

    def render(self) -> str:
        if self.is_empty:
            return "(no methodology changes)"
        lines: list[str] = []
        for label, added, removed in (
            ("layers", self.layers_added, self.layers_removed),
            ("types", self.types_added, self.types_removed),
            ("validators", self.validator_ids_added, self.validator_ids_removed),
            ("metrics", self.metric_ids_added, self.metric_ids_removed),
            ("hook points", self.hook_points_added, self.hook_points_removed),
        ):
            if added:
                lines.append(f"+ {label}: {', '.join(sorted(added))}")
            if removed:
                lines.append(f"- {label}: {', '.join(sorted(removed))}")
        return "\n".join(lines)


def _validator_ids(m: ResolvedMethodology) -> set[str]:
    ids: set[str] = set()
    for bucket in m.validators.values():
        for entry in bucket:
            vid = entry.get("id") or entry.get("func_ref")
            if isinstance(vid, str):
                ids.add(vid)
    return ids


def _metric_ids(m: ResolvedMethodology) -> set[str]:
    ids: set[str] = set()
    for bucket in m.metrics_by_layer.values():
        for entry in bucket:
            mid = entry.get("id") or entry.get("func_ref")
            if isinstance(mid, str):
                ids.add(mid)
    return ids


def diff_methodologies(
    before: ResolvedMethodology,
    after: ResolvedMethodology,
) -> MethodologyDiff:
    """Compare two snapshots and report the user-meaningful deltas."""
    before_layers = {layer.id for layer in before.layers}
    after_layers = {layer.id for layer in after.layers}

    before_types = set(before.spec_types)
    after_types = set(after.spec_types)

    before_vids = _validator_ids(before)
    after_vids = _validator_ids(after)

    before_mids = _metric_ids(before)
    after_mids = _metric_ids(after)

    before_hooks = set(before.hooks_by_point)
    after_hooks = set(after.hooks_by_point)

    return MethodologyDiff(
        layers_added=sorted(after_layers - before_layers),
        layers_removed=sorted(before_layers - after_layers),
        types_added=sorted(after_types - before_types),
        types_removed=sorted(before_types - after_types),
        validator_ids_added=sorted(after_vids - before_vids),
        validator_ids_removed=sorted(before_vids - after_vids),
        metric_ids_added=sorted(after_mids - before_mids),
        metric_ids_removed=sorted(before_mids - after_mids),
        hook_points_added=sorted(after_hooks - before_hooks),
        hook_points_removed=sorted(before_hooks - after_hooks),
    )


__all__ = ["MethodologyDiff", "diff_methodologies"]
