"""Refine planning — preview validator findings, group them, pick subsets.

When the corpus has many issues the user often wants to address them
one bucket at a time. ``build_refine_plan`` runs validators on the
current corpus and produces a per-spec list of error/warning groups
the chat REPL can then enumerate. ``select_plan_entries`` parses a
user's selection string ("1,3-5", "all", "warnings") and returns the
filtered subset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.methodology.validators import ValidationIssue
from spec_writer.methodology.validators.runner import run_validators
from spec_writer.types import Spec


def _empty_issues() -> list[ValidationIssue]:
    return []


def _empty_entries() -> list[RefinePlanEntry]:
    return []


@dataclass(frozen=True)
class RefinePlanEntry:
    """A planned bucket of work — one spec id with its issues."""

    spec_id: str | None
    """``None`` for corpus-level (cross-spec) issues that don't pin a single id."""

    issues: list[ValidationIssue] = field(default_factory=_empty_issues)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def severity_label(self) -> str:
        if self.has_errors:
            return "error"
        return "warning" if self.issues else "info"


@dataclass(frozen=True)
class RefinePlan:
    """Ordered preview of refine work — errors first, then warnings."""

    entries: list[RefinePlanEntry] = field(default_factory=_empty_entries)

    def render(self) -> str:
        if not self.entries:
            return "(corpus passes all validators)"
        lines: list[str] = ["# Refine plan"]
        for idx, entry in enumerate(self.entries, start=1):
            target = entry.spec_id or "(corpus-level)"
            lines.append(f"{idx}. [{entry.severity_label}] {target} — {len(entry.issues)} issue(s)")
            for issue in entry.issues[:5]:
                lines.append(f"     [{issue.code}] {issue.message}")
            if len(entry.issues) > 5:
                lines.append(f"     … and {len(entry.issues) - 5} more")
        return "\n".join(lines)


def build_refine_plan(
    specs: list[Spec],
    methodology: ResolvedMethodology,
) -> RefinePlan:
    """Run validators and group the result into per-spec buckets, errors first."""
    issues = run_validators(specs, methodology)
    buckets: dict[str | None, list[ValidationIssue]] = {}
    for issue in issues:
        buckets.setdefault(issue.spec_id, []).append(issue)

    def _sort_key(item: tuple[str | None, list[ValidationIssue]]) -> tuple[int, str]:
        spec_id, group = item
        has_error = any(i.severity == "error" for i in group)
        return (0 if has_error else 1, spec_id or "~")

    entries = [
        RefinePlanEntry(spec_id=spec_id, issues=group)
        for spec_id, group in sorted(buckets.items(), key=_sort_key)
    ]
    return RefinePlan(entries=entries)


def select_plan_entries(plan: RefinePlan, selection: str) -> list[RefinePlanEntry]:
    """Parse ``selection`` against ``plan.entries`` and return chosen subset.

    Supported syntax:
      - ``"all"`` — every entry.
      - ``"errors"`` — entries that contain at least one error issue.
      - ``"warnings"`` — entries with no errors but at least one issue.
      - ``"1,3,5"`` — explicit 1-based indices.
      - ``"1-3"`` — ranges (inclusive).
      - combinations: ``"1-3,7,errors"``.

    Unknown tokens are silently ignored — the caller can compare counts.
    """
    selection = selection.strip().lower()
    if not selection or selection == "none":
        return []
    chosen: list[RefinePlanEntry] = []
    seen_ids: set[int] = set()
    tokens = [t.strip() for t in selection.split(",") if t.strip()]
    for token in tokens:
        if token == "all":
            return list(plan.entries)
        if token == "errors":
            for idx, entry in enumerate(plan.entries):
                if entry.has_errors and idx not in seen_ids:
                    seen_ids.add(idx)
                    chosen.append(entry)
            continue
        if token == "warnings":
            for idx, entry in enumerate(plan.entries):
                if (not entry.has_errors) and entry.issues and idx not in seen_ids:
                    seen_ids.add(idx)
                    chosen.append(entry)
            continue
        if "-" in token:
            left, _, right = token.partition("-")
            try:
                lo = int(left)
                hi = int(right)
            except ValueError:
                continue
            for n in range(min(lo, hi), max(lo, hi) + 1):
                idx = n - 1
                if 0 <= idx < len(plan.entries) and idx not in seen_ids:
                    seen_ids.add(idx)
                    chosen.append(plan.entries[idx])
            continue
        try:
            idx = int(token) - 1
        except ValueError:
            continue
        if 0 <= idx < len(plan.entries) and idx not in seen_ids:
            seen_ids.add(idx)
            chosen.append(plan.entries[idx])
    return chosen


def scope_spec_ids(entries: list[RefinePlanEntry]) -> list[str]:
    """Return the unique non-None spec ids referenced by ``entries``."""
    out: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.spec_id and entry.spec_id not in seen:
            seen.add(entry.spec_id)
            out.append(entry.spec_id)
    return out


__all__ = [
    "RefinePlan",
    "RefinePlanEntry",
    "build_refine_plan",
    "scope_spec_ids",
    "select_plan_entries",
]
