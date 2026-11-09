"""Format validation issues as structured feedback for the agent.

When validators reject a draft, the runtime sends this string back
to the model so it can produce a corrective ``patch_spec`` call
instead of regenerating from scratch (principle 9 + docs/architecture.md).
"""

from __future__ import annotations

from spec_writer.methodology.validators import ValidationIssue


def format_validation_feedback(issues: list[ValidationIssue]) -> str:
    """Render ``issues`` as a single human-and-LLM-readable string."""
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    parts: list[str] = []
    if errors:
        parts.append("Validation failed. The following ERRORS must be fixed:")
        for issue in errors:
            line = f"- [{issue.code}] "
            if issue.spec_id:
                line += f"{issue.spec_id}"
                if issue.field:
                    line += f".{issue.field}"
                line += ": "
            line += issue.message
            if issue.fix_hint:
                line += f"\n  fix: {issue.fix_hint}"
            parts.append(line)
        parts.append(
            "\nUSE `patch_spec(spec_id, changes)` to fix each issue. "
            "DO NOT regenerate the whole spec — preserve everything else as-is."
        )
    if warnings:
        parts.append("\nWarnings (non-blocking, but consider fixing):")
        for issue in warnings:
            parts.append(
                f"- [{issue.code}] "
                + (f"{issue.spec_id}: " if issue.spec_id else "")
                + issue.message
            )
    return "\n".join(parts)


__all__ = ["format_validation_feedback"]
