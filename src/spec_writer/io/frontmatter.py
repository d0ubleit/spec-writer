"""Canonical frontmatter formatter + round-trip parser.

Implements the rules from docs/methodology.md:

- Field order: engine-mandatory → vertical links → code reference →
  methodology-mandatory (declaration order) → conventional → user-extra
  (alphabetical).
- Lists: always flow ``[a, b, c]``, never block.
- ``title`` always double-quoted; other strings bare unless they
  contain YAML reserved characters.
- Empty list → ``[]``; never ``null`` or absent.
- Numbers, booleans, ISO timestamps: bare.
- ``relates_to`` entries and ``provenance`` rendered as inline flow
  mappings.
- Indentation: 2 spaces (only inside flow values, never block).
- UTF-8 without BOM.

Reading accepts any valid YAML (block or flow). ``write_canonical``
is non-destructive in content — it only normalizes the format.
"""

from __future__ import annotations

import re
from typing import Any, Final, cast

import yaml

# ── ordering buckets ─────────────────────────────────────────────────────

_ENGINE_MANDATORY: Final[tuple[str, ...]] = ("id", "type", "layer", "title")
_VERTICAL_LINKS: Final[tuple[str, ...]] = ("up_to", "down_to")
_CODE_REFERENCE: Final[tuple[str, ...]] = ("implemented_by",)
_CONVENTIONAL: Final[tuple[str, ...]] = ("tags", "status", "provenance", "relates_to")

_FRONTMATTER_BLOCK: Final[re.Pattern[str]] = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)

# YAML reserved chars that force quoting when present in a scalar.
_RESERVED_CHARS: Final[str] = ":[]{}&*!|>'\"%@,#`?"

# ISO-8601 datetimes/dates that should stay bare (YAML parses them back).
_ISO_DATETIME: Final[re.Pattern[str]] = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?Z?)?$"
)

# A value is rendered bare only when it is a single identifier-like
# token (kebab-case lowercase, underscores allowed). Anything with
# spaces, dots, slashes, etc. is double-quoted for readability —
# matches the style of docs/methodology.md examples.
_SAFE_BARE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


# ── public API ───────────────────────────────────────────────────────────


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Split ``---\\nYAML\\n---\\nBODY`` into ``(frontmatter_dict, body)``."""
    m = _FRONTMATTER_BLOCK.match(content)
    if m is None:
        return {}, content
    raw = yaml.safe_load(m.group(1))
    if not isinstance(raw, dict):
        return {}, m.group(2)
    return cast(dict[str, Any], raw), m.group(2)


def write_canonical(
    frontmatter: dict[str, Any],
    body: str,
    methodology_required_order: list[str] | None = None,
) -> str:
    """Render ``frontmatter`` + ``body`` as a canonical markdown file."""
    methodology_order = list(methodology_required_order or [])
    ordered = _order_fields(frontmatter, methodology_order)
    yaml_lines = [f"{name}: {_render_value(name, value)}" for name, value in ordered]
    yaml_block = "\n".join(yaml_lines)
    if yaml_block:
        yaml_block += "\n"
    body_block = body.lstrip("\n")
    return f"---\n{yaml_block}---\n\n{body_block}"


# ── internals ────────────────────────────────────────────────────────────


def _order_fields(
    fields: dict[str, Any],
    methodology_order: list[str],
) -> list[tuple[str, Any]]:
    known: set[str] = set(_ENGINE_MANDATORY + _VERTICAL_LINKS + _CODE_REFERENCE + _CONVENTIONAL)
    known.update(methodology_order)

    ordered: list[tuple[str, Any]] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name in fields and name not in seen:
            ordered.append((name, fields[name]))
            seen.add(name)

    for group in (_ENGINE_MANDATORY, _VERTICAL_LINKS, _CODE_REFERENCE):
        for name in group:
            add(name)
    for name in methodology_order:
        add(name)
    for name in _CONVENTIONAL:
        add(name)

    user_extra = sorted(k for k in fields if k not in known and k not in seen)
    for name in user_extra:
        add(name)

    return ordered


def _render_value(name: str, value: Any) -> str:
    if name == "title":
        return _double_quote(str(value))
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, str):
        return _format_string(value)
    if isinstance(value, list):
        return _format_list(cast(list[Any], value))
    if isinstance(value, dict):
        return _format_mapping(cast(dict[str, Any], value))
    return _double_quote(str(value))


def _format_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        # keep decimal form so YAML round-trips to float
        return f"{value:.1f}"
    return str(value)


def _double_quote(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_string(value: str) -> str:
    if value == "":
        return '""'
    if _ISO_DATETIME.match(value):
        return value
    if any(c in _RESERVED_CHARS for c in value):
        return _double_quote(value)
    if value != value.strip():
        return _double_quote(value)
    if _SAFE_BARE.match(value):
        return value
    return _double_quote(value)


def _format_scalar_inline(value: Any) -> str:
    """Render a scalar inside a flow context (list / mapping value)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, str):
        return _format_string(value)
    if isinstance(value, list):
        return _format_list(cast(list[Any], value))
    if isinstance(value, dict):
        return _format_mapping(cast(dict[str, Any], value))
    return _double_quote(str(value))


def _format_list(value: list[Any]) -> str:
    if not value:
        return "[]"
    rendered = [_format_scalar_inline(item) for item in value]
    return "[" + ", ".join(rendered) + "]"


def _format_mapping(value: dict[str, Any]) -> str:
    if not value:
        return "{}"
    parts = [f"{k}: {_format_scalar_inline(v)}" for k, v in value.items()]
    return "{" + ", ".join(parts) + "}"


__all__ = ["parse_frontmatter", "write_canonical"]
