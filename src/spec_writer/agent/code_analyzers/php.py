"""PHP analyzer — regex-based, brace-balanced end-line estimate.

Catches ``function``, ``class``, ``interface``, ``trait`` declarations.
Visibility comes from the declaration prefix (``public`` / ``protected``
/ ``private``) when present; bare ``function`` defaults to ``public``.
"""

from __future__ import annotations

import re
from pathlib import Path

from spec_writer.agent.code_analyzers.protocol import Symbol, SymbolKind, Visibility
from spec_writer.agent.code_analyzers.registry import register_analyzer

_DECL_RE = re.compile(
    r"^\s*"
    r"(?:(?P<vis>public|protected|private)\s+)?"
    r"(?:(?:static|final|abstract|readonly)\s+)*"
    r"(?P<kw>function|class|interface|trait)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)


def _visibility_for(prefix: str | None, name: str) -> Visibility:
    if prefix == "public":
        return "public"
    if prefix == "protected":
        return "protected"
    if prefix == "private":
        return "private"
    return "internal" if name.startswith("_") else "public"


def _kind_for(keyword: str) -> SymbolKind:
    mapping: dict[str, SymbolKind] = {
        "function": "function",
        "class": "class",
        "interface": "interface",
        "trait": "trait",
    }
    return mapping[keyword]


def _balanced_end(lines: list[str], start_idx: int) -> int:
    """1-based last line of the block opened at ``start_idx`` (brace match)."""
    depth = 0
    seen_open = False
    for cursor in range(start_idx, len(lines)):
        line = lines[cursor]
        for char in line:
            if char == "{":
                depth += 1
                seen_open = True
            elif char == "}":
                depth -= 1
                if seen_open and depth == 0:
                    return cursor + 1
    return len(lines)


class PhpAnalyzer:
    language: str = "php"
    supported_extensions: list[str] = [".php"]  # noqa: RUF012 — registry-keyed only

    def discover_symbols(self, path: Path, source: str) -> list[Symbol]:
        del path
        lines = source.splitlines()
        out: list[Symbol] = []
        inside_class = False
        class_close_line = 0

        for idx, line in enumerate(lines):
            if inside_class and (idx + 1) > class_close_line:
                inside_class = False

            match = _DECL_RE.match(line)
            if match is None:
                continue
            keyword = match.group("kw")
            name = match.group("name")
            kind = _kind_for(keyword)
            visibility = _visibility_for(match.group("vis"), name)
            start_line = idx + 1
            end_line = _balanced_end(lines, idx)
            if keyword == "function" and inside_class:
                kind = "method"

            out.append(
                Symbol(
                    name=name,
                    kind=kind,
                    visibility=visibility,
                    start_line=start_line,
                    end_line=end_line,
                    signature=line.strip().rstrip("{").rstrip(),
                )
            )

            if keyword in {"class", "interface", "trait"}:
                inside_class = True
                class_close_line = end_line
        return out


register_analyzer(PhpAnalyzer())


__all__ = ["PhpAnalyzer"]
