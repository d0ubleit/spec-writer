"""Python analyzer — regex-based, no AST.

Recognises module-level ``def`` / ``async def`` / ``class`` declarations
with a coarse end-line estimate (next line at indentation <= the
declaration's indentation, or EOF). Methods inside a class are nested
under their containing class; the class still appears in the symbol
list, and methods carry ``kind=method``.

Visibility is inferred from the leading underscore convention:
``_name`` → ``private``, otherwise ``public``.
"""

from __future__ import annotations

import re
from pathlib import Path

from spec_writer.agent.code_analyzers.protocol import Symbol, SymbolKind, Visibility
from spec_writer.agent.code_analyzers.registry import register_analyzer

_DECL_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<kw>async\s+def|def|class)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)


def _visibility(name: str) -> Visibility:
    if name.startswith("__") and not name.endswith("__"):
        return "private"
    if name.startswith("_"):
        return "internal"
    return "public"


def _indent_width(indent: str) -> int:
    return len(indent.replace("\t", "    "))


def _block_end(lines: list[str], start_idx: int, indent_width: int) -> int:
    """1-based end line of the block opened at ``start_idx``.

    Stops at the first line whose indent is <= ``indent_width`` and which
    is non-blank / non-comment. ``start_idx`` is 0-based.
    """
    for cursor in range(start_idx + 1, len(lines)):
        line = lines[cursor]
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if _indent_width(line[: len(line) - len(stripped)]) <= indent_width:
            return cursor  # 0-based of next-outside == 1-based of last block line
    return len(lines)


class PythonAnalyzer:
    language: str = "python"
    supported_extensions: list[str] = [".py"]  # noqa: RUF012 — analyzer registry-keyed only

    def discover_symbols(self, path: Path, source: str) -> list[Symbol]:
        del path
        lines = source.splitlines()
        out: list[Symbol] = []
        # Stack of (indent_width, kind) — tracks the enclosing class for methods.
        class_stack: list[tuple[int, str]] = []

        for idx, line in enumerate(lines):
            match = _DECL_RE.match(line)
            if match is None:
                continue
            indent_w = _indent_width(match.group("indent"))
            keyword = match.group("kw")
            name = match.group("name")

            while class_stack and class_stack[-1][0] >= indent_w:
                class_stack.pop()

            start_line = idx + 1
            end_line = _block_end(lines, idx, indent_w)
            signature = line.strip().rstrip(":")

            inside_class = bool(class_stack)
            kind: SymbolKind
            if keyword == "class":
                kind = "class"
                class_stack.append((indent_w, name))
            else:
                kind = "method" if inside_class else "function"

            out.append(
                Symbol(
                    name=name,
                    kind=kind,
                    visibility=_visibility(name),
                    start_line=start_line,
                    end_line=end_line,
                    signature=signature,
                )
            )
        return out


register_analyzer(PythonAnalyzer())


__all__ = ["PythonAnalyzer"]
