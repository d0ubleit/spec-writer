"""TypeScript / JavaScript / TSX / JSX analyzer.

Recognises:
- ``export?`` (function|class|interface|enum|type|const) declarations
- ``async function`` variants
- exported arrow-function consts (``export const x = () => ...``)

Visibility is binary: exported → ``public``; module-local → ``internal``.
"""

from __future__ import annotations

import re
from pathlib import Path

from spec_writer.agent.code_analyzers.protocol import Symbol, SymbolKind, Visibility
from spec_writer.agent.code_analyzers.registry import register_analyzer

_DECL_RE = re.compile(
    r"^\s*"
    r"(?P<export>export(?:\s+default)?\s+)?"
    r"(?:async\s+)?"
    r"(?P<kw>function|class|interface|enum|type|const|let|var)\s+"
    r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)"
)


_KIND_MAP: dict[str, SymbolKind] = {
    "function": "function",
    "class": "class",
    "interface": "interface",
    "enum": "enum",
    "type": "type",
    "const": "const",
    "let": "const",
    "var": "const",
}


def _balanced_end(lines: list[str], start_idx: int) -> int:
    """End line of a braced block; falls back to single-line if no braces."""
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
    return start_idx + 1


class TypeScriptAnalyzer:
    language: str = "typescript"
    supported_extensions: list[str] = [  # noqa: RUF012 — registry-keyed only
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
    ]

    def discover_symbols(self, path: Path, source: str) -> list[Symbol]:
        del path
        lines = source.splitlines()
        out: list[Symbol] = []

        for idx, line in enumerate(lines):
            match = _DECL_RE.match(line)
            if match is None:
                continue
            keyword = match.group("kw")
            name = match.group("name")
            exported = bool(match.group("export"))
            visibility: Visibility = "public" if exported else "internal"

            kind = _KIND_MAP.get(keyword, "function")
            # const/let/var only count when they're exported (top-level API)
            # and look like arrow functions — otherwise too noisy.
            if keyword in {"const", "let", "var"}:
                if not exported:
                    continue
                if "=>" not in line and "function" not in line:
                    continue
                kind = "function"

            start_line = idx + 1
            end_line = _balanced_end(lines, idx)

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
        return out


register_analyzer(TypeScriptAnalyzer())


__all__ = ["TypeScriptAnalyzer"]
