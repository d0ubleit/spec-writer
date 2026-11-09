"""LanguageAnalyzer Protocol + analyzer registry.

REV-extraction primitives. The agent calls the
``discover_code_inventory`` MCP tool, which dispatches each file in
``code_scope.includes`` to an analyzer based on file extension; the
analyzer returns a best-effort list of :class:`Symbol`-s the agent
can navigate with ``Read`` / ``Grep``.

**Design notes:**

- Inventory is **best-effort**, not source-of-truth. Missing a
  symbol is acceptable; the agent walks the filesystem too.
- Analyzers are **stateless** functions wrapped in a small dataclass
  so they can be discovered via the registry.
- No tree-sitter / language SDK dependency — regex over file lines.
  Faster, fewer deps, language-portable (you can add a new language
  in 30 lines).
- Fallback :class:`FilesystemOnlyAnalyzer` returns one ``file``-kind
  symbol per file, so coverage stays meaningful on unrecognised
  extensions.
"""

# Importing these modules registers each analyzer at import time.
from spec_writer.agent.code_analyzers import php, python, typescript
from spec_writer.agent.code_analyzers.protocol import (
    FilesystemOnlyAnalyzer,
    LanguageAnalyzer,
    Symbol,
    SymbolKind,
    Visibility,
)
from spec_writer.agent.code_analyzers.registry import (
    analyzer_for,
    register_analyzer,
    registered_analyzers,
)

del php, python, typescript  # re-export not needed; import-only for side effects

__all__ = [
    "FilesystemOnlyAnalyzer",
    "LanguageAnalyzer",
    "Symbol",
    "SymbolKind",
    "Visibility",
    "analyzer_for",
    "register_analyzer",
    "registered_analyzers",
]
