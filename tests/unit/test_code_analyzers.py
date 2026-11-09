"""S4.1 + S4.2 — LanguageAnalyzer Protocol + Python / PHP / TS analyzers."""

from __future__ import annotations

from pathlib import Path

import spec_writer.agent.code_analyzers as ca  # noqa: F401 — registers analyzers
from spec_writer.agent.code_analyzers import (
    FilesystemOnlyAnalyzer,
    LanguageAnalyzer,
    Symbol,
    analyzer_for,
    registered_analyzers,
)

# ── registry + protocol ─────────────────────────────────────────────────


def test_protocol_runtime_check_filesystem() -> None:
    assert isinstance(FilesystemOnlyAnalyzer(), LanguageAnalyzer)


def test_analyzer_for_python_extension(tmp_path: Path) -> None:
    chosen = analyzer_for(tmp_path / "module.py")
    assert chosen.language == "python"


def test_analyzer_for_php_extension(tmp_path: Path) -> None:
    chosen = analyzer_for(tmp_path / "Controller.php")
    assert chosen.language == "php"


def test_analyzer_for_typescript_extension(tmp_path: Path) -> None:
    chosen = analyzer_for(tmp_path / "foo.tsx")
    assert chosen.language == "typescript"


def test_analyzer_for_unknown_extension_falls_back(tmp_path: Path) -> None:
    chosen = analyzer_for(tmp_path / "weird.zzz")
    assert chosen.language == "filesystem"


def test_registered_analyzers_includes_built_ins() -> None:
    by_ext = registered_analyzers()
    assert ".py" in by_ext
    assert ".php" in by_ext
    assert ".ts" in by_ext
    assert ".tsx" in by_ext
    assert ".js" in by_ext


# ── FilesystemOnlyAnalyzer ──────────────────────────────────────────────


def test_filesystem_analyzer_returns_one_symbol(tmp_path: Path) -> None:
    file = tmp_path / "binary.bin"
    file.write_bytes(b"\x00\x01\x02\n\n\n")
    analyzer = FilesystemOnlyAnalyzer()
    symbols = analyzer.discover_symbols(file, "")
    assert len(symbols) == 1
    assert symbols[0].kind == "file"
    assert symbols[0].name == "binary"


def test_filesystem_analyzer_counts_lines(tmp_path: Path) -> None:
    file = tmp_path / "data.txt"
    file.write_text("a\nb\nc\n")
    analyzer = FilesystemOnlyAnalyzer()
    symbols = analyzer.discover_symbols(file, "a\nb\nc\n")
    assert symbols[0].start_line == 1
    assert symbols[0].end_line == 3


# ── Python analyzer ─────────────────────────────────────────────────────


def _py_symbols(source: str) -> list[Symbol]:
    return analyzer_for(Path("x.py")).discover_symbols(Path("x.py"), source)


def test_python_discovers_module_function() -> None:
    syms = _py_symbols("def add(a, b):\n    return a + b\n")
    assert len(syms) == 1
    assert syms[0].name == "add"
    assert syms[0].kind == "function"
    assert syms[0].visibility == "public"
    assert syms[0].start_line == 1


def test_python_discovers_class_and_methods() -> None:
    source = (
        "class Foo:\n"
        "    def __init__(self):\n"
        "        self.x = 1\n"
        "\n"
        "    def bar(self):\n"
        "        return self.x\n"
    )
    syms = _py_symbols(source)
    names = {s.name: s for s in syms}
    assert "Foo" in names and names["Foo"].kind == "class"
    assert "__init__" in names and names["__init__"].kind == "method"
    assert "bar" in names and names["bar"].kind == "method"


def test_python_private_underscore_visibility() -> None:
    syms = _py_symbols("def _helper():\n    pass\n")
    assert syms[0].visibility == "internal"


def test_python_dunder_treated_as_private() -> None:
    syms = _py_symbols("def __secret():\n    pass\n")
    assert syms[0].visibility == "private"


def test_python_async_function() -> None:
    syms = _py_symbols("async def fetch():\n    return None\n")
    assert syms[0].name == "fetch"
    assert syms[0].kind == "function"


def test_python_end_line_includes_body() -> None:
    source = "def long():\n    a = 1\n    b = 2\n    return a + b\n\ndef other():\n    pass\n"
    syms = _py_symbols(source)
    long_sym = next(s for s in syms if s.name == "long")
    assert long_sym.start_line == 1
    assert long_sym.end_line >= 4  # body included


# ── PHP analyzer ────────────────────────────────────────────────────────


def _php_symbols(source: str) -> list[Symbol]:
    return analyzer_for(Path("x.php")).discover_symbols(Path("x.php"), source)


def test_php_class_and_method() -> None:
    source = (
        "<?php\n"
        "class AuthController {\n"
        "    public function login() {\n"
        "        return true;\n"
        "    }\n"
        "    private function _hash($pw) {\n"
        "        return md5($pw);\n"
        "    }\n"
        "}\n"
    )
    syms = _php_symbols(source)
    names = {s.name: s for s in syms}
    assert "AuthController" in names and names["AuthController"].kind == "class"
    assert "login" in names and names["login"].kind == "method"
    assert names["login"].visibility == "public"
    assert "_hash" in names and names["_hash"].visibility == "private"


def test_php_interface_detection() -> None:
    syms = _php_symbols("<?php\ninterface UserRepository {\n  public function find(int $id);\n}\n")
    interfaces = [s for s in syms if s.kind == "interface"]
    assert len(interfaces) == 1
    assert interfaces[0].name == "UserRepository"


def test_php_freestanding_function() -> None:
    syms = _php_symbols(
        "<?php\nfunction slugify(string $s): string {\n  return strtolower($s);\n}\n"
    )
    funcs = [s for s in syms if s.kind == "function"]
    assert len(funcs) == 1
    assert funcs[0].name == "slugify"


# ── TypeScript analyzer ─────────────────────────────────────────────────


def _ts_symbols(source: str, ext: str = ".ts") -> list[Symbol]:
    path = Path(f"x{ext}")
    return analyzer_for(path).discover_symbols(path, source)


def test_typescript_exported_function() -> None:
    syms = _ts_symbols("export function foo() { return 1; }\n")
    assert len(syms) == 1
    assert syms[0].name == "foo"
    assert syms[0].kind == "function"
    assert syms[0].visibility == "public"


def test_typescript_non_exported_internal_visibility() -> None:
    syms = _ts_symbols("function helper() { return 1; }\n")
    assert syms[0].visibility == "internal"


def test_typescript_interface_and_type() -> None:
    source = "export interface User { id: string; }\nexport type Id = string;\n"
    syms = _ts_symbols(source)
    kinds = {s.name: s.kind for s in syms}
    assert kinds["User"] == "interface"
    assert kinds["Id"] == "type"


def test_typescript_tsx_extension_supported() -> None:
    syms = _ts_symbols("export function Component() { return null; }\n", ext=".tsx")
    assert len(syms) == 1


def test_typescript_arrow_const_exported() -> None:
    syms = _ts_symbols("export const add = (a: number, b: number) => a + b;\n")
    assert len(syms) == 1
    assert syms[0].name == "add"
    assert syms[0].kind == "function"


def test_typescript_unexported_const_skipped() -> None:
    # bare consts without export are too noisy; analyzer skips them
    syms = _ts_symbols("const config = { foo: 1 };\n")
    assert syms == []


# ── failure-mode safety ─────────────────────────────────────────────────


def test_analyzer_never_raises_on_empty_source(tmp_path: Path) -> None:
    for ext in (".py", ".php", ".ts", ".tsx", ".js", ".unknown"):
        path = tmp_path / f"empty{ext}"
        path.write_text("")
        analyzer = analyzer_for(path)
        # Empty source must return empty list (or, for FS fallback, one
        # file-symbol with end_line=-1). Either way: no exception.
        result = analyzer.discover_symbols(path, "")
        assert isinstance(result, list)
