"""S1.0 setup test: verify all top-level packages import without error."""

from __future__ import annotations

import importlib


def test_root_package_imports() -> None:
    mod = importlib.import_module("spec_writer")
    assert hasattr(mod, "__version__")


def test_engine_package_imports() -> None:
    importlib.import_module("spec_writer.engine")


def test_agent_package_imports() -> None:
    importlib.import_module("spec_writer.agent")


def test_methodology_package_imports() -> None:
    importlib.import_module("spec_writer.methodology")


def test_io_package_imports() -> None:
    importlib.import_module("spec_writer.io")


def test_interface_package_imports() -> None:
    importlib.import_module("spec_writer.interface")
