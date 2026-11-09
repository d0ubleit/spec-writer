"""S1.0 setup test: pytest discovers and runs at least one test.

The mere fact this test runs proves discovery works — the assertion
itself is a placeholder so the run yields a green dot.
"""

from __future__ import annotations


def test_pytest_discovery_smoke() -> None:
    assert True
