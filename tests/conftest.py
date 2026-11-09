"""Shared pytest fixtures for spec-writer V2 test suite.

Story-specific fixtures live next to the tests that use them; this file
holds fixtures shared across multiple stories (tmp_project, fake_llm
factory, sample methodology, etc.). Phase 1 populates them as stories
require — S1.0 only needs the file to exist for pytest discovery.
"""

from __future__ import annotations
