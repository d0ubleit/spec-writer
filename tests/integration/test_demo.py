"""End-to-end smoke test for ``spec-writer demo`` (offline, no LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from spec_writer.interface.batch.cli import app


@pytest.mark.integration
def test_demo_runs_end_to_end_offline(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["--project-root", str(tmp_path), "demo"])

    assert result.exit_code == 0, result.output
    assert "Pipeline: OK" in result.output
    assert "L1-VIS-001" in result.output
    assert "L2-CAP-001" in result.output
    assert "L3-up-to-completeness" not in result.output  # empty layers are hidden

    specs = sorted((tmp_path / "demo-output" / "spec-editor" / "specs").rglob("*.md"))
    assert [path.name for path in specs] == [
        "manage-the-customer-lifecycle-CAP-001.md",
        "mini-crm-one-place-for-every-customer-record-VIS-001.md",
    ]


@pytest.mark.integration
def test_demo_refuses_to_delete_unrelated_directory(tmp_path: Path) -> None:
    unrelated = tmp_path / "demo-output"
    unrelated.mkdir()
    (unrelated / "important.txt").write_text("keep me", encoding="utf-8")

    result = CliRunner().invoke(app, ["--project-root", str(tmp_path), "demo"])

    assert result.exit_code != 0
    assert (unrelated / "important.txt").exists()
