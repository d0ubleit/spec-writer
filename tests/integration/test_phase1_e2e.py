"""S1.14: Phase 1 smoke E2E — init + generate via CLI with stub producer.

Walks the full pipeline once on a tmp project: ``spec-writer init``
followed by ``spec-writer generate``. The draft producer is
monkey-patched to deterministically return one L1 spec — Phase 1
has no real LLM, so this stands in for the FakeLLM-scripted flow
the IMPLEMENTATION-PLAN calls out. Phase 2 swaps the stub for the
agent runtime + Claude provider while leaving the CLI surface
unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from spec_writer.interface.batch.cli import app
from spec_writer.types import Spec


async def _stub_producer(layer, _seed, _corpus, _methodology) -> list[Spec]:
    if layer.id == "L1":
        return [
            Spec(
                id="L1-VIS-001",
                type="vision",
                layer="L1",
                title="Phase 1 Smoke Vision",
                up_to=[],
                body="## Purpose\n\nE2E smoke spec.\n",
            )
        ]
    return []


@pytest.mark.integration
def test_phase1_e2e_init_then_generate(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()

    # 1. init
    result = runner.invoke(
        app,
        [
            "--project-root",
            str(tmp_path),
            "init",
            "--no-wizard",
            "--preset",
            "minimal",
        ],
    )
    assert result.exit_code == 0, result.output

    spec_editor = tmp_path / "spec-editor"
    assert spec_editor.is_dir()
    methodology_yaml = spec_editor / "methodology.yaml"
    assert methodology_yaml.exists()

    # 2. place a source document
    src = spec_editor / "sources-raw" / "biz.md"
    src.write_text("# Business idea\n\nA platform.", encoding="utf-8")

    # 3. inject the stub producer (signature: project_ctx, methodology, config)
    monkeypatch.setattr(
        "spec_writer.interface.batch.commands.generate.build_producer",
        lambda _p, _m, _c: _stub_producer,
    )

    # 4. generate
    result = runner.invoke(
        app,
        [
            "--project-root",
            str(tmp_path),
            "generate",
            "--layer",
            "L1",
            "--max-iterations",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "converged" in result.output.lower() or "OK" in result.output

    # 5a. source sync ran
    assert (spec_editor / "sources" / "biz.md").exists()

    # 5b. spec landed on disk
    spec_path = spec_editor / "specs" / "vision" / "phase-1-smoke-vision-VIS-001.md"
    assert spec_path.exists()
    content = spec_path.read_text(encoding="utf-8")
    assert content.startswith("---\nid: L1-VIS-001\n")
    assert 'title: "Phase 1 Smoke Vision"' in content
    assert "## Purpose" in content
