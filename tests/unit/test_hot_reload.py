"""S3.8 — methodology hot-reload diff + /reload slash command."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from spec_writer.agent.chat_session import ChatSession
from spec_writer.engine.project_context import ProjectContext
from spec_writer.interface.batch.cli import app
from spec_writer.interface.batch.commands.chat import ChatContext, _cmd_reload
from spec_writer.methodology.hot_reload import diff_methodologies
from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.sources import FilesystemMixinSource

# ── diff_methodologies ──────────────────────────────────────────────────


def _methodology(tmp_path: Path, mixins: list[str]):
    meta = tmp_path / "meta"
    meta.mkdir(exist_ok=True)
    (meta / "base.yaml").write_text(
        (
            "version: 1\n"
            "layers:\n"
            "  - id: L1\n    role: source-boundary\n    name: vision\n"
            "  - id: L2\n    role: middle\n    name: capabilities\n"
        ),
        encoding="utf-8",
    )
    mixins_dir = meta / "mixins"
    mixins_dir.mkdir(exist_ok=True)
    (mixins_dir / "vision.yaml").write_text(
        "mixin: vision\nadds_types:\n  - id: vision\n    layer_role: source-boundary\n    prefix: VIS\n",
        encoding="utf-8",
    )
    (mixins_dir / "cap.yaml").write_text(
        "mixin: cap\nadds_types:\n  - id: capability\n    layer_name: capabilities\n    prefix: CAP\n",
        encoding="utf-8",
    )
    (mixins_dir / "gate.yaml").write_text(
        "mixin: gate\nmetrics:\n  - id: m1\n    layer: L2\n    func_ref: metrics.spec_count_at_layer\n",
        encoding="utf-8",
    )
    return load_methodology(
        meta / "base.yaml", [FilesystemMixinSource(mixins_dir)], active_mixins=mixins
    )


def test_diff_empty_when_identical(tmp_path: Path) -> None:
    a = _methodology(tmp_path, ["vision", "cap"])
    b = _methodology(tmp_path, ["vision", "cap"])
    diff = diff_methodologies(a, b)
    assert diff.is_empty


def test_diff_detects_added_type(tmp_path: Path) -> None:
    a = _methodology(tmp_path, ["vision"])
    b = _methodology(tmp_path, ["vision", "cap"])
    diff = diff_methodologies(a, b)
    assert "capability" in diff.types_added
    assert diff.types_removed == []


def test_diff_detects_removed_type(tmp_path: Path) -> None:
    a = _methodology(tmp_path, ["vision", "cap"])
    b = _methodology(tmp_path, ["vision"])
    diff = diff_methodologies(a, b)
    assert "capability" in diff.types_removed


def test_diff_detects_added_metric(tmp_path: Path) -> None:
    a = _methodology(tmp_path, ["vision", "cap"])
    b = _methodology(tmp_path, ["vision", "cap", "gate"])
    diff = diff_methodologies(a, b)
    assert "m1" in diff.metric_ids_added


def test_render_empty_says_no_changes(tmp_path: Path) -> None:
    a = _methodology(tmp_path, ["vision"])
    diff = diff_methodologies(a, a)
    assert "no methodology changes" in diff.render()


def test_render_summarises_changes(tmp_path: Path) -> None:
    a = _methodology(tmp_path, ["vision"])
    b = _methodology(tmp_path, ["vision", "cap", "gate"])
    diff = diff_methodologies(a, b)
    rendered = diff.render()
    assert "types" in rendered
    assert "capability" in rendered
    assert "metrics" in rendered


# ── /reload chat command ────────────────────────────────────────────────


def _bootstrap(tmp_path: Path) -> None:
    runner = CliRunner()
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
    assert result.exit_code == 0


def _chat_ctx(tmp_path: Path) -> ChatContext:
    project_ctx = ProjectContext.from_cwd(tmp_path)
    session = ChatSession(project_ctx=project_ctx)
    return ChatContext(
        project_ctx=project_ctx, session=session, default_layers=[], max_iterations=5
    )


def test_cmd_reload_records_first_load(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path)
    cctx.methodology = None  # simulate not-yet-loaded
    rc = _cmd_reload(cctx, [])
    assert rc == 0
    assert cctx.methodology is not None
    out = capsys.readouterr().out
    assert "first time" in out


def test_cmd_reload_reports_no_changes_when_idempotent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path)
    # first reload primes the context
    _cmd_reload(cctx, [])
    capsys.readouterr()
    _cmd_reload(cctx, [])
    out = capsys.readouterr().out
    assert "no methodology changes" in out


def test_cmd_reload_detects_edit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path)
    _cmd_reload(cctx, [])
    capsys.readouterr()

    # Append another mixin to methodology.yaml — minimal preset doesn't have
    # 'capabilities', so adding it should show up as a type delta.
    methodology_path = tmp_path / "spec-editor" / "methodology.yaml"
    text = methodology_path.read_text(encoding="utf-8")
    methodology_path.write_text(text + "  - capabilities\n", encoding="utf-8")

    _cmd_reload(cctx, [])
    out = capsys.readouterr().out
    assert "types" in out
    assert "capability" in out


def test_cmd_reload_records_session_event(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    cctx = _chat_ctx(tmp_path)
    _cmd_reload(cctx, [])
    import json

    payload = json.loads(cctx.session.path.read_text(encoding="utf-8"))
    notes = [e for e in payload["events"] if e["kind"] == "system-note"]
    assert any("methodology" in str(n).lower() for n in notes)


def test_reload_appears_in_help(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat"],
        input="/help\n/quit\n",
    )
    assert "/reload" in result.output
