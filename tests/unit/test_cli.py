"""S1.13: CLI entry point + init subcommand."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from spec_writer.interface.batch.cli import app


def _run(args: list[str], **kwargs):
    runner = CliRunner()
    return runner.invoke(app, args, **kwargs)


def test_help_exits_zero() -> None:
    result = _run(["--help"])
    assert result.exit_code == 0
    assert "spec-writer" in result.output


def test_help_lists_all_subcommands() -> None:
    result = _run(["--help"])
    for cmd in (
        "init",
        "generate",
        "chat",
        "refine",
        "sync",
        "validate",
        "metrics",
        "prompt-debug",
        "status",
        "reset",
    ):
        assert cmd in result.output


def test_placeholder_subcommand_exits_two() -> None:
    result = _run(["sync"])
    assert result.exit_code == 2
    assert "Phase" in result.output or "Phase" in (result.stderr or "")


def test_init_no_wizard_minimal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = _run(
        [
            "--project-root",
            str(tmp_path),
            "init",
            "--no-wizard",
            "--preset",
            "minimal",
        ]
    )
    assert result.exit_code == 0
    spec_editor = tmp_path / "spec-editor"
    assert spec_editor.is_dir()
    for sub in ("sources-raw", "sources", "specs", "_meta"):
        assert (spec_editor / sub).is_dir()
    methodology = spec_editor / "methodology.yaml"
    assert methodology.exists()
    assert "preset: minimal" in methodology.read_text(encoding="utf-8")


def test_init_wizard_uses_default_preset(tmp_path: Path, monkeypatch) -> None:
    """With stdin '\\n', the wizard accepts its proposed default."""
    monkeypatch.chdir(tmp_path)
    result = _run(
        ["--project-root", str(tmp_path), "init"],
        input="\n",
    )
    assert result.exit_code == 0
    methodology = (tmp_path / "spec-editor" / "methodology.yaml").read_text(encoding="utf-8")
    assert "preset:" in methodology


def test_init_auto_detect_backend(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    result = _run(
        [
            "--project-root",
            str(tmp_path),
            "init",
            "--no-wizard",
        ]
    )
    assert result.exit_code == 0
    methodology = (tmp_path / "spec-editor" / "methodology.yaml").read_text(encoding="utf-8")
    assert "preset: default-backend" in methodology


def test_init_existing_without_force_errors(tmp_path: Path) -> None:
    (tmp_path / "spec-editor").mkdir()
    result = _run(["--project-root", str(tmp_path), "init", "--no-wizard", "--preset", "minimal"])
    assert result.exit_code == 1
    assert "refusing" in (result.output or result.stderr or "").lower()


def test_init_existing_with_force(tmp_path: Path) -> None:
    (tmp_path / "spec-editor").mkdir()
    (tmp_path / "spec-editor" / "old.txt").write_text("keep me", encoding="utf-8")
    result = _run(
        [
            "--project-root",
            str(tmp_path),
            "init",
            "--no-wizard",
            "--preset",
            "minimal",
            "--force",
        ]
    )
    assert result.exit_code == 0
    # old file preserved (we don't wipe; just allow re-init)
    assert (tmp_path / "spec-editor" / "old.txt").exists()


def test_generate_without_project_root_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = _run(["generate", "--layer", "L1"])
    assert result.exit_code == 11


def test_metrics_no_methodology_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = _run(["metrics"])
    # No project root → 11
    assert result.exit_code == 11


def test_metrics_on_empty_corpus(tmp_path: Path) -> None:
    _run(
        [
            "--project-root",
            str(tmp_path),
            "init",
            "--no-wizard",
            "--preset",
            "minimal",
        ]
    )
    result = _run(["--project-root", str(tmp_path), "metrics"])
    # minimal preset has no metrics declared yet → exits 0 with stub message
    assert result.exit_code == 0
