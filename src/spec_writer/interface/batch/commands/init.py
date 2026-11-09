"""``spec-writer init`` — bootstrap a new project directory."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import typer

_BASE_ACTIVE_MIXINS = {
    "minimal": ["core-traceability", "vision"],
    "default-backend": [
        "core-traceability",
        "vision",
        "capabilities",
        "behaviors",
        "acceptance-criteria",
        "api-contracts",
        "entity-schemas",
        "tagging",
        "lifecycle-fields",
    ],
    # Full-fat preset for web products that ship UI + API + persistence
    # + jobs + integrations + ops. Activates every built-in mixin so the
    # methodology is ready for stage-2 / stage-3 growth without re-init.
    "web-product": [
        "core-traceability",
        "vision",
        "capabilities",
        "behaviors",
        "acceptance-criteria",
        "api-contracts",
        "entity-schemas",
        "tagging",
        "lifecycle-fields",
        # cross-cutting (project semantics)
        "glossary",
        "requirements-traceability",
        "architecture-views",
        "conventions",
        # functional layers
        "user-scenarios",
        "workflows",
        "quality-requirements",
        # UI + persistence + behaviour
        "ui-screens",
        "ui-components",
        "integration-contracts",
        "async-jobs",
        "state-machines",
        "rbac-matrix",
        "auth-flow",
        "fixtures",
        "configuration",
        "error-codes",
        "test-cases",
        # operations
        "deployment-topology",
        "migrations",
        "observability-specs",
        "tech-stack",
    ],
}


def _methodology_template(preset: str) -> str:
    mixins = _BASE_ACTIVE_MIXINS.get(preset, _BASE_ACTIVE_MIXINS["minimal"])
    mixin_lines = "\n".join(f"  - {name}" for name in mixins)
    return f"# spec-writer methodology config\npreset: {preset}\nactive_mixins:\n{mixin_lines}\n"


def _detect_default_preset(root: Path) -> str:
    markers = {
        "package.json": "default-backend",
        "composer.json": "default-backend",
        "Cargo.toml": "default-backend",
        "pyproject.toml": "default-backend",
        "go.mod": "default-backend",
    }
    for marker, preset in markers.items():
        if (root / marker).exists():
            return preset
    return "minimal"


def _write_methodology_file(spec_editor: Path, preset: str) -> None:
    target = spec_editor / "methodology.yaml"
    target.write_text(_methodology_template(preset), encoding="utf-8")


def bootstrap_project(root: Path, preset: str = "default-backend") -> Path:
    """Create the standard ``spec-editor/`` layout without any prompting.

    Public helper shared by ``spec-writer init`` and ``spec-writer demo``.
    Returns the ``spec-editor`` directory.
    """
    spec_editor = root / "spec-editor"
    for sub in ("sources-raw", "sources", "specs", "_meta"):
        (spec_editor / sub).mkdir(parents=True, exist_ok=True)
    _write_methodology_file(spec_editor, preset)
    return spec_editor


def run(
    ctx: typer.Context,
    preset: str | None = typer.Option(
        None, "--preset", help="Methodology preset (e.g. default-backend, minimal)."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing spec-editor/."),
    no_wizard: bool = typer.Option(
        False, "--no-wizard", help="Skip interactive prompts; use flags/defaults."
    ),
) -> None:
    """Create the standard ``spec-editor/`` layout in the project root.

    Without ``--no-wizard``, missing options are filled in by simple
    stdin prompts. With ``--no-wizard`` the command fails when required
    info is missing.
    """
    state = cast(dict[str, object], ctx.obj or {})
    explicit_root = cast(Path | None, state.get("project_root"))
    root = explicit_root.resolve() if explicit_root else Path.cwd().resolve()

    spec_editor = root / "spec-editor"
    if spec_editor.exists() and not force:
        typer.echo(
            f"refusing to overwrite existing {spec_editor} — pass --force to proceed",
            err=True,
        )
        raise typer.Exit(code=1)

    if preset is None:
        default = _detect_default_preset(root)
        if no_wizard:
            preset = default
        else:
            preset = typer.prompt("Methodology preset", default=default, show_default=True)

    for sub in ("sources-raw", "sources", "specs", "_meta"):
        (spec_editor / sub).mkdir(parents=True, exist_ok=True)
    _write_methodology_file(spec_editor, preset or "minimal")

    typer.echo(f"initialized {spec_editor} (preset={preset})")


__all__ = ["bootstrap_project", "run"]
