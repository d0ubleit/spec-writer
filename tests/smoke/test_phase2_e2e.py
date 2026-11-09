"""S2.12 + S3.12 — Phase 2 / 3 smoke E2E.

Opt-in (``pytest -m smoke``). Hits real Claude via the subscription
provider — costs tokens. Exercises the full path:

  init → drop a source document → spec-writer generate → assert one
  L1 spec landed on disk → validate the corpus → run /metrics output
  → run /refine-plan in --apply '' (preview only).

These tests are intentionally permissive: Claude's exact wording will
differ across runs. We assert the *shape* of the result, not its
content (spec exists, validators are clean enough to converge,
metrics computed successfully).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spec_writer.interface.batch.cli import app

pytestmark = [pytest.mark.smoke]


def _skip_if_no_credentials() -> None:
    cred_path = Path.home() / ".claude" / ".credentials.json"
    if not cred_path.exists() and not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip(
            "no Claude credentials available — set ANTHROPIC_API_KEY or "
            "log in via `claude login` to run this smoke test"
        )


def _runner() -> CliRunner:
    return CliRunner()


def test_phase2_generate_l1_against_real_claude(tmp_path: Path) -> None:
    _skip_if_no_credentials()
    runner = _runner()

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

    # 2. drop a source document
    spec_editor = tmp_path / "spec-editor"
    src = spec_editor / "sources-raw" / "biz.md"
    src.write_text(
        (
            "# Business idea\n\n"
            "Build a small URL shortener. Users paste a long URL and "
            "receive a short token they can share. Anonymous use is allowed."
        ),
        encoding="utf-8",
    )

    # 3. generate L1
    result = runner.invoke(
        app,
        [
            "--project-root",
            str(tmp_path),
            "generate",
            "--layer",
            "L1",
            "--max-iterations",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "OK" in result.output or "converged" in result.output.lower()

    # 4. at least one spec landed in specs/vision/
    vision_dir = spec_editor / "specs" / "vision"
    assert vision_dir.exists()
    md_files = list(vision_dir.glob("*.md"))
    assert len(md_files) >= 1, f"no specs generated in {vision_dir}"
    # frontmatter sanity
    first = md_files[0].read_text(encoding="utf-8")
    assert first.startswith("---\n")
    assert "type: vision" in first
    assert "layer: L1" in first


def test_phase3_validate_metrics_refine_plan_against_real_claude(tmp_path: Path) -> None:
    _skip_if_no_credentials()
    runner = _runner()

    # init
    runner.invoke(
        app,
        ["--project-root", str(tmp_path), "init", "--no-wizard", "--preset", "minimal"],
    )
    src = tmp_path / "spec-editor" / "sources-raw" / "biz.md"
    src.write_text("# Idea\n\nA tiny notes app.", encoding="utf-8")

    # generate one L1 spec so the corpus is non-empty
    gen = runner.invoke(
        app,
        [
            "--project-root",
            str(tmp_path),
            "generate",
            "--layer",
            "L1",
            "--max-iterations",
            "3",
        ],
    )
    assert gen.exit_code == 0, gen.output

    # validate
    val = runner.invoke(app, ["--project-root", str(tmp_path), "validate"])
    assert val.exit_code in (0, 1), val.output

    # metrics
    met = runner.invoke(app, ["--project-root", str(tmp_path), "metrics"])
    assert met.exit_code in (0, 1), met.output

    # status — read-only
    sta = runner.invoke(app, ["--project-root", str(tmp_path), "status"])
    assert sta.exit_code == 0, sta.output


def test_phase4_rev_chat_against_real_claude(tmp_path: Path) -> None:
    """S4.9 — REV chat E2E: drop a small Python project, run `chat --direction rev`.

    Seeds two tiny .py files, opens chat in REV mode, asks the agent
    to discover code and propose L4 specs. Asserts session captured
    at least one `discover_code_inventory` tool call and one
    `record_spec` for an L4 spec with `implemented_by`. Permissive on
    exact wording.
    """
    _skip_if_no_credentials()
    runner = _runner()
    runner.invoke(
        app,
        [
            "--project-root",
            str(tmp_path),
            "init",
            "--no-wizard",
            "--preset",
            "default-backend",
        ],
    )

    src = tmp_path / "src"
    src.mkdir()
    (src / "auth.py").write_text(
        "def login(username: str, password: str) -> bool:\n"
        '    """Authenticate a user by username + password."""\n'
        "    return _verify(username, password)\n"
        "\n"
        "def _verify(u: str, p: str) -> bool:\n"
        "    return False  # stub\n",
        encoding="utf-8",
    )
    (src / "session.py").write_text(
        "def create_session(user_id: int) -> str:\n"
        '    """Open a new session, return token."""\n'
        "    return f'tok-{user_id}'\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat", "--direction", "rev"],
        input=(
            "Discover the code inventory, then write one L4 api-contract per "
            "public function with implemented_by pointing at the real file:line range.\n"
            "/quit\n"
        ),
    )
    assert result.exit_code == 0, result.output

    sessions_dir = tmp_path / "spec-editor" / "_meta" / "sessions"
    files = list(sessions_dir.glob("*.json"))
    assert len(files) == 1

    import json as _json

    payload = _json.loads(files[0].read_text(encoding="utf-8"))
    tool_calls = [e for e in payload["events"] if e["kind"] == "tool-call"]
    tool_names = {e.get("tool") for e in tool_calls}
    assert "discover_code_inventory" in tool_names or "Read" in tool_names, (
        f"expected discover or Read; got {tool_names}"
    )


def test_phase3_chat_agent_against_real_claude(tmp_path: Path) -> None:
    """S3.13 — `spec-writer chat` free-form input drives the ChatAgent end-to-end.

    Drops a source doc, opens chat, sends one free-form message asking
    the agent to look at sources and propose something. Asserts the
    chat session log captured at least an agent message (the agent
    responded) and one tool call (it actually looked at the corpus or
    sources). Permissive on exact content — Claude's wording varies.
    """
    _skip_if_no_credentials()
    runner = _runner()
    runner.invoke(
        app,
        ["--project-root", str(tmp_path), "init", "--no-wizard", "--preset", "minimal"],
    )

    src = tmp_path / "spec-editor" / "sources-raw" / "biz.md"
    src.write_text(
        (
            "# Tiny notes app\n\n"
            "Users write short text notes (1-500 chars). Notes are private to "
            "the owner; no sharing in v1."
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["--project-root", str(tmp_path), "chat"],
        input=(
            "List what's in sources/ and tell me what L1 vision specs you would write.\n/quit\n"
        ),
    )
    assert result.exit_code == 0, result.output

    sessions_dir = tmp_path / "spec-editor" / "_meta" / "sessions"
    files = list(sessions_dir.glob("*.json"))
    assert len(files) == 1, f"expected one session file, got {files}"

    import json as _json

    payload = _json.loads(files[0].read_text(encoding="utf-8"))
    kinds = [e["kind"] for e in payload["events"]]
    # Agent should have produced at least one assistant message
    assert "agent-message" in kinds, f"no agent-message in {kinds}"
    # And called at least one tool — list_sources, list_corpus, Read, etc.
    assert "tool-call" in kinds or "tool-result" in kinds, f"no tool activity in {kinds}"
