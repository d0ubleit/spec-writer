"""S2.0: agent.yaml loader with 3-level resolution + env substitution."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from spec_writer.config import AgentConfig, load_agent_config


def test_builtin_defaults(tmp_path: Path) -> None:
    cfg = load_agent_config(
        project_root=None,
        user_config=tmp_path / "nope.yaml",
    )
    assert cfg.provider == "claude-subscription"
    assert cfg.model == "claude-opus-4-7"
    assert cfg.limits.per_layer == 500_000
    assert cfg.history.compression_strategy == "auto"
    assert cfg.tools.bash.enabled is False


def test_user_overrides_defaults(tmp_path: Path) -> None:
    user_yaml = tmp_path / "user.yaml"
    user_yaml.write_text(
        "provider: claude-apikey\nmodel: custom\n",
        encoding="utf-8",
    )
    cfg = load_agent_config(user_config=user_yaml)
    assert cfg.provider == "claude-apikey"
    assert cfg.model == "custom"


def test_project_overrides_user(tmp_path: Path) -> None:
    user_yaml = tmp_path / "user.yaml"
    user_yaml.write_text("model: from-user\n", encoding="utf-8")

    project_root = tmp_path / "project"
    (project_root / "spec-editor").mkdir(parents=True)
    (project_root / "spec-editor" / "agent.yaml").write_text(
        "model: from-project\n", encoding="utf-8"
    )

    cfg = load_agent_config(project_root=project_root, user_config=user_yaml)
    assert cfg.model == "from-project"


def test_deep_merge_preserves_unset_user_fields(tmp_path: Path) -> None:
    user_yaml = tmp_path / "user.yaml"
    user_yaml.write_text(
        ("limits:\n  per_layer: 1000\n  per_project: 9999\n"),
        encoding="utf-8",
    )
    project_root = tmp_path / "p"
    (project_root / "spec-editor").mkdir(parents=True)
    (project_root / "spec-editor" / "agent.yaml").write_text(
        "limits:\n  per_layer: 250\n",
        encoding="utf-8",
    )
    cfg = load_agent_config(project_root=project_root, user_config=user_yaml)
    assert cfg.limits.per_layer == 250  # project overrides user
    assert cfg.limits.per_project == 9999  # user value preserved


def test_env_var_substitution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MY_API_KEY", "secret-123")
    user_yaml = tmp_path / "user.yaml"
    user_yaml.write_text(
        ("provider: claude-apikey\napi_key: ${MY_API_KEY}\n"),
        encoding="utf-8",
    )
    cfg = load_agent_config(user_config=user_yaml)
    assert cfg.api_key == "secret-123"


def test_env_var_substitution_unset_leaves_literal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DOES_NOT_EXIST", raising=False)
    user_yaml = tmp_path / "user.yaml"
    user_yaml.write_text(
        ("provider: claude-apikey\napi_key: ${DOES_NOT_EXIST}\n"),
        encoding="utf-8",
    )
    cfg = load_agent_config(user_config=user_yaml)
    assert cfg.api_key == "${DOES_NOT_EXIST}"


def test_invalid_provider_raises(tmp_path: Path) -> None:
    user_yaml = tmp_path / "user.yaml"
    user_yaml.write_text("provider: not-a-provider\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_agent_config(user_config=user_yaml)


def test_unknown_top_level_field_warns(tmp_path: Path) -> None:
    user_yaml = tmp_path / "user.yaml"
    user_yaml.write_text("provider: fake\nbogus_field: 42\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_agent_config(user_config=user_yaml)
    assert cfg.provider == "fake"
    assert any("bogus_field" in str(w.message) for w in caught)


def test_invalid_nested_field_raises(tmp_path: Path) -> None:
    user_yaml = tmp_path / "user.yaml"
    user_yaml.write_text("history:\n  compression_strategy: nonsense\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_agent_config(user_config=user_yaml)


def test_history_strategy_values(tmp_path: Path) -> None:
    for strategy in ("auto", "sdk-only", "sliding", "none"):
        user_yaml = tmp_path / "u.yaml"
        user_yaml.write_text(f"history:\n  compression_strategy: {strategy}\n", encoding="utf-8")
        cfg = load_agent_config(user_config=user_yaml)
        assert cfg.history.compression_strategy == strategy


def test_bash_default_disabled(tmp_path: Path) -> None:
    cfg = load_agent_config(user_config=tmp_path / "absent.yaml")
    assert cfg.tools.bash.enabled is False
    assert cfg.tools.bash.allowlist == []


def test_bash_with_allowlist(tmp_path: Path) -> None:
    user_yaml = tmp_path / "user.yaml"
    user_yaml.write_text(
        ("tools:\n  bash:\n    enabled: true\n    allowlist: [git, ls]\n"),
        encoding="utf-8",
    )
    cfg = load_agent_config(user_config=user_yaml)
    assert cfg.tools.bash.enabled is True
    assert cfg.tools.bash.allowlist == ["git", "ls"]


def test_agent_config_json_schema_exported() -> None:
    schema = AgentConfig.model_json_schema()
    assert "properties" in schema
    assert "provider" in schema["properties"]


def test_missing_project_config_no_crash(tmp_path: Path) -> None:
    project_root = tmp_path / "p"
    project_root.mkdir()
    # no spec-editor/ at all
    cfg = load_agent_config(project_root=project_root, user_config=tmp_path / "u.yaml")
    assert cfg.provider == "claude-subscription"
