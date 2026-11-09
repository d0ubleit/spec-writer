"""``agent.yaml`` loader with 3-level resolution.

Resolution order (lowest → highest precedence):
  1. Built-in defaults baked into :class:`AgentConfig`
  2. User config at ``~/.spec-writer/agent.yaml``
  3. Project config at ``<project>/spec-editor/agent.yaml``

Within a level, deep-merge: nested dicts merge key-by-key, scalar /
list values from the higher level overwrite the lower one. After
merging we apply ``${ENV_VAR}`` substitution on string values and
validate the resulting payload against :class:`AgentConfig`.

Unknown fields are kept (Pydantic ``extra="allow"``) but recorded
as warnings — they don't block the load.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any, Final, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from spec_writer.types import Direction  # re-export typing prevent unused

del Direction


_ENV_VAR_PATTERN: Final[re.Pattern[str]] = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


# ── nested config models ─────────────────────────────────────────────────


def _empty_str_list() -> list[str]:
    return []


class TokenLimits(BaseModel):
    """Per-scope token caps. ``null`` means no cap."""

    model_config = ConfigDict(extra="forbid")

    per_session: int | None = None
    per_layer: int | None = 500_000
    per_project: int | None = 5_000_000


class HistoryConfig(BaseModel):
    """Conversation-history compression knobs (S2.7)."""

    model_config = ConfigDict(extra="forbid")

    tool_result_truncate_after_turns: int = 10
    sliding_window_size: int = 20
    compression_strategy: Literal["auto", "sdk-only", "sliding", "none"] = "auto"


class BashConfig(BaseModel):
    """Bash builtin gating — off by default."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    allowlist: list[str] = Field(default_factory=_empty_str_list)


def _default_bash() -> BashConfig:
    return BashConfig()


class ToolsConfig(BaseModel):
    """Per-tool agent capabilities."""

    model_config = ConfigDict(extra="forbid")

    bash: BashConfig = Field(default_factory=_default_bash)


def _default_limits() -> TokenLimits:
    return TokenLimits()


def _default_history() -> HistoryConfig:
    return HistoryConfig()


def _default_tools() -> ToolsConfig:
    return ToolsConfig()


class AgentConfig(BaseModel):
    """Resolved agent-runtime configuration."""

    model_config = ConfigDict(extra="allow")

    provider: Literal["claude-subscription", "claude-apikey", "openai-apikey", "fake"] = (
        "claude-subscription"
    )
    model: str = "claude-opus-4-7"
    api_key: str | None = None
    limits: TokenLimits = Field(default_factory=_default_limits)
    history: HistoryConfig = Field(default_factory=_default_history)
    tools: ToolsConfig = Field(default_factory=_default_tools)


# ── loader ───────────────────────────────────────────────────────────────


_KNOWN_TOP_LEVEL = frozenset({"provider", "model", "api_key", "limits", "history", "tools"})


def _read_yaml_dict(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return cast(dict[str, Any], raw)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(cast(dict[str, Any], existing), cast(dict[str, Any], value))
        else:
            merged[key] = value
    return merged


def _substitute_env_vars(value: Any) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), match.group(0))

        return _ENV_VAR_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in cast(dict[str, Any], value).items()}
    if isinstance(value, list):
        return [_substitute_env_vars(item) for item in cast(list[Any], value)]
    return value


def _warn_unknown_fields(payload: dict[str, Any], origin: str) -> None:
    extras = [key for key in payload if key not in _KNOWN_TOP_LEVEL]
    if extras:
        warnings.warn(
            f"agent config {origin}: unknown top-level fields ignored: {extras}",
            stacklevel=3,
        )


def load_agent_config(
    project_root: Path | None = None,
    user_config: Path | None = None,
) -> AgentConfig:
    """Resolve agent.yaml across built-in / user / project layers."""
    config: dict[str, Any] = AgentConfig().model_dump()

    user_path = (
        user_config if user_config is not None else Path.home() / ".spec-writer" / "agent.yaml"
    )
    if user_path.exists():
        user_data = _read_yaml_dict(user_path)
        _warn_unknown_fields(user_data, str(user_path))
        config = _deep_merge(config, user_data)

    if project_root is not None:
        project_path = project_root / "spec-editor" / "agent.yaml"
        if project_path.exists():
            project_data = _read_yaml_dict(project_path)
            _warn_unknown_fields(project_data, str(project_path))
            config = _deep_merge(config, project_data)

    substituted = cast(dict[str, Any], _substitute_env_vars(config))
    return AgentConfig.model_validate(substituted)


__all__ = [
    "AgentConfig",
    "BashConfig",
    "HistoryConfig",
    "TokenLimits",
    "ToolsConfig",
    "load_agent_config",
]
