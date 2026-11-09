"""S1.1: Scenario / Scope literal validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spec_writer.types import Scenario, Scope

MODES = (
    "generate",
    "refine",
    "incremental-add",
    "cascade-fix",
    "targeted-edit-cascade",
    "sync",
)

DIRECTIONS = ("fwd", "rev")


@pytest.mark.parametrize("mode", MODES)
def test_scenario_all_modes_accepted(mode: str) -> None:
    sc = Scenario(direction="fwd", layers_in_scope=["L1"], mode=mode)  # type: ignore[arg-type]
    assert sc.mode == mode


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_scenario_both_directions(direction: str) -> None:
    sc = Scenario(direction=direction, layers_in_scope=["L1"], mode="generate")  # type: ignore[arg-type]
    assert sc.direction == direction


def test_scenario_invalid_mode_fails() -> None:
    with pytest.raises(ValidationError):
        Scenario(direction="fwd", layers_in_scope=["L1"], mode="bogus")  # type: ignore[arg-type]


def test_scenario_invalid_direction_fails() -> None:
    with pytest.raises(ValidationError):
        Scenario(direction="sideways", layers_in_scope=["L1"], mode="generate")  # type: ignore[arg-type]


def test_scenario_defaults() -> None:
    sc = Scenario(direction="fwd", layers_in_scope=["L1", "L2"], mode="generate")
    assert sc.scope is None
    assert sc.max_iterations == 5
    assert sc.target_metrics == {}
    assert sc.auto_apply is False


def test_scope_all_filters_optional() -> None:
    s = Scope()
    assert s.layers is None
    assert s.types is None
    assert s.tags is None
    assert s.folder_pattern is None
    assert s.spec_ids is None
    assert s.from_finding is None


def test_scope_combined_filters() -> None:
    s = Scope(layers=["L3"], tags=["auth"], from_finding=["dangling-ref"])
    assert s.layers == ["L3"]
    assert s.tags == ["auth"]
    assert s.from_finding == ["dangling-ref"]


def test_scenario_with_scope() -> None:
    sc = Scenario(
        direction="fwd",
        layers_in_scope=["L3"],
        mode="refine",
        scope=Scope(tags=["auth"]),
    )
    assert sc.scope is not None
    assert sc.scope.tags == ["auth"]
