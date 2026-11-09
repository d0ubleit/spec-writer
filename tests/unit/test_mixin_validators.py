"""Tests for the built-in mixin validators + metrics.

Covers:
- The web-product preset loads every built-in mixin successfully.
- Each new spec type is registered with the right layer and prefix.
- New required_frontmatter / sections are surfaced as expected.
- Each validator detects its target failure modes.
- Metrics report sensible values for trivial corpora.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import cast

import pytest

from spec_writer.methodology.loader import load_methodology
from spec_writer.methodology.metrics import MetricContext
from spec_writer.methodology.metrics.mixin_metrics import (
    _behavior_test_coverage,
    _priority_tagged,
    _rbac_coverage,
    _scenario_step_linkage,
    _scenario_test_coverage,
)
from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.methodology.sources import FilesystemMixinSource
from spec_writer.methodology.validators import ValidatorContext
from spec_writer.methodology.validators.builtin import check_pyramid_adjacency
from spec_writer.methodology.validators.mixin_validators import (
    check_arch_view_singleton_per_kind,
    check_configuration_namespace_unique,
    check_error_code_http_status,
    check_error_code_uniqueness,
    check_glossary_aliases_unique,
    check_glossary_canonical_unique,
    check_rbac_matrix_completeness,
    check_rbac_matrix_singleton,
    check_rbac_roles_resolved,
    check_scenario_steps_linked,
    check_state_machine_reachability,
    check_tech_choice_uniqueness_per_category,
    check_test_case_id_unique,
    check_test_case_kind_matches_subject,
    check_traces_to_format,
    check_ui_component_testids_unique,
    check_ui_screen_routes_unique,
)
from spec_writer.types import Spec

# ── helpers ──────────────────────────────────────────────────────────────


def _builtin_base() -> Path:
    base = resources.files("spec_writer.methodology._builtin").joinpath("base.yaml")
    return Path(str(base))


def _builtin_mixins() -> Path:
    return Path(str(resources.files("spec_writer.methodology._builtin").joinpath("mixins")))


_WEB_PRODUCT_ACTIVE = [
    "core-traceability",
    "vision",
    "capabilities",
    "behaviors",
    "acceptance-criteria",
    "api-contracts",
    "entity-schemas",
    "tagging",
    "lifecycle-fields",
    "glossary",
    "requirements-traceability",
    "architecture-views",
    "conventions",
    "user-scenarios",
    "workflows",
    "quality-requirements",
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
    "deployment-topology",
    "migrations",
    "observability-specs",
    "tech-stack",
]


@pytest.fixture(scope="module")
def methodology() -> ResolvedMethodology:
    return load_methodology(
        _builtin_base(),
        [FilesystemMixinSource(_builtin_mixins())],
        active_mixins=_WEB_PRODUCT_ACTIVE,
    )


def _spec(
    spec_id: str,
    type_: str,
    layer: str,
    *,
    title: str = "X",
    up_to: list[str] | None = None,
    body: str = "",
    **extras: object,
) -> Spec:
    return Spec(
        id=spec_id,
        type=type_,
        layer=layer,
        title=title,
        up_to=up_to or [],
        body=body,
        **extras,  # type: ignore[arg-type]
    )


def _ctx(specs: list[Spec], methodology: ResolvedMethodology) -> ValidatorContext:
    return ValidatorContext(corpus={s.id: s for s in specs}, methodology=methodology)


# ── 1. Methodology load smoke ────────────────────────────────────────────


def test_web_product_preset_loads(methodology: ResolvedMethodology) -> None:
    rm = methodology
    expected = {
        # baseline
        "vision",
        "capability",
        "behavior",
        "api-contract",
        "entity-schema",
        # cross-cutting
        "term",
        "arch-view",
        "convention",
        # functional
        "user-scenario",
        "workflow",
        "quality-req",
        # boundary
        "ui-screen",
        "ui-component",
        "integration-contract",
        "async-job",
        "state-machine",
        "rbac-matrix",
        "auth-flow",
        "fixture",
        "configuration",
        "error-code",
        "test-case",
        # operations
        "deployment-topology",
        "migration",
        "observability-spec",
        "tech-choice",
    }
    assert expected <= set(rm.spec_types.keys())


def test_user_scenario_lives_on_L3(methodology: ResolvedMethodology) -> None:
    st = methodology.spec_types["user-scenario"]
    assert st.layer_id == "L3"
    assert st.prefix == "USC"
    section_ids = {s.id for s in st.sections}
    assert {"purpose", "actor", "steps", "postconditions"} <= section_ids


def test_ui_screen_requires_route_and_roles(methodology: ResolvedMethodology) -> None:
    st = methodology.spec_types["ui-screen"]
    required = {f.name for f in st.required_frontmatter}
    assert {"route", "layout_kind", "roles"} <= required


def test_api_contract_now_requires_roles(methodology: ResolvedMethodology) -> None:
    st = methodology.spec_types["api-contract"]
    required = {f.name for f in st.required_frontmatter}
    assert "roles" in required


def test_tech_choice_category_enum(methodology: ResolvedMethodology) -> None:
    enums = methodology.field_enum_values("category")
    # quality-requirements + tech-stack both add to `category`; both must be present
    assert "backend-language" in enums
    assert "performance" in enums


def test_tech_choice_status_enum(methodology: ResolvedMethodology) -> None:
    enums = methodology.field_enum_values("status")
    # lifecycle-fields contributes draft/review/..; tech-stack adds chosen/...
    assert {"chosen", "considered", "rejected", "deprecated", "superseded"} <= set(enums)


def test_phase_enum_present(methodology: ResolvedMethodology) -> None:
    enums = methodology.field_enum_values("phase")
    assert {"mvp", "stage-2", "stage-3"} <= set(enums)


# ── 2. user-scenarios step linkage ───────────────────────────────────────


def _scenario_body(steps_yaml: str) -> str:
    return f"## Purpose\n\nx\n\n## Actor\n\noperator\n\n## Steps\n\n```yaml\n{steps_yaml}\n```\n\n## Postconditions\n\nok\n"


def test_scenario_steps_linked_ok(methodology: ResolvedMethodology) -> None:
    screen = _spec("L4-SCR-001", "ui-screen", "L4")
    component = _spec("L4-CMP-001", "ui-component", "L4")
    scenario = _spec(
        "L3-USC-001",
        "user-scenario",
        "L3",
        body=_scenario_body(
            "- step: 1\n  screen: L4-SCR-001\n  action: click\n  component: L4-CMP-001\n"
        ),
    )
    issues = check_scenario_steps_linked(scenario, _ctx([screen, component, scenario], methodology))
    assert issues == []


def test_scenario_steps_unknown_screen(methodology: ResolvedMethodology) -> None:
    scenario = _spec(
        "L3-USC-002",
        "user-scenario",
        "L3",
        body=_scenario_body("- step: 1\n  screen: L4-SCR-999\n  action: click\n"),
    )
    issues = check_scenario_steps_linked(scenario, _ctx([scenario], methodology))
    codes = {i.code for i in issues}
    assert "user_scenario_step_unknown_screen" in codes


def test_scenario_steps_missing_screen(methodology: ResolvedMethodology) -> None:
    scenario = _spec(
        "L3-USC-003",
        "user-scenario",
        "L3",
        body=_scenario_body("- step: 1\n  action: observe\n"),
    )
    issues = check_scenario_steps_linked(scenario, _ctx([scenario], methodology))
    codes = {i.code for i in issues}
    assert "user_scenario_step_missing_screen" in codes


def test_scenario_steps_unparseable(methodology: ResolvedMethodology) -> None:
    scenario = _spec(
        "L3-USC-004",
        "user-scenario",
        "L3",
        body="## Steps\n\nnot a list, just prose\n",
    )
    issues = check_scenario_steps_linked(scenario, _ctx([scenario], methodology))
    codes = {i.code for i in issues}
    assert "user_scenario_steps_unparsed" in codes


# ── 3. ui-screen route uniqueness ────────────────────────────────────────


def test_ui_screen_routes_unique_ok(methodology: ResolvedMethodology) -> None:
    a = _spec("L4-SCR-001", "ui-screen", "L4", route="/admin/sites")
    b = _spec("L4-SCR-002", "ui-screen", "L4", route="/admin/profiles")
    assert check_ui_screen_routes_unique([a, b], _ctx([a, b], methodology)) == []


def test_ui_screen_routes_duplicate_flagged(methodology: ResolvedMethodology) -> None:
    a = _spec("L4-SCR-001", "ui-screen", "L4", route="/admin/sites")
    b = _spec("L4-SCR-002", "ui-screen", "L4", route="/admin/sites")
    issues = check_ui_screen_routes_unique([a, b], _ctx([a, b], methodology))
    assert any(i.code == "ui_screen_route_duplicate" for i in issues)


# ── 4. ui-component testid uniqueness ────────────────────────────────────


def test_ui_component_testid_duplicate(methodology: ResolvedMethodology) -> None:
    a = _spec("L4-CMP-001", "ui-component", "L4", testid="btn-create")
    b = _spec("L4-CMP-002", "ui-component", "L4", testid="btn-create")
    issues = check_ui_component_testids_unique([a, b], _ctx([a, b], methodology))
    assert any(i.code == "ui_component_testid_duplicate" for i in issues)


# ── 5. state-machine reachability ────────────────────────────────────────


def _sm_body(states_yaml: str, transitions_yaml: str) -> str:
    return (
        "## Purpose\n\nx\n\n"
        "## Subject\n\nentity\n\n"
        "## States\n\n```yaml\n"
        + states_yaml
        + "\n```\n\n## Transitions\n\n```yaml\n"
        + transitions_yaml
        + "\n```\n"
    )


def test_state_machine_happy_path(methodology: ResolvedMethodology) -> None:
    sm = _spec(
        "L4-SM-001",
        "state-machine",
        "L4",
        body=_sm_body(
            "- id: draft\n  initial: true\n- id: published\n- id: archived\n  terminal: true\n",
            "- from: draft\n  to: published\n  trigger: publish\n- from: published\n  to: archived\n  trigger: archive\n",
        ),
    )
    assert check_state_machine_reachability(sm, _ctx([sm], methodology)) == []


def test_state_machine_unreachable(methodology: ResolvedMethodology) -> None:
    sm = _spec(
        "L4-SM-002",
        "state-machine",
        "L4",
        body=_sm_body(
            "- id: draft\n  initial: true\n- id: published\n- id: orphan\n  terminal: true\n",
            "- from: draft\n  to: published\n  trigger: publish\n",
        ),
    )
    issues = check_state_machine_reachability(sm, _ctx([sm], methodology))
    codes = {i.code for i in issues}
    assert "state_machine_unreachable" in codes
    assert "state_machine_dead_end" in codes  # published has no outgoing and is non-terminal


def test_state_machine_no_initial(methodology: ResolvedMethodology) -> None:
    sm = _spec(
        "L4-SM-003",
        "state-machine",
        "L4",
        body=_sm_body(
            "- id: a\n- id: b\n",
            "- from: a\n  to: b\n  trigger: t\n",
        ),
    )
    issues = check_state_machine_reachability(sm, _ctx([sm], methodology))
    assert any(i.code == "state_machine_no_initial" for i in issues)


def test_state_machine_multiple_initials(methodology: ResolvedMethodology) -> None:
    sm = _spec(
        "L4-SM-004",
        "state-machine",
        "L4",
        body=_sm_body(
            "- id: a\n  initial: true\n- id: b\n  initial: true\n  terminal: true\n",
            "- from: a\n  to: b\n  trigger: t\n",
        ),
    )
    issues = check_state_machine_reachability(sm, _ctx([sm], methodology))
    assert any(i.code == "state_machine_multiple_initials" for i in issues)


# ── 6. tech-stack: one-chosen-per-category ───────────────────────────────


def test_tech_choice_uniqueness_ok(methodology: ResolvedMethodology) -> None:
    fastapi = _spec(
        "L4-TECH-001",
        "tech-choice",
        "L4",
        category="backend-framework",
        phase="mvp",
        status="chosen",
    )
    django = _spec(
        "L4-TECH-002",
        "tech-choice",
        "L4",
        category="backend-framework",
        phase="mvp",
        status="considered",
    )
    assert (
        check_tech_choice_uniqueness_per_category(
            [fastapi, django], _ctx([fastapi, django], methodology)
        )
        == []
    )


def test_tech_choice_uniqueness_violated(methodology: ResolvedMethodology) -> None:
    a = _spec(
        "L4-TECH-001",
        "tech-choice",
        "L4",
        category="backend-framework",
        phase="mvp",
        status="chosen",
    )
    b = _spec(
        "L4-TECH-002",
        "tech-choice",
        "L4",
        category="backend-framework",
        phase="mvp",
        status="chosen",
    )
    issues = check_tech_choice_uniqueness_per_category([a, b], _ctx([a, b], methodology))
    assert any(i.code == "tech_choice_duplicate_chosen" for i in issues)


def test_tech_choice_same_category_different_phase_ok(methodology: ResolvedMethodology) -> None:
    drupal = _spec(
        "L4-TECH-001",
        "tech-choice",
        "L4",
        category="cms",
        phase="mvp",
        status="chosen",
    )
    joomla = _spec(
        "L4-TECH-002",
        "tech-choice",
        "L4",
        category="cms",
        phase="stage-2",
        status="chosen",
    )
    assert (
        check_tech_choice_uniqueness_per_category(
            [drupal, joomla], _ctx([drupal, joomla], methodology)
        )
        == []
    )


# ── 7. rbac-matrix validators ────────────────────────────────────────────


def _rbac_body(roles: list[str], extra_roles_yaml: str = "") -> str:
    role_block = "\n".join(f"- id: {r}\n  description: x" for r in roles)
    if extra_roles_yaml:
        role_block = extra_roles_yaml
    return (
        "## Purpose\n\nx\n\n"
        f"## Roles\n\n```yaml\n{role_block}\n```\n\n"
        "## Resources\n\n```yaml\n- id: site\n  actions: [create, read]\n```\n\n"
        "## Matrix\n\n```yaml\n- role: operator\n  grants:\n    - resource: site\n      actions: [create, read]\n```\n"
    )


def test_rbac_singleton_ok(methodology: ResolvedMethodology) -> None:
    m = _spec("L4-RBAC-001", "rbac-matrix", "L4", body=_rbac_body(["operator", "admin"]))
    assert check_rbac_matrix_singleton([m], _ctx([m], methodology)) == []


def test_rbac_singleton_violated(methodology: ResolvedMethodology) -> None:
    m1 = _spec("L4-RBAC-001", "rbac-matrix", "L4", body=_rbac_body(["a"]))
    m2 = _spec("L4-RBAC-002", "rbac-matrix", "L4", body=_rbac_body(["b"]))
    issues = check_rbac_matrix_singleton([m1, m2], _ctx([m1, m2], methodology))
    assert any(i.code == "rbac_matrix_not_singleton" for i in issues)


def test_rbac_unknown_role_in_api_contract(methodology: ResolvedMethodology) -> None:
    m = _spec("L4-RBAC-001", "rbac-matrix", "L4", body=_rbac_body(["operator", "admin"]))
    bad_api = _spec(
        "L4-API-001",
        "api-contract",
        "L4",
        roles=["sysadmin"],  # typo
    )
    issues = check_rbac_roles_resolved([m, bad_api], _ctx([m, bad_api], methodology))
    assert any(i.code == "rbac_unknown_role" for i in issues)


def test_rbac_pseudo_roles_pass(methodology: ResolvedMethodology) -> None:
    m = _spec("L4-RBAC-001", "rbac-matrix", "L4", body=_rbac_body(["operator"]))
    anon_api = _spec("L4-API-001", "api-contract", "L4", roles=["anonymous"])
    sys_job = _spec("L4-JOB-001", "async-job", "L4", roles=["system"])
    assert (
        check_rbac_roles_resolved([m, anon_api, sys_job], _ctx([m, anon_api, sys_job], methodology))
        == []
    )


def test_rbac_orphan_role_warning(methodology: ResolvedMethodology) -> None:
    m = _spec(
        "L4-RBAC-001",
        "rbac-matrix",
        "L4",
        body=_rbac_body(["operator", "unused-role"]),
    )
    user_api = _spec("L4-API-001", "api-contract", "L4", roles=["operator"])
    issues = check_rbac_matrix_completeness([m, user_api], _ctx([m, user_api], methodology))
    assert any(i.code == "rbac_role_orphan" and "unused-role" in i.message for i in issues)
    assert all(i.severity == "warning" for i in issues if i.code == "rbac_role_orphan")


# ── 8. metrics ───────────────────────────────────────────────────────────


def _metric_ctx(specs: list[Spec], methodology: ResolvedMethodology, layer: str) -> MetricContext:
    return MetricContext(
        corpus={s.id: s for s in specs},
        methodology=methodology,
        layer_id=layer,
        params={},
    )


def test_scenario_step_linkage_metric(methodology: ResolvedMethodology) -> None:
    screen = _spec("L4-SCR-001", "ui-screen", "L4", route="/x")
    good = _spec(
        "L3-USC-001",
        "user-scenario",
        "L3",
        body=_scenario_body("- step: 1\n  screen: L4-SCR-001\n  action: click\n"),
    )
    bad = _spec(
        "L3-USC-002",
        "user-scenario",
        "L3",
        body=_scenario_body("- step: 1\n  screen: L4-SCR-999\n  action: click\n"),
    )
    result = _scenario_step_linkage(
        [screen, good, bad], _metric_ctx([screen, good, bad], methodology, "L3")
    )
    assert result.value == 0.5


def test_rbac_coverage_metric(methodology: ResolvedMethodology) -> None:
    a = _spec("L4-API-001", "api-contract", "L4", roles=["operator"])
    b = _spec("L4-SCR-001", "ui-screen", "L4")  # missing roles
    result = _rbac_coverage([a, b], _metric_ctx([a, b], methodology, "L4"))
    assert result.value == 0.5


def test_rbac_coverage_metric_no_targets(methodology: ResolvedMethodology) -> None:
    # No rbac-aware specs at all → vacuously 1.0
    cap = _spec("L2-CAP-001", "capability", "L2")
    result = _rbac_coverage([cap], _metric_ctx([cap], methodology, "L4"))
    assert result.value == 1.0


# ── 9. init preset ───────────────────────────────────────────────────────


def test_web_product_preset_in_init() -> None:
    from spec_writer.interface.batch.commands.init import _BASE_ACTIVE_MIXINS

    assert "web-product" in _BASE_ACTIVE_MIXINS
    mixins = _BASE_ACTIVE_MIXINS["web-product"]
    for required in (
        "glossary",
        "requirements-traceability",
        "architecture-views",
        "conventions",
        "user-scenarios",
        "workflows",
        "quality-requirements",
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
        "deployment-topology",
        "migrations",
        "observability-specs",
        "tech-stack",
    ):
        assert required in mixins, f"{required} missing from web-product preset"


# ── 10. glossary ─────────────────────────────────────────────────────────


def test_glossary_canonical_duplicate(methodology: ResolvedMethodology) -> None:
    a = _spec("L1-TERM-001", "term", "L1", canonical="customer")
    b = _spec("L1-TERM-002", "term", "L1", canonical="customer")
    issues = check_glossary_canonical_unique([a, b], _ctx([a, b], methodology))
    assert any(i.code == "glossary_canonical_duplicate" for i in issues)


def test_glossary_alias_duplicate(methodology: ResolvedMethodology) -> None:
    a = _spec("L1-TERM-001", "term", "L1", canonical="customer", aliases=["клиент"])
    b = _spec("L1-TERM-002", "term", "L1", canonical="user", aliases=["клиент"])
    issues = check_glossary_aliases_unique([a, b], _ctx([a, b], methodology))
    assert any(i.code == "glossary_alias_duplicate" for i in issues)


# ── 11. traces_to format ─────────────────────────────────────────────────


def test_traces_to_accepts_valid_format(methodology: ResolvedMethodology) -> None:
    spec = _spec("L3-BHV-001", "behavior", "L3", traces_to=["FR-12", "NFR-04", "US-19"])
    assert check_traces_to_format(spec, _ctx([spec], methodology)) == []


def test_traces_to_flags_bad_format(methodology: ResolvedMethodology) -> None:
    spec = _spec("L3-BHV-001", "behavior", "L3", traces_to=["fr-12", "weird"])
    issues = check_traces_to_format(spec, _ctx([spec], methodology))
    assert all(i.code == "traces_to_bad_format" for i in issues)
    assert len(issues) == 2


def test_traces_to_rejects_non_list(methodology: ResolvedMethodology) -> None:
    spec = _spec("L3-BHV-001", "behavior", "L3", traces_to="FR-12")  # type: ignore[arg-type]
    issues = check_traces_to_format(spec, _ctx([spec], methodology))
    assert any(i.code == "traces_to_not_list" for i in issues)


# ── 12. architecture-views singleton ─────────────────────────────────────


def test_arch_view_multiple_system_contexts(methodology: ResolvedMethodology) -> None:
    a = _spec("L2-ARCH-001", "arch-view", "L2", view_kind="system-context")
    b = _spec("L2-ARCH-002", "arch-view", "L2", view_kind="system-context")
    issues = check_arch_view_singleton_per_kind([a, b], _ctx([a, b], methodology))
    assert any(i.code == "arch_view_multiple_system_context" for i in issues)


def test_arch_view_multiple_container_views_ok(methodology: ResolvedMethodology) -> None:
    a = _spec("L2-ARCH-001", "arch-view", "L2", view_kind="container-view")
    b = _spec("L2-ARCH-002", "arch-view", "L2", view_kind="container-view")
    issues = check_arch_view_singleton_per_kind([a, b], _ctx([a, b], methodology))
    assert issues == []


# ── 13. error-codes ──────────────────────────────────────────────────────


def test_error_code_duplicate(methodology: ResolvedMethodology) -> None:
    a = _spec("L4-ERR-001", "error-code", "L4", code="SITE_DUP", http_status=409)
    b = _spec("L4-ERR-002", "error-code", "L4", code="SITE_DUP", http_status=409)
    issues = check_error_code_uniqueness([a, b], _ctx([a, b], methodology))
    assert any(i.code == "error_code_duplicate" for i in issues)


def test_error_code_http_status_out_of_range(methodology: ResolvedMethodology) -> None:
    bad = _spec("L4-ERR-001", "error-code", "L4", code="X", http_status=200)
    issues = check_error_code_http_status(bad, _ctx([bad], methodology))
    assert any(i.code == "error_code_http_status_out_of_range" for i in issues)


def test_error_code_http_status_zero_allowed(methodology: ResolvedMethodology) -> None:
    """0 = non-HTTP errors (queue / internal)"""
    spec = _spec("L4-ERR-001", "error-code", "L4", code="X", http_status=0)
    assert check_error_code_http_status(spec, _ctx([spec], methodology)) == []


# ── 14. configuration namespace ──────────────────────────────────────────


def test_configuration_namespace_duplicate(methodology: ResolvedMethodology) -> None:
    a = _spec("L4-CFG-001", "configuration", "L4", namespace="app", changeable_at_runtime=False)
    b = _spec("L4-CFG-002", "configuration", "L4", namespace="app", changeable_at_runtime=True)
    issues = check_configuration_namespace_unique([a, b], _ctx([a, b], methodology))
    assert any(i.code == "configuration_namespace_duplicate" for i in issues)


# ── 15. test-cases ───────────────────────────────────────────────────────


def test_test_case_e2e_without_user_scenario_warning(methodology: ResolvedMethodology) -> None:
    api = _spec("L4-API-001", "api-contract", "L4")
    tc = _spec(
        "L4-TC-001",
        "test-case",
        "L4",
        up_to=["L4-API-001"],
        kind="e2e",
        runner="playwright",
        severity="high",
    )
    issues = check_test_case_kind_matches_subject(tc, _ctx([api, tc], methodology))
    assert any(i.code == "test_case_e2e_without_scenario" for i in issues)


def test_test_case_e2e_with_user_scenario_ok(methodology: ResolvedMethodology) -> None:
    sc = _spec("L3-USC-001", "user-scenario", "L3")
    tc = _spec(
        "L4-TC-001",
        "test-case",
        "L4",
        up_to=["L3-USC-001"],
        kind="e2e",
        runner="playwright",
        severity="high",
    )
    assert check_test_case_kind_matches_subject(tc, _ctx([sc, tc], methodology)) == []


def test_test_case_contract_targets_integration(methodology: ResolvedMethodology) -> None:
    bh = _spec("L3-BHV-001", "behavior", "L3")
    tc = _spec(
        "L4-TC-001",
        "test-case",
        "L4",
        up_to=["L3-BHV-001"],
        kind="contract",
        runner="pytest",
        severity="medium",
    )
    issues = check_test_case_kind_matches_subject(tc, _ctx([bh, tc], methodology))
    assert any(i.code == "test_case_contract_without_integration" for i in issues)


def test_test_case_contract_up_to_integration_contract_ok(
    methodology: ResolvedMethodology,
) -> None:
    """a known limitation: kind=contract anchored on integration-contract is a valid shape.

    Regression guard for the contradiction where `up_to_types` excluded
    `integration-contract` while this validator demanded exactly that.
    Both validators must accept the same configuration.
    """
    ic = _spec("L4-IC-001", "integration-contract", "L4")
    tc = _spec(
        "L4-TC-001",
        "test-case",
        "L4",
        up_to=["L4-IC-001"],
        kind="contract",
        runner="pytest",
        severity="medium",
    )
    ctx = _ctx([ic, tc], methodology)
    assert check_test_case_kind_matches_subject(tc, ctx) == []
    assert check_pyramid_adjacency(tc, ctx) == []


def test_test_case_duplicate_warning(methodology: ResolvedMethodology) -> None:
    a = _spec(
        "L4-TC-001",
        "test-case",
        "L4",
        title="login happy path",
        up_to=["L3-BHV-001"],
        kind="integration",
        runner="pytest",
        severity="high",
    )
    b = _spec(
        "L4-TC-002",
        "test-case",
        "L4",
        title="login happy path",
        up_to=["L3-BHV-001"],
        kind="integration",
        runner="pytest",
        severity="high",
    )
    issues = check_test_case_id_unique([a, b], _ctx([a, b], methodology))
    assert any(i.code == "test_case_duplicate_subject_kind" for i in issues)


# ── 16. metrics: priority + test-coverage ────────────────────────────────


def test_priority_tagged_metric(methodology: ResolvedMethodology) -> None:
    tagged = _spec("L3-BHV-001", "behavior", "L3", priority="must")
    untagged = _spec("L3-BHV-002", "behavior", "L3")
    result = _priority_tagged(
        [tagged, untagged],
        _metric_ctx([tagged, untagged], methodology, "L3"),
    )
    assert result.value == 0.5


def test_behavior_test_coverage_metric(methodology: ResolvedMethodology) -> None:
    covered = _spec("L3-BHV-001", "behavior", "L3")
    uncovered = _spec("L3-BHV-002", "behavior", "L3")
    tc = _spec(
        "L4-TC-001",
        "test-case",
        "L4",
        up_to=["L3-BHV-001"],
        kind="unit",
        runner="pytest",
        severity="medium",
    )
    result = _behavior_test_coverage(
        [covered, uncovered, tc],
        _metric_ctx([covered, uncovered, tc], methodology, "L3"),
    )
    assert result.value == 0.5


def test_scenario_test_coverage_metric(methodology: ResolvedMethodology) -> None:
    sc1 = _spec("L3-USC-001", "user-scenario", "L3")
    sc2 = _spec("L3-USC-002", "user-scenario", "L3")
    e2e = _spec(
        "L4-TC-001",
        "test-case",
        "L4",
        up_to=["L3-USC-001"],
        kind="e2e",
        runner="playwright",
        severity="high",
    )
    # sc2 has a test-case but it's a `unit` — doesn't count for scenario coverage
    unit_for_scenario = _spec(
        "L4-TC-002",
        "test-case",
        "L4",
        up_to=["L3-USC-002"],
        kind="unit",
        runner="pytest",
        severity="medium",
    )
    result = _scenario_test_coverage(
        [sc1, sc2, e2e, unit_for_scenario],
        _metric_ctx([sc1, sc2, e2e, unit_for_scenario], methodology, "L3"),
    )
    assert result.value == 0.5


# ── 17. unused-import sanity ─────────────────────────────────────────────

_ = cast  # keep import alive across versions of Spec extras handling
