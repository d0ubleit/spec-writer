"""Validators for built-in mixins shipped beyond core-traceability.

Project-agnostic. Each mixin under ``methodology/_builtin/mixins/``
that needs custom rules references its functions here via the
``validators.<name>`` ``func_ref`` strings. Registered at import time.

Mixin → validator mapping (one block per mixin):

- glossary             → check_glossary_canonical_unique,
                         check_glossary_aliases_unique
- requirements-traceability → check_traces_to_format
- architecture-views   → check_arch_view_singleton_per_kind
- error-codes          → check_error_code_uniqueness,
                         check_error_code_http_status
- configuration        → check_configuration_namespace_unique
- test-cases           → check_test_case_id_unique,
                         check_test_case_kind_matches_subject
- user-scenarios       → check_scenario_steps_linked
- ui-screens           → check_ui_screen_routes_unique
- ui-components        → check_ui_component_testids_unique
- state-machines       → check_state_machine_reachability
- tech-stack           → check_tech_choice_uniqueness_per_category
- rbac-matrix          → check_rbac_matrix_singleton,
                         check_rbac_roles_resolved,
                         check_rbac_matrix_completeness

All validators are tolerant of partially-formed specs: a malformed
yaml-block is reported as a single issue, not raised, so downstream
validators can still run on the rest of the corpus.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, cast

import yaml

from spec_writer.methodology.validators import (
    ValidationIssue,
    ValidatorContext,
    register_validator,
)
from spec_writer.types import Spec

# ── shared helpers ───────────────────────────────────────────────────────


def _extract_section(body: str, title: str) -> str | None:
    """Return the markdown content under a heading, or None if missing.

    Matches `^##+ Title` (any heading level, case-insensitive). Body
    runs until the next heading at any level or EOF.
    """
    pattern = re.compile(
        rf"^(#+)\s*{re.escape(title)}\s*$\n(.*?)(?=^#+\s|\Z)",
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(body)
    if match is None:
        return None
    return match.group(2).strip()


def _extract_yaml_block(section_content: str) -> Any | None:
    """Pull the first ```yaml … ``` fence and parse it; fall back to plain text."""
    fenced = re.search(
        r"```(?:ya?ml)?\s*\n(.*?)\n```",
        section_content,
        re.DOTALL,
    )
    payload = fenced.group(1) if fenced else section_content
    try:
        return yaml.safe_load(payload)
    except yaml.YAMLError:
        return None


def _extras(spec: Spec) -> dict[str, Any]:
    return spec.model_extra or {}


def _roles_of(spec: Spec) -> list[str] | None:
    """Read the `roles` extras field. None means the type doesn't declare it."""
    value = _extras(spec).get("roles")
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in cast(list[Any], value)]
    if isinstance(value, str):
        return [value]
    return []


# ── user-scenarios: steps linkage ────────────────────────────────────────


@register_validator("validators.check_scenario_steps_linked", scope="per-spec")
def check_scenario_steps_linked(spec: Spec, ctx: ValidatorContext) -> list[ValidationIssue]:
    """Each user-scenario step must reference an existing ui-screen
    (and optionally an existing ui-component).
    """
    if spec.type != "user-scenario":
        return []
    issues: list[ValidationIssue] = []
    section = _extract_section(spec.body, "Steps")
    if section is None:
        return issues  # required-sections validator handles missing heading

    parsed = _extract_yaml_block(section)
    if not isinstance(parsed, list):
        issues.append(
            ValidationIssue(
                severity="error",
                code="user_scenario_steps_unparsed",
                spec_id=spec.id,
                field="body",
                message=f"{spec.id}: Steps section is not a YAML list",
                fix_hint="Steps must be a YAML list of `{step, screen, action, ...}` blocks.",
            )
        )
        return issues

    by_type: dict[str, set[str]] = {"ui-screen": set(), "ui-component": set()}
    for other in ctx.corpus.values():
        if other.type in by_type:
            by_type[other.type].add(other.id)

    items = cast(list[Any], parsed)
    for idx, raw_step in enumerate(items):
        if not isinstance(raw_step, dict):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="user_scenario_step_shape",
                    spec_id=spec.id,
                    field="body",
                    message=f"{spec.id}: Steps[{idx}] must be a mapping, got {type(raw_step).__name__}",
                )
            )
            continue
        step = cast(dict[str, Any], raw_step)
        screen_ref = step.get("screen")
        if not isinstance(screen_ref, str) or not screen_ref:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="user_scenario_step_missing_screen",
                    spec_id=spec.id,
                    field="body",
                    message=f"{spec.id}: Steps[{idx}] missing `screen:` reference",
                    fix_hint="Every step must point at a ui-screen id (e.g. screen: L4-SCR-007).",
                )
            )
        elif screen_ref not in by_type["ui-screen"]:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="user_scenario_step_unknown_screen",
                    spec_id=spec.id,
                    field="body",
                    message=f"{spec.id}: Steps[{idx}].screen={screen_ref!r} is not an existing ui-screen",
                    fix_hint="Record the ui-screen spec first or fix the reference.",
                )
            )
        component_ref = step.get("component")
        if (
            isinstance(component_ref, str)
            and component_ref
            and component_ref not in by_type["ui-component"]
        ):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="user_scenario_step_unknown_component",
                    spec_id=spec.id,
                    field="body",
                    message=f"{spec.id}: Steps[{idx}].component={component_ref!r} is not an existing ui-component",
                )
            )
    return issues


# ── ui-screens: route uniqueness ─────────────────────────────────────────


@register_validator("validators.check_ui_screen_routes_unique", scope="cross-spec")
def check_ui_screen_routes_unique(
    specs: list[Spec], ctx: ValidatorContext
) -> list[ValidationIssue]:
    # Route uniqueness is corpus-wide .
    del specs
    issues: list[ValidationIssue] = []
    seen: dict[str, str] = {}
    for spec in ctx.corpus.values():
        if spec.type != "ui-screen":
            continue
        route = _extras(spec).get("route")
        if not isinstance(route, str) or not route:
            continue  # required-frontmatter handles missing
        prior = seen.get(route)
        if prior is not None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="ui_screen_route_duplicate",
                    spec_id=spec.id,
                    field="route",
                    message=f"{spec.id}.route={route!r} already used by {prior}",
                    fix_hint="Routes must be unique across ui-screens.",
                )
            )
        else:
            seen[route] = spec.id
    return issues


# ── ui-components: testid uniqueness ─────────────────────────────────────


@register_validator("validators.check_ui_component_testids_unique", scope="cross-spec")
def check_ui_component_testids_unique(
    specs: list[Spec], ctx: ValidatorContext
) -> list[ValidationIssue]:
    # testids globally unique for e2e .
    del specs
    issues: list[ValidationIssue] = []
    seen: dict[str, str] = {}
    for spec in ctx.corpus.values():
        if spec.type != "ui-component":
            continue
        testid = _extras(spec).get("testid")
        if not isinstance(testid, str) or not testid:
            continue
        prior = seen.get(testid)
        if prior is not None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="ui_component_testid_duplicate",
                    spec_id=spec.id,
                    field="testid",
                    message=f"{spec.id}.testid={testid!r} already used by {prior}",
                    fix_hint="data-testid values must be unique across ui-components.",
                )
            )
        else:
            seen[testid] = spec.id
    return issues


# ── state-machines: reachability ─────────────────────────────────────────


@register_validator("validators.check_state_machine_reachability", scope="per-spec")
def check_state_machine_reachability(spec: Spec, ctx: ValidatorContext) -> list[ValidationIssue]:
    del ctx
    if spec.type != "state-machine":
        return []
    issues: list[ValidationIssue] = []

    states_section = _extract_section(spec.body, "States")
    transitions_section = _extract_section(spec.body, "Transitions")
    if states_section is None or transitions_section is None:
        return issues  # required-sections will surface

    states = _extract_yaml_block(states_section)
    transitions = _extract_yaml_block(transitions_section)
    if not isinstance(states, list) or not isinstance(transitions, list):
        issues.append(
            ValidationIssue(
                severity="error",
                code="state_machine_unparsed",
                spec_id=spec.id,
                field="body",
                message=f"{spec.id}: States and Transitions must each be a YAML list",
            )
        )
        return issues

    state_items = cast(list[Any], states)
    state_ids: set[str] = set()
    initials: set[str] = set()
    terminals: set[str] = set()
    for raw_state in state_items:
        if not isinstance(raw_state, dict):
            continue
        s = cast(dict[str, Any], raw_state)
        sid = s.get("id")
        if not isinstance(sid, str):
            continue
        state_ids.add(sid)
        if s.get("initial") is True:
            initials.add(sid)
        if s.get("terminal") is True:
            terminals.add(sid)

    if len(initials) == 0:
        issues.append(
            ValidationIssue(
                severity="error",
                code="state_machine_no_initial",
                spec_id=spec.id,
                field="body",
                message=f"{spec.id}: no state marked `initial: true`",
            )
        )
    elif len(initials) > 1:
        issues.append(
            ValidationIssue(
                severity="error",
                code="state_machine_multiple_initials",
                spec_id=spec.id,
                field="body",
                message=f"{spec.id}: multiple initial states: {sorted(initials)}",
            )
        )

    edges: list[tuple[str, str]] = []
    for raw_t in cast(list[Any], transitions):
        if not isinstance(raw_t, dict):
            continue
        t = cast(dict[str, Any], raw_t)
        frm, to = t.get("from"), t.get("to")
        if isinstance(frm, str) and isinstance(to, str):
            edges.append((frm, to))
            for endpoint, label in ((frm, "from"), (to, "to")):
                if endpoint not in state_ids:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="state_machine_unknown_state",
                            spec_id=spec.id,
                            field="body",
                            message=f"{spec.id}: transition `{label}: {endpoint!r}` references unknown state",
                        )
                    )

    # Reachability from initial(s)
    if initials:
        reachable: set[str] = set(initials)
        changed = True
        while changed:
            changed = False
            for frm, to in edges:
                if frm in reachable and to not in reachable:
                    reachable.add(to)
                    changed = True
        unreachable = state_ids - reachable
        for state_id in sorted(unreachable):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="state_machine_unreachable",
                    spec_id=spec.id,
                    field="body",
                    message=f"{spec.id}: state {state_id!r} is unreachable from initial state(s)",
                )
            )

    # Non-terminal states must have ≥1 outgoing
    has_outgoing = {frm for frm, _to in edges}
    for state_id in sorted(state_ids - terminals - has_outgoing):
        issues.append(
            ValidationIssue(
                severity="error",
                code="state_machine_dead_end",
                spec_id=spec.id,
                field="body",
                message=(
                    f"{spec.id}: state {state_id!r} is non-terminal but has no outgoing transitions"
                ),
                fix_hint="Either mark it `terminal: true` or add a transition out of it.",
            )
        )

    return issues


# ── tech-stack: per-category uniqueness ──────────────────────────────────


@register_validator("validators.check_tech_choice_uniqueness_per_category", scope="cross-spec")
def check_tech_choice_uniqueness_per_category(
    specs: list[Spec], ctx: ValidatorContext
) -> list[ValidationIssue]:
    # Corpus-wide uniqueness .
    del specs
    issues: list[ValidationIssue] = []
    chosen: dict[tuple[str, str], str] = {}  # (category, phase) → spec_id
    for spec in ctx.corpus.values():
        if spec.type != "tech-choice":
            continue
        # `status` is an explicit Spec attribute (lifecycle-fields mixin),
        # not in model_extra; phase/category live in extras.
        if spec.status != "chosen":
            continue
        extras = _extras(spec)
        category = extras.get("category")
        phase = extras.get("phase")
        if not isinstance(category, str) or not isinstance(phase, str):
            continue
        key = (category, phase)
        prior = chosen.get(key)
        if prior is not None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="tech_choice_duplicate_chosen",
                    spec_id=spec.id,
                    field="status",
                    message=(
                        f"{spec.id}: category={category!r} phase={phase!r} already "
                        f"has a chosen tech-choice ({prior})"
                    ),
                    fix_hint=(
                        "At most one tech-choice per (category, phase) may be "
                        "`status: chosen`. Mark the older one `superseded` or `deprecated`."
                    ),
                )
            )
        else:
            chosen[key] = spec.id
    return issues


# ── rbac-matrix: singleton, role resolution, completeness ────────────────


def _find_rbac_matrix(specs: Iterable[Spec]) -> list[Spec]:
    return [s for s in specs if s.type == "rbac-matrix"]


def _matrix_roles(matrix: Spec) -> set[str]:
    section = _extract_section(matrix.body, "Roles")
    if section is None:
        return set()
    parsed = _extract_yaml_block(section)
    if not isinstance(parsed, list):
        return set()
    out: set[str] = set()
    for entry in cast(list[Any], parsed):
        if isinstance(entry, dict):
            entry_dict = cast(dict[str, Any], entry)
            rid = entry_dict.get("id")
            if isinstance(rid, str):
                out.add(rid)
        elif isinstance(entry, str):
            out.add(entry)
    return out


def _roles_used_in_corpus(specs: Iterable[Spec]) -> dict[str, list[str]]:
    """role_id → list of spec_ids that reference it."""
    used: dict[str, list[str]] = {}
    for spec in specs:
        roles = _roles_of(spec)
        if roles is None:
            continue
        for role in roles:
            used.setdefault(role, []).append(spec.id)
    return used


@register_validator("validators.check_rbac_matrix_singleton", scope="cross-spec")
def check_rbac_matrix_singleton(specs: list[Spec], ctx: ValidatorContext) -> list[ValidationIssue]:
    # Singleton check is corpus-wide: a duplicate matrix anywhere in
    # the corpus must surface even from a scoped validate ().
    del specs
    matrices = _find_rbac_matrix(ctx.corpus.values())
    if len(matrices) <= 1:
        return []
    return [
        ValidationIssue(
            severity="error",
            code="rbac_matrix_not_singleton",
            spec_id=m.id,
            field="type",
            message=(
                f"{m.id}: multiple rbac-matrix specs exist ({len(matrices)}); only one allowed"
            ),
            fix_hint="Merge the matrices into one spec; mark the rest as deprecated and delete.",
        )
        for m in matrices
    ]


@register_validator("validators.check_rbac_roles_resolved", scope="cross-spec")
def check_rbac_roles_resolved(specs: list[Spec], ctx: ValidatorContext) -> list[ValidationIssue]:
    # Find the matrix anywhere in the corpus, then check roles in the
    # scoped subset only (so issue attribution stays inside scope).
    matrices = _find_rbac_matrix(ctx.corpus.values())
    if not matrices:
        return []  # no matrix → singleton validator already errored, or matrix isn't active
    matrix = matrices[0]
    known = _matrix_roles(matrix)
    # Built-in pseudo-roles that don't need to appear in matrix.Roles
    pseudo = {"anonymous", "system", "external"}
    issues: list[ValidationIssue] = []
    for spec in specs:
        roles = _roles_of(spec)
        if roles is None:
            continue
        for role in roles:
            if role in known or role in pseudo:
                continue
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="rbac_unknown_role",
                    spec_id=spec.id,
                    field="roles",
                    message=(
                        f"{spec.id}.roles references {role!r} which is not in "
                        f"the rbac-matrix Roles list"
                    ),
                    fix_hint=(
                        f"Either add {role!r} to the rbac-matrix Roles section "
                        f"or fix the typo in this spec."
                    ),
                )
            )
    return issues


@register_validator("validators.check_rbac_matrix_completeness", scope="cross-spec")
def check_rbac_matrix_completeness(
    specs: list[Spec], ctx: ValidatorContext
) -> list[ValidationIssue]:
    # Completeness is a corpus-wide property: a role used anywhere in
    # the corpus is not an orphan even if scoped validate hides the
    # caller from `specs`.
    del specs
    corpus_specs = list(ctx.corpus.values())
    matrices = _find_rbac_matrix(corpus_specs)
    if not matrices:
        return []
    matrix = matrices[0]
    matrix_roles = _matrix_roles(matrix)
    used = _roles_used_in_corpus(corpus_specs)
    pseudo = {"anonymous", "system", "external"}
    issues: list[ValidationIssue] = []
    for role in sorted(matrix_roles):
        if role in pseudo:
            continue
        callers = used.get(role, [])
        if not callers:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="rbac_role_orphan",
                    spec_id=matrix.id,
                    field="body",
                    message=(
                        f"role {role!r} defined in rbac-matrix but not used by any "
                        f"api-contract / ui-screen / async-job / integration-contract"
                    ),
                    fix_hint=(
                        "Either remove the role or grant it on at least one resource "
                        "and reference it from a code-boundary spec."
                    ),
                )
            )
    return issues


# ── glossary: canonical + alias uniqueness ───────────────────────────────


@register_validator("validators.check_glossary_canonical_unique", scope="cross-spec")
def check_glossary_canonical_unique(
    specs: list[Spec], ctx: ValidatorContext
) -> list[ValidationIssue]:
    # Corpus-wide uniqueness .
    del specs
    issues: list[ValidationIssue] = []
    seen: dict[str, str] = {}
    for spec in ctx.corpus.values():
        if spec.type != "term":
            continue
        canonical = _extras(spec).get("canonical")
        if not isinstance(canonical, str) or not canonical:
            continue
        prior = seen.get(canonical)
        if prior is not None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="glossary_canonical_duplicate",
                    spec_id=spec.id,
                    field="canonical",
                    message=f"{spec.id}.canonical={canonical!r} already defined by {prior}",
                    fix_hint="Canonical forms must be unique across terms.",
                )
            )
        else:
            seen[canonical] = spec.id
    return issues


@register_validator("validators.check_glossary_aliases_unique", scope="cross-spec")
def check_glossary_aliases_unique(
    specs: list[Spec], ctx: ValidatorContext
) -> list[ValidationIssue]:
    # Corpus-wide uniqueness .
    del specs
    issues: list[ValidationIssue] = []
    seen: dict[str, str] = {}  # alias → owning spec_id
    for spec in ctx.corpus.values():
        if spec.type != "term":
            continue
        aliases = _extras(spec).get("aliases")
        if not isinstance(aliases, list):
            continue
        for raw_alias in cast(list[Any], aliases):
            if not isinstance(raw_alias, str) or not raw_alias:
                continue
            prior = seen.get(raw_alias)
            if prior is not None and prior != spec.id:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="glossary_alias_duplicate",
                        spec_id=spec.id,
                        field="aliases",
                        message=f"{spec.id}.aliases[{raw_alias!r}] also owned by {prior}",
                        fix_hint="Move the alias to one canonical owner or rename it.",
                    )
                )
            else:
                seen[raw_alias] = spec.id
    return issues


# ── requirements-traceability: format check ──────────────────────────────


_TRACES_TO_PATTERN = re.compile(r"^[A-Z]{2,6}-\d{1,4}$")


@register_validator("validators.check_traces_to_format", scope="per-spec")
def check_traces_to_format(spec: Spec, ctx: ValidatorContext) -> list[ValidationIssue]:
    """Each entry in `traces_to` must match `<UPPER>-<digits>` (e.g. FR-12, NFR-04)."""
    del ctx
    traces = _extras(spec).get("traces_to")
    if traces is None:
        return []
    if not isinstance(traces, list):
        return [
            ValidationIssue(
                severity="error",
                code="traces_to_not_list",
                spec_id=spec.id,
                field="traces_to",
                message=f"{spec.id}.traces_to must be a list of requirement ids",
            )
        ]
    issues: list[ValidationIssue] = []
    for entry in cast(list[Any], traces):
        if not isinstance(entry, str) or not _TRACES_TO_PATTERN.match(entry):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="traces_to_bad_format",
                    spec_id=spec.id,
                    field="traces_to",
                    message=(
                        f"{spec.id}.traces_to has entry {entry!r}; expected "
                        f"`PREFIX-<digits>` like FR-12 / NFR-04 / US-19"
                    ),
                )
            )
    return issues


# ── architecture-views: singleton per kind ───────────────────────────────


@register_validator("validators.check_arch_view_singleton_per_kind", scope="cross-spec")
def check_arch_view_singleton_per_kind(
    specs: list[Spec], ctx: ValidatorContext
) -> list[ValidationIssue]:
    """system-context: exactly one. container-view / component-view:
    any number."""
    # Corpus-wide singleton check .
    del specs
    issues: list[ValidationIssue] = []
    by_kind: dict[str, list[str]] = {}
    for spec in ctx.corpus.values():
        if spec.type != "arch-view":
            continue
        kind = _extras(spec).get("view_kind")
        if isinstance(kind, str):
            by_kind.setdefault(kind, []).append(spec.id)
    sc = by_kind.get("system-context", [])
    if len(sc) > 1:
        for sid in sc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="arch_view_multiple_system_context",
                    spec_id=sid,
                    field="view_kind",
                    message=(
                        f"{sid}: multiple `system-context` views exist ({len(sc)}); only one allowed"
                    ),
                    fix_hint="Keep one system-context; demote others to container-view.",
                )
            )
    return issues


# ── error-codes: code uniqueness + http_status sanity ────────────────────


_HTTP_STATUS_RANGE = (400, 599)


@register_validator("validators.check_error_code_uniqueness", scope="cross-spec")
def check_error_code_uniqueness(specs: list[Spec], ctx: ValidatorContext) -> list[ValidationIssue]:
    # Corpus-wide uniqueness .
    del specs
    issues: list[ValidationIssue] = []
    seen: dict[str, str] = {}
    for spec in ctx.corpus.values():
        if spec.type != "error-code":
            continue
        code = _extras(spec).get("code")
        if not isinstance(code, str) or not code:
            continue
        prior = seen.get(code)
        if prior is not None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="error_code_duplicate",
                    spec_id=spec.id,
                    field="code",
                    message=f"{spec.id}.code={code!r} already used by {prior}",
                    fix_hint="Error codes must be unique across the project.",
                )
            )
        else:
            seen[code] = spec.id
    return issues


@register_validator("validators.check_error_code_http_status", scope="per-spec")
def check_error_code_http_status(spec: Spec, ctx: ValidatorContext) -> list[ValidationIssue]:
    del ctx
    if spec.type != "error-code":
        return []
    status_raw = _extras(spec).get("http_status")
    if status_raw is None or status_raw == "" or status_raw == 0:
        return []  # 0 = non-HTTP context, allowed
    try:
        status = int(status_raw)
    except (TypeError, ValueError):
        return [
            ValidationIssue(
                severity="error",
                code="error_code_http_status_invalid",
                spec_id=spec.id,
                field="http_status",
                message=f"{spec.id}.http_status={status_raw!r} is not an integer",
            )
        ]
    if status == 0:
        return []
    if not (_HTTP_STATUS_RANGE[0] <= status <= _HTTP_STATUS_RANGE[1]):
        return [
            ValidationIssue(
                severity="error",
                code="error_code_http_status_out_of_range",
                spec_id=spec.id,
                field="http_status",
                message=(
                    f"{spec.id}.http_status={status} outside 4xx/5xx range; "
                    f"use 0 for non-HTTP errors"
                ),
            )
        ]
    return []


# ── configuration: namespace uniqueness ──────────────────────────────────


@register_validator("validators.check_configuration_namespace_unique", scope="cross-spec")
def check_configuration_namespace_unique(
    specs: list[Spec], ctx: ValidatorContext
) -> list[ValidationIssue]:
    # Corpus-wide uniqueness .
    del specs
    issues: list[ValidationIssue] = []
    seen: dict[str, str] = {}
    for spec in ctx.corpus.values():
        if spec.type != "configuration":
            continue
        namespace = _extras(spec).get("namespace")
        if not isinstance(namespace, str) or not namespace:
            continue
        prior = seen.get(namespace)
        if prior is not None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="configuration_namespace_duplicate",
                    spec_id=spec.id,
                    field="namespace",
                    message=f"{spec.id}.namespace={namespace!r} already declared by {prior}",
                    fix_hint="One configuration spec per namespace; merge or rename.",
                )
            )
        else:
            seen[namespace] = spec.id
    return issues


# ── test-cases: id uniqueness + kind/subject coherence ───────────────────


_E2E_KINDS = frozenset({"e2e", "acceptance"})
_CONTRACT_KIND = "contract"


@register_validator("validators.check_test_case_id_unique", scope="cross-spec")
def check_test_case_id_unique(specs: list[Spec], ctx: ValidatorContext) -> list[ValidationIssue]:
    """spec.id is already unique by engine invariant; this validator
    instead guards against duplicate (kind, subject) anti-pattern."""
    # Corpus-wide duplicate check .
    del specs
    issues: list[ValidationIssue] = []
    seen: dict[tuple[str, str, str], str] = {}
    for spec in ctx.corpus.values():
        if spec.type != "test-case":
            continue
        kind = _extras(spec).get("kind")
        if not isinstance(kind, str):
            continue
        for parent_id in spec.up_to:
            key = (kind, parent_id, spec.title.strip().lower())
            prior = seen.get(key)
            if prior is not None:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="test_case_duplicate_subject_kind",
                        spec_id=spec.id,
                        field="title",
                        message=(
                            f"{spec.id}: another test-case ({prior}) already covers "
                            f"kind={kind!r} of {parent_id} with the same title"
                        ),
                        fix_hint="Rename one of them to clarify what it verifies.",
                    )
                )
            else:
                seen[key] = spec.id
    return issues


@register_validator("validators.check_test_case_kind_matches_subject", scope="per-spec")
def check_test_case_kind_matches_subject(
    spec: Spec, ctx: ValidatorContext
) -> list[ValidationIssue]:
    """e2e/acceptance tests should target user-scenario; contract
    tests should target integration-contract."""
    if spec.type != "test-case":
        return []
    kind = _extras(spec).get("kind")
    if not isinstance(kind, str):
        return []
    issues: list[ValidationIssue] = []
    parent_types: list[str] = []
    for parent_id in spec.up_to:
        parent = ctx.corpus.get(parent_id)
        if parent is not None:
            parent_types.append(parent.type)

    if kind in _E2E_KINDS:
        if parent_types and "user-scenario" not in parent_types:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="test_case_e2e_without_scenario",
                    spec_id=spec.id,
                    field="up_to",
                    message=(
                        f"{spec.id}: kind={kind!r} but up_to references "
                        f"{parent_types!r}; e2e/acceptance tests should up_to a user-scenario"
                    ),
                )
            )
    elif kind == _CONTRACT_KIND:
        if parent_types and "integration-contract" not in parent_types:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="test_case_contract_without_integration",
                    spec_id=spec.id,
                    field="up_to",
                    message=(
                        f"{spec.id}: kind='contract' but up_to references "
                        f"{parent_types!r}; contract tests should up_to an integration-contract"
                    ),
                )
            )
    return issues


__all__ = [
    "check_arch_view_singleton_per_kind",
    "check_configuration_namespace_unique",
    "check_error_code_http_status",
    "check_error_code_uniqueness",
    "check_glossary_aliases_unique",
    "check_glossary_canonical_unique",
    "check_rbac_matrix_completeness",
    "check_rbac_matrix_singleton",
    "check_rbac_roles_resolved",
    "check_scenario_steps_linked",
    "check_state_machine_reachability",
    "check_tech_choice_uniqueness_per_category",
    "check_test_case_id_unique",
    "check_test_case_kind_matches_subject",
    "check_traces_to_format",
    "check_ui_component_testids_unique",
    "check_ui_screen_routes_unique",
]
