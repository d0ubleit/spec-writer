"""Built-in validators for the ``core-traceability`` mixin.

- ``check_pyramid_adjacency`` (per-spec): ``up_to`` must reference a
  spec one layer up (principle 1).
- ``check_dangling_refs`` (cross-spec): every cross-spec reference
  must point to an existing spec id (principle 2).
- ``check_vertical_symmetry`` (cross-spec): ``A.up_to=[B]`` ↔
  ``B.down_to=[A]`` invariant.
- ``check_english_output`` (per-spec): ``title`` and string-typed
  frontmatter fields must be Latin-script; ``[SOURCE]`` blocks are
  exempt (principle 8).
- ``check_required_sections`` (per-spec): every section marked
  ``required: true`` in the methodology must appear as a heading in
  the spec body.
- ``check_required_frontmatter`` (per-spec): every methodology-declared
  required field beyond the engine-mandatory ones must be present.
"""

from __future__ import annotations

import re
import unicodedata

from spec_writer.methodology.validators import (
    ValidationIssue,
    ValidatorContext,
    register_validator,
)
from spec_writer.types import Spec

# ── pyramid adjacency ────────────────────────────────────────────────────


@register_validator("validators.check_pyramid_adjacency", scope="per-spec")
def check_pyramid_adjacency(spec: Spec, ctx: ValidatorContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    layers = ctx.methodology.layers
    layer_index = {layer.id: idx for idx, layer in enumerate(layers)}
    role_by_id = {layer.id: layer.role for layer in layers}

    cur = layer_index.get(spec.layer)
    if cur is None:
        return issues  # unknown layer — out of scope

    # Cross-cutting types are exempt from pyramid-adjacency by design:
    # their up_to may reference any upper layer, not just the immediate
    # one. See docs/methodology.md.
    spec_type = ctx.methodology.spec_types.get(spec.type)
    if spec_type is not None and spec_type.cross_cutting:
        return issues

    role = role_by_id.get(spec.layer)
    if role == "source-boundary":
        if spec.up_to:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="pyramid_source_has_up_to",
                    spec_id=spec.id,
                    field="up_to",
                    message=(
                        f"{spec.id} is on source-boundary layer {spec.layer}; "
                        f"up_to must be [], got {spec.up_to!r}"
                    ),
                    fix_hint="Remove up_to entries on the source-boundary layer.",
                )
            )
        return issues

    if not spec.up_to:
        issues.append(
            ValidationIssue(
                severity="error",
                code="pyramid_missing_up_to",
                spec_id=spec.id,
                field="up_to",
                message=(
                    f"{spec.id} on layer {spec.layer} must reference a spec on the "
                    f"immediately-higher layer via up_to."
                ),
                fix_hint="Add the parent spec id to up_to.",
            )
        )
        return issues

    # If the spec type declares `up_to_types`, validate against THAT
    # list — type author has explicit intent about which parent types
    # are valid (e.g. tech-choice anchors at capability OR arch-view OR
    # vision; not strictly L3). See the design rationale.
    declared_parent_types = list(spec_type.up_to_types) if spec_type else []

    expected_parent = layers[cur - 1].id
    for parent_id in spec.up_to:
        parent_spec = ctx.corpus.get(parent_id)
        if parent_spec is None:
            # dangling_refs validator will catch the missing reference;
            # adjacency can't be checked without knowing the parent's layer.
            continue
        if declared_parent_types:
            if parent_spec.type not in declared_parent_types:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="pyramid_wrong_parent_type",
                        spec_id=spec.id,
                        field="up_to",
                        message=(
                            f"{spec.id}.up_to[{parent_id}] points to a spec of "
                            f"type {parent_spec.type!r}; type {spec.type!r} "
                            f"declares up_to_types={declared_parent_types}."
                        ),
                        fix_hint=(
                            f"Re-anchor on a spec whose type is in "
                            f"{declared_parent_types}, or extend the mixin's "
                            f"up_to_types declaration."
                        ),
                    )
                )
            continue
        if parent_spec.layer != expected_parent:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="pyramid_skip_level",
                    spec_id=spec.id,
                    field="up_to",
                    message=(
                        f"{spec.id}.up_to[{parent_id}] points to layer "
                        f"{parent_spec.layer}; expected {expected_parent} "
                        f"(immediately above {spec.layer})."
                    ),
                    fix_hint=(
                        f"Decompose through {expected_parent} first; "
                        f"skip-level references are forbidden."
                    ),
                )
            )
    return issues


# ── dangling refs ────────────────────────────────────────────────────────


@register_validator("validators.check_dangling_refs", scope="cross-spec")
def check_dangling_refs(specs: list[Spec], ctx: ValidatorContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    known = ctx.corpus
    for spec in specs:
        for parent_id in spec.up_to:
            if parent_id not in known:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="dangling_ref",
                        spec_id=spec.id,
                        field="up_to",
                        message=f"{spec.id}.up_to[{parent_id}] does not exist in corpus.",
                    )
                )
        for child_id in spec.down_to:
            if child_id not in known:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="dangling_ref",
                        spec_id=spec.id,
                        field="down_to",
                        message=f"{spec.id}.down_to[{child_id}] does not exist in corpus.",
                    )
                )
        for rel in spec.relates_to:
            if rel.id not in known:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="dangling_ref",
                        spec_id=spec.id,
                        field="relates_to",
                        message=(f"{spec.id}.relates_to[{rel.id}] does not exist in corpus."),
                    )
                )
    return issues


# ── vertical symmetry ────────────────────────────────────────────────────


@register_validator("validators.check_vertical_symmetry", scope="cross-spec")
def check_vertical_symmetry(specs: list[Spec], ctx: ValidatorContext) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    by_id = ctx.corpus
    for spec in specs:
        for parent_id in spec.up_to:
            parent = by_id.get(parent_id)
            if parent is None:
                continue  # dangling_refs handles
            if spec.id not in parent.down_to:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="vertical_asymmetry",
                        spec_id=spec.id,
                        field="up_to",
                        message=(
                            f"{spec.id}.up_to includes {parent_id}, but "
                            f"{parent_id}.down_to is missing {spec.id}."
                        ),
                        fix_hint=(
                            "Engine should symmetrize automatically; if this "
                            "appears in hand-written specs, add the reverse edge."
                        ),
                    )
                )
        for child_id in spec.down_to:
            child = by_id.get(child_id)
            if child is None:
                continue
            if spec.id not in child.up_to:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="vertical_asymmetry",
                        spec_id=spec.id,
                        field="down_to",
                        message=(
                            f"{spec.id}.down_to includes {child_id}, but "
                            f"{child_id}.up_to is missing {spec.id}."
                        ),
                    )
                )
    return issues


# ── english-output ───────────────────────────────────────────────────────


def _contains_non_latin_letter(text: str) -> bool:
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "LATIN" not in name:
            return True
    return False


# ── structural completeness ──────────────────────────────────────────────


_ENGINE_MANDATORY_FIELDS = frozenset({"id", "type", "layer", "title", "up_to", "down_to"})


def _heading_pattern(title: str) -> re.Pattern[str]:
    return re.compile(rf"^#+\s*{re.escape(title)}\s*$", re.MULTILINE | re.IGNORECASE)


@register_validator("validators.check_required_sections", scope="per-spec")
def check_required_sections(spec: Spec, ctx: ValidatorContext) -> list[ValidationIssue]:
    """Every methodology-declared required section must appear as a heading."""
    spec_type = ctx.methodology.spec_types.get(spec.type)
    if spec_type is None:
        return []
    issues: list[ValidationIssue] = []
    body = spec.body or ""
    for section in spec_type.sections:
        if not section.required:
            continue
        if not _heading_pattern(section.title).search(body):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_required_section",
                    spec_id=spec.id,
                    field="body",
                    message=(
                        f"{spec.id}: required section {section.title!r} "
                        f"(id={section.id}) missing from body."
                    ),
                    fix_hint=(
                        f"Add a `## {section.title}` heading and the corresponding "
                        f"content. Methodology guidance: {section.guidance or '(none)'}"
                    ),
                )
            )
    return issues


@register_validator("validators.check_required_frontmatter", scope="per-spec")
def check_required_frontmatter(spec: Spec, ctx: ValidatorContext) -> list[ValidationIssue]:
    """Methodology-declared required frontmatter fields must be present.

    Looks first in Spec's declared attributes (tags, status, implemented_by,
    ...) and falls back to ``model_extra`` for methodology-specific fields
    like ``method``/``path``/``auth``.
    """
    spec_type = ctx.methodology.spec_types.get(spec.type)
    if spec_type is None:
        return []
    issues: list[ValidationIssue] = []
    extras = spec.model_extra or {}
    for field in spec_type.required_frontmatter:
        if field.name in _ENGINE_MANDATORY_FIELDS:
            continue
        # Prefer attribute on Spec (covers tags/status/implemented_by/etc.)
        # else fall back to extras dict.
        if hasattr(spec, field.name):
            value = getattr(spec, field.name)
        else:
            value = extras.get(field.name)
        # tags / implemented_by may be empty lists — count those as missing-data
        # for required fields, because empty defeats the point.
        missing = value is None
        if not missing and isinstance(value, str):
            missing = len(value) == 0
        if not missing and isinstance(value, list):
            missing = len(value) == 0  # type: ignore[arg-type]
        if not missing and isinstance(value, dict):
            missing = len(value) == 0  # type: ignore[arg-type]
        if missing:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_required_frontmatter",
                    spec_id=spec.id,
                    field=field.name,
                    message=(
                        f"{spec.id}: required frontmatter field {field.name!r} missing or empty."
                    ),
                    fix_hint=(
                        f"{field.description or '(no description)'} "
                        f"{field.llm_guidance.strip() if field.llm_guidance else ''}"
                    ).strip(),
                )
            )
    return issues


@register_validator("validators.check_surface_when_missing", scope="per-spec")
def check_surface_when_missing(spec: Spec, ctx: ValidatorContext) -> list[ValidationIssue]:
    """Emit WARNING for optional fields marked `surface_when_missing: true`.

    Use case: project marks an *optional* field (e.g. `priority`, `owner`,
    `confidence`) as "near-required" — not strict enough to block, but the
    quiet skip is hidden debt. Validator surfaces every missing instance
    as a warning so they're discoverable. Generic mechanism — no field
    name is hard-coded. See the design rationale.
    """
    spec_type = ctx.methodology.spec_types.get(spec.type)
    if spec_type is None:
        return []
    issues: list[ValidationIssue] = []
    extras = spec.model_extra or {}
    for field in spec_type.optional_frontmatter:
        if not field.surface_when_missing:
            continue
        if field.name in _ENGINE_MANDATORY_FIELDS:
            continue
        if hasattr(spec, field.name):
            value = getattr(spec, field.name)
        else:
            value = extras.get(field.name)
        missing = value is None
        if not missing and isinstance(value, str):
            missing = len(value) == 0
        if not missing and isinstance(value, list):
            missing = len(value) == 0  # type: ignore[arg-type]
        if not missing and isinstance(value, dict):
            missing = len(value) == 0  # type: ignore[arg-type]
        if missing:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="surfaced_missing_optional_frontmatter",
                    spec_id=spec.id,
                    field=field.name,
                    message=(
                        f"{spec.id}: optional frontmatter field {field.name!r} is "
                        f"missing; mixin marked it `surface_when_missing: true` to "
                        f"prevent quiet accumulation."
                    ),
                    fix_hint=(
                        f"{field.description or '(no description)'} "
                        f"{field.llm_guidance.strip() if field.llm_guidance else ''}"
                    ).strip(),
                )
            )
    return issues


# ── english output ───────────────────────────────────────────────────────


@register_validator("validators.check_english_output", scope="per-spec")
def check_english_output(spec: Spec, ctx: ValidatorContext) -> list[ValidationIssue]:
    del ctx  # unused; signature uniformity
    issues: list[ValidationIssue] = []

    if _contains_non_latin_letter(spec.title) and "[SOURCE]" not in spec.title:
        issues.append(
            ValidationIssue(
                severity="error",
                code="non_english_title",
                spec_id=spec.id,
                field="title",
                message=(
                    f"{spec.id}.title contains non-Latin letters; output must "
                    f"be English (principle 8)."
                ),
                fix_hint="Translate the title to English; quote source text only with [SOURCE] gloss.",
            )
        )

    if spec.status and _contains_non_latin_letter(spec.status):
        issues.append(
            ValidationIssue(
                severity="error",
                code="non_english_field",
                spec_id=spec.id,
                field="status",
                message=f"{spec.id}.status contains non-Latin letters.",
            )
        )

    for tag in spec.tags:
        if _contains_non_latin_letter(tag):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="non_english_field",
                    spec_id=spec.id,
                    field="tags",
                    message=f"{spec.id}.tags[{tag!r}] contains non-Latin letters.",
                )
            )
            break  # one issue per spec on tags

    return issues
