"""Methodology loader — base.yaml + active mixins → ResolvedMethodology.

Phase 1 supports:
- ``adds_types`` (atomic-mode mixin)
- ``extends_types`` (section-mode mixin) for sections + frontmatter
- ``extends_all_types`` for cross-cutting required/optional frontmatter
- ``extends_field`` enum merging
- ``validators``, ``hooks``, ``metrics`` — stored as opaque configs
- ``generator_hints`` per spec type
- mixin targeting via ``layer_role`` / ``layer_name`` / ``layer``

Conflicts (duplicate type id, duplicate section id within a type,
unknown target layer, unknown active mixin) are **collected** and
raised together as a single ``MethodologyLoadError`` per
No prioritization / silent winners.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, cast

import yaml

from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.methodology.sources import MixinSource
from spec_writer.types import FrontmatterField, Layer, Section, SpecType


class MethodologyLoadError(RuntimeError):
    """Raised when base/mixin composition has unresolved conflicts."""


JsonDict = dict[str, Any]


def _as_dict(value: object) -> JsonDict:
    """Return ``value`` as a ``dict[str, Any]`` or an empty dict."""
    if isinstance(value, dict):
        return cast(JsonDict, value)
    return {}


def _as_list(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return cast(list[Any], value)
    return [value]


# ── public API ───────────────────────────────────────────────────────────


def load_methodology(
    base_path: Path,
    sources: list[MixinSource],
    active_mixins: list[str],
) -> ResolvedMethodology:
    """Load base + active mixin yamls and return a ResolvedMethodology."""
    errors: list[str] = []

    base_raw = _read_yaml(base_path)
    layers, defaults = _parse_base(base_raw, errors)

    available = _discover_mixins(sources)
    resolved_paths = _resolve_active(active_mixins, available, errors)
    _raise_if_errors(errors)

    spec_types: dict[str, SpecType] = {}
    types_per_layer: dict[str, list[str]] = {layer.id: [] for layer in layers}
    validators: dict[str, list[JsonDict]] = {}
    hooks_by_point: dict[str, list[JsonDict]] = {}
    metrics_by_layer: dict[str, list[JsonDict]] = {}
    tools_by_layer: dict[str, list[JsonDict]] = {}
    generator_hints: dict[tuple[str, str], str] = {}
    field_enums: dict[str, list[str]] = {}

    extension_actions: list[tuple[str, JsonDict]] = []
    cross_cutting_actions: list[tuple[str, JsonDict]] = []
    pending_hints: list[tuple[str, JsonDict]] = []

    # Pass 1 — collect adds_types + global facts
    for mixin_name in active_mixins:
        path = resolved_paths.get(mixin_name)
        if path is None:
            continue
        mixin_raw = _read_yaml(path)

        for entry in _as_list(mixin_raw.get("adds_types")):
            _ingest_add_type(
                mixin_name,
                _as_dict(entry),
                layers,
                spec_types,
                types_per_layer,
                errors,
            )

        for entry in _as_list(mixin_raw.get("validators")):
            _ingest_validator(mixin_name, _as_dict(entry), validators)

        for point, items in _as_dict(mixin_raw.get("hooks")).items():
            for item in _as_list(items):
                item_dict = _as_dict(item)
                if item_dict:
                    hooks_by_point.setdefault(point, []).append(item_dict)

        for entry in _as_list(mixin_raw.get("metrics")):
            entry_dict = _as_dict(entry)
            layer_id = entry_dict.get("layer")
            if isinstance(layer_id, str):
                metrics_by_layer.setdefault(layer_id, []).append(entry_dict)

        for layer_id, tools in _as_dict(mixin_raw.get("tools_by_layer")).items():
            for tool in _as_list(tools):
                tool_dict = _as_dict(tool)
                if tool_dict:
                    tools_by_layer.setdefault(layer_id, []).append(tool_dict)

        hints = _as_dict(mixin_raw.get("generator_hints"))
        if hints:
            pending_hints.append((mixin_name, hints))

        for entry in _as_list(mixin_raw.get("extends_types")):
            extension_actions.append((mixin_name, _as_dict(entry)))

        cross = _as_dict(mixin_raw.get("extends_all_types"))
        if cross:
            cross_cutting_actions.append((mixin_name, cross))

        for entry in _as_list(mixin_raw.get("extends_field")):
            _ingest_field_extension(_as_dict(entry), field_enums)

    # Pass 2 — bind generator hints (types are known by now)
    for _mixin_name, hints in pending_hints:
        for type_id, text in hints.items():
            if not isinstance(text, str):
                continue
            st = spec_types.get(type_id)
            if st is not None:
                generator_hints[(st.layer_id, type_id)] = text

    # Pass 3 — extends_types
    for mixin_name, entry in extension_actions:
        _apply_extends_types(mixin_name, entry, spec_types, errors)

    # Pass 4 — extends_all_types
    for mixin_name, cross in cross_cutting_actions:
        _apply_extends_all_types(cross, spec_types)
        _ = mixin_name  # reserved for future error context

    _raise_if_errors(errors)

    return ResolvedMethodology(
        layers=layers,
        spec_types=spec_types,
        types_per_layer=types_per_layer,
        validators=validators,
        hooks_by_point=hooks_by_point,
        metrics_by_layer=metrics_by_layer,
        tools_by_layer=tools_by_layer,
        generator_hints_map=generator_hints,
        field_enums=field_enums,
        defaults=defaults,
    )


# ── internals ────────────────────────────────────────────────────────────


def _read_yaml(path: Path) -> JsonDict:
    text = path.read_text(encoding="utf-8")
    return _as_dict(yaml.safe_load(text))


def _parse_base(raw: JsonDict, errors: list[str]) -> tuple[list[Layer], JsonDict]:
    layers: list[Layer] = []
    for entry in _as_list(raw.get("layers")):
        entry_dict = _as_dict(entry)
        if not entry_dict:
            errors.append(f"layers entry is not a mapping: {entry!r}")
            continue
        try:
            layers.append(Layer.model_validate(entry_dict))
        except Exception as exc:  # pydantic validation error
            errors.append(f"invalid layer entry {entry_dict!r}: {exc}")
    return layers, _as_dict(raw.get("defaults"))


def _discover_mixins(sources: list[MixinSource]) -> dict[str, Path]:
    available: dict[str, Path] = {}
    for source in sources:
        for path in source.discover():
            try:
                raw = _read_yaml(path)
            except yaml.YAMLError:
                continue
            name = raw.get("mixin")
            if isinstance(name, str) and name not in available:
                available[name] = path
    return available


def _resolve_active(
    requested: list[str], available: dict[str, Path], errors: list[str]
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    known = list(available.keys())
    for name in requested:
        if name in available:
            resolved[name] = available[name]
            continue
        suggestions = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
        if suggestions:
            errors.append(
                f"active mixin {name!r} not found; did you mean: {', '.join(suggestions)}?"
            )
        else:
            errors.append(f"active mixin {name!r} not found")
    return resolved


def _resolve_target_layer(entry: JsonDict, layers: list[Layer]) -> tuple[Layer | None, str | None]:
    if "layer" in entry:
        target = entry["layer"]
        for layer in layers:
            if layer.id == target:
                return layer, None
        return None, f"layer {target!r} not found"
    if "layer_role" in entry:
        role = entry["layer_role"]
        matches = [layer for layer in layers if layer.role == role]
        if len(matches) == 1:
            return matches[0], None
        if not matches:
            return None, f"no layer with role {role!r}"
        return None, f"multiple layers with role {role!r}; specify layer_name"
    if "layer_name" in entry:
        name = entry["layer_name"]
        for layer in layers:
            if layer.name == name:
                return layer, None
        return None, f"layer_name {name!r} not found"
    return None, "missing layer/layer_role/layer_name"


def _ingest_add_type(
    mixin_name: str,
    entry: JsonDict,
    layers: list[Layer],
    spec_types: dict[str, SpecType],
    types_per_layer: dict[str, list[str]],
    errors: list[str],
) -> None:
    type_id = entry.get("id")
    if not isinstance(type_id, str):
        errors.append(f"[{mixin_name}] adds_types entry missing string 'id'")
        return
    if type_id in spec_types:
        errors.append(
            f"[{mixin_name}] duplicate spec type id {type_id!r} (already declared elsewhere)"
        )
        return
    layer, layer_err = _resolve_target_layer(entry, layers)
    if layer is None:
        errors.append(f"[{mixin_name}] type {type_id!r}: {layer_err}")
        return

    prefix = entry.get("prefix")
    if not isinstance(prefix, str):
        errors.append(f"[{mixin_name}] type {type_id!r} missing string 'prefix'")
        return

    sections = _parse_sections(mixin_name, type_id, entry.get("sections"), errors)
    required_fm = _parse_fm_fields(entry.get("required_frontmatter"))
    optional_fm = _parse_fm_fields(entry.get("optional_frontmatter"))
    up_to_types_raw = _as_list(entry.get("up_to_types"))
    up_to_types = [str(x) for x in up_to_types_raw]
    file_pattern_raw = entry.get("file_pattern")
    file_pattern = file_pattern_raw if isinstance(file_pattern_raw, str) else None
    cross_cutting = bool(entry.get("cross_cutting", False))

    spec_types[type_id] = SpecType(
        id=type_id,
        layer_id=layer.id,
        prefix=prefix,
        sections=sections,
        required_frontmatter=required_fm,
        optional_frontmatter=optional_fm,
        file_pattern=file_pattern,
        up_to_types=up_to_types,
        cross_cutting=cross_cutting,
    )
    types_per_layer.setdefault(layer.id, []).append(type_id)


def _parse_sections(
    mixin_name: str,
    type_id: str,
    raw: object,
    errors: list[str],
) -> list[Section]:
    sections: list[Section] = []
    seen_ids: set[str] = set()
    for entry in _as_list(raw):
        entry_dict = _as_dict(entry)
        if not entry_dict:
            continue
        sid = entry_dict.get("id")
        if not isinstance(sid, str):
            errors.append(f"[{mixin_name}] type {type_id!r}: section missing 'id'")
            continue
        if sid in seen_ids:
            errors.append(f"[{mixin_name}] type {type_id!r}: duplicate section id {sid!r}")
            continue
        seen_ids.add(sid)
        try:
            sections.append(Section.model_validate(entry_dict))
        except Exception as exc:
            errors.append(f"[{mixin_name}] type {type_id!r}: invalid section {sid!r}: {exc}")
    return sections


def _parse_fm_fields(raw: object) -> list[FrontmatterField]:
    fields: list[FrontmatterField] = []
    for entry in _as_list(raw):
        if isinstance(entry, str):
            fields.append(FrontmatterField(name=entry))
            continue
        entry_dict = _as_dict(entry)
        if not entry_dict:
            continue
        if "name" in entry_dict:
            name_val = entry_dict.get("name")
            if not isinstance(name_val, str):
                continue
            fields.append(
                FrontmatterField(
                    name=name_val,
                    description=str(entry_dict.get("description", "")),
                    llm_guidance=str(entry_dict.get("llm_guidance", "")),
                    validation=_as_dict(entry_dict.get("validation")) or None,
                    surface_when_missing=bool(entry_dict.get("surface_when_missing", False)),
                )
            )
        elif len(entry_dict) == 1:
            ((name_key, spec_val),) = entry_dict.items()
            spec_dict = _as_dict(spec_val)
            fields.append(
                FrontmatterField(
                    name=name_key,
                    description=str(spec_dict.get("description", "")),
                    llm_guidance=str(spec_dict.get("llm_guidance", "")),
                    validation=_as_dict(spec_dict.get("validation")) or None,
                    surface_when_missing=bool(spec_dict.get("surface_when_missing", False)),
                )
            )
    return fields


def _ingest_validator(
    mixin_name: str,
    entry: JsonDict,
    validators: dict[str, list[JsonDict]],
) -> None:
    if not entry:
        return
    annotated: JsonDict = dict(entry)
    annotated.setdefault("_mixin", mixin_name)
    target = annotated.get("on_type")
    key = target if isinstance(target, str) else "*"
    validators.setdefault(key, []).append(annotated)


def _ingest_field_extension(entry: JsonDict, field_enums: dict[str, list[str]]) -> None:
    name = entry.get("name")
    if not isinstance(name, str):
        return
    add_values = entry.get("add_values")
    if not isinstance(add_values, list):
        return
    bucket = field_enums.setdefault(name, [])
    for value in cast(list[Any], add_values):
        if isinstance(value, str) and value not in bucket:
            bucket.append(value)


def _apply_extends_types(
    mixin_name: str,
    entry: JsonDict,
    spec_types: dict[str, SpecType],
    errors: list[str],
) -> None:
    target_id = entry.get("id")
    if not isinstance(target_id, str):
        errors.append(f"[{mixin_name}] extends_types entry missing string 'id'")
        return
    target = spec_types.get(target_id)
    if target is None:
        errors.append(f"[{mixin_name}] extends_types references unknown type {target_id!r}")
        return

    existing_section_ids = {s.id for s in target.sections}
    for section_entry in _as_list(entry.get("add_sections")):
        section_dict = _as_dict(section_entry)
        if not section_dict:
            continue
        sid = section_dict.get("id")
        if not isinstance(sid, str):
            errors.append(f"[{mixin_name}] extends_types({target_id!r}): section missing 'id'")
            continue
        if sid in existing_section_ids:
            errors.append(
                f"[{mixin_name}] extends_types({target_id!r}): duplicate section id {sid!r}"
            )
            continue
        try:
            target.sections.append(Section.model_validate(section_dict))
            existing_section_ids.add(sid)
        except Exception as exc:
            errors.append(
                f"[{mixin_name}] extends_types({target_id!r}): invalid section {sid!r}: {exc}"
            )

    for fm in _parse_fm_fields(entry.get("add_required_frontmatter")):
        target.required_frontmatter.append(fm)
    for fm in _parse_fm_fields(entry.get("add_optional_frontmatter")):
        target.optional_frontmatter.append(fm)


def _apply_extends_all_types(
    cross: JsonDict,
    spec_types: dict[str, SpecType],
) -> None:
    required_extra = _parse_fm_fields(cross.get("required_frontmatter"))
    optional_extra = _parse_fm_fields(cross.get("optional_frontmatter"))
    for st in spec_types.values():
        for fm in required_extra:
            if not any(existing.name == fm.name for existing in st.required_frontmatter):
                st.required_frontmatter.append(fm)
        for fm in optional_extra:
            if not any(existing.name == fm.name for existing in st.optional_frontmatter):
                st.optional_frontmatter.append(fm)


def _raise_if_errors(errors: list[str]) -> None:
    if not errors:
        return
    bullets = "\n  - ".join(errors)
    raise MethodologyLoadError(f"methodology load failed:\n  - {bullets}")


def load_methodology_for_project(project_ctx: Any) -> ResolvedMethodology:
    """High-level helper: read the project's ``methodology.yaml`` and
    compose base + active mixins into a :class:`ResolvedMethodology`.

    Accepts anything with a ``methodology_yaml`` :class:`Path` attribute
    so both :class:`ProjectContext` and lightweight test fakes work.
    """
    from importlib import resources

    from spec_writer.methodology.sources import FilesystemMixinSource

    methodology_yaml = cast(Path, project_ctx.methodology_yaml)
    if not methodology_yaml.exists():
        raise MethodologyLoadError(f"missing {methodology_yaml} — run `spec-writer init` first")
    raw: Any = yaml.safe_load(methodology_yaml.read_text(encoding="utf-8")) or {}
    config = _as_dict(raw)
    active = [str(x) for x in _as_list(config.get("active_mixins"))]
    base_path = Path(str(resources.files("spec_writer.methodology._builtin").joinpath("base.yaml")))
    mixins_dir = Path(str(resources.files("spec_writer.methodology._builtin").joinpath("mixins")))
    return load_methodology(base_path, [FilesystemMixinSource(mixins_dir)], active)


__all__ = ["MethodologyLoadError", "load_methodology", "load_methodology_for_project"]
