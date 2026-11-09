# Methodology model

Spec-writer ships with a methodology, but it does not *have* one. Everything about
what a specification looks like — types, layers, sections, frontmatter, rules, quality
metrics, generation guidance — is configuration. This document explains that
configuration format and how to extend it.

## Why configuration

Hardcoding a methodology (DDD, EARS, ISO/IEC/IEEE 29148, your team's homegrown
process) into prompts and validators means every methodology change is a code change.
Instead, the engine reads a **resolved methodology**: a registry assembled from YAML
at startup. Change the YAML, change the method.

The engine knows no spec-type names. The only strings it relies on are the ones it
receives from the registry.

## Layers

Layers declare position in the pyramid:

```yaml
version: 1
layers:
  - id: L1
    role: source-boundary
    name: vision
  - id: L2
    role: middle
    name: capabilities
  - id: L3
    role: middle
    name: behaviors
  - id: L4
    role: code-boundary
    name: technical
    is_final: true
```

- `source-boundary` is the top layer; it is the only layer allowed to read raw
  sources, and its specs must have `up_to: []`.
- `code-boundary` is the bottom layer; it must be code-generation ready.
- `middle` layers are arbitrary in number. The built-in presets use two, but the
  engine supports zero to N.

`role` (not `L<n>`) is what mixins bind to, which keeps mixins portable across
methodologies with a different number of layers.

## Mixin anatomy

A mixin is one YAML file. Top-level keys:

| Key | Meaning |
|---|---|
| `mixin` | Name (must match the file stem) |
| `description` | Human/LLM-readable summary |
| `adds_types` | Declare a new atomic spec type (creates separate files) |
| `extends_types` | Add sections/frontmatter to an existing type (section-mode) |
| `extends_all_types` | Apply frontmatter rules to every type (cross-cutting) |
| `extends_field` | Append allowed values to an enum field |
| `validators` | Rules attached to types or to every spec |
| `metrics` | Quality metrics computed at a layer |
| `hooks` | Lifecycle callbacks |
| `generator_hints` | Per-type guidance injected into the generation prompt |
| `tools_by_layer` | Extra tools advertised at a layer |

### Declaring a type

```yaml
mixin: capabilities
adds_types:
  - id: capability
    layer_name: capabilities        # or layer_role: middle
    prefix: CAP
    file_pattern: "{slug}-CAP-{number}.md"
    up_to_types: [vision]
    sections:
      - id: purpose
        title: "Purpose"
        required: true
        format: markdown
      - id: actors
        title: "Actors"
        required: false
        format: markdown
        guidance: "Who initiates or participates in this capability."
    required_frontmatter:
      - tags:
          description: "Topic tags"
          llm_guidance: "1–5 lowercase kebab-case identifiers."
```

Section formats: `markdown`, `yaml-block`, `json-block`. `body_format_guidance`
describes the expected shape in prose.

`required_frontmatter` entries are single-key maps whose value carries
`description`, `llm_guidance`, and optionally `validation`. Optional fields are
declared the same way under `optional_frontmatter`.

### Extending an existing type

Section-mode mixins keep middle-layer atoms from proliferating:

```yaml
mixin: acceptance-criteria
extends_types:
  - id: behavior
    adds_sections:
      - id: acceptance_criteria
        title: "Acceptance Criteria"
        required: true
        format: yaml-block
```

Cross-cutting rules use `extends_all_types`:

```yaml
mixin: tagging
extends_all_types:
  required_frontmatter:
    - tags:
        description: "Topic tags categorizing this spec"
```

### Enumerations

```yaml
extends_field:
  - name: priority
    add_values: [must, should, could, wont]
```

## Validators

Declarative validators are declared in the mixin and implemented once in the engine:

```yaml
validators:
  - id: pyramid-adjacency
    scope: per-spec
    func_ref: validators.check_pyramid_adjacency
  - id: dangling-ref
    scope: cross-spec
    func_ref: validators.check_dangling_refs
```

- `scope: per-spec` — runs against one spec.
- `scope: cross-spec` — runs against the corpus.
- `on_type: <type>` restricts the rule to one type; omit it for cross-cutting rules.
- `rule_kind: python` marks a function-backed rule; omit for fully declarative ones.

All validators run **collect-all**: one bad spec does not hide the rest. Issues carry
`severity`, `code`, optional `spec_id`/`field`, a message and an optional `fix_hint`
that the agent can act on directly.

## Metrics

```yaml
metrics:
  - id: L3-scenario-test-coverage
    layer: L3
    func_ref: metrics.scenario_test_coverage_ratio
    target: 0.9
    warn_below: 0.6
    critical: true        # below target blocks convergence
```

Metrics are the difference between "the validator is green" and "the corpus is
actually good". Batch convergence requires every critical metric at target.

## Presets

`spec-writer init --preset …` writes a methodology file that activates a set of
built-in mixins:

| Preset | Intended for |
|---|---|
| `minimal` | vision only — a three-line start |
| `default-backend` | backend services: capabilities, behaviours, API and entity contracts, tests |
| `web-product` | full product surface: UI, jobs, RBAC, ops, deployment, observability |

The preset only selects mixins; projects can add their own mixins alongside the
built-ins and reference them in `methodology.yaml`:

```yaml
active_mixins:
  - core-traceability
  - vision
  - capabilities
  - my-company-conventions
```

## Writing a new mixin — checklist

1. Pick the layer role/name and the prefix for a new type (atomic mode), or the type
   to extend (section mode).
2. Declare required sections — each one should answer a question a reader would ask.
3. Declare required frontmatter only for data that validators or metrics actually use.
4. Add `up_to_types` when the type legitimately anchors outside the adjacent layer.
5. Register validators for the failure modes you care about; write a `fix_hint`.
6. Add a metric if the type has a quality bar that a boolean validator cannot express.
7. Write `generator_hints` — this text goes into the generation prompt.
8. Add a test spec to `tests/` that exercises the mixin through the loader.

Keep examples in mixin text generic (`customer`, `payment`, `mail-provider`). A
built-in mixin ships to every project; project-specific vocabulary belongs in the
project's own mixins.
