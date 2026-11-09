# Architecture

This document explains how spec-writer is put together and *why* the boundaries sit
where they do. For the methodology configuration model see
[`methodology.md`](methodology.md).

## Layers of abstraction

```text
interfaces        chat · batch CLI · stdio MCP bridge
     ↓
agent runtime     ChatAgent (session) · AgentRuntime (per-layer producer)
     ↓
tools             17 MCP tools, thin wrappers over engine primitives
     ↓
engine            ids · scope · seed · traceability · validators · metrics · loop
     ↓
IO                frontmatter · deterministic paths · atomic writes
     ↓
project           spec-editor/{sources, specs, _meta}
```

Imports only ever point downwards. The engine never imports the agent, the
methodology never imports the engine: everything methodology-specific arrives through
a resolved registry built from YAML.

## Invariants the engine guarantees

Spec-writer is only useful if its corpus can be trusted. Four invariants do most of
that work:

### 1. Adjacent-only derivation

`L(n+1)` may reference only `L(n)`. The `pyramid-adjacency` validator rejects
skip-level edges. Types that legitimately need another anchor declare it explicitly:

```yaml
up_to_types: [capability, behavior]   # e.g. entity-schema
```

When `up_to_types` is present it *replaces* the layer rule for that type; when it is
absent the engine falls back to strict adjacency.

### 2. Symmetric links

Corpus-internal references are symmetric. The engine maintains the inverse edge at
write time (`record_spec`, `patch_spec`) and during batch evaluation:

| forward | inverse |
|---|---|
| `up_to` | `down_to` |
| `relates_to` | `relates_to` |
| `depends_on` | `used_by` |

External references (source documents, code locations, URLs) are intentionally
asymmetric: there is no frontmatter on the other side to update.

### 3. Unique, stable identifiers

IDs are `L<n>-<PREFIX>-<NNN>`, allocated by a gap-filling registry rebuilt from disk at
the start of every run. Prefixes come from the methodology, never from code. File paths
are derived deterministically from ID + slug, so a spec's location is a function of its
content.

### 4. Canonical artefacts

Frontmatter is written in a deterministic field order with canonical formatting. This
makes diffs reviewable and snapshot tests meaningful.

## Execution paths

### Chat — agent-led

`ChatAgent` owns one conversation. It builds a single system prompt describing the
methodology, available tools and workflow rules, then loops:

```text
user message → history + compression → llm.chat()
    → text deltas / tool calls (dispatched by the provider)
    → tool results → … until quiet → persist turn → REPL
```

The engine does not orchestrate anything here. "Go layer by layer" is a *behavioural
constraint in the prompt*, not a pipeline. That is what allows the agent to stop,
ask a clarifying question, and resume with new information.

### Batch — explicit orchestration

For CI and scripts there is nobody to ask. Two interchangeable engines:

| | `engine/loop.py` | `interface/batch/graph.py` |
|---|---|---|
| Model | hand-rolled convergence loop | LangGraph `StateGraph` |
| Persistence | none | checkpoint after every node |
| HITL | no | `interrupt()` before each layer |
| Default | yes | `--engine graph` |

Both compute convergence through the same function:

```python
evaluate_iteration(layer, methodology, corpus, drafts, iteration, ...)
  → symmetrise → validate → metrics → LoopState
```

`LoopState` is one of `CONVERGED`, `CONTINUE`, `STOPPED_SHORT`, `FAILED`. Convergence
requires zero errors *and* zero unmet critical metrics. No progress between iterations
fails fast instead of burning tokens.

The LangGraph variant keeps only JSON-friendly state in checkpoints
(`layer_index`, `iteration`, counters, serialised drafts). Heavy collaborators —
corpus, methodology, producer — live in a `GraphContext` passed through LangGraph's
`context_schema`, which keeps checkpoints small.

### Agent bridge

`interface/agent_bridge` exposes a chat session over stdio MCP. Another agent can call
`chat_open`, `chat_send`, `chat_answer`, `chat_resume`, `chat_history`, `chat_close`.
A single-answerer invariant prevents two orchestrators from resuming the same session.
This is how spec-writer can be composed as a tool inside a larger agent system.

## Prompt strategy

The system prompt is stable per session, which matters for cost: the Claude provider
passes it through `--system-prompt` so the CLI can attach a cache breakpoint to it.
Conversation history is flattened per turn, fingerprinted, and resumed through the
SDK's session id — a mismatch (after compression rewrote history) forces a fresh
session instead of sending inconsistent context.

Compression strategies are pluggable: truncate oversized tool results, sliding window,
inject a corpus snapshot, or let Claude compact its own context.

## Reverse direction and sync

The same primitives serve code → specs:

- `discover_code_inventory` walks the codebase with per-language analyzers
  (Python, PHP, TypeScript; filesystem fallback).
- `coverage_report` compares what the agent read against what it claimed to implement
  (`implemented_by`), backed by a passive ledger.
- `detect_code_changes` uses git when available, falling back to a content hash.
- `build_reverse_index` answers "which specs claim this file?".
- `record_sync_complete` stores the new baseline in `_meta/sync_state.json`.

## Extending

| Goal | Where |
|---|---|
| New spec type / rule / metric | a YAML mixin (see `docs/methodology.md`) |
| New LLM provider | implement `LLMClient` in `agent/providers/` |
| New engine primitive | add under `engine/`, expose a tool in `agent/tools/` |
| New orchestration policy | `interface/batch/` (loop or graph) |
| New interface | `interface/` (CLI command, HTTP, bridge tool) |

Keep the dependency direction intact and the engine project-agnostic: no product name,
spec-type name or domain vocabulary belongs in `src/spec_writer/`.
