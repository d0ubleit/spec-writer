"""AgentRuntime — drive LLMClient ↔ ToolRegistry for one layer-generation pass.

Phase 2 implementation supports:
- Multi-turn tool dispatch loop until the model stops calling tools.
- Single-call validation feedback retry: after the loop ends, validators
  run against accepted drafts; if errors come back, the runtime feeds
  a structured feedback message and re-enters the LLM loop. Hard cap
  via ``max_feedback_rounds``.
- ``record_spec`` tool-call outputs are reified back into ``Spec``
  objects (read from disk after the tool wrote them) so the engine
  sees a clean DraftProducer return value.

The :func:`AgentRuntime.as_producer` adapter returns a coroutine
matching :data:`spec_writer.engine.loop.DraftProducer`, so the
existing engine pipeline keeps its interface unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from spec_writer.agent.compression import CompressionPolicy, apply_policy
from spec_writer.agent.events import (
    EngineEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from spec_writer.agent.feedback import format_validation_feedback
from spec_writer.agent.llm_client import LLMClient, Message
from spec_writer.agent.tool_registry import ToolContext, ToolRegistry
from spec_writer.engine.seed import Seed
from spec_writer.engine.traceability import symmetrize_refs
from spec_writer.io.spec_io import find_spec_by_id, read_spec
from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.methodology.validators.runner import run_validators
from spec_writer.types import FrontmatterField, Layer, Section, Spec, SpecType

_LOG = logging.getLogger(__name__)


# ── prompt builders (minimal Phase 2; S2.6 replaces these) ───────────────


def _fmt_section(s: Section) -> str:
    line = f"  - **{s.title}** (`{s.id}`, {s.format})"
    if s.guidance:
        line += f"\n      {s.guidance.strip()}"
    return line


def _fmt_fm_field(field: FrontmatterField) -> str:
    line = f"    - `{field.name}`"
    if field.description:
        line += f": {field.description}"
    if field.llm_guidance:
        line += f" — {field.llm_guidance.strip()}"
    return line


def _type_block(spec_type: SpecType, hint: str) -> str:
    required_sections = [s for s in spec_type.sections if s.required]
    optional_sections = [s for s in spec_type.sections if not s.required]

    sec_block = ""
    if required_sections:
        sec_block += "\n  Required sections:\n" + "\n".join(
            _fmt_section(s) for s in required_sections
        )
    if optional_sections:
        sec_block += "\n  Optional sections:\n" + "\n".join(
            _fmt_section(s) for s in optional_sections
        )

    fm_block = ""
    if spec_type.required_frontmatter:
        fm_block = (
            "\n  Required frontmatter fields (in addition to id/type/layer/title/up_to):\n"
            + "\n".join(_fmt_fm_field(f) for f in spec_type.required_frontmatter)
        )

    prefix_line = f"  ID prefix: `{spec_type.prefix}`"
    up_to_line = (
        f"  Each spec's `up_to` must reference parent type(s): {spec_type.up_to_types}"
        if spec_type.up_to_types
        else "  Each spec's `up_to` may be empty (this is a source-boundary type)."
    )

    header = f"### Spec type `{spec_type.id}`"
    if hint:
        header += f"\n  {hint.strip()}"
    return "\n".join([header, prefix_line, up_to_line, sec_block, fm_block]).rstrip()


_DIRECTION_GUIDANCE: dict[str, str] = {
    "fwd": (
        "## Direction: FWD (source → specs)\n"
        "You are translating **source documents / parent specs** into specs at\n"
        "this layer. Anchor every new spec to something concrete in the source —\n"
        "do not invent requirements that aren't documented. `up_to` must point\n"
        "to a real parent in the corpus or source-boundary."
    ),
    "rev": (
        "## Direction: REV (code → specs)\n"
        "You are reverse-engineering the corpus from **implementation code**.\n"
        "Read the codebase first (use `Read`/`Grep`), then record specs that\n"
        "describe what the code actually does. Set `implemented_by` to the\n"
        "concrete code symbols; `up_to` typically references the spec below\n"
        "this one in the pyramid (already-recovered child specs)."
    ),
}


def _build_system_prompt(
    layer: Layer,
    methodology: ResolvedMethodology,
    direction: str = "fwd",
) -> str:
    types = methodology.types_at_layer(layer.id)

    type_blocks: list[str] = [
        _type_block(st, methodology.generator_hints(layer.id, st.id) or "") for st in types
    ]

    types_section = (
        "\n\n".join(type_blocks) if type_blocks else "(no spec types declared for this layer)"
    )

    direction_block = _DIRECTION_GUIDANCE.get(direction, _DIRECTION_GUIDANCE["fwd"])

    return (
        f"You are the spec-writer agent. Generate specs for **layer {layer.id} "
        f"({layer.name})**.\n"
        f"Role: {layer.role}. Purpose: {layer.purpose or '(unspecified)'}.\n\n"
        f"{direction_block}\n\n"
        "## Workflow\n"
        "1. If source documents are listed, use the `Read` tool to consume them.\n"
        "2. For each new spec, call `mcp__spec-writer__record_spec` with:\n"
        "   - `type` — one of the spec types declared below\n"
        "   - `layer` — this layer's id\n"
        "   - `title` — short English title\n"
        "   - `up_to` — list of parent spec ids (empty for source-boundary)\n"
        "   - `body` — markdown content covering the required sections (use\n"
        "     `## Section Title` headings)\n"
        "   - any required frontmatter fields listed per type\n"
        "3. If validation feedback comes back, use `mcp__spec-writer__patch_spec`\n"
        "   to apply targeted fixes — do NOT regenerate the whole spec.\n"
        "4. Output must be English. Keep one idea per file (cohesion over\n"
        "   fragmentation).\n\n"
        "## Discovery tools\n"
        "- `mcp__spec-writer__query_traceability` — walk up_to/down_to/relates_to\n"
        "  from a spec to see its neighbours (useful when fixing trace errors).\n"
        "- `mcp__spec-writer__resolve_source_chain` — walk up_to recursively to\n"
        "  the source-boundary; returns the ancestor chain so you can re-anchor.\n"
        "- `mcp__spec-writer__resolve_scope` — translate a free-form description\n"
        "  ('capabilities on L2 tagged #auth') into a structured filter.\n"
        "- `mcp__spec-writer__ask_user` — chat/dialog mode only; ask one\n"
        "  clarification at a time. In batch this errors out, so make a\n"
        "  reasonable default decision and proceed.\n\n"
        f"## Spec types at this layer\n\n{types_section}\n\n"
        "When you have produced all the specs the source/parent layer warrants,\n"
        "reply with a one-line summary."
    )


_MODE_INSTRUCTIONS = {
    "generate": (
        "**Mode: generate.** Produce the specs warranted by the source / parent "
        "documents below. Use `mcp__spec-writer__record_spec` for each new spec."
    ),
    "refine": (
        "**Mode: refine.** Existing specs are listed below — your job is to apply "
        "TARGETED PATCHES via `mcp__spec-writer__patch_spec` to fix validator "
        "issues. DO NOT call `record_spec` to create new specs in this run; only "
        "patch what exists."
    ),
    "incremental-add": (
        "**Mode: incremental-add.** Add NEW specs that integrate with the existing "
        "corpus — reuse parent ids, don't duplicate concepts. Use `record_spec` "
        "for additions and `patch_spec` to update parents' down_to if needed."
    ),
    "cascade-fix": (
        "**Mode: cascade-fix.** A manual edit propagated changes upstream/"
        "downstream. Inspect existing specs and apply patches to keep them "
        "consistent. Mostly `patch_spec`; record_spec only if a new dependency "
        "is genuinely required."
    ),
    "targeted-edit-cascade": (
        "**Mode: targeted-edit-cascade.** Apply the requested edit via "
        "`patch_spec`, then cascade implications to related specs."
    ),
    "sync": (
        "**Mode: sync.** External state (code/sources) changed; reconcile the "
        "corpus. Use `patch_spec` for updates and `record_spec` only when a new "
        "spec is genuinely needed to represent the new state."
    ),
}


def _build_user_message(layer: Layer, seed: Seed, corpus: dict[str, Spec]) -> str:
    parts: list[str] = [_MODE_INSTRUCTIONS.get(seed.mode, _MODE_INSTRUCTIONS["generate"])]
    if seed.paths:
        sample = [str(p) for p in seed.paths[:10]]
        parts.append("Input documents at this layer:\n- " + "\n- ".join(sample))
    if seed.existing_corpus_summary:
        existing = "\n".join(
            f"  {sid}: {title}" for sid, title in seed.existing_corpus_summary.items()
        )
        parts.append(f"Existing specs at this layer:\n{existing}")
    if seed.notes:
        parts.append("Notes:\n- " + "\n- ".join(seed.notes))
    if not seed.paths and not seed.existing_corpus_summary:
        parts.append(f"Generate the first spec for layer {layer.id}.")
    if corpus:
        parts.append(f"(corpus size across all layers: {len(corpus)})")
    return "\n\n".join(parts)


# ── runtime ──────────────────────────────────────────────────────────────


class AgentRuntime:
    """Phase 2 agent loop: LLM ↔ tools, with one feedback retry round.

    Two dispatch modes:

    - **Local** (default): runtime sees ``ToolCallEvent`` from the LLM
      stream and dispatches via the in-process ``ToolRegistry``. Used
      by FakeLLM-based unit tests where there is no SDK underneath.
    - **SDK-MCP** (when ``recorded_ids_buffer`` is provided): the LLM
      provider has registered custom tools as an MCP server and the
      SDK handles dispatch internally. The runtime observes events for
      visibility but does NOT call the local registry — that would
      execute the tool twice. After ``llm.chat()`` finishes, the
      runtime reads ``recorded_ids_buffer`` (populated by the MCP tool
      closure) to learn what was written.
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        ctx: ToolContext,
        *,
        max_tool_turns: int = 8,
        max_feedback_rounds: int = 2,
        recorded_ids_buffer: list[str] | None = None,
        compression_policy: CompressionPolicy | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._ctx = ctx
        self._max_tool_turns = max_tool_turns
        self._max_feedback_rounds = max_feedback_rounds
        self._recorded_ids_buffer = recorded_ids_buffer
        self._compression_policy = compression_policy or CompressionPolicy(
            strategies=["truncate_tool_results", "sliding_window"]
        )

    async def produce_drafts(
        self,
        layer: Layer,
        seed: Seed,
        corpus: dict[str, Spec],
        methodology: ResolvedMethodology,
    ) -> list[Spec]:
        messages: list[Message] = [
            Message(
                role="system",
                content=_build_system_prompt(layer, methodology, direction=seed.direction),
            ),
            Message(role="user", content=_build_user_message(layer, seed, corpus)),
        ]

        if self._recorded_ids_buffer is not None:
            # SDK-MCP mode — buffer is filled by tool closures inside the SDK.
            recorded_ids = self._recorded_ids_buffer
        else:
            recorded_ids = []

        for feedback_round in range(self._max_feedback_rounds + 1):
            await self._drive_until_quiet(messages, recorded_ids)

            drafts = self._load_drafts(recorded_ids)
            if not drafts:
                return []

            # `record_spec` symmetrises neighbours on disk, but the
            # in-memory corpus still holds pre-symmetrisation copies.
            # Symmetrise the provisional corpus before validating so the
            # feedback round does not report false `vertical_asymmetry`
            # errors (mirrors `engine.loop.evaluate_iteration`).
            candidates = list(corpus.values()) + drafts
            symmetrize_refs(candidates)

            issues = run_validators(candidates, methodology)
            errors = [i for i in issues if i.severity == "error"]
            if not errors:
                return drafts
            if feedback_round == self._max_feedback_rounds:
                _LOG.warning("agent runtime: feedback cap reached with %d errors", len(errors))
                return drafts

            feedback = format_validation_feedback(issues)
            messages.append(Message(role="user", content=feedback))
            messages[:] = await apply_policy(
                messages,
                self._compression_policy,
                corpus={spec.id: spec for spec in (list(corpus.values()) + drafts)},
            )

        return self._load_drafts(recorded_ids)

    async def _drive_until_quiet(
        self,
        messages: list[Message],
        recorded_ids: list[str],
    ) -> None:
        sdk_mode = self._recorded_ids_buffer is not None
        for _turn in range(self._max_tool_turns):
            tool_calls_this_turn: list[tuple[str, str, dict[str, Any]]] = []
            text_buffer: list[str] = []

            async for event in self._llm.chat(messages, tools=[]):
                if isinstance(event, TextDeltaEvent):
                    text_buffer.append(event.text)
                elif isinstance(event, ToolCallEvent):
                    if sdk_mode:
                        # SDK handles dispatch via its MCP server; runtime
                        # is observe-only. recorded_ids buffer is appended
                        # to by the MCP tool closure (see mcp_tools.py).
                        continue
                    result = await self._dispatch_tool(event)
                    if event.tool == "record_spec" and result.get("ok"):
                        spec_id = result.get("spec_id")
                        if isinstance(spec_id, str):
                            recorded_ids.append(spec_id)
                    tool_calls_this_turn.append((event.tool, str(event.args), result))
                elif isinstance(event, ToolResultEvent):
                    # Either local-dispatch echoes or SDK's own result blocks;
                    # both already-handled paths.
                    pass

            full_text = "".join(text_buffer)
            if full_text:
                messages.append(Message(role="assistant", content=full_text))

            if sdk_mode:
                # SDK consumed the whole conversation in a single chat() call.
                return

            if not tool_calls_this_turn:
                return

            for tool, args_repr, result in tool_calls_this_turn:
                messages.append(
                    Message(
                        role="tool",
                        content=f"{tool}({args_repr}) -> {result}",
                    )
                )
            messages[:] = await apply_policy(messages, self._compression_policy)

    async def _dispatch_tool(self, event: ToolCallEvent) -> dict[str, Any]:
        if not self._tools.is_registered(event.tool):
            _LOG.warning("agent runtime: unknown tool %r", event.tool)
            return {"ok": False, "error": f"tool {event.tool!r} not registered"}
        try:
            return await self._tools.call(event.tool, event.args, self._ctx)
        except Exception as exc:  # surface tool errors as structured results
            _LOG.exception("tool %s failed", event.tool)
            return {"ok": False, "error": str(exc)}

    def _load_drafts(self, recorded_ids: list[str]) -> list[Spec]:
        drafts: list[Spec] = []
        seen: set[str] = set()
        for spec_id in recorded_ids:
            if spec_id in seen:
                continue
            seen.add(spec_id)
            path = find_spec_by_id(self._ctx.project_ctx.specs, spec_id)
            if path is None:
                continue
            try:
                drafts.append(read_spec(path))
            except Exception:
                _LOG.exception("could not reload recorded spec %s", spec_id)
        return drafts

    def as_producer(
        self,
    ) -> Callable[[Layer, Seed, dict[str, Spec], ResolvedMethodology], Awaitable[list[Spec]]]:
        async def _adapter(
            layer: Layer,
            seed: Seed,
            corpus: dict[str, Spec],
            methodology: ResolvedMethodology,
        ) -> list[Spec]:
            return await self.produce_drafts(layer, seed, corpus, methodology)

        return _adapter


__all__ = ["AgentRuntime", "format_validation_feedback"]


# Re-export common engine events for convenience.
del EngineEvent
