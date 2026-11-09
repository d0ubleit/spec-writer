"""ChatAgent — session-level conversation loop (primary chat path).

This is the **single agent** that runs through an entire chat session
([the ownership principle](docs/architecture.md)). One
context, one conversation history; the agent decides what to do,
calls engine primitives as MCP tools, asks the user when ambiguous,
and reports back. There is **no per-layer producer** in this path;
`run_layered_pipeline` lives only in batch CLI.

The agent emits :class:`EngineEvent`-s as it works so the REPL can
stream output (text deltas, tool calls, tool results) to the user
immediately rather than waiting for the LLM to finish.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from spec_writer.agent.chat_session import ChatSession
from spec_writer.agent.compression import (
    CompressionPolicy,
    apply_policy,
    summarizer_for,
)
from spec_writer.agent.events import (
    EngineEvent,
    StatusEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from spec_writer.agent.llm_client import LLMClient, Message
from spec_writer.agent.mcp_tools import default_allowed_tools
from spec_writer.engine.project_context import ProjectContext
from spec_writer.engine.token_ledger import TokenLedger
from spec_writer.io.spec_io import read_spec
from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.types import Spec

_LOG = logging.getLogger(__name__)


# ── system prompt ───────────────────────────────────────────────────────


def _layer_summary(methodology: ResolvedMethodology) -> str:
    lines: list[str] = []
    for layer in methodology.layers:
        types = [st.id for st in methodology.types_at_layer(layer.id)]
        lines.append(
            f"- **{layer.id}** ({layer.role}, dir: `{layer.name}`) — types: "
            f"{', '.join(types) if types else '(none)'}"
        )
    return "\n".join(lines)


_FWD_GUIDANCE = (
    "## Direction: FWD (sources → code)\n"
    "Work top-down. Start at L1 (source-boundary) using `list_sources` +\n"
    "`Read` on the documents in `spec-editor/sources/`. Anchor every spec\n"
    "to something concrete in the sources. After L1, move to L2 using L1\n"
    "specs as parents (`up_to` references); continue down to the\n"
    "code-boundary layer. `implemented_by` stays empty in FWD — that's a\n"
    "REV-mode concern.\n"
)


_REV_GUIDANCE = (
    "## Direction: REV (code → sources)\n"
    "Work bottom-up. Start at the **code-boundary layer** (the last one in\n"
    "the methodology). Call `discover_code_inventory` to enumerate the\n"
    "project's symbols, then `Read` / `Grep` to inspect the code. Record\n"
    "each significant symbol or route as a spec at the code-boundary\n"
    "layer — set `implemented_by` to the real file path(s) and line\n"
    'range(s) (e.g. `"src/auth/login.php:42-58"`). After writing each\n'
    "batch, call `coverage_report` to see what is still uncovered; keep\n"
    "walking until `ratio_covered` reaches the methodology target\n"
    "(default 0.85) or you have a justified reason to stop. Below the\n"
    "`warn_below` threshold (default 0.60), you MUST call `ask_user`\n"
    "to surface options (continue_walking / mark_intentional_gaps /\n"
    "lower_threshold) before declaring the layer done.\n"
    "After the code-boundary layer is covered, synthesise upward —\n"
    "L3 behaviors from L4 contracts, L2 capabilities from L3 behaviors,\n"
    "L1 vision from L2 capabilities. Use `up_to` on each intermediate\n"
    "spec to link to its parent (the layer immediately above it).\n"
)


def build_chat_system_prompt(
    methodology: ResolvedMethodology,
    project_ctx: ProjectContext,
    direction: str = "fwd",
) -> str:
    """System prompt for the session-level ChatAgent.

    Tells the agent the methodology, the available tools, and the
    workflow expectations from the ownership principle (proactive ask_user,
    research → propose → validate → patch loop, one idea per file).
    ``direction`` toggles between FWD (sources→code) and REV
    (code→sources) workflow guidance.
    """
    del project_ctx  # currently only methodology shapes the prompt
    direction_block = _REV_GUIDANCE if direction == "rev" else _FWD_GUIDANCE
    return (
        "You are the spec-writer agent — a methodology-aware partner that\n"
        "produces and maintains a layered specification corpus.\n\n"
        "## Ownership\n"
        "You own the whole session. One conversation, one context. You decide\n"
        "what to do at each step: read sources, list the corpus, write specs,\n"
        "validate, patch errors, ask the user when ambiguous, report progress.\n\n"
        f"{direction_block}\n"
        "## Active methodology\n"
        f"Layers (top → bottom):\n{_layer_summary(methodology)}\n\n"
        "Layered traversal is your responsibility, not a hard-coded loop.\n"
        "Work top-down for forward direction (sources → code) and bottom-up\n"
        "for reverse (code → sources). On each layer: research what's needed,\n"
        "propose specs, write them, validate, patch any errors, then move on.\n\n"
        "## Tools (MCP)\n"
        "- `mcp__spec-writer__list_sources` — what's in spec-editor/sources/\n"
        "- `mcp__spec-writer__list_corpus(layer?, type?, tag?)` — frontmatter\n"
        "  of existing specs (no bodies)\n"
        "- `Read(path)` — built-in; for source bodies, spec bodies, code\n"
        "- `Grep(pattern, path)` — built-in; for searching\n"
        "- `mcp__spec-writer__record_spec` — write a new spec (atomic)\n"
        "- `mcp__spec-writer__read_spec(spec_id)` — return one spec's full\n"
        "  frontmatter + body. USE BEFORE PATCHING — patching without\n"
        "  reading risks overwriting unrelated content. Preferred over the\n"
        "  built-in Read when you only have the spec_id (it resolves the\n"
        "  on-disk path automatically).\n"
        "- `mcp__spec-writer__patch_spec` — targeted update of an existing spec;\n"
        "  prefer this over rewriting bodies whole. Always `read_spec` first\n"
        "  to see current body before composing the patch.\n"
        "- `mcp__spec-writer__validate(scope?)` — run methodology validators;\n"
        "  returns list of issues. Use after writes, before declaring done\n"
        "- `mcp__spec-writer__metrics(layer_ids?)` — methodology metrics\n"
        "- `mcp__spec-writer__query_traceability(spec_id, direction, depth?)`\n"
        "- `mcp__spec-writer__resolve_scope(text)` — free-form text → Scope\n"
        "- `mcp__spec-writer__resolve_source_chain(spec_id)` — walk up_to to\n"
        "  source-boundary\n"
        "- `mcp__spec-writer__ask_user(question)` — only available in chat;\n"
        "  one focused question at a time, wait for the answer. DO NOT spam\n"
        "- `mcp__spec-writer__discover_code_inventory` — REV-mode only;\n"
        "  walks the project tree and lists per-file symbols (best-effort\n"
        "  hint). Use it on the first L4 pass when direction=rev; combine\n"
        "  with `Read` to inspect actual code.\n"
        "- `mcp__spec-writer__coverage_report` — REV-mode only; returns\n"
        "  visited / covered / uncovered file lists and ratios vs the\n"
        "  current inventory. Use it after generation passes to see where\n"
        "  you stand. If a code-boundary layer's `ratio_covered` is below\n"
        "  the methodology threshold (typically 0.6 `warn_below`), you\n"
        "  MUST call `ask_user` to surface options\n"
        "  (continue_walking / mark_intentional_gaps / lower_threshold)\n"
        "  before declaring the layer done.\n"
        "- `mcp__spec-writer__detect_code_changes` — sync-mode: git-first\n"
        "  (or hash-fallback) detection of added / modified / deleted /\n"
        "  renamed code files since the last sync.\n"
        "- `mcp__spec-writer__build_reverse_index` — sync-mode: produces\n"
        "  {code_file: [spec_ids]} mapping so you know which specs need\n"
        "  refining for a changed file.\n"
        "- `mcp__spec-writer__record_sync_complete` — sync-mode:\n"
        "  finalises a sync run by updating sync_state.json with the new\n"
        "  baseline commit / scope hash.\n"
        "- `mcp__spec-writer__delete_spec` — destructive: remove a spec\n"
        "  from disk and release its id. Use ONLY after `ask_user`\n"
        "  confirmation, or when handling an orphan the user has\n"
        "  explicitly approved for deletion.\n\n"
        "## Orphan workflow\n"
        "When code referenced by a spec is deleted, prefer marking the\n"
        "spec as orphan rather than deleting it: `patch_spec(spec_id,\n"
        "{status: 'orphan'})` keeps the history. Only call `delete_spec`\n"
        "after the user explicitly approves removal — confirm one orphan\n"
        'at a time via `ask_user` ("keep / delete"); never bulk-delete\n'
        "without per-spec confirmation. After deleting, also `patch_spec`\n"
        "any siblings whose `relates_to` / `up_to` / `down_to` still\n"
        "reference the deleted id.\n\n"
        "## Sync workflow\n"
        "When the user asks you to /sync (or you detect drift):\n"
        "1. `detect_code_changes` → classify each entry as new-feature /\n"
        "   existing-realization / utility / test.\n"
        "2. `build_reverse_index` → know which specs each changed file\n"
        "   touches.\n"
        "3. For each change, decide: `record_spec` (new feature),\n"
        "   `patch_spec` (existing realization needs updating), or skip\n"
        "   (utility / test). Always validate after batches.\n"
        "4. When done and the corpus is clean, call\n"
        "   `record_sync_complete` so the next /sync starts from the\n"
        "   right baseline.\n\n"
        "## Workflow guidance\n"
        "1. **Research first.** Before writing any spec, call `list_sources`\n"
        "   and `list_corpus` so you know what's actually there. Read what's\n"
        "   relevant.\n"
        "2. **Ask when uncertain.** Use `ask_user` for genuine ambiguity,\n"
        "   missing context, contradictions in sources, or design choices\n"
        "   the user should own. Do not invent. Do not guess and apologise.\n"
        "3. **Write incrementally.** One idea = one file. Anchor each spec\n"
        "   to a real parent (source or upper-layer spec).\n"
        "4. **Validate and patch.** After writing, call `validate`. If errors,\n"
        "   `patch_spec` each problem and re-validate. Don't regenerate whole\n"
        "   files for trivial fixes.\n"
        "5. **Report.** Tell the user what you did, what's next, where you\n"
        "   got stuck. Be concise.\n\n"
        "## Output\n"
        "All spec content (titles, bodies, frontmatter) MUST be English.\n"
        "Quotes from non-English sources are allowed with an English gloss.\n\n"
        "## What NOT to do\n"
        "- Don't call any tool not listed above. The SDK enforces this.\n"
        "- Don't write files outside `spec-editor/specs/` (use `record_spec`).\n"
        "- Don't ask the same question twice; remember answers within this\n"
        "  session.\n"
        "- Don't move on to a new layer with validator errors still pending.\n"
    )


# ── ChatAgent ───────────────────────────────────────────────────────────


class ChatAgent:
    """Single conversational agent that owns the whole session.

    Holds a persistent message history across multiple ``handle_user``
    calls; applies compression when the history grows past the policy
    trigger; streams events back to the caller for live UI rendering;
    logs every event into the underlying :class:`ChatSession`.
    """

    def __init__(
        self,
        llm: LLMClient,
        project_ctx: ProjectContext,
        methodology: ResolvedMethodology,
        session: ChatSession,
        *,
        direction: str = "fwd",
        compression_policy: CompressionPolicy | None = None,
        initial_history: list[Message] | None = None,
        token_ledger: TokenLedger | None = None,
    ) -> None:
        self._llm = llm
        self._project_ctx = project_ctx
        self._methodology = methodology
        self._session = session
        self._direction = direction
        self._compression_policy = compression_policy or CompressionPolicy(
            strategies=["truncate_tool_results", "sliding_window"]
        )
        # Optional ledger — when present, UsageEvents from the provider
        # are recorded to _meta/usage/<date>.jsonl . None ledger
        # means no usage capture (kept that way to preserve backward
        # compat in tests that don't supply one).
        self._token_ledger = token_ledger
        self._turn_count = 0
        if initial_history is not None:
            self.history: list[Message] = list(initial_history)
        else:
            self.history = [
                Message(
                    role="system",
                    content=build_chat_system_prompt(methodology, project_ctx, direction),
                )
            ]

    @property
    def session(self) -> ChatSession:
        return self._session

    @property
    def allowed_tools(self) -> list[str]:
        return default_allowed_tools()

    async def handle_user(self, text: str) -> AsyncIterator[EngineEvent]:
        """Drive the agent for one user message, yielding events as they arrive.

        SDK-MCP path: the underlying provider dispatches tool calls
        internally via the registered MCP server (see
        :mod:`agent.mcp_tools`). We observe the resulting events and
        thread them into both the stream and the session log. Multi-
        round tool use happens inside a single ``llm.chat`` invocation.
        """
        if not text.strip():
            return

        self._session.log_user(text)
        self.history.append(Message(role="user", content=text))
        self._turn_count += 1

        self.history = await apply_policy(
            self.history,
            self._compression_policy,
            corpus=self._snapshot_corpus(),
            summarizer=summarizer_for(self._llm),
        )

        text_buffer: list[str] = []
        try:
            async for event in self._llm.chat(self.history, tools=[]):
                if isinstance(event, TextDeltaEvent):
                    text_buffer.append(event.text)
                elif isinstance(event, ToolCallEvent):
                    self._session.log_tool_call(event.tool, event.args)
                elif isinstance(event, ToolResultEvent):
                    result_payload = _normalise_tool_result(event.result)
                    self._session.log_tool_result(event.tool, result_payload)
                elif isinstance(event, UsageEvent) and self._token_ledger is not None:
                    # a known limitation: append a JSONL line; "chat" layer-id since
                    # this is the interactive bridge path (vs batch).
                    self._token_ledger.record(
                        "chat",
                        event.usage,
                        call_label=f"{self._session.session_id}:turn-{self._turn_count}",
                    )
                yield event
        except Exception as exc:
            _LOG.exception("chat_agent: llm.chat raised")
            self._session.log_note("llm_error", error=str(exc))
            yield StatusEvent(phase="error", details={"reason": str(exc)})
            return

        full_text = "".join(text_buffer).strip()
        if full_text:
            self.history.append(Message(role="assistant", content=full_text))
            self._session.log_agent(full_text)

    def _snapshot_corpus(self) -> dict[str, Spec]:
        """Best-effort read of the current corpus for inject_spec_state."""
        specs_root: Path = self._project_ctx.specs
        out: dict[str, Spec] = {}
        if not specs_root.exists():
            return out
        for path in sorted(specs_root.rglob("*.md")):
            try:
                spec = read_spec(path)
            except Exception:
                continue
            out[spec.id] = spec
        return out


def _normalise_tool_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        from typing import cast

        return cast(dict[str, Any], result)
    return {"raw": str(result)}


__all__ = ["ChatAgent", "build_chat_system_prompt"]
