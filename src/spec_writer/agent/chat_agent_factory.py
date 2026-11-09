"""Construction helpers shared by both ChatAgent surfaces.

``spec-writer chat`` (interactive REPL) and the agent_bridge MCP
server need exactly the same boot sequence: load agent.yaml, build the
spec-writer MCP server, wire it into a ClaudeSubscriptionClient, and
instantiate a ChatAgent (optionally seeded with replayed history from a
prior session log). This module exposes that as a single function so
neither surface has to know about provider wiring details.
"""

from __future__ import annotations

from typing import Any

from spec_writer.agent.chat_agent import ChatAgent, build_chat_system_prompt
from spec_writer.agent.chat_session import ChatSession
from spec_writer.agent.llm_client import LLMClient, Message
from spec_writer.agent.mcp_tools import build_mcp_server, default_allowed_tools
from spec_writer.config import TokenLimits, load_agent_config
from spec_writer.engine.id_registry import IdRegistry
from spec_writer.engine.project_context import ProjectContext
from spec_writer.engine.token_ledger import TokenLedger
from spec_writer.methodology.registry import ResolvedMethodology


class UnsupportedProviderError(RuntimeError):
    """Raised when agent.yaml requests a provider this code path can't drive."""


def build_chat_agent(
    project_ctx: ProjectContext,
    methodology: ResolvedMethodology,
    session: ChatSession,
    *,
    direction: str = "fwd",
    resume_from: bool = False,
    llm_override: LLMClient | None = None,
) -> ChatAgent:
    """Construct a fully wired ChatAgent.

    ``llm_override`` lets tests inject a :class:`FakeLLM`; in production
    the LLM is built from ``agent.yaml`` (must be ``claude-subscription``
    until other providers are wired in).

    When ``resume_from`` is true, the agent's history is seeded with the
    user/agent message pairs replayed from ``session.events``. Tool
    calls are not replayed — the agent will re-discover state via its
    tools on the next user message; sources / corpus may have changed
    since the original run.
    """
    if llm_override is not None:
        llm: LLMClient = llm_override
    else:
        agent_cfg = load_agent_config(project_root=project_ctx.root)
        if agent_cfg.provider != "claude-subscription":
            raise UnsupportedProviderError(
                f"chat path supports only 'claude-subscription' provider; "
                f"agent.yaml has {agent_cfg.provider!r}"
            )
        id_registry = IdRegistry.from_disk(project_ctx.specs)
        mcp_server, _recorded_ids = build_mcp_server(project_ctx, methodology, id_registry)

        from spec_writer.agent.providers.claude_subscription import (
            ClaudeSubscriptionClient,
        )

        llm = ClaudeSubscriptionClient(
            model=agent_cfg.model,
            mcp_servers={"spec-writer": mcp_server},
            allowed_tools=default_allowed_tools(),
        )

    initial_history: list[Message] | None = None
    if resume_from:
        initial_history = [
            Message(
                role="system",
                content=build_chat_system_prompt(methodology, project_ctx, direction),
            ),
            *replay_history_tail(session),
        ]

    # a known limitation: per-session ledger so the chat path actually records LLM
    # usage to _meta/usage/<date>.jsonl. No caps in chat mode (caps are
    # a batch-runtime concept where one wants a hard stop).
    token_ledger = TokenLedger(
        limits=TokenLimits(),
        usage_dir=project_ctx.meta / "usage",
    )

    return ChatAgent(
        llm=llm,
        project_ctx=project_ctx,
        methodology=methodology,
        session=session,
        direction=direction,
        initial_history=initial_history,
        token_ledger=token_ledger,
    )


def replay_history_tail(session: ChatSession) -> list[Message]:
    """Replay user utterances + agent messages from a logged session.

    Tool calls are NOT replayed; the agent will re-discover state via
    its tools on the next user message. We deliberately keep the
    reconstruction minimal — sources / corpus may have changed since
    the original run, and we don't want to mislead the agent with
    stale tool results.
    """
    out: list[Message] = []
    for event in session.events:
        kind = event.get("kind")
        if kind == "user-utterance":
            text = str(event.get("text", "")).strip()
            if text:
                out.append(Message(role="user", content=text))
        elif kind == "agent-message":
            text = str(event.get("text", "")).strip()
            if text:
                out.append(Message(role="assistant", content=text))
    return out


_ = Any  # placeholder for future typed config hooks


__all__ = [
    "UnsupportedProviderError",
    "build_chat_agent",
    "replay_history_tail",
]
