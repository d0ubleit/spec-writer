"""ToolRegistry — vendor-agnostic dispatch table for agent tools.

Tools are async callables that take a ``ToolContext`` plus an argument
dict and return a serializable result dict. The registry guards
against accidental dispatch of unknown tool names (principle 6 —
fenced agent).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from spec_writer.engine.id_registry import IdRegistry
from spec_writer.engine.project_context import ProjectContext
from spec_writer.methodology.registry import ResolvedMethodology


@dataclass
class ToolContext:
    """Shared runtime state passed to every tool invocation."""

    project_ctx: ProjectContext
    methodology: ResolvedMethodology
    id_registry: IdRegistry


ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[dict[str, Any]]]


class ToolNotRegisteredError(KeyError):
    """Raised when an agent invokes a tool name not in the registry."""


class ToolRegistry:
    """Static dispatch table for agent tools."""

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        if name in self._handlers:
            raise ValueError(f"tool {name!r} already registered")
        self._handlers[name] = handler

    def names(self) -> list[str]:
        return sorted(self._handlers)

    def is_registered(self, name: str) -> bool:
        return name in self._handlers

    async def call(self, name: str, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            raise ToolNotRegisteredError(f"tool {name!r} not registered (known: {self.names()})")
        return await handler(args, ctx)


__all__ = ["ToolContext", "ToolHandler", "ToolNotRegisteredError", "ToolRegistry"]
