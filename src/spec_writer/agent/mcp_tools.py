"""Build a claude-agent-sdk MCP server exposing spec-writer custom tools.

claude-agent-sdk's native runtime owns the tool round-trip for tools
declared on an MCP server: when Claude calls one, the SDK invokes the
registered Python function in-process and feeds the result back as a
ToolResult message. Our :mod:`agent.runtime` therefore must NOT also
dispatch the same tool locally — that would execute it twice.

This module wraps the existing ``record_spec`` / ``patch_spec`` handlers
so the SDK can call them. The returned ``recorded_ids`` list is appended
to by ``record_spec`` invocations; the producer reads it after the
chat stream finishes and reifies the IDs into :class:`Spec` drafts via
disk lookup.
"""

from __future__ import annotations

import json
from typing import Any

from claude_agent_sdk import (  # type: ignore[import-untyped]
    create_sdk_mcp_server,
    tool,
)

from spec_writer.agent.tool_registry import ToolContext
from spec_writer.agent.tools.ask_user import ask_user
from spec_writer.agent.tools.build_reverse_index import build_reverse_index
from spec_writer.agent.tools.coverage_report import coverage_report
from spec_writer.agent.tools.delete_spec import delete_spec
from spec_writer.agent.tools.detect_code_changes import detect_code_changes
from spec_writer.agent.tools.discover_code_inventory import discover_code_inventory
from spec_writer.agent.tools.list_corpus import list_corpus
from spec_writer.agent.tools.list_sources import list_sources
from spec_writer.agent.tools.metrics import metrics as metrics_tool
from spec_writer.agent.tools.patch_spec import patch_spec
from spec_writer.agent.tools.query_traceability import query_traceability
from spec_writer.agent.tools.read_spec import read_spec_tool
from spec_writer.agent.tools.record_spec import record_spec
from spec_writer.agent.tools.record_sync_complete import record_sync_complete
from spec_writer.agent.tools.resolve_scope import resolve_scope
from spec_writer.agent.tools.resolve_source_chain import resolve_source_chain
from spec_writer.agent.tools.validate import validate as validate_tool
from spec_writer.engine.id_registry import IdRegistry
from spec_writer.engine.project_context import ProjectContext
from spec_writer.methodology.registry import ResolvedMethodology

MCP_SERVER_NAME = "spec-writer"
"""Server slug — must match the prefix in MCP-qualified tool names."""


def mcp_allowed_tool_names() -> list[str]:
    """Tool names Claude is permitted to invoke on the MCP server."""
    return [
        f"mcp__{MCP_SERVER_NAME}__record_spec",
        f"mcp__{MCP_SERVER_NAME}__patch_spec",
        f"mcp__{MCP_SERVER_NAME}__query_traceability",
        f"mcp__{MCP_SERVER_NAME}__resolve_scope",
        f"mcp__{MCP_SERVER_NAME}__resolve_source_chain",
        f"mcp__{MCP_SERVER_NAME}__ask_user",
        f"mcp__{MCP_SERVER_NAME}__validate",
        f"mcp__{MCP_SERVER_NAME}__metrics",
        f"mcp__{MCP_SERVER_NAME}__list_sources",
        f"mcp__{MCP_SERVER_NAME}__list_corpus",
        f"mcp__{MCP_SERVER_NAME}__discover_code_inventory",
        f"mcp__{MCP_SERVER_NAME}__coverage_report",
        f"mcp__{MCP_SERVER_NAME}__detect_code_changes",
        f"mcp__{MCP_SERVER_NAME}__build_reverse_index",
        f"mcp__{MCP_SERVER_NAME}__record_sync_complete",
        f"mcp__{MCP_SERVER_NAME}__delete_spec",
        f"mcp__{MCP_SERVER_NAME}__read_spec",
    ]


def default_allowed_tools() -> list[str]:
    """Read-only SDK builtins plus our MCP tools.

    The agent needs to ``Read`` source files and ``Grep`` for context;
    we keep ``Edit`` / ``Write`` / ``Bash`` out so all on-disk spec
    writes go through our atomic ``record_spec`` / ``patch_spec``.
    """
    return ["Read", "Grep", "Glob", *mcp_allowed_tool_names()]


# Pydantic-typed sub-schemas. Declared explicitly so the MCP wire
# serializer preserves nested structure rather than coercing the value
# to a string (a design constraint).
_RELATES_TO_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "neighbor spec id"},
        "kind": {
            "type": "string",
            "description": "relation kind (relates_to | depends_on | used_by | ...)",
        },
        "reason": {"type": "string", "description": "free-text justification"},
    },
    "required": ["id", "kind"],
}

_PROVENANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string"},
        "ref": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["kind", "ref"],
}

_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "description": "methodology spec type id"},
        "layer": {"type": "string", "description": "layer id (e.g. L1)"},
        "title": {"type": "string", "description": "English title"},
        "up_to": {
            "type": "array",
            "items": {"type": "string"},
            "description": "parent spec IDs (empty for source-boundary)",
        },
        "down_to": {
            "type": "array",
            "items": {"type": "string"},
            "description": "child spec IDs (engine auto-fills via symmetrise on write)",
        },
        "relates_to": {
            "type": "array",
            "items": _RELATES_TO_ITEM_SCHEMA,
            "description": "horizontal references (relates_to/depends_on/used_by/...)",
        },
        "provenance": _PROVENANCE_SCHEMA,
        "body": {"type": "string", "description": "markdown body"},
        "id": {
            "type": "string",
            "description": "optional explicit id; auto-allocated if omitted",
        },
        "tags": {"type": "array", "items": {"type": "string"}},
        "implemented_by": {"type": "array", "items": {"type": "string"}},
        "status": {"type": "string"},
    },
    "required": ["type", "layer", "title"],
}

_BODY_REPLACE_OP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "old": {
            "type": "string",
            "description": ("exact substring to replace; must appear once in the body"),
        },
        "new": {"type": "string", "description": "replacement text"},
    },
    "required": ["old", "new"],
}

_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spec_id": {"type": "string"},
        "changes": {
            "type": "object",
            "description": (
                "field → new value mapping (id/type/layer ignored). "
                "Reserved ops: 'body' (full replace), 'body_replace' "
                "(ordered surgical substitutions: list of {old, new}). "
                "relates_to expects list of {id, kind, reason?}; "
                "provenance expects {kind, ref, confidence?}. "
                "Unknown keys raise ."
            ),
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "body_replace": {
                    "type": ["array", "null"],
                    "items": _BODY_REPLACE_OP_SCHEMA,
                    "description": (
                        "ordered surgical edits; each old must match "
                        "exactly once in current body (post prior ops). "
                        "Pass null to clear a stale body_replace "
                        "frontmatter extra from "
                    ),
                },
                "up_to": {"type": "array", "items": {"type": "string"}},
                "down_to": {"type": "array", "items": {"type": "string"}},
                "relates_to": {
                    "type": "array",
                    "items": _RELATES_TO_ITEM_SCHEMA,
                },
                "provenance": _PROVENANCE_SCHEMA,
                "tags": {"type": "array", "items": {"type": "string"}},
                "implemented_by": {"type": "array", "items": {"type": "string"}},
                "status": {"type": ["string", "null"]},
            },
        },
    },
    "required": ["spec_id", "changes"],
}

_QUERY_TRACEABILITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spec_id": {"type": "string"},
        "direction": {
            "type": "string",
            "enum": ["up", "down", "both"],
            "description": "up: walk up_to; down: walk down_to; both: include relates_to too",
        },
        "depth": {
            "type": "integer",
            "minimum": 1,
            "description": "BFS depth (default 1)",
        },
    },
    "required": ["spec_id"],
}

_RESOLVE_SCOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "free-form scope description (layers, types, tags, ids, from:FIN-…)",
        },
    },
    "required": ["text"],
}

_RESOLVE_SOURCE_CHAIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spec_id": {"type": "string"},
        "max_depth": {
            "type": "integer",
            "minimum": 1,
            "description": "ancestor levels to walk (default 8)",
        },
    },
    "required": ["spec_id"],
}

_ASK_USER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "clarification question for the user (chat mode only)",
        },
    },
    "required": ["question"],
}

_VALIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scope": {
            "type": "object",
            "description": "optional Scope filter (layers / types / tags / spec_ids / folder_pattern)",
        },
    },
}

_METRICS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "layer_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "optional list of layer ids to compute metrics for; default = all",
        },
    },
}

_LIST_SOURCES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

_LIST_CORPUS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "layer": {"type": "string"},
        "type": {"type": "string"},
        "tag": {"type": "string"},
    },
}

_DISCOVER_CODE_INVENTORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "includes": {"type": "array", "items": {"type": "string"}},
        "excludes": {"type": "array", "items": {"type": "string"}},
        "max_files": {"type": "integer", "minimum": 1},
        "visibility": {"type": "string", "enum": ["public", "all"]},
    },
}

_COVERAGE_REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "includes": {"type": "array", "items": {"type": "string"}},
        "excludes": {"type": "array", "items": {"type": "string"}},
        "max_files": {"type": "integer", "minimum": 1},
    },
}

_DETECT_CODE_CHANGES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "includes": {"type": "array", "items": {"type": "string"}},
        "excludes": {"type": "array", "items": {"type": "string"}},
    },
}

_BUILD_REVERSE_INDEX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}

_RECORD_SYNC_COMPLETE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "last_sync_commit": {"type": "string"},
        "code_scope_hash": {"type": "string"},
    },
}

_DELETE_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"spec_id": {"type": "string"}},
    "required": ["spec_id"],
}

_READ_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "spec_id": {
            "type": "string",
            "description": "canonical spec id (e.g. L2-CAP-007)",
        },
    },
    "required": ["spec_id"],
}


def build_mcp_server(
    project_ctx: ProjectContext,
    methodology: ResolvedMethodology,
    id_registry: IdRegistry,
) -> tuple[Any, list[str]]:
    """Return (server_config, recorded_spec_ids) for use with claude-agent-sdk.

    The list is shared with the MCP tool closures; appending to it
    happens whenever Claude successfully invokes ``record_spec``.
    """
    recorded_ids: list[str] = []
    ctx = ToolContext(project_ctx=project_ctx, methodology=methodology, id_registry=id_registry)

    @tool("record_spec", "Write a new spec to disk", _RECORD_SCHEMA)
    async def record_spec_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await record_spec(args, ctx)
        except Exception as exc:  # surface to Claude as tool error
            result = {"ok": False, "error": str(exc)}
        if result.get("ok") is True:
            spec_id = result.get("spec_id")
            if isinstance(spec_id, str):
                recorded_ids.append(spec_id)
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool("patch_spec", "Apply a partial update to an existing spec", _PATCH_SCHEMA)
    async def patch_spec_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await patch_spec(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "query_traceability",
        "Walk up_to / down_to / relates_to edges around a spec",
        _QUERY_TRACEABILITY_SCHEMA,
    )
    async def query_traceability_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await query_traceability(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "resolve_scope",
        "Translate free-form text into a structured Scope filter",
        _RESOLVE_SCOPE_SCHEMA,
    )
    async def resolve_scope_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await resolve_scope(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "resolve_source_chain",
        "Walk up_to to the source-boundary; returns ancestor chain and source roots",
        _RESOLVE_SOURCE_CHAIN_SCHEMA,
    )
    async def resolve_source_chain_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await resolve_source_chain(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "ask_user",
        "Ask the user a clarification question (chat session only)",
        _ASK_USER_SCHEMA,
    )
    async def ask_user_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await ask_user(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "validate",
        "Run all methodology validators on the corpus (or a Scope slice)",
        _VALIDATE_SCHEMA,
    )
    async def validate_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await validate_tool(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "metrics",
        "Compute methodology-declared metrics over the corpus",
        _METRICS_SCHEMA,
    )
    async def metrics_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await metrics_tool(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "list_sources",
        "List input documents available in spec-editor/sources/",
        _LIST_SOURCES_SCHEMA,
    )
    async def list_sources_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await list_sources(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "list_corpus",
        "List existing specs (frontmatter only, no bodies)",
        _LIST_CORPUS_SCHEMA,
    )
    async def list_corpus_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await list_corpus(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "discover_code_inventory",
        "REV-mode: walk project source tree + return per-file symbol inventory",
        _DISCOVER_CODE_INVENTORY_SCHEMA,
    )
    async def discover_code_inventory_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await discover_code_inventory(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "coverage_report",
        "REV-mode: visited / covered / uncovered file lists vs current inventory",
        _COVERAGE_REPORT_SCHEMA,
    )
    async def coverage_report_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await coverage_report(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "detect_code_changes",
        "Sync-mode: git-first (or hash-fallback) detection of added/modified/deleted code files",
        _DETECT_CODE_CHANGES_SCHEMA,
    )
    async def detect_code_changes_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await detect_code_changes(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "build_reverse_index",
        "Sync-mode: build code-file → [spec_id] map from corpus implemented_by",
        _BUILD_REVERSE_INDEX_SCHEMA,
    )
    async def build_reverse_index_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await build_reverse_index(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "record_sync_complete",
        "Sync-mode: finalise a sync run by updating sync_state.json baseline",
        _RECORD_SYNC_COMPLETE_SCHEMA,
    )
    async def record_sync_complete_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await record_sync_complete(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "delete_spec",
        "Delete a spec from disk and release its id (destructive — confirm first)",
        _DELETE_SPEC_SCHEMA,
    )
    async def delete_spec_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await delete_spec(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    @tool(
        "read_spec",
        "Read one spec's full frontmatter + body by id — use before patching",
        _READ_SPEC_SCHEMA,
    )
    async def read_spec_mcp(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await read_spec_tool(args, ctx)
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    server = create_sdk_mcp_server(
        name=MCP_SERVER_NAME,
        version="0.1.0",
        tools=[
            record_spec_mcp,
            patch_spec_mcp,
            query_traceability_mcp,
            resolve_scope_mcp,
            resolve_source_chain_mcp,
            ask_user_mcp,
            validate_mcp,
            metrics_mcp,
            list_sources_mcp,
            list_corpus_mcp,
            discover_code_inventory_mcp,
            coverage_report_mcp,
            detect_code_changes_mcp,
            build_reverse_index_mcp,
            record_sync_complete_mcp,
            delete_spec_mcp,
            read_spec_mcp,
        ],
    )
    return server, recorded_ids


__all__ = [
    "MCP_SERVER_NAME",
    "build_mcp_server",
    "default_allowed_tools",
    "mcp_allowed_tool_names",
]
