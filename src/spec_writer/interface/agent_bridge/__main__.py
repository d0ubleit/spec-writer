"""Entry point: ``python -m spec_writer.interface.agent_bridge``.

Starts the stdio MCP server; intended to be launched by an MCP client
(e.g. Claude Code) via ``.mcp.json`` configuration.
"""

from __future__ import annotations

import asyncio

from spec_writer.interface.agent_bridge.server import run_stdio


def main() -> None:
    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
