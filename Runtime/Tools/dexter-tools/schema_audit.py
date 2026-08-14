"""Report the live Dexter MCP catalog size without maintaining a duplicate schema list."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import server


def _dump(tool: Any) -> dict:
    if hasattr(tool, "model_dump"):
        return tool.model_dump(by_alias=True, exclude_none=True)
    if hasattr(tool, "dict"):
        return tool.dict(by_alias=True, exclude_none=True)
    raise TypeError(f"Unsupported tool object: {type(tool)!r}")


async def _tools() -> list[Any]:
    # Prefer the public async catalog; fall back to the pinned SDK manager for
    # environments where list_tools is not exposed on MCPServer.
    method = getattr(server.mcp, "list_tools", None)
    if method is not None:
        value = method()
        if hasattr(value, "__await__"):
            value = await value
        return list(value)
    return list(server.mcp._tool_manager.list_tools())


def main() -> int:
    tools = asyncio.run(_tools())
    rows = []
    description_chars = 0
    input_schema_chars = 0
    serialized_chars = 0

    for tool in tools:
        payload = _dump(tool)
        description = str(payload.get("description") or "")
        schema = payload.get("inputSchema") or payload.get("input_schema") or {}
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        row = {
            "name": payload.get("name", "<unnamed>"),
            "description_chars": len(description),
            "schema_chars": len(schema_text),
            "serialized_chars": len(serialized),
        }
        rows.append(row)
        description_chars += row["description_chars"]
        input_schema_chars += row["schema_chars"]
        serialized_chars += row["serialized_chars"]

    print(f"tools: {len(rows)}")
    print(f"description chars: {description_chars}")
    print(f"input-schema chars: {input_schema_chars}")
    print(f"serialized tool chars: {serialized_chars}")
    print("\nlargest tools:")
    for row in sorted(rows, key=lambda item: item["serialized_chars"], reverse=True)[:10]:
        print(
            f"  {row['name']:<28} total={row['serialized_chars']:5d}  "
            f"schema={row['schema_chars']:5d}  description={row['description_chars']:4d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
