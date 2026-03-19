from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from streamlit_python_only.tooling_shared import ToolMetadata


@dataclass(slots=True)
class McpToolAdapter:
    name: str
    description: str
    args_schema: type[Any]
    handler: Callable[..., Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_langchain_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.handler,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )

    def to_tool_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            tool=self.to_langchain_tool(),
            risk_level=str(self.metadata.get("risk_level") or "safe"),
            module_id=str(self.metadata.get("module_id") or "mcp"),
            supports_thales=bool(self.metadata.get("supports_thales", True)),
            supports_native_tools=bool(self.metadata.get("supports_native_tools", True)),
            supports_textual_replay=bool(self.metadata.get("supports_textual_replay", True)),
            returns_evidence_kind=str(self.metadata.get("returns_evidence_kind") or "mcp"),
        )


def build_mcp_tool(tool_name: str, description: str, args_schema: type[Any], handler: Callable[..., Any], **metadata: Any) -> McpToolAdapter:
    return McpToolAdapter(
        name=tool_name,
        description=description,
        args_schema=args_schema,
        handler=handler,
        metadata=metadata,
    )


def build_mcp_tool_definition(tool_name: str, description: str, args_schema: type[Any], handler: Callable[..., Any], **metadata: Any) -> ToolMetadata:
    return build_mcp_tool(tool_name, description, args_schema, handler, **metadata).to_tool_metadata()


def build_mcp_tool_catalog(adapters: list[McpToolAdapter]) -> list[ToolMetadata]:
    return [adapter.to_tool_metadata() for adapter in adapters]
