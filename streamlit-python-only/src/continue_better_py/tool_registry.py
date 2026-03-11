from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import StructuredTool

from continue_better_py.settings import get_settings
from continue_better_py.tooling import ToolDefinition, build_tool_definitions


@dataclass(slots=True)
class ToolModule:
    id: str
    label: str
    category: str
    version: str
    available: bool
    enabled: bool
    active: bool
    status: str
    tool_names: list[str]


@dataclass(slots=True)
class RegisteredTool:
    definition: ToolDefinition
    enabled: bool

    @property
    def name(self) -> str:
        return self.definition.tool.name

    @property
    def tool(self) -> StructuredTool:
        return self.definition.tool

    @property
    def risk_level(self) -> str:
        return self.definition.risk_level

    @property
    def module_id(self) -> str:
        return self.definition.module_id


class ToolRegistry:
    def __init__(self, modules: list[ToolModule], tools: list[RegisteredTool]) -> None:
        self._modules = modules
        self._tools = tools
        self._tool_lookup = {tool.name: tool for tool in tools}

    def get(self, name: str) -> RegisteredTool | None:
        return self._tool_lookup.get(name)

    def enabled_tools(self) -> list[StructuredTool]:
        return [tool.tool for tool in self._tools if tool.enabled]

    def module_payloads(self) -> list[dict]:
        payloads = []
        for module in self._modules:
            enabled_tool_names = [tool.name for tool in self._tools if tool.enabled and tool.module_id == module.id]
            payloads.append(
                {
                    "id": module.id,
                    "label": module.label,
                    "category": module.category,
                    "version": module.version,
                    "available": module.available,
                    "enabled": module.enabled,
                    "active": module.active,
                    "status": module.status,
                    "toolNames": module.tool_names,
                    "enabledToolNames": enabled_tool_names,
                }
            )
        return payloads


def create_default_tool_registry(workspace_root: str, session_id: str | None = None) -> ToolRegistry:
    settings = get_settings()
    definitions = build_tool_definitions(workspace_root, session_id=session_id)
    module_flags = {
        "files": settings.enable_file_tools,
        "clarification": settings.enable_tool_clarification,
        "rag": settings.enable_rag,
    }
    tools = [
        RegisteredTool(definition=definition, enabled=module_flags.get(definition.module_id, True))
        for definition in definitions
    ]
    modules = [
        ToolModule(
            id="files",
            label="Files",
            category="core",
            version="0.1.0",
            available=settings.enable_file_tools,
            enabled=settings.enable_file_tools,
            active=False,
            status="idle",
            tool_names=[tool.name for tool in tools if tool.module_id == "files"],
        ),
        ToolModule(
            id="clarification",
            label="Clarification",
            category="control",
            version="0.1.0",
            available=settings.enable_tool_clarification,
            enabled=settings.enable_tool_clarification,
            active=False,
            status="idle",
            tool_names=[tool.name for tool in tools if tool.module_id == "clarification"],
        ),
        ToolModule(
            id="rag",
            label="RAG",
            category="knowledge",
            version="0.1.0",
            available=settings.enable_rag,
            enabled=settings.enable_rag,
            active=False,
            status="idle",
            tool_names=[tool.name for tool in tools if tool.module_id == "rag"],
        ),
    ]
    return ToolRegistry(modules=modules, tools=tools)
