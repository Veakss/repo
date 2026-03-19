from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from streamlit_python_only.tooling_shared import ToolMetadata


class ClarificationInput(BaseModel):
    question: str = Field(description="Clarification question for the user.")
    option_a: str | None = Field(default=None, description="First suggested answer.")
    option_b: str | None = Field(default=None, description="Second suggested answer.")
    option_c: str | None = Field(default=None, description="Third suggested answer.")


def build_control_tool_definitions() -> list[ToolMetadata]:
    def request_clarification(question: str, option_a: str | None = None, option_b: str | None = None, option_c: str | None = None) -> str:
        options = [value for value in [option_a, option_b, option_c] if value]
        if options:
            return question + "\n" + "\n".join(f"- {item}" for item in options)
        return question

    return [
        ToolMetadata(
            tool=StructuredTool.from_function(
                request_clarification,
                name="request_clarification",
                description="Ask the user for clarification when the request is ambiguous or risky. Provide likely options when possible.",
                args_schema=ClarificationInput,
            ),
            risk_level="safe",
            module_id="clarification",
            supports_thales=True,
            supports_native_tools=True,
            supports_textual_replay=True,
            returns_evidence_kind="clarification",
        )
    ]
