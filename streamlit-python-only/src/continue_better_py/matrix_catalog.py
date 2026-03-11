from __future__ import annotations

from pathlib import Path
from typing import Any

from continue_better_py.settings import Settings, get_settings


PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "baseline_current": {
        "description": "Use the current environment as-is.",
        "env": {},
    },
    "router_on": {
        "description": "Force ENABLE_TOOL_GROUP_ROUTER=1.",
        "env": {"ENABLE_TOOL_GROUP_ROUTER": "1"},
    },
    "router_off": {
        "description": "Force ENABLE_TOOL_GROUP_ROUTER=0.",
        "env": {"ENABLE_TOOL_GROUP_ROUTER": "0"},
    },
    "playbooks_on": {
        "description": "Force ENABLE_TOOL_PLAYBOOKS=1.",
        "env": {"ENABLE_TOOL_PLAYBOOKS": "1"},
    },
    "playbooks_off": {
        "description": "Force ENABLE_TOOL_PLAYBOOKS=0.",
        "env": {"ENABLE_TOOL_PLAYBOOKS": "0"},
    },
    "verify_relaxed_rag_on": {
        "description": "Force ENABLE_VERIFY_RELAXED_RAG_CITATIONS=1.",
        "env": {"ENABLE_VERIFY_RELAXED_RAG_CITATIONS": "1"},
    },
    "verify_relaxed_rag_off": {
        "description": "Force ENABLE_VERIFY_RELAXED_RAG_CITATIONS=0.",
        "env": {"ENABLE_VERIFY_RELAXED_RAG_CITATIONS": "0"},
    },
}

SURFACE_PRESETS: dict[str, dict[str, Any]] = {
    "direct_runtime": {
        "description": "Direct sidecar/runtime invocation over the sidecar SSE surface.",
        "entrypoint": "sidecar",
        "env": {},
    },
    "backend_relay": {
        "description": "FastAPI backend relay over HTTP SSE.",
        "entrypoint": "backend",
        "env": {},
    },
    "mcp_bridge": {
        "description": "Compatibility surface that preserves the MCP bridge preset naming.",
        "entrypoint": "backend",
        "env": {"ENABLE_MCP_SERVER": "1", "USE_MCP_BRIDGE": "1"},
    },
}


def matrix_workspace_root(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.project_root.resolve()


def matrix_fixture_relative_path() -> str:
    return "matrix_fixtures/roadmap_status.md"


def matrix_fixture_absolute_path(settings: Settings | None = None) -> Path:
    return matrix_workspace_root(settings).joinpath(matrix_fixture_relative_path()).resolve()


def matrix_sandbox_relative_root() -> str:
    return "artifacts/matrix_sandbox"


def matrix_sandbox_absolute_root(settings: Settings | None = None) -> Path:
    return matrix_workspace_root(settings).joinpath(matrix_sandbox_relative_root()).resolve()


def ensure_matrix_fixtures(settings: Settings | None = None) -> dict[str, str]:
    settings = settings or get_settings()
    fixture_path = matrix_fixture_absolute_path(settings)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    if not fixture_path.exists():
        fixture_path.write_text(
            "\n".join(
                [
                    "# Matrix Fixture",
                    "Current checkpoint: AURORA_PHASE4",
                    "Phase 4 status: very advanced and close to closure.",
                    "Preferred editor: VS Code.",
                    "Preferred stack: Next.js with FastAPI.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    sandbox_root = matrix_sandbox_absolute_root(settings)
    sandbox_root.mkdir(parents=True, exist_ok=True)
    return {
        "workspace_root": str(matrix_workspace_root(settings)),
        "fixture_relative_path": matrix_fixture_relative_path(),
        "fixture_absolute_path": str(fixture_path),
        "sandbox_relative_root": matrix_sandbox_relative_root(),
        "sandbox_absolute_root": str(sandbox_root),
    }


def build_matrix_scenarios(settings: Settings | None = None) -> list[dict[str, Any]]:
    refs = ensure_matrix_fixtures(settings)
    fixture_rel = refs["fixture_relative_path"]
    fixture_abs = refs["fixture_absolute_path"]
    sandbox_rel = refs["sandbox_relative_root"]
    hello_rel = f"{sandbox_rel}/hello.txt"
    followup_rel = f"{sandbox_rel}/followup.txt"

    return [
        {
            "id": "social_no_clarify",
            "category": "clarification",
            "description": "Short social turns should not trigger clarification or tool use.",
            "prompt": "salut",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": False},
            "expectedState": ["completed"],
            "clarification": "must_not_happen",
            "expectedToolsMode": "none",
            "final": {"minNonWhitespaceChars": 2},
        },
        {
            "id": "clarification_food_app",
            "category": "clarification",
            "description": "Broad product prompts should use request_clarification.",
            "prompt": "Help me create a food application.",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": False},
            "expectedState": ["awaiting_clarification"],
            "clarification": "must_happen",
            "expectedToolsMode": "must_include",
            "expectedTools": ["request_clarification"],
        },
        {
            "id": "ambiguous_bugfix",
            "category": "clarification",
            "description": "Ambiguous bugfix prompts should clarify before acting.",
            "prompt": "Fix the issue",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": False},
            "expectedState": ["awaiting_clarification"],
            "clarification": "must_happen",
            "expectedToolsMode": "must_include",
            "expectedTools": ["request_clarification"],
        },
        {
            "id": "open_url_explicit",
            "category": "apps",
            "description": "Explicit URL opening should prefer open_url and complete cleanly.",
            "prompt": "Open https://example.com and then reply done.",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": True, "webSearch": False, "rag": False},
            "expectedState": ["completed"],
            "clarification": "must_not_happen",
            "expectedToolsMode": "must_include",
            "expectedTools": ["open_url"],
            "forbiddenTools": ["run_terminal"],
            "final": {"includesOneOf": ["/done/i", "/opened/i"]},
        },
        {
            "id": "open_file_explicit",
            "category": "apps",
            "description": "Explicit file opening should prefer open_file and complete cleanly.",
            "prompt": f"Open the file {fixture_rel} and then reply opened.",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": True, "webSearch": False, "rag": False},
            "expectedState": ["completed"],
            "clarification": "must_not_happen",
            "expectedToolsMode": "must_include",
            "expectedTools": ["open_file", "open_resource"],
            "forbiddenTools": ["run_terminal", "read_file"],
            "final": {"includesOneOf": ["/done/i", "/opened/i"]},
        },
        {
            "id": "open_file_explicit_absolute",
            "category": "apps",
            "description": "Explicit absolute file opening should still prefer open_file/open_resource.",
            "prompt": f"Open the file {fixture_abs} and then reply done.",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": True, "webSearch": False, "rag": False},
            "expectedState": ["completed"],
            "clarification": "must_not_happen",
            "expectedToolsMode": "must_include",
            "expectedTools": ["open_file", "open_resource"],
            "forbiddenTools": ["run_terminal", "read_file"],
            "final": {"includesOneOf": ["/done/i", "/opened/i"]},
        },
        {
            "id": "web_latest_with_sources",
            "category": "web",
            "description": "Current external info requests should use web_search and return at least two URLs.",
            "prompt": "Give me a very short summary of the latest OpenAI news with two source links.",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": True, "rag": False},
            "expectedState": ["completed"],
            "clarification": "must_not_happen",
            "expectedToolsMode": "must_include",
            "expectedTools": ["web_search"],
            "final": {"urlCountMin": 2, "minNonWhitespaceChars": 40},
        },
        {
            "id": "read_file_phase_status",
            "category": "files",
            "description": "Explicit local file reading should prefer read_file over terminal improvisation.",
            "prompt": f"Read {fixture_rel} and answer very briefly: what checkpoint is recorded in the file?",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": False},
            "expectedState": ["completed"],
            "clarification": "must_not_happen",
            "expectedToolsMode": "must_include",
            "expectedTools": ["read_file"],
            "forbiddenTools": ["run_terminal", "open_file", "open_resource", "web_search"],
            "final": {"includesOneOf": ["/aurora_phase4/i"], "minNonWhitespaceChars": 8},
        },
        {
            "id": "terminal_sequential_inspect",
            "category": "terminal",
            "description": "The model should use sequential terminal inspection without chaining.",
            "prompt": "Use the terminal to print the current directory and list the workspace root, then summarize in one line.",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": False},
            "expectedState": ["completed"],
            "clarification": "must_not_happen",
            "expectedToolsMode": "must_include",
            "expectedTools": ["run_terminal"],
            "final": {"minNonWhitespaceChars": 25, "includesAll": ["/continue better/i"]},
        },
        {
            "id": "terminal_policy_chaining",
            "category": "terminal",
            "description": "Unsafe shell chaining should fail with a useful policy diagnostic.",
            "prompt": "Execute the exact command ls && pwd and explain the result.",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": False},
            "expectedState": ["failed", "completed"],
            "clarification": "must_not_happen",
            "expectedToolsMode": "must_include",
            "expectedTools": ["run_terminal"],
            "policyExpectation": "blocked_or_safe_rewrite",
        },
        {
            "id": "files_write_then_readback",
            "category": "files",
            "description": "The model should create a file, then read it back exactly.",
            "prompt": f"Create the file {hello_rel} with the text bonjour-matrice, then read it back and reply with the exact content only.",
            "allowWrites": True,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": False},
            "expectedState": ["completed"],
            "clarification": "must_not_happen",
            "expectedToolsMode": "must_include_all",
            "expectedTools": ["write_file", "read_file"],
            "artifacts": {
                "fileExists": [hello_rel],
                "fileContains": [{"path": hello_rel, "includes": "bonjour-matrice"}],
            },
            "final": {"exactNormalizedOneOf": ["bonjour-matrice"]},
        },
        {
            "id": "memory_store_explicit",
            "category": "memory",
            "description": "Explicit remember requests should use session_memory_upsert.",
            "prompt": "Remember for this session that the preferred editor is VS Code, then confirm in one line.",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": True},
            "expectedState": ["completed"],
            "clarification": "must_not_happen",
            "expectedToolsMode": "must_include",
            "expectedTools": ["session_memory_upsert"],
            "final": {"includesOneOf": ["/vs code/i", "/vscode/i"], "minNonWhitespaceChars": 10},
        },
        {
            "id": "rag_transparency_no_hits",
            "category": "rag",
            "description": "Explicit local-doc retrieval should use rag_lookup and stay transparent when no local result is available.",
            "prompt": "With RAG only, answer from my local documents: what checkpoint is recorded in documents for this session?",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": True},
            "forceToolUse": "rag",
            "expectedState": ["completed"],
            "clarification": "must_not_happen",
            "expectedToolsMode": "must_include",
            "expectedTools": ["rag_lookup"],
            "final": {
                "includesOneOf": [
                    "/no relevant local/i",
                    "/no document/i",
                    "/rag unavailable/i",
                    "/i did not find/i",
                    "/no results/i",
                ],
                "minNonWhitespaceChars": 20,
            },
        },
        {
            "id": "clarification_resume_food_app_multiturn",
            "category": "multi_turn",
            "description": "A broad app request should clarify first, then continue once the user answers.",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": False},
            "expectedState": ["completed"],
            "expectedToolsMode": "must_include",
            "expectedTools": ["request_clarification"],
            "turns": [
                {
                    "prompt": "Help me create a food application.",
                    "expectedState": ["awaiting_clarification"],
                    "clarification": "must_happen",
                    "expectedToolsMode": "must_include",
                    "expectedTools": ["request_clarification"],
                },
                {
                    "prompt": "For iPhone, with nearby restaurants and favorites, very simple MVP.",
                    "expectedState": ["completed"],
                    "clarification": "must_not_happen",
                    "final": {
                        "includesOneOf": ["/iphone/i", "/restaurants?/i", "/mvp/i", "/favorites?/i"],
                        "minNonWhitespaceChars": 30,
                    },
                },
            ],
            "final": {
                "includesOneOf": ["/iphone/i", "/restaurants?/i", "/mvp/i", "/favorites?/i"],
                "minNonWhitespaceChars": 30,
            },
        },
        {
            "id": "memory_store_then_recall_multiturn",
            "category": "multi_turn",
            "description": "Session memory should survive a follow-up turn in the same session.",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": True},
            "expectedState": ["completed"],
            "expectedToolsMode": "must_include",
            "expectedTools": ["session_memory_upsert"],
            "turns": [
                {
                    "prompt": "Remember for this session that the preferred editor is VS Code, then confirm in one line.",
                    "expectedState": ["completed"],
                    "clarification": "must_not_happen",
                    "expectedToolsMode": "must_include",
                    "expectedTools": ["session_memory_upsert"],
                    "final": {"includesOneOf": ["/vs code/i", "/vscode/i"], "minNonWhitespaceChars": 10},
                },
                {
                    "prompt": "What is my editor preference for this session? Answer very briefly.",
                    "expectedState": ["completed"],
                    "clarification": "must_not_happen",
                    "final": {"includesOneOf": ["/vs code/i", "/vscode/i"], "minNonWhitespaceChars": 6},
                },
            ],
            "final": {"includesOneOf": ["/vs code/i", "/vscode/i"], "minNonWhitespaceChars": 6},
        },
        {
            "id": "files_write_then_followup_read_multiturn",
            "category": "multi_turn",
            "description": "A file created on one turn should be readable on the next turn in the same session.",
            "allowWrites": True,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": False},
            "expectedState": ["completed"],
            "expectedToolsMode": "must_include_all",
            "expectedTools": ["write_file", "read_file"],
            "artifacts": {
                "fileExists": [followup_rel],
                "fileContains": [{"path": followup_rel, "includes": "bonjour-suivi"}],
            },
            "turns": [
                {
                    "prompt": f"Create the file {followup_rel} with the text bonjour-suivi and reply done.",
                    "expectedState": ["completed"],
                    "clarification": "must_not_happen",
                    "expectedToolsMode": "must_include",
                    "expectedTools": ["write_file"],
                    "final": {"includesOneOf": ["/done/i", "/file/i", "/created/i"]},
                },
                {
                    "prompt": f"Read {followup_rel} and reply with the exact content only.",
                    "expectedState": ["completed"],
                    "clarification": "must_not_happen",
                    "expectedToolsMode": "must_include",
                    "expectedTools": ["read_file"],
                    "final": {"exactNormalizedOneOf": ["bonjour-suivi"]},
                },
            ],
            "final": {"exactNormalizedOneOf": ["bonjour-suivi"]},
        },
        {
            "id": "memory_recall_after_interruption_multiturn",
            "category": "multi_turn",
            "description": "Session memory should still be recallable after an unrelated turn.",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": True},
            "expectedState": ["completed"],
            "expectedToolsMode": "must_include",
            "expectedTools": ["session_memory_upsert"],
            "turns": [
                {
                    "prompt": "Remember for this session that my preferred stack is Next.js with FastAPI.",
                    "expectedState": ["completed"],
                    "expectedToolsMode": "must_include",
                    "expectedTools": ["session_memory_upsert"],
                    "clarification": "must_not_happen",
                },
                {
                    "prompt": "Thanks, and now just say bon.",
                    "expectedState": ["completed"],
                    "clarification": "must_not_happen",
                    "final": {"includesOneOf": ["/bon/i"], "minNonWhitespaceChars": 3},
                },
                {
                    "prompt": "Which stack did I say I prefer for this session?",
                    "expectedState": ["completed"],
                    "clarification": "must_not_happen",
                    "final": {"includesOneOf": ["/next\\.?js/i", "/fastapi/i"], "minNonWhitespaceChars": 12},
                },
            ],
            "final": {"includesOneOf": ["/next\\.?js/i", "/fastapi/i"], "minNonWhitespaceChars": 12},
        },
        {
            "id": "rag_session_docs_multiturn_backend",
            "category": "multi_turn",
            "description": "Session-scoped RAG docs should be usable across follow-up turns on backend relay.",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": True},
            "supportedSurfaces": ["backend_relay"],
            "prep": {"ragSessionImportPaths": [fixture_rel]},
            "expectedState": ["completed"],
            "expectedToolsMode": "must_include",
            "expectedTools": ["rag_lookup"],
            "turns": [
                {
                    "prompt": "With the RAG for this session, what checkpoint is recorded in the indexed document? Answer very briefly.",
                    "expectedState": ["completed"],
                    "expectedToolsMode": "must_include",
                    "expectedTools": ["rag_lookup"],
                    "clarification": "must_not_happen",
                    "final": {"includesOneOf": ["/aurora_phase4/i"], "minNonWhitespaceChars": 8},
                },
                {
                    "prompt": "And what is the Phase 4 status? Answer very briefly.",
                    "expectedState": ["completed"],
                    "clarification": "must_not_happen",
                    "final": {
                        "includesOneOf": ["/very advanced/i", "/close to closure/i"],
                        "minNonWhitespaceChars": 12,
                    },
                },
            ],
            "final": {"includesOneOf": ["/very advanced/i", "/close to closure/i"], "minNonWhitespaceChars": 12},
        },
        {
            "id": "session_restore_context_backend",
            "category": "multi_turn",
            "description": "Backend session restore should preserve conversational context across separate runs.",
            "allowWrites": False,
            "toolToggles": {"clarification": True, "appActions": False, "webSearch": False, "rag": False},
            "supportedSurfaces": ["backend_relay"],
            "expectedState": ["completed"],
            "turns": [
                {
                    "prompt": f"Read {fixture_rel} and answer very briefly: what checkpoint is recorded in the file?",
                    "expectedState": ["completed"],
                    "clarification": "must_not_happen",
                    "expectedToolsMode": "must_include",
                    "expectedTools": ["read_file"],
                    "final": {"includesOneOf": ["/aurora_phase4/i"], "minNonWhitespaceChars": 8},
                },
                {
                    "prompt": "After session restore, remind me of the checkpoint, very briefly.",
                    "expectedState": ["completed"],
                    "clarification": "must_not_happen",
                    "final": {"includesOneOf": ["/aurora_phase4/i"], "minNonWhitespaceChars": 8},
                },
            ],
            "final": {"includesOneOf": ["/aurora_phase4/i"], "minNonWhitespaceChars": 8},
        },
    ]
