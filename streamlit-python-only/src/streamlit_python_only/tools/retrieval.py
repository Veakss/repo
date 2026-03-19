from __future__ import annotations

import json
import re
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from streamlit_python_only.rag import RagService, render_lookup_for_model
from streamlit_python_only.settings import get_settings
from streamlit_python_only.tooling_shared import ToolMetadata


class RagLookupInput(BaseModel):
    question: str = Field(description="The question to answer using the active RAG sources.")
    profiles: str | None = Field(default=None, description="Optional comma-separated profile names to restrict profile retrieval.")


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query.")
    limit: int = Field(default=5, ge=1, le=10, description="Maximum number of search results to return.")


class SessionMemoryUpsertInput(BaseModel):
    content: str = Field(description="Important session memory to store.")
    kind: str = Field(default="preference", description="Short memory kind label.")


def build_retrieval_tool_definitions(session_id: str | None, rag_service: RagService) -> list[ToolMetadata]:
    def _search_with_searxng(query: str, limit: int) -> list[tuple[str, str]]:
        settings = get_settings()
        if not settings.enable_searxng_web_search:
            return []
        base_url = str(settings.searxng_base_url or "").strip().rstrip("/")
        if not base_url:
            return []
        timeout = max(1000, int(settings.searxng_timeout_ms or 8000)) / 1000.0
        response = httpx.get(
            f"{base_url}/search",
            params={"q": query, "format": "json"},
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "ContinueBetterPython/1.0"},
        )
        response.raise_for_status()
        payload_data = response.json()
        payload = payload_data if isinstance(payload_data, dict) else {}
        rows = payload.get("results", []) if isinstance(payload.get("results"), list) else []
        matches: list[tuple[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or "").strip()
            title = str(row.get("title") or "").strip() or url
            if not url:
                continue
            matches.append((title, url))
            if len(matches) >= limit:
                break
        return matches

    def _search_with_duckduckgo(query: str, limit: int) -> list[tuple[str, str]]:
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        response = httpx.get(search_url, timeout=15.0, follow_redirects=True, headers={"User-Agent": "ContinueBetterPython/1.0"})
        response.raise_for_status()
        matches: list[tuple[str, str]] = []
        for url, title in re.findall(r'<a[^>]+class=\"result__a\"[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>', response.text, flags=re.IGNORECASE):
            if url.startswith("//"):
                url = f"https:{url}"
            parsed = urlparse(url)
            if "duckduckgo.com" in parsed.netloc:
                target = parse_qs(parsed.query).get("uddg", [])
                if target:
                    url = unquote(target[0])
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            matches.append((clean_title, url))
            if len(matches) >= limit:
                break
        return matches

    def rag_lookup(question: str, profiles: str | None = None) -> str:
        profile_list = [item.strip() for item in (profiles or "").split(",") if item.strip()] or None
        lookup = rag_service.lookup(
            question=question,
            session_id=session_id,
            scope={"profiles": profile_list} if profile_list else None,
        )
        return json.dumps({"kind": "rag_tool", "lookup": lookup, "rendered": render_lookup_for_model(lookup)}, ensure_ascii=False)

    def web_search(query: str, limit: int = 5) -> str:
        bounded_limit = max(1, min(10, int(limit)))
        matches: list[str] = []
        provider = "none"
        try:
            searx_rows = _search_with_searxng(query, bounded_limit)
            if searx_rows:
                provider = "searxng"
                matches = [f"- {title}: {url}" for title, url in searx_rows]
        except Exception:
            matches = []
        if not matches:
            ddg_rows = _search_with_duckduckgo(query, bounded_limit)
            provider = "duckduckgo"
            matches = [f"- {title}: {url}" for title, url in ddg_rows]
        if not matches:
            return "No search results found."
        formatted = []
        for row in matches:
            title, _, url = row.partition(": ")
            clean_title = title.strip("- ").strip() or "Source"
            clean_url = url.strip()
            formatted.append(f"- [{clean_title}]({clean_url})")
        return f"Web search results ({provider}):\n" + "\n".join(formatted)

    definitions = [
        ToolMetadata(
            tool=StructuredTool.from_function(web_search, name="web_search", description="Search the web and return compact source links.", args_schema=WebSearchInput),
            risk_level="safe",
            module_id="web",
            supports_thales=True,
            supports_native_tools=True,
            supports_textual_replay=True,
            returns_evidence_kind="web",
        )
    ]
    if session_id:
        def session_memory_upsert(content: str, kind: str = "preference") -> str:
            state = rag_service.append_memory_note(session_id=session_id, content=content, kind=kind)
            return f"Stored session memory ({kind}). Entry count: {state['summary']['entryCount']}"

        definitions.extend(
            [
                ToolMetadata(
                    tool=StructuredTool.from_function(rag_lookup, name="rag_lookup", description="Search Session Docs, Profiles, and Session Memory for relevant context with citations and transparency.", args_schema=RagLookupInput),
                    risk_level="safe",
                    module_id="rag",
                    supports_thales=True,
                    supports_native_tools=True,
                    supports_textual_replay=True,
                    returns_evidence_kind="rag",
                ),
                ToolMetadata(
                    tool=StructuredTool.from_function(session_memory_upsert, name="session_memory_upsert", description="Store an explicit fact or preference into the current session memory.", args_schema=SessionMemoryUpsertInput),
                    risk_level="safe",
                    module_id="memory",
                    supports_thales=True,
                    supports_native_tools=True,
                    supports_textual_replay=True,
                    returns_evidence_kind="memory",
                ),
            ]
        )
    return definitions
