from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
from streamlit_python_only.mcp.gateway.registry import GatewayRegistry
from streamlit_python_only.settings import get_settings


def _search_with_searxng(query: str, limit: int) -> list[dict[str, str]]:
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
    results: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        title = str(row.get("title") or "").strip() or url
        if not url:
            continue
        results.append({"title": title, "url": url})
        if len(results) >= limit:
            break
    return results


def _search_with_duckduckgo(query: str, limit: int) -> list[dict[str, str]]:
    search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    response = httpx.get(
        search_url,
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": "ContinueBetterPython/1.0"},
    )
    response.raise_for_status()
    results: list[dict[str, str]] = []
    for url, title in re.findall(r'<a[^>]+class=\"result__a\"[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>', response.text, flags=re.IGNORECASE):
        if url.startswith("//"):
            url = f"https:{url}"
        parsed = urlparse(url)
        if "duckduckgo.com" in parsed.netloc:
            target = parse_qs(parsed.query).get("uddg", [])
            if target:
                url = unquote(target[0])
        clean_title = re.sub(r"<[^>]+>", "", title).strip()
        results.append({"title": clean_title, "url": url})
        if len(results) >= limit:
            break
    return results


def register_web_tools(registry: GatewayRegistry) -> None:
    def web_search(arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        limit = int(arguments.get("limit", 5) or 5)
        if not query:
            return {"status": "error", "payload": {"error": "Missing query"}, "evidence_kind": "web"}
        bounded_limit = max(1, min(10, limit))
        try:
            results = _search_with_searxng(query, bounded_limit)
            provider = "searxng" if results else "duckduckgo"
            if not results:
                results = _search_with_duckduckgo(query, bounded_limit)
        except Exception as exc:
            return {"status": "error", "payload": {"error": str(exc)}, "evidence_kind": "web"}
        if not results:
            return {"status": "error", "payload": {"error": "No search results found"}, "evidence_kind": "web"}
        return {"status": "ok", "payload": {"results": results, "provider": provider}, "evidence_kind": "web"}

    registry.register("web_search", web_search)
