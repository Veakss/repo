import asyncio
import httpx
import json

async def main():
    async with httpx.AsyncClient() as client:
        req = {
            "sessionId": "test-session-123",
            "messages": [{"role": "user", "content": "crée une app"}],
            "workspaceRoot": "/Users/victor/Documents/continue-better/streamlit-python-only",
            "model": "google/gemini-2.5-flash-lite-preview-09-2025",
            "policyProfile": "ask_when_necessary",
            "toolToggles": {
                "webSearch": True,
                "rag": True,
                "appActions": True,
                "clarification": True
            }
        }
        async with client.stream("POST", "http://127.0.0.1:4001/v1/chat/stream", json=req, timeout=30.0) as response:
            async for line in response.aiter_lines():
                if line:
                    print(line)

asyncio.run(main())
