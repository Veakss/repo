from __future__ import annotations

import json
from typing import Iterator

import httpx
import streamlit as st

from continue_better_py.settings import get_settings


def iter_sse_events(response: httpx.Response) -> Iterator[dict]:
    buffer = ""
    for chunk in response.iter_text():
        buffer += chunk
        parts = buffer.split("\n\n")
        buffer = parts.pop() or ""
        for part in parts:
            if not part.startswith("data: "):
                continue
            payload = part[6:].strip()
            if not payload:
                continue
            yield json.loads(payload)


def ensure_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", "default")
    st.session_state.setdefault("timeline", [])


def render_sidebar(api_base: str) -> None:
    st.sidebar.title("Continue Better")
    st.sidebar.caption("Python rewrite")
    st.sidebar.text_input("Backend URL", value=api_base, key="backend_url")
    st.sidebar.text_input("Session ID", key="session_id")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Panels")
    st.sidebar.caption("File explorer, RAG, approvals, terminal, and Matrix panels are scaffolded next.")


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="Continue Better Python", layout="wide")
    ensure_state()

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255,255,255,0.08), transparent 30%),
                linear-gradient(180deg, #09111b, #111926 48%, #141e2a);
            color: #e9eef6;
        }
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            backdrop-filter: blur(12px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    api_base = st.session_state.get("backend_url", settings.backend_base_url)
    render_sidebar(api_base)

    left, right = st.columns([2.3, 1.2], gap="large")

    with left:
        st.title("Continue Better")
        st.caption("Streamlit shell for the Python-only migration")

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask the agent")
        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                assistant_text = ""
                with httpx.Client(timeout=None) as client:
                    with client.stream(
                        "POST",
                        f"{api_base}/chat/stream",
                        json={
                            "session_id": st.session_state["session_id"],
                            "message": prompt,
                            "workspace_root": str(settings.resolved_workspace_root),
                        },
                    ) as response:
                        response.raise_for_status()
                        for event in iter_sse_events(response):
                            if event.get("type") == "token":
                                assistant_text += event.get("token", "")
                                placeholder.markdown(assistant_text)
                            elif event.get("type") == "run_phase_changed":
                                st.session_state["timeline"].append(event)
                            elif event.get("type") == "run_state":
                                st.session_state["timeline"].append(event)
                st.session_state["messages"].append({"role": "assistant", "content": assistant_text or "(empty response)"})

    with right:
        st.subheader("Timeline")
        for event in reversed(st.session_state["timeline"][-20:]):
            st.code(json.dumps(event, indent=2), language="json")

        st.subheader("Planned Panels")
        st.write("- Sessions")
        st.write("- RAG")
        st.write("- Files")
        st.write("- Approvals")
        st.write("- Terminal")
        st.write("- Matrix Lab")


if __name__ == "__main__":
    main()
