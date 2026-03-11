from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
from typing import Any

import streamlit as st

from continue_better_py.settings import get_settings
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from frontend.api_client import BackendClient
    from frontend.ui_state import build_timeline_html, derive_pending_items, flatten_tree, merge_timeline
except ModuleNotFoundError:
    from api_client import BackendClient
    from ui_state import build_timeline_html, derive_pending_items, flatten_tree, merge_timeline


ClientFactory = Callable[[], BackendClient]


def ensure_state() -> None:
    settings = get_settings()
    defaults: dict[str, Any] = {
        "backend_url": settings.backend_base_url,
        "sessions": [],
        "session_id": None,
        "messages": [],
        "timeline": [],
        "known_runs": [],
        "active_run_id": None,
        "capabilities": None,
        "models": [],
        "pending_approvals": [],
        "pending_clarification": None,
        "selected_file": None,
        "file_tree": None,
        "file_content": "",
        "status_message": None,
        "ui_error": None,
        "selected_model": settings.llm_model,
        "composer_value": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_client(client_factory: ClientFactory | None = None) -> BackendClient:
    if client_factory:
        return client_factory()
    return BackendClient(st.session_state["backend_url"])


def set_status(message: str | None = None, error: str | None = None) -> None:
    st.session_state["status_message"] = message
    st.session_state["ui_error"] = error


def refresh_bootstrap(client: BackendClient) -> None:
    capabilities = client.capabilities()
    models_payload = client.models()
    sessions = client.list_sessions()
    st.session_state["capabilities"] = capabilities
    st.session_state["models"] = models_payload.get("models", [])
    st.session_state["sessions"] = sessions
    if sessions and st.session_state["session_id"] not in {session["id"] for session in sessions}:
        st.session_state["session_id"] = sessions[0]["id"]
    if not sessions:
        st.session_state["session_id"] = None


def refresh_session_state(client: BackendClient, session_id: str | None) -> None:
    if not session_id:
        st.session_state["messages"] = []
        st.session_state["known_runs"] = []
        st.session_state["timeline"] = []
        st.session_state["active_run_id"] = None
        st.session_state["pending_approvals"] = []
        st.session_state["pending_clarification"] = None
        return

    messages = client.get_session_messages(session_id, limit=600)
    runs = client.list_runs(session_id)
    timeline: list[dict[str, Any]] = []
    active_run_id = runs[0]["run_id"] if runs else None
    if active_run_id:
        run = client.get_run(active_run_id)
        timeline = run.get("events", [])
    st.session_state["messages"] = messages
    st.session_state["known_runs"] = runs
    st.session_state["timeline"] = timeline
    st.session_state["active_run_id"] = active_run_id
    approvals, clarification = derive_pending_items(timeline)
    st.session_state["pending_approvals"] = approvals
    st.session_state["pending_clarification"] = clarification


def refresh_files(client: BackendClient) -> None:
    tree = client.fs_tree()
    st.session_state["file_tree"] = tree
    files = flatten_tree(tree)
    current = st.session_state.get("selected_file")
    if files and current not in files:
        st.session_state["selected_file"] = files[0]
    if not files:
        st.session_state["selected_file"] = None
        st.session_state["file_content"] = ""
        return
    selected = st.session_state["selected_file"]
    if selected:
        file_payload = client.fs_read(selected)
        st.session_state["file_content"] = file_payload.get("content", "")


def bootstrap(client: BackendClient) -> None:
    refresh_bootstrap(client)
    refresh_session_state(client, st.session_state["session_id"])
    try:
        refresh_files(client)
    except Exception:
        st.session_state["file_tree"] = None
        st.session_state["file_content"] = ""


def consume_events(client: BackendClient, session_id: str, events: list[dict[str, Any]], assistant_text: str | None = None) -> None:
    st.session_state["timeline"] = merge_timeline(st.session_state["timeline"], events)
    approvals, clarification = derive_pending_items(st.session_state["timeline"])
    st.session_state["pending_approvals"] = approvals
    st.session_state["pending_clarification"] = clarification
    for event in reversed(events):
        run_id = event.get("runId")
        if isinstance(run_id, str) and run_id:
            st.session_state["active_run_id"] = run_id
            break
    if assistant_text:
        st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
    refresh_session_state(client, session_id)


def create_session(client: BackendClient, title: str | None = None) -> None:
    created = client.create_session(title)
    st.session_state["session_id"] = created["session_id"]
    refresh_bootstrap(client)
    refresh_session_state(client, st.session_state["session_id"])
    set_status(message="Session created")


def on_send(prompt: str, client: BackendClient) -> None:
    session_id = st.session_state["session_id"]
    if not session_id:
        create_session(client, "New session")
        session_id = st.session_state["session_id"]
    assert session_id is not None

    payload = {
        "session_id": session_id,
        "message": prompt,
        "workspace_root": str(get_settings().resolved_workspace_root),
        "model": st.session_state["selected_model"],
        "allow_writes": True,
    }
    events: list[dict[str, Any]] = []
    assistant_text = ""
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for event in client.stream_chat(payload):
            events.append(event)
            if event.get("type") == "token":
                assistant_text += event.get("token", "")
                placeholder.markdown(assistant_text)
        if not assistant_text:
            placeholder.markdown("_Waiting for tool result or approval._")
    consume_events(client, session_id, events, assistant_text=assistant_text or None)
    set_status(message="Run completed" if assistant_text else "Run paused for action")


def respond_to_approval(client: BackendClient, approval_id: str, decision: str) -> None:
    session_id = st.session_state["session_id"]
    if not session_id:
        return
    events = list(client.stream_approval(approval_id, decision))
    assistant_text = "".join(event.get("token", "") for event in events if event.get("type") == "token").strip()
    consume_events(client, session_id, events, assistant_text=assistant_text or None)
    set_status(message=f"Approval {decision}")


def respond_to_clarification(client: BackendClient, clarification_id: str, answer: str) -> None:
    session_id = st.session_state["session_id"]
    if not session_id:
        return
    events = list(client.stream_clarification(clarification_id, answer))
    assistant_text = "".join(event.get("token", "") for event in events if event.get("type") == "token").strip()
    consume_events(client, session_id, events, assistant_text=assistant_text or None)
    set_status(message="Clarification sent")


def render_session_sidebar(client: BackendClient) -> None:
    st.sidebar.title("Continue Better")
    st.sidebar.caption("Python rewrite")
    st.sidebar.text_input("Backend URL", key="backend_url")
    left, right = st.sidebar.columns(2)
    if left.button("Refresh", use_container_width=True):
        bootstrap(get_client())
        set_status(message="Backend refreshed")
    if right.button("New Session", use_container_width=True):
        create_session(client)

    with st.sidebar.expander("Session Actions", expanded=True):
        selected = st.session_state["session_id"]
        sessions = st.session_state["sessions"]
        current = next((session for session in sessions if session["id"] == selected), None)
        title_value = current["title"] if current else ""
        new_title = st.text_input("Rename", value=title_value, key="rename_session_value")
        rename_col, delete_col = st.columns(2)
        if rename_col.button("Rename", use_container_width=True, disabled=not current):
            client.update_session(selected, new_title)
            refresh_bootstrap(client)
            refresh_session_state(client, selected)
            set_status(message="Session renamed")
        if delete_col.button("Delete", use_container_width=True, disabled=not current):
            client.delete_session(selected)
            refresh_bootstrap(client)
            refresh_session_state(client, st.session_state["session_id"])
            set_status(message="Session deleted")

    st.sidebar.markdown("### Sessions")
    for session in st.session_state["sessions"]:
        selected = session["id"] == st.session_state["session_id"]
        label = f"{'●' if selected else '○'} {session['title']}"
        help_text = f"Messages: {session.get('message_count', 0)}"
        if st.sidebar.button(label, key=f"session-{session['id']}", use_container_width=True, help=help_text):
            st.session_state["session_id"] = session["id"]
            refresh_session_state(client, session["id"])
            refresh_files(client)


def render_header() -> None:
    capabilities = st.session_state.get("capabilities") or {}
    models = st.session_state.get("models") or []
    model_labels = [row.get("id") or row.get("name") or "unknown-model" for row in models] or [get_settings().llm_model]
    if st.session_state["selected_model"] not in model_labels:
        st.session_state["selected_model"] = model_labels[0]

    title_col, meta_col = st.columns([2.3, 1.1])
    with title_col:
        st.title("Continue Better")
        st.caption("Streamlit control plane for the Python-only stack")
    with meta_col:
        st.selectbox("Model", options=model_labels, key="selected_model")
        badges = []
        if capabilities.get("webSearch"):
            badges.append("Web")
        if capabilities.get("rag"):
            badges.append("RAG")
        if capabilities.get("interactiveTerminal"):
            badges.append("Terminal")
        if capabilities.get("clarification"):
            badges.append("Clarification")
        st.caption(" | ".join(badges) or "No capabilities reported")

    metrics = st.columns(4)
    metrics[0].metric("Session", st.session_state["session_id"] or "none")
    metrics[1].metric("Active Run", st.session_state["active_run_id"] or "none")
    metrics[2].metric("Pending Approvals", len(st.session_state["pending_approvals"]))
    metrics[3].metric("Pending Clarification", "yes" if st.session_state["pending_clarification"] else "no")


def render_chat_panel(client: BackendClient) -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask the agent")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        on_send(prompt, client)
        st.rerun()


def render_timeline_panel() -> None:
    st.markdown(build_timeline_html(st.session_state["timeline"]), unsafe_allow_html=True)


def render_approvals_panel(client: BackendClient) -> None:
    approvals = st.session_state["pending_approvals"]
    if not approvals:
        st.info("No pending approvals.")
        return
    for approval in approvals:
        st.markdown(f"**{approval['name']}**")
        st.caption(f"Risk: {approval['risk_level']} | Run: {approval['run_id']}")
        st.code(approval["arguments"] or "{}", language="json")
        approve_col, reject_col = st.columns(2)
        if approve_col.button("Approve", key=f"approve-{approval['approval_id']}", use_container_width=True):
            respond_to_approval(client, approval["approval_id"], "approved")
            st.rerun()
        if reject_col.button("Reject", key=f"reject-{approval['approval_id']}", use_container_width=True):
            respond_to_approval(client, approval["approval_id"], "rejected")
            st.rerun()
        st.markdown("---")


def render_clarification_panel(client: BackendClient) -> None:
    clarification = st.session_state["pending_clarification"]
    if not clarification:
        st.info("No clarification needed.")
        return
    st.markdown(f"**{clarification['question'] or 'Clarification required'}**")
    for idx, question in enumerate(clarification.get("questions", [])[:3], start=1):
        st.write(f"{idx}. {question}")
    options = [option["label"] for option in clarification.get("options", []) if isinstance(option, dict) and option.get("label")]
    if options:
        st.caption("Quick options")
        st.write(" | ".join(options))
    with st.form("clarification_form"):
        answer = st.text_area("Answer", value="")
        submitted = st.form_submit_button("Send Clarification", use_container_width=True)
    if submitted and answer.strip():
        respond_to_clarification(client, clarification["clarification_id"], answer.strip())
        st.rerun()


def render_files_panel(client: BackendClient) -> None:
    controls = st.columns([1.5, 1])
    if controls[1].button("Refresh Files", use_container_width=True):
        refresh_files(client)
    tree = st.session_state.get("file_tree")
    if not tree:
        st.info("Filesystem panel unavailable.")
        return
    files = flatten_tree(tree)
    if not files:
        st.info("Workspace is empty.")
        return
    selected = st.selectbox("Workspace Files", options=files, key="selected_file")
    if selected:
        content = client.fs_read(selected)
        st.session_state["file_content"] = content.get("content", "")
    st.code(st.session_state.get("file_content", ""), language="python")


def render_status_panels(client: BackendClient) -> None:
    tabs = st.tabs(["Timeline", "Approvals", "Clarification", "Files", "RAG", "Terminal", "Matrix"])
    with tabs[0]:
        render_timeline_panel()
    with tabs[1]:
        render_approvals_panel(client)
    with tabs[2]:
        render_clarification_panel(client)
    with tabs[3]:
        render_files_panel(client)
    with tabs[4]:
        st.info("RAG UI panel is wired as a placeholder until the RAG backend surfaces are ported in phase 4.")
    with tabs[5]:
        st.info("Interactive terminal UI waits on the Python terminal runtime surfaces.")
    with tabs[6]:
        st.info("Matrix Lab moves in phase 5. This tab remains reserved to preserve the product layout.")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255,255,255,0.08), transparent 28%),
                radial-gradient(circle at bottom right, rgba(126, 231, 135, 0.08), transparent 20%),
                linear-gradient(180deg, #09111b, #111926 48%, #141e2a);
            color: #e9eef6;
        }
        [data-testid="stSidebar"] {
            background: rgba(13, 20, 29, 0.78);
            border-right: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(14px);
        }
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            backdrop-filter: blur(12px);
        }
        .cb-timeline-wrap {
            display: grid;
            gap: 12px;
        }
        .cb-timeline-card, .cb-empty-card {
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 14px 16px;
            backdrop-filter: blur(10px);
        }
        .cb-timeline-head {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            font-size: 0.82rem;
            color: rgba(233, 238, 246, 0.9);
            margin-bottom: 8px;
        }
        .cb-timeline-card pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            color: rgba(220, 232, 245, 0.85);
            font-size: 0.75rem;
        }
        .cb-tone-error {
            border-color: rgba(248, 113, 113, 0.4);
        }
        .cb-tone-warn {
            border-color: rgba(251, 191, 36, 0.35);
        }
        .cb-tone-success {
            border-color: rgba(74, 222, 128, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_app(client_factory: ClientFactory | None = None) -> None:
    st.set_page_config(page_title="Continue Better Python", layout="wide")
    ensure_state()
    inject_css()
    client = get_client(client_factory)

    try:
        if st.session_state["capabilities"] is None:
            bootstrap(client)
    except Exception as exc:
        set_status(error=f"Bootstrap failed: {exc}")

    render_session_sidebar(client)
    render_header()
    if st.session_state["status_message"]:
        st.success(st.session_state["status_message"])
    if st.session_state["ui_error"]:
        st.error(st.session_state["ui_error"])

    left, right = st.columns([1.85, 1.15], gap="large")
    with left:
        render_chat_panel(client)
    with right:
        render_status_panels(client)


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
