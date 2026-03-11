from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from continue_better_py.settings import get_settings
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from frontend.api_client import BackendClient
    from frontend.ui_state import build_terminal_component_html, build_terminal_html, build_timeline_html, derive_pending_items, flatten_tree, merge_timeline
except ModuleNotFoundError:
    from api_client import BackendClient
    from ui_state import build_terminal_component_html, build_terminal_html, build_timeline_html, derive_pending_items, flatten_tree, merge_timeline


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
        "rag_profiles": [],
        "rag_session_files": [],
        "rag_profile_files": [],
        "rag_jobs": [],
        "rag_memory": None,
        "selected_rag_profile": None,
        "rag_lookup_response": None,
        "matrix_catalog": None,
        "matrix_reports": [],
        "matrix_jobs": [],
        "matrix_selected_report_id": None,
        "matrix_report": None,
        "matrix_compare_report_id": None,
        "matrix_compare_payload": None,
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
    refresh_rag_state(client)
    refresh_matrix_state(client)


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
    refresh_rag_state(client)
    refresh_matrix_state(client)


def refresh_rag_state(client: BackendClient) -> None:
    try:
        profiles = client.rag_list_profiles().get("profiles", [])
        st.session_state["rag_profiles"] = profiles
        if profiles and st.session_state.get("selected_rag_profile") not in profiles:
            st.session_state["selected_rag_profile"] = profiles[0]
        if not profiles:
            st.session_state["selected_rag_profile"] = None
        session_id = st.session_state.get("session_id")
        st.session_state["rag_session_files"] = client.rag_get_session_files(session_id).get("files", []) if session_id else []
        selected_profile = st.session_state.get("selected_rag_profile")
        st.session_state["rag_profile_files"] = client.rag_get_profile_files(selected_profile).get("files", []) if selected_profile else []
        st.session_state["rag_jobs"] = client.rag_list_index_jobs(limit=20).get("jobs", [])
        st.session_state["rag_memory"] = client.rag_get_session_memory(session_id, 80) if session_id else None
    except Exception:
        pass


def refresh_matrix_state(client: BackendClient) -> None:
    try:
        catalog = client.matrix_catalog()
        reports = client.matrix_list_reports(limit=30)
        jobs = client.matrix_list_jobs(limit=20)
        st.session_state["matrix_catalog"] = catalog
        st.session_state["matrix_reports"] = reports
        st.session_state["matrix_jobs"] = jobs
        selected_report_id = st.session_state.get("matrix_selected_report_id")
        available_ids = {report["report_id"] for report in reports}
        if reports and selected_report_id not in available_ids:
            selected_report_id = reports[0]["report_id"]
            st.session_state["matrix_selected_report_id"] = selected_report_id
        if selected_report_id:
            st.session_state["matrix_report"] = client.matrix_get_report(selected_report_id)
        else:
            st.session_state["matrix_report"] = None
            st.session_state["matrix_selected_report_id"] = None
        compare_id = st.session_state.get("matrix_compare_report_id")
        if compare_id and compare_id not in available_ids:
            st.session_state["matrix_compare_report_id"] = None
            st.session_state["matrix_compare_payload"] = None
    except Exception:
        pass


def create_session(client: BackendClient, title: str | None = None) -> None:
    created = client.create_session(title)
    st.session_state["session_id"] = created["session_id"]
    refresh_bootstrap(client)
    refresh_session_state(client, st.session_state["session_id"])
    refresh_rag_state(client)
    refresh_matrix_state(client)
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
            refresh_rag_state(client)
            set_status(message="Session renamed")
        if delete_col.button("Delete", use_container_width=True, disabled=not current):
            client.delete_session(selected)
            refresh_bootstrap(client)
            refresh_session_state(client, st.session_state["session_id"])
            refresh_rag_state(client)
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
            refresh_rag_state(client)


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


def render_rag_panel(client: BackendClient) -> None:
    session_id = st.session_state.get("session_id")
    top_cols = st.columns([1.3, 1])
    if top_cols[1].button("Refresh RAG", use_container_width=True):
        refresh_rag_state(client)

    with st.expander("Session Docs", expanded=True):
        available_files = flatten_tree(st.session_state.get("file_tree") or {"children": []})
        import_choices = [path for path in available_files if Path(path).suffix.lower() in {".md", ".txt", ".json", ".yml", ".yaml"}]
        selected_import = st.selectbox("Import workspace file", options=import_choices or ["No compatible files"], key="rag_session_import")
        import_disabled = not session_id or not import_choices
        if st.button("Import To Session Docs", disabled=import_disabled, use_container_width=True):
            client.rag_import_session_file(session_id, selected_import)
            refresh_rag_state(client)
            set_status(message="Session document imported")
        if st.button("Index Session Docs", disabled=not session_id or not st.session_state["rag_session_files"], use_container_width=True):
            client.rag_enqueue_session_index_job(session_id)
            refresh_rag_state(client)
            set_status(message="Session indexing job enqueued")
        for file_row in st.session_state.get("rag_session_files", []):
            cols = st.columns([2.4, 1, 0.8])
            cols[0].write(file_row.get("source_display_path") or file_row.get("original_name"))
            cols[1].caption(file_row.get("status", "unknown"))
            if cols[2].button("Remove", key=f"rag-session-delete-{file_row['id']}", use_container_width=True):
                client.rag_delete_session_file(session_id, file_row["id"])
                refresh_rag_state(client)
                st.rerun()

    with st.expander("Profiles", expanded=True):
        new_profile = st.text_input("New profile", key="rag_new_profile")
        if st.button("Create Profile", disabled=not new_profile.strip(), use_container_width=True):
            client.rag_create_profile(new_profile.strip())
            refresh_rag_state(client)
            st.rerun()
        profiles = st.session_state.get("rag_profiles", [])
        if profiles:
            selected_profile = st.selectbox("Active profile", options=profiles, key="selected_rag_profile")
            profile_name = st.text_input("Rename selected profile", value=selected_profile, key="rag_profile_rename")
            rename_col, delete_col = st.columns(2)
            if rename_col.button("Rename Profile", use_container_width=True):
                client.rag_rename_profile(selected_profile, profile_name)
                refresh_rag_state(client)
                st.rerun()
            if delete_col.button("Delete Profile", use_container_width=True):
                client.rag_delete_profile(selected_profile)
                refresh_rag_state(client)
                st.rerun()
            profile_import = st.selectbox("Import into profile", options=import_choices or ["No compatible files"], key="rag_profile_import")
            if st.button("Import To Profile", disabled=not import_choices, use_container_width=True):
                client.rag_import_profile_file(selected_profile, profile_import)
                refresh_rag_state(client)
                st.rerun()
            if st.button("Index Profile", disabled=not st.session_state["rag_profile_files"], use_container_width=True):
                client.rag_enqueue_profile_index_job(selected_profile)
                refresh_rag_state(client)
                st.rerun()
            for file_row in st.session_state.get("rag_profile_files", []):
                cols = st.columns([2.4, 1, 0.8])
                cols[0].write(file_row.get("source_display_path") or file_row.get("original_name"))
                cols[1].caption(file_row.get("status", "unknown"))
                if cols[2].button("Remove", key=f"rag-profile-delete-{file_row['id']}", use_container_width=True):
                    client.rag_delete_profile_file(selected_profile, file_row["id"])
                    refresh_rag_state(client)
                    st.rerun()
        else:
            st.info("No profiles yet.")

    with st.expander("Session Memory", expanded=True):
        memory = st.session_state.get("rag_memory")
        if not memory or not session_id:
            st.info("Session memory unavailable.")
        else:
            config = memory["config"]
            enabled = st.checkbox("Enabled", value=bool(config.get("enabled", True)), key="rag_memory_enabled")
            threshold = st.slider("Threshold", min_value=0.5, max_value=0.99, value=float(config.get("thresholdPct", 0.9)), step=0.01, key="rag_memory_threshold")
            token_budget = st.number_input("Token Budget", min_value=1000, max_value=200000, value=int(config.get("tokenBudget", 12000)), step=500, key="rag_memory_budget")
            config_col, compact_col, clear_col = st.columns(3)
            if config_col.button("Save Memory Config", use_container_width=True):
                client.rag_patch_session_memory(session_id, {"enabled": enabled, "threshold_pct": threshold, "token_budget": int(token_budget)})
                refresh_rag_state(client)
                st.rerun()
            if compact_col.button("Compact Memory", use_container_width=True):
                client.rag_compact_session_memory(session_id)
                refresh_rag_state(client)
                st.rerun()
            if clear_col.button("Clear Memory", use_container_width=True):
                client.rag_clear_session_memory(session_id)
                refresh_rag_state(client)
                st.rerun()
            summary = memory["summary"]
            st.caption(f"Entries: {summary.get('entryCount', 0)} | Estimated Tokens: {summary.get('estimatedTokens', 0)}")
            if summary.get("text"):
                st.code(summary["text"], language="markdown")
            for entry in memory.get("entries", [])[:10]:
                st.markdown(f"**{entry.get('kind', 'context')}** · {entry.get('ts', '')}")
                st.write(entry.get("content", ""))

    with st.expander("Index Jobs", expanded=False):
        jobs = st.session_state.get("rag_jobs", [])
        if not jobs:
            st.info("No indexing jobs yet.")
        for job in jobs:
            st.markdown(f"**{job['job_id']}**")
            st.caption(f"{job.get('scope_kind')}:{job.get('scope_id')} | {job.get('status')} | {job.get('stage')}")
            st.progress(int(job.get("progress", {}).get("percent", 0)))

    with st.expander("Lookup", expanded=False):
        question = st.text_input("Lookup question", key="rag_lookup_question")
        if st.button("Run RAG Lookup", disabled=not question.strip(), use_container_width=True):
            scope = {}
            if st.session_state.get("selected_rag_profile"):
                scope["profiles"] = [st.session_state["selected_rag_profile"]]
            st.session_state["rag_lookup_response"] = client.rag_lookup(question.strip(), session_id=session_id, scope=scope or None)
        response = st.session_state.get("rag_lookup_response")
        if response:
            st.caption(f"Status: {response.get('status')}")
            if response.get("message"):
                st.info(response["message"])
            for hit in response.get("hits", [])[:5]:
                citation = hit.get("citation", {})
                st.markdown(f"**{citation.get('path', 'unknown')}**")
                st.caption(f"Score: {hit.get('score')}")
                st.write(hit.get("snippet", ""))


def render_terminal_panel() -> None:
    timeline = st.session_state["timeline"]
    latest_terminal_id = next((str(event.get("terminalId")) for event in reversed(timeline) if event.get("terminalId")), None)
    components.html(
        build_terminal_component_html(
            backend_url=st.session_state["backend_url"],
            workspace_root=str(get_settings().resolved_workspace_root),
            session_id=st.session_state.get("session_id"),
            run_id=st.session_state.get("active_run_id"),
            initial_terminal_id=latest_terminal_id,
        ),
        height=470,
        scrolling=False,
    )
    st.markdown("##### Recent Terminal Events")
    st.markdown(build_terminal_html(timeline), unsafe_allow_html=True)


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def render_matrix_panel(client: BackendClient) -> None:
    catalog = st.session_state.get("matrix_catalog") or {"scenarios": [], "profiles": [], "surfaces": []}
    reports = st.session_state.get("matrix_reports") or []
    jobs = st.session_state.get("matrix_jobs") or []

    controls = st.columns([1.4, 1])
    if controls[1].button("Refresh Matrix", use_container_width=True):
        refresh_matrix_state(client)
        st.rerun()

    model_labels = [row.get("id") or row.get("name") or "unknown-model" for row in st.session_state.get("models") or []] or [get_settings().llm_model]
    scenario_options = [scenario["id"] for scenario in catalog.get("scenarios", [])]
    profile_options = [profile["id"] for profile in catalog.get("profiles", [])]
    surface_options = [surface["id"] for surface in catalog.get("surfaces", [])]

    with st.expander("Launch Run", expanded=True):
        selected_models = st.multiselect("Models", options=model_labels, default=model_labels[:1], key="matrix_models")
        selected_scenarios = st.multiselect("Scenarios", options=scenario_options, default=scenario_options[:3], key="matrix_scenarios")
        selected_profiles = st.multiselect("Profiles", options=profile_options, default=profile_options[:1], key="matrix_profiles")
        selected_surfaces = st.multiselect("Surfaces", options=surface_options, default=surface_options[:1], key="matrix_surfaces")
        repeat = int(st.number_input("Repeat", min_value=1, max_value=5, value=1, step=1, key="matrix_repeat"))
        if st.button("Start Matrix Run", use_container_width=True, disabled=not selected_models or not selected_scenarios or not selected_profiles or not selected_surfaces):
            job = client.matrix_start_job(
                {
                    "models": selected_models,
                    "scenario_ids": selected_scenarios,
                    "profiles": selected_profiles,
                    "surfaces": selected_surfaces,
                    "repeat": repeat,
                }
            )
            st.session_state["matrix_jobs"] = [job] + jobs
            set_status(message=f"Matrix job started: {job['job_id']}")
            refresh_matrix_state(client)
            st.rerun()

    with st.expander("Jobs", expanded=True):
        if not jobs:
            st.info("No matrix jobs yet.")
        for job in jobs[:8]:
            st.markdown(f"**{job['job_id']}**")
            progress = job.get("progress", {})
            total = int(progress.get("total") or 0)
            completed = int(progress.get("completed") or 0)
            label = progress.get("label") or job.get("status") or "idle"
            st.caption(f"Status: {job.get('status')} | {label}")
            st.progress((completed / total) if total else 0.0)
            if job.get("report_id"):
                st.caption(f"Report: {job['report_id']}")

    with st.expander("Reports", expanded=True):
        if not reports:
            st.info("No matrix reports yet.")
            return
        report_ids = [report["report_id"] for report in reports]
        selected_report_id = st.selectbox("Active report", options=report_ids, key="matrix_selected_report_id")
        if selected_report_id:
            st.session_state["matrix_report"] = client.matrix_get_report(selected_report_id)
        report = st.session_state.get("matrix_report")
        if not report:
            return
        summary = report.get("aggregate", {}).get("byModel", {})
        results = report.get("results", [])
        runs = len(results)
        pass_count = sum(1 for row in results if row["grade"]["overall"] == "pass")
        hard_fail_count = sum(1 for row in results if row["grade"]["overall"] == "hard_fail")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Runs", str(runs))
        metric_cols[1].metric("Pass Rate", _format_ratio(pass_count / runs if runs else None))
        metric_cols[2].metric("Hard Fail Rate", _format_ratio(hard_fail_count / runs if runs else None))
        avg_score = sum(row["grade"]["score"] / row["grade"]["maxScore"] for row in results) / runs if runs else None
        metric_cols[3].metric("Avg Score", _format_ratio(avg_score))

        filters = st.columns(4)
        filter_model = filters[0].selectbox("Filter model", options=["all"] + sorted(report.get("models", [])), index=0, key="matrix_filter_model")
        filter_profile = filters[1].selectbox("Filter profile", options=["all"] + sorted(report.get("profiles", [])), index=0, key="matrix_filter_profile")
        filter_surface = filters[2].selectbox("Filter surface", options=["all"] + sorted(report.get("surfaces", [])), index=0, key="matrix_filter_surface")
        filter_status = filters[3].selectbox("Filter status", options=["all", "pass", "soft_fail", "hard_fail"], index=0, key="matrix_filter_status")

        filtered_rows = []
        for row in results:
            if filter_model != "all" and row["model"] != filter_model:
                continue
            if filter_profile != "all" and row["profile"] != filter_profile:
                continue
            if filter_surface != "all" and row["surface"] != filter_surface:
                continue
            if filter_status != "all" and row["grade"]["overall"] != filter_status:
                continue
            filtered_rows.append(
                {
                    "scenario": row["scenarioId"],
                    "model": row["model"],
                    "profile": row["profile"],
                    "surface": row["surface"],
                    "status": row["grade"]["overall"],
                    "score": f"{row['grade']['score']}/{row['grade']['maxScore']}",
                    "tools": ", ".join(row["summary"].get("tools", [])),
                    "state": row["summary"].get("state"),
                    "latencyMs": row["summary"].get("latencyMs"),
                }
            )
        st.dataframe(filtered_rows or [{"scenario": "none", "status": "n/a"}], use_container_width=True, hide_index=True)

        axis_rows = []
        for key, bucket in sorted(report.get("aggregate", {}).get("byModelProfileSurface", {}).items()):
            axis_rows.append(
                {
                    "axis": key,
                    "runs": bucket.get("runs"),
                    "passRate": _format_ratio(bucket.get("passRate")),
                    "avgScore": _format_ratio(bucket.get("avgScore")),
                    "hardFails": bucket.get("hard_fail"),
                }
            )
        if axis_rows:
            st.caption("Axis summary")
            st.dataframe(axis_rows, use_container_width=True, hide_index=True)

        compare_options = [report_id for report_id in report_ids if report_id != selected_report_id]
        compare_cols = st.columns([1.6, 1])
        compare_choice = compare_cols[0].selectbox(
            "Compare against",
            options=["none"] + compare_options,
            key="matrix_compare_report_id",
        )
        if compare_cols[1].button("Run Compare", use_container_width=True, disabled=compare_choice == "none"):
            st.session_state["matrix_compare_payload"] = client.matrix_compare(selected_report_id, compare_choice)
            st.rerun()

        comparison = st.session_state.get("matrix_compare_payload")
        if comparison and compare_choice != "none":
            st.caption("Comparison summary")
            st.json(comparison["summary"])
            st.dataframe(comparison["byModel"], use_container_width=True, hide_index=True)


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
        render_rag_panel(client)
    with tabs[5]:
        render_terminal_panel()
    with tabs[6]:
        render_matrix_panel(client)


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
        .cb-terminal-meta {
            font-size: 0.74rem;
            color: rgba(220, 232, 245, 0.7);
            margin-bottom: 8px;
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
