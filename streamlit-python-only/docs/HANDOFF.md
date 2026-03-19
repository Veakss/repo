# Handoff Notes

## Current Intent

Port AI Technical Assistant to a Python-only stack inside `streamlit-python-only/`, while keeping Continue Better as the legacy reference baseline.

## Current Status At A Glance

- Frontend scope from this phase is in a good state:
  - `frontend/app.py` is now the single canonical Streamlit shell
  - active Streamlit shell behavior is now frontend-autonomous on the canonical path
  - user-first / assistant-after rendering is enforced more cleanly in chat, clarification resume, and approval resume flows
  - clarification state handling is less noisy and more robust
- Matrix Lab is now strong enough to support a tighter improvement loop:
  - reports distinguish infra issues from product-quality misses
  - reports surface failed dimensions directly in the Streamlit UI
  - exact-output, source quality, message ordering, and RAG quality now have dedicated grading dimensions
- The main remaining work is not broad frontend plumbing anymore.
  It is concentrated on a few live behavior gaps:
  - `web_latest_with_sources`
  - `terminal_sequential_inspect`
  - `rag_session_docs_multiturn_backend`

## Decisions Already Locked

- The rewrite lives in a dedicated folder inside this repo
- UI uses Streamlit plus custom components where needed
- Architecture remains split into frontend, backend, and sidecar
- MongoDB is the canonical persistence layer
- JSON exports remain available
- Providers:
  - OpenRouter
  - Thales
- Thales must support `.env` and YAML config
- enterprise Thales mode must support `config.yaml`-style resolution in addition to the local provider catalog
- Runtime mode is provider-aware and can choose `auto`

## Immediate Next Steps

1. Maintain parity:
   - keep terminal, Matrix, RAG, and provider probes green
   - preserve the current event contract used by the Streamlit shell
2. Future work is now incremental rather than milestone-blocking:
   - keep closing UI gaps against the original JS shell
   - improve targeted Matrix hard fails on live Gemini:
     - `web_latest_with_sources`
     - `terminal_sequential_inspect`
     - `rag_session_docs_multiturn_backend`
   - deeper Thales live validation when credentials are available
   - keep validating the new enterprise YAML + sequential-tool provider path against a real Thales endpoint
   - add an explicit Thales enterprise mode:
     - support `config.yaml` as the expected Windows/Thales provider configuration input
     - model Thales provider capabilities explicitly instead of treating it as a generic OpenAI-compatible provider
     - enforce single-tool-per-turn behavior for Thales rather than assuming multi-tool native orchestration
   - UI refinements and performance polish
   - continue replacing Streamlit-native rough edges with custom surfaces where the JS shell is materially cleaner
   - keep manually verifying the live browser result when overriding Streamlit accent colors or fixed-position controls
3. Keep using the live verification scripts when changing contracts:
   - `verify_lot2_live_mongo.py`
   - `verify_lot3_live.py`
   - `verify_lot4_live.py`
   - `verify_lot5_live.py`
   - `verify_lot6_live.py`
   - `evaluate_capabilities.py`
4. Use `docs/FUTURE_PRIORITIES.md` as the reference for the next hardening phase:
   - domain-by-domain priorities
   - what is essential now vs secondary
   - recommended sequencing for the next improvements

## Practical Constraints

- Docker is not installed on this machine right now, so the local Mongo bootstrap must not assume Docker
- MongoDB 8.0 is now installed locally through Homebrew and running through `brew services`
- port `27017` is open and `mongosh` ping succeeds on this machine

## Legacy Reference Areas

- backend API reference:
  - `/Users/victor/Documents/continue-better/backend/fastapi_app/main.py`
- sidecar runtime reference:
  - `/Users/victor/Documents/continue-better/orchestrator-sidecar/src/continue_adapter.ts`
- frontend UI reference:
  - `/Users/victor/Documents/continue-better/frontend/app/page.tsx`
  - `/Users/victor/Documents/continue-better/frontend/components/`
- Matrix reference:
  - `/Users/victor/Documents/continue-better/frontend/app/matrix/page.tsx`
  - `/Users/victor/Documents/continue-better/scripts/cross_model_matrix_runner.mjs`

## Provider-Specific Reminder

The Windows Thales variant confirmed that standard OpenAI replay with `assistant.tool_calls` and `role=tool` is not reliable for Thales on follow-up turns. Preserve a dedicated textual replay path in the Python sidecar.

Also keep in mind that the target Windows/Thales environment uses a `config.yaml`-driven provider configuration path, and the Thales API should be treated as a single-tool-per-turn runtime surface rather than a provider that safely supports multi-tool native turns.

## Verified So Far

- sidecar starts locally
- backend starts locally
- `/health` works on both services
- `/v1/models` returns a live OpenRouter catalog
- provider unit tests pass
- first real chat attempt reached the model call boundary and failed only because the new subtree has no OpenRouter key configured yet
- approval and clarification resume endpoints are wired and reachable
- capabilities now expose both `files` and `clarification` modules
- `write_file` is now classified as a risky tool in the Python tool registry
- pending approval/clarification snapshots are now persisted under `artifacts/runtime_state/`
- runtime emits `tool_call` and `tool_result` events
- sidecar HTTP contract is covered by API tests
- `scripts/verify_lot1.py` exists for reproducible runtime/provider checks
- `streamlit-python-only/.env` is now locally configured from the parent project credentials mapping to OpenRouter
- live OpenRouter validation succeeded with `google/gemini-2.5-flash-lite-preview-09-2025`
- live approval flow succeeded end-to-end on a real `write_file` tool call
- backend now persists sessions, messages, runs, run events, approvals, and clarifications in Mongo collections
- run artifacts now mirror both event JSONL and metadata JSON files under `artifacts/runs/`
- backend exposes `GET/PATCH/DELETE /v1/sessions/{session_id}`, `GET /v1/runs`, and `GET /v1/runs/{run_id}/events`
- approval and clarification backend relays now write durable state before and during replay
- `tests/test_store.py` covers persistence and cascading delete behavior
- `tests/test_backend_api.py` covers backend session CRUD, streaming persistence, approvals, and clarifications
- `scripts/verify_lot2.py` runs the full lot 2 verification set and compile checks
- `scripts/verify_lot2_live_mongo.py` verifies the backend against a real local Mongo daemon while keeping the sidecar in-process
- `frontend/app.py` is now the active Streamlit control-plane shell, with helper modules in `frontend/api_client.py` and `frontend/ui_state.py`
- Streamlit currently supports:
  - session list and session CRUD
  - persisted chat history
  - run timeline
  - approval actions
  - clarification answers
  - workspace file tree and file preview
- RAG currently supports:
  - Session Docs import/list/delete/index
  - Profiles CRUD plus import/list/delete/index
  - Session Memory config, append, compact, clear
  - lookup with transparency statuses and citations
- Matrix currently supports:
  - Python scenario catalog
  - background job launch from the backend
  - Mongo + JSON report persistence
  - Streamlit report browsing and compare view
  - live verification against the real Python stack
- Terminal currently supports:
  - PTY-backed terminal sessions
  - create/write/interrupt/resize/control/close/stream APIs
  - agent-facing interactive terminal tools in the runtime registry
  - blocked-command diagnostics
  - terminal lifecycle events in the run timeline
  - custom Streamlit terminal surface backed by browser-side JS
- JS-like orchestration controls currently support:
  - `policy_profile`
  - module toggles for Web/RAG/Apps/Clarification
  - one-run forced modes via `/rag`, `/web`, `/apps`, `/clarify`
  - timeline filtering (`all`, `errors`, `approvals`, `terminal`, `files`)
- the Streamlit shell now follows a cleaner 3-column structure:
  - left rail for sessions
  - center conversation column
  - right inspector column
- the shell panels now also use dedicated framed containers for header, left panel, chat panel, and inspector panel to reduce the "raw Streamlit page" look
- the fixed bottom control area now includes:
  - centered chat composer
  - adjacent floating tool dock
  - stronger blue primary overrides for the active controls
- timeline and terminal event feeds now render through custom HTML cards with concise summaries and collapsible raw payloads instead of dumping noisy raw blocks inline
- `tests/test_streamlit_app.py` covers the core phase 3 layout through `streamlit.testing.v1`
- `scripts/verify_lot3.py` runs the full project test suite plus compile checks
- `scripts/verify_lot3_live.py` starts real sidecar and backend processes, uses the local Mongo daemon, drives the Streamlit app through `AppTest`, and verifies persisted assistant output
- `src/streamlit_python_only/rag.py` is the new shared RAG implementation used by the sidecar and runtime tool
- `scripts/verify_lot4.py` runs the full test suite and compile checks after RAG changes
- `scripts/verify_lot4_live.py` verifies real import, indexing, lookup, session memory availability, and a live model answer against indexed session docs
- `src/streamlit_python_only/matrix.py` is the new shared Matrix implementation used by the backend

- package migration status:
  - backend, sidecar, tests, and scripts now import `streamlit_python_only.*`
  - `continue_better_py` is now only a residual compatibility shim; the canonical frontend path no longer depends on it
- `scripts/verify_lot5.py` runs deterministic Matrix verification
- `scripts/verify_lot5_live.py` verifies real Matrix job execution, report persistence, and compare on backend + sidecar + Mongo
- `scripts/probe_provider.py` probes the configured provider profile directly
- `scripts/verify_lot6.py` runs the hardened regression and provider metadata checks
- `scripts/verify_lot6_live.py` verifies the real provider, terminal tool path, backend, sidecar, and Mongo together
- `scripts/evaluate_capabilities.py` now runs three live capability tasks against the real model with per-task subprocess timeouts
- latest live capability result on this Mac with OpenRouter/Gemini:
  - pass: `terminal_single_shot`
  - pass: `interactive_terminal`
  - pass: `approval_write_file`
- latest parity pass also fixed a live sidecar/runtime signature mismatch that mocks did not catch; keep real backend+sidecar smokes in the loop when changing runtime factory signatures
- runtime now emits periodic heartbeat diagnostics during long LangGraph turns so SSE stays alive during multi-step tool chains
- backend RAG imports now accept allowed absolute paths inside the project, which fixed the previous live Matrix RAG prep failure on backend relay
- approval policy parity improved:
  - `always_allow`: never ask
  - `always_ask`: always ask
  - `ask_when_necessary`: approval for any non-`safe` tool
- early clarification heuristics now exist for broad product prompts and ambiguous bugfix prompts, which fixed the live `clarification_food_app` Matrix scenario
- latest targeted Matrix live slice on `backend_relay` with Gemini:
  - pass: `clarification_food_app`
  - pass: `read_file_phase_status`
  - hard fail: `web_latest_with_sources`
  - hard fail: `terminal_sequential_inspect`
  - hard fail: `rag_session_docs_multiturn_backend`
- the remaining Matrix failures are now answer-quality/contract issues, not backend plumbing failures
- latest UI cleanup pass kept the full suite green: `39 passed`
- latest UI implementation pass also kept targeted UI/runtime/backend/sidecar checks green: `25 passed`
- latest UI reliability pass focused on live issues found through Chrome DevTools MCP:
  - replaced the inspector panel selector with explicit Streamlit buttons to avoid fragile click targets
  - suppressed the misleading `Session created` notice on the first implicit send
  - added a visible `_Thinking…_` placeholder while chat responses stream
  - set clarification forms to `clear_on_submit=True` and return the inspector to `Run` after resolution
  - latest targeted checks stayed green: `25 passed`
  - latest live browser smoke confirmed:
    - a normal prompt send returns a real assistant reply
    - the `Files` inspector panel opens and shows file preview content
- latest frontend/package alignment pass:
  - `frontend/app.py` is now the canonical Streamlit entrypoint
  - the temporary duplicate frontend shell has been removed
  - the temporary `continue_better_py` shim should no longer be needed by the active Streamlit frontend path
  - assistant streaming and clarification-resume rendering now happen inside the main chat column instead of a detached area below the shell
- latest frontend UX parity follow-up:
  - approval-resume rendering now follows the same pattern and also streams back into the main chat column
  - clarification dialog and clarification inspector now share deduplicated question/option extraction, which reduces repeated prompts in multi-round clarification flows
- latest Matrix precision pass:
  - grading now exposes distinct `infra`, `flow`, `clarificationQuality`, `messageOrder`, and `finalContract` dimensions
  - backend-relay Matrix runs now snapshot persisted session messages so `user first / assistant after` can be checked in reports
  - new scenario coverage was added for:
    - workspace exploration before clarification
    - terminal exact-output
  - existing clarification and multiturn scenarios now also assert message ordering, and clarification scenarios assert option/question quality
  - backend message snapshotting now degrades safely to an empty capture when a test surface/mock does not expose `/v1/sessions/{id}/messages`, so Matrix jobs fail on grading instead of crashing at collection time
- latest evaluation-loop follow-up:
  - `scripts/evaluate_capabilities.py` now covers:
    - basic persisted message ordering
    - terminal exact-output
    - clarification on broad product prompts
    - previous terminal/interactive/approval checks
  - the Streamlit Matrix report view now exposes:
    - per-run failed dimensions
    - a priority failures table
    - a scenario summary with top weak dimensions
  - deterministic/live lot 5 verification now checks that the new Matrix dimensions are really present in generated reports, not just that report generation succeeds
- latest Matrix precision follow-up:
  - Matrix now also exposes dedicated dimensions for:
    - `sourceQuality`
    - `exactOutput`
    - `ragQuality`
  - these are now wired into the scenarios that matter for current parity gaps:
    - `web_latest_with_sources`
    - exact-output terminal/file scenarios
    - `rag_session_docs_multiturn_backend`
  - this makes the remaining live hard fails easier to read as product-quality misses instead of a generic `finalContract` bucket
- latest operator-UX follow-up for Matrix/capability eval:
  - the Streamlit Matrix panel now sorts and surfaces failures by severity first, then by number of failed dimensions
  - it also exposes an overall weakest-dimensions table plus a scenario summary sorted by weakest pass rate
  - `scripts/evaluate_capabilities.py` now provides a top-level triage summary (`failed_task_ids`, `ordering_failures`, `clarification_tasks`) so live runs are easier to scan without opening each task payload
- known limitation:
  - Matrix can validate persisted chat ordering and run/event flow, but it still cannot fully score visual Streamlit details such as modal centering, perceived polish, or animation smoothness; keep a manual UI smoke in the loop for those

## Frontend Canonical Path

- use `frontend/app.py` for real Streamlit runs
- `scripts/run_frontend.sh` and `scripts/run_all.sh` now launch `app.py`
