# Handoff Notes

## Current Intent

Port the full Continue Better product to a Python-only stack inside `streamlit-python-only/`.

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
- `src/continue_better_py/rag.py` is the new shared RAG implementation used by the sidecar and runtime tool
- `scripts/verify_lot4.py` runs the full test suite and compile checks after RAG changes
- `scripts/verify_lot4_live.py` verifies real import, indexing, lookup, session memory availability, and a live model answer against indexed session docs
- `src/continue_better_py/matrix.py` is the new shared Matrix implementation used by the backend
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
