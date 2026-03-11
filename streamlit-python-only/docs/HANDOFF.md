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

1. Start lot 6:
   - terminal hardening
   - replay diagnostics
   - packaging and polish
   - provider-specific closure checks for OpenRouter and Thales
2. Keep lot 5 stable while lot 6 lands:
   - preserve the Matrix catalog and report schema
   - keep the Streamlit Matrix tab wired to the real backend endpoints
   - avoid breaking Mongo + JSON report persistence
3. Keep using the live verification scripts when changing contracts:
   - `verify_lot2_live_mongo.py`
   - `verify_lot3_live.py`
   - `verify_lot4_live.py`
   - `verify_lot5_live.py`

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
- `tests/test_streamlit_app.py` covers the core phase 3 layout through `streamlit.testing.v1`
- `scripts/verify_lot3.py` runs the full project test suite plus compile checks
- `scripts/verify_lot3_live.py` starts real sidecar and backend processes, uses the local Mongo daemon, drives the Streamlit app through `AppTest`, and verifies persisted assistant output
- `src/continue_better_py/rag.py` is the new shared RAG implementation used by the sidecar and runtime tool
- `scripts/verify_lot4.py` runs the full test suite and compile checks after RAG changes
- `scripts/verify_lot4_live.py` verifies real import, indexing, lookup, session memory availability, and a live model answer against indexed session docs
- `src/continue_better_py/matrix.py` is the new shared Matrix implementation used by the backend
- `scripts/verify_lot5.py` runs deterministic Matrix verification
- `scripts/verify_lot5_live.py` verifies real Matrix job execution, report persistence, and compare on backend + sidecar + Mongo
