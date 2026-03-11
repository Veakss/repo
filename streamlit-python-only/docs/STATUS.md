# Status Log

## Snapshot

- Branch target: `codex/streamlit-python-only`
- Remote push: available on `origin`
- Current phase: lot 4 complete, ready for lot 5

## Done

- dedicated Python-only subtree created
- roadmap written
- architecture target written
- handoff notes written
- branch created for the rewrite
- project scaffold created for frontend/backend/sidecar/shared
- local venv created and dependencies installed
- provider unit tests passing
- Python compilation passing
- sidecar runtime starts locally
- backend runtime starts locally
- live OpenRouter model listing works from sidecar and backend
- approval and clarification HTTP flows scaffolded
- risky tool classification introduced for `write_file`
- backend proxy endpoints added for approvals and clarifications
- runtime engine extracted from the sidecar app
- file-backed pending approval/clarification snapshots added
- tool registry abstraction added
- sidecar API tests added
- lot 1 verification script added
- local OpenRouter/Gemini config wired into `streamlit-python-only/.env`
- live Gemini 2.5 Flash Lite smoke passed
- live approval flow on `write_file` passed
- Mongo-backed backend persistence layer expanded for sessions, messages, runs, approvals, and clarifications
- run JSON mirrors now include `.jsonl` events and `.meta.json` metadata per run
- backend now supports session get/rename/delete and run list/detail endpoints
- backend approval and clarification relays now persist state transitions and assistant outputs
- backend API tests added for session CRUD, run persistence, approval persistence, and clarification persistence
- store unit tests added for cascade delete and artifact mirroring
- lot 2 verification script added
- live MongoDB 8.0 installed locally via Homebrew
- live lot 2 backend verification against a real Mongo daemon passed
- Streamlit UI now talks to the Python backend for sessions, messages, runs, approvals, clarifications, and workspace files
- frontend helper modules added for backend access and timeline/UI state shaping
- backend filesystem endpoints added for the Streamlit files panel
- Streamlit test coverage added with `streamlit.testing.v1`
- lot 3 verification scripts added, including a live Streamlit smoke against backend + sidecar + Mongo
- RAG service added with Mongo-backed profiles, imported files, chunks, indexing jobs, and session memory
- sidecar now exposes Python RAG endpoints for profiles, session docs, jobs, memory, and lookup
- backend proxies the RAG contract and appends assistant turns into session memory
- runtime now exposes a `rag_lookup` tool for session-aware retrieval
- Streamlit RAG tab now supports imports, profile CRUD, indexing jobs, memory config, and manual lookup
- lot 4 verification scripts added, including a live retrieval smoke

## In Progress

- Matrix Python port planning
- terminal parity planning

## Next Verification Point

- Matrix reports can be generated, stored, and browsed from the Python stack
- benchmark runs can be launched from Streamlit without the TS matrix runner

## Update Log

### 2026-03-10

- created `streamlit-python-only/`
- documented roadmap, target architecture, and handoff instructions
- created local branch `codex/streamlit-python-only`
- identified missing remote as the current blocker for real `git push`
- added Python project scaffold with FastAPI, Streamlit, LangGraph, Mongo store, and provider config
- installed dependencies in `streamlit-python-only/.venv`
- verified `pytest tests/test_providers.py` passes
- verified `python -m compileall backend sidecar frontend src tests` passes
- verified sidecar `/health` and `/v1/models`
- verified backend `/health`, `/models`, and `/capabilities`
- attempted a real OpenRouter chat call; it reached the provider boundary and failed with `401 Missing Authentication header` because the new subtree does not yet have an API key configured
- added `approval_required`, `approval_decision`, and `clarification_required` event scaffolding
- verified sidecar resume endpoints return clean `404` responses for unknown approval/clarification ids
- verified backend capabilities now expose the clarification module
- extracted the sidecar runtime to `runtime.py`, with dedicated `tool_registry.py` and `run_state.py`
- verified `pytest tests/test_providers.py tests/test_runtime.py tests/test_sidecar_api.py`
- verified actual sidecar SSE failure mode is clean when no OpenRouter key is configured: `error` event then `failed` state then `done`
- mapped the parent project OpenRouter-compatible credentials into `streamlit-python-only/.env`
- verified `scripts/verify_lot1.py` with real OpenRouter auth and live completion
- verified a real `write_file` approval cycle against `google/gemini-2.5-flash-lite-preview-09-2025`
- verified the created file `/Users/victor/Documents/continue-better/agent_playground/lot1_live.txt` contains `ok`
- rebuilt the backend persistence layer around Mongo canonical collections plus run JSON mirrors
- added session CRUD routes and run query routes to the Python backend
- added durable approval and clarification state persistence in Mongo
- verified `pytest tests/test_store.py tests/test_backend_api.py tests/test_providers.py tests/test_runtime.py tests/test_sidecar_api.py`
- verified `python scripts/verify_lot2.py`
- verified `python -m compileall src backend sidecar frontend tests scripts`
- installed `mongodb-community@8.0` with Homebrew and started it via `brew services`
- verified `mongosh --eval 'db.adminCommand({ ping: 1 })'` returns `{ ok: 1 }`
- verified `python scripts/verify_lot2_live_mongo.py` against a real local Mongo daemon
- added Streamlit `frontend/api_client.py` and `frontend/ui_state.py` to separate backend calls from presentation logic
- rebuilt the Streamlit shell to support session selection, rename/delete, persisted history, timeline rendering, approvals, clarifications, and file preview
- added backend filesystem routes `GET /v1/fs/tree` and `GET /v1/fs/read`
- verified `pytest` across the full Python subtree including Streamlit tests: `20 passed`
- verified `python scripts/verify_lot3.py`
- verified `python scripts/verify_lot3_live.py` with a real backend, sidecar, Mongo daemon, and OpenRouter-backed response
- added `src/continue_better_py/rag.py` for Mongo-backed RAG storage, indexing, lookup, and session memory
- added backend and sidecar RAG routes for profiles, file import/list/delete, index jobs, memory, and lookup
- added the `rag_lookup` runtime tool and exposed the RAG control surface in Streamlit
- verified `pytest` across the full Python subtree including RAG coverage: `23 passed`
- verified `python scripts/verify_lot4.py`
- verified `python scripts/verify_lot4_live.py` with real import, indexing, lookup, memory, backend, sidecar, Mongo, and a real model response of `AURORA_PHASE4`
