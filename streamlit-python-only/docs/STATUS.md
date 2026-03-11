# Status Log

## Snapshot

- Branch target: `codex/streamlit-python-only`
- Remote push: available on `origin`
- Current phase: lot 6 complete

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
- Matrix scenario catalog ported to Python with stable local fixtures and sandbox paths
- Matrix runner added with backend relay, direct runtime, and compatibility surface presets
- Matrix reports now persist to Mongo and JSON mirrors under `artifacts/matrix/`
- backend now exposes Matrix catalog, jobs, reports, and compare endpoints
- Streamlit Matrix tab now supports run launch, job polling, report browsing, filtering, and compare
- lot 5 verification scripts added, including a live Matrix smoke against backend + sidecar + Mongo
- terminal tool added with policy blocking, execution diagnostics, and terminal timeline events
- app actions, web search, and explicit session memory upsert tools added to the Python registry
- Streamlit terminal tab now renders real terminal activity from run events
- provider probe and lot 6 verification scripts added for operational hardening
- run scripts now prefer the local `.venv` interpreter and load `.env`
- PTY-backed terminal manager added with create/write/interrupt/resize/control/close/stream APIs
- runtime terminal tools expanded to cover `open_terminal`, `terminal_write`, `terminal_interrupt`, `terminal_request_control`, `terminal_release_control`, `terminal_snapshot`, `terminal_wait_for_output`, and `terminal_close`
- backend and sidecar now expose local CORS-compatible terminal endpoints for the Streamlit custom terminal component
- Streamlit terminal tab now includes a custom interactive terminal surface instead of a static event-only view
- real capability evaluation script added at `scripts/evaluate_capabilities.py`
- live capability evaluation against OpenRouter/Gemini currently passes 2 out of 3 tasks:
  - `terminal_single_shot`: pass
  - `approval_write_file`: pass
  - `interactive_terminal`: timeout with the real model

## In Progress

- post-parity cleanup and future refinements

## Next Verification Point

- optional future enhancements stay regression-free
- Thales-specific live validation can be rerun when credentials are available on this machine
- interactive terminal live prompting should be improved until `scripts/evaluate_capabilities.py` is fully green

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
- added `src/continue_better_py/matrix_catalog.py` and `src/continue_better_py/matrix.py` for the Python Matrix Lab catalog, runner, grading, aggregation, and persistence
- added backend Matrix routes for catalog, jobs, reports, and compare
- rebuilt the Streamlit Matrix tab to launch jobs, inspect job progress, browse reports, filter results, and compare reports
- added `tests/test_matrix_service.py` and expanded backend/frontend coverage for Matrix flows
- verified `pytest` across the full Python subtree including Matrix coverage: `26 passed`
- verified `python scripts/verify_lot5.py`
- verified `python scripts/verify_lot5_live.py` with real backend, sidecar, Mongo, job execution, report persistence, and compare
- expanded the tool registry with terminal, app actions, web search, and explicit session memory upsert support
- runtime now emits provider diagnostics plus terminal lifecycle events and blocked-command diagnostics
- Streamlit terminal tab now renders real terminal output instead of a placeholder
- added `scripts/probe_provider.py`, `scripts/verify_lot6.py`, and `scripts/verify_lot6_live.py`
- updated the run scripts to use `.venv/bin/python` and auto-load `.env`
- verified `pytest` across the full Python subtree after lot 6 changes: `28 passed`
- verified `python scripts/verify_lot6.py`
- verified `python scripts/verify_lot6_live.py` with real provider probe, backend, sidecar, Mongo, and terminal execution
- added `src/continue_better_py/terminal_manager.py` to port the old PTY terminal contract into Python
- expanded runtime/tool registry support for interactive terminal tools and added backend + sidecar terminal endpoints
- updated the Streamlit terminal tab to embed a custom browser-side terminal surface that talks directly to the backend terminal APIs
- added runtime/API regression coverage for interactive terminal flows
- verified `pytest -q` across the full Python subtree after the terminal parity pass: `31 passed`
- ran `python scripts/evaluate_capabilities.py` live against OpenRouter/Gemini and recorded:
  - pass: single-shot terminal inspection
  - pass: approval-gated file creation
  - fail: interactive terminal scenario timed out under the real model
