# Status Log

## Snapshot

- Branch target: `codex/streamlit-python-only`
- Remote push: available on `origin`
- Current phase: lot 2 complete, ready for lot 3

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

## In Progress

- Streamlit shell expansion
- UI parity implementation

## Next Verification Point

- Streamlit can drive a real backend session end to end
- UI session and run panels reflect persisted backend state

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
- confirmed no local Mongo process is available on this machine, so live backend smoke against a real Mongo daemon could not be run here
