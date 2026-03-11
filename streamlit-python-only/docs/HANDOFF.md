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

1. Start lot 3:
   - wire Streamlit to backend session CRUD and chat streaming
   - add a run timeline panel backed by `/v1/runs` and `/v1/runs/{run_id}/events`
   - expose approval and clarification actions from the UI
2. Preserve backend contract while growing the UI:
   - keep session list, messages, and run details sourced from Mongo-backed endpoints
   - reuse the new run metadata counters in the timeline and status pills
3. Add file, RAG, and Matrix panels progressively on top of the current Streamlit shell
4. When a Mongo daemon is available locally, run a live backend smoke using the real backend plus sidecar stack
5. Port Matrix runner and dashboard after the Streamlit app can drive normal runs end to end

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
