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

1. Add root project files:
   - move pending control-flow persistence from local files into the canonical backend persistence layer
   - add richer run diagnostics and replay support
2. Add backend run/session coverage:
   - better run metadata
   - graceful degraded mode when Mongo is unavailable
3. Add Streamlit feature panels:
   - sessions
   - files
   - RAG
   - approvals
   - Matrix
4. Add real OpenRouter smoke once an API key is present in the new subtree `.env`
5. Port Matrix runner and dashboard

## Practical Constraints

- No Git remote is configured right now
- Local commits can be made
- `git push` is blocked until a remote is added
- Docker is not installed on this machine right now, so the local Mongo bootstrap must not assume Docker

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
