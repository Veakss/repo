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
   - `pyproject.toml`
   - `.env.example`
   - `docker-compose.yml`
   - `config/providers.example.yaml`
2. Scaffold shared package and settings
3. Implement sidecar provider adapters and LangGraph entrypoint
4. Add backend proxy and minimal Mongo persistence
5. Add Streamlit shell wired to backend

## Practical Constraints

- No Git remote is configured right now
- Local commits can be made
- `git push` is blocked until a remote is added

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
