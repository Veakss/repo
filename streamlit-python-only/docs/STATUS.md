# Status Log

## Snapshot

- Branch target: `codex/streamlit-python-only`
- Remote push: blocked, no remote configured
- Current phase: bootstrap and migration planning

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

## In Progress

- Mongo handling hardening
- runtime chat path beyond the auth boundary
- Streamlit shell expansion

## Next Verification Point

- project files exist and import cleanly
- sidecar starts and exposes `health`

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
