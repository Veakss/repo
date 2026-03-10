# Continue Better Python

Parallel rewrite of Continue Better with a Python-only product stack:

- `frontend/`: Streamlit UI with custom components when Streamlit alone is not enough
- `backend/`: FastAPI control plane, persistence, SSE relay, session/run APIs
- `sidecar/`: FastAPI orchestration runtime using LangChain, LangGraph, and LangMem when useful
- `src/`: shared Python package for config, schemas, providers, storage, and runtime helpers
- `config/`: provider catalogs and local environment templates
- `docs/`: roadmap, handoff notes, architecture decisions, and progress logs

This subtree is the new source of truth for the Python migration. The legacy TypeScript implementation remains in the repo as the reference baseline until parity is reached.

Current focus:

1. Lot 1: provider/runtime foundation for OpenRouter and Thales
2. Lot 2: backend persistence and API parity
3. Lot 3: Streamlit UI parity
4. Lot 4: Matrix Lab Python port

See:

- [Roadmap](./docs/ROADMAP.md)
- [Architecture](./docs/ARCHITECTURE.md)
- [Handoff](./docs/HANDOFF.md)
- [Status](./docs/STATUS.md)
