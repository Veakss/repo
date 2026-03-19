# Python Rewrite Roadmap

## Objective

Reach functional parity with the current Continue Better baseline in a Python-only implementation branded as AI Technical Assistant:

- UI in Streamlit, with custom embedded components where needed
- Runtime/orchestration only through LangChain, LangGraph, and LangMem when appropriate
- MongoDB for persistence
- Support for OpenRouter and Thales providers
- Matrix Lab rewritten in Python

The target is not a simplified clone. The target is the same product, ported and maintainable.

## Non-Negotiables

- Keep the same product surface:
  - sessions
  - streaming chat
  - runs and replay
  - approvals
  - clarification
  - file explorer and preview
  - terminal support
  - RAG with Session Docs, Profiles, and Session Memory
  - Matrix Lab and report browsing
- Mono-user local deployment
- Frontend, backend, and sidecar remain separate services
- Provider strategy:
  - OpenRouter: native tool calling when supported
  - Thales: provider-compatible textual replay mode
  - Runtime mode can be `auto`, `native`, or `textual_replay`

## Working Rules

- New implementation lives under `streamlit-python-only/`
- Legacy app remains untouched unless needed for reference
- Every meaningful milestone must update:
  - `docs/STATUS.md`
  - `docs/HANDOFF.md`
- Future-priority planning should also stay aligned with:
  - `docs/FUTURE_PRIORITIES.md`
- Commit locally at the end of each stable milestone
- Push whenever a remote is available

## Milestones

### Lot 1: Runtime Foundation

Goal:

- establish Python sidecar parity foundations
- lock provider compatibility strategy
- build the event model and orchestration skeleton

Deliverables:

- provider config system: `.env` + YAML model catalog
- provider abstraction for OpenRouter and Thales
- LangGraph run loop with explicit state transitions
- tool registry abstraction
- SSE stream event contract matching the legacy app
- tests for provider mode selection and Thales compatibility helpers

Exit criteria:

- sidecar `health`, `capabilities`, `models`, and `chat/stream` work locally
- OpenRouter path runs with the configured model
- Thales mode can be configured and routed through textual replay logic

### Lot 2: Backend and Persistence

Goal:

- rebuild the control plane and persistence layer in Python

Deliverables:

- FastAPI backend with API shape close to the legacy backend
- Mongo repositories for sessions, messages, runs, events, RAG metadata, and Matrix reports
- JSON export mirrors for runs and Matrix outputs
- SSE relay between backend and sidecar
- approval and clarification persistence/recovery model

Exit criteria:

- sessions can be created, listed, renamed, deleted
- chat runs stream through backend
- run events are persisted in Mongo and exported to JSON

### Lot 3: Streamlit UI Parity

Goal:

- rebuild the UI as closely as possible using Streamlit and custom components

Deliverables:

- chat window
- composer and tool toggles
- sessions menu
- run header and timeline
- file explorer and preview
- RAG panel
- terminal panel
- approval and clarification UX

Exit criteria:

- a real run can be initiated from Streamlit and monitored end to end
- the UI remains visually and behaviorally close to the current app

### Lot 4: RAG Parity

Goal:

- rebuild all three RAG layers and operational flows

Deliverables:

- Session Docs
- Profiles
- Session Memory
- indexing job queue
- citations and retrieval formatting
- transparency states: disabled, indexing, no hits, error

Exit criteria:

- indexed session docs are retrievable from real runs
- memory persists and can be compacted/cleared

### Lot 5: Matrix Lab Python Port

Goal:

- reproduce the current benchmark/reporting subsystem in Python

Deliverables:

- Python scenario catalog
- Python runner
- Mongo + JSON report persistence
- Streamlit dashboard equivalent to `/matrix`
- report compare view
- direct run from UI

Exit criteria:

- a matrix run can be launched and inspected without the legacy TS stack

### Lot 6: Hardening and Parity Closure

Goal:

- close the remaining parity gaps and make the Python stack the default

Deliverables:

- regression checklist
- provider-specific probes
- runtime diagnostics
- replay reliability fixes
- packaging and run scripts

Exit criteria:

- Python stack can replace the legacy stack for daily use

## Prioritized Build Order

1. Sidecar provider/runtime core
2. Backend API and persistence
3. Streamlit shell and streaming integration
4. RAG implementation
5. Matrix Lab port
6. Terminal hardening and UI refinement

## Cross-Cutting Decisions

### Provider Compatibility

- Thales is first-class, not a fallback afterthought
- Thales-specific compatibility lives in provider adapters, not scattered across tools
- OpenRouter should be used during development whenever live provider testing is possible on this machine

### Persistence

- Mongo is canonical
- JSON exports are secondary operational artifacts for portability and debugging

### UI Strategy

- Streamlit first
- custom components only where parity or interaction density requires them

### Tooling Strategy

- tools must be modular and independently testable
- approval-sensitive tools must declare risk metadata
- all runtime orchestration remains graph-driven

## Definition of Done for Any Step

- code added
- docs updated
- at least one local verification run completed when feasible
- next step captured in `docs/HANDOFF.md`
