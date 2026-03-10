# Target Architecture

## Services

### Frontend

- Technology: Streamlit plus custom components
- Role:
  - render the product UI
  - open SSE streams to backend
  - show chat, timeline, sessions, RAG, files, Matrix, approvals, and terminal panels

### Backend

- Technology: FastAPI
- Role:
  - product-facing API
  - session and run persistence
  - SSE relay to the sidecar
  - JSON export generation
  - filesystem sandbox access
  - Matrix report APIs

### Sidecar

- Technology: FastAPI + LangChain + LangGraph
- Role:
  - LLM orchestration
  - tool execution
  - approvals and clarification state
  - provider-specific compatibility handling
  - RAG orchestration

## Shared Package

`src/continue_better_py/` will centralize:

- settings
- schemas
- event types
- provider definitions
- repository interfaces
- storage helpers
- common utility code

## Storage

### MongoDB Collections

- `sessions`
- `messages`
- `runs`
- `run_events`
- `rag_profiles`
- `rag_files`
- `rag_chunks`
- `rag_memory_entries`
- `matrix_reports`
- `matrix_runs`

### JSON Mirror Artifacts

- `artifacts/runs/`
- `artifacts/matrix/`
- `artifacts/debug/`

## Provider Layer

### OpenRouter

- OpenAI-compatible transport
- native tool calling when model supports it
- preferred development provider on this machine

### Thales

- `.env` plus YAML configuration
- tool-compatible first turn
- textual replay compatibility for follow-up turns
- optional probing and recovery helpers

## Orchestration Model

LangGraph state machine with explicit phases:

- planning
- execute
- verify
- repair
- finish

Run states preserved from the legacy app:

- running
- awaiting_approval
- awaiting_clarification
- completed
- failed
- stopped

## UI Fidelity Strategy

Approximate the existing UI using:

- Streamlit layout primitives for core structure
- injected CSS for transparency-heavy styling
- custom components for:
  - timeline density
  - terminal
  - richer tabular Matrix views
  - advanced panel interactions

## Constraints

- mono-user local app
- maintainable code over framework cleverness
- preserve operational transparency
- keep provider-specific logic isolated
