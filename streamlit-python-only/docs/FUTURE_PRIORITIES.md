# Future Priorities

## Purpose

This document captures the recommended next improvement phase for the Python rewrite after the current structural migration.

The goal is not to reopen the architecture. The goal is to harden the existing base, improve product behavior, and prepare the stack for more realistic Windows/Thales usage.

## Executive Summary

The project is now on top of a credible and manageable architecture:

- one canonical Streamlit frontend
- one FastAPI backend acting as the product-facing control plane
- one FastAPI sidecar carrying the orchestration runtime
- one shared Python package for runtime, tools, providers, RAG, Matrix, storage, and schemas
- one Mongo-backed persistence model with JSON artifacts for local traceability

The next phase should focus on reliability, operator usability, Windows/Thales readiness, and product-quality behavior rather than broad rewrites.

The runtime/provider hardening phase has now landed:

- final-answer contract detection and repair exists in the Python runtime
- provider capabilities are explicit and exposed through the sidecar
- sequential single-tool handling now exists for Thales/textual replay style providers
- enterprise Thales configuration now supports `tools/config.yaml` / `config.yaml` and AMA/seamlessdiag-style `llm.default` + `llm.models`
- targeted Matrix progress improved on:
  - `web_latest_with_sources`
  - `terminal_sequential_inspect`
  - `terminal_exact_output`

The highest-value remaining work is no longer broad runtime plumbing. It is now:

- multi-turn RAG answer quality on backend relay
- live capability stability on real providers
- provider-aware behavior discipline, especially for Thales follow-up turns

## Global Priorities

### Priority 1

- runtime quality for multi-turn grounded answers
- provider-aware behavior stability in live runs
- RAG answer quality on backend relay
- provider and environment validation for real-world usage

### Priority 2

- frontend UX reliability and visible correctness
- Windows/Thales operational readiness
- terminal robustness across workflows
- RAG industrialization beyond the current lexical retrieval path
- better operator-facing diagnostics
- Thales enterprise-mode hardening after the first implementation pass:
  - startup validation against enterprise SSL/proxy/path constraints
  - runbook and troubleshooting guidance for `config.yaml`
  - live validation against a real Windows/Thales endpoint
  - Matrix/capability coverage for enterprise sequential-tool behavior

### Priority 3

- codebase modularization where files are now too large
- visual polish and non-blocking UX refinement
- additional Matrix breadth once core weak points are addressed

## Domain Review

### Frontend

Current role:

- render sessions, chat, approvals, clarifications, files, RAG, terminal, and Matrix
- reflect backend/runtime state clearly
- stay Streamlit-native enough to remain maintainable

Essential now:

- fix visible UX defects first
- make session creation and session switching feel deterministic
- remove noisy or misleading notices
- keep `user first / assistant after` behavior stable
- ensure approval and clarification resumes always land back in the central chat flow

Important but secondary:

- split the current `frontend/app.py` into smaller UI modules over time
- reduce CSS overrides that fight Streamlit unnecessarily
- improve inspector readability and information density

Lower priority:

- visual polish
- animation or styling refinements
- non-functional layout experiments

### Backend

Current role:

- product-facing API
- session/run persistence
- SSE relay to the sidecar
- filesystem, terminal, RAG, and Matrix access surface

Essential now:

- keep contracts stable
- improve API-side error clarity and diagnostics
- reduce hard-coded local assumptions where configuration should drive behavior
- make environment validation easier to understand at startup

Important but secondary:

- tighten route-level consistency for payloads and failure cases
- add clearer operator diagnostics around replay, persistence, and relay failures

Lower priority:

- internal cosmetic refactors
- performance tuning before real bottlenecks are observed

### Matrix

Current role:

- measure product behavior quality
- separate infra failures from answer-quality failures
- provide a repeatable evaluation loop for the Python stack

Essential now:

- keep the current dimensions meaningful and stable
- keep `web_latest_with_sources`, `terminal_sequential_inspect`, and `terminal_exact_output` green after the runtime hardening pass
- improve the main remaining weak live scenario:
  - `rag_session_docs_multiturn_backend`
- improve live capability stability on:
  - `interactive_terminal`
  - `message_order_basic`
  - clarification-heavy scenarios
- keep reports directly actionable for engineers and operators

Important but secondary:

- add more Windows/Thales-oriented scenarios
- add more scenarios around exact-output and multi-step tool chains

Lower priority:

- additional dashboard polish
- expanding scenario breadth before core weak spots are stabilized

### Runtime

Current role:

- orchestrate the model/tool loop with LangGraph
- manage approvals and clarifications
- normalize tool behavior and final response flow

Essential now:

- harden the new final-answer contract against follow-up turns and provider drift
- improve multi-turn grounded synthesis after terminal and RAG tool use
- reduce cases where the correct tool is chosen but the final answer still loses grounding, sources, or ordering
- avoid unnecessary clarifications
- keep one-pass repair logic predictable and observable

Important but secondary:

- split dense logic inside `runtime.py`
- separate heuristics, replay logic, and final-answer shaping more clearly
- refine provider-aware sequential execution for Thales:
  - keep at most one tool call per assistant turn
  - improve the follow-up prompt after a collapsed multi-tool response
  - strengthen provider-specific prompting so Thales stays on a sequential tool loop
  - make final-answer repair less heuristic for RAG-heavy follow-up turns

Lower priority:

- architectural redesign
- graph elegance improvements that do not move product behavior

### Sidecar

Current role:

- expose runtime capabilities as an API surface
- host terminal, RAG, and orchestration endpoints

Essential now:

- keep sidecar contracts stable
- improve diagnostics and predictability of long-running multi-step runs
- avoid responsibility drift from frontend concerns into the sidecar

Important but secondary:

- refine internal organization of route groups
- document the intended scope of each sidecar surface more explicitly

Lower priority:

- major reshaping of sidecar structure while contracts are still being used actively

### Providers

Current role:

- resolve OpenRouter and Thales behavior
- select native vs textual replay mode
- isolate provider-specific compatibility logic

Essential now:

- validate Thales flows against a real endpoint
- document provider-specific expectations and failure modes
- improve handling guidance for SSL, proxy, and enterprise environment constraints
- keep the new provider capabilities model authoritative and visible, including whether a provider supports:
  - native tool calls
  - textual replay
  - multi-tool calls in a single turn
  - enterprise YAML-driven configuration
- encode more provider-specific behavior in the provider/runtime layer instead of in prompts alone

Important but secondary:

- strengthen startup-time configuration validation
- improve provider probe coverage and reporting
- make startup diagnostics report the effective config source:
  - local `.env`
  - provider catalog
  - enterprise `config.yaml`
- add explicit live validation notes for:
  - OpenRouter dev mode on macOS
  - Thales enterprise mode on Windows
- add operator-facing examples of supported enterprise YAML structure:
  - `llm.default.provider`
  - `llm.default.name`
  - `llm.models[]`

Lower priority:

- adding more providers before the main two are fully hardened

### Documentation

Current role:

- capture architecture, status, handoff, and roadmap
- support project continuity through migration work

Essential now:

- add clearer Windows/Thales operational documentation
- document known limitations explicitly
- document the recommended verification loop for:
  - frontend
  - tools
  - terminal
  - RAG
  - Matrix
- document the current Thales-specific operating assumptions explicitly:
  - Windows enterprise machines use `config.yaml`
  - Thales API follow-up turns rely on textual replay
  - Thales should be treated as sequential single-tool execution, not multi-tool native orchestration

Important but secondary:

- produce a concise onboarding document for a new engineer joining the project
- produce a runbook for debugging chat/tool/runtime issues

Lower priority:

- historical narrative or non-operational project storytelling

## Recommended Sequencing

1. frontend reliability issues that hurt confidence immediately
2. runtime final-answer discipline and exact-output behavior
3. provider and environment hardening for real operational contexts
4. Thales-specific enterprise validation and ops hardening:
   - startup validation and diagnostics
   - real-endpoint validation
   - operator runbook for `config.yaml`
5. Windows/Thales run scripts and operational readiness
6. terminal robustness across real workflows
7. RAG industrialization
8. code modularization and non-blocking polish

## Practical Interpretation

What is most urgent is not adding features.

What is most urgent is making the current stack:

- more predictable
- easier to operate
- more accurate in its final answers
- less noisy in its UI behavior
- more credible on Windows/Thales constraints

That should be the guiding principle for the next evolution phase.
