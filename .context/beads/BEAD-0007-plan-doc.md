---
id: BEAD-0007
title: Implementation plan document
status: done
created: 2026-05-14
---

## Summary

Wrote the implementation plan at `docs/specs/plan.md`. Derived from the design doc. Breaks each
phase into numbered work units with dependencies, file deliverables, acceptance criteria, and
TCK targets.

Phases 0–2 (MVP) are planned in full detail:
- Phase 0: 6 work units (probe grammar, JSON-RPC schema, canonical vocabulary, capability spec, Gherkin TCK, ADRs)
- Phase 1: 12 work units (daemon skeleton, protocol types, probe registry, tensor store, model host, adapter, hooks, shm, step+inspect, built-in views, subscribe, e2e smoke test)
- Phase 2: 6 work units (intervention engine, intervene verb, session bundles, MCP adapter, IOI acceptance test, overhead benchmark)

Phases 3–7 planned at task level (10–11 tasks each), to be detailed when execution reaches them.

Cross-cutting: CI strategy, testing strategy (unit/TCK/acceptance), documentation, dependency management, git workflow.

## Next

Phase 0 execution begins. First step of each execution cycle is TCK specs.
