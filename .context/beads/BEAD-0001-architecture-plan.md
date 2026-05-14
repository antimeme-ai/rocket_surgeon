---
id: BEAD-0001
title: Write architecture plan from lit review synthesis
status: done
priority: high
created: 2026-05-14
completed: 2026-05-14
---

## Description

Lit review sweep complete (13 reviews across 13 domains). Synthesize into a written architecture plan before any implementation begins.

## Context

Three-layer architecture emerged from research: Core Engine, Machine Interface, TUI. Multiple design decisions pending (language split, protocol design, hook strategy, MoE approach, SAE integration).

## Resolution

Architecture plan written to `docs/specs/architecture.md`. Synthesizes all 13 lit reviews into a three-layer design (Core Engine in Rust, Machine Interface via JSON-RPC 2.0, TUI via Ratatui) with DTrace-inspired probe model, 7 composable protocol primitives, and checkpoint-based replay for reverse stepping. ADRs for load-bearing decisions follow as separate beads.

## Acceptance

- [x] Written plan in docs/specs/
- [ ] ADRs for load-bearing decisions (BEAD-0003, BEAD-0004, BEAD-0005)
- [ ] Reviewed and approved before any code
