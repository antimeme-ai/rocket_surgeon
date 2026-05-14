---
id: BEAD-0005
title: "ADR: Probe model — DTrace-inspired naming with composable hooks"
status: done
priority: high
created: 2026-05-14
completed: 2026-05-14
---

## Description

Decide on the observation/intervention abstraction. Options: raw PyTorch hooks, DAP breakpoints, DTrace-inspired probes.

## Resolution

ADR-0003 written. Decision: DTrace-inspired probe model with `model:layer:component:event` naming, composable hook registry, wildcard queries, zero-cost-when-off lifecycle. Single abstraction covers observation, checkpointing, intervention, and interpretability analysis.

See `docs/adr/ADR-0003-probe-model.md`.
