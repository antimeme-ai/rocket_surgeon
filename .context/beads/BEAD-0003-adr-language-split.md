---
id: BEAD-0003
title: "ADR: Language split — Rust + Python via PyO3"
status: done
priority: high
created: 2026-05-14
completed: 2026-05-14
---

## Description

Decide on the language split for rocket_surgeon. Options: pure Python, pure Rust, Rust + Python.

## Resolution

ADR-0001 written. Decision: Rust core (state machine, protocol server, TUI, checkpoint storage) + Python hook layer (PyTorch hooks, model loading, SAE integration), bridged via PyO3. Python process is host, Rust compiled as extension module. TUI runs as separate Rust process over protocol.

See `docs/adr/ADR-0001-language-split.md`.
