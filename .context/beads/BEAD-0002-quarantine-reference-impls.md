---
id: BEAD-0002
title: Clone reference implementations to quarantine
status: open
priority: medium
created: 2026-05-14
---

## Description

Clone and study reference implementations identified in lit review. These go in quarantine/ (gitignored).

## Repos to quarantine
- nnsight (ndif-team/nnsight) — primary interception approach reference
- TransformerLens (TransformerLensOrg/TransformerLens) — HookPoint abstraction, naming conventions
- pyvene (stanfordnlp/pyvene) — intervention-as-data pattern
- CircuitsVis (TransformerLensOrg/CircuitsVis) — visualization components
- nnterp (Butanium/nnterp) — standardized naming layer
- OpenAI Transformer Debugger (openai/transformer-debugger) — closest prior art
- Anthropic circuit-tracer (decoderesearch/circuit-tracer) — attribution graphs

## Context

Per JSMNTL: reference implementations studied before building. These repos inform architecture decisions.

## Acceptance

- All repos cloned to quarantine/
- Quick notes on what to study in each
