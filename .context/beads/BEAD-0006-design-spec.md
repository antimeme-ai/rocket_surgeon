---
id: BEAD-0006
title: Comprehensive design specification
status: done
created: 2026-05-14
---

## Summary

Wrote the comprehensive design specification at `docs/specs/design.md` (1,779 lines). Supersedes the
original `architecture.md`. Incorporates sky claude's architectural consultation, 6 deep research
agent studies (NNsight internals, protocol patterns, multi-GPU/MoE, visualization/TUI, vLLM-Lens,
tokio-console), and the full 101-paper + 60-repo research base.

Key decisions codified:
- Three-process architecture (Rust daemon ↔ Python model host(s) ↔ Rust TUI)
- 10 protocol verbs over JSON-RPC 2.0 (lifecycle + rocket/* namespace)
- Five-level probe namespace with rank dimension (model:rank:layer:component:event)
- Three-tier hook strategy (eager / Dynamo-FX / CUDA-graph)
- Content-addressable tensors (BLAKE3), summary-then-slice protocol
- Shared-memory ring buffer for tensor handoff
- √N activation checkpoints, ULP-close replay (not bit-exact)
- Pre/post-collective barriers (never inside NCCL collectives)
- MoE four tick granularities designed into schema from day one
- Perfetto protobuf trace format
- Session bundles for reproducibility
- Reimplementation of all core abstractions (model adapter, intervention engine, hook manager)
  informed by but not dependent on NNsight, pyvene, baukit, nnterp, vLLM-Lens

Also cloned 13 additional repos identified by sky claude (penzai, treescope, vllm-lens, vllm-hook,
tokio-console, goodfire-sdk, maia, MixtureKit, gpt-neox, Megatron-LM, text-generation-inference,
Liger-Kernel, mamba). EasySteer was not found (likely private/removed).

## Next

Per JSMNTL: TCK specs (Phase 0), then red tests, then implementation.
