# Sub-project D: Conformance + Probe Ordering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate component firing order during forward pass. Fix GPT-2 conformance for lm_head addition. Add Llama conformance test.

**Architecture:** Step one-tick-at-a-time, record (layer, component) pairs, validate monotonic layer ordering and component completeness.

**Tech Stack:** Python pytest, e2e_harness

---

### Task 1: Fix GPT-2 conformance + add firing order test

**Files:**
- Modify: `python/tests/conformance/test_gpt2.py`

Update `test_component_vocabulary` to account for `lm_head` (added in Sub-project B). Add `test_probe_firing_order` that steps through the full forward pass one tick at a time and validates properties.

### Task 2: Add Llama conformance test

**Files:**
- Create: `python/tests/conformance/test_llama.py`

Same structure as GPT-2 but using `hf-internal-testing/tiny-random-LlamaForCausalLM` (already used by e2e_harness). Tests component vocabulary and firing order for Llama architecture.

### Task 3: Full verification + PR

Lint, test, push, create PR.
