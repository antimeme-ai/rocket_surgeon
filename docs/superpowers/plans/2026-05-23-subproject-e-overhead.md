# Sub-project E: Overhead Benchmarking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure intervention overhead and assert <= 2% regression vs baseline stepping.

**Architecture:** Python E2E benchmark: spawn daemon, attach model, step N times baseline, step N times with interventions, compare wall times.

**Tech Stack:** Python, e2e_harness, time module

---

### Task 1: Write E2E intervention overhead benchmark

**Files:**
- Create: `tests/bench_intervention_overhead.py`

### Task 2: Full verification + PR
