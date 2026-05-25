---
id: BEAD-0021
title: Workspace lints duplication — worker/shm copy-paste the whole [lints] block to override `unsafe_code`
status: open
priority: medium
created: 2026-05-25
---

## Description

`Cargo.toml` defines a thorough `[workspace.lints.rust]` + `[workspace.lints.clippy]` + `[workspace.lints.rustdoc]` configuration that crates inherit via `[lints] workspace = true`. Two crates need an exception — `rocket-surgeon-worker` (PyO3 + libc + cudaHostRegister) and `rocket-surgeon-shm` (mmap + shared memory) both legitimately need `unsafe_code = "allow"` instead of the workspace's `"forbid"`.

Because Cargo's `[lints]` table is all-or-nothing (you can either inherit via `workspace = true` OR specify your own — you can't inherit *and* override a single key), these two crates each carry a near-verbatim copy of the entire workspace `[lints]` block, differing only in that one line. `perfetto-writer` also has a full copy even though it doesn't need an override (likely historical — its overrides could be deleted in favor of `workspace = true`).

This is drift-prone: every time the workspace lints get tightened or loosened, the three forked crates silently keep the old policy until someone notices and updates each one by hand.

## Proposed fix

Move `unsafe_code` enforcement out of the workspace lints table and into per-crate inner attributes:

1. Drop `unsafe_code = "forbid"` from `[workspace.lints.rust]` in `Cargo.toml`.
2. Add `#![forbid(unsafe_code)]` to the `lib.rs` / `main.rs` of every crate that should forbid unsafe (every crate except worker and shm).
3. Drop the `[lints.*]` blocks in `crates/rocket-surgeon-worker/Cargo.toml`, `crates/rocket-surgeon-shm/Cargo.toml`, `crates/perfetto-writer/Cargo.toml`. Replace each with `[lints]\nworkspace = true`.
4. Worker and shm get nothing — they inherit everything else from workspace, and unsafe is allowed by absence.

Net result: one source of truth for the workspace lint policy, two crate-level inner attributes for the unsafe exception. Adding a new lint to the workspace propagates automatically to all 11 crates instead of needing three manual updates.

## Why not now

Touches all 11 crates' `lib.rs`/`main.rs` (one-line addition each) plus three Cargo.toml refactors. Mechanical but wide; better as its own focused PR than rolled into adjacent toolchain hygiene work.

## Discovered

Audit during the rust-toolchain.toml / Cargo.lock / cargo-deny cleanup session, 2026-05-25.
