# CHARLIE — shm + transport properties — findings

**Lane:** `crates/rocket-surgeon-shm`, `crates/rocket-surgeon-transport`.
**Branch:** `platoon/test-shm-transport`. Authored by worker CHARLIE; clippy
cleanup (loom cfg registration + lint fixes), findings doc, and commit finalized
by the commander after CHARLIE's session ended green-on-test but not
clippy-clean.

## What was built

New test files (all hand-written strategies; production code untouched):
- `rocket-surgeon-shm/tests/prop_ring.rs` — DOOMRING SPSC ring properties.
- `rocket-surgeon-shm/tests/stateful_ring.rs` — tier-6 model-based test driving
  the real POSIX-shm ring in lockstep with a `VecDeque<Vec<u8>>` model.
- `rocket-surgeon-shm/tests/prop_config_frame.rs` — `RingConfig` validation
  (tier-6 model) + `serialize_probe_frame` byte-offset encoder (tier-7 spec).
- `rocket-surgeon-shm/tests/loom_ring.rs` — `#![cfg(loom)]` exhaustive
  interleaving test of the cursor protocol (run via `RUSTFLAGS="--cfg loom"`).
- `rocket-surgeon-transport/tests/prop_framing.rs` — Content-Length framing
  roundtrip / FIFO / exception-raising.

Oracle tiers climbed:
- **Tier 7 spec** — every field of `serialize_probe_frame` round-trips from its
  documented little-endian byte offset. The offsets *are* the cross-language
  (Rust↔Python) wire spec, so this is the strongest oracle available.
- **Tier 6 model** — ring vs `VecDeque`; `RingConfig::new` vs an independent
  reference predicate over the full `u64` range (overflow path).
- **Tier 4 roundtrip/metamorphic** — bytes published == consumed across the full
  boundary set (empty .. slot capacity); framing identity even when bodies
  *contain* `\r\n\r\n` / `Content-Length:`; header is the single source of truth
  for consumed length; Content-Length case-insensitivity; extra-header invariance.
- **Exception-raising** — oversized payloads, stale generations, corrupt header
  size, `message_too_large`, truncated body, missing Content-Length each produce
  the specific typed error, never a panic or silent truncation.
- Generators are **measured** (`*_generator_distribution`, boundary-bucket counts).

## loom

The SPSC ordering protocol is re-expressed over `loom`'s model atomics (loom
cannot instrument the raw `mmap`'d region directly). loom explores every legal
interleaving / memory-ordering outcome and asserts FIFO with no stale/torn slot
read across wrap-around. Wired as a `cfg(loom)`-gated dev-dependency with a
`build.rs` `check-cfg` registration so default `cargo test` / clippy stay clean;
run it with `RUSTFLAGS="--cfg loom" cargo test -p rocket-surgeon-shm --test loom_ring`.

## Findings / weak oracles

- No production bug surfaced in this lane — the ring's `Release`/`Acquire`
  discipline holds under loom, and framing roundtrips cleanly. The value here is
  the *coverage*: the byte-offset spec oracle and the loom interleaving proof did
  not exist before and pin behaviour the example tests never touched.
- The proptest config-validation property found and is regression-pinned on a
  capacity boundary (`RingConfig { backuptics: 1, slot_size: 128 }, excess = 1`);
  green after the geometry was exercised exhaustively. Saved in
  `prop_ring.proptest-regressions`.

## Gate status

`cargo test -p rocket-surgeon-shm -p rocket-surgeon-transport` → all green.
`cargo clippy -p rocket-surgeon-shm -p rocket-surgeon-transport --all-targets
-- -D warnings` → clean. loom test compiles under `--cfg loom`.
