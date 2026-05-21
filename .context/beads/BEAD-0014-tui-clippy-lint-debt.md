---
id: BEAD-0014
title: rocket-surgeon-tui carries ~88 clippy errors — blocks the local commit gate
status: closed
priority: medium
created: 2026-05-21
---

## Description

`crates/rocket-surgeon-tui` fails `cargo clippy --all-targets -- -D warnings`
with 88 errors. The crate is a scaffold: its modules exist
(`client/connection`, `client/subscription`, `input`, `render/compositor`,
`state/reducer`, `state/cache`, `tiling`) but are not wired together, so most
of the surface is unreachable.

The TUI skeleton was merged via a web-UI PR, which bypasses the lefthook
pre-commit gate (see the `master-goes-red-gate-bypass` note). Consequence:
`master` is red on clippy, and because the lefthook pre-commit hook runs
clippy **workspace-wide with `-D warnings`**, this debt now blocks *every*
local commit — including unrelated, clean changes in other crates.

The daemon/protocol crates (`rocket-surgeon`, `rocket-surgeon-protocol`, etc.)
are clippy-clean; the debt is entirely in `rocket-surgeon-tui`.

## Findings (88 errors, by class)

- **Dead code (~50)** — entire client layer never constructed (`Connection`,
  `ReconnectingClient`, `OutgoingMessage`, `ClientError`, `PendingMap`,
  `ConnectFn`, `lock_pending`, `read_loop`, `write_loop`,
  `read_content_length_message`, `MAX_MESSAGE_SIZE`, `MAX_HEADER_COUNT`,
  `MAX_HEADER_LINE_LEN`); `SubscriptionState`, `TensorCache`, `CacheKey`;
  `tiling` helpers (`hsplit`, `adjust_ratio`, `propose_layout`, …); and many
  never-constructed enum variants across `input`, `render`, `state`.
- **`use_self` (26)** — "unnecessary structure name repetition".
- **Misc style (~12)** — `derivable_impls`, `unnested_or_patterns`,
  `needless_pass_by_value`, `items_after_statements`, `large_enum_variant`,
  `cast_lossless`, `uninlined_format_args`, `match_single_binding`,
  `manual_let_else`, `significant_drop_in_scrutinee`, etc.

## Resolution

Kept the TUI scaffold and added module- or item-scoped `dead_code` allowances
to the unwired modules/items that the next daemon-connected TUI slice will use. Fixed the
mechanical clippy findings in the reachable code and tests. Verified with
`cargo clippy -p rocket-surgeon-tui --all-targets -- -D warnings`,
`cargo test -p rocket-surgeon-tui`, workspace clippy/tests, and
`cargo xtask ci`.
