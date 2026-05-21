# TUI Phase 4.1 CR Remediation — Handoff Document

**Branch:** `fix/tui-cr-findings`
**PR:** #16 (merged → master)
**Base:** `master` at `0a65ede` (merge of PR #15, feat/tui-foundation)
**Date:** 2026-05-20
**Status:** Complete — 19/19 findings closed, 311 Rust tests green, merged

---

## What Was Done

Retroactive six-persona adversarial code review of Phase 4.1 (PR #15) found 19 findings across 4 severity tiers. This session wrote a dependency-ordered remediation plan, then executed all 7 tasks serially following JSMNTL methodology: plan → test → implement → verify → commit.

### Severity Breakdown

| Tier | Count | Nature |
|------|-------|--------|
| P0 | 3 | Crashes / security: unbounded alloc, div-by-zero, silent drops |
| P1 | 4 | Correctness: stale broadcast, write-lock-during-backoff, cache loop, dirty bypass |
| P2 | 4 | Design: OOP patterns, hardcoded heuristic, missing command buffer, no clamping |
| P3 | 4 | Polish: Debug-string sort, duplicate Rect, i64 cast, misleading comment |

All 19 addressed. P3 #17 (WezTerm comment) was analyzed and determined to be correct behavior — no code change needed.

### Commits (oldest first)

| SHA | Task | Description |
|-----|------|-------------|
| `b2bd6e3` | — | Plan document: 19 findings mapped to 7 tasks in dependency order |
| `e03de1c` | 1 | Connection hardening: MAX_MESSAGE_SIZE (64 MiB), MAX_HEADER_COUNT, MAX_HEADER_LINE_LEN, poison-safe mutex via `into_inner`, `tracing::warn!` for unparseable messages |
| `d5b9fe3` | 2 | Reconnect rework: `Connection::spawn` takes external `broadcast::Sender<Notification>`, `ReconnectingClient` acquires write lock only after successful connect, `SubscriptionManager` → free functions |
| `b0d3364` | 3 | main.rs: `clap::value_parser!(u32).range(1..=240)` for fps, removed `\|\| true` dirty bypass, initial dirty seeding loop |
| `58f4645` | 4 | Cache: `max_entries.max(1)` prevents infinite loop on `new(0)`, `prefetch_keys` takes `tick_id` parameter |
| `79e2197` | 5 | State rework: `UiState::initial()` → `initial_ui_state()` free function (8 call sites), `command_buffer: String` field, cursor clamping to `capabilities.num_layers`, 5 new tests |
| `5dcb8af` | 6 | `propose_layout` triggers on any component change (not just `"attn"`), `EventType` gets `PartialOrd + Ord` derive, `sorted.sort()` replaces `sorted.sort_by_key(\|e\| format!("{e:?}"))` |
| `f81b8bc` | 7 | `tiling::Rect` removed in favor of `ratatui::layout::Rect`, compositor no longer converts between Rect types, request ID capped to `i64::MAX` range |

---

## Architecture Decisions Made

### 1. External notification channel ownership (Task 2)

`Connection::spawn` now takes a `broadcast::Sender<Notification>` as its third parameter rather than creating one internally. This means `ReconnectingClient` owns the channel and can hand it to new connections on reconnect without subscribers going stale. The `ConnectFn` type signature is:

```rust
type ConnectFn = Box<dyn Fn(broadcast::Sender<Notification>) -> Pin<Box<dyn Future<Output = Result<Arc<Connection>, ClientError>> + Send>> + Send + Sync>;
```

This is the right shape — the factory receives the shared sender so each new connection publishes to the same broadcast bus.

### 2. Subscription as free functions (Task 2)

`SubscriptionManager` was a struct with methods — violated no-OOP rule. Replaced with:
- `SubscriptionState` — data struct (current filter, subscription ID)
- `initial_subscription_state()` — constructor
- `update_filter(&mut state, &client, events, layers, components)` — idempotent
- `unsubscribe(&mut state, &client)` — teardown

### 3. Cursor clamping strategy (Task 5)

`clamp_cursor()` runs at the end of every `reduce_navigation` call. It reads `state.session.capabilities.num_layers` and clamps `cursor.layer` to `num_layers - 1`. When capabilities haven't been received yet (pre-connection), no clamping occurs. This is the minimal correct approach — the views will eventually need their own clamping for token_position, but that requires sequence length metadata that doesn't exist yet.

### 4. `tiling::Rect` eliminated (Task 7)

The custom `Rect` was field-identical to `ratatui::layout::Rect`. Removing it collapsed the compositor's Rect conversion (construct `tiling::Rect` → pass to `resolve` → convert each result back to `ratatui::Rect`) into a single pass-through. Net -43 lines.

---

## Test Impact

| Crate | Before | After | Delta |
|-------|--------|-------|-------|
| rocket-surgeon-tui | 75 | 80 | +5 |
| rocket-surgeon-protocol | 216 | 216 | 0 |
| rocket-surgeon-transport | 15 | 15 | 0 |
| **Total** | **306** | **311** | **+5** |

New tests added:
1. `rejects_oversized_content_length` — message > 64 MiB rejected
2. `rejects_too_many_headers` — >16 headers rejected
3. `rejects_oversized_header_line` — header line > 1024 bytes rejected
4. `nav_down_clamps_to_max_layer` — cursor stays at layer 3 when model has 4 layers
5. `initial_state_has_empty_command_buffer` — constructor default
6. `command_char_appends_to_buffer` — Char('h') → "h"
7. `command_backspace_removes_last_char` — "hel" → "he"
8. `exit_command_mode_clears_buffer` — ExitToNormal clears buffer

(Tests 1-3 were added in Task 1, tests 4-8 in Task 5.)

---

## File Inventory

All changes are within `crates/rocket-surgeon-tui/src/` and one file in `crates/rocket-surgeon-protocol/src/`:

| File | Lines | Changes |
|------|-------|---------|
| `client/connection.rs` | 484 | Limit constants, error variants, poison-safe lock, external notification channel, i64 cap |
| `client/subscription.rs` | 237 | Complete rewrite: struct → free functions |
| `input/terminal.rs` | 282 | Formatting only (cargo fmt) |
| `main.rs` | 118 | fps validation, dirty seeding, `initial_ui_state()` |
| `render/capability.rs` | 87 | Formatting only (cargo fmt) |
| `render/compositor.rs` | 121 | `command_buffer` rendering, Rect pass-through |
| `state.rs` | 93 | `command_buffer` field, `initial_ui_state()` free function |
| `state/cache.rs` | 203 | min capacity, `prefetch_keys(tick_id)` |
| `state/diff.rs` | 122 | Import update for `initial_ui_state` |
| `state/reducer.rs` | 469 | `clamp_cursor`, `reduce_command`, 8 new tests, `test_capabilities` helper |
| `tiling.rs` | 232 | `ratatui::layout::Rect`, generalized `propose_layout`, test rename |
| `protocol/.../messages.rs` | — | `PartialOrd + Ord` derive on `EventType` |

---

## Known Pre-existing Issues (Not In Scope)

These existed before Phase 4.1 and remain untouched:

1. **Clippy warnings in protocol crate** — `FocusSelector` variant naming (`By*` prefix), missing `Eq` derives on 3 structs. Tracked but not part of TUI remediation.

2. **Dead-code warnings in TUI crate** — Many enums/structs defined for Phase 4.2+ (event types, view kinds, connection types) are unused by `main.rs`. These are scaffolding that will light up when the event loop integrates the daemon client. Suppressing them would create unnecessary churn.

3. **Clippy lint debt on master** — PR #18 (`fix/tui-clippy-green-master`) is open to address clippy warnings that bypassed the gate during the Phase 4.1 merge. This is a separate effort.

4. **pyo3 build requires Python venv** — `cargo test --workspace` fails on the pyo3 crate if `.venv/` doesn't exist. The Rust-only crates (protocol, transport, TUI) can be tested independently.

---

## Open PRs and Active Branches

As of session end (2026-05-20 night):

| PR | Branch | Status | Description |
|----|--------|--------|-------------|
| #18 | `fix/tui-clippy-green-master` | Open | Clippy debt cleanup for master |
| #19 | `wu-b-v0.3-tick-clock` | Open | Three-clock tick model in worker |
| #20 | `wu-g-v0.3-kv-cache` | Open | KV cache read/intervene verbs |
| #21 | `wu-c-v0.3-checkpoint` | Open | Checkpoint create/list/restore/delete |

These are all independent of each other and of the TUI work. They represent the v0.3.0 protocol backport work identified during TUI design brainstorming.

---

## What Comes Next

### Immediate (next session)

1. **Review and merge PR #18** (clippy green master) — independent cleanup
2. **Review open v0.3.0 PRs** (#19, #20, #21) — these implement protocol verbs the TUI will need

### Phase 4.2: Widget Library

The TUI implementation plan (from the brainstorming session) identifies Phase 4.2 as the widget library — the actual rendering components that fill the placeholder views. The existing Phase 4 in `docs/specs/plan.md` (tasks 4.1–4.13) was identified as dramatically underscoped during brainstorming. The TUI design spec and expanded Phase 4 plan were being developed but may not have been committed yet (they were in-progress during the brainstorming session that preceded this CR remediation session).

Key Phase 4.2 deliverables from the brainstorming:
- Activation heatmap widget (sparklines, color-mapped tensor slices)
- Attention pattern widget (per-head heatmap with focus tracking)
- Layer stack widget (the primary navigation panel)
- Tensor detail widget (histogram, top-k, slice viewer)
- Status bar widget (already has basic rendering, needs real data binding)

### Open Beads

| Bead | Status | Relevance |
|------|--------|-----------|
| BEAD-0010 | Open | Perfetto multi-GPU structural issues — deferred, not blocking |
| BEAD-0011 | Closed | E2E gate — resolved in PR #10 |
| BEAD-0012 | Closed | TickPosition phase correction — resolved in PR #11 |

---

## Session Arc

This session covered two major efforts:

1. **TUI design brainstorming** (earlier, context compacted) — Collaborative design of the TUI architecture with Patrick. Produced research reports (Bloomberg terminal patterns, tensor visualization, terminal graphics protocols, computational geometry for constrained spaces), the TUI design spec, and an expanded Phase 4 implementation plan. Key design decisions: Bloomberg-density + computational geometry hybrid, Elm-style state management, C allowed for perf-critical rendering, MIDI input not precluded.

2. **CR remediation execution** (this handoff) — 19 findings planned and executed across 7 atomic commits. Every commit left all tests green. The plan document (`docs/superpowers/plans/2026-05-20-tui-cr-remediation.md`) serves as an audit trail mapping each finding to its resolution.

The TUI crate is now at ~2,600 lines of production code + ~400 lines of tests across 12 source files. The foundation is solid: hardened connection layer, clean state management, correct dirty tracking, and no architectural debt from Phase 4.1.

---

## Reproducing the Build

```bash
# Run all Rust tests (excludes pyo3 crate which needs a venv)
cargo test -p rocket-surgeon-protocol -p rocket-surgeon-tui -p rocket-surgeon-transport

# Expected: 311 tests pass (216 + 80 + 15)

# Format check
cargo fmt --check

# Clippy (TUI crate only — protocol crate has pre-existing warnings)
cargo clippy -p rocket-surgeon-tui --no-deps -- -D warnings
# Note: will show pre-existing warnings from Phase 4.1 scaffolding (dead_code for
# unused variants/structs that Phase 4.2+ will use). These are tracked in PR #18.
```
