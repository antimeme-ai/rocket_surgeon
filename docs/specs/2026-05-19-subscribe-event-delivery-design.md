# WU 1.14: Subscribe + Event Delivery — Design Spec

## Goal

Wire event delivery end-to-end: client sends `rocket/subscribe` to enable push notifications, daemon emits `tick.stopped`, `probe.fired`, and `tick.heartbeat` as JSON-RPC Notifications with monotonic `seq`. Unsubscribe turns events off. No subscription state, no per-event filtering, no multi-client fan-out — push everything to the single connected client.

## Dependencies

- WU 1.12 (probe events): worker produces `ProbeFiredEvent`s in step response — done
- WU 1.10 (step integration): barrier-driven stepping, tick state — done

## TCK Contract

10 scenarios in `tck/protocol/subscribe.feature` (full rewrite):

1. Subscribe enables events — send `rocket/subscribe`, get back available event types, step, receive `tick.stopped`
2. Events not sent before subscribe — step without subscribing, verify no notifications
3. Unsubscribe stops events — subscribe, step (get events), unsubscribe, step (no events)
4. Heartbeat while stopped — subscribe, wait ~3s, receive >= 2 heartbeat notifications
5. Probe.fired delivered after step — subscribe, define probe, step through matching layer, receive `probe.fired`
6. Notifications have monotonic seq — subscribe, step twice, verify seq values are strictly increasing
7. Subscribe is idempotent — subscribe twice, no error, events still flow
8. Unsubscribe is idempotent — unsubscribe without prior subscribe, no error
9. Subscribe requires stopped state — attempt subscribe in wrong state, get error
10. Notification wire format — verify no `id` field, has `method`, has `params.seq`

---

## 1. Event Delivery Model

No subscription state. One boolean: `events_enabled`. `rocket/subscribe` flips it to `true`, `rocket/unsubscribe` flips it to `false`. When enabled, the daemon pushes every event to the client as a JSON-RPC Notification. The client receives and correlates by `method` field.

This follows the DAP/GDB model — the debugger pushes everything, the client decides what to ignore. Reference implementations studied: DAP (VS Code), GDB/MI, LLDB/SB, Carmack's Quake event loops.

No multi-client fan-out. rocket_surgeon is one-client-per-session over stdin/stdout. If we ever need multiple clients, that's a multiplexer concern above the daemon, not subscription state inside it.

## 2. Daemon Loop

The current loop is blocking: `read_message → process → respond → repeat`. With events enabled, we need a read-with-timeout to emit heartbeats during idle periods.

Two-path loop:

```
loop {
    if events_enabled {
        // heartbeat at top of loop
        if elapsed_since_last_heartbeat >= 1s {
            emit tick.heartbeat notification
        }

        match read_message_timeout(&mut reader, stdin_fd, 1s) {
            None => continue,         // timeout, loop back for heartbeat check
            Some(raw) => process(raw)  // got a message, handle it
        }
    } else {
        // original blocking path — zero overhead when events are off
        raw = read_message(&mut reader)?;
        process(raw)
    }

    // after processing a step response:
    if events_enabled && step_just_completed {
        emit tick.stopped notification
        for each probe_event in step_response.events {
            emit probe.fired notification
        }
    }
}
```

### Wrinkles

**BufReader + poll() interaction.** `BufReader` may hold data in its internal buffer that `poll()` on the raw fd cannot see. The `read_message_timeout` function must check `reader.buffer().is_empty()` before falling through to `poll()`. If the buffer has data, skip the poll and read immediately.

**Batched probe events.** `ProbeFiredEvent`s are produced by the worker and returned in `HostStepResponse.events` after the step completes. They are not streamed during the step. The daemon emits them as individual notifications after sending the step response.

**tick.stopped is semi-redundant with the step response.** The step response already tells the client where it stopped. `tick.stopped` exists for symmetry with future event sources (barriers, breakpoints) that stop the session without a client-initiated step. For now, both are sent.

### Time note

_This is Carmack's idle-loop-with-heartbeat pattern. When rocket_surgeon eventually has to wrangle real time (timeline scrubbing, replay at wall-clock rate, time-synchronized multi-rank coordination), this 1s poll loop becomes the nucleus of a proper time-aware event loop. That reckoning won't be simple — but this foundation is the right starting point._

## 3. Event Types

Three event types for WU 1.14:

| Event | Method | When |
|-------|--------|------|
| `tick.stopped` | `tick.stopped` | After step completes (session stops at new position) |
| `probe.fired` | `probe.fired` | After step, for each probe that matched during the step |
| `tick.heartbeat` | `tick.heartbeat` | Every ~1s while stopped and events enabled |

### Wire format

JSON-RPC Notification — no `id` field on the envelope, has `method` and `params`:

```json
{
    "jsonrpc": "2.0",
    "method": "tick.stopped",
    "params": {
        "seq": 1,
        "position": { "tick_id": 42, "layer": 12, "rank": 0 },
        "state": "stopped"
    }
}
```

**`seq`** is a `u64`, monotonically increasing, daemon-scoped. Incremented per notification sent. Lives in `params`, not on the envelope — we do not step on JSON-RPC semantics with `id`. Purpose: audit ordering, gap detection, log correlation.

## 4. Subscribe / Unsubscribe

### `rocket/subscribe`

Flips `events_enabled = true`. Idempotent. Requires stopped state (error otherwise).

Request: empty object `{}`.

Response:
```json
{
    "available_events": ["tick.stopped", "tick.heartbeat", "probe.fired"],
    "status": "stopped"
}
```

### `rocket/unsubscribe`

Flips `events_enabled = false`. Idempotent. No error if already unsubscribed.

Request: empty object `{}`.

Response:
```json
{
    "status": "stopped"
}
```

Both use `method::SUBSCRIBE` (`"rocket/subscribe"`) and `method::UNSUBSCRIBE` (`"rocket/unsubscribe"`).

## 5. Transport Layer Changes

### `read_message_timeout`

New function in `crates/rocket-surgeon-transport/src/framing.rs`:

```rust
fn read_message_timeout(
    reader: &mut BufReader<impl Read + AsRawFd>,
    timeout_ms: i32,
) -> Result<Option<String>, FramingError>
```

Behavior:
1. Check `reader.buffer().is_empty()` — if buffer has data, skip poll, read immediately
2. If buffer empty, `libc::poll()` on the reader's raw fd with `timeout_ms`
3. If poll returns 0 (timeout), return `Ok(None)`
4. If poll returns > 0 (data ready), call `read_message(reader)` and return `Ok(Some(msg))`
5. If poll returns -1, return `Err`

`poll()` is POSIX, works on macOS and Linux. No platform-specific code needed. Requires `use std::os::unix::io::AsRawFd` (or `AsFd` with `BorrowedFd` if we prefer the newer API — `AsRawFd` is simpler for `libc::poll`).

### `send_notification`

Thin helper in the daemon:

```rust
fn send_notification(writer: &mut impl Write, seq: &mut u64, method: &str, params: Value) {
    // inject seq into params
    // construct Notification
    // serialize + write_message
    // increment *seq
}
```

## 6. Protocol Type Changes

### Modify

- **`SubscribeRequest`** — remove `events` and `filter` fields. Empty struct (push everything, no filtering).
- **`SubscribeResponse`** — remove `subscription_id`. Rename `subscribed_events` to `available_events`. Add `status: SessionState`.
- **`TickStoppedEvent`** — add `seq: u64`.
- **`TickHeartbeatEvent`** — add `seq: u64`.
- **`ProbeFiredEvent`** — add `seq: u64`.

### Add

- **`UnsubscribeRequest`** — empty struct.
- **`UnsubscribeResponse`** — `status: SessionState`.
- **`method::UNSUBSCRIBE`** — `"rocket/unsubscribe"`.

### Delete

- **`SubscriptionFilter`** — no filtering in push-everything model.

## 7. File Map

| File | Action |
|------|--------|
| `crates/rocket-surgeon-transport/src/framing.rs` | Add `read_message_timeout()` |
| `crates/rocket-surgeon-protocol/src/messages.rs` | Simplify `SubscribeRequest`/`SubscribeResponse`, delete `SubscriptionFilter`, add `UnsubscribeRequest`/`UnsubscribeResponse`, add `seq` to event structs, add `method::UNSUBSCRIBE` |
| `crates/rocket-surgeon/src/main.rs` | Restructure main loop (two-path: blocking / poll+heartbeat), add `events_enabled` + `seq` state, add `send_notification` helper, handle subscribe/unsubscribe dispatch, emit events after step |
| `tck/protocol/subscribe.feature` | Full rewrite — 10 scenarios for push-everything model |
| `tests/test_e2e_subscribe.py` | New E2E test: subscribe → step → events → unsubscribe lifecycle |
