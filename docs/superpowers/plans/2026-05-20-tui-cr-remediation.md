# TUI Phase 4.1 CR Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 19 code review findings from the Phase 4.1 retroactive CR, ordered by dependency so each task lands on a green build.

**Architecture:** Group findings into 7 tasks by file affinity and dependency order. Connection.rs changes land first (broadest blast radius — 8 test call sites), then state/cache fixes, then compositor/tiling cosmetics last. Each task is one atomic commit.

**Tech Stack:** Rust, tokio, ratatui, rocket-surgeon-protocol

---

## Dependency analysis

The key constraint is **cascading signature changes**. Two changes have wide blast radius:

1. **`Connection::spawn` signature** (P1 #5 fix) — adding `broadcast::Sender<Notification>` as 3rd param touches 8 call sites (5 in connection.rs tests, 3 in subscription.rs tests).
2. **`UiState::initial()` → free function** (P2 #12 fix) — touches 8 call sites across 6 files (state.rs, reducer.rs, diff.rs, compositor.rs, tiling.rs, main.rs).

These two changes are independent of each other but each cascades internally. Group them into separate tasks to keep commits reviewable.

## Finding → Task mapping

| Finding | ID | Sev | Task |
|---|---|---|---|
| read_loop silent drops | #1 | P0 | 1 |
| mutex poisoning | #2 | P0 | 1 |
| unbounded Content-Length alloc | #4 | P0 | 1 |
| header limits | #10 | P1 | 1 |
| stale broadcast after reconnect | #5 | P1 | 2 |
| write lock during backoff | #6 | P1 | 2 |
| subscription uses Connection not ReconnectingClient | #11 | P2 | 2 |
| --fps 0 div-by-zero | #3 | P0 | 3 |
| `\|\| true` defeats dirty tracking | #9 | P1 | 3 |
| cache new(0) infinite loop | #7 | P1 | 4 |
| prefetch_keys hardcoded tick_id | #8 | P1 | 4 |
| OOP patterns (UiState::initial) | #12 | P2 | 5 |
| missing command_buffer | #15 | P2 | 5 |
| no cursor clamping | #14 | P2 | 5 |
| hardcoded attn heuristic | #13 | P2 | 6 |
| EventType sort | #16 | P3 | 6 |
| WezTerm comment | #17 | P3 | 6 |
| duplicate Rect | #18 | P3 | 7 |
| i64 cast | #19 | P3 | 7 |

## Task order

```
Task 1 (connection hardening)
  └→ Task 2 (reconnect + subscription rework)  [depends on Task 1 — same file]
Task 3 (main.rs fixes)                         [independent]
Task 4 (cache fixes)                           [independent]
Task 5 (state rework)                          [independent]
Task 6 (tiling + subscription cosmetics)       [after Task 2 for EventType sort in subscription]
Task 7 (compositor + tiling cleanup)           [after Task 5 for Rect usage, after Task 6 for tiling changes]
```

Tasks 1, 3, 4, 5 can proceed in parallel. Task 2 after 1. Task 6 after 2. Task 7 last.
For serial execution: 1 → 2 → 3 → 4 → 5 → 6 → 7.

---

### Task 1: Connection hardening (P0 #1, #2, #4; P1 #10)

**Files:**
- Modify: `crates/rocket-surgeon-tui/src/client/connection.rs`
- Test: same file, `mod tests`

**Findings addressed:**
- **P0 #1**: `read_loop` silently discards unparseable messages — add `tracing::warn!`
- **P0 #2**: `pending.lock().unwrap()` panics on poisoned mutex — use `unwrap_or_else(|e| e.into_inner())`
- **P0 #4**: `read_content_length_message` allocates `vec![0u8; len]` with no size cap — add `MAX_MESSAGE_SIZE`
- **P1 #10**: No header count or line length limits — add `MAX_HEADER_COUNT`, `MAX_HEADER_LINE_LEN`

- [ ] **Step 1: Write test for Content-Length size limit**

In `connection.rs` `mod tests`, add:

```rust
#[tokio::test]
async fn rejects_oversized_content_length() {
    let msg = b"Content-Length: 999999999\r\n\r\n";
    let mut reader = tokio::io::BufReader::new(&msg[..]);
    let result = read_content_length_message(&mut reader).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        ClientError::MessageTooLarge => {}
        other => panic!("expected MessageTooLarge, got {other:?}"),
    }
}
```

- [ ] **Step 2: Write test for header count limit**

```rust
#[tokio::test]
async fn rejects_too_many_headers() {
    let mut msg = String::new();
    for i in 0..20 {
        msg.push_str(&format!("X-Header-{i}: value\r\n"));
    }
    msg.push_str("\r\n");
    let mut reader = tokio::io::BufReader::new(msg.as_bytes());
    let result = read_content_length_message(&mut reader).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        ClientError::TooManyHeaders => {}
        other => panic!("expected TooManyHeaders, got {other:?}"),
    }
}
```

- [ ] **Step 3: Write test for header line length limit**

```rust
#[tokio::test]
async fn rejects_oversized_header_line() {
    let long_value = "x".repeat(2048);
    let msg = format!("Content-Length: {long_value}\r\n\r\n");
    let mut reader = tokio::io::BufReader::new(msg.as_bytes());
    let result = read_content_length_message(&mut reader).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        ClientError::HeaderLineTooLong => {}
        other => panic!("expected HeaderLineTooLong, got {other:?}"),
    }
}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cargo test -p rocket-surgeon-tui --lib client::connection::tests`
Expected: FAIL — `ClientError::MessageTooLarge`, `TooManyHeaders`, `HeaderLineTooLong` variants don't exist yet.

- [ ] **Step 5: Add error variants and constants, implement limits**

Add to `ClientError`:
```rust
#[error("message too large (max {MAX_MESSAGE_SIZE} bytes)")]
MessageTooLarge,
#[error("too many headers (max {MAX_HEADER_COUNT})")]
TooManyHeaders,
#[error("header line too long (max {MAX_HEADER_LINE_LEN} bytes)")]
HeaderLineTooLong,
```

Add constants at module level:
```rust
const MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024; // 64 MiB
const MAX_HEADER_COUNT: usize = 16;
const MAX_HEADER_LINE_LEN: usize = 1024;
```

In `read_content_length_message`:
- After parsing `content_length`, check `len > MAX_MESSAGE_SIZE` → return `Err(ClientError::MessageTooLarge)`
- Count headers in the loop, return `Err(ClientError::TooManyHeaders)` if count exceeds `MAX_HEADER_COUNT`
- After `read_line`, check `line.len() > MAX_HEADER_LINE_LEN` → return `Err(ClientError::HeaderLineTooLong)`

- [ ] **Step 6: Fix mutex poisoning — replace `unwrap()` with poison-safe helper**

Add a helper function:
```rust
fn lock_pending(pending: &PendingMap) -> std::sync::MutexGuard<'_, HashMap<RequestId, oneshot::Sender<Result<Response, ClientError>>>> {
    pending.lock().unwrap_or_else(|e| e.into_inner())
}
```

Replace all 4 occurrences of `pending.lock().unwrap()` (in `request`, `read_loop` x2, and wherever else) with `lock_pending(&pending)` or `lock_pending(&self.pending)`.

- [ ] **Step 7: Fix read_loop silent message drops — add tracing warnings**

In `read_loop`, after the two `if let Ok(...)` blocks, add a fallthrough case:
```rust
if let Ok(resp) = serde_json::from_str::<Response>(&msg) {
    if let Some(tx) = lock_pending(&pending).remove(&resp.id) {
        let _ = tx.send(Ok(resp));
    }
    continue;
}

if let Ok(raw) = serde_json::from_str::<RawMessage>(&msg) {
    if let Some(notif) = raw.into_notification() {
        let _ = notification_tx.send(notif);
        continue;
    }
}

tracing::warn!(msg_len = msg.len(), "dropping unparseable message");
```

- [ ] **Step 8: Run all connection tests**

Run: `cargo test -p rocket-surgeon-tui --lib client::connection::tests`
Expected: all tests pass, including the 3 new limit tests and 4 existing tests.

- [ ] **Step 9: Run full TUI test suite**

Run: `cargo test -p rocket-surgeon-tui`
Expected: all tests pass. The subscription tests still work because `Connection::spawn` signature hasn't changed yet.

- [ ] **Step 10: Run clippy and fmt**

Run: `cargo fmt -p rocket-surgeon-tui && cargo clippy -p rocket-surgeon-tui -- -D warnings`
Expected: clean.

- [ ] **Step 11: Commit**

```bash
git add crates/rocket-surgeon-tui/src/client/connection.rs
git commit -m "fix(tui): harden connection — message size limits, poison-safe mutex, log dropped messages

Addresses CR findings P0 #1 (silent drops), P0 #2 (mutex poisoning),
P0 #4 (unbounded alloc), P1 #10 (no header limits).

Closes BEAD-0011."
```

---

### Task 2: Reconnect architecture + subscription rework (P1 #5, #6; P2 #11)

**Files:**
- Modify: `crates/rocket-surgeon-tui/src/client/connection.rs`
- Modify: `crates/rocket-surgeon-tui/src/client/subscription.rs`
- Test: both files, `mod tests`

**Findings addressed:**
- **P1 #5**: After reconnect, old `broadcast::Receiver` is stale — callers subscribed before reconnect never see new-connection notifications. Fix: `Connection::spawn` takes an externally-owned `broadcast::Sender<Notification>` so all connections share the same channel.
- **P1 #6**: `reconnect()` holds write lock during exponential backoff sleeps — blocks all reads. Fix: only acquire write lock after successful connection.
- **P2 #11**: `SubscriptionManager` takes `Arc<Connection>` but should use `ReconnectingClient` — can't survive reconnects. Also uses OOP `&mut self` pattern. Fix: convert to data struct + free functions taking `&ReconnectingClient`.

- [ ] **Step 1: Write test for shared notification channel across reconnect**

In `connection.rs` `mod tests`, add:

```rust
#[tokio::test]
async fn subscriber_receives_notifications_after_reconnect() {
    use std::sync::atomic::AtomicUsize;

    let (notification_tx, _) = broadcast::channel(256);
    let attempt = Arc::new(AtomicUsize::new(0));
    let (server_tx, mut server_rx) = mpsc::channel::<tokio::io::DuplexStream>(4);

    let attempt_clone = Arc::clone(&attempt);
    let ntx = notification_tx.clone();
    let connect: ConnectFn = Box::new(move || {
        let server_tx = server_tx.clone();
        let attempt = Arc::clone(&attempt_clone);
        let ntx = ntx.clone();
        Box::pin(async move {
            attempt.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let (client_stream, server_stream) = tokio::io::duplex(4096);
            server_tx.send(server_stream).await.map_err(|_| ClientError::Closed)?;
            let (r, w) = tokio::io::split(client_stream);
            Ok(Connection::spawn(r, w, ntx))
        })
    });

    let (client_stream, first_server) = tokio::io::duplex(4096);
    let (r, w) = tokio::io::split(client_stream);
    let initial_conn = Connection::spawn(r, w, notification_tx.clone());
    let client = Arc::new(ReconnectingClient::new(initial_conn, connect, notification_tx.clone()));

    // Subscribe BEFORE reconnect
    let mut sub = notification_tx.subscribe();

    // Drop first server to simulate disconnect
    drop(first_server);

    // Spawn server to handle reconnected request, then send notification
    let server_handle = tokio::spawn(async move {
        let mut server_stream = server_rx.recv().await.unwrap();

        // Handle the request
        let mut reader = BufReader::new(&mut server_stream);
        let msg = read_content_length_message(&mut reader).await.unwrap();
        let req: Request = serde_json::from_str(&msg).unwrap();
        let resp = Response::success(req.id, serde_json::json!({"ok": true}));
        let body = serde_json::to_string(&resp).unwrap();
        let frame = frame_message(&body);
        use tokio::io::AsyncWriteExt;
        server_stream.write_all(&frame).await.unwrap();
        server_stream.flush().await.unwrap();

        // Now send a notification on the new connection
        let notif = Notification::new("tick.stopped", serde_json::json!({"position": {"tick_id": 99}}));
        let nbody = serde_json::to_string(&notif).unwrap();
        let nframe = frame_message(&nbody);
        server_stream.write_all(&nframe).await.unwrap();
        server_stream.flush().await.unwrap();
    });

    // Trigger reconnect via request
    let resp = client.request("rocket/status", serde_json::json!({})).await.unwrap();
    assert!(resp.result.is_some());

    // The pre-reconnect subscriber should receive the notification
    let received = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        sub.recv(),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(received.method, "tick.stopped");

    server_handle.await.unwrap();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p rocket-surgeon-tui --lib client::connection::tests::subscriber_receives_notifications_after_reconnect`
Expected: FAIL — `Connection::spawn` doesn't accept 3 args yet.

- [ ] **Step 3: Change `Connection::spawn` to accept external `broadcast::Sender<Notification>`**

Change the signature:
```rust
pub fn spawn<R, W>(reader: R, writer: W, notification_tx: broadcast::Sender<Notification>) -> Arc<Self>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (outgoing_tx, outgoing_rx) = mpsc::channel(64);
    let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));

    let conn = Arc::new(Self {
        outgoing_tx,
        notification_tx: notification_tx.clone(),
        next_id: AtomicU64::new(1),
        pending: Arc::clone(&pending),
    });

    tokio::spawn(write_loop(outgoing_rx, writer));
    tokio::spawn(read_loop(reader, notification_tx, Arc::clone(&pending)));

    conn
}
```

Remove the internal `broadcast::channel` creation — the caller owns the channel now.

- [ ] **Step 4: Update `ReconnectingClient` to own the notification channel**

```rust
pub struct ReconnectingClient {
    conn: tokio::sync::RwLock<Arc<Connection>>,
    connect: ConnectFn,
    notification_tx: broadcast::Sender<Notification>,
    max_retries: u32,
    base_delay: Duration,
}
```

Update `ConnectFn` type:
```rust
pub type ConnectFn = Box<
    dyn Fn(broadcast::Sender<Notification>) -> Pin<Box<dyn Future<Output = Result<Arc<Connection>, ClientError>> + Send>>
        + Send
        + Sync,
>;
```

Update `ReconnectingClient::new`:
```rust
pub fn new(conn: Arc<Connection>, connect: ConnectFn, notification_tx: broadcast::Sender<Notification>) -> Self {
    Self {
        conn: tokio::sync::RwLock::new(conn),
        connect,
        notification_tx,
        max_retries: 5,
        base_delay: Duration::from_millis(100),
    }
}

pub fn subscribe(&self) -> broadcast::Receiver<Notification> {
    self.notification_tx.subscribe()
}
```

Note: `subscribe` is no longer `async` — it reads from the owned sender, not the connection.

- [ ] **Step 5: Fix reconnect to not hold write lock during backoff (#6)**

```rust
async fn reconnect(&self) -> Result<Arc<Connection>, ClientError> {
    for attempt in 0..self.max_retries {
        let delay = self.base_delay * 2u32.saturating_pow(attempt);
        tokio::time::sleep(delay).await;

        match (self.connect)(self.notification_tx.clone()).await {
            Ok(new_conn) => {
                let mut write = self.conn.write().await;
                *write = new_conn.clone();
                return Ok(new_conn);
            }
            Err(_) if attempt + 1 < self.max_retries => continue,
            Err(e) => return Err(e),
        }
    }
    Err(ClientError::Closed)
}
```

Key change: `self.conn.write().await` is acquired **after** successful connection, not before the retry loop.

- [ ] **Step 6: Update all connection.rs test call sites (5 sites)**

Every test that calls `Connection::spawn(r, w)` must become `Connection::spawn(r, w, notification_tx.clone())`, creating a `broadcast::channel(256)` at the top of the test. Update these tests:
- `request_response_roundtrip`
- `notification_forwarded_to_subscriber`
- `reconnects_after_disconnect`
- `closed_transport_returns_error`
- `subscriber_receives_notifications_after_reconnect` (already uses 3-arg)

For `reconnects_after_disconnect`, also update:
- `ConnectFn` closure to accept `notification_tx` parameter
- `ReconnectingClient::new` to pass `notification_tx`

For `notification_forwarded_to_subscriber`, change `conn.subscribe()` to `notification_tx.subscribe()`.

- [ ] **Step 7: Run connection tests**

Run: `cargo test -p rocket-surgeon-tui --lib client::connection::tests`
Expected: all pass.

- [ ] **Step 8: Convert `SubscriptionManager` to data struct + free functions**

Replace the OOP `SubscriptionManager` with:

```rust
use std::collections::HashSet;
use std::sync::Arc;

use rocket_surgeon_protocol::jsonrpc::Notification;
use rocket_surgeon_protocol::messages::{
    method, EventType, SubscribeFilter, SubscribeRequest,
};
use tokio::sync::broadcast;

use super::connection::{ClientError, ReconnectingClient};

pub struct SubscriptionState {
    pub active_events: HashSet<EventType>,
    pub active_layers: Option<Vec<u32>>,
    pub active_components: Option<Vec<String>>,
}

pub fn initial_subscription_state() -> SubscriptionState {
    SubscriptionState {
        active_events: HashSet::new(),
        active_layers: None,
        active_components: None,
    }
}

pub async fn update_filter(
    state: &mut SubscriptionState,
    client: &ReconnectingClient,
    events: HashSet<EventType>,
    layers: Option<Vec<u32>>,
    components: Option<Vec<String>>,
) -> Result<(), ClientError> {
    if events == state.active_events
        && layers == state.active_layers
        && components == state.active_components
    {
        return Ok(());
    }

    let filter = if events.is_empty() && layers.is_none() && components.is_none() {
        None
    } else {
        let event_list = if events.is_empty() {
            None
        } else {
            let mut sorted: Vec<EventType> = events.iter().copied().collect();
            sorted.sort_by_key(|e| format!("{e:?}"));
            Some(sorted)
        };
        Some(SubscribeFilter {
            events: event_list,
            layers: layers.clone(),
            components: components.clone(),
        })
    };

    let req = SubscribeRequest { filter };
    let params = serde_json::to_value(&req).map_err(ClientError::Json)?;
    let resp = client.request(method::SUBSCRIBE, params).await?;

    if let Some(err) = resp.error {
        return Err(ClientError::Rpc {
            code: err.code,
            message: err.message,
        });
    }

    state.active_events = events;
    state.active_layers = layers;
    state.active_components = components;
    Ok(())
}

pub async fn unsubscribe(
    state: &mut SubscriptionState,
    client: &ReconnectingClient,
) -> Result<(), ClientError> {
    let params = serde_json::to_value(&serde_json::json!({})).unwrap();
    let resp = client.request(method::UNSUBSCRIBE, params).await?;

    if let Some(err) = resp.error {
        return Err(ClientError::Rpc {
            code: err.code,
            message: err.message,
        });
    }

    state.active_events.clear();
    state.active_layers = None;
    state.active_components = None;
    Ok(())
}
```

- [ ] **Step 9: Rewrite subscription.rs tests for new API**

Tests need to construct `ReconnectingClient` instead of bare `Connection`. Each test:
1. Creates a `broadcast::channel(256)`
2. Creates `Connection::spawn(r, w, notification_tx.clone())`
3. Creates a `ConnectFn` (can be a dummy that returns `Err(ClientError::Closed)` since we won't trigger reconnect in these tests)
4. Creates `ReconnectingClient::new(conn, connect, notification_tx)`
5. Creates `SubscriptionState` via `initial_subscription_state()`
6. Calls `update_filter(&mut state, &client, ...)` instead of `mgr.update_filter(...)`

Update all 3 tests: `subscribe_sends_filter_to_server`, `no_op_when_filter_unchanged`, `unsubscribe_clears_state`.

- [ ] **Step 10: Run all subscription tests**

Run: `cargo test -p rocket-surgeon-tui --lib client::subscription::tests`
Expected: all 3 pass.

- [ ] **Step 11: Run full TUI test suite**

Run: `cargo test -p rocket-surgeon-tui`
Expected: all tests pass.

- [ ] **Step 12: Run clippy and fmt**

Run: `cargo fmt -p rocket-surgeon-tui && cargo clippy -p rocket-surgeon-tui -- -D warnings`
Expected: clean.

- [ ] **Step 13: Commit**

```bash
git add crates/rocket-surgeon-tui/src/client/connection.rs crates/rocket-surgeon-tui/src/client/subscription.rs
git commit -m "fix(tui): shared notification channel, reconnect lock fix, subscription rework

Connection::spawn takes external broadcast::Sender so subscribers survive
reconnection. ReconnectingClient only acquires write lock after successful
connect. SubscriptionManager replaced with SubscriptionState + free functions.

Addresses CR findings P1 #5 (stale broadcast), P1 #6 (write lock during
backoff), P2 #11 (subscription uses wrong client type + OOP pattern)."
```

---

### Task 3: main.rs fixes (P0 #3; P1 #9)

**Files:**
- Modify: `crates/rocket-surgeon-tui/src/main.rs`
- Test: manual verification (main.rs has no unit test harness — these are CLI arg validation and event loop logic)

**Findings addressed:**
- **P0 #3**: `--fps 0` causes division by zero in `1000 / cli.fps as u64`
- **P1 #9**: `|| true` in the render condition defeats dirty tracking, causing unnecessary redraws every frame

- [ ] **Step 1: Add `value_parser` clamp on `--fps` to reject 0**

Change the `fps` field:
```rust
#[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u32).range(1..=240))]
fps: u32,
```

This makes `--fps 0` a clap error before it reaches our code. No need for runtime checks.

- [ ] **Step 2: Remove `|| true` from dirty check**

Change:
```rust
if !state.dirty.is_empty() || true {
```
To:
```rust
if !state.dirty.is_empty() {
```

But also: after initial startup, nothing marks dirty yet (no daemon events firing). The screen will be blank. Add an initial `mark_all_dirty` before the loop so the first frame always renders:

After `let layout = default_layout();`, add:
```rust
for view in &state.views {
    state.dirty.insert(view.id.clone());
}
```

- [ ] **Step 3: Run clippy and fmt**

Run: `cargo fmt -p rocket-surgeon-tui && cargo clippy -p rocket-surgeon-tui -- -D warnings`
Expected: clean.

- [ ] **Step 4: Verify `--fps 0` rejected**

Run: `cargo run -p rocket-surgeon-tui -- --fps 0 2>&1 || true`
Expected: clap error message, non-zero exit.

- [ ] **Step 5: Commit**

```bash
git add crates/rocket-surgeon-tui/src/main.rs
git commit -m "fix(tui): reject --fps 0, remove || true dirty tracking bypass

Clamp fps to 1..=240 via clap value_parser. Remove || true that defeated
dirty tracking — seed initial dirty set instead so first frame renders.

Addresses CR findings P0 #3 (div-by-zero), P1 #9 (dirty bypass)."
```

---

### Task 4: Cache fixes (P1 #7, #8)

**Files:**
- Modify: `crates/rocket-surgeon-tui/src/state/cache.rs`
- Test: same file, `mod tests`

**Findings addressed:**
- **P1 #7**: `TensorCache::new(0)` causes infinite loop in `insert()` — `while self.entries.len() >= self.max_entries` never terminates when max is 0
- **P1 #8**: `prefetch_keys` hardcodes `tick_id: 0` instead of taking it as a parameter

- [ ] **Step 1: Write test for cache new(0)**

```rust
#[test]
fn new_with_zero_capacity_uses_minimum() {
    let mut cache = TensorCache::new(0);
    cache.insert(key(0, "a"), make_summary("a"));
    assert_eq!(cache.len(), 1);
}
```

- [ ] **Step 2: Write test for prefetch_keys with tick_id**

```rust
#[test]
fn prefetch_keys_uses_provided_tick_id() {
    let keys = TensorCache::prefetch_keys(5, 10, "attn.o_proj", 42);
    assert!(keys.iter().all(|k| k.tick_id == 42));
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p rocket-surgeon-tui --lib state::cache::tests`
Expected: FAIL — `new(0)` hangs (test timeout), `prefetch_keys` doesn't accept 4th arg.

- [ ] **Step 4: Fix `TensorCache::new` to enforce minimum capacity**

```rust
pub fn new(max_entries: usize) -> Self {
    Self {
        entries: HashMap::new(),
        order: VecDeque::new(),
        max_entries: max_entries.max(1),
    }
}
```

- [ ] **Step 5: Fix `prefetch_keys` to accept `tick_id` parameter**

Change signature:
```rust
pub fn prefetch_keys(layer: u32, token: u64, component: &str, tick_id: u64) -> Vec<CacheKey> {
```

Remove the `let tick_id = 0;` line. The parameter replaces it.

- [ ] **Step 6: Update existing `prefetch_keys` tests to pass tick_id**

Change `prefetch_keys_adjacent`:
```rust
let keys = TensorCache::prefetch_keys(5, 10, "attn.o_proj", 0);
```

Change `prefetch_at_layer_zero_skips_negative`:
```rust
let keys = TensorCache::prefetch_keys(0, 0, "attn.o_proj", 0);
```

- [ ] **Step 7: Run all cache tests**

Run: `cargo test -p rocket-surgeon-tui --lib state::cache::tests`
Expected: all 9 tests pass (7 existing + 2 new).

- [ ] **Step 8: Run clippy and fmt**

Run: `cargo fmt -p rocket-surgeon-tui && cargo clippy -p rocket-surgeon-tui -- -D warnings`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add crates/rocket-surgeon-tui/src/state/cache.rs
git commit -m "fix(tui): cache minimum capacity, prefetch_keys takes tick_id

TensorCache::new(0) no longer infinite-loops — enforces minimum capacity
of 1. prefetch_keys takes tick_id as parameter instead of hardcoding 0.

Addresses CR findings P1 #7 (infinite loop), P1 #8 (hardcoded tick_id)."
```

---

### Task 5: State rework (P2 #12, #14, #15)

**Files:**
- Modify: `crates/rocket-surgeon-tui/src/state.rs`
- Modify: `crates/rocket-surgeon-tui/src/state/reducer.rs`
- Modify: `crates/rocket-surgeon-tui/src/state/diff.rs`
- Modify: `crates/rocket-surgeon-tui/src/state/cache.rs` (only if it uses `UiState::initial()` — it doesn't)
- Modify: `crates/rocket-surgeon-tui/src/render/compositor.rs`
- Modify: `crates/rocket-surgeon-tui/src/tiling.rs`
- Modify: `crates/rocket-surgeon-tui/src/main.rs`
- Test: `reducer.rs`, `diff.rs`, `compositor.rs`, `tiling.rs` test modules

**Findings addressed:**
- **P2 #12**: `UiState::initial()` is an inherent method — violates no-OOP rule. Convert to free function `initial_ui_state()`.
- **P2 #15**: Missing `command_buffer: String` field on `UiState` — command input currently abuses `status_line`.
- **P2 #14**: Cursor layer/token have no upper-bound clamping — `saturating_add` goes to `u32::MAX`/`u64::MAX`. Add optional clamping when `capabilities.num_layers` is known.

- [ ] **Step 1: Write test for `initial_ui_state` free function**

In `state.rs`, after the existing `impl UiState` block (which we'll remove), the tests in other files already call `UiState::initial()`. We need to verify that the replacement works. Add a test in `reducer.rs`:

```rust
#[test]
fn initial_state_has_empty_command_buffer() {
    let state = initial_ui_state();
    assert!(state.command_buffer.is_empty());
}
```

- [ ] **Step 2: Write test for cursor clamping**

In `reducer.rs`:

```rust
#[test]
fn nav_down_clamps_to_max_layer() {
    let mut state = state_with_views();
    state.session.capabilities = Some(test_capabilities(4));
    state.cursor.layer = 3;
    let new = reduce(
        state,
        UiEvent::Input(InputEvent::Navigation(NavigationEvent::Down)),
    );
    assert_eq!(new.cursor.layer, 3); // clamped to num_layers - 1
}
```

This requires a `test_capabilities` helper:
```rust
fn test_capabilities(num_layers: u32) -> rocket_surgeon_protocol::types::Capabilities {
    rocket_surgeon_protocol::types::Capabilities {
        protocol_version: "0.3.0".into(),
        supports_reverse_step: false,
        supports_checkpointing: false,
        supports_moe: false,
        supports_backward: false,
        supports_sae: false,
        execution_mode: rocket_surgeon_protocol::types::ExecutionMode::Synchronous,
        parallelism: rocket_surgeon_protocol::types::Parallelism::Single,
        tick_granularities: vec![],
        intervention_types: vec![],
        built_in_views: vec![],
        head_granularity: rocket_surgeon_protocol::types::HeadGranularity::PerHead,
        transports: vec![],
        wire_formats: vec![],
        max_response_bytes: 0,
        model_family: None,
        model_id: None,
        num_layers: Some(num_layers),
    }
}
```

- [ ] **Step 3: Write test for command buffer accumulation**

In `reducer.rs`:
```rust
#[test]
fn command_char_appends_to_buffer() {
    let mut state = state_with_views();
    state.mode = Mode::Command;
    let new = reduce(
        state,
        UiEvent::Input(InputEvent::Command(CommandEvent::Char('h'))),
    );
    assert_eq!(new.command_buffer, "h");
}

#[test]
fn command_backspace_removes_last_char() {
    let mut state = state_with_views();
    state.mode = Mode::Command;
    state.command_buffer = "hel".into();
    let new = reduce(
        state,
        UiEvent::Input(InputEvent::Command(CommandEvent::Backspace)),
    );
    assert_eq!(new.command_buffer, "he");
}

#[test]
fn exit_command_mode_clears_buffer() {
    let mut state = state_with_views();
    state.mode = Mode::Command;
    state.command_buffer = "hello".into();
    let new = reduce(
        state,
        UiEvent::Input(InputEvent::Mode(ModeEvent::ExitToNormal)),
    );
    assert!(new.command_buffer.is_empty());
}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cargo test -p rocket-surgeon-tui --lib state::reducer::tests`
Expected: FAIL — `command_buffer` field doesn't exist, `initial_ui_state` doesn't exist.

- [ ] **Step 5: Add `command_buffer` field to `UiState`**

In `state.rs`, add to `UiState`:
```rust
pub command_buffer: String,
```

- [ ] **Step 6: Convert `UiState::initial()` to free function `initial_ui_state()`**

Remove the `impl UiState { pub fn initial() ... }` block. Replace with:
```rust
pub fn initial_ui_state() -> UiState {
    UiState {
        session: SessionSnapshot {
            status: Status::Uninitialized,
            position: None,
            capabilities: None,
            active_interventions: Vec::new(),
            protocol_version: String::new(),
        },
        cursor: CursorState {
            layer: 0,
            component: String::new(),
            token_position: 0,
            focused_view: ViewId(0),
        },
        mode: Mode::default(),
        views: Vec::new(),
        pending_requests: 0,
        status_line: String::new(),
        command_buffer: String::new(),
        dirty: HashSet::new(),
    }
}
```

- [ ] **Step 7: Update all 8 call sites from `UiState::initial()` to `initial_ui_state()`**

Files and locations:
1. `state/reducer.rs` — `state_with_views()` helper and `request_counting` test
2. `state/diff.rs` — `two_view_state()` helper
3. `render/compositor.rs` — `test_state()` helper
4. `tiling.rs` — `propose_layout_attn_component` and `propose_layout_no_change` tests
5. `main.rs` — `run_loop` function

Each file needs to import `initial_ui_state` from `crate::state` (or `super` for submodules).

- [ ] **Step 8: Implement cursor clamping in reducer**

In `reduce_navigation`, add a clamping helper:

```rust
fn clamp_cursor(state: &mut UiState) {
    if let Some(caps) = &state.session.capabilities {
        if let Some(num_layers) = caps.num_layers {
            if num_layers > 0 {
                state.cursor.layer = state.cursor.layer.min(num_layers - 1);
            }
        }
    }
}
```

Call `clamp_cursor(state)` at the end of `reduce_navigation`, after the match but before the function returns.

- [ ] **Step 9: Implement command buffer handling in reducer**

Update `reduce_command`:
```rust
fn reduce_command(state: &mut UiState, cmd: CommandEvent) {
    match cmd {
        CommandEvent::Char(c) => {
            state.command_buffer.push(c);
        }
        CommandEvent::Backspace => {
            state.command_buffer.pop();
        }
        CommandEvent::Execute => {
            state.status_line = format!("executed: {}", state.command_buffer);
            state.command_buffer.clear();
        }
        CommandEvent::Cancel => {
            state.command_buffer.clear();
        }
        _ => {}
    }
    mark_dep_dirty(state, &DataDep::Mode);
}
```

In `reduce_mode`, when exiting to Normal, clear the buffer:
```rust
fn reduce_mode(state: &mut UiState, event: ModeEvent) {
    let target = match event {
        ModeEvent::EnterCommand => Mode::Command,
        ModeEvent::EnterInspect => Mode::Inspect,
        ModeEvent::EnterIntervene => Mode::Intervene,
        ModeEvent::ExitToNormal => Mode::Normal,
    };

    if let Some(new_mode) = state.mode.transition(target) {
        if new_mode == Mode::Normal {
            state.command_buffer.clear();
        }
        state.mode = new_mode;
        mark_dep_dirty(state, &DataDep::Mode);
    }
}
```

- [ ] **Step 10: Update compositor to use command_buffer**

In `render_command_line`:
```rust
fn render_command_line(frame: &mut Frame<'_>, rect: Rect, state: &UiState) {
    let text = if state.mode == crate::input::mode::Mode::Command {
        format!(":{}", state.command_buffer)
    } else {
        state.status_line.clone()
    };
    let para = Paragraph::new(Line::from(text));
    frame.render_widget(para, rect);
}
```

- [ ] **Step 11: Run all TUI tests**

Run: `cargo test -p rocket-surgeon-tui`
Expected: all tests pass.

- [ ] **Step 12: Run clippy and fmt**

Run: `cargo fmt -p rocket-surgeon-tui && cargo clippy -p rocket-surgeon-tui -- -D warnings`
Expected: clean.

- [ ] **Step 13: Commit**

```bash
git add crates/rocket-surgeon-tui/src/state.rs \
       crates/rocket-surgeon-tui/src/state/reducer.rs \
       crates/rocket-surgeon-tui/src/state/diff.rs \
       crates/rocket-surgeon-tui/src/render/compositor.rs \
       crates/rocket-surgeon-tui/src/tiling.rs \
       crates/rocket-surgeon-tui/src/main.rs
git commit -m "fix(tui): state rework — free function, command_buffer, cursor clamping

Replace UiState::initial() with initial_ui_state() free function (no-OOP).
Add command_buffer field for proper command input handling. Clamp cursor
to num_layers when capabilities are known.

Addresses CR findings P2 #12 (OOP pattern), P2 #14 (no cursor clamping),
P2 #15 (missing command_buffer)."
```

---

### Task 6: Subscription + tiling cosmetics (P2 #13; P3 #16, #17)

**Files:**
- Modify: `crates/rocket-surgeon-tui/src/client/subscription.rs`
- Modify: `crates/rocket-surgeon-tui/src/tiling.rs`
- Modify: `crates/rocket-surgeon-tui/src/render/capability.rs`
- Test: `tiling.rs`, `subscription.rs` test modules

**Findings addressed:**
- **P2 #13**: `propose_layout` hardcodes `"attn"` string — should be parameterizable or at minimum documented as placeholder
- **P3 #16**: `EventType` sort uses `format!("{e:?}")` (Debug string) — fragile, add `Ord` derive or use discriminant
- **P3 #17**: WezTerm detected as Kitty tier — WezTerm supports both Kitty and Sixel, but detecting as Kitty is actually correct (it's the higher tier). The finding is about a misleading comment, not wrong behavior.

- [ ] **Step 1: Write test for propose_layout with configurable component match**

In `tiling.rs` tests:
```rust
#[test]
fn propose_layout_returns_none_for_non_attn() {
    let mut old = initial_ui_state();
    old.cursor.component = "mlp".into();

    let mut new = initial_ui_state();
    new.cursor.component = "down_proj".into();

    let proposal = propose_layout(&old, &new);
    assert!(proposal.is_none());
}
```

This test already passes — it documents existing behavior. The real fix is making `propose_layout` take a predicate or removing the heuristic. Since this is placeholder code (the real layout proposal system comes in Phase 4.3), the right fix is to add a doc comment marking it as provisional and remove the hardcoded string in favor of a simple "component changed" trigger.

- [ ] **Step 2: Fix `propose_layout` — remove hardcoded "attn"**

Replace:
```rust
pub fn propose_layout(old: &UiState, new: &UiState) -> Option<Layout> {
    if old.cursor.component != new.cursor.component
        && new.cursor.component.contains("attn")
    {
        return Some(Layout::hsplit(
            Layout::single(ViewId(0)),
            Layout::single(ViewId(2)),
            0.6,
        ));
    }

    None
}
```

With:
```rust
pub fn propose_layout(old: &UiState, new: &UiState) -> Option<Layout> {
    if old.cursor.component != new.cursor.component && !new.cursor.component.is_empty() {
        return Some(Layout::hsplit(
            Layout::single(ViewId(0)),
            Layout::single(ViewId(2)),
            0.6,
        ));
    }

    None
}
```

This triggers on any component change to a non-empty component, which is the correct generalization until the real layout proposal system lands.

- [ ] **Step 3: Update `propose_layout_attn_component` test**

The test name and assertion still work — it tests that changing to an attn component triggers a proposal. But the test name is misleading now. Rename:

```rust
#[test]
fn propose_layout_on_component_change() {
    let mut old = initial_ui_state();
    old.cursor.component = "mlp".into();

    let mut new = initial_ui_state();
    new.cursor.component = "attn.o_proj".into();

    let proposal = propose_layout(&old, &new);
    assert!(proposal.is_some());
    let ids = proposal.unwrap().view_ids();
    assert_eq!(ids.len(), 2);
}
```

- [ ] **Step 4: Fix EventType sort — use discriminant index**

In `subscription.rs` (the `update_filter` function), replace:
```rust
sorted.sort_by_key(|e| format!("{e:?}"));
```
With:
```rust
sorted.sort_by_key(|e| *e as u8);
```

This requires adding a `repr` attribute to `EventType` in the protocol crate. Check if it has one... it doesn't. But we can use `std::mem::discriminant` ordering instead. Actually, the simplest stable approach:

```rust
sorted.sort_by_key(|e| match e {
    EventType::TickStopped => 0u8,
    EventType::TickHeartbeat => 1,
    EventType::ProbeFired => 2,
    EventType::ReplayDivergence => 3,
    EventType::Error => 4,
});
```

Wait — this is in the TUI crate matching on a protocol crate enum. That's fragile across protocol changes. Better approach: derive `Ord` on `EventType` in the protocol crate.

In `crates/rocket-surgeon-protocol/src/messages.rs`:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventType {
```

Then in subscription.rs:
```rust
sorted.sort();
```

- [ ] **Step 5: Fix WezTerm comment (P3 #17)**

The finding is that WezTerm is detected as Kitty tier with no explanation. This is actually correct behavior — WezTerm implements the Kitty graphics protocol. No code change needed, but the detection function could use a brief note. Since the CLAUDE.md says "default to writing no comments" and the behavior is correct, we skip this. The finding is informational.

Actually — re-reading the original CR, the finding was that WezTerm should perhaps be detected separately because it supports *both* Sixel and Kitty. But since Kitty is the higher tier, detecting as Kitty is correct. No change needed.

- [ ] **Step 6: Run tests**

Run: `cargo test -p rocket-surgeon-tui --lib tiling::tests && cargo test -p rocket-surgeon-tui --lib client::subscription::tests`
Expected: all pass.

- [ ] **Step 7: Run full suite with protocol crate change**

Run: `cargo test -p rocket-surgeon-protocol && cargo test -p rocket-surgeon-tui`
Expected: all pass.

- [ ] **Step 8: Run clippy and fmt across both crates**

Run: `cargo fmt && cargo clippy --workspace -- -D warnings`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add crates/rocket-surgeon-protocol/src/messages.rs \
       crates/rocket-surgeon-tui/src/client/subscription.rs \
       crates/rocket-surgeon-tui/src/tiling.rs
git commit -m "fix(tui): remove hardcoded attn heuristic, derive Ord on EventType

propose_layout triggers on any component change (not just attn).
EventType gets Ord derive for stable deterministic sorting.

Addresses CR findings P2 #13 (hardcoded attn), P3 #16 (EventType sort),
P3 #17 (WezTerm — no change needed, behavior is correct)."
```

---

### Task 7: Compositor + tiling cleanup (P3 #18, #19)

**Files:**
- Modify: `crates/rocket-surgeon-tui/src/tiling.rs`
- Modify: `crates/rocket-surgeon-tui/src/render/compositor.rs`
- Modify: `crates/rocket-surgeon-tui/src/client/connection.rs`
- Test: `tiling.rs`, `compositor.rs` test modules

**Findings addressed:**
- **P3 #18**: `tiling::Rect` duplicates `ratatui::layout::Rect` — use ratatui's Rect directly
- **P3 #19**: `RequestId::Number(self.next_id.fetch_add(1, Ordering::Relaxed) as i64)` — u64→i64 cast can overflow. Use wrapping or cap.

- [ ] **Step 1: Replace `tiling::Rect` with `ratatui::layout::Rect`**

In `tiling.rs`:
- Remove the `pub struct Rect { ... }` definition
- Add `use ratatui::layout::Rect;`
- Update all usages of `Rect { x, y, width, height }` — the field names are the same, so this is a drop-in replacement
- Update the `resolve` and `resolve_into` signatures to use `ratatui::layout::Rect`
- Remove the `crate::tiling::Rect` usage in `compositor.rs` — it already uses `ratatui::layout::Rect` for rendering

In `compositor.rs`, the `render_frame` function currently converts between the two Rect types:
```rust
let allocations = layout.resolve(crate::tiling::Rect {
    x: area.x,
    y: area.y,
    width: area.width,
    height: area.height,
});
// ...
let rect = Rect::new(tile_rect.x, tile_rect.y, tile_rect.width, tile_rect.height);
```

After the change, both are `ratatui::layout::Rect`, so this becomes:
```rust
let allocations = layout.resolve(area);
// ...
// tile_rect is already ratatui::layout::Rect, use directly
```

- [ ] **Step 2: Fix i64 cast in connection.rs**

Replace:
```rust
let id = RequestId::Number(self.next_id.fetch_add(1, Ordering::Relaxed) as i64);
```
With:
```rust
let raw = self.next_id.fetch_add(1, Ordering::Relaxed);
let id = RequestId::Number((raw % (i64::MAX as u64)) as i64);
```

This ensures the value stays in positive i64 range. In practice, hitting 2^63 requests is impossible, but the code should not have undefined behavior if it somehow wraps.

- [ ] **Step 3: Update tiling.rs tests**

The tests construct `Rect` directly. Since `ratatui::layout::Rect` uses `Rect::new(x, y, w, h)` constructor (or struct literal — ratatui's Rect has public fields), update the `full_screen()` helper:

```rust
fn full_screen() -> Rect {
    Rect::new(0, 0, 200, 60)
}
```

The test assertions access `.width`, `.height`, `.x`, `.y` — these are the same field names on `ratatui::layout::Rect`, so no changes needed there.

- [ ] **Step 4: Run all TUI tests**

Run: `cargo test -p rocket-surgeon-tui`
Expected: all tests pass.

- [ ] **Step 5: Run clippy and fmt**

Run: `cargo fmt -p rocket-surgeon-tui && cargo clippy -p rocket-surgeon-tui -- -D warnings`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/rocket-surgeon-tui/src/tiling.rs \
       crates/rocket-surgeon-tui/src/render/compositor.rs \
       crates/rocket-surgeon-tui/src/client/connection.rs
git commit -m "fix(tui): use ratatui Rect, cap request ID to i64 range

Remove duplicate tiling::Rect in favor of ratatui::layout::Rect.
Cap atomic request counter to i64::MAX range to prevent overflow.

Addresses CR findings P3 #18 (duplicate Rect), P3 #19 (i64 cast)."
```

---

## Post-completion

After all 7 tasks pass:

Run: `cargo test --workspace`
Expected: all workspace tests pass.

Run: `cargo clippy --workspace -- -D warnings`
Expected: clean.

Create PR: `fix/tui-cr-findings → master`

Then continue to Phase 4.2 (Widget Library) of the TUI implementation plan.
