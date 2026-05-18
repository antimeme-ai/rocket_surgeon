# Chunk A CR Remediation Plan

**Goal:** Fix all code review findings from the retroactive Python + Rust CRs on Chunk A (WU 1.6 + WU 1.7). Each fix follows JSMNTL: test first (red), then implement (green), then lint.

**What's already done (uncommitted):**
- Rust: `path_contains` on `ModuleMatcher::TypeAndName` (adapter.rs)
- Rust: `rank: u32` on `resolve()`, probe_point generation (adapter.rs)
- Rust: i64-before-f64 in `python_to_json_value` (bridge.rs)
- Rust: `model_handle` + `direction` on `HostStepRequest`, `model_handle` on `HostUpdateProbesRequest` (messages.rs)
- Rust: `Default` derive on `StepDirection` (types.rs)
- Rust: `discover_execution_order` + `apply_execution_order` call in `handle_host_attach` (dispatch.rs)
- Rust: test constructors updated, all 297 tests pass, fmt + clippy clean
- Python: `tensor_to_bytes` bf16 → fp16 conversion (bridge.py:193-198)
- Python: `compute_tensor_stats` empty tensor guard (bridge.py:163-174)
- Python: tests for bf16 bytes, empty tensor, nan/inf (test_bridge_stats.py)

**What's left — Python CR findings:**

| ID | Severity | Finding | File:Line | Root cause |
|----|----------|---------|-----------|------------|
| C2 | Critical | `call_counts` dict in `install_capture_hooks` accumulates across forward passes | bridge.py:225,240-241 | Closure captures dict by reference; never cleared between passes |
| C4 | Critical | `Mailbox.wait()` blocks forever — no timeout | mailbox.py:38-41 | `_lock.acquire()` has no timeout arg |
| M5 | Minor | Sentinel hook returns `out` instead of `None` | bridge.py:209 | Returning output re-allocates; `None` means "don't modify" |
| I1 | Important | Capture hook output typed as `torch.Tensor` but could be tuple | bridge.py:236 | PyTorch hooks receive tuples for multi-output modules |
| I3 | Important | `Mailbox.restore()` doesn't re-acquire lock for next cycle | mailbox.py:48-49 | After restore, `wait()` will immediately return `None` unless producer calls `put()` first |

**What's deferred (out of Chunk A scope):**

| ID | Why deferred |
|----|-------------|
| Rust #2 | GQA split_size comment — cosmetic |
| Rust #4 | Stateless dispatch — architectural, belongs to session management WU |
| Rust #6 | TickState hardcoded Forward/Output — not wired to actual stepping yet |
| Rust #8 | CaptureMode::Full — not wired to actual capture yet |
| M1 | metadata/model_config overlap — requires protocol change, separate WU |
| I4 | StopIteration in discover_execution_order — theoretical, defensive |
| I5 | active_probes mutation docs — documentation only |
| I6 | Multi-probe test — testing concern, address below |
| I7 | NaN/Inf test — already addressed by nan_inf test in test_bridge_stats.py |

---

## Analysis of remaining findings

### C2: call_counts accumulation

**Problem:** `call_counts` (bridge.py:225) is a dict created once per `install_capture_hooks` call and closed over by all hook closures. If the same capture hooks stay installed across multiple forward passes, call_index values continue incrementing from the previous pass instead of resetting to 0.

**Impact:** Wrong call_index values in probe point addresses after the first forward pass.

**Fix:** Return `call_counts` alongside handles so the caller can `.clear()` it between passes. This keeps the bridge thin (no implicit state reset) and lets the Rust orchestrator control the lifecycle.

**Alternative considered:** Reset inside `run_forward`. Rejected — `run_forward` and `install_capture_hooks` are decoupled by design; adding coupling would violate the thin-bridge principle.

### C4: mailbox timeout

**Problem:** `Mailbox.wait()` calls `self._lock.acquire()` with no timeout. If the forward thread crashes between hooks, the controller blocks forever.

**Impact:** Deadlock in production.

**Fix:** Add an optional `timeout` parameter to `wait()`. Use `self._lock.acquire(timeout=timeout)` (available on `_thread.lock`). Return a sentinel or raise `TimeoutError` on expiry. The caller (Rust orchestrator via PyO3) can then handle the timeout.

**Note:** `_thread.allocate_lock().acquire()` accepts a `timeout` kwarg since Python 3.2.

### M5: sentinel return value

**Problem:** `lambda _m, _i, out: out` returns the output tensor. PyTorch forward hooks that return `None` mean "don't modify output." Returning `out` creates a no-op that forces PyTorch to check if the output changed, which is wasted work.

**Impact:** Minor performance. Correct semantics.

**Fix:** Change to `lambda _m, _i, _o: None`.

### I1: capture hook output type

**Problem:** The capture hook type signature says `output: torch.Tensor` but PyTorch hooks can receive tuples for multi-output modules (e.g., attention returning `(attn_output, attn_weights)`).

**Impact:** Would crash on modules that return tuples.

**Fix:** Widen the type to `Any`. The hook already just passes output through to the mailbox — no tensor-specific operations are performed on it before the put.

### I3: restore() doesn't re-acquire lock

**Problem:** After `restore()`, the lock is still released (from the prior `put()`). A subsequent `wait()` would succeed immediately and return `None`.

**Impact:** In practice this doesn't bite us because the barrier protocol always goes put → wait → restore, and the producer always calls put() before the consumer calls wait(). The lock is re-acquired by `wait()` itself. But the Mailbox is fragile to misuse.

**Assessment:** This is correct behavior for the current protocol. The mailbox is a single-slot synchronization primitive, not a general-purpose queue. The lock states:
- After `__init__`: lock acquired (blocked)
- After `put`: lock released (unblocked) 
- After `wait`: lock acquired (blocked again)
- After `restore`: value cleared, lock state unchanged (still acquired from wait)

So the cycle is: `put()` releases → `wait()` acquires → back to blocked state. `restore()` just clears the value. This is actually correct. The CR finding was wrong — `restore()` doesn't need to re-acquire because `wait()` already did.

**Decision:** No fix needed. Document the protocol invariants in a comment on `Mailbox`.

---

## Task plan

### Task R1: Fix C2 — return call_counts from install_capture_hooks

**Files:**
- Modify: `python/rocket_surgeon/bridge.py:214-255`
- Modify: `python/tests/test_hooks.py` (update callers)
- Add test: `python/tests/test_hooks.py` (multi-pass call_index test)
- Modify: `python/tests/test_tck_hooks.py` (update callers)

**Steps:**
1. Write test: two forward passes with same hooks, verify call_index resets after manual clear
2. Run test → red (install_capture_hooks returns list, not tuple)
3. Change return type to `tuple[list[Any], dict[str, int]]` — return `(handles, call_counts)`
4. Update all callers (test_hooks.py, test_tck_hooks.py) to unpack tuple
5. Run tests → green
6. Lint

### Task R2: Fix C4 — add timeout to Mailbox.wait()

**Files:**
- Modify: `python/rocket_surgeon/hooks/mailbox.py:38-41`
- Add test: `python/tests/test_mailbox.py`

**Steps:**
1. Write test: `wait(timeout=0.1)` on empty mailbox raises `TimeoutError`
2. Run test → red
3. Add `timeout: float | None = None` param to `wait()`. Use `self._lock.acquire(timeout=timeout)`. If acquire returns False, raise `TimeoutError`.
4. Run tests → green (existing tests pass because default timeout=None means block forever)
5. Lint

### Task R3: Fix M5 + I1 — sentinel return None, capture hook output type

**Files:**
- Modify: `python/rocket_surgeon/bridge.py:209` (sentinel lambda)
- Modify: `python/rocket_surgeon/bridge.py:236` (capture hook signature)
- Add test: verify sentinel hook doesn't modify output

**Steps:**
1. Write test: install sentinel hooks, run forward, verify output unchanged (existing tests cover this implicitly, but add explicit assertion)
2. Change sentinel lambda from `lambda _m, _i, out: out` to `lambda _m, _i, _o: None`
3. Widen capture hook output type from `torch.Tensor` to `Any`
4. Run tests → green
5. Lint

### Task R4: Document I3 — mailbox protocol invariants

**Files:**
- Modify: `python/rocket_surgeon/hooks/mailbox.py`

**Steps:**
1. Add protocol invariant comment to Mailbox class docstring explaining lock state transitions
2. Lint

### Task R5: Commit and push

**Steps:**
1. Run full Python test suite
2. Run full Rust test suite (excluding worker binary)
3. Run ruff + mypy
4. Run cargo fmt + clippy
5. Commit all CR fixes (Rust + Python together)
6. Push
