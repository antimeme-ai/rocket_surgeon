# Chunk A: Model Adapter + Hook Manager — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given a loaded HuggingFace model, map its module tree to canonical probe-point names, register hooks, and implement a lock-based barrier gate for tick-by-tick forward pass stepping.

**Architecture:** Two-layer adapter: core framework in Rust (module mapping, resolution, tick state) + per-family static declarations. Python bridge stays thin — holds PyTorch object refs, calls PyTorch APIs. Lock-based single-slot mailboxes (not threading.Event) for barrier synchronization.

**Tech Stack:** Rust (adapter, tick, capture), Python (bridge, hooks, mailbox), PyO3 0.24 (FFI), PyTorch (hooks, tensor ops), `_thread.allocate_lock()` (barrier primitive)

**Design spec:** `docs/specs/2026-05-18-adapter-hook-manager-design.md`

**JSMNTL discipline:** Each task follows TCK-red-first cycle. Write failing tests, then implement to green, then lint. Subagents MUST be briefed: "JSMNTL methodology is mandatory — write the failing test first, verify it fails, then implement to make it pass."

**Test model:** `hf-internal-testing/tiny-random-LlamaForCausalLM` for all Python tests. This is a tiny model that loads fast on CPU.

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `python/rocket_surgeon/bridge.py` | Renamed from `skin.py`. Thin PyTorch bridge: model loading, module discovery, execution order tracing, tensor stats, fused splitting |
| `python/rocket_surgeon/hooks/mailbox.py` | Lock-based single-slot mailbox (`_thread.allocate_lock()`). Two instances per barrier: result + resume |
| `python/tests/test_mailbox.py` | Mailbox unit tests: put/wait/restore across two threads |
| `python/tests/test_bridge_discovery.py` | Tests for discover_modules, model_config, discover_execution_order |
| `python/tests/test_bridge_stats.py` | Tests for compute_tensor_stats, split_fused_output, tensor_to_bytes |
| `python/tests/test_hooks.py` | Tests for sentinel/capture hook installation, barrier cycle, forward lifecycle |
| `crates/rocket-surgeon-worker/src/bridge.rs` | Renamed from `skin.rs`. PyO3 bindings for all bridge.py functions |
| `crates/rocket-surgeon-worker/src/adapter.rs` | Core adapter framework: types, family declarations, resolution pipeline |
| `crates/rocket-surgeon-worker/src/tick.rs` | Tick state: position tracking, tick_id generation, step counting |
| `crates/rocket-surgeon-worker/src/capture.rs` | Capture policy: probe matching, stats packaging |

### Modified files

| File | Changes |
|------|---------|
| `crates/rocket-surgeon-worker/src/main.rs` | `mod skin` → `mod bridge`, add `mod adapter`, `mod tick`, `mod capture` |
| `crates/rocket-surgeon-worker/src/dispatch.rs` | Add `_host/configure_hooks`, `_host/step`, `_host/update_probes` handlers |
| `crates/rocket-surgeon-protocol/src/messages.rs` | Add internal method constants + request/response types for new host commands; extend `HostAttachResponse` with `model_type` and `component_vocabulary` |
| `python/rocket_surgeon/hooks/__init__.py` | Export hook installation functions |
| `python/tests/test_skin.py` | Rename to `test_bridge.py`, update imports |

### Deleted files

| File | Replaced by |
|------|------------|
| `python/rocket_surgeon/skin.py` | `python/rocket_surgeon/bridge.py` |
| `crates/rocket-surgeon-worker/src/skin.rs` | `crates/rocket-surgeon-worker/src/bridge.rs` |

---

## Task 1: Rename skin → bridge + verify TCK red

Mechanical rename. Establishes the new naming convention from the design spec. Verify TCK feature files exist and describe functionality we haven't built yet.

**Files:**
- Rename: `python/rocket_surgeon/skin.py` → `python/rocket_surgeon/bridge.py`
- Rename: `crates/rocket-surgeon-worker/src/skin.rs` → `crates/rocket-surgeon-worker/src/bridge.rs`
- Modify: `crates/rocket-surgeon-worker/src/main.rs`
- Modify: `crates/rocket-surgeon-worker/src/dispatch.rs`
- Modify: `python/tests/test_skin.py` → `python/tests/test_bridge.py`

- [ ] **Step 1: Verify TCK feature files exist and are red**

Run:
```bash
cat tck/model/adapter.feature | head -5
cat tck/model/hooks.feature | head -5
echo "TCK scenarios exist. adapter.feature has $(grep -c 'Scenario:' tck/model/adapter.feature) scenarios, hooks.feature has $(grep -c 'Scenario:' tck/model/hooks.feature) scenarios."
```

Expected: Both files exist. adapter.feature has 10 scenarios, hooks.feature has 8 scenarios. None are implemented yet — the features they test (component_vocabulary, probe.fired notifications, etc.) don't exist in the codebase.

- [ ] **Step 2: Rename Python skin → bridge**

```bash
git mv python/rocket_surgeon/skin.py python/rocket_surgeon/bridge.py
```

Then update the module docstring in `python/rocket_surgeon/bridge.py`:

```python
"""Minimal Python bridge for PyTorch model operations.

Called from Rust worker via PyO3. No logic, no state management,
no IPC — just the thinnest possible bridge to PyTorch.
"""
```

- [ ] **Step 3: Rename Rust skin → bridge**

```bash
git mv crates/rocket-surgeon-worker/src/skin.rs crates/rocket-surgeon-worker/src/bridge.rs
```

Update all Python imports in `bridge.rs` — change every `py.import("rocket_surgeon.skin")` to `py.import("rocket_surgeon.bridge")`:

```rust
// In load_model:
let bridge = py.import("rocket_surgeon.bridge")?;
let handle = bridge
    .getattr("load_model")?
    .call1((source, device, dtype))?
    .extract::<u64>()?;

// In unload_model:
let bridge = py.import("rocket_surgeon.bridge")?;
bridge.getattr("unload_model")?.call1((handle,))?;

// In model_metadata:
let bridge = py.import("rocket_surgeon.bridge")?;
let result = bridge.getattr("model_metadata")?.call1((handle,))?;
```

- [ ] **Step 4: Update mod declarations and imports**

In `crates/rocket-surgeon-worker/src/main.rs`, change:
```rust
mod bridge;
mod dispatch;
```

In `crates/rocket-surgeon-worker/src/dispatch.rs`, change the import:
```rust
use crate::bridge;
```

And update all references from `skin::` to `bridge::`:
```rust
let handle = match bridge::load_model(&req.model_source, &req.device, dtype_str) {
    Ok(h) => h,
    Err(e) => return internal_error(request.id.clone(), format!("load_model failed: {e}")),
};

let info = match bridge::model_metadata(handle) {
    Ok(i) => i,
    Err(e) => {
        return internal_error(request.id.clone(), format!("model_metadata failed: {e}"));
    }
};
```

And in `handle_host_detach`:
```rust
match bridge::unload_model(req.model_handle) {
```

- [ ] **Step 5: Rename Python test file and update imports**

```bash
git mv python/tests/test_skin.py python/tests/test_bridge.py
```

Update imports in `python/tests/test_bridge.py`:
```python
"""Tests for the Python bridge: load_model, unload_model, model_metadata."""

from __future__ import annotations

import pytest

from rocket_surgeon.bridge import load_model, model_metadata, unload_model
```

- [ ] **Step 6: Run all tests to verify rename is clean**

Run:
```bash
cargo test --workspace --all-targets
pytest python/tests/test_bridge.py -v
```

Expected: All existing tests pass. No references to `skin` remain.

- [ ] **Step 7: Verify no stale references**

```bash
grep -r "rocket_surgeon.skin" --include="*.py" --include="*.rs" python/ crates/ tests/
grep -r "use crate::skin" --include="*.rs" crates/
grep -r "mod skin" --include="*.rs" crates/
```

Expected: No output (no stale references).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: rename skin → bridge (Python + Rust)

Design spec calls this the bridge layer. Mechanical rename,
no behavioral changes."
```

---

## Task 2: Protocol message extensions

Add internal protocol message types for the new host commands that Chunk A introduces. These types are needed by the worker dispatch before we can wire the adapter and hooks.

**Files:**
- Modify: `crates/rocket-surgeon-protocol/src/messages.rs`

- [ ] **Step 1: Write failing test for new method constants**

Add to the existing `tests` module at the bottom of `crates/rocket-surgeon-protocol/src/messages.rs`:

```rust
#[test]
fn internal_configure_hooks_constant() {
    assert_eq!(internal::HOST_CONFIGURE_HOOKS, "_host/configure_hooks");
}

#[test]
fn internal_step_constant() {
    assert_eq!(internal::HOST_STEP, "_host/step");
}

#[test]
fn internal_update_probes_constant() {
    assert_eq!(internal::HOST_UPDATE_PROBES, "_host/update_probes");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p rocket-surgeon-protocol`

Expected: FAIL — `HOST_CONFIGURE_HOOKS`, `HOST_STEP`, `HOST_UPDATE_PROBES` not found.

- [ ] **Step 3: Add method constants to internal module**

In `crates/rocket-surgeon-protocol/src/messages.rs`, extend the `internal` module:

```rust
pub mod internal {
    pub const HOST_ATTACH: &str = "_host/attach";
    pub const HOST_DETACH: &str = "_host/detach";
    pub const HOST_CONFIGURE_HOOKS: &str = "_host/configure_hooks";
    pub const HOST_STEP: &str = "_host/step";
    pub const HOST_UPDATE_PROBES: &str = "_host/update_probes";
}
```

- [ ] **Step 4: Write failing test for new request/response types**

Add to tests module:

```rust
#[test]
fn host_configure_hooks_request_round_trip() {
    let req = HostConfigureHooksRequest {
        model_handle: 1,
        active_probes: vec!["model:0:*:*:0:fwd".to_owned()],
    };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: HostConfigureHooksRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(req, parsed);
}

#[test]
fn host_configure_hooks_response_round_trip() {
    let resp = HostConfigureHooksResponse {
        sentinel_count: 50,
        capture_count: 12,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: HostConfigureHooksResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(resp, parsed);
}

#[test]
fn host_step_request_round_trip() {
    let req = HostStepRequest { count: 1 };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: HostStepRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(req, parsed);
}

#[test]
fn host_step_response_round_trip() {
    let resp = HostStepResponse {
        position: TickPosition {
            tick_id: 42,
            direction: StepDirection::Forward,
            rank: Some(0),
            layer: 3,
            component: "q_proj".to_owned(),
            event: TickEvent::Output,
            replay_of: None,
        },
        capture: None,
        forward_complete: false,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: HostStepResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(resp.position.tick_id, parsed.position.tick_id);
    assert_eq!(resp.forward_complete, parsed.forward_complete);
}

#[test]
fn host_update_probes_round_trip() {
    let req = HostUpdateProbesRequest {
        active_probes: vec!["model:0:3:q_proj:0:fwd".to_owned()],
    };
    let json = serde_json::to_string(&req).unwrap();
    let parsed: HostUpdateProbesRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(req, parsed);
}

#[test]
fn host_attach_response_includes_component_vocabulary() {
    let resp = HostAttachResponse {
        model_handle: 1,
        num_layers: 4,
        num_heads: 4,
        hidden_dim: 32,
        module_tree: vec!["model.layers.0".to_owned()],
        model_type: "llama".to_owned(),
        component_vocabulary: vec!["q_proj".to_owned(), "k_proj".to_owned()],
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: HostAttachResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.model_type, "llama");
    assert_eq!(parsed.component_vocabulary.len(), 2);
}
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `cargo test -p rocket-surgeon-protocol`

Expected: FAIL — types not defined yet.

- [ ] **Step 6: Add request/response structs**

Add after the `HostDetachResponse` struct in `crates/rocket-surgeon-protocol/src/messages.rs`:

```rust
// ---------------------------------------------------------------------------
// _host/configure_hooks (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostConfigureHooksRequest {
    pub model_handle: u64,
    pub active_probes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostConfigureHooksResponse {
    pub sentinel_count: u32,
    pub capture_count: u32,
}

// ---------------------------------------------------------------------------
// _host/step (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostStepRequest {
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostStepResponse {
    pub position: TickPosition,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture: Option<TensorSummary>,
    pub forward_complete: bool,
}

// ---------------------------------------------------------------------------
// _host/update_probes (internal: daemon → orchestrator → worker)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostUpdateProbesRequest {
    pub active_probes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostUpdateProbesResponse {
    pub probes_active: u32,
}
```

Extend `HostAttachResponse` to include adapter info:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostAttachResponse {
    pub model_handle: u64,
    pub num_layers: u32,
    pub num_heads: u32,
    pub hidden_dim: u32,
    pub module_tree: Vec<String>,
    pub model_type: String,
    pub component_vocabulary: Vec<String>,
}
```

- [ ] **Step 7: Fix existing tests that construct HostAttachResponse**

The existing `host_attach_response_round_trip` test in `messages.rs` and the worker's `dispatch.rs` both construct `HostAttachResponse` without the new fields. Update the existing test:

```rust
#[test]
fn host_attach_response_round_trip() {
    let resp = HostAttachResponse {
        model_handle: 1,
        num_layers: 32,
        num_heads: 32,
        hidden_dim: 4096,
        module_tree: vec!["model.embed_tokens".to_owned(), "model.layers.0".to_owned()],
        model_type: "llama".to_owned(),
        component_vocabulary: vec!["q_proj".to_owned()],
    };
    let json = serde_json::to_string(&resp).unwrap();
    let parsed: HostAttachResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(resp, parsed);
}
```

Update the worker's `dispatch.rs` `handle_host_attach` to include the new fields (placeholder values for now — Task 8 wires real adapter resolution):

```rust
let resp = HostAttachResponse {
    model_handle: info.handle,
    num_layers: info.num_layers,
    num_heads: info.num_heads,
    hidden_dim: info.hidden_dim,
    module_tree: info.module_tree,
    model_type: String::new(),
    component_vocabulary: Vec::new(),
};
```

- [ ] **Step 8: Run all tests**

Run: `cargo test --workspace --all-targets`

Expected: All tests pass (both protocol crate and worker crate).

- [ ] **Step 9: Commit**

```bash
git add crates/rocket-surgeon-protocol/src/messages.rs crates/rocket-surgeon-worker/src/dispatch.rs
git commit -m "feat(protocol): add internal message types for configure_hooks, step, update_probes

Extends HostAttachResponse with model_type and component_vocabulary.
Adds HostConfigureHooksRequest/Response, HostStepRequest/Response,
HostUpdateProbesRequest/Response for new worker commands."
```

---

## Task 3: Lock-based Mailbox primitive

The foundation of the barrier mechanism. A single-slot mailbox built on `_thread.allocate_lock()`, mirroring nnsight's `Mediator.Value` pattern. Two mailboxes per barrier (result + resume) enable the ping-pong between the forward thread and the Rust IPC thread.

**Files:**
- Create: `python/rocket_surgeon/hooks/mailbox.py`
- Create: `python/tests/test_mailbox.py`
- Modify: `python/rocket_surgeon/hooks/__init__.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_mailbox.py`:

```python
"""Tests for lock-based single-slot mailbox."""

from __future__ import annotations

import threading
import time

import pytest

from rocket_surgeon.hooks.mailbox import Mailbox


def test_put_then_wait_returns_value() -> None:
    m = Mailbox()
    m.put("hello")
    assert m.wait() == "hello"


def test_wait_blocks_until_put() -> None:
    m = Mailbox()
    result: list[str] = []

    def producer() -> None:
        time.sleep(0.05)
        m.put("from-producer")

    t = threading.Thread(target=producer)
    t.start()
    value = m.wait()
    t.join()
    assert value == "from-producer"


def test_get_returns_stored_value_without_blocking() -> None:
    m = Mailbox()
    m.put(42)
    m.wait()
    assert m.get() == 42


def test_get_returns_none_when_empty() -> None:
    m = Mailbox()
    assert m.get() is None


def test_restore_clears_value() -> None:
    m = Mailbox()
    m.put("data")
    m.wait()
    m.restore()
    assert m.get() is None


def test_ping_pong_two_mailboxes() -> None:
    """Simulate the barrier pattern: forward thread sends result, waits for resume."""
    result_mb = Mailbox()
    resume_mb = Mailbox()
    captured: list[tuple[str, str]] = []

    def forward_thread() -> None:
        result_mb.put("tensor_at_layer_3")
        value = resume_mb.wait()
        resume_mb.restore()
        captured.append(("forward_got", value))

    def rust_thread() -> None:
        value = result_mb.wait()
        result_mb.restore()
        captured.append(("rust_got", value))
        resume_mb.put("continue")

    fwd = threading.Thread(target=forward_thread)
    rust = threading.Thread(target=rust_thread)
    fwd.start()
    rust.start()
    fwd.join(timeout=2.0)
    rust.join(timeout=2.0)

    assert ("rust_got", "tensor_at_layer_3") in captured
    assert ("forward_got", "continue") in captured


def test_multiple_rounds() -> None:
    """Multiple put/wait/restore cycles work correctly."""
    m = Mailbox()
    for i in range(10):
        m.put(i)
        assert m.wait() == i
        m.restore()
        assert m.get() is None


def test_put_overwrites_unconsumed_value() -> None:
    """Second put before wait overwrites the slot."""
    m = Mailbox()
    m.put("first")
    m.put("second")
    assert m.wait() == "second"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest python/tests/test_mailbox.py -v`

Expected: FAIL — `ImportError: cannot import name 'Mailbox' from 'rocket_surgeon.hooks.mailbox'`

- [ ] **Step 3: Implement Mailbox**

Create `python/rocket_surgeon/hooks/mailbox.py`:

```python
"""Lock-based single-slot mailbox for barrier synchronization.

Uses _thread.allocate_lock() — a thin C wrapper around a pthread mutex.
No Python-level bookkeeping, no flag-based race conditions.

Pattern mirrors nnsight's Mediator.Value from
src/nnsight/intervention/interleaver.py.
"""

from __future__ import annotations

from _thread import allocate_lock
from typing import Any


class Mailbox:
    """Single-slot mailbox: one producer, one consumer.

    - put(value): store value, release lock (wakes consumer)
    - wait() -> value: acquire lock (blocks until put), return value
    - get() -> value: non-blocking read of current value
    - restore(): clear value, drop references
    """

    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._lock = allocate_lock()
        self._lock.acquire()
        self._value: Any = None

    def put(self, value: Any) -> None:
        """Store value and release the lock, waking any blocked consumer."""
        self._value = value
        if self._lock.locked():
            self._lock.release()

    def wait(self) -> Any:
        """Block until a value is put, then return it.

        Releases the GIL while blocked (via PyThread_acquire_lock_timed).
        """
        self._lock.acquire()
        return self._value

    def get(self) -> Any:
        """Non-blocking read of the current stored value (or None)."""
        return self._value

    def restore(self) -> None:
        """Clear the stored value and drop references.

        Must be called after consumption to prevent activation memory leaks.
        """
        self._value = None
```

- [ ] **Step 4: Export from hooks package**

Update `python/rocket_surgeon/hooks/__init__.py`:

```python
"""PyTorch hook registration and management."""

from __future__ import annotations

from rocket_surgeon.hooks.mailbox import Mailbox

__all__ = ["Mailbox"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest python/tests/test_mailbox.py -v`

Expected: All 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add python/rocket_surgeon/hooks/mailbox.py python/rocket_surgeon/hooks/__init__.py python/tests/test_mailbox.py
git commit -m "feat(hooks): lock-based single-slot mailbox for barrier synchronization

Uses _thread.allocate_lock() to avoid threading.Event set/clear race.
Mirrors nnsight's Mediator.Value pattern."
```

---

## Task 4: Adapter core types and family declarations

Define the Rust data structures for the adapter framework and the static family declaration tables for llama and gpt2.

**Files:**
- Create: `crates/rocket-surgeon-worker/src/adapter.rs`
- Modify: `crates/rocket-surgeon-worker/src/main.rs`

- [ ] **Step 1: Write failing tests for adapter types**

Create `crates/rocket-surgeon-worker/src/adapter.rs` with just the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_mapping_creation() {
        let m = ModuleMapping::Direct {
            canonical: "q_proj".to_owned(),
        };
        assert!(matches!(m, ModuleMapping::Direct { .. }));
    }

    #[test]
    fn fused_mapping_with_three_components() {
        let m = ModuleMapping::Fused {
            components: vec![
                FusedComponent {
                    canonical: "q_proj".to_owned(),
                    split_dim: -1,
                    split_size: 64,
                },
                FusedComponent {
                    canonical: "k_proj".to_owned(),
                    split_dim: -1,
                    split_size: 64,
                },
                FusedComponent {
                    canonical: "v_proj".to_owned(),
                    split_dim: -1,
                    split_size: 64,
                },
            ],
        };
        if let ModuleMapping::Fused { components } = &m {
            assert_eq!(components.len(), 3);
            assert_eq!(components[0].canonical, "q_proj");
        } else {
            panic!("expected Fused");
        }
    }

    #[test]
    fn llama_family_lookup() {
        let decl = family_declaration("llama");
        assert!(decl.is_some());
        let decl = decl.unwrap();
        assert_eq!(decl.model_types, &["llama", "mistral", "codellama"]);
    }

    #[test]
    fn gpt2_family_lookup() {
        let decl = family_declaration("gpt2");
        assert!(decl.is_some());
    }

    #[test]
    fn unknown_family_returns_none() {
        assert!(family_declaration("unknown_arch_xyz").is_none());
    }

    #[test]
    fn llama_has_q_proj_mapping() {
        let decl = family_declaration("llama").unwrap();
        let found = decl.mappings.iter().any(|(matcher, mapping)| {
            matches!(matcher, ModuleMatcher::TypeAndName { attr_name, .. } if *attr_name == "q_proj")
                && matches!(mapping, ModuleMapping::Direct { canonical } if canonical == "q_proj")
        });
        assert!(found, "llama should have a Direct mapping for q_proj");
    }

    #[test]
    fn gpt2_has_fused_c_attn() {
        let decl = family_declaration("gpt2").unwrap();
        let found = decl.mappings.iter().any(|(matcher, mapping)| {
            matches!(matcher, ModuleMatcher::TypeAndName { attr_name, .. } if *attr_name == "c_attn")
                && matches!(mapping, ModuleMapping::Fused { .. })
        });
        assert!(found, "gpt2 should have a Fused mapping for c_attn");
    }

    #[test]
    fn gpt2_c_attn_has_three_equal_splits() {
        let decl = family_declaration("gpt2").unwrap();
        for (matcher, mapping) in decl.mappings {
            if matches!(matcher, ModuleMatcher::TypeAndName { attr_name, .. } if *attr_name == "c_attn") {
                if let ModuleMapping::Fused { components } = mapping {
                    assert_eq!(components.len(), 3);
                    assert_eq!(components[0].canonical, "q_proj");
                    assert_eq!(components[1].canonical, "k_proj");
                    assert_eq!(components[2].canonical, "v_proj");
                    assert!(components.iter().all(|c| c.split_dim == -1));
                    return;
                }
            }
        }
        panic!("c_attn fused mapping not found");
    }

    #[test]
    fn llama_skip_rotary_emb() {
        let decl = family_declaration("llama").unwrap();
        let found = decl.mappings.iter().any(|(matcher, _)| {
            matches!(matcher, ModuleMatcher::TypeOnly { type_name } if *type_name == "LlamaRotaryEmbedding")
        });
        assert!(found, "llama should have a Skip for LlamaRotaryEmbedding");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Add `mod adapter;` to `crates/rocket-surgeon-worker/src/main.rs` (after `mod bridge;`), then run:

```bash
cargo test -p rocket-surgeon-worker -- adapter
```

Expected: FAIL — types not defined.

- [ ] **Step 3: Implement adapter types**

Add to the top of `crates/rocket-surgeon-worker/src/adapter.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModuleMapping {
    Direct { canonical: String },
    Fused { components: Vec<FusedComponent> },
    Container,
    Skip,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedComponent {
    pub canonical: String,
    pub split_dim: i64,
    pub split_size: usize,
}

#[derive(Debug, Clone)]
pub enum ModuleMatcher {
    TypeAndName {
        type_name: &'static str,
        attr_name: &'static str,
    },
    TypeOnly {
        type_name: &'static str,
    },
}

#[derive(Debug, Clone)]
pub struct FamilyDeclaration {
    pub model_types: &'static [&'static str],
    pub mappings: Vec<(ModuleMatcher, ModuleMapping)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MappedComponent {
    pub module_path: String,
    pub canonical: String,
    pub layer_index: Option<u32>,
    pub call_index: u32,
    pub mapping: ModuleMapping,
    pub probe_point: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentMap {
    pub components: Vec<MappedComponent>,
    pub model_family: String,
    pub vocabulary: Vec<String>,
}
```

- [ ] **Step 4: Implement family declarations**

Add below the types in `adapter.rs`:

```rust
fn llama_declaration() -> FamilyDeclaration {
    use ModuleMapping::*;
    use ModuleMatcher::*;
    FamilyDeclaration {
        model_types: &["llama", "mistral", "codellama"],
        mappings: vec![
            (TypeOnly { type_name: "LlamaAttention" }, Container),
            (TypeOnly { type_name: "LlamaSdpaAttention" }, Container),
            (TypeOnly { type_name: "LlamaFlashAttention2" }, Container),
            (TypeOnly { type_name: "MistralAttention" }, Container),
            (TypeOnly { type_name: "MistralSdpaAttention" }, Container),
            (TypeOnly { type_name: "MistralFlashAttention2" }, Container),
            (TypeOnly { type_name: "LlamaMLP" }, Container),
            (TypeOnly { type_name: "MistralMLP" }, Container),
            (TypeOnly { type_name: "LlamaDecoderLayer" }, Container),
            (TypeOnly { type_name: "MistralDecoderLayer" }, Container),
            (TypeAndName { type_name: "LlamaRMSNorm", attr_name: "input_layernorm" }, Direct { canonical: "ln1".to_owned() }),
            (TypeAndName { type_name: "LlamaRMSNorm", attr_name: "post_attention_layernorm" }, Direct { canonical: "ln2".to_owned() }),
            (TypeAndName { type_name: "LlamaRMSNorm", attr_name: "norm" }, Direct { canonical: "ln_final".to_owned() }),
            (TypeAndName { type_name: "MistralRMSNorm", attr_name: "input_layernorm" }, Direct { canonical: "ln1".to_owned() }),
            (TypeAndName { type_name: "MistralRMSNorm", attr_name: "post_attention_layernorm" }, Direct { canonical: "ln2".to_owned() }),
            (TypeAndName { type_name: "MistralRMSNorm", attr_name: "norm" }, Direct { canonical: "ln_final".to_owned() }),
            (TypeAndName { type_name: "Linear", attr_name: "q_proj" }, Direct { canonical: "q_proj".to_owned() }),
            (TypeAndName { type_name: "Linear", attr_name: "k_proj" }, Direct { canonical: "k_proj".to_owned() }),
            (TypeAndName { type_name: "Linear", attr_name: "v_proj" }, Direct { canonical: "v_proj".to_owned() }),
            (TypeAndName { type_name: "Linear", attr_name: "o_proj" }, Direct { canonical: "o_proj".to_owned() }),
            (TypeAndName { type_name: "Linear", attr_name: "gate_proj" }, Direct { canonical: "gate_proj".to_owned() }),
            (TypeAndName { type_name: "Linear", attr_name: "up_proj" }, Direct { canonical: "up_proj".to_owned() }),
            (TypeAndName { type_name: "Linear", attr_name: "down_proj" }, Direct { canonical: "down_proj".to_owned() }),
            (TypeOnly { type_name: "LlamaRotaryEmbedding" }, Skip),
            (TypeAndName { type_name: "Embedding", attr_name: "embed_tokens" }, Direct { canonical: "embed".to_owned() }),
            (TypeAndName { type_name: "Linear", attr_name: "lm_head" }, Direct { canonical: "lm_head".to_owned() }),
        ],
    }
}

fn gpt2_declaration() -> FamilyDeclaration {
    use ModuleMapping::*;
    use ModuleMatcher::*;
    FamilyDeclaration {
        model_types: &["gpt2"],
        mappings: vec![
            (TypeOnly { type_name: "GPT2Attention" }, Container),
            (TypeOnly { type_name: "GPT2MLP" }, Container),
            (TypeOnly { type_name: "GPT2Block" }, Container),
            (TypeAndName { type_name: "LayerNorm", attr_name: "ln_1" }, Direct { canonical: "ln1".to_owned() }),
            (TypeAndName { type_name: "LayerNorm", attr_name: "ln_2" }, Direct { canonical: "ln2".to_owned() }),
            (TypeAndName { type_name: "LayerNorm", attr_name: "ln_f" }, Direct { canonical: "ln_final".to_owned() }),
            (TypeAndName { type_name: "Conv1D", attr_name: "c_attn" }, Fused {
                components: vec![
                    FusedComponent { canonical: "q_proj".to_owned(), split_dim: -1, split_size: 0 },
                    FusedComponent { canonical: "k_proj".to_owned(), split_dim: -1, split_size: 0 },
                    FusedComponent { canonical: "v_proj".to_owned(), split_dim: -1, split_size: 0 },
                ],
            }),
            (TypeAndName { type_name: "Conv1D", attr_name: "c_proj" }, Direct { canonical: "o_proj".to_owned() }),
            (TypeAndName { type_name: "Conv1D", attr_name: "c_fc" }, Direct { canonical: "up_proj".to_owned() }),
            (TypeAndName { type_name: "Conv1D", attr_name: "c_proj" }, Direct { canonical: "down_proj".to_owned() }),
            (TypeAndName { type_name: "Embedding", attr_name: "wte" }, Direct { canonical: "embed".to_owned() }),
            (TypeAndName { type_name: "Embedding", attr_name: "wpe" }, Direct { canonical: "pos_embed".to_owned() }),
            (TypeAndName { type_name: "Linear", attr_name: "lm_head" }, Direct { canonical: "lm_head".to_owned() }),
            (TypeOnly { type_name: "Dropout" }, Skip),
            (TypeOnly { type_name: "NewGELUActivation" }, Skip),
        ],
    }
}

static FAMILIES: &[fn() -> FamilyDeclaration] = &[llama_declaration, gpt2_declaration];

pub fn family_declaration(model_type: &str) -> Option<FamilyDeclaration> {
    for factory in FAMILIES {
        let decl = factory();
        if decl.model_types.iter().any(|&t| t == model_type) {
            return Some(decl);
        }
    }
    None
}
```

Note: `split_size: 0` for GPT-2's fused c_attn components means "fill from config at resolution time" — the resolution pipeline (Task 5) resolves these to `n_embd` using the model config.

- [ ] **Step 5: Run tests**

```bash
cargo test -p rocket-surgeon-worker -- adapter
```

Expected: All 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/rocket-surgeon-worker/src/adapter.rs crates/rocket-surgeon-worker/src/main.rs
git commit -m "feat(adapter): core types and family declarations for llama + gpt2

ModuleMapping (Direct/Fused/Container/Skip), FusedComponent,
ComponentMap, FamilyDeclaration. Static declaration tables for
llama (+ mistral, codellama) and gpt2 (with fused c_attn)."
```

---

## Task 5: Adapter resolution pipeline

Given a raw module inventory from Python and a model config, resolve modules to canonical names with layer detection and unknown fallback.

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/adapter.rs`

- [ ] **Step 1: Write failing tests for resolution**

Add to the tests module in `adapter.rs`:

```rust
#[test]
fn resolve_llama_modules() {
    let modules = vec![
        RawModule { path: "model".into(), type_name: "LlamaModel".into(), attr_name: "model".into() },
        RawModule { path: "model.embed_tokens".into(), type_name: "Embedding".into(), attr_name: "embed_tokens".into() },
        RawModule { path: "model.layers.0.self_attn".into(), type_name: "LlamaSdpaAttention".into(), attr_name: "self_attn".into() },
        RawModule { path: "model.layers.0.self_attn.q_proj".into(), type_name: "Linear".into(), attr_name: "q_proj".into() },
        RawModule { path: "model.layers.0.self_attn.k_proj".into(), type_name: "Linear".into(), attr_name: "k_proj".into() },
        RawModule { path: "model.layers.0.self_attn.v_proj".into(), type_name: "Linear".into(), attr_name: "v_proj".into() },
        RawModule { path: "model.layers.0.self_attn.o_proj".into(), type_name: "Linear".into(), attr_name: "o_proj".into() },
        RawModule { path: "model.layers.0.input_layernorm".into(), type_name: "LlamaRMSNorm".into(), attr_name: "input_layernorm".into() },
        RawModule { path: "model.layers.0.mlp".into(), type_name: "LlamaMLP".into(), attr_name: "mlp".into() },
        RawModule { path: "model.layers.0.mlp.gate_proj".into(), type_name: "Linear".into(), attr_name: "gate_proj".into() },
        RawModule { path: "model.layers.0.mlp.up_proj".into(), type_name: "Linear".into(), attr_name: "up_proj".into() },
        RawModule { path: "model.layers.0.mlp.down_proj".into(), type_name: "Linear".into(), attr_name: "down_proj".into() },
        RawModule { path: "model.layers.0.post_attention_layernorm".into(), type_name: "LlamaRMSNorm".into(), attr_name: "post_attention_layernorm".into() },
        RawModule { path: "lm_head".into(), type_name: "Linear".into(), attr_name: "lm_head".into() },
    ];
    let config = ModelConfig { model_type: "llama".into(), num_layers: 1, num_heads: 4, hidden_size: 32, num_kv_heads: Some(4) };
    let map = resolve(&modules, &config).unwrap();
    assert_eq!(map.model_family, "llama");

    let canonicals: Vec<&str> = map.components.iter().map(|c| c.canonical.as_str()).collect();
    assert!(canonicals.contains(&"q_proj"));
    assert!(canonicals.contains(&"k_proj"));
    assert!(canonicals.contains(&"v_proj"));
    assert!(canonicals.contains(&"o_proj"));
    assert!(canonicals.contains(&"gate_proj"));
    assert!(canonicals.contains(&"up_proj"));
    assert!(canonicals.contains(&"down_proj"));
    assert!(canonicals.contains(&"ln1"));
    assert!(canonicals.contains(&"ln2"));
    assert!(canonicals.contains(&"embed"));
    assert!(canonicals.contains(&"lm_head"));
}

#[test]
fn resolve_detects_layer_index() {
    let modules = vec![
        RawModule { path: "model.layers.3.self_attn.q_proj".into(), type_name: "Linear".into(), attr_name: "q_proj".into() },
    ];
    let config = ModelConfig { model_type: "llama".into(), num_layers: 4, num_heads: 4, hidden_size: 32, num_kv_heads: Some(4) };
    let map = resolve(&modules, &config).unwrap();
    let q = map.components.iter().find(|c| c.canonical == "q_proj").unwrap();
    assert_eq!(q.layer_index, Some(3));
}

#[test]
fn resolve_unknown_module_gets_raw_fallback() {
    let modules = vec![
        RawModule { path: "model.weird_thing".into(), type_name: "UnknownModule".into(), attr_name: "weird_thing".into() },
    ];
    let config = ModelConfig { model_type: "llama".into(), num_layers: 1, num_heads: 4, hidden_size: 32, num_kv_heads: Some(4) };
    let map = resolve(&modules, &config).unwrap();
    let raw = map.components.iter().find(|c| c.canonical.starts_with("_raw.")).unwrap();
    assert_eq!(raw.canonical, "_raw.model.weird_thing");
}

#[test]
fn resolve_unsupported_family_returns_error() {
    let modules = vec![];
    let config = ModelConfig { model_type: "unknown_arch".into(), num_layers: 1, num_heads: 4, hidden_size: 32, num_kv_heads: None };
    let result = resolve(&modules, &config);
    assert!(result.is_err());
}

#[test]
fn resolve_builds_vocabulary() {
    let modules = vec![
        RawModule { path: "model.layers.0.self_attn.q_proj".into(), type_name: "Linear".into(), attr_name: "q_proj".into() },
        RawModule { path: "model.layers.0.self_attn.k_proj".into(), type_name: "Linear".into(), attr_name: "k_proj".into() },
    ];
    let config = ModelConfig { model_type: "llama".into(), num_layers: 1, num_heads: 4, hidden_size: 32, num_kv_heads: Some(4) };
    let map = resolve(&modules, &config).unwrap();
    assert!(map.vocabulary.contains(&"q_proj".to_owned()));
    assert!(map.vocabulary.contains(&"k_proj".to_owned()));
}

#[test]
fn apply_execution_order_reorders_components() {
    let mut map = ComponentMap {
        components: vec![
            MappedComponent { module_path: "a".into(), canonical: "first".into(), layer_index: Some(0), call_index: 0, mapping: ModuleMapping::Direct { canonical: "first".into() }, probe_point: String::new() },
            MappedComponent { module_path: "b".into(), canonical: "second".into(), layer_index: Some(0), call_index: 0, mapping: ModuleMapping::Direct { canonical: "second".into() }, probe_point: String::new() },
        ],
        model_family: "test".into(),
        vocabulary: vec![],
    };
    let execution_order = vec![
        ("b".to_owned(), 0u32),
        ("a".to_owned(), 0u32),
    ];
    apply_execution_order(&mut map, &execution_order);
    assert_eq!(map.components[0].module_path, "b");
    assert_eq!(map.components[1].module_path, "a");
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p rocket-surgeon-worker -- adapter
```

Expected: FAIL — `RawModule`, `ModelConfig`, `resolve`, `apply_execution_order` not defined.

- [ ] **Step 3: Implement RawModule and ModelConfig types**

Add to `adapter.rs` (above the family declarations):

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawModule {
    pub path: String,
    pub type_name: String,
    pub attr_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub num_layers: u32,
    pub num_heads: u32,
    pub hidden_size: u32,
    pub num_kv_heads: Option<u32>,
}
```

- [ ] **Step 4: Implement resolve()**

Add to `adapter.rs`:

```rust
use std::collections::HashSet;
use std::num::ParseIntError;

fn extract_layer_index(path: &str) -> Option<u32> {
    for segment in path.split('.') {
        if let Ok(idx) = segment.parse::<u32>() {
            return Some(idx);
        }
    }
    None
}

fn matches_module(matcher: &ModuleMatcher, module: &RawModule) -> bool {
    match matcher {
        ModuleMatcher::TypeOnly { type_name } => module.type_name == *type_name,
        ModuleMatcher::TypeAndName { type_name, attr_name } => {
            module.type_name == *type_name && module.attr_name == *attr_name
        }
    }
}

pub fn resolve(
    modules: &[RawModule],
    config: &ModelConfig,
) -> Result<ComponentMap, String> {
    let decl = family_declaration(&config.model_type)
        .ok_or_else(|| format!("unsupported model family: {}", config.model_type))?;

    let family_name = config.model_type.clone();
    let mut components = Vec::new();
    let mut vocabulary = HashSet::new();

    for module in modules {
        let mut matched = false;

        for (matcher, mapping) in &decl.mappings {
            if matches_module(matcher, module) {
                matched = true;
                match mapping {
                    ModuleMapping::Skip | ModuleMapping::Container => break,
                    ModuleMapping::Direct { canonical } => {
                        let layer_index = extract_layer_index(&module.path);
                        vocabulary.insert(canonical.clone());
                        components.push(MappedComponent {
                            module_path: module.path.clone(),
                            canonical: canonical.clone(),
                            layer_index,
                            call_index: 0,
                            mapping: mapping.clone(),
                            probe_point: String::new(),
                        });
                        break;
                    }
                    ModuleMapping::Fused { components: fused } => {
                        let layer_index = extract_layer_index(&module.path);
                        let mut resolved_fused = fused.clone();
                        for fc in &mut resolved_fused {
                            if fc.split_size == 0 {
                                fc.split_size = config.hidden_size as usize;
                            }
                            vocabulary.insert(fc.canonical.clone());
                        }
                        components.push(MappedComponent {
                            module_path: module.path.clone(),
                            canonical: format!("_fused.{}", module.attr_name),
                            layer_index,
                            call_index: 0,
                            mapping: ModuleMapping::Fused { components: resolved_fused },
                            probe_point: String::new(),
                        });
                        break;
                    }
                }
            }
        }

        if !matched {
            let canonical = format!("_raw.{}", module.path);
            vocabulary.insert(canonical.clone());
            components.push(MappedComponent {
                module_path: module.path.clone(),
                canonical,
                layer_index: extract_layer_index(&module.path),
                call_index: 0,
                mapping: ModuleMapping::Direct {
                    canonical: format!("_raw.{}", module.path),
                },
                probe_point: String::new(),
            });
        }
    }

    let mut vocab_sorted: Vec<String> = vocabulary.into_iter().collect();
    vocab_sorted.sort();

    Ok(ComponentMap {
        components,
        model_family: family_name,
        vocabulary: vocab_sorted,
    })
}

pub fn apply_execution_order(
    map: &mut ComponentMap,
    execution_order: &[(String, u32)],
) {
    let order_map: std::collections::HashMap<(&str, u32), usize> = execution_order
        .iter()
        .enumerate()
        .map(|(i, (path, ci))| ((path.as_str(), *ci), i))
        .collect();

    map.components.sort_by_key(|c| {
        order_map
            .get(&(c.module_path.as_str(), c.call_index))
            .copied()
            .unwrap_or(usize::MAX)
    });

    for (i, (path, call_index)) in execution_order.iter().enumerate() {
        if let Some(comp) = map.components.iter_mut().find(|c| c.module_path == *path && c.call_index == *call_index) {
            comp.call_index = *call_index;
        }
    }
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p rocket-surgeon-worker -- adapter
```

Expected: All adapter tests PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/rocket-surgeon-worker/src/adapter.rs
git commit -m "feat(adapter): resolution pipeline — module matching, layer detection, unknown fallback

resolve() maps raw module inventory to canonical names using family
declarations. apply_execution_order() reorders ComponentMap to match
actual forward-pass hook firing order."
```

---

## Task 6: Bridge Python — module discovery functions

Add `discover_modules`, `model_config`, and `discover_execution_order` to the Python bridge. These are the Python-side functions that feed data to the Rust adapter.

**Files:**
- Modify: `python/rocket_surgeon/bridge.py`
- Create: `python/tests/test_bridge_discovery.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_bridge_discovery.py`:

```python
"""Tests for bridge discovery functions: discover_modules, model_config, discover_execution_order."""

from __future__ import annotations

import pytest

from rocket_surgeon.bridge import (
    discover_execution_order,
    discover_modules,
    load_model,
    model_config,
    unload_model,
)

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture
def model_handle() -> int:
    handle = load_model(source=TINY_MODEL, device="cpu", dtype="float32")
    yield handle
    unload_model(handle)


def test_discover_modules_returns_list_of_dicts(model_handle: int) -> None:
    modules = discover_modules(model_handle)
    assert isinstance(modules, list)
    assert len(modules) > 0
    first = modules[0]
    assert "path" in first
    assert "type_name" in first
    assert "attr_name" in first
    assert isinstance(first["path"], str)
    assert isinstance(first["type_name"], str)
    assert isinstance(first["attr_name"], str)


def test_discover_modules_includes_linear_layers(model_handle: int) -> None:
    modules = discover_modules(model_handle)
    type_names = {m["type_name"] for m in modules}
    assert "Linear" in type_names


def test_discover_modules_includes_layer_structure(model_handle: int) -> None:
    modules = discover_modules(model_handle)
    paths = {m["path"] for m in modules}
    has_layer_path = any("layers.0" in p for p in paths)
    assert has_layer_path, f"Expected layer paths in {paths}"


def test_model_config_returns_expected_keys(model_handle: int) -> None:
    config = model_config(model_handle)
    assert "model_type" in config
    assert "num_layers" in config
    assert "num_heads" in config
    assert "hidden_size" in config
    assert isinstance(config["model_type"], str)
    assert isinstance(config["num_layers"], int)


def test_model_config_llama_model_type(model_handle: int) -> None:
    config = model_config(model_handle)
    assert config["model_type"] == "llama"


def test_discover_execution_order_returns_list_of_tuples(model_handle: int) -> None:
    order = discover_execution_order(model_handle)
    assert isinstance(order, list)
    assert len(order) > 0
    first = order[0]
    assert isinstance(first, tuple)
    assert len(first) == 2
    assert isinstance(first[0], str)
    assert isinstance(first[1], int)


def test_discover_execution_order_consistent(model_handle: int) -> None:
    order1 = discover_execution_order(model_handle)
    order2 = discover_execution_order(model_handle)
    assert order1 == order2


def test_discover_execution_order_call_index_zero_for_simple_model(model_handle: int) -> None:
    order = discover_execution_order(model_handle)
    for path, call_index in order:
        assert call_index == 0, f"Expected call_index 0 for {path}, got {call_index}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest python/tests/test_bridge_discovery.py -v`

Expected: FAIL — `ImportError: cannot import name 'discover_modules' from 'rocket_surgeon.bridge'`

- [ ] **Step 3: Implement discover_modules**

Add to `python/rocket_surgeon/bridge.py`:

```python
def discover_modules(handle: int) -> list[dict[str, Any]]:
    """Walk model.named_modules() and return module inventory.

    Each entry: {path, type_name, attr_name}.
    attr_name is the last segment of the path (the attribute name on the parent).
    """
    model = _models[handle]
    result = []
    for name, module in model.named_modules():
        if not name:
            continue
        type_name = type(module).__name__
        attr_name = name.rsplit(".", 1)[-1]
        result.append({
            "path": name,
            "type_name": type_name,
            "attr_name": attr_name,
        })
    return result
```

- [ ] **Step 4: Implement model_config**

Add to `python/rocket_surgeon/bridge.py`:

```python
def model_config(handle: int) -> dict[str, Any]:
    """Extract model configuration attributes.

    Returns: {model_type, num_layers, num_heads, hidden_size, num_kv_heads}.
    """
    model = _models[handle]
    config = model.config
    return {
        "model_type": getattr(config, "model_type", "unknown"),
        "num_layers": getattr(config, "num_hidden_layers", 0),
        "num_heads": getattr(config, "num_attention_heads", 0),
        "hidden_size": getattr(config, "hidden_size", 0),
        "num_kv_heads": getattr(config, "num_key_value_heads", None),
    }
```

- [ ] **Step 5: Implement discover_execution_order**

Add to `python/rocket_surgeon/bridge.py`:

```python
def discover_execution_order(handle: int) -> list[tuple[str, int]]:
    """Run a tracing forward pass and record hook firing order.

    Returns ordered list of (module_path, call_index) pairs.
    call_index tracks how many times the same module fires (0-based).
    """
    model = _models[handle]
    call_counts: dict[str, int] = {}
    order: list[tuple[str, int]] = []
    handles: list[Any] = []

    def make_hook(path: str) -> Any:
        def hook(module: Any, input: Any, output: Any) -> None:
            idx = call_counts.get(path, 0)
            call_counts[path] = idx + 1
            order.append((path, idx))
        return hook

    for name, module in model.named_modules():
        if not name:
            continue
        h = module.register_forward_hook(make_hook(name))
        handles.append(h)

    with torch.inference_mode():
        dummy_input = torch.zeros(1, 2, dtype=torch.long, device=next(model.parameters()).device)
        model(dummy_input)

    for h in handles:
        h.remove()

    return order
```

- [ ] **Step 6: Run tests**

Run: `pytest python/tests/test_bridge_discovery.py -v`

Expected: All 9 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add python/rocket_surgeon/bridge.py python/tests/test_bridge_discovery.py
git commit -m "feat(bridge): discover_modules, model_config, discover_execution_order

discover_modules returns module inventory with path/type_name/attr_name.
model_config extracts model_type/num_layers/num_heads/hidden_size.
discover_execution_order traces a forward pass to record (path, call_index)
hook firing order."
```

---

## Task 7: Bridge Python — compute_tensor_stats, split_fused_output, tensor_to_bytes

Stats computation with fp32 cast for autocast-aware correctness, fused output splitting, and full tensor serialization.

**Files:**
- Modify: `python/rocket_surgeon/bridge.py`
- Create: `python/tests/test_bridge_stats.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_bridge_stats.py`:

```python
"""Tests for bridge tensor operations: compute_tensor_stats, split_fused_output, tensor_to_bytes."""

from __future__ import annotations

import math

import pytest
import torch

from rocket_surgeon.bridge import compute_tensor_stats, split_fused_output, tensor_to_bytes


def test_compute_tensor_stats_returns_expected_keys() -> None:
    t = torch.randn(4, 8)
    stats = compute_tensor_stats(t)
    expected_keys = {"mean", "std", "min", "max", "abs_max", "l2_norm", "sparsity", "shape", "dtype"}
    assert expected_keys.issubset(stats.keys())


def test_compute_tensor_stats_correct_values() -> None:
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    stats = compute_tensor_stats(t)
    assert math.isclose(stats["mean"], 2.5, rel_tol=1e-5)
    assert math.isclose(stats["min"], 1.0)
    assert math.isclose(stats["max"], 4.0)
    assert math.isclose(stats["abs_max"], 4.0)
    assert stats["shape"] == [4]
    assert stats["dtype"] == "float32"


def test_compute_tensor_stats_fp16_uses_fp32_for_reduction() -> None:
    t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
    stats = compute_tensor_stats(t)
    expected_mean = torch.tensor([1.0, 2.0, 3.0, 4.0]).float().mean().item()
    assert math.isclose(stats["mean"], expected_mean, rel_tol=1e-3)
    assert stats["dtype"] == "float16"


def test_compute_tensor_stats_bf16_uses_fp32_for_reduction() -> None:
    t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    stats = compute_tensor_stats(t)
    expected_mean = torch.tensor([1.0, 2.0, 3.0, 4.0]).float().mean().item()
    assert math.isclose(stats["mean"], expected_mean, rel_tol=1e-2)
    assert stats["dtype"] == "bfloat16"


def test_compute_tensor_stats_sparsity() -> None:
    t = torch.tensor([0.0, 1.0, 0.0, 2.0])
    stats = compute_tensor_stats(t)
    assert math.isclose(stats["sparsity"], 0.5, rel_tol=1e-5)


def test_compute_tensor_stats_population_std() -> None:
    t = torch.tensor([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    stats = compute_tensor_stats(t)
    expected_std = t.float().std(correction=0).item()
    assert math.isclose(stats["std"], expected_std, rel_tol=1e-5)


def test_split_fused_output_equal_chunks() -> None:
    t = torch.randn(2, 3, 12)
    parts = split_fused_output(t, dim=-1, sizes=[4, 4, 4])
    assert len(parts) == 3
    assert parts[0].shape == (2, 3, 4)
    assert parts[1].shape == (2, 3, 4)
    assert parts[2].shape == (2, 3, 4)
    assert torch.allclose(torch.cat(parts, dim=-1), t)


def test_split_fused_output_unequal_chunks() -> None:
    t = torch.randn(2, 3, 10)
    parts = split_fused_output(t, dim=-1, sizes=[6, 2, 2])
    assert len(parts) == 3
    assert parts[0].shape == (2, 3, 6)
    assert parts[1].shape == (2, 3, 2)
    assert parts[2].shape == (2, 3, 2)


def test_tensor_to_bytes_roundtrip() -> None:
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    data = tensor_to_bytes(t)
    assert isinstance(data, bytes)
    assert len(data) == 3 * 4
    reconstructed = torch.frombuffer(bytearray(data), dtype=torch.float32)
    assert torch.allclose(t, reconstructed)


def test_tensor_to_bytes_preserves_dtype() -> None:
    t = torch.tensor([1.0, 2.0], dtype=torch.float16)
    data = tensor_to_bytes(t)
    assert len(data) == 2 * 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest python/tests/test_bridge_stats.py -v`

Expected: FAIL — `ImportError: cannot import name 'compute_tensor_stats'`

- [ ] **Step 3: Implement compute_tensor_stats**

Add to `python/rocket_surgeon/bridge.py`:

```python
_DTYPE_NAME_MAP: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def compute_tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    """Compute summary stats on a tensor, casting to fp32 for reduction accuracy.

    Returns: {mean, std, min, max, abs_max, l2_norm, sparsity, shape, dtype}.
    The dtype field reports the ORIGINAL tensor dtype, not fp32.
    """
    original_dtype = tensor.dtype
    t = tensor.detach().float()
    numel = t.numel()
    return {
        "mean": t.mean().item(),
        "std": t.std(correction=0).item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "abs_max": t.abs().max().item(),
        "l2_norm": t.norm(2).item(),
        "sparsity": (t == 0).sum().item() / numel if numel > 0 else 0.0,
        "shape": list(tensor.shape),
        "dtype": _DTYPE_NAME_MAP.get(original_dtype, str(original_dtype)),
    }
```

- [ ] **Step 4: Implement split_fused_output**

Add to `python/rocket_surgeon/bridge.py`:

```python
def split_fused_output(tensor: torch.Tensor, dim: int, sizes: list[int]) -> list[torch.Tensor]:
    """Split a fused module output tensor along the given dimension.

    Handles both equal and unequal split sizes (e.g., Phi-3 GQA).
    """
    return list(tensor.split(sizes, dim=dim))
```

- [ ] **Step 5: Implement tensor_to_bytes**

Add to `python/rocket_surgeon/bridge.py`:

```python
def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize tensor to raw bytes. Dtype-preserving — an fp16 tensor stays fp16."""
    return tensor.detach().contiguous().cpu().numpy().tobytes()
```

- [ ] **Step 6: Run tests**

Run: `pytest python/tests/test_bridge_stats.py -v`

Expected: All 11 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add python/rocket_surgeon/bridge.py python/tests/test_bridge_stats.py
git commit -m "feat(bridge): compute_tensor_stats, split_fused_output, tensor_to_bytes

Stats cast to fp32 before reduction (autocast contract). split_fused_output
handles equal and unequal chunk sizes. tensor_to_bytes is dtype-preserving."
```

---

## Task 8: Bridge Rust — PyO3 bindings for new bridge functions

Extend `bridge.rs` with PyO3 bindings that call the new Python bridge functions. These are used by the Rust adapter resolution pipeline and worker dispatch.

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/bridge.rs`

- [ ] **Step 1: Write failing test for model_config binding**

Add a test module to `bridge.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_config_fields() {
        let cfg = ModelConfig {
            model_type: "llama".to_owned(),
            num_layers: 4,
            num_heads: 4,
            hidden_size: 32,
            num_kv_heads: Some(4),
        };
        assert_eq!(cfg.model_type, "llama");
        assert_eq!(cfg.num_layers, 4);
    }
}
```

(Note: full integration tests of PyO3 calls require Python and run in the e2e suite. Here we test that the Rust types compile correctly.)

- [ ] **Step 2: Add ModelConfig and RawModule re-exports to bridge.rs**

Update the imports in `bridge.rs`:

```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

pub use crate::adapter::{ModelConfig, RawModule};
```

- [ ] **Step 3: Implement model_config binding**

Add to `bridge.rs`:

```rust
pub fn model_config(handle: u64) -> anyhow::Result<ModelConfig> {
    Python::with_gil(|py| {
        let bridge = py.import("rocket_surgeon.bridge")?;
        let result = bridge.getattr("model_config")?.call1((handle,))?;
        let dict = result
            .downcast::<PyDict>()
            .map_err(|e| anyhow::anyhow!("expected dict from model_config, got: {e}"))?;

        let model_type: String = dict
            .get_item("model_type")?
            .ok_or_else(|| anyhow::anyhow!("missing model_type"))?
            .extract()?;
        let num_layers: u32 = dict
            .get_item("num_layers")?
            .ok_or_else(|| anyhow::anyhow!("missing num_layers"))?
            .extract()?;
        let num_heads: u32 = dict
            .get_item("num_heads")?
            .ok_or_else(|| anyhow::anyhow!("missing num_heads"))?
            .extract()?;
        let hidden_size: u32 = dict
            .get_item("hidden_size")?
            .ok_or_else(|| anyhow::anyhow!("missing hidden_size"))?
            .extract()?;
        let num_kv_heads: Option<u32> = dict
            .get_item("num_kv_heads")?
            .and_then(|v| v.extract().ok());

        Ok(ModelConfig {
            model_type,
            num_layers,
            num_heads,
            hidden_size,
            num_kv_heads,
        })
    })
}
```

- [ ] **Step 4: Implement discover_modules binding**

Add to `bridge.rs`:

```rust
pub fn discover_modules(handle: u64) -> anyhow::Result<Vec<RawModule>> {
    Python::with_gil(|py| {
        let bridge = py.import("rocket_surgeon.bridge")?;
        let result = bridge.getattr("discover_modules")?.call1((handle,))?;
        let list = result
            .downcast::<PyList>()
            .map_err(|e| anyhow::anyhow!("expected list from discover_modules, got: {e}"))?;

        let mut modules = Vec::with_capacity(list.len());
        for item in list.iter() {
            let dict = item
                .downcast::<PyDict>()
                .map_err(|e| anyhow::anyhow!("expected dict in modules list, got: {e}"))?;
            let path: String = dict
                .get_item("path")?
                .ok_or_else(|| anyhow::anyhow!("missing path"))?
                .extract()?;
            let type_name: String = dict
                .get_item("type_name")?
                .ok_or_else(|| anyhow::anyhow!("missing type_name"))?
                .extract()?;
            let attr_name: String = dict
                .get_item("attr_name")?
                .ok_or_else(|| anyhow::anyhow!("missing attr_name"))?
                .extract()?;
            modules.push(RawModule {
                path,
                type_name,
                attr_name,
            });
        }
        Ok(modules)
    })
}
```

- [ ] **Step 5: Implement discover_execution_order binding**

Add to `bridge.rs`:

```rust
pub fn discover_execution_order(handle: u64) -> anyhow::Result<Vec<(String, u32)>> {
    Python::with_gil(|py| {
        let bridge = py.import("rocket_surgeon.bridge")?;
        let result = bridge
            .getattr("discover_execution_order")?
            .call1((handle,))?;
        let list = result
            .downcast::<PyList>()
            .map_err(|e| anyhow::anyhow!("expected list, got: {e}"))?;

        let mut order = Vec::with_capacity(list.len());
        for item in list.iter() {
            let tuple = item
                .downcast::<PyTuple>()
                .map_err(|e| anyhow::anyhow!("expected tuple, got: {e}"))?;
            let path: String = tuple.get_item(0)?.extract()?;
            let call_index: u32 = tuple.get_item(1)?.extract()?;
            order.push((path, call_index));
        }
        Ok(order)
    })
}
```

- [ ] **Step 6: Implement compute_tensor_stats and tensor_to_bytes bindings**

Add to `bridge.rs`:

```rust
pub fn compute_tensor_stats(py: Python<'_>, tensor: &Bound<'_, pyo3::PyAny>) -> anyhow::Result<std::collections::HashMap<String, serde_json::Value>> {
    let bridge = py.import("rocket_surgeon.bridge")?;
    let result = bridge.getattr("compute_tensor_stats")?.call1((tensor,))?;
    let dict = result
        .downcast::<PyDict>()
        .map_err(|e| anyhow::anyhow!("expected dict from compute_tensor_stats, got: {e}"))?;

    let mut stats = std::collections::HashMap::new();
    for (key, value) in dict.iter() {
        let k: String = key.extract()?;
        let v = python_to_json_value(&value)?;
        stats.insert(k, v);
    }
    Ok(stats)
}

fn python_to_json_value(obj: &Bound<'_, pyo3::PyAny>) -> anyhow::Result<serde_json::Value> {
    if let Ok(v) = obj.extract::<f64>() {
        Ok(serde_json::Value::from(v))
    } else if let Ok(v) = obj.extract::<i64>() {
        Ok(serde_json::Value::from(v))
    } else if let Ok(v) = obj.extract::<String>() {
        Ok(serde_json::Value::from(v))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let items: Vec<serde_json::Value> = list
            .iter()
            .map(|item| python_to_json_value(&item))
            .collect::<anyhow::Result<_>>()?;
        Ok(serde_json::Value::Array(items))
    } else {
        Ok(serde_json::Value::String(obj.str()?.to_string()))
    }
}

pub fn tensor_to_bytes(py: Python<'_>, tensor: &Bound<'_, pyo3::PyAny>) -> anyhow::Result<Vec<u8>> {
    let bridge = py.import("rocket_surgeon.bridge")?;
    let result = bridge.getattr("tensor_to_bytes")?.call1((tensor,))?;
    let bytes: Vec<u8> = result.extract()?;
    Ok(bytes)
}

pub fn split_fused_output(
    py: Python<'_>,
    tensor: &Bound<'_, pyo3::PyAny>,
    dim: i64,
    sizes: &[usize],
) -> anyhow::Result<Vec<Bound<'_, pyo3::PyAny>>> {
    let bridge = py.import("rocket_surgeon.bridge")?;
    let py_sizes = pyo3::types::PyList::new(py, sizes.iter().map(|&s| s as i64))?;
    let result = bridge
        .getattr("split_fused_output")?
        .call1((tensor, dim, py_sizes))?;
    let list = result
        .downcast::<PyList>()
        .map_err(|e| anyhow::anyhow!("expected list, got: {e}"))?;
    Ok(list.iter().collect())
}
```

- [ ] **Step 7: Run cargo check**

```bash
cargo check -p rocket-surgeon-worker
```

Expected: Compiles successfully.

- [ ] **Step 8: Run existing tests to verify nothing broke**

```bash
cargo test --workspace --all-targets --exclude rocket-surgeon-worker
```

Expected: All tests pass (worker tests that call Python may need the Python environment, which is tested separately).

- [ ] **Step 9: Commit**

```bash
git add crates/rocket-surgeon-worker/src/bridge.rs
git commit -m "feat(bridge): PyO3 bindings for discover_modules, model_config, execution_order, stats

Bindings call through to Python bridge functions. model_config returns
ModelConfig struct. discover_execution_order returns (path, call_index) pairs.
compute_tensor_stats converts Python dict to HashMap<String, Value>."
```

---

## Task 9: Tick state management

Rust module for tick position tracking, tick_id generation, and step counting.

**Files:**
- Create: `crates/rocket-surgeon-worker/src/tick.rs`
- Modify: `crates/rocket-surgeon-worker/src/main.rs`

- [ ] **Step 1: Write failing tests**

Create `crates/rocket-surgeon-worker/src/tick.rs` with tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tick_state_starts_at_zero() {
        let state = TickState::new(0);
        assert_eq!(state.tick_id(), 0);
        assert_eq!(state.layer(), 0);
        assert_eq!(state.component(), "");
        assert_eq!(state.call_index(), 0);
    }

    #[test]
    fn advance_increments_tick_id() {
        let mut state = TickState::new(0);
        state.advance("q_proj", 0, 0);
        assert_eq!(state.tick_id(), 1);
        state.advance("k_proj", 0, 0);
        assert_eq!(state.tick_id(), 2);
    }

    #[test]
    fn advance_updates_position() {
        let mut state = TickState::new(0);
        state.advance("q_proj", 3, 0);
        assert_eq!(state.layer(), 3);
        assert_eq!(state.component(), "q_proj");
        assert_eq!(state.call_index(), 0);
    }

    #[test]
    fn advance_tracks_call_index() {
        let mut state = TickState::new(0);
        state.advance("embed", 0, 0);
        assert_eq!(state.tick_id(), 1);
        state.advance("embed", 0, 1);
        assert_eq!(state.tick_id(), 2);
        assert_eq!(state.call_index(), 1);
    }

    #[test]
    fn tick_id_is_monotonic() {
        let mut state = TickState::new(0);
        let mut prev = state.tick_id();
        for i in 0..100 {
            state.advance("comp", i % 4, 0);
            assert!(state.tick_id() > prev);
            prev = state.tick_id();
        }
    }

    #[test]
    fn to_tick_position() {
        let mut state = TickState::new(0);
        state.advance("q_proj", 5, 0);
        let pos = state.to_tick_position();
        assert_eq!(pos.tick_id, 1);
        assert_eq!(pos.layer, 5);
        assert_eq!(pos.component, "q_proj");
        assert_eq!(pos.rank, Some(0));
    }

    #[test]
    fn step_count_tracks_total_steps() {
        let mut state = TickState::new(0);
        assert_eq!(state.step_count(), 0);
        state.advance("a", 0, 0);
        assert_eq!(state.step_count(), 1);
        state.advance("b", 0, 0);
        assert_eq!(state.step_count(), 2);
    }
}
```

- [ ] **Step 2: Add mod declaration and run tests to verify they fail**

Add `mod tick;` to `crates/rocket-surgeon-worker/src/main.rs`.

```bash
cargo test -p rocket-surgeon-worker -- tick
```

Expected: FAIL — `TickState` not defined.

- [ ] **Step 3: Implement TickState**

Add above the tests in `tick.rs`:

```rust
use rocket_surgeon_protocol::types::{StepDirection, TickEvent, TickPosition};

pub struct TickState {
    tick_id: u64,
    rank: u32,
    layer: u32,
    component: String,
    call_index: u32,
    step_count: u64,
}

impl TickState {
    pub fn new(rank: u32) -> Self {
        Self {
            tick_id: 0,
            rank,
            layer: 0,
            component: String::new(),
            call_index: 0,
            step_count: 0,
        }
    }

    pub fn advance(&mut self, component: &str, layer: u32, call_index: u32) {
        self.tick_id += 1;
        self.layer = layer;
        self.component = component.to_owned();
        self.call_index = call_index;
        self.step_count += 1;
    }

    pub fn tick_id(&self) -> u64 {
        self.tick_id
    }

    pub fn layer(&self) -> u32 {
        self.layer
    }

    pub fn component(&self) -> &str {
        &self.component
    }

    pub fn call_index(&self) -> u32 {
        self.call_index
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    pub fn to_tick_position(&self) -> TickPosition {
        TickPosition {
            tick_id: self.tick_id,
            direction: StepDirection::Forward,
            rank: Some(self.rank),
            layer: self.layer,
            component: self.component.clone(),
            event: TickEvent::Output,
            replay_of: None,
        }
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p rocket-surgeon-worker -- tick
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/rocket-surgeon-worker/src/tick.rs crates/rocket-surgeon-worker/src/main.rs
git commit -m "feat(tick): tick state management — position tracking, tick_id generation

TickState tracks current layer, component, call_index, and produces
monotonic tick_ids. to_tick_position() generates protocol TickPosition."
```

---

## Task 10: Hook installation + forward pass lifecycle

Python functions for installing sentinel hooks, capture hooks with mailbox barrier, removing hooks, and running the forward pass on a separate thread.

**Files:**
- Modify: `python/rocket_surgeon/bridge.py`
- Modify: `python/rocket_surgeon/hooks/__init__.py`
- Create: `python/tests/test_hooks.py`

- [ ] **Step 1: Write failing tests**

Create `python/tests/test_hooks.py`:

```python
"""Tests for hook installation, barrier cycling, and forward pass lifecycle."""

from __future__ import annotations

import threading

import pytest
import torch

from rocket_surgeon.bridge import (
    discover_modules,
    install_capture_hooks,
    install_sentinel_hooks,
    load_model,
    remove_hooks,
    run_forward,
    unload_model,
)
from rocket_surgeon.hooks.mailbox import Mailbox

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture
def model_handle() -> int:
    handle = load_model(source=TINY_MODEL, device="cpu", dtype="float32")
    yield handle
    unload_model(handle)


def test_install_sentinel_hooks_returns_handles(model_handle: int) -> None:
    modules = discover_modules(model_handle)
    paths = [m["path"] for m in modules]
    handles = install_sentinel_hooks(model_handle, paths)
    assert isinstance(handles, list)
    assert len(handles) == len(paths)
    remove_hooks(handles)


def test_install_capture_hooks_returns_handles(model_handle: int) -> None:
    result_mb = Mailbox()
    resume_mb = Mailbox()
    paths = ["model.layers.0.self_attn.q_proj"]
    handles = install_capture_hooks(
        model_handle, paths, result_mb, resume_mb, active_probes={"model.layers.0.self_attn.q_proj"}
    )
    assert isinstance(handles, list)
    assert len(handles) == 1
    remove_hooks(handles)


def test_capture_hook_barrier_cycle(model_handle: int) -> None:
    """Full barrier cycle: hook fires, puts result, blocks, gets resumed."""
    result_mb = Mailbox()
    resume_mb = Mailbox()
    target_path = "model.layers.0.self_attn.q_proj"

    modules = discover_modules(model_handle)
    all_paths = [m["path"] for m in modules]
    sentinel_handles = install_sentinel_hooks(model_handle, all_paths)
    capture_handles = install_capture_hooks(
        model_handle, [target_path], result_mb, resume_mb,
        active_probes={target_path}
    )

    captured: list[tuple[str, int]] = []
    errors: list[str] = []

    def forward_thread() -> None:
        try:
            from rocket_surgeon.bridge import _models
            model = _models[model_handle]
            with torch.inference_mode():
                dummy = torch.zeros(1, 2, dtype=torch.long)
                model(dummy)
        except Exception as e:
            errors.append(str(e))

    fwd = threading.Thread(target=forward_thread)
    fwd.start()

    value = result_mb.wait()
    assert value is not None
    path, call_index, tensor = value
    assert path == target_path
    assert isinstance(call_index, int)
    assert isinstance(tensor, torch.Tensor)
    captured.append((path, call_index))
    result_mb.restore()

    resume_mb.put(None)

    fwd.join(timeout=10.0)
    assert not fwd.is_alive(), "Forward thread did not complete"
    assert len(errors) == 0, f"Forward thread errors: {errors}"
    assert len(captured) == 1

    remove_hooks(sentinel_handles + capture_handles)


def test_remove_hooks_cleans_up(model_handle: int) -> None:
    modules = discover_modules(model_handle)
    paths = [m["path"] for m in modules]
    handles = install_sentinel_hooks(model_handle, paths)
    remove_hooks(handles)


def test_run_forward_calls_done_callback(model_handle: int) -> None:
    done_event = threading.Event()
    error_ref: list[Exception | None] = [None]

    def done_callback(error: Exception | None) -> None:
        error_ref[0] = error
        done_event.set()

    input_ids = torch.zeros(1, 2, dtype=torch.long)
    run_forward(model_handle, input_ids, done_callback)
    done_event.wait(timeout=10.0)
    assert done_event.is_set()
    assert error_ref[0] is None


def test_run_forward_reports_error_on_bad_input(model_handle: int) -> None:
    done_event = threading.Event()
    error_ref: list[Exception | None] = [None]

    def done_callback(error: Exception | None) -> None:
        error_ref[0] = error
        done_event.set()

    bad_input = torch.zeros(0, dtype=torch.long)
    run_forward(model_handle, bad_input, done_callback)
    done_event.wait(timeout=10.0)
    assert done_event.is_set()
    assert error_ref[0] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest python/tests/test_hooks.py -v`

Expected: FAIL — `ImportError: cannot import name 'install_sentinel_hooks'`

- [ ] **Step 3: Implement install_sentinel_hooks**

Add to `python/rocket_surgeon/bridge.py`:

```python
def install_sentinel_hooks(handle: int, module_paths: list[str]) -> list[Any]:
    """Install no-op sentinel hooks on specified modules to defeat PyTorch's fast path.

    Returns list of RemovableHandle objects.
    """
    model = _models[handle]
    modules_by_path = dict(model.named_modules())
    handles = []
    for path in module_paths:
        module = modules_by_path.get(path)
        if module is not None:
            h = module.register_forward_hook(lambda _m, _i, out: out)
            handles.append(h)
    return handles
```

- [ ] **Step 4: Implement install_capture_hooks**

Add to `python/rocket_surgeon/bridge.py`:

```python
def install_capture_hooks(
    handle: int,
    module_paths: list[str],
    result_mailbox: Any,
    resume_mailbox: Any,
    active_probes: set[str] | None = None,
) -> list[Any]:
    """Install capture hooks with mailbox barrier on specified modules.

    Each hook:
    1. Fast-exit if module_path not in active_probes
    2. Put (path, call_index, tensor) on result_mailbox
    3. Block on resume_mailbox.wait()
    4. Check intervention, restore, return
    """
    from rocket_surgeon.hooks.mailbox import Mailbox

    model = _models[handle]
    modules_by_path = dict(model.named_modules())
    handles = []
    call_counts: dict[str, int] = {}

    if active_probes is None:
        active_probes = set()

    for path in module_paths:
        module = modules_by_path.get(path)
        if module is None:
            continue

        def make_hook(p: str) -> Any:
            def hook(mod: Any, inp: Any, output: torch.Tensor) -> torch.Tensor | None:
                if p not in active_probes:
                    return None

                idx = call_counts.get(p, 0)
                call_counts[p] = idx + 1

                result_mailbox.put((p, idx, output))
                intervention = resume_mailbox.wait()
                resume_mailbox.restore()

                if intervention is not None:
                    return intervention
                return None
            return hook

        h = module.register_forward_hook(make_hook(path), prepend=True)
        handles.append(h)
    return handles
```

- [ ] **Step 5: Implement remove_hooks and run_forward**

Add to `python/rocket_surgeon/bridge.py`:

```python
def remove_hooks(handles: list[Any]) -> None:
    """Remove all hooks referenced by the given handles."""
    for h in handles:
        h.remove()


def run_forward(
    handle: int,
    input_ids: torch.Tensor,
    done_callback: Any,
) -> None:
    """Spawn a thread that runs model(input_ids) and calls done_callback on completion.

    done_callback(None) on success, done_callback(exception) on error.
    """
    model = _models[handle]

    def _run() -> None:
        try:
            with torch.inference_mode():
                model(input_ids)
            done_callback(None)
        except Exception as e:
            done_callback(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
```

Also add `import threading` to the top of `bridge.py` if not already present.

- [ ] **Step 6: Run tests**

Run: `pytest python/tests/test_hooks.py -v`

Expected: All 6 tests PASS.

Note: The `test_capture_hook_barrier_cycle` test may only capture at one hook point because the hook blocks and we only do one resume. In the real system, the Rust thread drives the step loop. This test validates the fundamental barrier mechanics work.

- [ ] **Step 7: Commit**

```bash
git add python/rocket_surgeon/bridge.py python/tests/test_hooks.py
git commit -m "feat(bridge): hook installation + forward pass lifecycle

install_sentinel_hooks defeats PyTorch fast path. install_capture_hooks
uses lock-based mailbox barrier for tick-by-tick stepping.
run_forward spawns a thread and reports completion via callback."
```

---

## Task 11: Capture policy

Rust module for probe matching against the ComponentMap and packaging captured stats into protocol types.

**Files:**
- Create: `crates/rocket-surgeon-worker/src/capture.rs`
- Modify: `crates/rocket-surgeon-worker/src/main.rs`

- [ ] **Step 1: Write failing tests**

Create `crates/rocket-surgeon-worker/src/capture.rs` with tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{ComponentMap, MappedComponent, ModuleMapping};

    fn sample_component_map() -> ComponentMap {
        ComponentMap {
            components: vec![
                MappedComponent {
                    module_path: "model.layers.0.self_attn.q_proj".into(),
                    canonical: "q_proj".into(),
                    layer_index: Some(0),
                    call_index: 0,
                    mapping: ModuleMapping::Direct { canonical: "q_proj".into() },
                    probe_point: "model:0:0:q_proj:0:fwd".into(),
                },
                MappedComponent {
                    module_path: "model.layers.0.self_attn.k_proj".into(),
                    canonical: "k_proj".into(),
                    layer_index: Some(0),
                    call_index: 0,
                    mapping: ModuleMapping::Direct { canonical: "k_proj".into() },
                    probe_point: "model:0:0:k_proj:0:fwd".into(),
                },
            ],
            model_family: "llama".into(),
            vocabulary: vec!["q_proj".into(), "k_proj".into()],
        }
    }

    #[test]
    fn probe_matches_exact_path() {
        let map = sample_component_map();
        let active = vec!["model:0:0:q_proj:0:fwd".to_owned()];
        assert!(should_capture(&map, "model.layers.0.self_attn.q_proj", 0, &active));
    }

    #[test]
    fn probe_does_not_match_different_component() {
        let map = sample_component_map();
        let active = vec!["model:0:0:q_proj:0:fwd".to_owned()];
        assert!(!should_capture(&map, "model.layers.0.self_attn.k_proj", 0, &active));
    }

    #[test]
    fn wildcard_layer_matches_any_layer() {
        let map = sample_component_map();
        let active = vec!["model:0:*:q_proj:0:fwd".to_owned()];
        assert!(should_capture(&map, "model.layers.0.self_attn.q_proj", 0, &active));
    }

    #[test]
    fn wildcard_component_matches_all() {
        let map = sample_component_map();
        let active = vec!["model:0:0:*:0:fwd".to_owned()];
        assert!(should_capture(&map, "model.layers.0.self_attn.q_proj", 0, &active));
        assert!(should_capture(&map, "model.layers.0.self_attn.k_proj", 0, &active));
    }

    #[test]
    fn empty_active_probes_matches_nothing() {
        let map = sample_component_map();
        let active: Vec<String> = vec![];
        assert!(!should_capture(&map, "model.layers.0.self_attn.q_proj", 0, &active));
    }

    #[test]
    fn capture_policy_none_for_inactive() {
        assert_eq!(capture_mode(&[]), CaptureMode::None);
    }

    #[test]
    fn capture_policy_summary_is_default() {
        assert_eq!(capture_mode(&["model:0:0:q_proj:0:fwd".to_owned()]), CaptureMode::Summary);
    }
}
```

- [ ] **Step 2: Add mod declaration and run tests**

Add `mod capture;` to `main.rs`.

```bash
cargo test -p rocket-surgeon-worker -- capture
```

Expected: FAIL — types not defined.

- [ ] **Step 3: Implement capture module**

Add to `capture.rs`:

```rust
use crate::adapter::ComponentMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureMode {
    None,
    Summary,
    Full,
}

pub fn capture_mode(active_probes: &[String]) -> CaptureMode {
    if active_probes.is_empty() {
        CaptureMode::None
    } else {
        CaptureMode::Summary
    }
}

pub fn should_capture(
    map: &ComponentMap,
    module_path: &str,
    call_index: u32,
    active_probes: &[String],
) -> bool {
    let component = map
        .components
        .iter()
        .find(|c| c.module_path == module_path && c.call_index == call_index);

    let component = match component {
        Some(c) => c,
        None => return false,
    };

    for probe_pattern in active_probes {
        if probe_matches(&component.probe_point, probe_pattern) {
            return true;
        }
    }
    false
}

fn probe_matches(probe_point: &str, pattern: &str) -> bool {
    let point_parts: Vec<&str> = probe_point.split(':').collect();
    let pattern_parts: Vec<&str> = pattern.split(':').collect();

    if point_parts.len() != pattern_parts.len() {
        return false;
    }

    for (pp, pat) in point_parts.iter().zip(pattern_parts.iter()) {
        if *pat == "*" {
            continue;
        }
        if pp != pat {
            return false;
        }
    }
    true
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p rocket-surgeon-worker -- capture
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/rocket-surgeon-worker/src/capture.rs crates/rocket-surgeon-worker/src/main.rs
git commit -m "feat(capture): probe matching and capture policy

should_capture matches module paths against active probe patterns
with wildcard support. CaptureMode: None/Summary/Full."
```

---

## Task 12: Worker dispatch — _host/configure_hooks + _host/step

Wire the adapter, bridge, tick, and capture modules through the worker dispatch loop. Add handlers for the new internal protocol commands.

**Files:**
- Modify: `crates/rocket-surgeon-worker/src/dispatch.rs`
- Modify: `crates/rocket-surgeon-worker/src/main.rs`

- [ ] **Step 1: Write failing tests for new dispatch methods**

Add to the tests module in `dispatch.rs`:

```rust
#[test]
fn dispatch_configure_hooks_invalid_params() {
    let req = make_request(
        internal::HOST_CONFIGURE_HOOKS,
        serde_json::json!({"wrong_field": 42}),
    );
    let resp = dispatch(&req);
    assert!(resp.error.is_some());
    assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
}

#[test]
fn dispatch_step_invalid_params() {
    let req = make_request(
        internal::HOST_STEP,
        serde_json::json!({"wrong_field": 42}),
    );
    let resp = dispatch(&req);
    assert!(resp.error.is_some());
    assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
}

#[test]
fn dispatch_update_probes_invalid_params() {
    let req = make_request(
        internal::HOST_UPDATE_PROBES,
        serde_json::json!({"wrong_field": 42}),
    );
    let resp = dispatch(&req);
    assert!(resp.error.is_some());
    assert_eq!(resp.error.as_ref().unwrap().code, INVALID_PARAMS);
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p rocket-surgeon-worker -- dispatch
```

Expected: FAIL — HOST_CONFIGURE_HOOKS not recognized in dispatch match.

- [ ] **Step 3: Add new command routing to dispatch**

Update the `dispatch()` function in `dispatch.rs`:

```rust
use rocket_surgeon_protocol::messages::internal;
use rocket_surgeon_protocol::messages::{
    HostAttachRequest, HostAttachResponse,
    HostConfigureHooksRequest, HostConfigureHooksResponse,
    HostDetachRequest, HostDetachResponse,
    HostStepRequest, HostStepResponse,
    HostUpdateProbesRequest, HostUpdateProbesResponse,
};

pub fn dispatch(request: &Request) -> Response {
    match request.method.as_str() {
        internal::HOST_ATTACH => handle_host_attach(request),
        internal::HOST_DETACH => handle_host_detach(request),
        internal::HOST_CONFIGURE_HOOKS => handle_host_configure_hooks(request),
        internal::HOST_STEP => handle_host_step(request),
        internal::HOST_UPDATE_PROBES => handle_host_update_probes(request),
        _ => Response::error(
            request.id.clone(),
            RpcError {
                code: METHOD_NOT_FOUND,
                message: format!("Method not found: {}", request.method),
                data: None,
            },
        ),
    }
}
```

- [ ] **Step 4: Implement stub handlers**

Add to `dispatch.rs`:

```rust
fn handle_host_configure_hooks(request: &Request) -> Response {
    let req: HostConfigureHooksRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    let resp = HostConfigureHooksResponse {
        sentinel_count: 0,
        capture_count: 0,
    };

    match serde_json::to_value(resp) {
        Ok(value) => Response::success(request.id.clone(), value),
        Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
    }
}

fn handle_host_step(request: &Request) -> Response {
    let req: HostStepRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    let resp = HostStepResponse {
        position: rocket_surgeon_protocol::types::TickPosition {
            tick_id: 0,
            direction: rocket_surgeon_protocol::types::StepDirection::Forward,
            rank: Some(0),
            layer: 0,
            component: String::new(),
            event: rocket_surgeon_protocol::types::TickEvent::Output,
            replay_of: None,
        },
        capture: None,
        forward_complete: false,
    };

    match serde_json::to_value(resp) {
        Ok(value) => Response::success(request.id.clone(), value),
        Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
    }
}

fn handle_host_update_probes(request: &Request) -> Response {
    let req: HostUpdateProbesRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    let resp = HostUpdateProbesResponse {
        probes_active: req.active_probes.len() as u32,
    };

    match serde_json::to_value(resp) {
        Ok(value) => Response::success(request.id.clone(), value),
        Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
    }
}
```

- [ ] **Step 5: Update handle_host_attach to use adapter resolution**

Update the existing `handle_host_attach` to call through bridge → adapter:

```rust
fn handle_host_attach(request: &Request) -> Response {
    let req: HostAttachRequest = match parse_params(request) {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    let dtype_str = match req.dtype {
        Some(rocket_surgeon_protocol::types::DType::Float16) => "float16",
        Some(rocket_surgeon_protocol::types::DType::Bfloat16) => "bfloat16",
        Some(rocket_surgeon_protocol::types::DType::Float32) | None => "float32",
        Some(other) => {
            return Response::error(
                request.id.clone(),
                RpcError {
                    code: INVALID_PARAMS,
                    message: format!("Unsupported dtype: {other:?}"),
                    data: None,
                },
            );
        }
    };

    let handle = match bridge::load_model(&req.model_source, &req.device, dtype_str) {
        Ok(h) => h,
        Err(e) => return internal_error(request.id.clone(), format!("load_model failed: {e}")),
    };

    let info = match bridge::model_metadata(handle) {
        Ok(i) => i,
        Err(e) => {
            return internal_error(request.id.clone(), format!("model_metadata failed: {e}"));
        }
    };

    let config = match bridge::model_config(handle) {
        Ok(c) => c,
        Err(e) => {
            return internal_error(request.id.clone(), format!("model_config failed: {e}"));
        }
    };

    let modules = match bridge::discover_modules(handle) {
        Ok(m) => m,
        Err(e) => {
            return internal_error(request.id.clone(), format!("discover_modules failed: {e}"));
        }
    };

    let component_map = match crate::adapter::resolve(&modules, &config) {
        Ok(m) => m,
        Err(e) => {
            return internal_error(request.id.clone(), format!("adapter resolution failed: {e}"));
        }
    };

    let resp = HostAttachResponse {
        model_handle: info.handle,
        num_layers: info.num_layers,
        num_heads: info.num_heads,
        hidden_dim: info.hidden_dim,
        module_tree: info.module_tree,
        model_type: config.model_type,
        component_vocabulary: component_map.vocabulary,
    };

    match serde_json::to_value(resp) {
        Ok(value) => Response::success(request.id.clone(), value),
        Err(e) => internal_error(request.id.clone(), format!("serialization failed: {e}")),
    }
}
```

- [ ] **Step 6: Run all tests**

```bash
cargo test --workspace --all-targets --exclude rocket-surgeon-worker
cargo test -p rocket-surgeon-worker -- dispatch --test-threads=1
```

Expected: All tests pass. The dispatch tests that construct HostAttachResponse now include the new fields.

- [ ] **Step 7: Commit**

```bash
git add crates/rocket-surgeon-worker/src/dispatch.rs
git commit -m "feat(dispatch): add configure_hooks, step, update_probes handlers

Stub handlers for new internal commands. Attach handler now calls
bridge::model_config + discover_modules + adapter::resolve to build
component_vocabulary in the response."
```

---

## Task 13: Integration test — full attach + adapter resolution

End-to-end test that exercises the attach flow through the worker binary, verifying that the adapter resolution produces a component vocabulary.

**Files:**
- Modify: `tests/test_e2e_lifecycle.py` (or create new integration test)

- [ ] **Step 1: Check existing e2e test**

Read: `tests/test_e2e_lifecycle.py`

Understand the existing test pattern for spawning the worker and sending JSON-RPC messages.

- [ ] **Step 2: Write failing integration test**

Add a new test to the existing e2e test file (or create `tests/test_e2e_adapter.py`):

```python
"""Integration test: attach with adapter resolution returns component vocabulary."""

from __future__ import annotations

import json
import subprocess
import struct
import sys
from pathlib import Path

import pytest

WORKER_BIN = "target/debug/rs-worker"
TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def send_message(proc: subprocess.Popen, msg: dict) -> dict:
    """Send a JSON-RPC message and read the response."""
    body = json.dumps(msg).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    proc.stdin.write(header + body)
    proc.stdin.flush()

    header_line = b""
    while not header_line.endswith(b"\r\n\r\n"):
        byte = proc.stdout.read(1)
        if not byte:
            raise RuntimeError("Worker closed stdout")
        header_line += byte
    content_length = int(header_line.decode().split(":")[1].strip().split("\r\n")[0])
    body = proc.stdout.read(content_length)
    return json.loads(body)


@pytest.fixture
def worker():
    proc = subprocess.Popen(
        [WORKER_BIN, "--log-level", "warn"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    yield proc
    proc.terminate()
    proc.wait(timeout=5)


def test_attach_returns_component_vocabulary(worker) -> None:
    resp = send_message(worker, {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "_host/attach",
        "params": {
            "model_source": TINY_MODEL,
            "model_family": "llama",
            "device": "cpu",
            "rank": 0,
        },
    })
    assert resp.get("error") is None, f"Attach failed: {resp.get('error')}"
    result = resp["result"]
    assert "component_vocabulary" in result
    assert isinstance(result["component_vocabulary"], list)
    assert len(result["component_vocabulary"]) > 0
    assert "q_proj" in result["component_vocabulary"]
    assert "k_proj" in result["component_vocabulary"]
    assert "v_proj" in result["component_vocabulary"]
    assert "o_proj" in result["component_vocabulary"]
    assert result["model_type"] == "llama"
    assert result["num_layers"] > 0


def test_attach_returns_module_tree(worker) -> None:
    resp = send_message(worker, {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "_host/attach",
        "params": {
            "model_source": TINY_MODEL,
            "model_family": "llama",
            "device": "cpu",
            "rank": 0,
        },
    })
    result = resp["result"]
    assert "module_tree" in result
    assert len(result["module_tree"]) > 0
    has_layer = any("layers.0" in m for m in result["module_tree"])
    assert has_layer
```

- [ ] **Step 3: Build worker binary**

```bash
cargo build -p rocket-surgeon-worker
```

- [ ] **Step 4: Run integration tests**

```bash
pytest tests/test_e2e_adapter.py -v
```

Expected: Both tests PASS — attach returns component_vocabulary with canonical names.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_adapter.py
git commit -m "test(e2e): integration test for attach with adapter resolution

Verifies worker returns component_vocabulary with canonical names
(q_proj, k_proj, v_proj, o_proj, etc.) and model_type from config."
```

---

## Self-Review Checklist

### 1. Spec coverage

| Spec section | Task(s) | Status |
|---|---|---|
| §1 Core Concepts (ModuleMapping, FusedComponent) | Task 4 | Covered |
| §1 Per-Family Declarations (llama, gpt2) | Task 4 | Covered |
| §1 Adapter Resolution Pipeline | Task 5, 6, 8, 12 | Covered |
| §1 Execution Order Discovery (per-call, call_index) | Task 6 | Covered |
| §1 Probe-Point Construction | Task 5 | Covered (via resolve) |
| §2 Sentinel hooks | Task 10 | Covered |
| §2 Capture hooks (with mailbox barrier) | Task 10 | Covered |
| §2 Barrier Mechanics (lock-based mailboxes) | Task 3 | Covered |
| §2 Tick Cycle | Task 9, 12 | Covered |
| §2 Capture Policy (None/Summary/Full) | Task 11 | Covered |
| §2 Autocast contract (fp32 cast) | Task 7 | Covered |
| §2 Fused module splitting | Task 7 | Covered |
| §2 Forward Pass Lifecycle | Task 10 | Covered |
| §3 New Internal Protocol Commands | Task 2, 12 | Covered |
| §3 New Rust Modules (adapter, tick, capture) | Tasks 4-5, 9, 11 | Covered |
| §3 Bridge Growth | Tasks 6, 7, 8, 10 | Covered |
| §3 Rename skin → bridge | Task 1 | Covered |
| §4 Attach Sequence | Task 12 | Covered |
| §4 Step Sequence | Task 12 (stub) | Partial — full wiring needs forward thread management |
| §5 Error Handling | Task 10, 12 | Partial — hook exception path covered |
| §7 New Rust unit tests | Tasks 4, 5, 9, 11 | Covered |
| §7 New Python tests | Tasks 3, 6, 7, 10 | Covered |
| §7 Integration test | Task 13 | Covered |

**Known gaps**: The `_host/step` handler is a stub — full implementation requires the worker to manage persistent state across requests (ComponentMap, TickState, mailbox references, forward thread). This state management wiring is deferred because it requires the worker to evolve from stateless request/response to a stateful session, which is a design decision better handled as a follow-up task or early Chunk B work. The foundation pieces (all the types, Python functions, Rust modules) are in place for this wiring.

### 2. Placeholder scan
- No "TBD", "TODO", "implement later" in any step
- All code blocks contain actual code
- All commands include expected output

### 3. Type consistency
- `ModelConfig` defined in `adapter.rs`, re-exported from `bridge.rs` ✓
- `RawModule` defined in `adapter.rs`, used in `bridge.rs` and `resolve()` ✓
- `ComponentMap` used consistently across adapter, capture, dispatch ✓
- `Mailbox` API (put/wait/get/restore) used consistently in mailbox.py and bridge.py ✓
- `HostAttachResponse` new fields (`model_type`, `component_vocabulary`) updated everywhere ✓
