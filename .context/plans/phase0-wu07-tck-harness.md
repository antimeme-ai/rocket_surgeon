# Work Unit 0.7 — TCK Test Harness

## Scope

Build the pytest-bdd infrastructure that loads and runs all 16 Gherkin `.feature` files.
Deliverables are: the runner, fixtures, shared step definitions, per-feature test modules,
and the xtask `tck` subcommand. Step definitions are **stubs** — they match the patterns
but do not implement behavior. All scenarios report as `xfail` (no server yet).

## Deliverables

### Python test infrastructure
1. `python/tests/tck/__init__.py`
2. `python/tests/tck/conftest.py` — pytest-bdd fixtures + shared step imports
3. `python/tests/tck/steps/__init__.py`
4. `python/tests/tck/steps/common.py` — shared step definitions (Given/When/Then)
5. 16 test modules in `python/tests/tck/` — one per feature file, using `scenarios()`

### Config changes
6. `pyproject.toml` — add `pytest-bdd` to dev dependencies, configure bdd paths
7. `xtask/src/main.rs` — add `Tck` subcommand

## Architecture

### pytest-bdd wiring

Each feature file gets a test module that auto-collects scenarios:

```python
# python/tests/tck/test_lifecycle.py
from pytest_bdd import scenarios
scenarios("../../../tck/protocol/lifecycle.feature")
```

Step definitions are shared via `conftest.py` which imports from `steps/common.py`.

### Step definition strategy

Instead of 483 unique step strings, use **regex step definitions** that cover parametric
families. Group by verb/pattern:

**Given steps (~15 patterns):**
- `the session is in "{state}" state` (with optional `with model "{model}"`)
- `a rocket_surgeon server is running`
- `the session is initialized with protocol_version "{version}"`
- `a model "{name}" is attached` (with optional `with dtype`, `with model_family`)
- `the server capability "{cap}" is {value}`
- `the session has been stepped to tick {n} at layer {m}`
- `a defined probe "{id}" at point "{point}" with action "{action}"` (+ optional enabled/priority)
- `an active intervention "{id}" of type "{type}" on "{target}"` (+ optional mode/priority/params)
- `the session has an activation checkpoint "{id}" at tick {n} layer {m}`
- `no steps have been executed in this session`
- `the client has stepped forward {n} tick(s) at "{gran}" granularity`
- `the tensor store capacity is configured to hold at most {n} tensors`
- Catchall for remaining Given patterns

**When steps (~8 patterns):**
- `the client sends "{verb}" with:` (table or docstring)
- `the client sends "{verb}" with no parameters`
- `the client sends "{verb}" with direction "{dir}"`
- `the client executes {n} forward steps at "{gran}" granularity`
- `the request includes "{field}" array:` (docstring continuation)
- `the client subscribes to "{event}" events`
- `the client captures {n} tensors by inspecting {n} distinct components`
- Catchall for remaining When patterns

**Then steps (~20 patterns):**
- `the response status is "{status}"`
- `the response is a JSON-RPC error`
- `the response "{path}" is "{value}"`
- `the response "{path}" is not null` / `is null`
- `the response "{path}" is a non-empty string`
- `the response "{path}" has field "{field}" of type {type}`
- `the response "{path}" is an array with at least {n} element(s)`
- `the error "data.{field}" is "{value}"`
- `the error "{path}" is one of "{a}", "{b}"`
- `"{saved_a}" equals "{saved_b}"` / `< / >`
- `the client receives a "{event}" notification`
- `the client does not receive a "{event}" notification`
- Catchall for remaining Then patterns

**And steps** reuse the same definitions (pytest-bdd treats And/But as continuation of prior Given/When/Then).

### Fixture stack

```
conftest.py
├── daemon_process    — starts/stops the daemon (stub: yields None)
├── unix_socket_path  — tmpdir-scoped socket path
├── rpc_client        — JSON-RPC client over socket (stub: raises NotImplementedError)
├── session           — initialized session (stub)
├── session_state     — mutable dict tracking response state
└── model_fixture     — attached model (stub)
```

All fixtures are session/function-scoped stubs. Real implementations come in Phase 1.

### xfail strategy

Mark every test module with `pytestmark = pytest.mark.xfail(reason="no server implementation yet")`.
This ensures:
- pytest-bdd parses the feature files (catches Gherkin syntax errors)
- Step definitions are resolved (catches missing step defs)
- Tests actually run and fail gracefully (not import errors)
- Output shows xfail count, not error count

## Approach

1. Install pytest-bdd and verify it works with a trivial feature
2. Write conftest.py with fixture stubs
3. Write steps/common.py with all step definition patterns
4. Write 16 test modules (one per feature file)
5. Add xtask `tck` subcommand
6. Run `cargo xtask tck` — verify all scenarios are collected and xfail
7. Subagent code review
8. Fix findings

## Test module → feature file mapping

| Test module | Feature file |
|-------------|-------------|
| test_lifecycle.py | tck/protocol/lifecycle.feature |
| test_stepping.py | tck/protocol/stepping.feature |
| test_inspection.py | tck/protocol/inspection.feature |
| test_intervention.py | tck/protocol/intervention.feature |
| test_probes.py | tck/protocol/probes.feature |
| test_checkpoint.py | tck/protocol/checkpoint.feature |
| test_replay.py | tck/protocol/replay.feature |
| test_subscribe.py | tck/protocol/subscribe.feature |
| test_errors.py | tck/protocol/errors.feature |
| test_state_envelope.py | tck/protocol/state-envelope.feature |
| test_capabilities.py | tck/protocol/capabilities.feature |
| test_adapter.py | tck/model/adapter.feature |
| test_hooks.py | tck/model/hooks.feature |
| test_handles.py | tck/tensor/handles.feature |
| test_tick_granularity.py | tck/moe/tick-granularity.feature |
| test_bundle.py | tck/session/bundle.feature |

## Acceptance Criteria (from plan doc)

- [ ] `cargo xtask tck` runs all Gherkin scenarios
- [ ] All scenarios report as xfail — not import or missing step errors
- [ ] Fixture infrastructure for daemon lifecycle exists (stubs OK)
- [ ] JSON-RPC client helper exists (stub OK)
- [ ] Step definitions for common Given/When/Then patterns exist
- [ ] Clean teardown (no leaked processes/sockets)
