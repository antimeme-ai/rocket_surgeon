---
id: BEAD-0019
title: rocket-surgeon-python — _bridge.py stale vs v2 ProbeFrame signature, master red
status: open
priority: high
created: 2026-05-21
---

## Description

Commit `5553a5f` ("feat(probe-frame): v2 alignment fix — _pad0, data_off
rename, generation field") updated the Rust side of the PyO3 bridge — the
`#[pyfunction] serialize_probe_frame_header` in
`crates/rocket-surgeon-python/src/lib.rs` gained a `generation: u32`
parameter (11 params total) and renamed `offset` → `data_off`, and
`probe_frame.rs` gained the `generation` field plus `_pad0` alignment — but
**`python/rocket_surgeon/host/_bridge.py` was not updated to match**.

The Python wrapper `serialize_probe_frame_header` still carries the v1
signature and calls `_native_serialize(...)` with 10 positional args.
Against the v2 native extension this raises:

```
TypeError: serialize_probe_frame_header() missing 1 required positional
argument: 'generation'
```

The pure-Python fallback path is also still on the v1 layout.

## Impact

- `python/tests/test_bridge.py` is red on master — 12 failures across
  `TestSerializeProbeFrameHeader`, `TestPurePythonFallback`,
  `TestNativeExtension`.
- The lefthook **pre-push** hook (`pytest python/tests/`) therefore fails
  for every branch regardless of what the branch changed — pushes need
  `--no-verify` until this is fixed.
- `5553a5f` reached master without the gate catching the mismatch — a
  gate-bypass (web-UI merge), the recurring "master red" pattern. Once
  BEAD-0016 remote CI is on master this class of regression is caught.

## Scope

- Update `_bridge.py`'s `serialize_probe_frame_header` / parse wrappers to
  the v2 signature: add `generation`, rename `offset` → `data_off`.
- Update the pure-Python fallback serialization/parsing to the v2 128-byte
  layout (`_pad0`, `data_off`, `generation`) so it byte-matches the native
  path — the `TestPurePythonFallback` cross-path tests assert exactly this.
- Confirm `test_bridge.py` green and the pre-push gate green again.
- Cross-check other `ProbeFrame` consumers (model host, daemon) for the
  same v1/v2 drift.

## Notes

- Discovered while landing BEAD-0015 slice 3 (PR #29), which had to push
  with `--no-verify` to clear this unrelated breakage.
