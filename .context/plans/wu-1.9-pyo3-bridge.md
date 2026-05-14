# WU 1.9 — PyO3 Thin Bridge

## Goal

Expose hot-path operations (BLAKE3 hashing, ProbeFrame header serialization) as PyO3
functions callable from the Python host, with GIL-released computation and a pure-Python
fallback for development without Rust builds.

## Design Decisions

### ProbeFrame Header Layout (128 bytes, little-endian packed)

Derived from ADR-0004 and ADR-0006. Fields packed sequentially, no alignment padding:

```
Offset  Size  Type       Field
0       4     u32_le     rank
4       4     u32_le     layer
8       2     u16_le     comp_id
10      1     u8         dtype
11      1     u8         ndim
12      32    [u32_le;8] shape
44      8     u64_le     tick_id
52      8     u64_le     offset
60      8     u64_le     size
68      4     u32_le     flags
72      56    [u8;56]    _reserved (zeros)
                         ─────────────────
                         128 bytes total
```

### DType u8 encoding

Maps to protocol `DType` enum ordinals (matching serde order in types.rs):
  0=Float16, 1=BFloat16, 2=Float32, 3=Float64,
  4=Int8, 5=Int16, 6=Int32, 7=Int64, 8=UInt8, 9=Bool

### PyO3 Module Structure

Maturin config: `module-name = "rocket_surgeon._rs"`. The #[pymodule] generates
`rocket_surgeon._rs` which is imported by `python/rocket_surgeon/host/_bridge.py`.

Functions exposed:
- `blake3_hash(data: bytes) -> str` — hex-encoded 256-bit BLAKE3 digest. GIL released.
- `serialize_probe_frame_header(rank, layer, comp_id, dtype, ndim, shape, tick_id, offset, size, flags) -> bytes` — 128-byte header. GIL released.
- `parse_probe_frame_header(data: bytes) -> dict` — inverse of serialize, for testing/debugging.

### Pure-Python Fallback

`_bridge.py` tries `from rocket_surgeon._rs import ...`; on ImportError, falls back to
pure-Python implementations using `struct.pack` (for ProbeFrame) and `hashlib` (BLAKE3
unavailable in stdlib, so fallback uses SHA-256 with a flag indicating it's not BLAKE3).

Wait — `hashlib` doesn't have BLAKE3. Options:
1. Fallback uses hashlib.blake2b (different algorithm, clearly labeled)
2. Fallback raises RuntimeError when blake3_hash is called without Rust extension
3. Add blake3 PyPI package as optional dependency

Decision: Option 2 for blake3_hash (fast Rust path or nothing — hash consistency is critical
for content-addressable IDs). ProbeFrame serialize/parse have pure-Python fallbacks via struct.

## Files Modified

| File | Change |
|------|--------|
| `Cargo.toml` (workspace) | Add `blake3 = "1"` to workspace deps |
| `crates/rocket-surgeon-python/Cargo.toml` | Add `blake3.workspace = true` dep |
| `crates/rocket-surgeon-python/src/lib.rs` | Implement blake3_hash, probe_frame functions |
| `crates/rocket-surgeon-python/src/probe_frame.rs` | ProbeFrame header struct + serialization |
| `python/rocket_surgeon/host/__init__.py` | Create host subpackage |
| `python/rocket_surgeon/host/_bridge.py` | Python wrapper with fallback |
| `python/tests/test_bridge.py` | Unit tests for PyO3 functions |
| `crates/rocket-surgeon-python/src/tests.rs` | Rust-side unit tests |

## Test Plan (TCK: unit tests, not Gherkin)

### Rust-side tests
1. `blake3_hash_empty` — hash of empty input matches known BLAKE3 digest
2. `blake3_hash_known_input` — hash of b"hello" matches reference
3. `probe_frame_header_round_trip` — serialize then parse recovers all fields
4. `probe_frame_header_size_is_128` — serialized output is exactly 128 bytes
5. `probe_frame_header_reserved_zeros` — bytes 72..128 are all zero
6. `probe_frame_header_endianness` — first 4 bytes match rank.to_le_bytes()

### Python-side tests
7. `test_blake3_hash_matches_rust` — Python call matches Rust computation
8. `test_blake3_hash_empty` — empty bytes returns known digest
9. `test_blake3_hash_deterministic` — same input → same output
10. `test_serialize_probe_frame_header_size` — returns exactly 128 bytes
11. `test_probe_frame_round_trip` — serialize → parse recovers all fields
12. `test_probe_frame_shape_padding` — shape with ndim<8 pads remaining to 0
13. `test_bridge_fallback_probe_frame` — pure-Python fallback produces same bytes
14. `test_bridge_fallback_blake3_raises` — fallback raises without Rust extension

## Execution Order

1. Write failing Rust tests (red)
2. Write failing Python tests (red — will fail until PyO3 functions exist)
3. Implement Rust ProbeFrame module
4. Implement PyO3 bridge functions
5. Implement Python _bridge.py wrapper
6. Green: all tests pass
7. Clippy, fmt, CI
8. Code review → fix → commit
