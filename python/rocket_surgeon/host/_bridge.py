"""Thin wrapper around the Rust PyO3 extension for hot-path operations.

Provides BLAKE3 hashing and ProbeFrame header serialization. Falls back to
pure-Python for ProbeFrame operations when the Rust extension is unavailable.
BLAKE3 has no pure-Python fallback — hash consistency with the Rust daemon is
critical for content-addressable tensor IDs.
"""

from __future__ import annotations

import struct
from typing import Any

PROBE_FRAME_HEADER_SIZE = 128
_RESERVED_SIZE = 48
_SHAPE_SLOTS = 8

_HAS_NATIVE: bool

try:
    from rocket_surgeon import _rs as _native  # type: ignore[attr-defined]

    _native_blake3_hash = _native.blake3_hash
    _native_serialize = _native.serialize_probe_frame_header
    _native_parse = _native.parse_probe_frame_header
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False


def has_native_extension() -> bool:
    """Return True if the Rust PyO3 extension is available."""
    return _HAS_NATIVE


def blake3_hash(data: bytes) -> str:
    """Compute BLAKE3 hash of data, returning hex string.

    Requires the Rust extension — no pure-Python fallback because hash
    consistency with the Rust daemon is required for content-addressable IDs.
    """
    if not _HAS_NATIVE:
        msg = (
            "blake3_hash requires the Rust extension (rocket_surgeon._rs). "
            "Build with: maturin develop"
        )
        raise RuntimeError(msg)
    return _native_blake3_hash(data)  # type: ignore[no-any-return]


def serialize_probe_frame_header(
    *,
    rank: int,
    layer: int,
    comp_id: int,
    dtype: int,
    ndim: int,
    shape: list[int],
    tick_id: int,
    data_off: int,
    size: int,
    flags: int,
    generation: int,
) -> bytes:
    """Serialize a ProbeFrame header to 128 bytes (little-endian packed)."""
    if _HAS_NATIVE:
        return _native_serialize(  # type: ignore[no-any-return]
            rank,
            layer,
            comp_id,
            dtype,
            ndim,
            shape,
            tick_id,
            data_off,
            size,
            flags,
            generation,
        )
    return _py_serialize_probe_frame_header(
        rank=rank,
        layer=layer,
        comp_id=comp_id,
        dtype=dtype,
        ndim=ndim,
        shape=shape,
        tick_id=tick_id,
        data_off=data_off,
        size=size,
        flags=flags,
        generation=generation,
    )


def parse_probe_frame_header(data: bytes) -> dict[str, Any]:
    """Parse a 128-byte ProbeFrame header into a dict."""
    if _HAS_NATIVE:
        return _native_parse(data)  # type: ignore[no-any-return]
    return _py_parse_probe_frame_header(data)


def _py_serialize_probe_frame_header(
    *,
    rank: int,
    layer: int,
    comp_id: int,
    dtype: int,
    ndim: int,
    shape: list[int],
    tick_id: int,
    data_off: int,
    size: int,
    flags: int,
    generation: int,
) -> bytes:
    """Pure-Python ProbeFrame header serialization."""
    if len(shape) > _SHAPE_SLOTS:
        msg = f"shape has {len(shape)} dims, max is {_SHAPE_SLOTS}"
        raise ValueError(msg)
    if ndim != len(shape):
        msg = f"ndim ({ndim}) does not match shape length ({len(shape)})"
        raise ValueError(msg)

    padded_shape = list(shape) + [0] * (_SHAPE_SLOTS - len(shape))

    buf = struct.pack(
        "<IIHBB8IxxxxQQQII",
        rank,
        layer,
        comp_id,
        dtype,
        ndim,
        *padded_shape,
        tick_id,
        data_off,
        size,
        flags,
        generation,
    )
    buf += b"\x00" * _RESERVED_SIZE
    return buf


def _py_parse_probe_frame_header(data: bytes) -> dict[str, Any]:
    """Pure-Python ProbeFrame header parsing."""
    if len(data) < PROBE_FRAME_HEADER_SIZE:
        msg = f"buffer too small: expected {PROBE_FRAME_HEADER_SIZE} bytes, got {len(data)}"
        raise ValueError(msg)

    fields = struct.unpack_from("<IIHBB8IxxxxQQQII", data, 0)
    rank = fields[0]
    layer = fields[1]
    comp_id = fields[2]
    dtype = fields[3]
    ndim = fields[4]
    shape = list(fields[5:13])
    tick_id = fields[13]
    data_off = fields[14]
    size = fields[15]
    flags = fields[16]
    generation = fields[17]

    return {
        "rank": rank,
        "layer": layer,
        "comp_id": comp_id,
        "dtype": dtype,
        "ndim": ndim,
        "shape": shape,
        "tick_id": tick_id,
        "data_off": data_off,
        "size": size,
        "flags": flags,
        "generation": generation,
    }
