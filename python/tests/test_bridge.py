"""Tests for the PyO3 bridge: BLAKE3 hashing and ProbeFrame header."""

from __future__ import annotations

import pytest

from rocket_surgeon.host._bridge import (
    PROBE_FRAME_HEADER_SIZE,
    _py_parse_probe_frame_header,
    _py_serialize_probe_frame_header,
    blake3_hash,
    has_native_extension,
    parse_probe_frame_header,
    serialize_probe_frame_header,
)

BLAKE3_EMPTY_HEX = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"


@pytest.fixture
def sample_header_kwargs() -> dict:
    return {
        "rank": 0,
        "layer": 12,
        "comp_id": 3,
        "dtype": 2,
        "ndim": 3,
        "shape": [2, 4096, 4096],
        "tick_id": 42,
        "data_off": 0x1000,
        "size": 2 * 4096 * 4096 * 4,
        "flags": 0,
        "generation": 7,
    }


class TestBlake3:
    def test_empty_input(self) -> None:
        result = blake3_hash(b"")
        assert result == BLAKE3_EMPTY_HEX

    def test_known_input(self) -> None:
        result = blake3_hash(b"hello")
        assert len(result) == 64
        assert result != BLAKE3_EMPTY_HEX

    def test_deterministic(self) -> None:
        data = b"\x00" * 1024
        assert blake3_hash(data) == blake3_hash(data)

    def test_different_inputs_different_hashes(self) -> None:
        assert blake3_hash(b"hello") != blake3_hash(b"world")


class TestSerializeProbeFrameHeader:
    def test_size_is_128(self, sample_header_kwargs: dict) -> None:
        result = serialize_probe_frame_header(**sample_header_kwargs)
        assert len(result) == PROBE_FRAME_HEADER_SIZE

    def test_round_trip(self, sample_header_kwargs: dict) -> None:
        serialized = serialize_probe_frame_header(**sample_header_kwargs)
        parsed = parse_probe_frame_header(serialized)
        assert parsed["rank"] == sample_header_kwargs["rank"]
        assert parsed["layer"] == sample_header_kwargs["layer"]
        assert parsed["comp_id"] == sample_header_kwargs["comp_id"]
        assert parsed["dtype"] == sample_header_kwargs["dtype"]
        assert parsed["ndim"] == sample_header_kwargs["ndim"]
        assert parsed["tick_id"] == sample_header_kwargs["tick_id"]
        assert parsed["data_off"] == sample_header_kwargs["data_off"]
        assert parsed["generation"] == sample_header_kwargs["generation"]
        assert parsed["size"] == sample_header_kwargs["size"]
        assert parsed["flags"] == sample_header_kwargs["flags"]
        n_pad = 8 - len(sample_header_kwargs["shape"])
        padded_shape = sample_header_kwargs["shape"] + [0] * n_pad
        assert list(parsed["shape"]) == padded_shape

    def test_shape_padding(self) -> None:
        result = serialize_probe_frame_header(
            rank=0,
            layer=0,
            comp_id=0,
            dtype=0,
            ndim=2,
            shape=[1024, 768],
            tick_id=0,
            data_off=0,
            generation=0,
            size=0,
            flags=0,
        )
        parsed = parse_probe_frame_header(result)
        shape = list(parsed["shape"])
        assert shape[0] == 1024
        assert shape[1] == 768
        assert all(d == 0 for d in shape[2:])

    def test_reserved_bytes_are_zero(self, sample_header_kwargs: dict) -> None:
        result = serialize_probe_frame_header(**sample_header_kwargs)
        assert all(b == 0 for b in result[80:128])

    def test_little_endian(self) -> None:
        result = serialize_probe_frame_header(
            rank=0x04030201,
            layer=0,
            comp_id=0,
            dtype=0,
            ndim=0,
            shape=[],
            tick_id=0,
            data_off=0,
            generation=0,
            size=0,
            flags=0,
        )
        assert result[0] == 0x01
        assert result[1] == 0x02
        assert result[2] == 0x03
        assert result[3] == 0x04


class TestPurePythonFallback:
    def test_serialize_matches_native(self, sample_header_kwargs: dict) -> None:
        if not has_native_extension():
            pytest.skip("native extension not available")
        native = serialize_probe_frame_header(**sample_header_kwargs)
        pure_py = _py_serialize_probe_frame_header(**sample_header_kwargs)
        assert native == pure_py

    def test_parse_matches_native(self, sample_header_kwargs: dict) -> None:
        if not has_native_extension():
            pytest.skip("native extension not available")
        data = serialize_probe_frame_header(**sample_header_kwargs)
        native = parse_probe_frame_header(data)
        pure_py = _py_parse_probe_frame_header(data)
        assert native["rank"] == pure_py["rank"]
        assert native["layer"] == pure_py["layer"]
        assert native["comp_id"] == pure_py["comp_id"]
        assert native["dtype"] == pure_py["dtype"]
        assert native["ndim"] == pure_py["ndim"]
        assert list(native["shape"]) == list(pure_py["shape"])
        assert native["tick_id"] == pure_py["tick_id"]
        assert native["data_off"] == pure_py["data_off"]
        assert native["generation"] == pure_py["generation"]
        assert native["size"] == pure_py["size"]
        assert native["flags"] == pure_py["flags"]

    def test_pure_python_round_trip(self, sample_header_kwargs: dict) -> None:
        serialized = _py_serialize_probe_frame_header(**sample_header_kwargs)
        assert len(serialized) == PROBE_FRAME_HEADER_SIZE
        parsed = _py_parse_probe_frame_header(serialized)
        assert parsed["rank"] == sample_header_kwargs["rank"]
        assert parsed["layer"] == sample_header_kwargs["layer"]
        assert parsed["tick_id"] == sample_header_kwargs["tick_id"]

    def test_pure_python_shape_too_long(self) -> None:
        with pytest.raises(ValueError, match="max is 8"):
            _py_serialize_probe_frame_header(
                rank=0,
                layer=0,
                comp_id=0,
                dtype=0,
                ndim=9,
                shape=[1] * 9,
                tick_id=0,
                data_off=0,
                generation=0,
                size=0,
                flags=0,
            )

    def test_pure_python_parse_too_small(self) -> None:
        with pytest.raises(ValueError, match="buffer too small"):
            _py_parse_probe_frame_header(b"\x00" * 64)

    def test_pure_python_ndim_mismatch(self) -> None:
        with pytest.raises(ValueError, match=r"ndim.*does not match"):
            _py_serialize_probe_frame_header(
                rank=0,
                layer=0,
                comp_id=0,
                dtype=0,
                ndim=5,
                shape=[42],
                tick_id=0,
                data_off=0,
                generation=0,
                size=0,
                flags=0,
            )

    def test_cross_path_serialize_rust_parse_python(self, sample_header_kwargs: dict) -> None:
        if not has_native_extension():
            pytest.skip("native extension not available")
        rust_bytes = serialize_probe_frame_header(**sample_header_kwargs)
        py_parsed = _py_parse_probe_frame_header(rust_bytes)
        assert py_parsed["rank"] == sample_header_kwargs["rank"]
        assert py_parsed["tick_id"] == sample_header_kwargs["tick_id"]

    def test_cross_path_serialize_python_parse_rust(self, sample_header_kwargs: dict) -> None:
        if not has_native_extension():
            pytest.skip("native extension not available")
        py_bytes = _py_serialize_probe_frame_header(**sample_header_kwargs)
        rust_parsed = parse_probe_frame_header(py_bytes)
        assert rust_parsed["rank"] == sample_header_kwargs["rank"]
        assert rust_parsed["tick_id"] == sample_header_kwargs["tick_id"]


class TestNativeExtension:
    def test_has_native_extension(self) -> None:
        assert has_native_extension() is True

    def test_ndim_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match=r"ndim.*does not match"):
            serialize_probe_frame_header(
                rank=0,
                layer=0,
                comp_id=0,
                dtype=0,
                ndim=0,
                shape=[1024, 768],
                tick_id=0,
                data_off=0,
                generation=0,
                size=0,
                flags=0,
            )

    def test_empty_shape(self) -> None:
        result = serialize_probe_frame_header(
            rank=0,
            layer=0,
            comp_id=0,
            dtype=0,
            ndim=0,
            shape=[],
            tick_id=0,
            data_off=0,
            generation=0,
            size=0,
            flags=0,
        )
        assert len(result) == PROBE_FRAME_HEADER_SIZE
        parsed = parse_probe_frame_header(result)
        assert parsed["ndim"] == 0
        assert all(d == 0 for d in parsed["shape"])

    def test_max_shape_dims(self) -> None:
        result = serialize_probe_frame_header(
            rank=0,
            layer=0,
            comp_id=0,
            dtype=0,
            ndim=8,
            shape=[1] * 8,
            tick_id=0,
            data_off=0,
            generation=0,
            size=0,
            flags=0,
        )
        parsed = parse_probe_frame_header(result)
        assert parsed["ndim"] == 8
        assert all(d == 1 for d in parsed["shape"])
