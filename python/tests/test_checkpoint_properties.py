"""Property / roundtrip / exception-raising tests for checkpoint capture-restore.

MATERIA oracle tiers exercised:
  * tier-4 roundtrip   — capture(t) -> arena bytes -> restore == t for arbitrary
                         tensors across every supported dtype (the crown jewel).
  * tier-6 model       — the (dtype, shape) returned by capture equals the abstract
                         (str(dtype), list(shape)) of the source.
  * tier-2 exception   — missing keys, undersized slots raise the *right* error.

The CUDA RNG paths are exercised only in their CPU-fallback form (no CUDA here);
the CPU RNG roundtrip is a clean metamorphic identity on the random stream.

Generator distribution is annotated with ``hypothesis.event``; inspect with
``--hypothesis-show-statistics``.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from rocket_surgeon.checkpoint import (
    activation_available,
    capture_activation,
    capture_cpu_rng_state,
    capture_rng_state,
    restore_activation,
    restore_cpu_rng_state,
    restore_rng_state,
)

# dtypes the checkpoint bridge claims to support, with element sizes.
_DTYPES: list[tuple[torch.dtype, np.dtype, int]] = [
    (torch.float32, np.dtype(np.float32), 4),
    (torch.float64, np.dtype(np.float64), 8),
    (torch.float16, np.dtype(np.float16), 2),
]


@st.composite
def _tensor_and_dtype(draw: st.DrawFn) -> tuple[torch.Tensor, torch.dtype]:
    """An arbitrary small tensor in one of the supported dtypes."""
    torch_dt, np_dt, _ = draw(st.sampled_from(_DTYPES))
    shape = draw(st.lists(st.integers(1, 6), min_size=1, max_size=3))
    # bounded finite elements so float16 doesn't overflow to inf
    elements = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    arr = draw(hnp.arrays(np_dt, tuple(shape), elements=elements))
    return torch.from_numpy(arr.copy()), torch_dt


def _classify(t: torch.Tensor) -> None:
    event(f"dtype: {t.dtype}")
    event(f"ndim: {t.dim()}")
    event(f"nelem bucket: {2 ** (t.nelement().bit_length())}")


# --------------------------------------------------------------------------- #
# Roundtrip: capture -> arena -> restore == identity
# --------------------------------------------------------------------------- #
@given(_tensor_and_dtype())
@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
def test_capture_restore_roundtrip(payload: tuple[torch.Tensor, torch.dtype]) -> None:
    """Roundtrip: writing a tensor into arena memory and reading it back yields
    the identical tensor (bit-for-bit for the supported dtypes)."""
    original, _ = payload
    _classify(original)
    key = ("layer.x", 0)
    src = {key: original}

    nbytes = original.nelement() * original.element_size()
    buf = (ctypes.c_byte * nbytes)()
    ptr = ctypes.addressof(buf)

    dtype_str, shape = capture_activation(src, "layer.x", 0, ptr, nbytes)

    # restore into a *fresh* zero target of the same dtype/shape
    target = torch.zeros_like(original)
    dst = {key: target}
    restore_activation(dst, "layer.x", 0, ptr, nbytes, dtype_str, shape)

    assert torch.equal(target, original)


@given(_tensor_and_dtype())
@settings(max_examples=200)
def test_capture_returns_abstract_dtype_and_shape(
    payload: tuple[torch.Tensor, torch.dtype],
) -> None:
    """Model oracle: capture's (dtype, shape) return == (str(dtype), list(shape))
    of the source tensor."""
    original, _ = payload
    src = {("c", 3): original}
    dtype_str, shape = capture_activation(src, "c", 3, *_fresh_buffer(original))
    assert dtype_str == str(original.dtype)
    assert shape == list(original.shape)


@given(_tensor_and_dtype())
@settings(max_examples=200)
def test_tuple_wrapped_activation_uses_first_element(
    payload: tuple[torch.Tensor, torch.dtype],
) -> None:
    """Model oracle: a tuple-valued activation is treated as its first element
    (HF modules often return tuples)."""
    original, _ = payload
    wrapped: dict[tuple[str, int], Any] = {("c", 0): (original, "ignored", 42)}
    bare = {("c", 0): original}
    ptr_w, n_w = _fresh_buffer(original)
    ptr_b, n_b = _fresh_buffer(original)
    assert capture_activation(wrapped, "c", 0, ptr_w, n_w) == capture_activation(
        bare, "c", 0, ptr_b, n_b
    )


def _fresh_buffer(t: torch.Tensor) -> tuple[int, int]:
    nbytes = t.nelement() * t.element_size()
    buf = (ctypes.c_byte * nbytes)()
    # keep buffer alive on the object so the pointer stays valid for the call
    _BUFFERS.append(buf)
    return ctypes.addressof(buf), nbytes


_BUFFERS: list[Any] = []


# --------------------------------------------------------------------------- #
# Exception-raising properties
# --------------------------------------------------------------------------- #
@given(st.text(min_size=1, max_size=12), st.integers(0, 99))
@settings(max_examples=200)
def test_capture_missing_key_raises_keyerror(path: str, idx: int) -> None:
    """E1: capturing a component not in last_outputs raises KeyError."""
    assume((path, idx) not in {("present", 0)})
    src: dict[tuple[str, int], Any] = {("present", 0): torch.zeros(4)}
    buf = ctypes.create_string_buffer(64)
    with pytest.raises(KeyError):
        capture_activation(src, path, idx, ctypes.addressof(buf), len(buf))


@given(_tensor_and_dtype())
@settings(max_examples=200)
def test_capture_undersized_slot_raises_valueerror(
    payload: tuple[torch.Tensor, torch.dtype],
) -> None:
    """E2: a destination slot smaller than the tensor raises ValueError with a
    capacity message — never a silent truncation / buffer overrun."""
    original, _ = payload
    nbytes = original.nelement() * original.element_size()
    assume(nbytes > 1)
    src = {("c", 0): original}
    too_small = nbytes - 1
    buf = (ctypes.c_byte * too_small)()
    with pytest.raises(ValueError, match="exceeds slot capacity"):
        capture_activation(src, "c", 0, ctypes.addressof(buf), too_small)


@given(_tensor_and_dtype())
@settings(max_examples=200)
def test_restore_undersized_slot_raises_valueerror(
    payload: tuple[torch.Tensor, torch.dtype],
) -> None:
    """E3: restoring from a source slot smaller than the tensor raises ValueError."""
    original, _ = payload
    nbytes = original.nelement() * original.element_size()
    assume(nbytes > 1)
    dst = {("c", 0): torch.zeros_like(original)}
    too_small = nbytes - 1
    buf = (ctypes.c_byte * nbytes)()
    with pytest.raises(ValueError, match="exceeds slot capacity"):
        restore_activation(
            dst,
            "c",
            0,
            ctypes.addressof(buf),
            too_small,
            str(original.dtype),
            list(original.shape),
        )


@given(_tensor_and_dtype())
@settings(max_examples=100)
def test_restore_missing_target_raises_keyerror(
    payload: tuple[torch.Tensor, torch.dtype],
) -> None:
    """E4: restoring into a dict with no target tensor raises KeyError with a
    'cannot restore' message (the slot must pre-exist)."""
    original, _ = payload
    nbytes = original.nelement() * original.element_size()
    buf = (ctypes.c_byte * nbytes)()
    with pytest.raises(KeyError, match="cannot restore"):
        restore_activation(
            {},
            "missing",
            0,
            ctypes.addressof(buf),
            nbytes,
            str(original.dtype),
            list(original.shape),
        )


# --------------------------------------------------------------------------- #
# activation_available model oracle
# --------------------------------------------------------------------------- #
@given(
    st.dictionaries(
        st.tuples(st.text(max_size=6), st.integers(0, 9)),
        st.just(0),
        max_size=8,
    ),
    st.text(max_size=6),
    st.integers(0, 9),
)
@settings(max_examples=200)
def test_activation_available_matches_set_membership(
    store: dict[tuple[str, int], int], path: str, idx: int
) -> None:
    """Model oracle: activation_available agrees with key-set membership for all
    inputs (the dict-of-keys IS the abstract model)."""
    event("hit" if (path, idx) in store else "miss")
    assert activation_available(store, path, idx) == ((path, idx) in store)


# --------------------------------------------------------------------------- #
# CPU RNG roundtrip — metamorphic identity on the random stream
# --------------------------------------------------------------------------- #
@given(st.integers(0, 2**31 - 1), st.integers(1, 32))
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_cpu_rng_state_roundtrip(seed: int, n: int) -> None:
    """Roundtrip: restoring a captured CPU RNG state reproduces the exact stream
    that followed capture."""
    torch.manual_seed(seed)
    state = capture_cpu_rng_state()
    first = torch.rand(n)
    restore_cpu_rng_state(state)
    second = torch.rand(n)
    assert torch.equal(first, second)


def test_cuda_rng_roundtrip_is_noop_without_cuda() -> None:
    """E5: with no CUDA, capture_rng_state encodes device_count 0 and restoring it
    is a no-op that does not raise (defined behaviour on the CPU-only path)."""
    if torch.cuda.is_available():
        pytest.skip("CUDA present; this pins the CPU-only fallback")
    blob = capture_rng_state()
    # device_count == 0 encoded as a single little-endian u32
    assert blob == (0).to_bytes(4, "little")
    restore_rng_state(blob)  # must not raise
