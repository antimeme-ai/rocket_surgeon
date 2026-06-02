import ctypes

import pytest
import torch

from rocket_surgeon.checkpoint import activation_available, capture_activation


def test_capture_activation_present_key():
    t = torch.randn(4, 8)
    storage = {("transformer.h.0", 0): t}
    buf = ctypes.create_string_buffer(t.nelement() * t.element_size())
    ptr = ctypes.addressof(buf)
    dtype, shape = capture_activation(storage, "transformer.h.0", 0, ptr, len(buf))
    assert dtype == "torch.float32"
    assert shape == [4, 8]


def test_capture_activation_missing_key_raises():
    storage = {}
    buf = ctypes.create_string_buffer(128)
    ptr = ctypes.addressof(buf)
    with pytest.raises(KeyError):
        capture_activation(storage, "transformer.h.3", 0, ptr, len(buf))


def test_activation_available_true():
    t = torch.randn(4, 8)
    storage = {("transformer.h.0", 0): t}
    assert activation_available(storage, "transformer.h.0", 0) is True


def test_activation_available_false():
    storage = {}
    assert activation_available(storage, "transformer.h.3", 0) is False


def test_activation_available_partial_forward():
    t0 = torch.randn(4, 8)
    t1 = torch.randn(4, 8)
    storage = {
        ("transformer.h.0", 0): t0,
        ("transformer.h.1", 0): t1,
    }
    assert activation_available(storage, "transformer.h.0", 0) is True
    assert activation_available(storage, "transformer.h.1", 0) is True
    assert activation_available(storage, "transformer.h.3", 0) is False
    assert activation_available(storage, "transformer.h.11", 0) is False
