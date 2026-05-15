"""Tests for the minimal Python skin: load_model, unload_model, model_metadata."""

from __future__ import annotations

import pytest

from rocket_surgeon.skin import load_model, model_metadata, unload_model

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_load_model_returns_integer_handle() -> None:
    handle = load_model(source=TINY_MODEL, device="cpu", dtype="float32")
    assert isinstance(handle, int)
    assert handle > 0
    unload_model(handle)


def test_unload_model_removes_reference() -> None:
    handle = load_model(source=TINY_MODEL, device="cpu", dtype="float32")
    unload_model(handle)
    with pytest.raises(KeyError):
        unload_model(handle)


def test_model_metadata_returns_expected_keys() -> None:
    handle = load_model(source=TINY_MODEL, device="cpu", dtype="float32")
    meta = model_metadata(handle)
    assert "num_layers" in meta
    assert "num_heads" in meta
    assert "hidden_dim" in meta
    assert "module_tree" in meta
    assert isinstance(meta["num_layers"], int)
    assert isinstance(meta["module_tree"], list)
    assert len(meta["module_tree"]) > 0
    unload_model(handle)


def test_model_metadata_unknown_handle_raises() -> None:
    with pytest.raises(KeyError):
        model_metadata(99999)


def test_load_model_bad_source_raises() -> None:
    with pytest.raises(OSError, match="nonexistent"):
        load_model(source="/nonexistent/path/to/model", device="cpu", dtype="float32")
