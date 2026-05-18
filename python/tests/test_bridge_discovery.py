"""Tests for bridge discovery functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

from rocket_surgeon.bridge import (
    discover_execution_order,
    discover_modules,
    load_model,
    model_config,
    unload_model,
)

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture
def model_handle() -> Generator[int, None, None]:
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
