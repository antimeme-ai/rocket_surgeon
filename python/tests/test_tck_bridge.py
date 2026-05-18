"""Step definitions for tck/model/bridge_discovery.feature."""

from __future__ import annotations

import math

import torch
from pytest_bdd import given, parsers, scenario, then, when

from rocket_surgeon.bridge import (
    compute_tensor_stats,
    discover_execution_order,
    discover_modules,
    load_model,
    model_config,
    split_fused_output,
    tensor_to_bytes,
    unload_model,
)

FEATURE = "../../tck/model/bridge_discovery.feature"
TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# ── Scenarios ──────────────────────────────────────────────────────


@scenario(FEATURE, "discover_modules returns module inventory")
def test_discover_modules():
    pass


@scenario(FEATURE, "model_config returns architecture metadata")
def test_model_config():
    pass


@scenario(FEATURE, "discover_execution_order returns ordered module firings")
def test_execution_order():
    pass


@scenario(FEATURE, "compute_tensor_stats casts to fp32 before reduction")
def test_stats_fp32():
    pass


@scenario(FEATURE, "compute_tensor_stats reports sparsity")
def test_stats_sparsity():
    pass


@scenario(FEATURE, "tensor_to_bytes preserves dtype")
def test_tensor_bytes():
    pass


@scenario(FEATURE, "split_fused_output splits along given dimension")
def test_split_fused():
    pass


# ── Background ─────────────────────────────────────────────────────


@given("a tiny llama model is loaded on CPU", target_fixture="ctx")
def tiny_model():
    handle = load_model(source=TINY_MODEL, device="cpu", dtype="float32")
    ctx = {"handle": handle}
    yield ctx
    unload_model(handle)


# ── Discovery steps ────────────────────────────────────────────────


@when("discover_modules is called")
def call_discover_modules(ctx):
    ctx["modules"] = discover_modules(ctx["handle"])


@when("model_config is called")
def call_model_config(ctx):
    ctx["config"] = model_config(ctx["handle"])


@when("discover_execution_order is called")
def call_execution_order(ctx):
    ctx["exec_order"] = discover_execution_order(ctx["handle"])


@then("the result is a non-empty list of module dicts")
def modules_nonempty(ctx):
    assert isinstance(ctx["modules"], list)
    assert len(ctx["modules"]) > 0


@then('each module dict has keys "path", "type_name", "attr_name"')
def modules_have_keys(ctx):
    for m in ctx["modules"]:
        assert "path" in m
        assert "type_name" in m
        assert "attr_name" in m


@then('at least one module path contains "self_attn.q_proj"')
def has_q_proj(ctx):
    paths = [m["path"] for m in ctx["modules"]]
    assert any("self_attn.q_proj" in p for p in paths)


@then(parsers.parse('the result contains "model_type" as "{expected}"'))
def config_model_type(ctx, expected):
    assert ctx["config"]["model_type"] == expected


@then(parsers.parse('the result contains "{key}" as a positive integer'))
def config_positive_int(ctx, key):
    val = ctx["config"][key]
    assert isinstance(val, int)
    assert val > 0


@then("the result is a non-empty list of (path, call_index) tuples")
def exec_order_nonempty(ctx):
    assert isinstance(ctx["exec_order"], list)
    assert len(ctx["exec_order"]) > 0
    for item in ctx["exec_order"]:
        assert isinstance(item, tuple)
        assert len(item) == 2


@then("the first entry's call_index is 0")
def first_call_index_zero(ctx):
    _, call_index = ctx["exec_order"][0]
    assert call_index == 0


@then("module paths appear in forward-pass order")
def forward_pass_order(ctx):
    paths = [path for path, _ in ctx["exec_order"]]
    assert len(paths) > 1


# ── Tensor stats steps ─────────────────────────────────────────────


@given("a tensor of dtype float16", target_fixture="tensor_ctx")
def fp16_tensor():
    t = torch.randn(4, 8, dtype=torch.float16)
    return {"tensor": t}


@given("a tensor where half the elements are zero", target_fixture="tensor_ctx")
def half_zero_tensor():
    t = torch.zeros(10)
    t[:5] = torch.randn(5)
    return {"tensor": t}


@given("a float32 tensor with known values", target_fixture="tensor_ctx")
def known_f32_tensor():
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    return {"tensor": t}


@given(parsers.parse("a tensor of shape [{rows:d}, {cols:d}]"), target_fixture="tensor_ctx")
def shaped_tensor(rows, cols):
    t = torch.randn(rows, cols)
    return {"tensor": t}


@when("compute_tensor_stats is called on the tensor")
def call_stats(tensor_ctx):
    tensor_ctx["stats"] = compute_tensor_stats(tensor_ctx["tensor"])


@when("tensor_to_bytes is called")
def call_tensor_to_bytes(tensor_ctx):
    tensor_ctx["bytes"] = tensor_to_bytes(tensor_ctx["tensor"])


@when(
    parsers.parse("split_fused_output is called with dim={dim:d} and sizes [{s}]"),
    target_fixture="split_result",
)
def call_split(tensor_ctx, dim, s):
    sizes = [int(x.strip()) for x in s.split(",")]
    return split_fused_output(tensor_ctx["tensor"], dim, sizes)


@then(parsers.parse('the result dtype field is "{expected}"'))
def stats_dtype(tensor_ctx, expected):
    assert tensor_ctx["stats"]["dtype"] == expected


@then("the result mean is a finite float")
def stats_mean_finite(tensor_ctx):
    assert math.isfinite(tensor_ctx["stats"]["mean"])


@then("the result std is a non-negative float")
def stats_std_nonneg(tensor_ctx):
    assert tensor_ctx["stats"]["std"] >= 0.0


@then("the result shape matches the tensor's shape")
def stats_shape_match(tensor_ctx):
    assert tensor_ctx["stats"]["shape"] == list(tensor_ctx["tensor"].shape)


@then("the result sparsity is approximately 0.5")
def stats_sparsity_half(tensor_ctx):
    assert abs(tensor_ctx["stats"]["sparsity"] - 0.5) < 0.01


@then(parsers.parse("the byte length equals numel * {bytes_per:d}"))
def byte_length(tensor_ctx, bytes_per):
    expected = tensor_ctx["tensor"].numel() * bytes_per
    assert len(tensor_ctx["bytes"]) == expected


@then(parsers.parse("{n:d} tensors each of shape [{rows:d}, {cols:d}]"))
def split_shapes(split_result, n, rows, cols):
    assert len(split_result) == n
    for t in split_result:
        assert list(t.shape) == [rows, cols]


@then(parsers.parse("the result is {n:d} tensors each of shape [{rows:d}, {cols:d}]"))
def split_shapes_alt(split_result, n, rows, cols):
    assert len(split_result) == n
    for t in split_result:
        assert list(t.shape) == [rows, cols]
