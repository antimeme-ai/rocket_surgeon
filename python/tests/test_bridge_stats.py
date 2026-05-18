"""Tests for bridge tensor operations."""

from __future__ import annotations

import math

import torch

from rocket_surgeon.bridge import compute_tensor_stats, split_fused_output, tensor_to_bytes


def test_compute_tensor_stats_returns_expected_keys() -> None:
    t = torch.randn(4, 8)
    stats = compute_tensor_stats(t)
    expected_keys = {
        "mean",
        "std",
        "min",
        "max",
        "abs_max",
        "l2_norm",
        "sparsity",
        "shape",
        "dtype",
    }
    assert expected_keys.issubset(stats.keys())


def test_compute_tensor_stats_correct_values() -> None:
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    stats = compute_tensor_stats(t)
    assert math.isclose(stats["mean"], 2.5, rel_tol=1e-5)
    assert math.isclose(stats["min"], 1.0)
    assert math.isclose(stats["max"], 4.0)
    assert math.isclose(stats["abs_max"], 4.0)
    assert stats["shape"] == [4]
    assert stats["dtype"] == "float32"


def test_compute_tensor_stats_fp16_uses_fp32_for_reduction() -> None:
    t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
    stats = compute_tensor_stats(t)
    expected_mean = torch.tensor([1.0, 2.0, 3.0, 4.0]).float().mean().item()
    assert math.isclose(stats["mean"], expected_mean, rel_tol=1e-3)
    assert stats["dtype"] == "float16"


def test_compute_tensor_stats_bf16_uses_fp32_for_reduction() -> None:
    t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    stats = compute_tensor_stats(t)
    expected_mean = torch.tensor([1.0, 2.0, 3.0, 4.0]).float().mean().item()
    assert math.isclose(stats["mean"], expected_mean, rel_tol=1e-2)
    assert stats["dtype"] == "bfloat16"


def test_compute_tensor_stats_sparsity() -> None:
    t = torch.tensor([0.0, 1.0, 0.0, 2.0])
    stats = compute_tensor_stats(t)
    assert math.isclose(stats["sparsity"], 0.5, rel_tol=1e-5)


def test_compute_tensor_stats_population_std() -> None:
    t = torch.tensor([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    stats = compute_tensor_stats(t)
    expected_std = t.float().std(correction=0).item()
    assert math.isclose(stats["std"], expected_std, rel_tol=1e-5)


def test_split_fused_output_equal_chunks() -> None:
    t = torch.randn(2, 3, 12)
    parts = split_fused_output(t, dim=-1, sizes=[4, 4, 4])
    assert len(parts) == 3
    assert parts[0].shape == (2, 3, 4)
    assert parts[1].shape == (2, 3, 4)
    assert parts[2].shape == (2, 3, 4)
    assert torch.allclose(torch.cat(parts, dim=-1), t)


def test_split_fused_output_unequal_chunks() -> None:
    t = torch.randn(2, 3, 10)
    parts = split_fused_output(t, dim=-1, sizes=[6, 2, 2])
    assert len(parts) == 3
    assert parts[0].shape == (2, 3, 6)
    assert parts[1].shape == (2, 3, 2)
    assert parts[2].shape == (2, 3, 2)


def test_tensor_to_bytes_roundtrip() -> None:
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    data = tensor_to_bytes(t)
    assert isinstance(data, bytes)
    assert len(data) == 3 * 4
    reconstructed = torch.frombuffer(bytearray(data), dtype=torch.float32)
    assert torch.allclose(t, reconstructed)


def test_tensor_to_bytes_preserves_dtype() -> None:
    t = torch.tensor([1.0, 2.0], dtype=torch.float16)
    data = tensor_to_bytes(t)
    assert len(data) == 2 * 2
