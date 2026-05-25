import torch

from rocket_surgeon.replay import compare_activations


def test_identical_tensors_no_divergence():
    a = torch.randn(32, 128)
    result = compare_activations(a, a.clone(), cosine_threshold=0.999, mre_threshold=0.05)
    assert result is None


def test_different_tensors_reports_divergence():
    a = torch.randn(32, 128)
    b = torch.randn(32, 128)
    result = compare_activations(a, b, cosine_threshold=0.999, mre_threshold=0.05)
    assert result is not None
    assert "cosine_similarity" in result
    assert "max_relative_error" in result
    assert result["cosine_similarity"] < 0.999


def test_slightly_perturbed_within_tolerance():
    a = torch.randn(32, 128)
    noise = torch.randn_like(a) * 1e-5
    b = a + noise
    result = compare_activations(a, b, cosine_threshold=0.999, mre_threshold=0.05)
    assert result is None


def test_scaled_tensor_exceeds_mre():
    a = torch.ones(32, 128)
    b = a * 1.1  # 10% relative error
    result = compare_activations(a, b, cosine_threshold=0.0, mre_threshold=0.05)
    assert result is not None
    assert result["max_relative_error"] > 0.05
