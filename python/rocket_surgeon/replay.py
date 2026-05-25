import ctypes

import torch


def compare_activations(
    original: torch.Tensor,
    replayed: torch.Tensor,
    cosine_threshold: float,
    mre_threshold: float,
) -> dict[str, float] | None:
    original_flat = original.flatten().float()
    replayed_flat = replayed.flatten().float()

    dot = torch.dot(original_flat, replayed_flat)
    norm_a = torch.linalg.norm(original_flat)
    norm_b = torch.linalg.norm(replayed_flat)
    denom = norm_a * norm_b
    if denom == 0:
        cosine_sim = 1.0 if torch.equal(original_flat, replayed_flat) else 0.0
    else:
        cosine_sim = (dot / denom).item()

    abs_diff = torch.abs(original_flat - replayed_flat)
    abs_orig = torch.abs(original_flat)
    epsilon = 1e-8
    relative_error = abs_diff / (abs_orig + epsilon)
    max_rel_error = relative_error.max().item()

    if cosine_sim < cosine_threshold or max_rel_error > mre_threshold:
        return {
            "cosine_similarity": cosine_sim,
            "max_relative_error": max_rel_error,
        }
    return None


def compare_activations_from_ptr(
    original_ptr: int,
    original_len: int,
    original_dtype: str,
    original_shape: list[int],
    replayed: torch.Tensor,
    cosine_threshold: float,
    mre_threshold: float,
) -> dict[str, float] | None:
    dtype_map = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }
    dtype = dtype_map.get(original_dtype)
    if dtype is None:
        msg = f"unsupported dtype for divergence comparison: {original_dtype}"
        raise ValueError(msg)
    buf = (ctypes.c_char * original_len).from_address(original_ptr)
    original = torch.frombuffer(buf, dtype=dtype).reshape(original_shape)
    result = compare_activations(original, replayed, cosine_threshold, mre_threshold)
    del original
    return result
