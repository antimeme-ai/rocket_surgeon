"""Built-in view computations for rocket_surgeon.

Called from Rust worker via PyO3. Each view is a pure function:
last_outputs -> computed result dict. No state, no hooks.
"""

from __future__ import annotations

from typing import Any

import torch

from . import bridge


def compute_view(
    model_handle: int,
    last_outputs: dict[tuple[str, int], Any],
    view_name: str,
    params: Any = None,
) -> dict[str, Any]:
    """Dispatch to the appropriate view computation."""
    if view_name == "residual_stream_norm":
        return _residual_stream_norm(model_handle, last_outputs)
    if view_name == "attention_pattern":
        if params is None:
            msg = "INVALID_PARAMS: attention_pattern requires params with 'layer'"
            raise ValueError(msg)
        layer = params.get("layer")
        if layer is None:
            msg = "INVALID_PARAMS: attention_pattern requires 'layer' in params"
            raise ValueError(msg)
        head = params.get("head")
        return _attention_pattern(model_handle, last_outputs, int(layer), head)
    msg = f"INVALID_PARAMS: unknown view '{view_name}'"
    raise ValueError(msg)


def _residual_stream_norm(
    model_handle: int,
    last_outputs: dict[tuple[str, int], Any],
) -> dict[str, Any]:
    model = bridge._models[model_handle]  # noqa: SLF001

    layer_paths: list[tuple[int, str]] = []
    layer_path_depth = 3
    for name, _ in model.named_modules():
        parts = name.split(".")
        if len(parts) == layer_path_depth and parts[0] == "model" and parts[1] == "layers":
            try:
                layer_idx = int(parts[2])
                layer_paths.append((layer_idx, name))
            except ValueError:
                continue

    layer_paths.sort(key=lambda x: x[0])

    norms: list[float] = []
    for _layer_idx, path in layer_paths:
        key = (path, 0)
        if key not in last_outputs:
            continue
        tensor = last_outputs[key]
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        norm_val = torch.norm(tensor.float(), p=2).item()
        norms.append(norm_val)

    if not norms:
        msg = "VIEW_DATA_UNAVAILABLE: no layer outputs found in last_outputs"
        raise ValueError(msg)

    return {
        "norms": norms,
        "num_layers": len(norms),
        "norm_type": "l2",
    }


def _attention_pattern(
    model_handle: int,
    last_outputs: dict[tuple[str, int], Any],
    layer: int,
    head: int | None = None,
) -> dict[str, Any]:
    model = bridge._models[model_handle]  # noqa: SLF001
    num_layers: int = getattr(model.config, "num_hidden_layers", 0)
    num_heads: int = getattr(model.config, "num_attention_heads", 0)
    attn_impl = getattr(model.config, "_attn_implementation", "eager")

    if attn_impl != "eager":
        msg = (
            f"CAPABILITY_NOT_SUPPORTED: attention weights not materialized — "
            f"model uses '{attn_impl}' attention. Set attn_implementation='eager' at attach."
        )
        raise ValueError(msg)

    if layer < 0 or layer >= num_layers:
        msg = f"INVALID_PARAMS: layer {layer} out of range (model has {num_layers} layers)"
        raise ValueError(msg)

    if head is not None and (head < 0 or head >= num_heads):
        msg = f"INVALID_PARAMS: head {head} out of range (layer has {num_heads} heads)"
        raise ValueError(msg)

    attn_path = f"model.layers.{layer}.self_attn"
    key = (attn_path, 0)

    if key not in last_outputs:
        msg = (
            f"VIEW_DATA_UNAVAILABLE: no attention output for layer {layer} in last_outputs. "
            f"Ensure barrier hooks cover self_attn modules."
        )
        raise ValueError(msg)

    output = last_outputs[key]
    min_attn_tuple_len = 2
    if not isinstance(output, tuple) or len(output) < min_attn_tuple_len:
        msg = (
            "VIEW_DATA_UNAVAILABLE: self_attn output is not a tuple with attention weights. "
            "Ensure output_attentions=True is set on the model config."
        )
        raise ValueError(msg)

    attn_weights = output[1]
    seq_len = attn_weights.shape[-1]

    if head is not None:
        heads_data = [
            {"head": head, "weights": attn_weights[0, head].detach().cpu().tolist()},
        ]
    else:
        heads_data = [
            {"head": h, "weights": attn_weights[0, h].detach().cpu().tolist()}
            for h in range(attn_weights.shape[1])
        ]

    return {
        "layer": layer,
        "heads": heads_data,
        "seq_len": int(seq_len),
    }
