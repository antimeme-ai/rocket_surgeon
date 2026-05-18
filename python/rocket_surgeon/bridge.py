"""Minimal Python bridge for PyTorch model operations.

Called from Rust worker via PyO3. No logic, no state management,
no IPC — just the thinnest possible bridge to PyTorch.
"""

from __future__ import annotations

import gc
from typing import Any

import torch
from transformers import AutoModelForCausalLM

_models: dict[int, torch.nn.Module] = {}
_next_handle: int = 1

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def load_model(source: str, device: str, dtype: str) -> int:
    """Load a model from *source* onto *device* with *dtype* and return an integer handle."""
    global _next_handle  # noqa: PLW0603
    torch_dtype = _DTYPE_MAP.get(dtype, torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
    )
    module: torch.nn.Module = model
    if device == "cpu":
        module = model.to(device)  # type: ignore[arg-type]
    handle = _next_handle
    _next_handle += 1
    _models[handle] = module
    return handle


def unload_model(handle: int) -> None:
    """Remove the model referenced by *handle* from the registry and free memory.

    Raises KeyError if *handle* is not registered.
    """
    model = _models.pop(handle)  # raises KeyError if missing
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_metadata(handle: int) -> dict[str, Any]:
    """Return a metadata dict for the model referenced by *handle*.

    Raises KeyError if *handle* is not registered.
    """
    model = _models[handle]  # raises KeyError if missing
    config = model.config
    num_layers: int = getattr(config, "num_hidden_layers", 0)
    num_heads: int = getattr(config, "num_attention_heads", 0)
    hidden_dim: int = getattr(config, "hidden_size", 0)
    module_tree = [name for name, _ in model.named_modules() if name]
    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "hidden_dim": hidden_dim,
        "module_tree": module_tree,
    }
