"""Minimal Python bridge for PyTorch model operations.

Called from Rust worker via PyO3. No logic, no state management,
no IPC — just the thinnest possible bridge to PyTorch.
"""

from __future__ import annotations

import gc
import threading
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
    attn_impl = getattr(module.config, "_attn_implementation", "eager")
    if attn_impl == "eager":
        module.config.output_attentions = True  # type: ignore[union-attr]
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


def discover_modules(handle: int) -> list[dict[str, Any]]:
    """Walk model.named_modules() and return module inventory.

    Each entry: {path, type_name, attr_name}.
    """
    model = _models[handle]
    result = []
    for name, module in model.named_modules():
        if not name:
            continue
        type_name = type(module).__name__
        attr_name = name.rsplit(".", 1)[-1]
        result.append(
            {
                "path": name,
                "type_name": type_name,
                "attr_name": attr_name,
            }
        )
    return result


def model_config(handle: int) -> dict[str, Any]:
    """Extract model configuration attributes."""
    model = _models[handle]
    config = model.config
    return {
        "model_type": getattr(config, "model_type", "unknown"),
        "num_layers": getattr(config, "num_hidden_layers", 0),
        "num_heads": getattr(config, "num_attention_heads", 0),
        "hidden_size": getattr(config, "hidden_size", 0),
        "num_kv_heads": getattr(config, "num_key_value_heads", None),
    }


def discover_execution_order(handle: int) -> list[tuple[str, int]]:
    """Run a tracing forward pass and record hook firing order.

    Returns ordered list of (module_path, call_index) pairs.
    """
    model = _models[handle]
    call_counts: dict[str, int] = {}
    order: list[tuple[str, int]] = []
    handles: list[Any] = []

    def make_hook(path: str) -> Any:
        def hook(_module: Any, _input: Any, _output: Any) -> None:
            idx = call_counts.get(path, 0)
            call_counts[path] = idx + 1
            order.append((path, idx))

        return hook

    for name, module in model.named_modules():
        if not name:
            continue
        h = module.register_forward_hook(make_hook(name))
        handles.append(h)

    with torch.inference_mode():
        dummy_input = torch.zeros(1, 2, dtype=torch.long, device=next(model.parameters()).device)
        model(dummy_input)

    for h in handles:
        h.remove()

    return order


_DTYPE_NAME_MAP: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def compute_tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    """Compute summary stats on a tensor, casting to fp32 for reduction accuracy."""
    original_dtype = tensor.dtype
    t = tensor.detach().float()
    numel = t.numel()
    if numel == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "abs_max": 0.0,
            "l2_norm": 0.0,
            "sparsity": 0.0,
            "shape": list(tensor.shape),
            "dtype": _DTYPE_NAME_MAP.get(original_dtype, str(original_dtype)),
        }
    return {
        "mean": t.mean().item(),
        "std": t.std(correction=0).item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "abs_max": t.abs().max().item(),
        "l2_norm": t.norm(2).item(),
        "sparsity": (t == 0).sum().item() / numel,
        "shape": list(tensor.shape),
        "dtype": _DTYPE_NAME_MAP.get(original_dtype, str(original_dtype)),
    }


def split_fused_output(tensor: torch.Tensor, dim: int, sizes: list[int]) -> list[torch.Tensor]:
    """Split a fused module output tensor along the given dimension."""
    return list(tensor.split(sizes, dim=dim))  # type: ignore[no-untyped-call]


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize tensor to raw bytes. Dtype-preserving except bf16 → fp16 (NumPy has no bf16)."""
    t = tensor.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float16)
    return t.numpy().tobytes()


def install_passive_hooks(
    handle: int,
    module_paths: list[str],
    storage: dict[tuple[str, int], Any],
) -> list[Any]:
    """Install plain forward hooks that stash container outputs without barriers.

    Each hook writes (path, 0) -> output into *storage*. No mailbox,
    no barrier, no tick counting.
    """
    model = _models[handle]
    modules_by_path = dict(model.named_modules())
    handles: list[Any] = []
    for path in module_paths:
        module = modules_by_path.get(path)
        if module is None:
            continue

        def make_hook(p: str) -> Any:
            def hook(_mod: Any, _inp: Any, output: Any) -> None:
                storage[(p, 0)] = output

            return hook

        h = module.register_forward_hook(make_hook(path))
        handles.append(h)
    return handles


def install_sentinel_hooks(handle: int, module_paths: list[str]) -> list[Any]:
    """Install no-op sentinel hooks on specified modules to defeat PyTorch's fast path."""
    model = _models[handle]
    modules_by_path = dict(model.named_modules())
    handles: list[Any] = []
    for path in module_paths:
        module = modules_by_path.get(path)
        if module is not None:
            h = module.register_forward_hook(lambda _m, _i, _o: None)
            handles.append(h)
    return handles


def install_capture_hooks(
    handle: int,
    module_paths: list[str],
    result_mailbox: Any,
    resume_mailbox: Any,
    active_probes: set[str] | None = None,
) -> tuple[list[Any], dict[str, int]]:
    """Install capture hooks with mailbox barrier on specified modules.

    Returns (handles, call_counts). Caller should call call_counts.clear()
    between forward passes to reset per-module call indices.
    """
    model = _models[handle]
    modules_by_path = dict(model.named_modules())
    handles: list[Any] = []
    call_counts: dict[str, int] = {}

    if active_probes is None:
        active_probes = set()

    for path in module_paths:
        module = modules_by_path.get(path)
        if module is None:
            continue

        def make_hook(p: str) -> Any:
            def hook(_mod: Any, _inp: Any, output: Any) -> Any:
                if p not in active_probes:
                    return None

                idx = call_counts.get(p, 0)
                call_counts[p] = idx + 1

                result_mailbox.put((p, idx, output))
                intervention = resume_mailbox.wait()
                resume_mailbox.restore()

                if intervention is not None:
                    return intervention
                return None

            return hook

        h = module.register_forward_hook(make_hook(path), prepend=True)
        handles.append(h)
    return handles, call_counts


def remove_hooks(handles: list[Any]) -> None:
    """Remove all hooks referenced by the given handles."""
    for h in handles:
        h.remove()


def run_forward(
    handle: int,
    input_ids: torch.Tensor,
    done_callback: Any,
) -> None:
    """Spawn a thread that runs model(input_ids) and calls done_callback on completion."""
    model = _models[handle]

    def _run() -> None:
        try:
            with torch.inference_mode():
                model(input_ids)
            done_callback(None)
        except Exception as e:
            done_callback(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
