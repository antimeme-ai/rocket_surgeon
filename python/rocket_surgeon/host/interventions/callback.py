import ctypes
import importlib
import threading
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class InterventionContext:
    layer: int
    component: str
    event: str
    tick_id: int
    device: torch.device
    model_handle: int


_module_cache: dict[str, types.ModuleType] = {}


def invalidate_callback_cache(module_name: str | None = None) -> None:
    if module_name is None:
        _module_cache.clear()
    else:
        _module_cache.pop(module_name, None)


def resolve_callback(
    module_name: str, function_name: str, *, reload: bool = False
) -> Callable[..., Any]:
    if reload or module_name not in _module_cache:
        mod = importlib.import_module(module_name)
        if reload and module_name in _module_cache:
            importlib.reload(mod)
        _module_cache[module_name] = mod
    mod = _module_cache[module_name]
    fn: Callable[..., Any] = getattr(mod, function_name)
    return fn


def execute_callback(
    fn: Callable[..., Any],
    tensor: torch.Tensor,
    ctx: InterventionContext,
    timeout_s: float,
    nan_check: bool,
) -> tuple[torch.Tensor | None, str | None]:
    original = tensor.clone()
    result_holder: list[Any] = [None, None]
    thread_id_holder: list[int] = [0]

    def _run() -> None:
        thread_id_holder[0] = threading.current_thread().ident or 0
        try:
            result = fn(original, ctx)
            result_holder[0] = result
        except Exception as e:
            result_holder[1] = str(e)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)

    if worker.is_alive():
        tid = thread_id_holder[0]
        if tid:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid), ctypes.py_object(TimeoutError)
            )
        worker.join(timeout=timeout_s)
        suffix = "(uninterruptible)" if worker.is_alive() else ""
        return None, f"callback timeout after {timeout_s}s {suffix}".strip()

    if result_holder[1] is not None:
        return None, result_holder[1]

    result = result_holder[0]
    error = _validate_callback_result(result, tensor, nan_check)
    if error is not None:
        return None, error
    return result, None


def _validate_callback_result(
    result: Any,
    tensor: torch.Tensor,
    nan_check: bool,
) -> str | None:
    if result is None:
        return "callback returned None"
    if result.shape != tensor.shape:
        return f"shape mismatch: expected {tensor.shape}, got {result.shape}"
    if result.device != tensor.device:
        return f"device mismatch: expected {tensor.device}, got {result.device}"
    if nan_check and torch.isnan(result).any():
        return "callback output contains NaN"
    return None
