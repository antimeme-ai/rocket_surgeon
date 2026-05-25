"""Checkpoint tensor capture/restore bridge.

Called from Rust worker via PyO3. Wraps arena pointers as PyTorch
tensors for zero-copy CUDA DMA.

SAFETY: Arena pointers (dst_ptr/src_ptr) are raw addresses into the
mmap'd CheckpointArena owned by the Rust worker. The arena cannot be
dropped while these functions execute because dispatch is single-threaded:
the same serial dispatch loop that calls into Python via PyO3 owns the
arena reference, so the arena is alive for the duration of each call.
"""

from __future__ import annotations

import ctypes
import struct
from typing import Any

import numpy as np
import torch

_ELEMENT_SIZES = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}


def capture_activation(
    last_outputs: dict[tuple[str, int], Any],
    component_path: str,
    call_index: int,
    dst_ptr: int,
    dst_len: int,
) -> tuple[str, list[int]]:
    """Copy layer output from last_outputs into arena memory at dst_ptr."""
    key = (component_path, call_index)
    tensor = last_outputs[key]
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    t = tensor.detach().contiguous()
    nbytes = t.nelement() * t.element_size()
    if nbytes > dst_len:
        msg = f"tensor {nbytes} bytes exceeds slot capacity {dst_len}"
        raise ValueError(msg)
    buf = (ctypes.c_byte * dst_len).from_address(dst_ptr)
    cpu_view = torch.frombuffer(buf, dtype=t.dtype).reshape(t.shape)
    cpu_view.copy_(t)
    if t.is_cuda:
        torch.cuda.synchronize()
    del cpu_view
    return (str(t.dtype), list(t.shape))


def restore_activation(
    last_outputs: dict[tuple[str, int], Any],
    component_path: str,
    call_index: int,
    src_ptr: int,
    src_len: int,
    dtype_str: str,
    shape: list[int],
) -> None:
    """Copy activation from arena memory back into last_outputs tensor."""
    key = (component_path, call_index)
    torch_dtype = getattr(torch, dtype_str.replace("torch.", ""))
    nelement = 1
    for s in shape:
        nelement *= s
    nbytes = nelement * _ELEMENT_SIZES[torch_dtype]
    if nbytes > src_len:
        msg = f"restore tensor {nbytes} bytes exceeds slot capacity {src_len}"
        raise ValueError(msg)
    buf = (ctypes.c_byte * nbytes).from_address(src_ptr)
    cpu_view = torch.frombuffer(buf, dtype=torch_dtype).reshape(shape)
    target = last_outputs.get(key)
    if target is None:
        msg = f"no activation for {key} in last_outputs — cannot restore"
        raise KeyError(msg)
    if isinstance(target, tuple):
        target = target[0]
    target.copy_(cpu_view)
    if target.is_cuda:
        torch.cuda.synchronize()
    del cpu_view


def register_cuda_pinned(ptr: int, size: int) -> bool:
    """Register mmap'd memory with CUDA for pinned DMA."""
    if not torch.cuda.is_available():
        return False
    cudart = torch.cuda.cudart()  # type: ignore[no-untyped-call]
    result = cudart.cudaHostRegister(ptr, size, 0)
    return bool(result.value == 0 if hasattr(result, "value") else result == 0)


def unregister_cuda_pinned(ptr: int) -> bool:
    """Unregister mmap'd memory from CUDA. Call before munmap."""
    if not torch.cuda.is_available():
        return False
    cudart = torch.cuda.cudart()  # type: ignore[no-untyped-call]
    result = cudart.cudaHostUnregister(ptr)
    return bool(result.value == 0 if hasattr(result, "value") else result == 0)


def capture_rng_state() -> bytes:
    """Capture CUDA RNG state for all devices as length-prefixed raw bytes."""
    if not torch.cuda.is_available():
        return struct.pack("<I", 0)
    parts: list[bytes] = []
    device_count = torch.cuda.device_count()
    parts.append(struct.pack("<I", device_count))
    for i in range(device_count):
        rng_bytes = torch.cuda.get_rng_state(i).numpy().tobytes()
        parts.append(struct.pack("<II", i, len(rng_bytes)))
        parts.append(rng_bytes)
    return b"".join(parts)


def restore_rng_state(state: bytes) -> None:
    """Restore CUDA RNG state from bytes captured by capture_rng_state."""
    offset = 0
    (device_count,) = struct.unpack_from("<I", state, offset)
    offset += 4
    for _ in range(device_count):
        device_id, length = struct.unpack_from("<II", state, offset)
        offset += 8
        rng_bytes = state[offset : offset + length]
        offset += length
        t = torch.frombuffer(bytearray(rng_bytes), dtype=torch.uint8)
        torch.cuda.set_rng_state(t, device_id)


def capture_cpu_rng_state() -> bytes:
    """Capture CPU RNG state as raw bytes."""
    state = torch.random.get_rng_state()
    return bytes(state.numpy())


def restore_cpu_rng_state(state_bytes: bytes) -> None:
    """Restore CPU RNG state from bytes captured by capture_cpu_rng_state."""
    state = torch.from_numpy(np.frombuffer(state_bytes, dtype=np.uint8).copy())
    torch.random.set_rng_state(state)
