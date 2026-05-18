"""Tests for hook installation, barrier cycling, and forward pass lifecycle."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest
import torch

from rocket_surgeon.bridge import (
    _models,
    discover_modules,
    install_capture_hooks,
    install_sentinel_hooks,
    load_model,
    remove_hooks,
    run_forward,
    unload_model,
)
from rocket_surgeon.hooks.mailbox import Mailbox

if TYPE_CHECKING:
    from collections.abc import Generator

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture
def model_handle() -> Generator[int, None, None]:
    handle = load_model(source=TINY_MODEL, device="cpu", dtype="float32")
    yield handle
    unload_model(handle)


def test_install_sentinel_hooks_returns_handles(model_handle: int) -> None:
    modules = discover_modules(model_handle)
    paths = [m["path"] for m in modules]
    handles = install_sentinel_hooks(model_handle, paths)
    assert isinstance(handles, list)
    assert len(handles) == len(paths)
    remove_hooks(handles)


def test_install_capture_hooks_returns_handles(model_handle: int) -> None:
    result_mb = Mailbox()
    resume_mb = Mailbox()
    paths = ["model.layers.0.self_attn.q_proj"]
    handles, _call_counts = install_capture_hooks(
        model_handle,
        paths,
        result_mb,
        resume_mb,
        active_probes={"model.layers.0.self_attn.q_proj"},
    )
    assert isinstance(handles, list)
    assert len(handles) == 1
    remove_hooks(handles)


def test_capture_hook_barrier_cycle(model_handle: int) -> None:
    """Full barrier cycle: hook fires, puts result, blocks, gets resumed."""
    result_mb = Mailbox()
    resume_mb = Mailbox()
    target_path = "model.layers.0.self_attn.q_proj"

    modules = discover_modules(model_handle)
    all_paths = [m["path"] for m in modules]
    sentinel_handles = install_sentinel_hooks(model_handle, all_paths)
    capture_handles, _call_counts = install_capture_hooks(
        model_handle,
        [target_path],
        result_mb,
        resume_mb,
        active_probes={target_path},
    )

    captured: list[tuple[str, int]] = []
    errors: list[str] = []

    def forward_thread() -> None:
        try:
            model = _models[model_handle]
            with torch.inference_mode():
                dummy = torch.zeros(1, 2, dtype=torch.long)
                model(dummy)
        except Exception as e:
            errors.append(str(e))

    fwd = threading.Thread(target=forward_thread)
    fwd.start()

    value = result_mb.wait()
    assert value is not None
    path, call_index, tensor = value
    assert path == target_path
    assert isinstance(call_index, int)
    assert isinstance(tensor, torch.Tensor)
    captured.append((path, call_index))
    result_mb.restore()

    resume_mb.put(None)

    fwd.join(timeout=10.0)
    assert not fwd.is_alive(), "Forward thread did not complete"
    assert len(errors) == 0, f"Forward thread errors: {errors}"
    assert len(captured) == 1

    remove_hooks(sentinel_handles + capture_handles)


def test_capture_hook_call_index_resets_after_clear(model_handle: int) -> None:
    """call_counts dict accumulates across passes; caller must .clear() to reset."""
    result_mb = Mailbox()
    resume_mb = Mailbox()
    target_path = "model.layers.0.self_attn.q_proj"

    modules = discover_modules(model_handle)
    all_paths = [m["path"] for m in modules]
    sentinel_handles = install_sentinel_hooks(model_handle, all_paths)
    capture_handles, call_counts = install_capture_hooks(
        model_handle,
        [target_path],
        result_mb,
        resume_mb,
        active_probes={target_path},
    )

    def drain_one_pass() -> int:
        """Run one forward pass, drain all barrier events, return last call_index."""
        last_idx = -1

        def fwd() -> None:
            model = _models[model_handle]
            with torch.inference_mode():
                model(torch.zeros(1, 2, dtype=torch.long))

        t = threading.Thread(target=fwd)
        t.start()
        value = result_mb.wait()
        assert value is not None
        _, idx, _ = value
        last_idx = idx
        result_mb.restore()
        resume_mb.put(None)
        t.join(timeout=10.0)
        return last_idx

    idx_pass1 = drain_one_pass()
    assert idx_pass1 == 0

    idx_pass2_no_clear = drain_one_pass()
    assert idx_pass2_no_clear == 1, "Without clear, call_index should accumulate"

    call_counts.clear()

    idx_pass3_after_clear = drain_one_pass()
    assert idx_pass3_after_clear == 0, "After clear, call_index should reset to 0"

    remove_hooks(sentinel_handles + capture_handles)


def test_sentinel_hook_returns_none(model_handle: int) -> None:
    """Sentinel hooks should return None (don't modify), not the output tensor."""
    modules = discover_modules(model_handle)
    paths = [m["path"] for m in modules]
    handles = install_sentinel_hooks(model_handle, paths)

    model = _models[model_handle]
    first_module = dict(model.named_modules())[paths[0]]
    for hook_ref in first_module._forward_hooks.values():
        dummy_out = torch.randn(2, 3)
        result = hook_ref(None, None, dummy_out)
        assert result is None, f"Sentinel hook should return None, got {type(result)}"
        break

    remove_hooks(handles)


def test_remove_hooks_cleans_up(model_handle: int) -> None:
    modules = discover_modules(model_handle)
    paths = [m["path"] for m in modules]
    handles = install_sentinel_hooks(model_handle, paths)
    remove_hooks(handles)


def test_run_forward_calls_done_callback(model_handle: int) -> None:
    done_event = threading.Event()
    error_ref: list[Exception | None] = [None]

    def done_callback(error: Exception | None) -> None:
        error_ref[0] = error
        done_event.set()

    input_ids = torch.zeros(1, 2, dtype=torch.long)
    run_forward(model_handle, input_ids, done_callback)
    done_event.wait(timeout=10.0)
    assert done_event.is_set()
    assert error_ref[0] is None


def test_run_forward_reports_error_on_bad_input(model_handle: int) -> None:
    done_event = threading.Event()
    error_ref: list[Exception | None] = [None]

    def done_callback(error: Exception | None) -> None:
        error_ref[0] = error
        done_event.set()

    bad_input = torch.zeros(0, dtype=torch.long)
    run_forward(model_handle, bad_input, done_callback)
    done_event.wait(timeout=10.0)
    assert done_event.is_set()
    assert error_ref[0] is not None
