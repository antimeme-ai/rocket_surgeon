import time

import torch

from rocket_surgeon.host.interventions.callback import (
    InterventionContext,
    execute_callback,
)


def _identity(tensor: torch.Tensor, ctx: InterventionContext) -> torch.Tensor:
    return tensor


def _scale_by_two(tensor: torch.Tensor, ctx: InterventionContext) -> torch.Tensor:
    return tensor * 2


def _raise_error(tensor: torch.Tensor, ctx: InterventionContext) -> torch.Tensor:
    raise ValueError("intentional failure")


def _hang_forever(tensor: torch.Tensor, ctx: InterventionContext) -> torch.Tensor:
    time.sleep(100)
    return tensor


def _wrong_shape(tensor: torch.Tensor, ctx: InterventionContext) -> torch.Tensor:
    return tensor[0]


def _make_ctx(device: torch.device) -> InterventionContext:
    return InterventionContext(
        layer=0, component="mlp", event="output", tick_id=1, device=device, model_handle=0
    )


def test_callback_returns_modified_tensor():
    t = torch.ones(4, 8)
    ctx = _make_ctx(t.device)
    result, error = execute_callback(_scale_by_two, t, ctx, timeout_s=5.0, nan_check=False)
    assert result is not None
    assert torch.allclose(result, t * 2)
    assert error is None


def test_callback_identity_preserves_tensor():
    t = torch.randn(4, 8)
    ctx = _make_ctx(t.device)
    result, error = execute_callback(_identity, t, ctx, timeout_s=5.0, nan_check=False)
    assert result is not None
    assert torch.allclose(result, t)
    assert error is None


def test_callback_exception_returns_original():
    t = torch.ones(4, 8)
    ctx = _make_ctx(t.device)
    result, error = execute_callback(_raise_error, t, ctx, timeout_s=5.0, nan_check=False)
    assert result is None
    assert error is not None
    assert "intentional failure" in error


def test_callback_wrong_shape_returns_error():
    t = torch.ones(4, 8)
    ctx = _make_ctx(t.device)
    result, error = execute_callback(_wrong_shape, t, ctx, timeout_s=5.0, nan_check=False)
    assert result is None
    assert error is not None
    assert "shape" in error.lower()


def test_callback_timeout_returns_error():
    t = torch.ones(4, 8)
    ctx = _make_ctx(t.device)
    result, error = execute_callback(_hang_forever, t, ctx, timeout_s=0.1, nan_check=False)
    assert result is None
    assert error is not None
    assert "timeout" in error.lower()
