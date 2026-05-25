"""Core intervention engine — filter, sort, apply."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import torch

from rocket_surgeon.host.interventions.callback import (
    InterventionContext,
    execute_callback,
    resolve_callback,
)
from rocket_surgeon.host.interventions.composition import filter_recipes, sort_by_priority

log = logging.getLogger(__name__)


def apply_interventions(
    *,
    tensor: torch.Tensor,
    recipes: list[dict[str, Any]],
    family: str,
    rank: int,
    layer: int,
    component: str,
    event: str,
    tensor_store: Callable[[str], torch.Tensor | None] | None = None,
    tick_id: int = 0,
    model_handle: int = 0,
) -> tuple[torch.Tensor, list[str]]:
    """Apply matching intervention recipes to a tensor.

    Returns (modified_tensor, list_of_fired_recipe_ids).
    The input tensor is NOT mutated; a clone is used for modifications.
    """
    matched = filter_recipes(
        recipes,
        family=family,
        rank=rank,
        layer=layer,
        component=component,
        event=event,
    )
    if not matched:
        return tensor, []

    sorted_recipes = sort_by_priority(matched)
    original = tensor.clone()
    current = tensor.clone()
    fired: list[str] = []

    for recipe in sorted_recipes:
        if recipe["mode"] == "replace":
            current = original.clone()

        current = _apply_single(
            current,
            recipe,
            layer,
            component,
            event,
            tensor_store,
            tick_id,
            model_handle,
        )
        fired.append(recipe["id"])

    return current, fired


def _apply_single(
    tensor: torch.Tensor,
    recipe: dict[str, Any],
    layer: int,
    component: str,
    event: str,
    tensor_store: Callable[[str], torch.Tensor | None] | None,
    tick_id: int,
    model_handle: int,
) -> torch.Tensor:
    """Apply a single recipe to tensor. Returns the (possibly new) tensor."""
    itype = recipe["intervention_type"]
    params = recipe["params"]

    if itype == "ablate":
        _apply_ablate(tensor, params)
    elif itype == "scale":
        tensor.mul_(params["factor"])
    elif itype == "add":
        _apply_add(tensor, params, tensor_store)
    elif itype == "patch":
        _apply_patch(tensor, params, tensor_store)
    elif itype == "clamp":
        tensor.clamp_(min=params["min"], max=params["max"])
    elif itype == "callback":
        return _apply_callback(tensor, params, layer, component, event, tick_id, model_handle)

    return tensor


def _apply_ablate(tensor: torch.Tensor, params: dict[str, Any]) -> None:
    mode = params.get("mode", "zero")
    if mode == "zero":
        tensor.zero_()
    elif mode == "mean":
        tensor.fill_(tensor.mean().item())
    elif mode == "resample":
        mean = tensor.mean().item()
        std = tensor.std().item()
        tensor.normal_(mean, std)


def _apply_add(
    tensor: torch.Tensor,
    params: dict[str, Any],
    tensor_store: Callable[[str], torch.Tensor | None] | None,
) -> None:
    vector = params["vector"]
    if isinstance(vector, list):
        add_tensor = torch.tensor(vector, dtype=tensor.dtype, device=tensor.device)
    else:
        if tensor_store is None:
            log.warning(
                "add intervention references tensor %s but no tensor_store provided",
                vector,
            )
            return
        resolved = tensor_store(vector)
        if resolved is None:
            log.warning("tensor_store returned None for id %s", vector)
            return
        add_tensor = resolved
    tensor.add_(add_tensor)


def _apply_patch(
    tensor: torch.Tensor,
    params: dict[str, Any],
    tensor_store: Callable[[str], torch.Tensor | None] | None,
) -> None:
    source_id = params["source_tensor_id"]
    if tensor_store is None:
        log.warning(
            "patch intervention references tensor %s but no tensor_store provided",
            source_id,
        )
        return
    source = tensor_store(source_id)
    if source is None:
        log.warning("tensor_store returned None for id %s", source_id)
        return
    tensor.copy_(source)


def _apply_callback(
    tensor: torch.Tensor,
    params: dict[str, Any],
    layer: int,
    component: str,
    event: str,
    tick_id: int,
    model_handle: int,
) -> torch.Tensor:
    module_name = params["module"]
    function_name = params["function"]
    timeout_s = params.get("timeout_s", 5.0)
    nan_check = params.get("nan_check", False)
    fn = resolve_callback(module_name, function_name)
    ctx = InterventionContext(
        layer=layer,
        component=component,
        event=event,
        tick_id=tick_id,
        device=tensor.device,
        model_handle=model_handle,
    )
    result, error = execute_callback(fn, tensor, ctx, timeout_s, nan_check)
    if result is not None:
        return result
    log.warning("callback intervention failed: %s", error)
    return tensor
