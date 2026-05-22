"""Recipe deserialization and validation."""

from __future__ import annotations

from typing import Any

VALID_TYPES = frozenset({"ablate", "scale", "add", "patch", "clamp"})
VALID_MODES = frozenset({"additive", "replace"})
VALID_ABLATE_MODES = frozenset({"zero", "mean", "resample"})


class RecipeError(ValueError):
    pass


def parse_recipe(raw: dict[str, Any]) -> dict[str, Any]:
    """Parse and validate a raw recipe dict from JSON.

    Returns a normalized dict with guaranteed fields:
    id, intervention_type, target, params, priority, mode, condition.
    """
    recipe_id = raw.get("id")
    if not recipe_id:
        msg = "recipe missing required field: id"
        raise RecipeError(msg)

    intervention_type = raw.get("type", "")
    if intervention_type not in VALID_TYPES:
        msg = f"unknown intervention type: {intervention_type!r}"
        raise RecipeError(msg)

    target = raw.get("target", "")
    params = raw.get("params", {})
    _validate_params(intervention_type, params)

    priority = raw.get("priority", 0)
    mode = raw.get("mode", "additive")
    if mode not in VALID_MODES:
        msg = f"invalid composition mode: {mode!r}"
        raise RecipeError(msg)

    return {
        "id": recipe_id,
        "intervention_type": intervention_type,
        "target": target,
        "params": params,
        "priority": priority,
        "mode": mode,
        "condition": raw.get("condition"),
    }


def _validate_params(intervention_type: str, params: dict[str, Any]) -> None:
    if intervention_type == "scale":
        if "factor" not in params:
            msg = "scale intervention requires 'factor' in params"
            raise RecipeError(msg)
    elif intervention_type == "add":
        if "vector" not in params:
            msg = "add intervention requires 'vector' in params"
            raise RecipeError(msg)
    elif intervention_type == "patch":
        if "source_tensor_id" not in params:
            msg = "patch intervention requires 'source_tensor_id' in params"
            raise RecipeError(msg)
    elif intervention_type == "clamp":
        if "min" not in params or "max" not in params:
            msg = "clamp intervention requires 'min' and 'max' in params"
            raise RecipeError(msg)
    elif intervention_type == "ablate":
        mode = params.get("mode", "zero")
        if mode not in VALID_ABLATE_MODES:
            msg = f"invalid ablate mode: {mode!r}"
            raise RecipeError(msg)
