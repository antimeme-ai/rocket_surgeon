"""Intervention engine — applies surgical modifications to tensors during forward pass."""

from __future__ import annotations

from rocket_surgeon.host.interventions.engine import apply_interventions
from rocket_surgeon.host.interventions.recipes import RecipeError, parse_recipe

__all__ = ["RecipeError", "apply_interventions", "parse_recipe"]
