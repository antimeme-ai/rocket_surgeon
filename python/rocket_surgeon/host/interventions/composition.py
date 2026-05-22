"""Priority sorting and composition semantics for intervention recipes."""

from __future__ import annotations

import logging
from typing import Any

from rocket_surgeon.host.interventions.matching import target_matches

log = logging.getLogger(__name__)


def filter_recipes(
    recipes: list[dict[str, Any]],
    *,
    family: str,
    rank: int,
    layer: int,
    component: str,
    event: str,
) -> list[dict[str, Any]]:
    """Return recipes whose target matches the current point.

    Skips recipes with a `condition` field (reserved for Phase 3).
    """
    matched = []
    for recipe in recipes:
        if recipe.get("condition") is not None:
            log.debug("skipping conditional recipe %s (Phase 3)", recipe.get("id"))
            continue
        if target_matches(
            target=recipe["target"],
            family=family,
            rank=rank,
            layer=layer,
            component=component,
            event=event,
        ):
            matched.append(recipe)
    return matched


def sort_by_priority(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort recipes by priority ascending (lower = first). Stable sort."""
    return sorted(recipes, key=lambda r: r.get("priority", 0))
