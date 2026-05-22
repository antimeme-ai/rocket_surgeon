"""Probe-point target matching for intervention recipes."""

from __future__ import annotations


def target_matches(
    *,
    target: str,
    family: str,
    rank: int,
    layer: int,
    component: str,
    event: str,
) -> bool:
    """Return True if a recipe target matches the current execution point.

    Target format: family:rank:layer:component:event
    Wildcards: '*' matches any single segment.
    """
    segments = target.split(":")
    expected_segments = 5
    if len(segments) != expected_segments:
        return False

    actual = [family, str(rank), str(layer), component, event]
    for pattern_seg, actual_seg in zip(segments, actual, strict=True):
        if pattern_seg == "*":
            continue
        if pattern_seg != actual_seg:
            return False
    return True
