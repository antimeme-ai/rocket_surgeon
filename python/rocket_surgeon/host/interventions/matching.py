"""Probe-point target matching for intervention recipes."""

from __future__ import annotations

import re

_BRACKET_RE = re.compile(r"\[(\d+)\]$")


def strip_bracket(segment: str) -> str:
    return _BRACKET_RE.sub("", segment)


def extract_head_index(segment: str) -> int | None:
    m = _BRACKET_RE.search(segment)
    return int(m.group(1)) if m else None


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
    Bracket notation on component (e.g., o_proj[7]) is stripped before
    matching — head slicing is handled by the caller.
    """
    segments = target.split(":")
    expected_segments = 5
    if len(segments) != expected_segments:
        return False

    actual = [family, str(rank), str(layer), component, event]
    for pattern_seg, actual_seg in zip(segments, actual, strict=True):
        if pattern_seg == "*":
            continue
        if strip_bracket(pattern_seg) != actual_seg:
            return False
    return True
