"""Property / model-based / exception-raising tests for intervention target matching.

MATERIA oracle tiers exercised:
  * tier-6 model       — target_matches agrees with an INDEPENDENT reference
                         matcher (built with rpartition, not the production regex)
                         across generated targets and execution points.
  * tier-4 roundtrip   — strip_bracket / extract_head_index invert the
                         "base[index]" construction.
  * tier-2 exception   — malformed targets (wrong segment count, junk) return
                         False, never raise.

Generator distribution annotated with ``hypothesis.event``; inspect with
``--hypothesis-show-statistics``.
"""

from __future__ import annotations

from hypothesis import event, given, settings
from hypothesis import strategies as st

from rocket_surgeon.host.interventions.matching import (
    extract_head_index,
    strip_bracket,
    target_matches,
)

# identifier-ish base segments that never themselves end in "[digits]"
_BASE = st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,7}", fullmatch=True)
_FIELD = st.from_regex(r"[a-zA-Z0-9_]{1,8}", fullmatch=True)


# --------------------------------------------------------------------------- #
# Independent reference model for target_matches
# --------------------------------------------------------------------------- #
def _ref_strip(seg: str) -> str:
    """Reference: drop a single trailing ``[<digits>]`` group, if present.

    Implemented with rpartition (production uses an anchored regex) so this is a
    genuinely independent oracle, not a transcription.
    """
    if seg.endswith("]") and "[" in seg:
        base, _, inner = seg[:-1].rpartition("[")
        if inner.isdigit():
            return base
    return seg


def _ref_matches(target: str, fields: tuple[str, str, str, str, str]) -> bool:
    segs = target.split(":")
    if len(segs) != 5:
        return False
    for pat, act in zip(segs, fields, strict=True):
        if pat == "*":
            continue
        if _ref_strip(pat) != act:
            return False
    return True


@st.composite
def _target_and_point(
    draw: st.DrawFn,
) -> tuple[str, tuple[str, str, str, str, str]]:
    """Generate a (target, execution-point) pair, biased toward interesting cases:
    sometimes a guaranteed match built from the point, sometimes pure noise."""
    family = draw(_FIELD)
    rank = str(draw(st.integers(0, 7)))
    layer = str(draw(st.integers(0, 47)))
    component = draw(_BASE)
    ev = draw(st.sampled_from(["input", "output", "grad"]))
    point = (family, rank, layer, component, ev)

    mode = draw(st.sampled_from(["exact", "wildcard", "bracket", "noise", "wrong_count"]))
    if mode == "exact":
        target = ":".join(point)
    elif mode == "wildcard":
        segs = [draw(st.sampled_from([p, "*"])) for p in point]
        target = ":".join(segs)
    elif mode == "bracket":
        idx = draw(st.integers(0, 31))
        target = f"{family}:{rank}:{layer}:{component}[{idx}]:{ev}"
    elif mode == "wrong_count":
        target = ":".join(draw(st.lists(_FIELD, min_size=0, max_size=8)))
    else:  # noise
        target = ":".join(draw(st.lists(st.one_of(_FIELD, st.just("*")), max_size=6)))
    return target, point


# --------------------------------------------------------------------------- #
# Model-based property
# --------------------------------------------------------------------------- #
@given(_target_and_point())
@settings(max_examples=600)
def test_target_matches_agrees_with_reference(
    pair: tuple[str, tuple[str, str, str, str, str]],
) -> None:
    """Model oracle: target_matches == independent reference matcher for all
    generated (target, point) pairs."""
    target, point = pair
    family, rank, layer, component, ev = point
    actual = target_matches(
        target=target,
        family=family,
        rank=int(rank),
        layer=int(layer),
        component=component,
        event=ev,
    )
    expected = _ref_matches(target, point)
    event("match" if expected else "no-match")
    event(f"segments: {len(target.split(':'))}")
    assert actual == expected


# --------------------------------------------------------------------------- #
# strip_bracket / extract_head_index roundtrip + edges
# --------------------------------------------------------------------------- #
@given(_BASE, st.integers(0, 10**6))
@settings(max_examples=300)
def test_bracket_construction_roundtrips(base: str, idx: int) -> None:
    """Roundtrip: for base[idx], strip recovers base and extract recovers idx."""
    seg = f"{base}[{idx}]"
    event(f"idx digits: {len(str(idx))}")
    assert strip_bracket(seg) == base
    assert extract_head_index(seg) == idx


@given(_BASE)
@settings(max_examples=200)
def test_no_bracket_is_identity_and_none(base: str) -> None:
    """Edge: a segment without a trailing [digits] is unchanged by strip and has
    no head index."""
    assert strip_bracket(base) == base
    assert extract_head_index(base) is None


@given(_BASE, st.sampled_from(["[]", "[", "[-1]", "[x]", "[1", "1]"]))
@settings(max_examples=200)
def test_malformed_bracket_extracts_none(base: str, suffix: str) -> None:
    """Edge: malformed bracket suffixes ([], [-1], [x], unbalanced) yield no index
    (extract returns None) — only a well-formed trailing [<digits>] counts."""
    seg = base + suffix
    event(f"suffix: {suffix}")
    assert extract_head_index(seg) is None


# --------------------------------------------------------------------------- #
# Exception-raising / robustness
# --------------------------------------------------------------------------- #
@given(st.lists(_FIELD, min_size=0, max_size=8).map(":".join))
@settings(max_examples=300)
def test_wrong_segment_count_returns_false_not_raise(target: str) -> None:
    """E1: a target without exactly 5 segments returns False and never raises."""
    n = len(target.split(":"))
    event(f"segments: {n}")
    result = target_matches(
        target=target, family="gpt2", rank=0, layer=0, component="x", event="output"
    )
    if n != 5:
        assert result is False


@given(st.text(max_size=30))
@settings(max_examples=300)
def test_arbitrary_text_never_raises(target: str) -> None:
    """E2: target_matches tolerates arbitrary text input — returns a bool, no
    exception (implicit oracle reinforced with the bool postcondition)."""
    result = target_matches(
        target=target, family="gpt2", rank=0, layer=9, component="o_proj", event="output"
    )
    assert isinstance(result, bool)


# --------------------------------------------------------------------------- #
# Documented divergence pinned as a regression oracle
# --------------------------------------------------------------------------- #
def test_bracket_is_stripped_on_every_segment_not_just_component_known() -> None:
    """KNOWN doc/impl divergence (Finding M1): the docstring says bracket notation
    is stripped "on component", but the implementation strips a trailing
    [<digits>] on EVERY pattern segment. A bracket on the family segment is
    therefore silently accepted. Pinned so a future fix flips it deliberately."""
    assert target_matches(
        target="gpt2[1]:0:9:o_proj:output",
        family="gpt2",
        rank=0,
        layer=9,
        component="o_proj",
        event="output",
    )
