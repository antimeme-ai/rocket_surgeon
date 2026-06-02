"""Tests for bracket notation handling in intervention target matching."""

from __future__ import annotations

from rocket_surgeon.host.interventions.matching import (
    extract_head_index,
    strip_bracket,
    target_matches,
)


def test_strip_bracket_removes_index() -> None:
    assert strip_bracket("o_proj[7]") == "o_proj"


def test_strip_bracket_no_bracket() -> None:
    assert strip_bracket("o_proj") == "o_proj"


def test_extract_head_index_present() -> None:
    assert extract_head_index("o_proj[7]") == 7


def test_extract_head_index_absent() -> None:
    assert extract_head_index("o_proj") is None


def test_exact_component_matches() -> None:
    assert target_matches(
        target="gpt2:0:9:o_proj:output",
        family="gpt2",
        rank=0,
        layer=9,
        component="o_proj",
        event="output",
    )


def test_bracket_component_matches_base() -> None:
    assert target_matches(
        target="gpt2:0:9:o_proj[7]:output",
        family="gpt2",
        rank=0,
        layer=9,
        component="o_proj",
        event="output",
    )


def test_bracket_different_component_no_match() -> None:
    assert not target_matches(
        target="gpt2:0:9:o_proj[7]:output",
        family="gpt2",
        rank=0,
        layer=9,
        component="down_proj",
        event="output",
    )


def test_bracket_different_layer_no_match() -> None:
    assert not target_matches(
        target="gpt2:0:9:o_proj[7]:output",
        family="gpt2",
        rank=0,
        layer=8,
        component="o_proj",
        event="output",
    )


def test_wildcard_with_bracket() -> None:
    assert target_matches(
        target="*:*:*:o_proj[3]:output",
        family="gpt2",
        rank=0,
        layer=5,
        component="o_proj",
        event="output",
    )
