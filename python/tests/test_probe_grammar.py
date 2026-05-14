"""Tests for probe-point grammar parser — mirrors Rust tests in grammar.rs."""

from __future__ import annotations

import json

import pytest

from rocket_surgeon.probes.grammar import (
    IndexedSeg,
    NamedSeg,
    ParseError,
    ProbePoint,
    Wildcard,
)

# ---------------------------------------------------------------------------
# Design doc section 8 examples
# ---------------------------------------------------------------------------


class TestDesignDocExamples:
    def test_attn_output_all_ranks(self) -> None:
        p = ProbePoint.parse("llama:*:12:attn.o_proj:output")
        assert p.model == "llama"
        assert isinstance(p.rank, Wildcard)
        assert p.layer == 12
        assert p.component == (NamedSeg("attn"), NamedSeg("o_proj"))
        assert p.event == "output"

    def test_mlp_input_rank0(self) -> None:
        p = ProbePoint.parse("llama:0:*:mlp:input")
        assert p.model == "llama"
        assert p.rank == 0
        assert isinstance(p.layer, Wildcard)
        assert p.component == (NamedSeg("mlp"),)
        assert p.event == "input"

    def test_moe_router_pre_topk(self) -> None:
        p = ProbePoint.parse("mixtral:*:8:router:pre_topk")
        assert p.model == "mixtral"
        assert isinstance(p.rank, Wildcard)
        assert p.layer == 8
        assert p.component == (NamedSeg("router"),)
        assert p.event == "pre_topk"

    def test_all_wildcards(self) -> None:
        p = ProbePoint.parse("llama:*:*:residual_post:*")
        assert p.model == "llama"
        assert isinstance(p.rank, Wildcard)
        assert isinstance(p.layer, Wildcard)
        assert p.component == (NamedSeg("residual_post"),)
        assert isinstance(p.event, Wildcard)

    def test_attn_scores_virtual(self) -> None:
        p = ProbePoint.parse("llama:0:12:attn.scores:*")
        assert p.rank == 0
        assert p.layer == 12
        assert p.component == (NamedSeg("attn"), NamedSeg("scores"))

    def test_indexed_expert(self) -> None:
        p = ProbePoint.parse("mixtral:*:8:experts[3]:output")
        assert p.component == (IndexedSeg("experts", 3),)

    def test_indexed_expert_with_subcomponent(self) -> None:
        p = ProbePoint.parse("mixtral:*:8:experts[3].gate_proj:output")
        assert p.component == (
            IndexedSeg("experts", 3),
            NamedSeg("gate_proj"),
        )

    def test_wildcard_component(self) -> None:
        p = ProbePoint.parse("llama:*:12:*:output")
        assert isinstance(p.component, Wildcard)

    def test_full_wildcard(self) -> None:
        p = ProbePoint.parse("*:*:*:*:*")
        assert isinstance(p.model, Wildcard)
        assert isinstance(p.rank, Wildcard)
        assert isinstance(p.layer, Wildcard)
        assert isinstance(p.component, Wildcard)
        assert isinstance(p.event, Wildcard)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_complex(self) -> None:
        text = "mixtral:*:8:experts[3].gate_proj:output"
        assert str(ProbePoint.parse(text)) == text

    def test_all_wildcards(self) -> None:
        text = "*:*:*:*:*"
        assert str(ProbePoint.parse(text)) == text

    def test_all_concrete(self) -> None:
        text = "llama:0:12:attn.o_proj:output"
        assert str(ProbePoint.parse(text)) == text

    def test_leading_zeros_normalize(self) -> None:
        assert str(ProbePoint.parse("llama:0:012:mlp:output")) == "llama:0:12:mlp:output"


# ---------------------------------------------------------------------------
# Wildcard matching
# ---------------------------------------------------------------------------


class TestMatching:
    def test_wildcard_matches_any_rank(self) -> None:
        pattern = ProbePoint.parse("llama:*:12:mlp:output")
        target = ProbePoint.parse("llama:3:12:mlp:output")
        assert pattern.matches(target)

    def test_wildcard_matches_any_layer(self) -> None:
        pattern = ProbePoint.parse("llama:0:*:mlp:output")
        target = ProbePoint.parse("llama:0:7:mlp:output")
        assert pattern.matches(target)

    def test_wildcard_matches_any_component(self) -> None:
        pattern = ProbePoint.parse("llama:0:12:*:output")
        target = ProbePoint.parse("llama:0:12:attn.o_proj:output")
        assert pattern.matches(target)

    def test_concrete_does_not_match_different(self) -> None:
        pattern = ProbePoint.parse("llama:0:12:mlp:output")
        target = ProbePoint.parse("llama:0:12:attn:output")
        assert not pattern.matches(target)

    def test_different_model_does_not_match(self) -> None:
        pattern = ProbePoint.parse("llama:0:12:mlp:output")
        target = ProbePoint.parse("mixtral:0:12:mlp:output")
        assert not pattern.matches(target)

    def test_different_layer_does_not_match(self) -> None:
        pattern = ProbePoint.parse("llama:0:12:mlp:output")
        target = ProbePoint.parse("llama:0:13:mlp:output")
        assert not pattern.matches(target)


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


class TestInvalidInput:
    def test_reject_missing_segments(self) -> None:
        with pytest.raises(ParseError):
            ProbePoint.parse("llama:0:12:mlp")

    def test_reject_empty_string(self) -> None:
        with pytest.raises(ParseError):
            ProbePoint.parse("")

    def test_reject_extra_segment(self) -> None:
        with pytest.raises(ParseError):
            ProbePoint.parse("llama:0:12:mlp:output:extra")

    def test_reject_trailing_colon(self) -> None:
        with pytest.raises(ParseError):
            ProbePoint.parse("llama:0:12:mlp:output:")

    def test_reject_leading_colon(self) -> None:
        with pytest.raises(ParseError):
            ProbePoint.parse(":0:12:mlp:output")

    def test_reject_negative_layer(self) -> None:
        with pytest.raises(ParseError):
            ProbePoint.parse("llama:0:-1:mlp:output")

    def test_reject_non_numeric_rank(self) -> None:
        with pytest.raises(ParseError):
            ProbePoint.parse("llama:abc:12:mlp:output")

    def test_reject_empty_component_segment(self) -> None:
        with pytest.raises(ParseError):
            ProbePoint.parse("llama:0:12:.mlp:output")

    def test_reject_unclosed_bracket(self) -> None:
        with pytest.raises(ParseError):
            ProbePoint.parse("llama:0:12:experts[3:output")

    def test_reject_u32_overflow(self) -> None:
        with pytest.raises(ParseError):
            ProbePoint.parse("llama:0:4294967296:mlp:output")


# ---------------------------------------------------------------------------
# Serde round-trip (dict ↔ ProbePoint, mirrors Rust JSON serde)
# ---------------------------------------------------------------------------


class TestSerde:
    def test_round_trip(self) -> None:
        p = ProbePoint.parse("llama:0:12:attn.o_proj:output")
        d = p.to_dict()
        j = json.dumps(d)
        p2 = ProbePoint.from_dict(json.loads(j))
        assert p == p2

    def test_dict_shape_matches_rust(self) -> None:
        p = ProbePoint.parse("llama:*:12:experts[3].gate_proj:output")
        d = p.to_dict()
        assert d["model"] == {"Name": "llama"}
        assert d["rank"] == "Wildcard"
        assert d["layer"] == {"Num": 12}
        assert d["component"] == {
            "Path": [
                {"Indexed": {"name": "experts", "index": 3}},
                {"Named": "gate_proj"},
            ]
        }
        assert d["event"] == {"Name": "output"}
