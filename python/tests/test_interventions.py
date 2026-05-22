"""Tests for the intervention engine."""

from __future__ import annotations

from rocket_surgeon.host.interventions.matching import target_matches


class TestTargetMatching:
    def test_exact_match(self) -> None:
        assert target_matches(
            target="gpt2:0:11:attn.o_proj:output",
            family="gpt2",
            rank=0,
            layer=11,
            component="attn.o_proj",
            event="output",
        )

    def test_no_match_wrong_layer(self) -> None:
        assert not target_matches(
            target="gpt2:0:11:attn.o_proj:output",
            family="gpt2",
            rank=0,
            layer=5,
            component="attn.o_proj",
            event="output",
        )

    def test_wildcard_rank(self) -> None:
        assert target_matches(
            target="gpt2:*:11:attn.o_proj:output",
            family="gpt2",
            rank=7,
            layer=11,
            component="attn.o_proj",
            event="output",
        )

    def test_wildcard_all_layers(self) -> None:
        assert target_matches(
            target="gpt2:*:*:attn.o_proj:output",
            family="gpt2",
            rank=0,
            layer=99,
            component="attn.o_proj",
            event="output",
        )

    def test_wildcard_family(self) -> None:
        assert target_matches(
            target="*:*:11:*:output",
            family="llama",
            rank=0,
            layer=11,
            component="mlp.gate_proj",
            event="output",
        )

    def test_wrong_component_no_match(self) -> None:
        assert not target_matches(
            target="gpt2:0:11:attn.o_proj:output",
            family="gpt2",
            rank=0,
            layer=11,
            component="mlp.c_fc",
            event="output",
        )

    def test_malformed_target_too_few_segments(self) -> None:
        assert not target_matches(
            target="gpt2:0:11",
            family="gpt2",
            rank=0,
            layer=11,
            component="attn.o_proj",
            event="output",
        )
