"""Tests for the intervention engine."""

from __future__ import annotations

import pytest
import torch

from rocket_surgeon.host.interventions.composition import filter_recipes, sort_by_priority
from rocket_surgeon.host.interventions.engine import apply_interventions
from rocket_surgeon.host.interventions.matching import target_matches
from rocket_surgeon.host.interventions.recipes import RecipeError, parse_recipe


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


class TestRecipeParsing:
    def test_parse_ablate_zero(self) -> None:
        raw = {
            "id": "iv-1",
            "type": "ablate",
            "target": "gpt2:0:11:attn.o_proj:output",
            "params": {"mode": "zero"},
        }
        recipe = parse_recipe(raw)
        assert recipe["id"] == "iv-1"
        assert recipe["intervention_type"] == "ablate"
        assert recipe["params"]["mode"] == "zero"
        assert recipe["priority"] == 0
        assert recipe["mode"] == "additive"

    def test_parse_scale(self) -> None:
        raw = {
            "id": "iv-2",
            "type": "scale",
            "target": "gpt2:0:11:attn.o_proj:output",
            "params": {"factor": 0.5},
        }
        recipe = parse_recipe(raw)
        assert recipe["params"]["factor"] == 0.5

    def test_parse_add_inline(self) -> None:
        raw = {
            "id": "iv-3",
            "type": "add",
            "target": "gpt2:0:11:attn.o_proj:output",
            "params": {"vector": [1.0, 2.0, 3.0]},
        }
        recipe = parse_recipe(raw)
        assert recipe["params"]["vector"] == [1.0, 2.0, 3.0]

    def test_parse_add_tensor_ref(self) -> None:
        raw = {
            "id": "iv-4",
            "type": "add",
            "target": "gpt2:0:11:attn.o_proj:output",
            "params": {"vector": "abc123def456"},
        }
        recipe = parse_recipe(raw)
        assert recipe["params"]["vector"] == "abc123def456"

    def test_parse_patch(self) -> None:
        raw = {
            "id": "iv-5",
            "type": "patch",
            "target": "gpt2:0:11:attn.o_proj:output",
            "params": {"source_tensor_id": "abc123"},
        }
        recipe = parse_recipe(raw)
        assert recipe["params"]["source_tensor_id"] == "abc123"

    def test_parse_clamp(self) -> None:
        raw = {
            "id": "iv-6",
            "type": "clamp",
            "target": "gpt2:0:11:attn.o_proj:output",
            "params": {"min": -1.0, "max": 1.0},
        }
        recipe = parse_recipe(raw)
        assert recipe["params"]["min"] == -1.0
        assert recipe["params"]["max"] == 1.0

    def test_parse_with_priority_and_mode(self) -> None:
        raw = {
            "id": "iv-7",
            "type": "scale",
            "target": "gpt2:0:11:attn.o_proj:output",
            "params": {"factor": 2.0},
            "priority": 10,
            "mode": "replace",
        }
        recipe = parse_recipe(raw)
        assert recipe["priority"] == 10
        assert recipe["mode"] == "replace"

    def test_parse_missing_id_raises(self) -> None:
        raw = {
            "type": "ablate",
            "target": "gpt2:0:11:attn.o_proj:output",
            "params": {},
        }
        with pytest.raises(RecipeError, match=r"missing.*id"):
            parse_recipe(raw)

    def test_parse_unknown_type_raises(self) -> None:
        raw = {
            "id": "iv-bad",
            "type": "explode",
            "target": "gpt2:0:11:attn.o_proj:output",
            "params": {},
        }
        with pytest.raises(RecipeError, match=r"unknown.*type"):
            parse_recipe(raw)

    def test_parse_scale_missing_factor_raises(self) -> None:
        raw = {
            "id": "iv-bad",
            "type": "scale",
            "target": "gpt2:0:11:attn.o_proj:output",
            "params": {},
        }
        with pytest.raises(RecipeError, match="factor"):
            parse_recipe(raw)

    def test_condition_field_preserved(self) -> None:
        raw = {
            "id": "iv-cond",
            "type": "ablate",
            "target": "gpt2:0:11:attn.o_proj:output",
            "params": {"mode": "zero"},
            "condition": "tick_id > 10",
        }
        recipe = parse_recipe(raw)
        assert recipe["condition"] == "tick_id > 10"


def _recipe(
    recipe_id: str,
    target: str,
    itype: str = "scale",
    params: dict | None = None,
    priority: int = 0,
    mode: str = "additive",
    condition: str | None = None,
) -> dict:
    """Helper to build a normalized recipe dict for tests."""
    return {
        "id": recipe_id,
        "intervention_type": itype,
        "target": target,
        "params": params or {},
        "priority": priority,
        "mode": mode,
        "condition": condition,
    }


class TestComposition:
    def test_filter_by_target(self) -> None:
        recipes = [
            _recipe("a", "gpt2:0:11:attn.o_proj:output", params={"factor": 2.0}),
            _recipe("b", "gpt2:0:5:mlp.c_fc:output", params={"factor": 3.0}),
            _recipe("c", "gpt2:*:*:attn.o_proj:output", itype="ablate", params={"mode": "zero"}),
        ]
        matched = filter_recipes(
            recipes, family="gpt2", rank=0, layer=11, component="attn.o_proj", event="output"
        )
        assert len(matched) == 2
        assert matched[0]["id"] == "a"
        assert matched[1]["id"] == "c"

    def test_sort_by_priority_ascending(self) -> None:
        recipes = [
            {"id": "hi", "priority": 10},
            {"id": "lo", "priority": -5},
            {"id": "mid", "priority": 0},
        ]
        sorted_r = sort_by_priority(recipes)
        assert [r["id"] for r in sorted_r] == ["lo", "mid", "hi"]

    def test_stable_sort_same_priority(self) -> None:
        recipes = [
            {"id": "first", "priority": 0},
            {"id": "second", "priority": 0},
            {"id": "third", "priority": 0},
        ]
        sorted_r = sort_by_priority(recipes)
        assert [r["id"] for r in sorted_r] == ["first", "second", "third"]

    def test_filter_skips_conditional_recipes(self) -> None:
        recipes = [
            _recipe(
                "a",
                "gpt2:0:11:attn.o_proj:output",
                params={"factor": 2.0},
                condition="tick_id > 10",
            ),
        ]
        matched = filter_recipes(
            recipes, family="gpt2", rank=0, layer=11, component="attn.o_proj", event="output"
        )
        assert len(matched) == 0


class TestApplyInterventions:
    def test_no_matching_recipes_returns_unchanged(self) -> None:
        tensor = torch.ones(4)
        recipes = [
            _recipe("a", "gpt2:0:5:mlp:output", params={"factor": 0.0}),
        ]
        result, fired = apply_interventions(
            tensor=tensor,
            recipes=recipes,
            family="gpt2",
            rank=0,
            layer=11,
            component="attn.o_proj",
            event="output",
        )
        assert fired == []
        assert torch.equal(result, torch.ones(4))

    def test_ablate_zero(self) -> None:
        tensor = torch.randn(8)
        recipes = [
            _recipe(
                "z",
                "gpt2:0:11:attn.o_proj:output",
                itype="ablate",
                params={"mode": "zero"},
            ),
        ]
        result, fired = apply_interventions(
            tensor=tensor,
            recipes=recipes,
            family="gpt2",
            rank=0,
            layer=11,
            component="attn.o_proj",
            event="output",
        )
        assert fired == ["z"]
        assert torch.all(result == 0.0)

    def test_ablate_mean(self) -> None:
        tensor = torch.tensor([2.0, 4.0, 6.0])
        recipes = [
            _recipe("m", "*:*:*:*:*", itype="ablate", params={"mode": "mean"}),
        ]
        result, fired = apply_interventions(
            tensor=tensor,
            recipes=recipes,
            family="gpt2",
            rank=0,
            layer=0,
            component="x",
            event="output",
        )
        assert fired == ["m"]
        assert torch.allclose(result, torch.tensor([4.0, 4.0, 4.0]))

    def test_scale(self) -> None:
        tensor = torch.tensor([1.0, 2.0, 3.0])
        recipes = [
            _recipe("s", "*:*:*:*:*", params={"factor": 0.5}),
        ]
        result, fired = apply_interventions(
            tensor=tensor,
            recipes=recipes,
            family="gpt2",
            rank=0,
            layer=0,
            component="x",
            event="output",
        )
        assert fired == ["s"]
        assert torch.allclose(result, torch.tensor([0.5, 1.0, 1.5]))

    def test_add_inline(self) -> None:
        tensor = torch.tensor([1.0, 2.0, 3.0])
        recipes = [
            _recipe("a", "*:*:*:*:*", itype="add", params={"vector": [10.0, 20.0, 30.0]}),
        ]
        result, fired = apply_interventions(
            tensor=tensor,
            recipes=recipes,
            family="gpt2",
            rank=0,
            layer=0,
            component="x",
            event="output",
        )
        assert fired == ["a"]
        assert torch.allclose(result, torch.tensor([11.0, 22.0, 33.0]))

    def test_add_tensor_ref(self) -> None:
        tensor = torch.tensor([1.0, 2.0])
        store_tensor = torch.tensor([100.0, 200.0])
        recipes = [
            _recipe("a", "*:*:*:*:*", itype="add", params={"vector": "ref-id-1"}),
        ]
        result, fired = apply_interventions(
            tensor=tensor,
            recipes=recipes,
            family="gpt2",
            rank=0,
            layer=0,
            component="x",
            event="output",
            tensor_store=lambda tid: store_tensor if tid == "ref-id-1" else None,
        )
        assert fired == ["a"]
        assert torch.allclose(result, torch.tensor([101.0, 202.0]))

    def test_patch(self) -> None:
        tensor = torch.zeros(3)
        patch_tensor = torch.tensor([7.0, 8.0, 9.0])
        recipes = [
            _recipe(
                "p",
                "*:*:*:*:*",
                itype="patch",
                params={"source_tensor_id": "src-1"},
            ),
        ]
        result, fired = apply_interventions(
            tensor=tensor,
            recipes=recipes,
            family="gpt2",
            rank=0,
            layer=0,
            component="x",
            event="output",
            tensor_store=lambda tid: patch_tensor if tid == "src-1" else None,
        )
        assert fired == ["p"]
        assert torch.allclose(result, torch.tensor([7.0, 8.0, 9.0]))

    def test_clamp(self) -> None:
        tensor = torch.tensor([-5.0, 0.0, 5.0, 10.0])
        recipes = [
            _recipe(
                "c",
                "*:*:*:*:*",
                itype="clamp",
                params={"min": -1.0, "max": 1.0},
            ),
        ]
        result, fired = apply_interventions(
            tensor=tensor,
            recipes=recipes,
            family="gpt2",
            rank=0,
            layer=0,
            component="x",
            event="output",
        )
        assert fired == ["c"]
        assert torch.allclose(result, torch.tensor([-1.0, 0.0, 1.0, 1.0]))

    def test_priority_ordering(self) -> None:
        tensor = torch.tensor([10.0])
        recipes = [
            _recipe(
                "clamp-hi",
                "*:*:*:*:*",
                itype="clamp",
                params={"min": -1.0, "max": 1.0},
                priority=10,
            ),
            _recipe("scale-lo", "*:*:*:*:*", params={"factor": 0.05}, priority=0),
        ]
        # scale first (priority 0): 10 * 0.05 = 0.5
        # clamp second (priority 10): clamp(0.5, -1, 1) = 0.5
        result, fired = apply_interventions(
            tensor=tensor,
            recipes=recipes,
            family="gpt2",
            rank=0,
            layer=0,
            component="x",
            event="output",
        )
        assert fired == ["scale-lo", "clamp-hi"]
        assert torch.allclose(result, torch.tensor([0.5]))

    def test_replace_mode_discards_prior(self) -> None:
        tensor = torch.tensor([10.0])
        recipes = [
            _recipe(
                "add-first",
                "*:*:*:*:*",
                itype="add",
                params={"vector": [100.0]},
                priority=0,
            ),
            _recipe(
                "scale-replace",
                "*:*:*:*:*",
                params={"factor": 2.0},
                priority=5,
                mode="replace",
            ),
        ]
        # add first: 10 + 100 = 110, but then replace resets to original (10)
        # scale on original: 10 * 2 = 20
        result, fired = apply_interventions(
            tensor=tensor,
            recipes=recipes,
            family="gpt2",
            rank=0,
            layer=0,
            component="x",
            event="output",
        )
        assert fired == ["add-first", "scale-replace"]
        assert torch.allclose(result, torch.tensor([20.0]))

    def test_empty_recipe_list(self) -> None:
        tensor = torch.tensor([1.0, 2.0])
        result, fired = apply_interventions(
            tensor=tensor,
            recipes=[],
            family="gpt2",
            rank=0,
            layer=0,
            component="x",
            event="output",
        )
        assert fired == []
        assert torch.equal(result, torch.tensor([1.0, 2.0]))

    def test_ablate_resample_changes_values(self) -> None:
        torch.manual_seed(42)
        tensor = torch.tensor([1.0, 3.0, 5.0, 7.0])
        original = tensor.clone()
        recipes = [
            _recipe("r", "*:*:*:*:*", itype="ablate", params={"mode": "resample"}),
        ]
        result, fired = apply_interventions(
            tensor=tensor,
            recipes=recipes,
            family="gpt2",
            rank=0,
            layer=0,
            component="x",
            event="output",
        )
        assert fired == ["r"]
        assert not torch.equal(result, original)
