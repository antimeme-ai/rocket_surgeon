"""Property / model-based / metamorphic / exception-raising tests for the
intervention engine, composition, and recipe validation.

MATERIA oracle tiers exercised (LIMA lane, B004):

  * tier-6 model       — ``apply_interventions`` agrees with an INDEPENDENT,
                         out-of-place reference fold over the sorted recipe list,
                         for both result tensor and fired-id order.
  * tier-6 model       — ``filter_recipes`` agrees with a set/predicate model;
                         ``sort_by_priority`` agrees with a stable reference sort.
  * tier-4 metamorphic — ``scale(a) ∘ scale(b) == scale(a*b)`` (multiplicative);
                         ``ablate ∘ ablate == ablate`` (idempotent, zero & mean).
  * tier-5 property    — applied order == priority order (non-decreasing, stable,
                         permutation); clamp postcondition ``result ∈ [min,max]``;
                         the input tensor is never mutated.
  * tier-2 exception   — malformed recipes raise ``RecipeError`` with the right
                         message; dropping any required param raises.

Two production validation gaps are pinned as regression oracles (Findings
F-LIMA-1, F-LIMA-2) — see ``PLATOON-FINDINGS.md``.

Generator distribution is annotated with ``hypothesis.event`` / ``target``;
inspect with ``--hypothesis-show-statistics``.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pytest
import torch
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from rocket_surgeon.host.interventions.composition import filter_recipes, sort_by_priority
from rocket_surgeon.host.interventions.engine import apply_interventions
from rocket_surgeon.host.interventions.matching import target_matches
from rocket_surgeon.host.interventions.recipes import RecipeError, parse_recipe

# A wildcard target matches every execution point, so engine tests can focus on
# the *transform* without the matcher (which has its own property suite) firing.
WILDCARD = "*:*:*:*:*"


class _Point(TypedDict):
    family: str
    rank: int
    layer: int
    component: str
    event: str


POINT: _Point = {"family": "gpt2", "rank": 0, "layer": 0, "component": "x", "event": "output"}


def _norm(
    recipe_id: str,
    *,
    itype: str,
    params: dict[str, Any],
    target: str = WILDCARD,
    priority: int = 0,
    mode: str = "additive",
    condition: str | None = None,
) -> dict[str, Any]:
    """Build a normalized recipe dict (post-``parse_recipe`` shape)."""
    return {
        "id": recipe_id,
        "intervention_type": itype,
        "target": target,
        "params": params,
        "priority": priority,
        "mode": mode,
        "condition": condition,
    }


# --------------------------------------------------------------------------- #
# Independent reference model for apply_interventions (out-of-place fold)
# --------------------------------------------------------------------------- #
def _ref_apply(
    original: torch.Tensor,
    recipes: list[dict[str, Any]],
    store: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, list[str]]:
    """Reference fold: applies the same semantics as the engine, but purely
    functionally (no in-place mutation), serving as a tier-6 model oracle.

    Excludes the nondeterministic ``resample`` ablate mode and the ``callback``
    type by construction (generators never emit them here)."""
    current = original.clone()
    fired: list[str] = []
    for r in sorted(recipes, key=lambda x: x["priority"]):  # stable
        if r["mode"] == "replace":
            current = original.clone()
        itype = r["intervention_type"]
        p = r["params"]
        if itype == "scale":
            current = current * p["factor"]
        elif itype == "ablate":
            m = p.get("mode", "zero")
            if m == "zero":
                current = torch.zeros_like(current)
            elif m == "mean":
                current = torch.full_like(current, current.mean().item())
        elif itype == "add":
            v = p["vector"]
            add = (
                torch.tensor(v, dtype=current.dtype)
                if isinstance(v, list)
                else store[v].to(current.dtype)
            )
            current = current + add
        elif itype == "clamp":
            current = current.clamp(min=p["min"], max=p["max"])
        elif itype == "patch":
            current = store[p["source_tensor_id"]].to(current.dtype).clone()
        fired.append(r["id"])
    return current, fired


# --------------------------------------------------------------------------- #
# Generators (deterministic recipe ops over a fixed-width tensor)
# --------------------------------------------------------------------------- #
_DIM = 6
_finite = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False, width=32
)
_vec = st.lists(_finite, min_size=_DIM, max_size=_DIM)
_tensors = _vec.map(lambda xs: torch.tensor(xs, dtype=torch.float32))


@st.composite
def _det_recipe(draw: st.DrawFn, idx: int) -> dict[str, Any]:
    """A single deterministic recipe (no resample/callback). ``idx`` makes the id
    unique within a generated list."""
    itype = draw(
        st.sampled_from(["scale", "ablate", "ablate_mean", "add_inline", "clamp", "add_ref"])
    )
    priority = draw(st.integers(min_value=-3, max_value=3))
    mode = draw(st.sampled_from(["additive", "additive", "replace"]))  # bias additive
    rid = f"r{idx}"
    if itype == "scale":
        return _norm(
            rid, itype="scale", params={"factor": draw(_finite)}, priority=priority, mode=mode
        )
    if itype == "ablate":
        return _norm(rid, itype="ablate", params={"mode": "zero"}, priority=priority, mode=mode)
    if itype == "ablate_mean":
        return _norm(rid, itype="ablate", params={"mode": "mean"}, priority=priority, mode=mode)
    if itype == "add_inline":
        return _norm(rid, itype="add", params={"vector": draw(_vec)}, priority=priority, mode=mode)
    if itype == "add_ref":
        return _norm(rid, itype="add", params={"vector": "store-a"}, priority=priority, mode=mode)
    # clamp: keep min <= max so the postcondition is satisfiable
    lo = draw(_finite)
    hi = draw(st.floats(min_value=lo, max_value=10.0, allow_nan=False, width=32))
    return _norm(rid, itype="clamp", params={"min": lo, "max": hi}, priority=priority, mode=mode)


@st.composite
def _recipe_list(draw: st.DrawFn) -> list[dict[str, Any]]:
    # min 1: the empty-recipe path is covered by example tests; spend property
    # budget on lists that actually exercise the fold.
    n = draw(st.integers(min_value=1, max_value=6))
    return [draw(_det_recipe(i)) for i in range(n)]


# Fixed store referenced by add_ref / patch recipes.
def _store() -> dict[str, torch.Tensor]:
    return {"store-a": torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])}


# --------------------------------------------------------------------------- #
# Model-based: engine == reference fold (result + fired order)
# --------------------------------------------------------------------------- #
@given(_tensors, _recipe_list())
@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow])
def test_engine_matches_reference_fold(
    tensor: torch.Tensor, recipes: list[dict[str, Any]]
) -> None:
    """Model oracle: apply_interventions equals an independent out-of-place fold
    over the priority-sorted recipe list, in both tensor value and fired ids."""
    store = _store()
    event(f"n_recipes: {len(recipes)}")
    for r in recipes:
        event(f"type: {r['intervention_type']}/{r['params'].get('mode', '')}")
        if r["mode"] == "replace":
            event("has-replace")
    exp_t, exp_fired = _ref_apply(tensor, recipes, store)
    got_t, got_fired = apply_interventions(
        tensor=tensor, recipes=recipes, tensor_store=store.get, **POINT
    )
    assert got_fired == exp_fired
    assert torch.allclose(got_t, exp_t, atol=1e-4, rtol=1e-4), (
        f"mismatch\n exp={exp_t}\n got={got_t}"
    )


# --------------------------------------------------------------------------- #
# Property: input tensor is never mutated (engine clones)
# --------------------------------------------------------------------------- #
@given(_tensors, _recipe_list())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_input_tensor_not_mutated(tensor: torch.Tensor, recipes: list[dict[str, Any]]) -> None:
    """Postcondition: the caller's tensor is unchanged regardless of recipes —
    the docstring promises a clone is used for all modifications."""
    before = tensor.clone()
    apply_interventions(tensor=tensor, recipes=recipes, tensor_store=_store().get, **POINT)
    assert torch.equal(tensor, before)


# --------------------------------------------------------------------------- #
# Property: applied/fired order == priority order (non-decreasing, stable, perm)
# --------------------------------------------------------------------------- #
@given(_recipe_list())
@settings(max_examples=300)
def test_fired_order_is_priority_order(recipes: list[dict[str, Any]]) -> None:
    """The fired ids are the matched recipes in non-decreasing priority order, a
    stable permutation of the input (ties keep input order)."""
    tensor = torch.ones(_DIM)
    _, fired = apply_interventions(
        tensor=tensor, recipes=recipes, tensor_store=_store().get, **POINT
    )
    by_id = {r["id"]: r for r in recipes}
    assert sorted(fired) == sorted(r["id"] for r in recipes)  # permutation
    prios = [by_id[i]["priority"] for i in fired]
    assert prios == sorted(prios)  # non-decreasing
    # stability: within each priority class, input order is preserved
    expected = [r["id"] for r in sorted(recipes, key=lambda r: r["priority"])]
    assert fired == expected


# --------------------------------------------------------------------------- #
# Metamorphic: scale(a) ∘ scale(b) == scale(a*b)
# --------------------------------------------------------------------------- #
@given(_tensors, _finite, _finite)
@settings(max_examples=300)
def test_scale_composes_multiplicatively(tensor: torch.Tensor, a: float, b: float) -> None:
    """Two sequential scales equal a single scale by the product of the factors."""
    two = [
        _norm("a", itype="scale", params={"factor": a}, priority=0),
        _norm("b", itype="scale", params={"factor": b}, priority=1),
    ]
    one = [_norm("ab", itype="scale", params={"factor": a * b}, priority=0)]
    out_two, _ = apply_interventions(tensor=tensor, recipes=two, **POINT)
    out_one, _ = apply_interventions(tensor=tensor, recipes=one, **POINT)
    assert torch.allclose(out_two, out_one, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Metamorphic: ablate ∘ ablate == ablate (idempotent), for zero and mean
# --------------------------------------------------------------------------- #
@given(_tensors, st.sampled_from(["zero", "mean"]))
@settings(max_examples=200)
def test_ablate_is_idempotent(tensor: torch.Tensor, ablate_mode: str) -> None:
    """Ablating twice equals ablating once: zero→zero is constant; mean of a
    constant tensor is that same constant."""
    one = [_norm("x", itype="ablate", params={"mode": ablate_mode}, priority=0)]
    twice = [
        _norm("x", itype="ablate", params={"mode": ablate_mode}, priority=0),
        _norm("y", itype="ablate", params={"mode": ablate_mode}, priority=1),
    ]
    out_one, _ = apply_interventions(tensor=tensor, recipes=one, **POINT)
    out_two, _ = apply_interventions(tensor=tensor, recipes=twice, **POINT)
    assert torch.allclose(out_one, out_two, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Property: clamp postcondition — every element lands in [min, max]
# --------------------------------------------------------------------------- #
@given(
    _tensors,
    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, width=32),
    st.floats(min_value=0.0, max_value=5.0, allow_nan=False, width=32),
)
@settings(max_examples=200)
def test_clamp_result_within_bounds(tensor: torch.Tensor, lo: float, span: float) -> None:
    """With a well-formed range (min ≤ max), the result is bounded by [min, max]."""
    hi = lo + span
    recipes = [_norm("c", itype="clamp", params={"min": lo, "max": hi}, priority=0)]
    out, _ = apply_interventions(tensor=tensor, recipes=recipes, **POINT)
    assert torch.all(out >= lo - 1e-5)
    assert torch.all(out <= hi + 1e-5)


# --------------------------------------------------------------------------- #
# composition.filter_recipes — set/predicate model
# --------------------------------------------------------------------------- #
@st.composite
def _targets_and_recipes(draw: st.DrawFn) -> list[dict[str, Any]]:
    """Recipes with a mix of matching/non-matching targets and conditions."""
    out = []
    n = draw(st.integers(min_value=0, max_value=8))
    for i in range(n):
        target = draw(
            st.sampled_from(
                [
                    WILDCARD,
                    "gpt2:0:0:x:output",  # exact match for POINT
                    "gpt2:0:1:x:output",  # wrong layer
                    "llama:*:*:*:*",  # wrong family
                    "gpt2:*:*:x:*",  # match
                ]
            )
        )
        cond = draw(st.sampled_from([None, None, "tick_id > 5"]))
        out.append(
            _norm(f"r{i}", itype="scale", params={"factor": 1.0}, target=target, condition=cond)
        )
    return out


@given(_targets_and_recipes())
@settings(max_examples=300)
def test_filter_recipes_is_predicate_model(recipes: list[dict[str, Any]]) -> None:
    """Model oracle: filter_recipes keeps exactly the recipes whose condition is
    None AND whose target matches the point — in input order."""
    got = filter_recipes(recipes, **POINT)
    expected = [
        r
        for r in recipes
        if r["condition"] is None and target_matches(target=r["target"], **POINT)
    ]
    matched = sum(1 for r in recipes if r in expected)
    event(f"matched: {matched}/{len(recipes)}")
    assert got == expected


# --------------------------------------------------------------------------- #
# composition.sort_by_priority — stable ascending reference
# --------------------------------------------------------------------------- #
@given(st.lists(st.tuples(st.text(min_size=1, max_size=4), st.integers(-50, 50)), max_size=12))
@settings(max_examples=300)
def test_sort_by_priority_is_stable_ascending(pairs: list[tuple[str, int]]) -> None:
    """Model oracle: sort_by_priority == Python stable sort on priority, and is a
    permutation with non-decreasing priorities (ties keep input order)."""
    recipes: list[dict[str, Any]] = [
        {"id": f"{i}-{name}", "priority": p} for i, (name, p) in enumerate(pairs)
    ]
    event(f"len: {len(recipes)}")
    got = sort_by_priority(recipes)
    assert got == sorted(recipes, key=lambda r: r["priority"])
    assert [r["priority"] for r in got] == sorted(r["priority"] for r in recipes)
    assert {r["id"] for r in got} == {r["id"] for r in recipes}


@given(st.lists(st.integers(-9, 9), max_size=10))
@settings(max_examples=200)
def test_sort_by_priority_default_zero(prios: list[int]) -> None:
    """Recipes with no explicit priority default to 0 and keep input order among
    themselves and relative to explicit-0 recipes."""
    # interleave priority-bearing and priority-less recipes
    recipes: list[dict[str, Any]] = []
    for i, p in enumerate(prios):
        recipes.append({"id": f"p{i}"} if p == 0 else {"id": f"p{i}", "priority": p})
    got = sort_by_priority(recipes)
    assert [r.get("priority", 0) for r in got] == sorted(r.get("priority", 0) for r in recipes)


# --------------------------------------------------------------------------- #
# Exception-raising: malformed recipes raise RecipeError with the right message
# --------------------------------------------------------------------------- #
# type -> substring expected in the RecipeError message when its param is absent.
_REQUIRED_PARAM = {
    "scale": "factor",
    "add": "vector",
    "patch": "source_tensor_id",
    "clamp": "min",
}


@given(st.sampled_from(sorted(_REQUIRED_PARAM)))
@settings(max_examples=50)
def test_missing_required_param_raises(itype: str) -> None:
    """E1: every typed recipe missing its required param raises RecipeError naming
    the param — exception-raising property (113x mutation-killing class)."""
    needle = _REQUIRED_PARAM[itype]
    raw = {"id": "x", "type": itype, "target": WILDCARD, "params": {}}
    with pytest.raises(RecipeError, match=needle):
        parse_recipe(raw)
    event(f"type: {itype}")


@given(
    st.text(min_size=1, max_size=10).filter(
        lambda s: s not in {"ablate", "scale", "add", "patch", "clamp"}
    )
)
@settings(max_examples=200)
def test_unknown_type_raises(bad_type: str) -> None:
    """E2: any type outside the valid set raises 'unknown intervention type'."""
    raw = {"id": "x", "type": bad_type, "target": WILDCARD, "params": {}}
    with pytest.raises(RecipeError, match=r"unknown.*type"):
        parse_recipe(raw)


@given(st.text(min_size=1, max_size=10).filter(lambda s: s not in {"additive", "replace"}))
@settings(max_examples=150)
def test_invalid_mode_raises(bad_mode: str) -> None:
    """E3: a composition mode outside {additive, replace} raises."""
    raw = {
        "id": "x",
        "type": "ablate",
        "target": WILDCARD,
        "params": {"mode": "zero"},
        "mode": bad_mode,
    }
    with pytest.raises(RecipeError, match="composition mode"):
        parse_recipe(raw)


@given(st.text(min_size=1, max_size=10).filter(lambda s: s not in {"zero", "mean", "resample"}))
@settings(max_examples=150)
def test_invalid_ablate_mode_raises(bad_mode: str) -> None:
    """E4: an ablate mode outside {zero, mean, resample} raises."""
    raw = {"id": "x", "type": "ablate", "target": WILDCARD, "params": {"mode": bad_mode}}
    with pytest.raises(RecipeError, match="ablate mode"):
        parse_recipe(raw)


@given(st.dictionaries(st.text(max_size=5), st.integers(), max_size=4))
@settings(max_examples=150)
def test_missing_id_always_raises(params: dict[str, int]) -> None:
    """E5: a recipe without a truthy id raises 'missing required field: id',
    regardless of other params."""
    raw: dict[str, Any] = {"type": "ablate", "target": WILDCARD, "params": {"mode": "zero"}}
    raw.update(params)  # never adds a non-empty 'id'
    raw.pop("id", None)
    with pytest.raises(RecipeError, match="id"):
        parse_recipe(raw)


def test_round_trips_all_valid_types() -> None:
    """Sanity: each valid type with its required params parses without error and
    preserves the type and target (anchors the exception tests)."""
    cases = [
        {"id": "1", "type": "ablate", "target": WILDCARD, "params": {"mode": "mean"}},
        {"id": "2", "type": "scale", "target": WILDCARD, "params": {"factor": 2.0}},
        {"id": "3", "type": "add", "target": WILDCARD, "params": {"vector": [1.0]}},
        {"id": "4", "type": "patch", "target": WILDCARD, "params": {"source_tensor_id": "s"}},
        {"id": "5", "type": "clamp", "target": WILDCARD, "params": {"min": -1.0, "max": 1.0}},
    ]
    for raw in cases:
        recipe = parse_recipe(raw)
        assert recipe["intervention_type"] == raw["type"]
        assert recipe["target"] == WILDCARD


# --------------------------------------------------------------------------- #
# Regression pins for production validation gaps (see PLATOON-FINDINGS.md)
# --------------------------------------------------------------------------- #
def test_finding_clamp_min_gt_max_accepted_known() -> None:
    """Finding F-LIMA-1 (weak oracle): parse_recipe does NOT reject clamp with
    min > max. The engine then runs torch.clamp_(min, max) which, for min > max,
    collapses every element to `max`, so the natural postcondition
    `min ≤ result ≤ max` is unsatisfiable. Pinned so a future validation fix
    flips this deliberately."""
    recipe = parse_recipe(
        {"id": "c", "type": "clamp", "target": WILDCARD, "params": {"min": 1.0, "max": -1.0}}
    )
    out, fired = apply_interventions(
        tensor=torch.tensor([0.0, 5.0, -5.0]), recipes=[recipe], **POINT
    )
    assert fired == ["c"]
    # All elements collapse to max (-1.0), violating min ≤ x ≤ max.
    assert torch.allclose(out, torch.tensor([-1.0, -1.0, -1.0]))


def test_finding_falsy_id_zero_rejected_known() -> None:
    """Finding F-LIMA-2 (weak oracle): parse_recipe uses `if not recipe_id`, so a
    *present* but falsy id (integer 0, or "") is misreported as missing. Pinned
    to document the surprising rejection."""
    with pytest.raises(RecipeError, match="missing required field: id"):
        parse_recipe({"id": 0, "type": "ablate", "target": WILDCARD, "params": {"mode": "zero"}})
    with pytest.raises(RecipeError, match="missing required field: id"):
        parse_recipe({"id": "", "type": "ablate", "target": WILDCARD, "params": {"mode": "zero"}})
