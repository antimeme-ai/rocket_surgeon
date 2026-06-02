# PLATOON-FINDINGS — LIMA (B004, intervention engine)

**Lane:** `python/rocket_surgeon/host/interventions` — engine, composition,
matching, recipe validation.
**Branch:** `platoon2/interventions`.
**Artifact:** `python/tests/test_intervention_engine_properties.py` (new, 17 property
tests). Matching already had a Wave-1 property suite (`test_matching_properties.py`);
this wave covers the previously example-only modules `engine.py`, `composition.py`,
and `recipes.py`.

## Techniques applied (MATERIA oracle tiers)

| Tier | Technique | Tests |
| --- | --- | --- |
| 6 model | `apply_interventions` == independent out-of-place reference fold (tensor value **and** fired-id order) over scale/ablate/add/clamp/patch with additive+replace modes | `test_engine_matches_reference_fold` |
| 6 model | `filter_recipes` == set/predicate model (`condition is None AND target_matches`), order-preserving | `test_filter_recipes_is_predicate_model` |
| 6 model | `sort_by_priority` == Python stable sort; permutation + non-decreasing; default-0 handling | `test_sort_by_priority_is_stable_ascending`, `test_sort_by_priority_default_zero` |
| 5 property | applied/fired order == priority order (stable permutation, non-decreasing) | `test_fired_order_is_priority_order` |
| 5 property | input tensor never mutated (engine clones) | `test_input_tensor_not_mutated` |
| 5 property | clamp postcondition `min ≤ result ≤ max` (well-formed range) | `test_clamp_result_within_bounds` |
| 4 metamorphic | `scale(a) ∘ scale(b) == scale(a*b)` | `test_scale_composes_multiplicatively` |
| 4 metamorphic | `ablate ∘ ablate == ablate` (idempotent, zero & mean) | `test_ablate_is_idempotent` |
| 2 exception | malformed recipes raise `RecipeError` with the right message; every typed recipe missing its required param raises; bad type / mode / ablate-mode / id all raise | `test_missing_required_param_raises`, `test_unknown_type_raises`, `test_invalid_mode_raises`, `test_invalid_ablate_mode_raises`, `test_missing_id_always_raises` |

The **model-based fold** is the centerpiece (Hughes 2020: model-based properties
find bugs in ~8 tests where postconditions take 50) — it checks the *entire*
output tensor and the *entire* fired sequence on every example, against a reference
written out-of-place so it is a genuinely independent oracle, not a transcription.

## Generator distribution (evidence, `--hypothesis-show-statistics`)

`test_engine_matches_reference_fold` over 400 examples (after biasing list size to
≥1 so every example exercises the fold):
- recipe count spread 1–6 (no single bucket dominates; was 40% empty before the bias fix).
- intervention types all well-represented: `add` ~34%, `scale` ~26%, `ablate/zero`
  ~25%, `clamp` ~17%, `ablate/mean` ~14%.
- `has-replace` mode present in ~29% of lists — the replace/reset branch is exercised.

`test_filter_recipes_is_predicate_model`: `matched: k/n` events confirm a mix of
matched and filtered recipes (not all-match or all-filter). Exception generators
report low filter-retry rates (<1%) — inputs are not being silently discarded.

## Bugs / weak oracles found

Both are **validation gaps in production** (`recipes.py`). Per brief I recorded
them and pinned current behavior as regression oracles rather than changing
production semantics (these are judgment calls the protocol/recipe owner should
make deliberately, not style fixes).

### F-LIMA-1 — `parse_recipe` accepts clamp with `min > max`; engine silently collapses
- **Minimal case:** `parse_recipe({"id":"c","type":"clamp","target":"*:*:*:*:*","params":{"min":1.0,"max":-1.0}})` succeeds.
- **Consequence:** `engine._apply_single` runs `tensor.clamp_(min=1.0, max=-1.0)`. For
  `min > max`, torch collapses **every** element to `max` (`-1.0`), so the natural
  postcondition `min ≤ x ≤ max` is unsatisfiable — the intervention does something
  surprising and unannounced instead of erroring at parse time.
- **Root cause:** `recipes._validate_params` for `clamp` only checks that `min` and
  `max` keys are *present* (`recipes.py:66-69`); it never checks `min ≤ max`.
- **Pinned by:** `test_finding_clamp_min_gt_max_accepted_known`.
- **Suggested fix (owner's call):** raise `RecipeError("clamp requires min <= max")`
  in `_validate_params`. Trivial; left to the owner because it changes accepted input.

### F-LIMA-2 — falsy-but-present `id` (integer `0`, empty string) misreported as missing
- **Minimal case:** `parse_recipe({"id":0,"type":"ablate",...})` raises
  `RecipeError("recipe missing required field: id")`, as does `id=""`.
- **Root cause:** `recipes.py:23` guards with `if not recipe_id` (falsy test) rather
  than `if recipe_id is None` / `"id" not in raw`. Any falsy id is rejected as absent.
- **Pinned by:** `test_finding_falsy_id_zero_rejected_known`.
- **Impact:** low in practice (ids are hashes/strings), but the error message is wrong
  for a present-but-falsy id, which would mislead an LLM client debugging a recipe.

### Adjacent observations (no test pinned; out of lane scope to fix)
- `scale`/`add`/`clamp` params are **not type-validated** at parse time. A non-numeric
  `factor` (`"big"`) or a wrong-length inline `add` vector parses cleanly, then the
  engine raises a raw `TypeError`/`RuntimeError` from torch instead of a `RecipeError`.
  This is consistent across all numeric params; a parse-time numeric/shape check would
  move these to the exception-raising tier. Noted for the recipe-schema owner.

## Gaps left
- `ablate` mode `resample` and the `callback` intervention type are **excluded from
  the model-fold oracle** because they are nondeterministic / side-effecting
  (`resample` draws from `normal_`; `callback` runs arbitrary user code in a thread
  with a timeout). `callback` already has a dedicated suite (`test_callback.py`);
  `resample`'s "changes values" behavior is covered by the existing example test.
  A metamorphic oracle for `resample` (e.g. mean/std of the output ≈ requested) is a
  reasonable future addition but was out of scope for this wave.
- No stateful-sequence model here: the engine is a pure per-tick fold (no state
  carried across calls), so stateful model-based testing doesn't apply — the
  per-call fold model is the right abstraction.

## Status
- 17 new property tests green; full intervention suite (engine props + interventions
  + matching props + callback) = 63 passed.
- `ruff check`, `ruff format --check`, `mypy` clean on the new file.
- Production code unchanged (findings pinned, not fixed — owner's call).
