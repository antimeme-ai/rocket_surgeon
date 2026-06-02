"""Property / metamorphic / exception-raising tests for replay divergence math.

MATERIA oracle tiers exercised here:
  * tier-4 metamorphic  — cosine scale/sign invariance, append-monotonicity, the
                          from_ptr / direct equivalence relation.
  * tier-2 exception     — invalid dtype / shape / NaN behaviour.

`compare_activations` returns ``None`` to mean "within tolerance" (no divergence
reported) and a dict to mean "divergence reported". To *observe* the raw cosine
similarity regardless of tolerance we pass an impossible ``cosine_threshold`` (>1)
with ``mre_threshold = inf`` so the function always yields the dict — call this the
"probe" configuration.

Generator distribution is annotated with ``hypothesis.event``; run with
``--hypothesis-show-statistics`` to inspect it (evidence captured in
PLATOON-FINDINGS.md).
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest
import torch
from hypothesis import assume, event, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from rocket_surgeon.replay import compare_activations, compare_activations_from_ptr

# A probe configuration that forces the dict so we can read the raw metrics.
PROBE_COS = 1.1  # cosine_sim is always < 1.1, so the dict is always returned
PROBE_MRE = math.inf

_finite32 = st.floats(
    min_value=-1.0e4,
    max_value=1.0e4,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)


def _vectors(min_size: int = 1, max_size: int = 256) -> st.SearchStrategy[torch.Tensor]:
    """1-D float32 tensors with finite, bounded elements."""
    return hnp.arrays(
        np.float32,
        st.integers(min_value=min_size, max_value=max_size),
        elements=_finite32,
    ).map(lambda a: torch.from_numpy(a.copy()))


def _nonzero_vectors(size: int = 64) -> st.SearchStrategy[torch.Tensor]:
    """Vectors guaranteed to have non-negligible norm (cosine well-defined)."""
    return _vectors(min_size=size, max_size=size).filter(
        lambda t: torch.linalg.norm(t).item() > 1e-2
    )


def _classify(t: torch.Tensor) -> None:
    norm = torch.linalg.norm(t).item()
    if norm < 1.0:
        event("norm: small (<1)")
    elif norm < 100.0:
        event("norm: medium")
    else:
        event("norm: large (>=100)")
    event(f"len bucket: {len(t) // 32 * 32}")


# --------------------------------------------------------------------------- #
# Metamorphic relations
# --------------------------------------------------------------------------- #
@given(_nonzero_vectors())
@settings(max_examples=200)
def test_identical_tensors_report_no_divergence(a: torch.Tensor) -> None:
    """M1: divergence(x, x) is none. Abstraction: identical => zero divergence.

    Uses a strict-but-sub-unit cosine threshold so float rounding of
    dot/(norm*norm) around 1.0 cannot spuriously trip it.
    """
    _classify(a)
    assert compare_activations(a, a.clone(), cosine_threshold=0.999, mre_threshold=0.0) is None


@given(_nonzero_vectors(), _nonzero_vectors(), st.floats(0.01, 1e3))
@settings(max_examples=300)
def test_cosine_scale_invariance(a: torch.Tensor, b: torch.Tensor, c: float) -> None:
    """M2: cosine(a, c*b) == cosine(a, b) for c > 0."""
    _classify(b)
    base = compare_activations(a, b, PROBE_COS, PROBE_MRE)
    scaled = compare_activations(a, b * c, PROBE_COS, PROBE_MRE)
    assert base is not None
    assert scaled is not None
    assert math.isclose(
        base["cosine_similarity"], scaled["cosine_similarity"], rel_tol=1e-4, abs_tol=1e-4
    )


@given(_nonzero_vectors(), _nonzero_vectors())
@settings(max_examples=300)
def test_cosine_sign_flip_under_negation(a: torch.Tensor, b: torch.Tensor) -> None:
    """M3: cosine(a, -b) == -cosine(a, b)."""
    pos = compare_activations(a, b, PROBE_COS, PROBE_MRE)
    neg = compare_activations(a, -b, PROBE_COS, PROBE_MRE)
    assert pos is not None
    assert neg is not None
    assert math.isclose(
        neg["cosine_similarity"], -pos["cosine_similarity"], rel_tol=1e-4, abs_tol=1e-4
    )


@given(_nonzero_vectors(), _nonzero_vectors())
@settings(max_examples=300)
def test_cosine_is_symmetric(a: torch.Tensor, b: torch.Tensor) -> None:
    """M4: cosine(a, b) == cosine(b, a). (max_relative_error is NOT symmetric —
    its denominator uses the *original* tensor — so only cosine is asserted.)"""
    ab = compare_activations(a, b, PROBE_COS, PROBE_MRE)
    ba = compare_activations(b, a, PROBE_COS, PROBE_MRE)
    assert ab is not None
    assert ba is not None
    assert math.isclose(
        ab["cosine_similarity"], ba["cosine_similarity"], rel_tol=1e-4, abs_tol=1e-4
    )


@given(_nonzero_vectors(size=48), _nonzero_vectors(size=48), _nonzero_vectors(size=48))
@settings(max_examples=300)
def test_appending_matching_block_does_not_increase_max_rel_error(
    a: torch.Tensor, b: torch.Tensor, m: torch.Tensor
) -> None:
    """M5: appending an identical block to both sides never increases the
    max_relative_error.

    The appended block contributes relative error 0 (|m-m| / (|m|+eps) == 0), so
    the max over the union of (original errors, zeros) is unchanged. This is the per-tensor
    analogue of "appending matching ticks never increases replay divergence."

    NB: the analogous claim for *cosine* is FALSE and was refuted by this suite —
    cosine is scale-invariant, so for a=x, b=2x (cosine 1.0) appending a shared
    block injects absolute scale and DROPS cosine below 1.0. See PLATOON-FINDINGS
    (Finding R3): the divergence metric is not monotone under tick-append, so a
    higher-level "appending matching ticks" invariant cannot rely on cosine.
    """
    before = compare_activations(a, b, PROBE_COS, PROBE_MRE)
    after = compare_activations(torch.cat([a, m]), torch.cat([b, m]), PROBE_COS, PROBE_MRE)
    assert before is not None
    assert after is not None
    assert after["max_relative_error"] <= before["max_relative_error"] + 1e-4


@given(_nonzero_vectors(), st.floats(0.0, 1.0), st.floats(0.0, 1.0))
@settings(max_examples=200)
def test_threshold_monotonicity(a: torch.Tensor, t1: float, t2: float) -> None:
    """M6: tightening the cosine threshold can only turn 'no divergence' into
    'divergence', never the reverse, holding inputs fixed. (Monotone decision.)"""
    b = a + torch.randn_like(a) * 0.01
    lo, hi = min(t1, t2), max(t1, t2)
    # higher cosine_threshold is stricter
    strict = compare_activations(a, b, hi, PROBE_MRE)
    loose = compare_activations(a, b, lo, PROBE_MRE)
    if loose is not None:
        # if the looser (lower) threshold already reported divergence,
        # the stricter one must too
        assert strict is not None


# --------------------------------------------------------------------------- #
# from_ptr equivalence (roundtrip through the FFI byte path)
# --------------------------------------------------------------------------- #
@given(_nonzero_vectors(), _nonzero_vectors())
@settings(max_examples=200, deadline=None)
def test_from_ptr_matches_direct(a: torch.Tensor, b: torch.Tensor) -> None:
    """M7: compare_activations_from_ptr over a's raw bytes == compare_activations.

    The FFI variant reconstructs the original tensor from a raw pointer; it must
    produce identical metrics to the in-memory path for the same data.
    """
    a32 = a.float().contiguous()
    buf = (ctypes.c_char * (a32.nelement() * 4)).from_buffer_copy(a32.numpy().tobytes())
    ptr = ctypes.addressof(buf)
    direct = compare_activations(a32, b, PROBE_COS, PROBE_MRE)
    via_ptr = compare_activations_from_ptr(
        ptr, len(buf), "torch.float32", [a32.nelement()], b, PROBE_COS, PROBE_MRE
    )
    assert (direct is None) == (via_ptr is None)
    if direct is not None and via_ptr is not None:
        assert math.isclose(
            direct["cosine_similarity"], via_ptr["cosine_similarity"], abs_tol=1e-5
        )
        assert math.isclose(
            direct["max_relative_error"], via_ptr["max_relative_error"], abs_tol=1e-3
        )
    del buf


# --------------------------------------------------------------------------- #
# Exception-raising / boundary behaviour
# --------------------------------------------------------------------------- #
_KNOWN_DTYPES = {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}


@given(st.text(min_size=0, max_size=20))
@settings(max_examples=200)
def test_unsupported_dtype_string_raises_valueerror(dtype_str: str) -> None:
    """E1: any dtype string outside the supported set raises ValueError, never
    a panic or silent coercion."""
    assume(dtype_str not in _KNOWN_DTYPES)
    event(f"dtype rejected: {dtype_str[:12]!r}")
    with pytest.raises(ValueError, match="unsupported dtype"):
        compare_activations_from_ptr(0, 0, dtype_str, [1], torch.zeros(1), 0.999, 0.05)


@given(st.integers(1, 64), st.integers(1, 64))
@settings(max_examples=100)
def test_shape_mismatch_raises_runtimeerror(n: int, m: int) -> None:
    """E2: comparing tensors with different element counts raises (it does not
    silently truncate or hang). NOTE: this pins *current* behaviour — a generic
    torch RuntimeError. See PLATOON-FINDINGS.md: a clearer ValueError with a
    shape message would be the stronger contract."""
    assume(n != m)
    with pytest.raises(RuntimeError):
        compare_activations(torch.ones(n), torch.ones(m), 0.999, 0.05)


def test_zero_vs_zero_is_no_divergence() -> None:
    """E3: zero-tensor boundary — equal zero tensors report no divergence."""
    z = torch.zeros(64)
    assert compare_activations(z, z.clone(), 0.999, 0.05) is None


@given(_nonzero_vectors())
@settings(max_examples=100)
def test_zero_vs_nonzero_reports_divergence(a: torch.Tensor) -> None:
    """E4: a zero original against a non-zero replay must report divergence
    (denominator-zero branch falls through to cosine 0.0)."""
    z = torch.zeros_like(a)
    # original zero, replayed non-zero
    result = compare_activations(z, a, 0.999, 0.05)
    assert result is not None


@given(_nonzero_vectors(), st.integers(0, 47))
@settings(max_examples=200)
def test_nan_in_replay_is_silently_swallowed_known_bug(a: torch.Tensor, idx: int) -> None:
    """E5 (KNOWN BUG, pinned): a NaN in the replayed tensor is reported as
    *no divergence*.

    cosine_sim and max_relative_error both become NaN; the guard
    ``cosine_sim < thr or mre > thr`` is False for NaN on both sides, so the
    function returns None. For a debugger, a replay that produced NaN is the most
    important divergence to surface — and it is silently dropped. See
    PLATOON-FINDINGS.md (Finding R1). This test pins the current behaviour so a
    fix flips it deliberately."""
    a = a.clone()
    b = a.clone()
    b[idx % len(b)] = float("nan")
    result = compare_activations(a, b, cosine_threshold=0.999, mre_threshold=0.05)
    # Current (buggy) behaviour: NaN divergence is swallowed.
    assert result is None
