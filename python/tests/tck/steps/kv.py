"""Step definitions for the KV-cache feature (WU-G).

These steps cover `tck/protocol/kv-cache.feature` — the `rocket/kv.read` and
`rocket/kv.intervene` protocol surface. Like the rest of the TCK harness they
are stubs: they define every step pattern so pytest-bdd resolves the feature
without `StepDefinitionNotFoundError`. A real implementation is wired in once
the integrator runs the harness against a live daemon.
"""

from __future__ import annotations

from pytest_bdd import given, parsers, then, when

# ---------------------------------------------------------------------------
# Given steps
# ---------------------------------------------------------------------------


@given("an attached session with KV cache populated")
def given_kv_cache_populated() -> None:
    pass


@given(parsers.re(r"a session where position (?P<pos>\d+) was evicted"))
def given_position_evicted(pos: str) -> None:
    pass


# ---------------------------------------------------------------------------
# When steps
# ---------------------------------------------------------------------------


@when(
    parsers.re(
        r"the client sends kv\.read with layers \[(?P<layers>[^\]]*)\],"
        r" positions \[(?P<positions>[^\]]*)\]"
    )
)
def when_kv_read_layers_positions(layers: str, positions: str) -> None:
    pass


@when(parsers.re(r"the client sends kv\.read for position (?P<pos>\d+)"))
def when_kv_read_position(pos: str) -> None:
    pass


@when(
    parsers.re(
        r'the client sends kv\.intervene with op "(?P<op>[^"]+)"'
        r" on position (?P<pos>\d+)"
    )
)
def when_kv_intervene_on_position(op: str, pos: str) -> None:
    pass


@when(parsers.re(r'the client sends kv\.intervene with op "(?P<op>[^"]+)" and empty layers'))
def when_kv_intervene_empty_layers(op: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Then steps
# ---------------------------------------------------------------------------


@then("the response includes cache entries with norms per layer and position")
def then_response_has_cache_entries() -> None:
    pass


@then(parsers.re(r'the error code is "(?P<code>[^"]+)"'))
def then_error_code_is(code: str) -> None:
    pass


@then("error context includes evicted_at_tick and nearest_checkpoint")
def then_error_context_includes_eviction() -> None:
    pass


@then(parsers.re(r'the response reports the applied op "(?P<op>[^"]+)"'))
def then_response_reports_applied_op(op: str) -> None:
    pass


@then("the response reports a positive slots_modified count")
def then_response_reports_slots_modified() -> None:
    pass
