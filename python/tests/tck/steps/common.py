"""Shared step definitions for the TCK harness.

All step implementations are stubs. They define the patterns so pytest-bdd
can resolve every step in the 16 feature files without StepDefinitionNotFoundError.
Real implementations come in Phase 1 as each feature is built.
"""

from __future__ import annotations

from pytest_bdd import given, parsers, then, when

# ---------------------------------------------------------------------------
# Given steps
# ---------------------------------------------------------------------------


@given("a rocket_surgeon server is running")
def given_server_running() -> None:
    pass


@given(parsers.re(r'the session is in "(?P<state>[^"]+)" state.*'))
def given_session_in_state(state: str) -> None:
    pass


@given(parsers.re(r'the session is initialized with protocol_version "(?P<version>[^"]+)"'))
def given_session_initialized(version: str) -> None:
    pass


@given("the session is initialized and a model is attached")
def given_session_initialized_attached() -> None:
    pass


@given(parsers.re(r'a model "(?P<name>[^"]+)" is attached.*'))
def given_model_attached(name: str) -> None:
    pass


@given(parsers.re(r"the session has been stepped to tick (?P<tick>\d+) at layer (?P<layer>\d+)"))
def given_stepped_to(tick: str, layer: str) -> None:
    pass


@given(parsers.re(r'the server capability "(?P<cap>[^"]+)" is (?P<value>.+)'))
def given_server_capability(cap: str, value: str) -> None:
    pass


@given(
    parsers.re(
        r'the session has a[n]? (?P<tier>\w+) checkpoint "(?P<cid>[^"]+)"'
        r" at tick (?P<tick>\d+) layer (?P<layer>\d+)"
    )
)
def given_checkpoint(tier: str, cid: str, tick: str, layer: str) -> None:
    pass


@given(
    parsers.re(
        r'a defined probe "(?P<pid>[^"]+)" at point "(?P<point>[^"]+)"'
        r' with action "(?P<action>[^"]+)".*'
    )
)
def given_probe_defined(pid: str, point: str, action: str) -> None:
    pass


@given(
    parsers.re(
        r'an active intervention "(?P<iid>[^"]+)" of type "(?P<itype>[^"]+)"'
        r' on "(?P<target>[^"]+)".*'
    )
)
def given_active_intervention(iid: str, itype: str, target: str) -> None:
    pass


@given(parsers.re(r"no (?P<things>.+) have been (?P<action>.+) in this session"))
def given_nothing_done(things: str, action: str) -> None:
    pass


@given(
    parsers.re(
        r"the client has stepped forward (?:at least )?(?P<n>\d+) ticks?"
        r' at "(?P<gran>[^"]+)" granularity'
    )
)
def given_client_stepped(n: str, gran: str) -> None:
    pass


@given(parsers.re(r"the tensor store capacity is configured to hold at most (?P<n>\d+) tensors"))
def given_tensor_store_capacity(n: str) -> None:
    pass


@given(parsers.re(r'the session has a tensor "(?P<tid>[^"]+)" with shape (?P<shape>.+)'))
def given_tensor_exists(tid: str, shape: str) -> None:
    pass


@given(parsers.re(r"the host process is configured to simulate a crash.*"))
def given_simulate_crash() -> None:
    pass


@given(parsers.re(r"the GPU memory is configured to simulate OOM.*"))
def given_simulate_oom() -> None:
    pass


@given(parsers.re(r"the NCCL backend is configured to simulate a timeout"))
def given_simulate_nccl_timeout() -> None:
    pass


@given(parsers.re(r'a tensor with id "(?P<tid>[^"]+)" exists in the tensor store'))
def given_tensor_in_store(tid: str) -> None:
    pass


@given("the model has two probe points observing the same tensor content")
def given_two_probes_same_content() -> None:
    pass


@given(
    parsers.re(
        r'a bundle (?:file "(?P<path>[^"]+)" with protocol_version'
        r' "(?P<ver>[^"]+)"|has been exported to "(?P<epath>[^"]+)")'
    )
)
def given_bundle(
    path: str | None = None,
    ver: str | None = None,
    epath: str | None = None,
) -> None:
    pass


@given(parsers.re(r'the client has subscribed to "(?P<event>[^"]+)" events'))
def given_client_subscribed(event: str) -> None:
    pass


@given(parsers.re(r"the routing decision selected experts (?P<experts>.+)"))
def given_routing_decision(experts: str) -> None:
    pass


@given(parsers.re(r'the session has been advanced to "(?P<state>[^"]+)" state'))
def given_advanced_to_state(state: str) -> None:
    pass


@given(parsers.re(r'the client steps forward (?P<n>\d+) ticks? at "(?P<gran>[^"]+)" granularity'))
def given_client_steps_forward(n: str, gran: str) -> None:
    pass


@given(parsers.re(r'the resulting tick_id is saved as "(?P<name>[^"]+)"'))
def given_tick_saved(name: str) -> None:
    pass


@given(parsers.re(r'the client sends "(?P<verb>[^"]+)" with no parameters'))
def given_client_sends_no_params(verb: str) -> None:
    pass


@given(parsers.re(r'the client sends "(?P<verb>[^"]+)" with:'))
def given_client_sends_verb(verb: str) -> None:
    pass


# ---------------------------------------------------------------------------
# When steps
# ---------------------------------------------------------------------------


@when(parsers.re(r'the client sends "(?P<verb>[^"]+)" with:'))
def when_client_sends_verb(verb: str) -> None:
    pass


@when(parsers.re(r'the client sends "(?P<verb>[^"]+)" with no parameters'))
def when_client_sends_no_params(verb: str) -> None:
    pass


@when(parsers.re(r'the client sends "(?P<verb>[^"]+)" with direction "(?P<direction>[^"]+)"'))
def when_client_sends_direction(verb: str, direction: str) -> None:
    pass


@when(parsers.re(r'the client sends "(?P<verb>[^"]+)" expecting an error'))
def when_client_sends_expecting_error(verb: str) -> None:
    pass


@when(parsers.re(r'the request includes "(?P<field>[^"]+)" (?:array|object):'))
def when_request_includes(field: str) -> None:
    pass


@when(parsers.re(r'the client executes (?P<n>\d+) forward steps at "(?P<gran>[^"]+)" granularity'))
def when_client_executes_steps(n: str, gran: str) -> None:
    pass


@when(parsers.re(r'the client subscribes to "(?P<event>[^"]+)" events'))
def when_client_subscribes(event: str) -> None:
    pass


@when(
    parsers.re(
        r"the client captures (?P<n>\d+) tensors"
        r" by inspecting (?P<m>\d+) distinct components"
    )
)
def when_client_captures_tensors(n: str, m: str) -> None:
    pass


@when(parsers.re(r"the session is reset to stopped at tick (?P<tick>\d+)"))
def when_session_reset(tick: str) -> None:
    pass


@when(parsers.re(r'the session remains in "(?P<state>[^"]+)" state for (?P<n>\d+) seconds'))
def when_session_remains(state: str, n: str) -> None:
    pass


@when(parsers.re(r"the forward pass reaches layer (?P<layer>\d+).*"))
def when_forward_pass_reaches(layer: str) -> None:
    pass


@when(parsers.re(r'client "(?P<name>[^"]+)" sends "(?P<verb>[^"]+)" with:'))
def when_named_client_sends(name: str, verb: str) -> None:
    pass


@when(
    parsers.re(
        r"the client steps forward until the forward pass"
        r" (?:reaches|completes) layer (?P<layer>\d+)"
    )
)
def when_client_steps_to_layer(layer: str) -> None:
    pass


@when(parsers.re(r'the response "(?P<path>[^"]+)" is saved as "(?P<name>[^"]+)"'))
def when_response_saved(path: str, name: str) -> None:
    pass


@when(parsers.re(r'the first captured tensor_id is saved as "(?P<name>[^"]+)"'))
def when_first_tensor_saved(name: str) -> None:
    pass


@when(parsers.re(r'the first tensor "(?P<field>[^"]+)" is saved as "(?P<name>[^"]+)"'))
def when_tensor_field_saved(field: str, name: str) -> None:
    pass


@when(parsers.re(r'the client steps forward (?P<n>\d+) ticks? at "(?P<gran>[^"]+)" granularity'))
def when_client_steps_forward(n: str, gran: str) -> None:
    pass


@when(parsers.re(r'the resulting tick_id is saved as "(?P<name>[^"]+)"'))
def when_tick_saved(name: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Then steps
# ---------------------------------------------------------------------------


@then(parsers.re(r'the response status is "(?P<status>[^"]+)"'))
def then_response_status(status: str) -> None:
    pass


@then("the response is a JSON-RPC error")
def then_response_is_error() -> None:
    pass


@then(parsers.re(r'the response has a "(?P<field>[^"]+)" object'))
def then_response_has_object(field: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" is "(?P<value>[^"]*)"'))
def then_response_path_is(path: str, value: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" is not null'))
def then_response_not_null(path: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" is null'))
def then_response_null(path: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" is a non-empty string'))
def then_response_nonempty_string(path: str) -> None:
    pass


@then(
    parsers.re(
        r'the response "(?P<path>[^"]+)" has field'
        r' "(?P<field>[^"]+)" of type (?P<ftype>\w+)'
    )
)
def then_response_field_type(path: str, field: str, ftype: str) -> None:
    pass


@then(
    parsers.re(
        r'the response "(?P<path>[^"]+)" is an array'
        r" with (?:at least|exactly) (?P<n>\d+) elements?"
    )
)
def then_response_array_length(path: str, n: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" is of type (?P<ftype>\w+)'))
def then_response_of_type(path: str, ftype: str) -> None:
    pass


@then(
    parsers.re(
        r'the response "(?P<path>[^"]+)" matches'
        r' UUID format(?:\s+"(?P<pattern>[^"]+)")?'
    )
)
def then_response_matches_uuid(path: str, pattern: str | None = None) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" matches "(?P<pattern>[^"]+)"'))
def then_response_matches_pattern(path: str, pattern: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" includes (?:at least:)?.*'))
def then_response_includes(path: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" contains (?:an entry|"[^"]+").*'))
def then_response_contains(path: str) -> None:
    pass


@then(
    parsers.re(
        r'the response "(?P<path>[^"]+)"'
        r" (?:is a boolean|is an? .+|is the array .+"
        r"|has \d+ entr.+|does not contain .+|>= \d+)"
    )
)
def then_response_assertion(path: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" is a positive integer'))
def then_response_positive_int(path: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" is (?P<value>\d+)'))
def then_response_is_number(path: str, value: str) -> None:
    pass


@then(parsers.re(r'the error "(?P<path>[^"]+)" is "(?P<value>[^"]*)"'))
def then_error_field_is(path: str, value: str) -> None:
    pass


@then(parsers.re(r'the error "(?P<path>[^"]+)" is an integer'))
def then_error_is_integer(path: str) -> None:
    pass


@then(parsers.re(r'the error "(?P<path>[^"]+)" equals the error "(?P<path2>[^"]+)"'))
def then_error_equals_error(path: str, path2: str) -> None:
    pass


@then(parsers.re(r'the error "(?P<path>[^"]+)" is one of "(?P<a>[^"]+)", "(?P<b>[^"]+)"'))
def then_error_one_of(path: str, a: str, b: str) -> None:
    pass


@then(parsers.re(r'the error "(?P<path>[^"]+)" is a non-empty (?:string|array)'))
def then_error_nonempty(path: str) -> None:
    pass


@then(parsers.re(r'the error "(?P<path>[^"]+)" includes "(?P<value>[^"]+)"'))
def then_error_includes(path: str, value: str) -> None:
    pass


@then(parsers.re(r'each entry in (?:error )?"(?P<path>[^"]+)" is a valid (?P<what>.+)'))
def then_each_entry_valid(path: str, what: str) -> None:
    pass


@then(parsers.re(r'"(?P<a>[^"]+)" equals "(?P<b>[^"]+)"'))
def then_values_equal(a: str, b: str) -> None:
    pass


@then(parsers.re(r'"(?P<a>[^"]+)" < "(?P<b>[^"]+)" < "(?P<c>[^"]+)"'))
def then_values_ordered(a: str, b: str, c: str) -> None:
    pass


@then(parsers.re(r'"(?P<a>[^"]+)" > "(?P<b>[^"]+)"'))
def then_value_greater(a: str, b: str) -> None:
    pass


@then(parsers.re(r'"(?P<a>[^"]+)" advanced further in layer index than "(?P<b>[^"]+)"'))
def then_advanced_further(a: str, b: str) -> None:
    pass


@then(parsers.re(r'all observed (?:tick_ids|"[^"]+" values) are unique'))
def then_all_unique() -> None:
    pass


@then(parsers.re(r'the client receives (?:a|at least \d+) "(?P<event>[^"]+)" notifications?.*'))
def then_client_receives_notification(event: str) -> None:
    pass


@then(parsers.re(r'the client does not receive (?:a )?"(?P<event>[^"]+)" notification.*'))
def then_client_not_receives(event: str) -> None:
    pass


@then(parsers.re(r'the notification (?:includes|"params\.[^"]+") .*'))
def then_notification_includes() -> None:
    pass


@then(parsers.re(r'the most recent response "(?P<path>[^"]+)" is (?P<rest>.+)'))
def then_most_recent_response(path: str, rest: str) -> None:
    pass


@then(parsers.re(r'the response data field "(?P<field>[^"]+)" (?P<rest>.+)'))
def then_response_data_field(field: str, rest: str) -> None:
    pass


@then(parsers.re(r'the entry "(?P<eid>[^"]+)" has (?P<rest>.+)'))
def then_entry_has(eid: str, rest: str) -> None:
    pass


@then(parsers.re(r'the first tensor (?:in )?"(?P<path>[^"]+)" (?P<rest>.+)'))
def then_first_tensor(path: str, rest: str) -> None:
    pass


@then(parsers.re(r'the tensor "(?P<tid>[^"]+)" has been evicted'))
def then_tensor_evicted(tid: str) -> None:
    pass


@then(parsers.re(r"the tensor store contains at most (?P<n>\d+) tensors"))
def then_tensor_store_size(n: str) -> None:
    pass


@then(parsers.re(r"the decoded slice data length in bytes equals (?P<expr>.+)"))
def then_slice_data_length(expr: str) -> None:
    pass


@then(
    parsers.re(
        r'the bundle (?:at )?"(?P<path>[^"]+)" contains field'
        r' "(?P<field>[^"]+)" (?P<rest>.+)'
    )
)
def then_bundle_contains(path: str, field: str, rest: str) -> None:
    pass


@then(parsers.re(r'the bundle "(?P<path>[^"]+)" (?P<rest>.+)'))
def then_bundle_field(path: str, rest: str) -> None:
    pass


@then(parsers.re(r'the file "(?P<path>[^"]+)" (?:exists|is valid JSON)'))
def then_file_check(path: str) -> None:
    pass


@then(parsers.re(r'probe "(?P<pid>[^"]+)" (?:fires|captures) .*'))
def then_probe_fires(pid: str) -> None:
    pass


@then(parsers.re(r'intervention "(?P<iid>[^"]+)" (?:is applied|executes) .*'))
def then_intervention_applied(iid: str) -> None:
    pass


@then(parsers.re(r'the non-mutating probe "(?P<pid>[^"]+)" executes before .*'))
def then_probe_before_intervention(pid: str) -> None:
    pass


@then(parsers.re(r'only intervention "(?P<iid>[^"]+)" takes effect.*'))
def then_only_intervention(iid: str) -> None:
    pass


@then(parsers.re(r'the prior additive intervention "(?P<iid>[^"]+)" is overridden.*'))
def then_intervention_overridden(iid: str) -> None:
    pass


@then(parsers.re(r'execution pauses at "(?P<point>[^"]+)"'))
def then_execution_pauses(point: str) -> None:
    pass


@then(parsers.re(r'a "(?P<event>[^"]+)" event is emitted.*'))
def then_event_emitted(event: str) -> None:
    pass


@then(parsers.re(r"the event includes .*"))
def then_event_includes() -> None:
    pass


@then(parsers.re(r'layer (?P<layer>\d+) ticks at "(?P<gran>[^"]+)" granularity'))
def then_layer_granularity(layer: str, gran: str) -> None:
    pass


@then(parsers.re(r'all other layers tick at "(?P<gran>[^"]+)" granularity'))
def then_other_layers_granularity(gran: str) -> None:
    pass


@then(parsers.re(r'each "(?P<event>[^"]+)" notification includes .*'))
def then_each_notification_includes(event: str) -> None:
    pass


@then(parsers.re(r'client "(?P<name>[^"]+)" receives .*'))
def then_named_client_receives(name: str) -> None:
    pass


@then(parsers.re(r'client "(?P<name>[^"]+)" subscription_id differs from.*'))
def then_subscription_ids_differ(name: str) -> None:
    pass


@then(
    parsers.re(
        r'the notification tensor "(?P<field>[^"]+)"'
        r' matches the pattern "(?P<pattern>[^"]+)"'
    )
)
def then_notification_tensor_matches(field: str, pattern: str) -> None:
    pass


@then(parsers.re(r'every "(?P<event>[^"]+)" notification for "(?P<pid>[^"]+)" has .*'))
def then_every_notification_has(event: str, pid: str) -> None:
    pass


@then(parsers.re(r'every entry in "(?P<path>[^"]+)" (?P<rest>.+)'))
def then_every_entry(path: str, rest: str) -> None:
    pass


@then(parsers.re(r'each entry in "(?P<path>[^"]+)" includes:'))
def then_each_entry_includes(path: str) -> None:
    pass


@then(parsers.re(r'the capabilities field "(?P<field>[^"]+)" (?P<rest>.+)'))
def then_capabilities_field(field: str, rest: str) -> None:
    pass


@then(parsers.re(r'the response contains a "(?P<field>[^"]+)" object'))
def then_response_contains_object(field: str) -> None:
    pass


@then(parsers.re(r'the response data contains a "(?P<field>[^"]+)" object'))
def then_response_data_contains_object(field: str) -> None:
    pass


@then(parsers.re(r"the server does not return an error"))
def then_no_error() -> None:
    pass


@then(parsers.re(r'"(?P<a>[^"]+)" and "(?P<b>[^"]+)" are distinct'))
def then_values_distinct(a: str, b: str) -> None:
    pass


@then(parsers.re(r"the set \{.*\} equals \{.*\}"))
def then_set_equals() -> None:
    pass


@then(parsers.re(r"this is verified from the prior .*"))
def then_verified_from_prior() -> None:
    pass


@then(parsers.re(r'"(?P<a>[^"]+)" does not equal "(?P<b>[^"]+)"'))
def then_values_not_equal(a: str, b: str) -> None:
    pass


@then(parsers.re(r'each entry in "(?P<path>[^"]+)" is a non-empty string'))
def then_each_entry_nonempty_string(path: str) -> None:
    pass


@then(
    parsers.re(
        r'each entry in "(?P<path>[^"]+)"'
        r" is an integer in range \[(?P<lo>\d+), (?P<hi>\d+)\]"
    )
)
def then_each_entry_integer_range(path: str, lo: str, hi: str) -> None:
    pass


@then(parsers.re(r'the client receives "(?P<event>[^"]+)" notifications only for (?P<scope>.+)'))
def then_client_receives_filtered(event: str, scope: str) -> None:
    pass


@then(parsers.re(r'the most recent response "(?P<path>[^"]+)" equals (?P<value>.+)'))
def then_most_recent_equals(path: str, value: str) -> None:
    pass


@then(parsers.re(r'the most recent response "(?P<path>[^"]+)" includes "(?P<value>[^"]+)"'))
def then_most_recent_includes(path: str, value: str) -> None:
    pass


@then(parsers.re(r'the notification "(?P<path>[^"]+)" includes:'))
def then_notification_path_includes(path: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" contains at least:'))
def then_response_contains_at_least(path: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" is less than (?P<value>.+)'))
def then_response_less_than(path: str, value: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" starts with "(?P<prefix>[^"]*)"'))
def then_response_starts_with(path: str, prefix: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" is one of "(?P<a>[^"]+)" or "(?P<b>[^"]+)"'))
def then_response_one_of(path: str, a: str, b: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" is greater than (?P<value>\d+)'))
def then_response_greater_than(path: str, value: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" has exactly (?P<n>\d+) (?:elements?|entries)'))
def then_response_exact_count(path: str, n: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" includes:'))
def then_response_includes_table(path: str) -> None:
    pass


@then(parsers.re(r'the session is in "(?P<state>[^"]+)" state'))
def then_session_in_state(state: str) -> None:
    pass


@then(parsers.re(r'each entry in "(?P<path>[^"]+)" is a number > (?P<threshold>\d+)'))
def then_each_entry_number_gt(path: str, threshold: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" matches the bundle "(?P<field>[^"]+)"'))
def then_response_matches_bundle(path: str, field: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" is not empty'))
def then_response_not_empty(path: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Perfetto trace — Given steps
# ---------------------------------------------------------------------------


@given('a PerfettoSink has been created for session "test-session" with model "gpt2"')
def given_perfetto_sink_created() -> None:
    pass


@given(parsers.re(r"a TraceSink is opened for writing"))
def given_trace_sink_opened() -> None:
    pass


@given(parsers.re(r"rank (?P<rank>\d+) has been declared"))
def given_rank_declared(rank: str) -> None:
    pass


@given(parsers.re(r"layer (?P<layer>\d+) under rank (?P<rank>\d+) has been declared"))
def given_layer_declared(layer: str, rank: str) -> None:
    pass


@given(
    parsers.re(
        r'component "(?P<name>[^"]+)" at index (?P<idx>\d+) '
        r"under layer (?P<layer>\d+) rank (?P<rank>\d+) has been declared"
    )
)
def given_component_declared(name: str, idx: str, layer: str, rank: str) -> None:
    pass


@given(parsers.re(r"interned names have been emitted for rank (?P<rank>\d+)"))
def given_interned_names_emitted(rank: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Perfetto trace — When steps
# ---------------------------------------------------------------------------


@when(parsers.re(r"a TracePacket is written with timestamp (?P<ts>\d+)"))
def when_trace_packet_written(ts: str) -> None:
    pass


@when(
    parsers.re(
        r"a process track is written with uuid (?P<uuid>\d+) "
        r'and name "(?P<name>[^"]+)"'
    )
)
def when_process_track_written(uuid: str, name: str) -> None:
    pass


@when(
    parsers.re(
        r"a thread track is written with uuid (?P<uuid>\d+) "
        r"parent (?P<parent>\d+)"
    )
)
def when_thread_track_written(uuid: str, parent: str) -> None:
    pass


@when(
    parsers.re(
        r"on_tick_stopped is called with layer (?P<layer>\d+) "
        r'component "(?P<component>[^"]+)"'
    )
)
def when_on_tick_stopped(layer: str, component: str) -> None:
    pass


@when(
    parsers.re(
        r'on_probe_fired is called with probe_id "(?P<pid>[^"]+)" '
        r"and tensor summary"
    )
)
def when_on_probe_fired(pid: str) -> None:
    pass


@when(parsers.re(r"close is called on the PerfettoSink"))
def when_perfetto_close() -> None:
    pass


# ---------------------------------------------------------------------------
# Perfetto trace — Then steps
# ---------------------------------------------------------------------------


@then(parsers.re(r"the output begins with byte 0x0A"))
def then_output_begins_0a() -> None:
    pass


@then(parsers.re(r"the output is valid field-1 framed protobuf"))
def then_output_valid_field1() -> None:
    pass


@then(parsers.re(r"each packet in the output decodes as a valid TracePacket"))
def then_each_packet_valid() -> None:
    pass


@then(parsers.re(r"the re-encoded packet equals the original bytes"))
def then_reencoded_equals_original() -> None:
    pass


@then(parsers.re(r"the output contains exactly (?P<n>\d+) TracePackets?"))
def then_output_packet_count(n: str) -> None:
    pass


@then(parsers.re(r"the output contains at least (?P<n>\d+) TracePackets?"))
def then_output_min_packet_count(n: str) -> None:
    pass


@then(
    parsers.re(
        r"a TrackDescriptor packet exists with uuid (?P<uuid>\d+) "
        r'and name "(?P<name>[^"]+)"'
    )
)
def then_track_descriptor_exists(uuid: str, name: str) -> None:
    pass


@then(
    parsers.re(
        r"a TrackDescriptor packet exists with uuid (?P<uuid>\d+) "
        r"and parent_uuid (?P<parent>\d+)"
    )
)
def then_track_descriptor_parent(uuid: str, parent: str) -> None:
    pass


@then(parsers.re(r"the process track has a ProcessDescriptor"))
def then_process_track_has_descriptor() -> None:
    pass


@then(parsers.re(r"the rank track has a ThreadDescriptor"))
def then_rank_track_has_descriptor() -> None:
    pass


@then(parsers.re(r"every child track has child_ordering set to EXPLICIT"))
def then_child_ordering_explicit() -> None:
    pass


@then(parsers.re(r"a SLICE_BEGIN TrackEvent exists on the component track"))
def then_slice_begin_exists() -> None:
    pass


@then(parsers.re(r"a SLICE_END TrackEvent exists on the component track"))
def then_slice_end_exists() -> None:
    pass


@then(parsers.re(r'a TYPE_INSTANT TrackEvent exists with name "(?P<name>[^"]+)"'))
def then_instant_event_exists(name: str) -> None:
    pass


@then(
    parsers.re(
        r"the instant event has DebugAnnotations for "
        r'"(?P<fields>[^"]+)"'
    )
)
def then_instant_has_annotations(fields: str) -> None:
    pass


@then(
    parsers.re(
        r"the InternedData packet has sequence_flags "
        r"SEQ_INCREMENTAL_STATE_CLEARED"
    )
)
def then_interned_data_flags() -> None:
    pass


@then(parsers.re(r"each interned name has a unique iid starting from 1"))
def then_interned_iids_unique() -> None:
    pass


@then(parsers.re(r"all open slices have been terminated with SLICE_END"))
def then_all_slices_closed() -> None:
    pass


@then(
    parsers.re(
        r'the trace file exists at "(?P<pattern>[^"]+)" '
        r"with non-zero size"
    )
)
def then_trace_file_exists(pattern: str) -> None:
    pass


@then(parsers.re(r"the \.pftrace file is valid field-1 framed protobuf"))
def then_pftrace_valid() -> None:
    pass
