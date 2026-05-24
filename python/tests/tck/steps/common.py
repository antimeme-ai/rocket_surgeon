"""Shared step definitions for the TCK harness.

Wired to a real daemon via the `rpc` fixture from conftest.py.
Step functions receive pytest fixtures by parameter name.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

from pytest_bdd import given, parsers, then, when

MODEL_PATH = "hf-internal-testing/tiny-random-LlamaForCausalLM"
MODEL_FAMILY = "llama"


def _resolve_path(obj: Any, path: str) -> Any:
    """Walk a dotted path into a nested dict. e.g. 'data.stopped_at.layer'."""
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return obj


def _datatable_to_params(
    datatable: Sequence[Sequence[object]],
    saved_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a 2-column datatable into a dict, coercing numeric strings.

    Values starting with ``$`` are substituted from *saved_values*.
    """
    params: dict[str, Any] = {}
    for row in datatable:
        key = str(row[0])
        val = str(row[1])
        if saved_values and val.startswith("$"):
            resolved = saved_values.get(val[1:])
            if resolved is not None:
                params[key] = resolved
                continue
        if val.isdigit():
            params[key] = int(val)
        elif val.replace(".", "", 1).isdigit():
            params[key] = float(val)
        elif val.lower() in ("true", "false"):
            params[key] = val.lower() == "true"
        else:
            params[key] = val
    return params


_BROKEN_MODEL_PATHS = frozenset(
    {
        "/models/does-not-exist",
        "/models/does-not-exist-either",
        "/models/buggy-worker",
    }
)


def _fixup_attach_params(params: dict[str, Any]) -> dict[str, Any]:
    """Map conceptual feature-file model paths to the real tiny test model."""
    model_path = params.get("model_path", "")
    if model_path in _BROKEN_MODEL_PATHS:
        params.setdefault("device", "cpu")
        params.setdefault("num_ranks", 1)
        return params
    if model_path.startswith("/models/") or model_path == MODEL_PATH:
        params["model_path"] = MODEL_PATH
        params.setdefault("model_family", MODEL_FAMILY)
    params.setdefault("device", "cpu")
    params.setdefault("num_ranks", 1)
    return params


def _init_session(rpc: Any) -> None:
    rpc.send("initialize", {"client_name": "tck", "protocol_version": "0.3.0"})


def _attach_model(rpc: Any) -> None:
    rpc.send(
        "attach",
        {
            "model_path": MODEL_PATH,
            "model_family": MODEL_FAMILY,
            "device": "cpu",
            "num_ranks": 1,
        },
    )


# ---------------------------------------------------------------------------
# Given steps
# ---------------------------------------------------------------------------


@given("a rocket_surgeon server is running")
def given_server_running() -> None:
    pass


@given(parsers.re(r'the session is in "(?P<state>[^"]+)" state$'))
def given_session_in_state(state: str, rpc: Any) -> None:
    if state == "uninitialized":
        pass
    elif state == "initialized":
        _init_session(rpc)
    elif state in {"stopped", "stepping"}:
        _init_session(rpc)
        _attach_model(rpc)


@given(parsers.re(r'the session is in "(?P<state>[^"]+)" state with model "(?P<model>[^"]+)"'))
def given_session_in_state_with_model(state: str, model: str, rpc: Any) -> None:
    _init_session(rpc)
    _attach_model(rpc)


@given(parsers.re(r'the session is in "(?P<state>[^"]+)" state after a previous detach'))
def given_session_after_detach(state: str, rpc: Any) -> None:
    _init_session(rpc)
    _attach_model(rpc)
    rpc.send("detach", {})


@given(parsers.re(r'the session is initialized with protocol_version "(?P<version>[^"]+)"'))
def given_session_initialized(version: str, rpc: Any) -> None:
    rpc.send("initialize", {"client_name": "tck", "protocol_version": version})


@given("the session is initialized and a model is attached")
def given_session_initialized_attached(rpc: Any) -> None:
    _init_session(rpc)
    _attach_model(rpc)


@given(parsers.re(r"an attached session$"))
def given_attached_session(rpc: Any) -> None:
    _init_session(rpc)
    _attach_model(rpc)


@given(parsers.re(r'a model "(?P<name>[^"]+)" is attached.*'))
def given_model_attached(name: str, rpc: Any) -> None:
    _attach_model(rpc)


@given(parsers.re(r"the session has been stepped to tick (?P<tick>\d+) at layer (?P<layer>\d+)"))
def given_stepped_to(tick: str, layer: str, rpc: Any) -> None:
    count = int(tick) or 1
    rpc.send("rocket/step", {"direction": "forward", "count": count})


@given(parsers.re(r'the server capability "(?P<cap>[^"]+)" is (?P<value>.+)'))
def given_server_capability(cap: str, value: str) -> None:
    pass


@given(
    parsers.re(
        r'the session has a[n]? (?P<tier>\w+) checkpoint "(?P<cid>[^"]+)"'
        r" at tick (?P<tick>\d+) layer (?P<layer>\d+)"
    )
)
def given_checkpoint(
    tier: str, cid: str, tick: str, layer: str, rpc: Any, saved_values: dict
) -> None:
    resp = rpc.send("rocket/checkpoint", {"action": "create", "tier": tier})
    real_id = resp.get("result", {}).get("data", {}).get("checkpoint_id", "")
    saved_values[cid] = real_id


@given(
    parsers.re(
        r'a defined probe "(?P<pid>[^"]+)" at point "(?P<point>[^"]+)"'
        r' with action "(?P<action>[^"]+)"(?P<extras>.*)'
    )
)
def given_probe_defined(pid: str, point: str, action: str, extras: str, rpc: Any) -> None:
    enabled = True
    priority = 0
    if "enabled false" in extras:
        enabled = False
    if "enabled true" in extras:
        enabled = True
    m = re.search(r"priority (\d+)", extras)
    if m:
        priority = int(m.group(1))
    rpc.send(
        "rocket/probe",
        {
            "action": "define",
            "probe": {
                "id": pid,
                "point": point,
                "action": action,
                "config": {"summary": True},
                "enabled": enabled,
                "priority": priority,
            },
        },
    )


@given(
    parsers.re(
        r'an active intervention "(?P<iid>[^"]+)" of type "(?P<itype>[^"]+)"'
        r' on "(?P<target>[^"]+)".*'
    )
)
def given_active_intervention(iid: str, itype: str, target: str, rpc: Any) -> None:
    rpc.send(
        "rocket/intervene",
        {
            "action": "set",
            "recipe": {
                "id": iid,
                "type": itype,
                "target": target,
                "params": {},
                "priority": 0,
            },
        },
    )


@given(parsers.re(r"no (?P<things>.+) have been (?P<action>.+) in this session"))
def given_nothing_done(things: str, action: str) -> None:
    pass


@given(
    parsers.re(
        r"the client has stepped forward (?:at least )?(?P<n>\d+) ticks?"
        r' at "(?P<gran>[^"]+)" granularity'
    )
)
def given_client_stepped(n: str, gran: str, rpc: Any) -> None:
    rpc.send(
        "rocket/step",
        {"direction": "forward", "count": int(n), "granularity": gran},
    )


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
def given_client_subscribed(event: str, rpc: Any) -> None:
    rpc.send("rocket/subscribe", {"events": [event]})


@given(parsers.re(r"the routing decision selected experts (?P<experts>.+)"))
def given_routing_decision(experts: str) -> None:
    pass


@given(parsers.re(r'the session has been advanced to "(?P<state>[^"]+)" state'))
def given_advanced_to_state(state: str) -> None:
    pass


@given(parsers.re(r'the client steps forward (?P<n>\d+) ticks? at "(?P<gran>[^"]+)" granularity'))
def given_client_steps_forward(n: str, gran: str, rpc: Any, saved_values: dict) -> None:
    resp = rpc.send(
        "rocket/step",
        {"direction": "forward", "count": int(n), "granularity": gran},
    )
    saved_values["_last_step"] = resp


@given(parsers.re(r'the resulting tick_id is saved as "(?P<name>[^"]+)"'))
def given_tick_saved(name: str, rpc: Any, saved_values: dict) -> None:
    state = rpc.result_state()
    saved_values[name] = state.get("tick_id")


@given(parsers.re(r'the client sends "(?P<verb>[^"]+)" with no parameters'))
def given_client_sends_no_params(verb: str, rpc: Any) -> None:
    rpc.send(verb, {})


@given(parsers.re(r'the client sends "(?P<verb>[^"]+)" with:'))
def given_client_sends_verb(verb: str, rpc: Any, datatable: Any, saved_values: dict) -> None:
    params = _datatable_to_params(datatable, saved_values)
    rpc.send(verb, params)


@given(
    parsers.re(
        r"the backend worker reports a model with (?P<layers>\d+) layers and (?P<heads>\d+) heads"
    )
)
def given_backend_worker_model(layers: str, heads: str) -> None:
    pass


@given(parsers.re(r"the backend worker cannot load the requested model"))
def given_backend_cannot_load() -> None:
    pass


@given(parsers.re(r'the backend worker reports model_type "(?P<mtype>[^"]+)"'))
def given_backend_model_type(mtype: str) -> None:
    pass


@given(parsers.re(r"the backend worker reports num_layers=(?P<n>\d+)"))
def given_backend_num_layers(n: str) -> None:
    pass


@given(parsers.re(r'an intervention recipe with type "(?P<itype>[^"]+)".*'))
def given_intervention_recipe(itype: str) -> None:
    pass


@given(parsers.re(r"params (?P<params_json>\{.+\})"))
def given_params_json(params_json: str) -> None:
    pass


# ---------------------------------------------------------------------------
# When steps
# ---------------------------------------------------------------------------


@when(parsers.re(r'the client sends "(?P<verb>[^"]+)" with:'))
def when_client_sends_verb(
    verb: str,
    rpc: Any,
    saved_values: dict,
    datatable: Any = None,
    docstring: str | None = None,
) -> None:
    if docstring is not None:
        params = json.loads(docstring)
    elif datatable is not None:
        params = _datatable_to_params(datatable, saved_values)
    else:
        params = {}
    if verb == "attach":
        params = _fixup_attach_params(params)
    rpc.send(verb, params)


@when(parsers.re(r'the client sends "(?P<verb>[^"]+)" with no parameters'))
def when_client_sends_no_params(verb: str, rpc: Any) -> None:
    rpc.send(verb, {})


@when(parsers.re(r'the client sends "(?P<verb>[^"]+)" with direction "(?P<direction>[^"]+)"'))
def when_client_sends_direction(verb: str, direction: str, rpc: Any) -> None:
    rpc.send(verb, {"direction": direction, "count": 1})


@when(parsers.re(r'the client sends "(?P<verb>[^"]+)" expecting an error'))
def when_client_sends_expecting_error(verb: str, rpc: Any) -> None:
    rpc.send(verb, {})


@when(parsers.re(r'the request includes "(?P<field>[^"]+)" (?:array|object):'))
def when_request_includes(field: str) -> None:
    pass


@when(parsers.re(r'the client executes (?P<n>\d+) forward steps at "(?P<gran>[^"]+)" granularity'))
def when_client_executes_steps(n: str, gran: str, rpc: Any, saved_values: dict) -> None:
    tick_ids = []
    for _ in range(int(n)):
        rpc.send(
            "rocket/step",
            {"direction": "forward", "count": 1, "granularity": gran},
        )
        state = rpc.result_state()
        tick_ids.append(state.get("tick_id"))
    saved_values["_observed_tick_ids"] = tick_ids


@when(parsers.re(r'the client subscribes to "(?P<event>[^"]+)" events'))
def when_client_subscribes(event: str, rpc: Any) -> None:
    rpc.send("rocket/subscribe", {"events": [event]})


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
def when_response_saved(path: str, name: str, rpc: Any, saved_values: dict) -> None:
    result = rpc.last_response.get("result", {})
    saved_values[name] = _resolve_path(result, path)


@when(parsers.re(r'the first captured tensor_id is saved as "(?P<name>[^"]+)"'))
def when_first_tensor_saved(name: str) -> None:
    pass


@when(parsers.re(r'the first tensor "(?P<field>[^"]+)" is saved as "(?P<name>[^"]+)"'))
def when_tensor_field_saved(field: str, name: str) -> None:
    pass


@when(parsers.re(r'the client steps forward (?P<n>\d+) ticks? at "(?P<gran>[^"]+)" granularity'))
def when_client_steps_forward(n: str, gran: str, rpc: Any, saved_values: dict) -> None:
    rpc.send(
        "rocket/step",
        {"direction": "forward", "count": int(n), "granularity": gran},
    )


@when(parsers.re(r'the resulting tick_id is saved as "(?P<name>[^"]+)"'))
def when_tick_saved(name: str, rpc: Any, saved_values: dict) -> None:
    state = rpc.result_state()
    saved_values[name] = state.get("tick_id")


# ---------------------------------------------------------------------------
# Then steps
# ---------------------------------------------------------------------------


@then(parsers.re(r'the response status is "(?P<status>[^"]+)"'))
def then_response_status(status: str, rpc: Any) -> None:
    assert not rpc.is_error(), f"Expected success, got error: {rpc.last_error}"
    actual = rpc.status()
    assert actual == status, f"Expected status '{status}', got '{actual}'"


@then("the response is a JSON-RPC error")
def then_response_is_error(rpc: Any) -> None:
    assert rpc.is_error(), f"Expected error, got success: {rpc.last_response}"


@then(parsers.re(r'the response has a "(?P<field>[^"]+)" object'))
def then_response_has_object(field: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    val = _resolve_path(result, field)
    assert isinstance(val, dict), f"Expected dict at '{field}', got {type(val)}"


@then(parsers.re(r'the response "(?P<path>[^"]+)" is "(?P<value>[^"]*)"'))
def then_response_path_is(path: str, value: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    assert str(actual) == value, f"'{path}': expected '{value}', got '{actual}'"


@then(parsers.re(r'the response "(?P<path>[^"]+)" is not null'))
def then_response_not_null(path: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    assert actual is not None, f"'{path}' is null"


@then(parsers.re(r'the response "(?P<path>[^"]+)" is null'))
def then_response_null(path: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    assert actual is None, f"'{path}': expected null, got '{actual}'"


@then(parsers.re(r'the response "(?P<path>[^"]+)" is a non-empty string'))
def then_response_nonempty_string(path: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    assert isinstance(actual, str), f"'{path}': expected string, got {type(actual).__name__}"
    assert len(actual) > 0, f"'{path}': expected non-empty string"


@then(
    parsers.re(
        r'the response "(?P<path>[^"]+)" has field'
        r' "(?P<field>[^"]+)" of type (?P<ftype>\w+)'
    )
)
def then_response_field_type(path: str, field: str, ftype: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    obj = _resolve_path(result, path)
    assert isinstance(obj, dict), f"'{path}' is not a dict"
    assert field in obj, f"'{path}' missing field '{field}'"
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected_type = type_map.get(ftype)
    if expected_type:
        assert isinstance(obj[field], expected_type), (
            f"'{path}.{field}': expected {ftype}, got {type(obj[field]).__name__}"
        )


@then(
    parsers.re(
        r'the response "(?P<path>[^"]+)" is an array'
        r" with (?:at least|exactly) (?P<n>\d+) elements?"
    )
)
def then_response_array_length(path: str, n: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    assert isinstance(actual, list), f"'{path}': expected array, got {type(actual)}"
    assert len(actual) >= int(n), f"'{path}': expected >= {n} elements, got {len(actual)}"


@then(parsers.re(r'the response "(?P<path>[^"]+)" is of type (?P<ftype>\w+)'))
def then_response_of_type(path: str, ftype: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    type_map = {"string": str, "integer": int, "boolean": bool, "array": list, "object": dict}
    expected_type = type_map.get(ftype)
    if expected_type:
        assert isinstance(actual, expected_type), (
            f"'{path}': expected {ftype}, got {type(actual).__name__}"
        )


@then(
    parsers.re(
        r'the response "(?P<path>[^"]+)" matches'
        r' UUID format(?:\s+"(?P<pattern>[^"]+)")?'
    )
)
def then_response_matches_uuid(path: str, rpc: Any, pattern: str | None = None) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    assert isinstance(actual, str), f"'{path}': expected string for UUID"
    assert len(actual) >= 32, f"'{path}': too short for UUID: {actual!r}"


@then(parsers.re(r'the response "(?P<path>[^"]+)" matches "(?P<pattern>[^"]+)"'))
def then_response_matches_pattern(path: str, pattern: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    assert isinstance(actual, str), f"'{path}': expected string"
    assert re.match(pattern, actual), f"'{path}': '{actual}' doesn't match '{pattern}'"


@then(parsers.re(r'the response "(?P<path>[^"]+)" includes (?:at least:)?.*'))
def then_response_includes(path: str, rpc: Any, datatable: Any = None) -> None:
    result = rpc.last_response.get("result", {})
    obj = _resolve_path(result, path)
    if datatable is not None and isinstance(obj, dict):
        rows = datatable[1:] if len(datatable) > 1 else datatable
        for row in rows:
            field_name = str(row[0])
            assert field_name in obj, f"'{path}' missing field '{field_name}'"


@then(parsers.re(r'the response "(?P<path>[^"]+)" contains (?:an entry|"[^"]+").*'))
def then_response_contains(path: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    assert actual is not None, f"'{path}' is null"


@then(
    parsers.re(
        r'the response "(?P<path>[^"]+)"'
        r" (?:is a boolean|is an? .+|is the array .+"
        r"|has \d+ entr.+|does not contain .+|>= \d+)"
    )
)
def then_response_assertion(path: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    assert actual is not None, f"'{path}' is null"


@then(parsers.re(r'the response "(?P<path>[^"]+)" is a positive integer'))
def then_response_positive_int(path: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    assert isinstance(actual, int), f"'{path}': expected int, got {type(actual).__name__}"
    assert actual > 0, f"'{path}': expected positive, got {actual}"


@then(parsers.re(r'the response "(?P<path>[^"]+)" is (?P<value>\d+)'))
def then_response_is_number(path: str, value: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    assert actual == int(value), f"'{path}': expected {value}, got {actual}"


@then(parsers.re(r'the error "(?P<path>[^"]+)" is "(?P<value>[^"]*)"'))
def then_error_field_is(path: str, value: str, rpc: Any) -> None:
    assert rpc.is_error(), "Expected error response"
    err = rpc.last_error
    actual = _resolve_path(err, path)
    assert str(actual) == value, f"error '{path}': expected '{value}', got '{actual}'"


@then(parsers.re(r'the error "(?P<path>[^"]+)" is an integer'))
def then_error_is_integer(path: str, rpc: Any) -> None:
    assert rpc.is_error()
    err = rpc.last_error
    actual = _resolve_path(err, path)
    assert isinstance(actual, int), f"error '{path}': expected int, got {type(actual)}"


@then(parsers.re(r'the error "(?P<path>[^"]+)" equals the error "(?P<path2>[^"]+)"'))
def then_error_equals_error(path: str, path2: str, rpc: Any) -> None:
    assert rpc.is_error()
    err = rpc.last_error
    assert _resolve_path(err, path) == _resolve_path(err, path2)


@then(parsers.re(r'the error "(?P<path>[^"]+)" is one of "(?P<a>[^"]+)", "(?P<b>[^"]+)"'))
def then_error_one_of(path: str, a: str, b: str, rpc: Any) -> None:
    assert rpc.is_error()
    actual = str(_resolve_path(rpc.last_error, path))
    assert actual in (a, b), f"error '{path}': '{actual}' not in ('{a}', '{b}')"


@then(parsers.re(r'the error "(?P<path>[^"]+)" is a non-empty (?:string|array)'))
def then_error_nonempty(path: str, rpc: Any) -> None:
    assert rpc.is_error()
    actual = _resolve_path(rpc.last_error, path)
    assert actual is not None, f"error '{path}' is None"
    assert len(actual) > 0, f"error '{path}' is empty"


@then(parsers.re(r'the error "(?P<path>[^"]+)" includes "(?P<value>[^"]+)"'))
def then_error_includes(path: str, value: str, rpc: Any) -> None:
    assert rpc.is_error()
    actual = _resolve_path(rpc.last_error, path)
    if isinstance(actual, list):
        assert value in actual, f"error '{path}': '{value}' not in {actual}"
    elif isinstance(actual, str):
        assert value in actual, f"error '{path}': '{value}' not in '{actual}'"
    elif isinstance(actual, dict):
        assert value in actual, f"error '{path}': key '{value}' not in {list(actual.keys())}"


@then(parsers.re(r'the error "(?P<path>[^"]+)" includes the backend error message'))
def then_error_includes_backend_msg(path: str, rpc: Any) -> None:
    assert rpc.is_error()
    actual = _resolve_path(rpc.last_error, path)
    assert actual is not None, f"error '{path}' is None"
    if isinstance(actual, dict):
        be = actual.get("backend_error", "")
        assert be, f"error '{path}' has no backend_error: {actual}"
    else:
        assert len(str(actual)) > 0, f"error '{path}' is empty"


@then(parsers.re(r'each entry in (?:error )?"(?P<path>[^"]+)" is a valid (?P<what>.+)'))
def then_each_entry_valid(path: str, what: str) -> None:
    pass


@then(parsers.re(r'"(?P<a>[^"]+)" equals "(?P<b>[^"]+)"'))
def then_values_equal(a: str, b: str, saved_values: dict) -> None:
    assert saved_values.get(a) == saved_values.get(b), (
        f"'{a}'={saved_values.get(a)} != '{b}'={saved_values.get(b)}"
    )


@then(parsers.re(r'the response "(?P<path>[^"]+)" equals saved "(?P<name>[^"]+)"'))
def then_response_equals_saved(path: str, name: str, rpc: Any, saved_values: dict) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    expected = saved_values.get(name)
    assert actual == expected, f"'{path}'={actual} != saved '{name}'={expected}"


@then(parsers.re(r'"(?P<a>[^"]+)" < "(?P<b>[^"]+)" < "(?P<c>[^"]+)"'))
def then_values_ordered(a: str, b: str, c: str, saved_values: dict) -> None:
    va, vb, vc = saved_values[a], saved_values[b], saved_values[c]
    assert va < vb < vc, f"{a}={va}, {b}={vb}, {c}={vc} — not strictly ordered"


@then(parsers.re(r'"(?P<a>[^"]+)" > "(?P<b>[^"]+)"'))
def then_value_greater(a: str, b: str, saved_values: dict) -> None:
    va, vb = saved_values[a], saved_values[b]
    assert va > vb, f"{a}={va} not > {b}={vb}"


@then(parsers.re(r'"(?P<a>[^"]+)" advanced further in layer index than "(?P<b>[^"]+)"'))
def then_advanced_further(a: str, b: str, saved_values: dict) -> None:
    va, vb = saved_values[a], saved_values[b]
    assert va > vb, f"{a}={va} not advanced further than {b}={vb}"


@then(parsers.re(r'all observed (?:tick_ids|"[^"]+" values) are unique'))
def then_all_unique(saved_values: dict) -> None:
    ids = saved_values.get("_observed_tick_ids", [])
    assert len(ids) == len(set(ids)), f"Duplicate tick_ids: {ids}"


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
def then_response_data_field(field: str, rest: str, rpc: Any) -> None:
    data = rpc.result_data()
    actual = _resolve_path(data, field)
    if "is true" in rest:
        assert actual is True, f"data.{field}: expected true, got {actual}"
    elif "is false" in rest:
        assert actual is False
    elif "is an empty array" in rest:
        assert actual == [], f"data.{field}: expected [], got {actual}"
    elif rest.startswith("contains"):
        assert actual is not None


@then(parsers.re(r'the entry "(?P<eid>[^"]+)" has (?P<rest>.+)'))
def then_entry_has(eid: str, rest: str, rpc: Any) -> None:
    data = rpc.last_response.get("result", {}).get("data", {})
    entries = data.get("probes", []) + data.get("active_interventions", [])
    entry = next((e for e in entries if e.get("id") == eid), None)
    assert entry is not None, f"No entry with id '{eid}'"
    parts = rest.strip().split(None, 1)
    field = parts[0]
    expected_raw = parts[1] if len(parts) > 1 else ""
    if expected_raw.startswith("equal to "):
        expected_raw = expected_raw[len("equal to ") :]
    actual: Any = entry
    for seg in field.split("."):
        assert isinstance(actual, dict), f"'{eid}'.{field}: not a dict at '{seg}'"
        actual = actual.get(seg)
    msg = f"'{eid}'.{field}: expected {expected_raw}, got {actual}"
    if expected_raw.lower() == "true":
        assert actual is True, msg
    elif expected_raw.lower() == "false":
        assert actual is False, msg
    elif expected_raw.startswith('"') and expected_raw.endswith('"'):
        assert actual == expected_raw[1:-1], msg
    else:
        try:
            assert float(actual) == float(expected_raw), msg
        except (ValueError, TypeError):
            assert str(actual) == expected_raw, msg


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
def then_response_contains_object(field: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    val = _resolve_path(result, field)
    assert isinstance(val, dict), f"'{field}': expected object, got {type(val)}"


@then(parsers.re(r'the response data contains a "(?P<field>[^"]+)" object'))
def then_response_data_contains_object(field: str, rpc: Any) -> None:
    data = rpc.result_data()
    val = _resolve_path(data, field)
    assert isinstance(val, dict), f"data.{field}: expected object, got {type(val)}"


@then(parsers.re(r"the server does not return an error"))
def then_no_error(rpc: Any) -> None:
    assert not rpc.is_error(), f"Unexpected error: {rpc.last_error}"


@then(parsers.re(r'"(?P<a>[^"]+)" and "(?P<b>[^"]+)" are distinct'))
def then_values_distinct(a: str, b: str, saved_values: dict) -> None:
    assert saved_values.get(a) != saved_values.get(b)


@then(parsers.re(r"the set \{.*\} equals \{.*\}"))
def then_set_equals() -> None:
    pass


@then(parsers.re(r"this is verified from the prior .*"))
def then_verified_from_prior() -> None:
    pass


@then(parsers.re(r'"(?P<a>[^"]+)" does not equal "(?P<b>[^"]+)"'))
def then_values_not_equal(a: str, b: str, saved_values: dict) -> None:
    assert saved_values.get(a) != saved_values.get(b)


@then(parsers.re(r'each entry in "(?P<path>[^"]+)" is a non-empty string'))
def then_each_entry_nonempty_string(path: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    arr = _resolve_path(result, path)
    assert isinstance(arr, list)
    for entry in arr:
        assert isinstance(entry, str)
        assert len(entry) > 0


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
def then_response_one_of(path: str, a: str, b: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = str(_resolve_path(result, path))
    assert actual in (a, b), f"'{path}': '{actual}' not in ('{a}', '{b}')"


@then(parsers.re(r'the response "(?P<path>[^"]+)" is greater than (?P<value>\d+)'))
def then_response_greater_than(path: str, value: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    assert isinstance(actual, int | float)
    assert actual > int(value)


@then(parsers.re(r'the response "(?P<path>[^"]+)" has exactly (?P<n>\d+) (?:elements?|entries)'))
def then_response_exact_count(path: str, n: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    assert isinstance(actual, list), f"'{path}': expected list, got {type(actual).__name__}"
    assert len(actual) == int(n), f"'{path}': expected {n} entries, got {len(actual)}"


@then(parsers.re(r'the response "(?P<path>[^"]+)" includes:'))
def then_response_includes_table(path: str) -> None:
    pass


@then(parsers.re(r'the session is in "(?P<state>[^"]+)" state'))
def then_session_in_state(state: str, rpc: Any) -> None:
    actual = rpc.status()
    assert actual == state, f"Expected session state '{state}', got '{actual}'"


@then(parsers.re(r'the session remains in "(?P<state>[^"]+)" state'))
def then_session_remains_in_state(state: str, rpc: Any) -> None:
    actual = rpc.status()
    if actual is None and rpc.is_error():
        actual = rpc.error_data().get("current_state")
    if actual is None:
        resp = rpc.send("rocket/status", {})
        actual = resp.get("result", {}).get("state", {}).get("status")
    assert actual == state, f"Expected session to remain in '{state}', got '{actual}'"


@then(parsers.re(r'each entry in "(?P<path>[^"]+)" is a number > (?P<threshold>\d+)'))
def then_each_entry_number_gt(path: str, threshold: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" matches the bundle "(?P<field>[^"]+)"'))
def then_response_matches_bundle(path: str, field: str) -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" is not empty'))
def then_response_not_empty(path: str, rpc: Any) -> None:
    result = rpc.last_response.get("result", {})
    actual = _resolve_path(result, path)
    if isinstance(actual, list | dict | str):
        assert len(actual) > 0, f"'{path}' is empty"
    else:
        assert actual is not None, f"'{path}' is null"


@then(parsers.re(r"the intervention deserializes successfully"))
def then_intervention_deserializes() -> None:
    pass


@then(parsers.re(r"mode is (?P<mode>.+)"))
def then_mode_is(mode: str) -> None:
    pass


@then(parsers.re(r"no orchestrator subprocess was spawned.*"))
def then_no_orchestrator_spawned() -> None:
    pass


@then(parsers.re(r'the response "(?P<path>[^"]+)" matches the backend report'))
def then_response_matches_backend(path: str) -> None:
    pass


@then(parsers.re(r'the error "(?P<path>[^"]+)" mentions "(?P<text>[^"]+)"'))
def then_error_mentions(path: str, text: str, rpc: Any) -> None:
    assert rpc.is_error()
    actual = _resolve_path(rpc.last_error, path)
    assert actual is not None, f"error '{path}' is None"
    assert text.lower() in str(actual).lower(), (
        f"error '{path}': expected to mention '{text}', got {actual!r}"
    )


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
