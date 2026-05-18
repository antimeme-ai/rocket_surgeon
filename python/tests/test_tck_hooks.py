"""Step definitions for tck/model/hook_lifecycle.feature."""

from __future__ import annotations

import contextlib
import threading

import torch
from pytest_bdd import given, scenario, then, when

from rocket_surgeon.bridge import (
    _models,
    discover_modules,
    install_capture_hooks,
    install_sentinel_hooks,
    load_model,
    remove_hooks,
    run_forward,
    unload_model,
)
from rocket_surgeon.hooks.mailbox import Mailbox

FEATURE = "../../tck/model/hook_lifecycle.feature"
TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# ── Scenarios ──────────────────────────────────────────────────────


@scenario(FEATURE, "Sentinel hooks installed on all modules return handles")
def test_sentinel_hooks():
    pass


@scenario(FEATURE, "Capture hook delivers path, call_index, and tensor on forward pass")
def test_capture_barrier():
    pass


@scenario(FEATURE, "Capture hook with no active probes does not block")
def test_capture_no_active():
    pass


@scenario(FEATURE, "Removing hooks allows clean forward pass")
def test_remove_hooks():
    pass


@scenario(FEATURE, "run_forward calls done_callback with None on success")
def test_run_forward_success():
    pass


@scenario(FEATURE, "run_forward calls done_callback with exception on bad input")
def test_run_forward_error():
    pass


# ── Background ─────────────────────────────────────────────────────


@given("a tiny llama model is loaded on CPU", target_fixture="ctx")
def tiny_model():
    handle = load_model(source=TINY_MODEL, device="cpu", dtype="float32")
    ctx = {"handle": handle, "hooks": [], "errors": []}
    yield ctx
    for h_list in [ctx.get("sentinel_handles", []), ctx.get("capture_handles", [])]:
        with contextlib.suppress(Exception):
            remove_hooks(h_list)
    unload_model(handle)


@given("the model's module paths are discovered")
def discover_paths(ctx):
    modules = discover_modules(ctx["handle"])
    ctx["module_paths"] = [m["path"] for m in modules]


@given("a result mailbox and a resume mailbox")
def mailbox_pair(ctx):
    ctx["result_mb"] = Mailbox()
    ctx["resume_mb"] = Mailbox()


@given("sentinel hooks are installed on all module paths")
def install_sentinels(ctx):
    if "module_paths" not in ctx:
        modules = discover_modules(ctx["handle"])
        ctx["module_paths"] = [m["path"] for m in modules]
    handles = install_sentinel_hooks(ctx["handle"], ctx["module_paths"])
    ctx["sentinel_handles"] = handles


@given('a capture hook is installed on "model.layers.0.self_attn.q_proj"')
def install_capture_q_proj(ctx):
    target = "model.layers.0.self_attn.q_proj"
    handles, _call_counts = install_capture_hooks(
        ctx["handle"],
        [target],
        ctx["result_mb"],
        ctx["resume_mb"],
        active_probes={target},
    )
    ctx["capture_handles"] = handles
    ctx["capture_target"] = target


@given('a capture hook is installed on "model.layers.0.self_attn.q_proj" with no active probes')
def install_capture_no_probes(ctx):
    target = "model.layers.0.self_attn.q_proj"
    handles, _call_counts = install_capture_hooks(
        ctx["handle"],
        [target],
        ctx["result_mb"],
        ctx["resume_mb"],
        active_probes=set(),
    )
    ctx["capture_handles"] = handles


# ── When steps ─────────────────────────────────────────────────────


@when("sentinel hooks are installed on all module paths")
def when_install_sentinels(ctx):
    handles = install_sentinel_hooks(ctx["handle"], ctx["module_paths"])
    ctx["sentinel_handles"] = handles


@when("a forward pass is started in a background thread")
def start_forward_thread(ctx):
    errors = ctx["errors"]

    def forward_thread():
        try:
            model = _models[ctx["handle"]]
            with torch.inference_mode():
                dummy = torch.zeros(1, 2, dtype=torch.long)
                model(dummy)
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=forward_thread)
    t.start()
    ctx["forward_thread"] = t


@when("the result mailbox is waited on")
def wait_result(ctx):
    value = ctx["result_mb"].wait()
    ctx["captured_value"] = value


@when("the result mailbox is restored and the resume mailbox signals continue")
def restore_and_signal(ctx):
    ctx["result_mb"].restore()
    ctx["resume_mb"].put(None)


@when("all hooks are removed")
def remove_all_hooks(ctx):
    remove_hooks(ctx.get("sentinel_handles", []))
    remove_hooks(ctx.get("capture_handles", []))
    ctx["sentinel_handles"] = []
    ctx["capture_handles"] = []


@when("a forward pass runs to completion")
def forward_to_completion(ctx):
    done_event = threading.Event()
    error_ref = [None]

    def callback(error):
        error_ref[0] = error
        done_event.set()

    input_ids = torch.zeros(1, 2, dtype=torch.long)
    run_forward(ctx["handle"], input_ids, callback)
    done_event.wait(timeout=10.0)
    ctx["forward_done"] = done_event.is_set()
    ctx["forward_error"] = error_ref[0]


@when("run_forward is called with valid input")
def run_forward_valid(ctx):
    done_event = threading.Event()
    error_ref = [None]

    def callback(error):
        error_ref[0] = error
        done_event.set()

    input_ids = torch.zeros(1, 2, dtype=torch.long)
    run_forward(ctx["handle"], input_ids, callback)
    done_event.wait(timeout=10.0)
    ctx["callback_error"] = error_ref[0]
    ctx["callback_called"] = done_event.is_set()


@when("run_forward is called with invalid input")
def run_forward_invalid(ctx):
    done_event = threading.Event()
    error_ref = [None]

    def callback(error):
        error_ref[0] = error
        done_event.set()

    bad_input = torch.zeros(0, dtype=torch.long)
    run_forward(ctx["handle"], bad_input, callback)
    done_event.wait(timeout=10.0)
    ctx["callback_error"] = error_ref[0]
    ctx["callback_called"] = done_event.is_set()


# ── Then steps ─────────────────────────────────────────────────────


@then("a handle is returned for each module path")
def handle_per_module(ctx):
    assert len(ctx["sentinel_handles"]) == len(ctx["module_paths"])


@then("the handles can be removed without error")
def remove_without_error(ctx):
    remove_hooks(ctx["sentinel_handles"])
    ctx["sentinel_handles"] = []


@then('the captured value contains the module path "model.layers.0.self_attn.q_proj"')
def captured_path(ctx):
    path, _, _ = ctx["captured_value"]
    assert path == "model.layers.0.self_attn.q_proj"


@then("the captured value contains a non-negative call_index")
def captured_call_index(ctx):
    _, call_index, _ = ctx["captured_value"]
    assert isinstance(call_index, int)
    assert call_index >= 0


@then("the captured value contains a torch.Tensor")
def captured_tensor(ctx):
    _, _, tensor = ctx["captured_value"]
    assert isinstance(tensor, torch.Tensor)


@then("the forward pass completes without error")
def forward_complete(ctx):
    if "forward_thread" in ctx:
        ctx["forward_thread"].join(timeout=10.0)
        assert not ctx["forward_thread"].is_alive()
    assert len(ctx["errors"]) == 0
    if "forward_error" in ctx:
        assert ctx["forward_error"] is None


@then("the result mailbox was never written to")
def mailbox_empty(ctx):
    assert ctx["result_mb"].get() is None


@then("the done callback receives None (no error)")
def callback_no_error(ctx):
    assert ctx["callback_called"]
    assert ctx["callback_error"] is None


@then("the done callback receives an exception")
def callback_has_error(ctx):
    assert ctx["callback_called"]
    assert ctx["callback_error"] is not None
