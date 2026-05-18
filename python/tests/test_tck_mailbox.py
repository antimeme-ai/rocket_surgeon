"""Step definitions for tck/model/mailbox_barrier.feature."""

from __future__ import annotations

import threading

from pytest_bdd import given, parsers, scenario, then, when

from rocket_surgeon.hooks.mailbox import Mailbox

FEATURE = "../../tck/model/mailbox_barrier.feature"


# ── Scenarios ──────────────────────────────────────────────────────


@scenario(FEATURE, "Put then wait delivers the value")
def test_put_then_wait():
    pass


@scenario(FEATURE, "Restore makes the mailbox reusable")
def test_restore_reusable():
    pass


@scenario(FEATURE, "Get retrieves value without blocking")
def test_get_nonblocking():
    pass


@scenario(FEATURE, "Two-mailbox barrier cycle completes")
def test_two_mailbox_barrier():
    pass


@scenario(FEATURE, "Multiple barrier rounds succeed")
def test_multiple_rounds():
    pass


# ── Steps ──────────────────────────────────────────────────────────


@given("a fresh mailbox", target_fixture="mailbox")
def fresh_mailbox():
    return Mailbox()


@given("a result mailbox and a resume mailbox", target_fixture="barrier_pair")
def barrier_pair():
    return {"result": Mailbox(), "resume": Mailbox()}


@when("a value is put into the mailbox")
def put_value(mailbox):
    mailbox.put("test-value")


@when("the value is consumed via wait")
def consume_wait(mailbox):
    mailbox.wait()


@when("the mailbox is restored")
def restore_mailbox(mailbox):
    mailbox.restore()


@then("wait returns that value")
def wait_returns_value(mailbox):
    assert mailbox.wait() == "test-value"
    mailbox.restore()


@then("the mailbox can accept a new value")
def accept_new_value(mailbox):
    mailbox.put("second-value")
    assert mailbox.wait() == "second-value"
    mailbox.restore()


@then("get returns that value without blocking")
def get_returns_value(mailbox):
    assert mailbox.get() == "test-value"
    mailbox.restore()


@when(
    "a producer puts a value on the result mailbox and waits on the resume mailbox",
    target_fixture="producer_thread",
)
def producer_puts_and_waits(barrier_pair):
    result_mb = barrier_pair["result"]
    resume_mb = barrier_pair["resume"]
    unblocked = threading.Event()

    def producer():
        result_mb.put("produced-value")
        resume_mb.wait()
        resume_mb.restore()
        unblocked.set()

    t = threading.Thread(target=producer)
    t.start()
    barrier_pair["_thread"] = t
    barrier_pair["_unblocked"] = unblocked
    return t


@when("the consumer waits on the result mailbox")
def consumer_waits(barrier_pair):
    val = barrier_pair["result"].wait()
    barrier_pair["_received"] = val


@then("the consumer receives the produced value")
def consumer_received(barrier_pair):
    assert barrier_pair["_received"] == "produced-value"


@when("the consumer restores the result mailbox and puts a signal on the resume mailbox")
def consumer_restores_and_signals(barrier_pair):
    barrier_pair["result"].restore()
    barrier_pair["resume"].put(None)


@then("the producer is unblocked")
def producer_unblocked(barrier_pair):
    barrier_pair["_unblocked"].wait(timeout=5.0)
    assert barrier_pair["_unblocked"].is_set()
    barrier_pair["_thread"].join(timeout=5.0)


@when(parsers.parse("{n:d} barrier rounds are executed"))
def execute_barrier_rounds(barrier_pair, n):
    result_mb = barrier_pair["result"]
    resume_mb = barrier_pair["resume"]
    completed = []

    for i in range(n):
        done = threading.Event()

        def producer(val=i, evt=done):
            result_mb.put(val)
            resume_mb.wait()
            resume_mb.restore()
            evt.set()

        t = threading.Thread(target=producer)
        t.start()

        received = result_mb.wait()
        result_mb.restore()
        resume_mb.put(None)

        done.wait(timeout=5.0)
        t.join(timeout=5.0)
        completed.append(received)

    barrier_pair["_completed"] = completed
    barrier_pair["_n"] = n


@then(parsers.parse("all {n:d} rounds complete successfully"))
def all_rounds_complete(barrier_pair, n):
    assert len(barrier_pair["_completed"]) == n
    assert barrier_pair["_completed"] == list(range(n))
