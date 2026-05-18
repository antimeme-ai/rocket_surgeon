"""Tests for lock-based single-slot mailbox."""

from __future__ import annotations

import threading
import time

from rocket_surgeon.hooks.mailbox import Mailbox


def test_put_then_wait_returns_value() -> None:
    m = Mailbox()
    m.put("hello")
    assert m.wait() == "hello"


def test_wait_blocks_until_put() -> None:
    m = Mailbox()

    def producer() -> None:
        time.sleep(0.05)
        m.put("from-producer")

    t = threading.Thread(target=producer)
    t.start()
    value = m.wait()
    t.join()
    assert value == "from-producer"


def test_get_returns_stored_value_without_blocking() -> None:
    m = Mailbox()
    m.put(42)
    m.wait()
    assert m.get() == 42


def test_get_returns_none_when_empty() -> None:
    m = Mailbox()
    assert m.get() is None


def test_restore_clears_value() -> None:
    m = Mailbox()
    m.put("data")
    m.wait()
    m.restore()
    assert m.get() is None


def test_ping_pong_two_mailboxes() -> None:
    """Simulate the barrier pattern: forward thread sends result, waits for resume."""
    result_mb = Mailbox()
    resume_mb = Mailbox()
    captured: list[tuple[str, str]] = []

    def forward_thread() -> None:
        result_mb.put("tensor_at_layer_3")
        value = resume_mb.wait()
        resume_mb.restore()
        captured.append(("forward_got", value))

    def rust_thread() -> None:
        value = result_mb.wait()
        result_mb.restore()
        captured.append(("rust_got", value))
        resume_mb.put("continue")

    fwd = threading.Thread(target=forward_thread)
    rust = threading.Thread(target=rust_thread)
    fwd.start()
    rust.start()
    fwd.join(timeout=2.0)
    rust.join(timeout=2.0)

    assert ("rust_got", "tensor_at_layer_3") in captured
    assert ("forward_got", "continue") in captured


def test_multiple_rounds() -> None:
    """Multiple put/wait/restore cycles work correctly."""
    m = Mailbox()
    for i in range(10):
        m.put(i)
        assert m.wait() == i
        m.restore()
        assert m.get() is None


def test_put_overwrites_unconsumed_value() -> None:
    """Second put before wait overwrites the slot."""
    m = Mailbox()
    m.put("first")
    m.put("second")
    assert m.wait() == "second"
