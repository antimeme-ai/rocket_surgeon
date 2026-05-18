"""Lock-based single-slot mailbox for barrier synchronization.

Uses _thread.allocate_lock() — a thin C wrapper around a pthread mutex.
No Python-level bookkeeping, no flag-based race conditions.

Pattern mirrors nnsight's Mediator.Value from
src/nnsight/intervention/interleaver.py.
"""

from __future__ import annotations

from _thread import allocate_lock
from typing import Any


class Mailbox:
    """Single-slot mailbox: one producer, one consumer.

    - put(value): store value, release lock (wakes consumer)
    - wait() -> value: acquire lock (blocks until put), return value
    - get() -> value: non-blocking read of current value
    - restore(): clear value, drop references
    """

    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._lock = allocate_lock()
        self._lock.acquire()
        self._value: Any = None

    def put(self, value: Any) -> None:
        """Store value and release the lock, waking any blocked consumer."""
        self._value = value
        if self._lock.locked():
            self._lock.release()

    def wait(self) -> Any:
        """Block until a value is put, then return it."""
        self._lock.acquire()
        return self._value

    def get(self) -> Any:
        """Non-blocking read of the current stored value (or None)."""
        return self._value

    def restore(self) -> None:
        """Clear the stored value and drop references."""
        self._value = None
