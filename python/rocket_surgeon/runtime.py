"""Worker-process runtime fixes applied at ``rs-worker`` startup.

The worker embeds CPython via PyO3. An embedded interpreter reports the
*host binary* (``rs-worker``) as :data:`sys.executable`. Any library that
re-launches the interpreter — :mod:`multiprocessing` with the ``spawn`` start
method, ``torch`` inductor compile workers, ``torch.distributed`` — would then
exec ``rs-worker -c ...``, which the worker's CLI rejects. This module
repoints those launch paths at a real Python interpreter.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def _looks_like_python(name: str) -> bool:
    """Return whether *name* is the basename of a Python interpreter."""
    return name.lower().startswith("python")


def find_real_interpreter() -> str | None:
    """Return the path to a real Python interpreter for the running process.

    The embedded interpreter's prefixes still point at a genuine CPython
    install whose ``bin/`` holds an interpreter ABI-identical to the worker.
    Spawned subprocesses inherit ``PYTHONPATH``, so that interpreter resolves
    the same packages (torch, ``rocket_surgeon``) the worker itself uses.

    Returns ``None`` if no interpreter is found — callers treat that as
    best-effort and continue.
    """
    minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for prefix in (sys.base_prefix, sys.prefix, sys.exec_prefix):
        for name in (minor, "python3", "python"):
            candidate = Path(prefix) / "bin" / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def align_subprocess_interpreter() -> str | None:
    """Repoint subprocess and multiprocessing launches at a real interpreter.

    Idempotent and best-effort. Returns the chosen interpreter path, or
    ``None`` when :data:`sys.executable` already names a Python interpreter
    (the process is not embedded) or no real interpreter could be found.
    """
    if _looks_like_python(Path(sys.executable).name):
        return None
    real = find_real_interpreter()
    if real is None:
        return None
    sys.executable = real
    # `_base_executable` backs venv creation and some subprocess launch paths.
    # setattr (not direct assignment) keeps this clean of typeshed/lint noise
    # for an attribute the `sys` stubs do not declare.
    setattr(sys, "_base_executable", real)  # noqa: B010
    multiprocessing.set_executable(real)
    return real
