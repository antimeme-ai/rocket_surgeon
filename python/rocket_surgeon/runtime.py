"""Worker-process runtime fixes applied at ``rs-worker`` startup.

The worker embeds CPython via PyO3. An embedded interpreter reports the
*host binary* (``rs-worker``) as :data:`sys.executable`. Any library that
re-launches the interpreter — :mod:`multiprocessing` with the ``spawn`` start
method, ``torch`` inductor compile workers, ``torch.distributed`` — would then
exec ``rs-worker -c ...``, which the worker's CLI rejects. This module
repoints those launch paths at a real Python interpreter.

POSIX layout (``<prefix>/bin/python*``) is assumed — consistent with the rest
of the project, which is macOS/Linux only.
"""

from __future__ import annotations

import multiprocessing
import os
import re
import sys
from pathlib import Path

# Basename of a Python interpreter: `python`, `python3`, `python3.11`, ...
_PYTHON_EXE = re.compile(r"^python(\d+(\.\d+)?)?$", re.IGNORECASE)

# A venv's site-packages sits this many directories below the venv root:
# <venv>/lib/pythonX.Y/site-packages.
_VENV_ROOT_DEPTH = 3


def _looks_like_python(name: str) -> bool:
    """Return whether *name* is the basename of a Python interpreter."""
    return _PYTHON_EXE.match(name) is not None


def _interpreter_in(prefix: Path) -> str | None:
    """Return the path to a Python interpreter under ``prefix/bin``, or None."""
    minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for name in (minor, "python3", "python"):
        candidate = prefix / "bin" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _find_venv_interpreter() -> str | None:
    """Locate a virtualenv interpreter from a ``site-packages`` entry on sys.path.

    A virtualenv is marked by a ``pyvenv.cfg`` at its root, with its interpreter
    in ``bin/``. This is the most robust target: the venv's own ``site-packages``
    carries the worker's dependencies, so a spawned child resolves them even
    without inheriting ``PYTHONPATH``. Returns ``None`` if no venv is on the path.
    """
    for entry in sys.path:
        if not entry:
            continue
        site_packages = Path(entry)
        if site_packages.name != "site-packages" or len(site_packages.parents) < _VENV_ROOT_DEPTH:
            continue
        venv_root = site_packages.parents[_VENV_ROOT_DEPTH - 1]
        if not (venv_root / "pyvenv.cfg").is_file():
            continue
        interpreter = _interpreter_in(venv_root)
        if interpreter is not None:
            return interpreter
    return None


def find_real_interpreter() -> str | None:
    """Return a real Python interpreter the worker can re-launch as a subprocess.

    Prefers a virtualenv interpreter discovered on :data:`sys.path` — its own
    ``site-packages`` carries the worker's dependencies. Falls back to the base
    CPython install backing the embedded interpreter; spawned children inherit
    the worker's ``PYTHONPATH``, which is what makes the worker's packages
    importable there.

    Returns ``None`` if no interpreter is found — callers treat that as
    best-effort and continue.
    """
    venv = _find_venv_interpreter()
    if venv is not None:
        return venv
    for prefix in (sys.base_prefix, sys.prefix, sys.exec_prefix):
        interpreter = _interpreter_in(Path(prefix))
        if interpreter is not None:
            return interpreter
    return None


def align_subprocess_interpreter() -> str | None:
    """Repoint subprocess and multiprocessing launches at a real interpreter.

    Call once, on the main thread, at worker startup — before any
    subprocess-spawning library runs. Idempotent and best-effort: returns the
    chosen interpreter path, or ``None`` when :data:`sys.executable` already
    names a Python interpreter (the process is not embedded) or no real
    interpreter could be found.
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
