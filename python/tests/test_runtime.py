"""Tests for rocket_surgeon.runtime — subprocess interpreter alignment.

The worker embeds CPython, so :data:`sys.executable` names the host binary
rather than a real interpreter. These tests cover the repair that repoints
subprocess/multiprocessing launches back at a genuine interpreter.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.spawn
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from rocket_surgeon.runtime import align_subprocess_interpreter, find_real_interpreter

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def restore_interpreter_state() -> Iterator[None]:
    """Save and restore the global interpreter-launch state a test mutates."""
    saved_executable = sys.executable
    saved_base = getattr(sys, "_base_executable", None)
    saved_mp = multiprocessing.spawn.get_executable()
    yield
    sys.executable = saved_executable
    if saved_base is not None:
        sys._base_executable = saved_base
    multiprocessing.set_executable(saved_mp)


def test_find_real_interpreter_returns_existing_python() -> None:
    """A real, executable Python interpreter is discoverable from sys prefixes."""
    found = find_real_interpreter()
    assert found is not None
    path = Path(found)
    assert path.is_file()
    assert path.name.lower().startswith("python")


def test_find_real_interpreter_result_actually_runs() -> None:
    """The chosen interpreter executes and reports the running Python version.

    This is the property the worker depends on: the repaired `sys.executable`
    must be a launchable interpreter, not just an existing file.
    """
    found = find_real_interpreter()
    assert found is not None
    result = subprocess.run(
        [found, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True,
        check=True,
    )
    expected = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert result.stdout.strip() == expected


def test_find_real_interpreter_prefers_venv_when_on_path() -> None:
    """Under the project venv, the discovered interpreter lives in that venv."""
    found = find_real_interpreter()
    assert found is not None
    # A venv interpreter sits at <venv>/bin/python*; <venv> has a pyvenv.cfg.
    venv_root = Path(found).parent.parent
    assert (venv_root / "pyvenv.cfg").is_file()


def test_find_real_interpreter_returns_none_when_no_interpreter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With no venv on the path and empty prefixes, discovery yields None."""
    monkeypatch.setattr(sys, "path", [])
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path))
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(sys, "exec_prefix", str(tmp_path))
    assert find_real_interpreter() is None


def test_align_is_noop_when_executable_is_already_python(
    restore_interpreter_state: None,
) -> None:
    """The test process runs under a real interpreter, so alignment skips."""
    assert align_subprocess_interpreter() is None
    assert Path(sys.executable).name.lower().startswith("python")


def test_align_repairs_embedded_host_binary(restore_interpreter_state: None) -> None:
    """When sys.executable is a non-Python host binary, all launch paths repair."""
    sys.executable = "/opt/rocket-surgeon/bin/rs-worker"

    chosen = align_subprocess_interpreter()

    assert chosen is not None
    assert Path(chosen).name.lower().startswith("python")
    assert sys.executable == chosen
    assert getattr(sys, "_base_executable", None) == chosen
    # multiprocessing fsencodes the executable on POSIX; normalise before compare.
    assert os.fsdecode(multiprocessing.spawn.get_executable()) == chosen


def test_align_is_idempotent(restore_interpreter_state: None) -> None:
    """A second call after repair is a no-op (executable already Python)."""
    sys.executable = "/opt/rocket-surgeon/bin/rs-worker"
    first = align_subprocess_interpreter()
    second = align_subprocess_interpreter()
    assert first is not None
    assert second is None
