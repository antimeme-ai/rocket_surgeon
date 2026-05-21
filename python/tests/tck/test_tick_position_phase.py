"""TCK: protocol / TickPosition phase and token_position."""

from __future__ import annotations

import pytest
from pytest_bdd import scenarios

pytestmark = pytest.mark.xfail(reason="stub: no server implementation yet", strict=False)

scenarios("../../../tck/protocol/tick-position-phase.feature")
