"""TCK: session / bundle."""

from __future__ import annotations

import pytest
from pytest_bdd import scenarios

pytestmark = pytest.mark.xfail(reason="stub: no server implementation yet", strict=False)

scenarios("../../../tck/session/bundle.feature")
