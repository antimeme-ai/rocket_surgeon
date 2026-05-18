"""Shared pytest fixtures and configuration for rocket_surgeon tests.

Ensures the tiny test model is cached locally before any test runs.
After first download, tests run fully offline.
"""

from __future__ import annotations

import os

import pytest
from transformers import AutoModelForCausalLM

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def pytest_configure(config: pytest.Config) -> None:
    """Pre-cache the tiny model so tests never need the network after first run."""
    try:
        AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    except Exception:
        pytest.exit(
            f"Failed to download test model {TINY_MODEL}. "
            "Run once with network access to cache it.",
            returncode=1,
        )

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
