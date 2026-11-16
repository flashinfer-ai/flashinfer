"""Keep runner-module imports local to the sharding infrastructure tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _nightly_pytest_addopts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nightly package tests export PYTEST_ADDOPTS="--full". Simulate it in every
    # run so a child pytest that inherits the variable but does not load a
    # plugin registering ``--full`` fails in regular CI instead of only nightly.
    monkeypatch.setenv("PYTEST_ADDOPTS", "--full")
