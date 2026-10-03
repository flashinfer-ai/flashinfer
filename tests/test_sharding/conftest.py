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
    # Nightly package tests export PYTEST_ADDOPTS="--full", but the isolated
    # child pytest processes spawned here do not load the FlashInfer plugin
    # that registers ``--full``. Always simulate it so a child that inherits
    # the variable fails in every CI run instead of only in nightly.
    monkeypatch.setenv("PYTEST_ADDOPTS", "--full")


@pytest.fixture
def isolated_pytest_addopts(
    _nightly_pytest_addopts: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clear PYTEST_ADDOPTS for tests whose runner spawns pytest in-process."""
    monkeypatch.delenv("PYTEST_ADDOPTS")
