"""Keep runner-module imports local to the sharding infrastructure tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _isolate_child_pytest_addopts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nightly package tests export PYTEST_ADDOPTS="--full", but the child pytest
    # processes spawned here run isolated suites that do not load the FlashInfer
    # plugin registering ``--full``. Tests needing specific options set them.
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
