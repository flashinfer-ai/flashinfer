# Copyright (c) 2025 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Negative control: a normal test file that should NOT trigger skip-all warnings."""

import pytest


def test_always_passes():
    """This test always executes — no skip."""
    assert 1 + 1 == 2


def test_also_passes():
    """Another unconditionally passing test."""
    assert True


@pytest.mark.skipif(False, reason="this skip condition is never true")
def test_conditional_that_runs():
    """A conditional skip whose condition is False, so the test runs."""
    assert 42 == 42
