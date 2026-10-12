"""Throwaway test file: every test is unconditionally skipped.

Used to validate skip-all detection in CI. Delete after verification.
"""
import pytest

pytestmark = pytest.mark.skipif(True, reason="Dummy skip-all for CI validation (IKL-569)")


def test_dummy_alpha():
    assert True


def test_dummy_beta():
    assert True


def test_dummy_gamma():
    assert True
