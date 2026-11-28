"""Tests for the mock experimental feature (a). CPU-only."""

import pytest

from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.mock_verify_a import mock_scale_a


def test_mock_scale_warns_on_first_use():
    with pytest.warns(ExperimentalWarning, match="mock_scale_a"):
        assert mock_scale_a(3) == 6


def test_mock_scale_honors_factor():
    assert mock_scale_a(4, factor=3) == 12


def test_mock_scale_is_marked_experimental():
    assert mock_scale_a.is_experimental is True
