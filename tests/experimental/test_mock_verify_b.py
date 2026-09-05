"""Tests for the mock experimental feature (b). CPU-only."""

import pytest

from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.mock_verify_b import mock_scale_b


def test_mock_scale_warns_on_first_use():
    with pytest.warns(ExperimentalWarning, match="mock_scale_b"):
        assert mock_scale_b(3) == 6


def test_mock_scale_honors_factor():
    assert mock_scale_b(4, factor=3) == 12


def test_mock_scale_is_marked_experimental():
    assert mock_scale_b.is_experimental is True
