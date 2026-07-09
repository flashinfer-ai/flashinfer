import pytest
import flashinfer.utils

def test_version_at_least():
    assert flashinfer.utils.version_at_least("12.3", "12.1") is True
    assert flashinfer.utils.version_at_least("12.1", "12.3") is False
    assert flashinfer.utils.version_at_least("12.1", "12.1") is True
    assert flashinfer.utils.version_at_least("13.0", "12.9") is True
