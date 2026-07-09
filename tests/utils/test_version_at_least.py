import pytest
import flashinfer.utils


@pytest.fixture(autouse=True)
def clear_version_cache():
    # Clear the lru_cache before each test to ensure isolation
    flashinfer.utils.version_at_least.cache_clear()
    yield
    flashinfer.utils.version_at_least.cache_clear()


def test_version_at_least():
    assert flashinfer.utils.version_at_least("12.3", "12.1") is True
    assert flashinfer.utils.version_at_least("12.1", "12.3") is False
    assert flashinfer.utils.version_at_least("12.1", "12.1") is True
    assert flashinfer.utils.version_at_least("13.0", "12.9") is True


def test_version_at_least_none():
    """Verify that version_at_least handles None inputs gracefully,
    returning False instead of raising TypeError, which is crucial
    for CPU-only setups where torch.version.cuda is None.
    """
    assert flashinfer.utils.version_at_least(None, "12.1") is False
    assert flashinfer.utils.version_at_least("12.3", None) is False
    assert flashinfer.utils.version_at_least(None, None) is False


def test_version_at_least_caching():
    # Cache should be clear initially
    info_before = flashinfer.utils.version_at_least.cache_info()
    assert info_before.hits == 0
    assert info_before.misses == 0

    # First call: miss
    assert flashinfer.utils.version_at_least("12.3", "12.1") is True
    info_1 = flashinfer.utils.version_at_least.cache_info()
    assert info_1.hits == 0
    assert info_1.misses == 1

    # Second call (same args): hit!
    assert flashinfer.utils.version_at_least("12.3", "12.1") is True
    info_2 = flashinfer.utils.version_at_least.cache_info()
    assert info_2.hits == 1
    assert info_2.misses == 1
