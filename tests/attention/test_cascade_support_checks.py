import pytest

import flashinfer


_CASCADE_MERGE_APIS = (
    flashinfer.merge_state,
    flashinfer.merge_state_in_place,
    flashinfer.merge_states,
)


@pytest.mark.parametrize(
    "api",
    _CASCADE_MERGE_APIS,
    ids=lambda api: api.__name__,
)
def test_cascade_merge_support_checks(api):
    assert hasattr(api, "is_compute_capability_supported")
    assert hasattr(api, "has_backend_choices")

    assert api.is_compute_capability_supported(75)
    assert api.is_compute_capability_supported(86)
    assert api.is_compute_capability_supported(107)
    assert api.is_compute_capability_supported(121)
    assert not api.is_compute_capability_supported(70)

    assert not api.has_backend_choices()
