import pytest

from flashinfer.tllm_enums import ActivationType, is_gated_activation


@pytest.mark.cpu_only
def test_geglu_tanh_is_gated_activation():
    assert is_gated_activation(ActivationType.GegluTanh)
    assert is_gated_activation(ActivationType.GegluTanh.value)
    assert ActivationType.GegluTanh.is_gated
