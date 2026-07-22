from flashinfer.tllm_enums import ActivationType, is_gated_activation


def test_geglu_tanh_is_gated_activation():
    assert is_gated_activation(ActivationType.GegluTanh)
    assert is_gated_activation(ActivationType.GegluTanh.value)
    assert ActivationType.GegluTanh.is_gated


def test_situ_activation_abi_and_gated_classification():
    assert ActivationType.Situ.value == 10
    assert ActivationType.InvalidType.value == 11
    assert is_gated_activation(ActivationType.Situ)
    assert is_gated_activation(ActivationType.Situ.value)
    assert ActivationType.Situ.is_gated
