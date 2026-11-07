"""Host-only tests of the Cake SM120 NVFP4 (W4A4) weight helpers (no GPU)."""

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm120.nvfp4_nvfp4_bf16_cake import (
    mxfp4_scales_to_nvfp4,
    mxfp4_scales_to_nvfp4_shifted,
)


def _e4m3_value(byte: torch.Tensor) -> torch.Tensor:
    return byte.view(torch.float8_e4m3fn).float()


def _ue8m0_value(byte: torch.Tensor) -> torch.Tensor:
    return torch.ldexp(torch.ones(byte.shape), byte.to(torch.int32) - 127)


def test_shifted_equals_plain_when_in_range() -> None:
    exps = torch.randint(-8, 4, (3, 5, 4), dtype=torch.int32)
    sf = (exps + 127).to(torch.uint8)
    plain = mxfp4_scales_to_nvfp4(sf)
    shifted, alpha = mxfp4_scales_to_nvfp4_shifted(sf)
    assert torch.equal(plain, shifted)
    assert torch.equal(alpha, torch.ones(3))


@pytest.mark.parametrize(
    "lo,hi", [(-16, 0), (-17, 0), (-9, 8), (-12, 5), (-1, 16), (-16, 1)]
)
def test_shifted_is_bitwise_lossless(lo: int, hi: int) -> None:
    g = torch.Generator().manual_seed(lo * 31 + hi)
    exps = torch.randint(lo, hi + 1, (4, 6, 8), dtype=torch.int32, generator=g)
    exps[:, 0, 0], exps[:, 0, 1] = lo, hi  # every expert hits both ends
    sf = (exps + 127).to(torch.uint8)
    e4m3, alpha = mxfp4_scales_to_nvfp4_shifted(sf)
    assert e4m3.shape == (4, 6, 16) and alpha.shape == (4,)
    true = _ue8m0_value(sf).repeat_interleave(2, dim=-1)
    recon = _e4m3_value(e4m3) * alpha.view(-1, 1, 1)
    assert torch.equal(recon, true)  # exact in fp32: only exponents move


def test_per_expert_shift_differs() -> None:
    low = torch.full((1, 2, 4), -16 + 127, dtype=torch.uint8)
    low[0, 0, 0] = 127  # expert 0 spans -16..0
    ok = torch.full((1, 2, 4), -5 + 127, dtype=torch.uint8)  # expert 1 fits as is
    e4m3, alpha = mxfp4_scales_to_nvfp4_shifted(torch.cat([low, ok]))
    assert (
        alpha[0] == 2.0**-7 and alpha[1] == 1.0
    )  # shift +7 (lo -16 -> -9) => alpha = 2^-7


def test_span_too_large_and_bad_bytes_raise() -> None:
    with pytest.raises(ValueError):
        mxfp4_scales_to_nvfp4_shifted(
            torch.tensor([[[127 - 10, 127 + 8]]], dtype=torch.uint8)
        )  # span 18
    with pytest.raises(ValueError):
        mxfp4_scales_to_nvfp4_shifted(torch.tensor([[[255, 127]]], dtype=torch.uint8))
    with pytest.raises(ValueError):
        mxfp4_scales_to_nvfp4_shifted(torch.tensor([[[0, 127]]], dtype=torch.uint8))


# ----------------------------------------------------------------------- config


def _nvfp4_config(**kwargs):
    from flashinfer.moe_ep import Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig

    return Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig(**kwargs)


def test_config_is_a_sibling_of_the_w4a8_config() -> None:
    from flashinfer.moe_ep import Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig

    config = _nvfp4_config()
    assert not isinstance(config, Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig)
    assert config.kernel_name == "sm120_nvfp4_nvfp4_bf16_cake"
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_backend import (
        Sm120Mxfp4Mxfp8CakeMegaKernelBackend,
    )

    with pytest.raises(TypeError):
        Sm120Mxfp4Mxfp8CakeMegaKernelBackend(config)


def test_config_return_codec_aliases_resolve_or_raise() -> None:
    assert _nvfp4_config().return_codec == "bf16"
    for kwargs in (
        {"combine_dtype": "int8"},
        {"return_codec": "int8"},
        {"combine_dtype": "int8", "return_codec": "int8"},
    ):
        config = _nvfp4_config(**kwargs)
        assert config.combine_dtype == config.return_codec == "int8"
    with pytest.raises(ValueError, match="conflicting return codecs"):
        _nvfp4_config(combine_dtype="bf16", return_codec="int8")
    with pytest.raises(ValueError):
        _nvfp4_config(combine_dtype="fp8")


def test_config_clamp_aliases_resolve_or_raise() -> None:
    config = _nvfp4_config()
    assert config.swiglu_limit == 0 and config.gate_up_clamp is None
    for kwargs in (
        {"gate_up_clamp": 7},
        {"activation_clamp": 7.0},
        {"swiglu_limit": 7},
        {"gate_up_clamp": 7, "activation_clamp": 7, "swiglu_limit": 7},
    ):
        config = _nvfp4_config(**kwargs)
        assert config.swiglu_limit == 7 and config.gate_up_clamp == 7.0
    for kwargs in (
        {"gate_up_clamp": 7, "activation_clamp": 10},
        {"gate_up_clamp": 7, "swiglu_limit": 10},
        {"gate_up_clamp": 7, "swiglu_limit": 0},
    ):
        with pytest.raises(ValueError, match="conflicting activation clamps"):
            _nvfp4_config(**kwargs)
    with pytest.raises(ValueError):
        _nvfp4_config(gate_up_clamp=7.5)


def test_config_replace_keeps_resolved_aliases() -> None:
    import dataclasses

    config = _nvfp4_config(combine_dtype="int8", gate_up_clamp=7)
    again = dataclasses.replace(config, knobs={"x": 1})
    assert again.return_codec == "int8" and again.swiglu_limit == 7


def test_config_rejects_host_or_malformed_scales() -> None:
    for bad in (
        torch.ones(4),  # host tensor: a pageable copy per launch, not capturable
        torch.ones(4, dtype=torch.float64),
        torch.ones(2, 2),
        [1.0, 1.0],
    ):
        with pytest.raises(ValueError, match="1-D float32 CUDA tensor"):
            _nvfp4_config(fc1_alpha=bad)


def test_expert_scale_length_is_checked() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm120.nvfp4_nvfp4_bf16_cake.cake_nvfp4_config import (
        check_expert_scale,
    )

    check_expert_scale("fc1_alpha", None, 64)
    if torch.cuda.is_available():
        value = torch.ones(64, device="cuda")
        check_expert_scale("fc1_alpha", value, 64)
        with pytest.raises(ValueError, match="local experts"):
            check_expert_scale("fc1_alpha", value, 32)


def test_staging_quantizes_into_the_workspace_rows() -> None:
    """The fused quantizer writes the leading rows of the staging buffers in place (no
    temporary + copy) and matches the torch reference bit for bit."""

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from flashinfer.moe_ep.backends.mega.kernel.sm120.nvfp4_nvfp4_bf16_cake.cake_nvfp4_quant import (
        _HAVE_TRITON,
        nvfp4_quantize_fused,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.nvfp4_nvfp4_bf16_cake.cake_nvfp4_weights import (
        nvfp4_quantize_torch,
    )

    if not _HAVE_TRITON:
        pytest.skip("needs triton")
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(37, 1024, device="cuda", generator=g).to(torch.bfloat16)
    codes = torch.full((64, 512), 0xAB, dtype=torch.uint8, device="cuda")
    sf = torch.full((64, 64), 0xCD, dtype=torch.uint8, device="cuda")
    nvfp4_quantize_fused(x, 2.0, codes=codes, sf=sf)
    ref_codes, ref_sf = nvfp4_quantize_torch(x, 2.0)
    assert torch.equal(codes[:37], ref_codes) and torch.equal(sf[:37], ref_sf)
    assert bool((codes[37:] == 0xAB).all()) and bool((sf[37:] == 0xCD).all())
