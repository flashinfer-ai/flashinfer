"""Host-only tests of the Cake SM120 MXFP4 x MXFP4 (W4A4) backend: config, geometry, quantizer rule."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp4_bf16_cake import (
    mxfp4_dequantize_torch,
    mxfp4_quantize_torch,
)


def _geometry():
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        cake_geometry,
    )

    return cake_geometry


# ----------------------------------------------------------------------- quantizer rule


def _seed_rule(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The Cake seed's MXFP4 activation rule, written independently: e = ceil(log2(amax / 6)) from the
    FP32 quotient, clamped to [1, 254] as a UE8M0 byte, codes = RNE e2m1(x / 2^(e - 127)) (true division)."""

    rows, cols = values.shape
    blocks = values.float().reshape(rows, cols // 32, 32)
    bits = (blocks.abs().amax(dim=-1) / 6.0).contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exponent = exponent.clamp(min=1, max=254)
    scale = torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent - 127)
    scaled = blocks / scale[..., None]
    grid = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
    magnitude = scaled.abs().unsqueeze(-1)
    # nearest grid point, ties to the even code (codes 0, 2, 4, 6 are even)
    distance = (magnitude - grid).abs()
    best = distance.min(dim=-1, keepdim=True).values
    candidates = torch.where(distance == best, torch.arange(8), 99)
    even = torch.where(candidates % 2 == 0, candidates, 99).min(dim=-1).values
    code = torch.where(even < 99, even, candidates.min(dim=-1).values)
    code = torch.where(scaled.abs() > 6.0, torch.tensor(7), code)
    nibbles = (code | (torch.signbit(scaled).to(torch.int64) << 3)).reshape(rows, cols)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)
    return packed, exponent.to(torch.uint8)


def _samples() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    e = torch.randint(-20, 20, (16, 32, 1), generator=g).float()
    ties = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -0.25, -5.0, 0.0, 3.0)
    )
    pow2 = torch.zeros(8, 64)
    pow2[:, 0] = torch.tensor(
        (6.0, 3.0, 1.5, 12.0, 6.0 * 2.0**-20, 6.0 * 2.0**20, 0.75, 96.0)
    )
    pow2[:, 1:32] = 0.37
    return {
        "randn": torch.randn(9, 256, generator=g),
        "wide": (torch.randn(16, 32, 32, generator=g) * torch.exp2(e)).reshape(
            16, 1024
        ),
        "ties": ties[torch.randint(0, len(ties), (8, 128), generator=g)],
        "pow2_amax": pow2,  # amax / 6 exactly a power of two (no round-up)
        "tiny": torch.randn(4, 64, generator=g) * 1e-39,  # subnormal amax / 6
        "zeros": torch.zeros(3, 64),
        "bf16": torch.randn(5, 512, generator=g).to(torch.bfloat16).float(),
    }


@pytest.mark.parametrize("kind", list(_samples()))
def test_quantizer_matches_the_seed_rule(kind: str) -> None:
    x = _samples()[kind]
    packed, sf = mxfp4_quantize_torch(x)
    ref_packed, ref_sf = _seed_rule(x)
    assert torch.equal(sf, ref_sf)
    assert torch.equal(packed, ref_packed)


def test_quantizer_scale_is_the_smallest_covering_power_of_two() -> None:
    x = _samples()["wide"]
    packed, sf = mxfp4_quantize_torch(x)
    scale = torch.ldexp(torch.ones(sf.shape), sf.to(torch.int32) - 127)
    amax = x.reshape(x.shape[0], -1, 32).abs().amax(dim=-1)
    assert bool((amax <= 6.0 * scale).all())
    assert bool((amax > 3.0 * scale).all())  # half the scale would not cover the block
    # values on the E2M1 grid times the scale survive exactly
    y = mxfp4_dequantize_torch(packed, sf)
    p2, s2 = mxfp4_quantize_torch(y)
    assert torch.equal(mxfp4_dequantize_torch(p2, s2), y)


def test_quantizer_zero_block_and_norm() -> None:
    packed, sf = mxfp4_quantize_torch(torch.zeros(2, 64))
    assert bool((packed == 0).all()) and bool((sf == 1).all())
    with pytest.raises(ValueError, match="no global scale"):
        mxfp4_quantize_torch(torch.zeros(2, 64), 2.0)
    with pytest.raises(ValueError):
        mxfp4_quantize_torch(torch.zeros(2, 48))


# ----------------------------------------------------------------------- config / backend


def test_config_is_a_sibling_of_the_other_cake_configs() -> None:
    from flashinfer.moe_ep import (
        Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig,
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
        Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp4_bf16_cake import (
        Sm120Mxfp4Mxfp4CakeMegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_backend import (
        Sm120Mxfp4Mxfp8CakeMegaKernelBackend,
    )

    config = Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig(
        swiglu_limit=7, return_codec="int8", shared_expert="fp8_in_kernel"
    )
    assert config.kernel_name == "sm120_mxfp4_mxfp4_bf16_cake"
    assert config.variant == "fp8shared" and config.kernel_codec_overflow_blocks == 16
    assert not isinstance(config, Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig)
    assert not isinstance(Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(), type(config))
    with pytest.raises(TypeError):
        Sm120Mxfp4Mxfp8CakeMegaKernelBackend(config)
    for other in (
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(),
        Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig(),
    ):
        with pytest.raises(TypeError):
            Sm120Mxfp4Mxfp4CakeMegaKernelBackend(other)
    backend = Sm120Mxfp4Mxfp4CakeMegaKernelBackend(config)
    assert backend.kernel_name() == "sm120_mxfp4_mxfp4_bf16_cake"
    assert backend.ACT_CODEC == 2
    assert backend._activation_shapes(256, 4096) == ((256, 2048), (256, 128))
    fleet = SimpleNamespace(
        num_experts=256, max_tokens_per_rank=4096, token_hidden_size=4096
    )
    params = backend.kernel_params(4, fleet)
    assert (
        params.act_codec == 2 and params.return_codec == 1 and params.swiglu_limit == 7
    )
    assert params.device_parameters()[-1] == 2


def test_backend_is_registered() -> None:
    from flashinfer.moe_ep import Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    backend = create_mega_kernel(Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig())
    assert backend.ACT_CODEC == 2


def test_geometry_region_and_key() -> None:
    G = _geometry()
    with pytest.raises(G.GeometryError, match="must be 0"):
        G.check_supported_region(
            G.KernelParams(4, 256, 6, 8192, 4096, 2048, False, act_codec=3)
        )
    with pytest.raises(G.GeometryError, match="hidden_size % 1024"):
        G.check_supported_region(
            G.KernelParams(4, 256, 6, 8192, 3584, 2048, False, act_codec=2)
        )
    with pytest.raises(G.GeometryError, match="intermediate_size % 256"):
        G.check_supported_region(
            G.KernelParams(4, 256, 6, 8192, 4096, 1920, True, act_codec=2)
        )
    G.check_supported_region(
        G.KernelParams(4, 256, 6, 8192, 4096, 1920, False, act_codec=2)
    )


@pytest.mark.parametrize("hidden", [4096, 5120])
def test_geometry_mxfp4_records_and_pool(hidden: int) -> None:
    """Codec 2 records carry H / 2 + H / 32 bytes (+ the 128 B meta, padded to 128 B), the pool rows
    are packed e2m1, the scale words those of MXFP8 and there is no per-expert scale tail."""

    G = _geometry()
    kernels = {
        c: G.describe_kernel(
            G.KernelParams(4, 256, 6, 8192, hidden, 2048, True, act_codec=c)
        )
        for c in (0, 1, 2)
    }
    assert kernels[2]["key"].endswith("_mxfp4") and kernels[1]["key"].endswith("_nvfp4")
    assert not kernels[0]["key"].endswith("_mxfp8")
    record = -(-(128 + hidden // 2 + hidden // 32) // 128) * 128
    assert kernels[2]["transport"]["record_bytes"] == record
    c0, c1, c2 = (
        kernels[c]["instances"]["m128_copier"]["constants"] for c in (0, 1, 2)
    )
    assert c2["POOL_ACT_WORDS"] == hidden // 8 == c1["POOL_ACT_WORDS"]
    assert c2["SCALE_WORDS"] == hidden // 128 == c0["SCALE_WORDS"]
    assert c2["EXPERT_SCALE_WORDS"] == 0 < c1["EXPERT_SCALE_WORDS"]
    assert c2["INTER_ROW_BYTES"] == 2048 // 2


def test_prequantized_input_validation() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp4_bf16_cake.cake_mxfp4_backend import (
        MXFP4_CODEC,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.nvfp4_nvfp4_bf16_cake.cake_nvfp4_staging import (
        MoEEpConfigError,
        validate_fp4_forward_inputs,
    )

    fleet = SimpleNamespace(
        num_experts=256, max_tokens_per_rank=64, token_hidden_size=1024
    )
    ids = torch.zeros(5, 6, dtype=torch.int64)
    weights = torch.zeros(5, 6, dtype=torch.float32)
    packed = torch.zeros(5, 512, dtype=torch.uint8)
    ok = torch.zeros(5, 32, dtype=torch.uint8)

    def check(x, scales):
        validate_fp4_forward_inputs(
            MXFP4_CODEC,
            x,
            ids,
            weights,
            fleet,
            top_k=6,
            quantize_input=False,
            scales=scales,
        )

    check(packed, ok)
    check(packed, ok.view(torch.float8_e8m0fnu))
    for x, scales in (
        (packed, None),
        (packed, torch.zeros(5, 64, dtype=torch.uint8)),  # NVFP4-sized scales
        (packed, ok.view(torch.float8_e4m3fn)),
        (torch.zeros(5, 1024, dtype=torch.uint8), ok),  # MXFP8-sized codes
    ):
        with pytest.raises(MoEEpConfigError):
            check(x, scales)


def test_staging_quantizes_into_the_workspace_rows() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp4_bf16_cake.cake_mxfp4_backend import (
        MXFP4_CODEC,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.nvfp4_nvfp4_bf16_cake.cake_nvfp4_staging import (
        FUSED_QUANT,
        stage_fp4_inputs,
    )

    if not FUSED_QUANT:
        pytest.skip("needs triton")
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(37, 1024, device="cuda", generator=g).to(torch.bfloat16)
    ids = torch.randint(0, 64, (37, 6), device="cuda", generator=g)
    w = torch.rand(37, 6, device="cuda", generator=g)
    codes = torch.full((64, 512), 0xAB, dtype=torch.uint8, device="cuda")
    sf = torch.full((64, 32), 0xCD, dtype=torch.uint8, device="cuda")
    idx_out = torch.full((64, 6), -7, dtype=torch.int64, device="cuda")
    w_out = torch.zeros(64, 6, device="cuda")
    rows = stage_fp4_inputs(
        MXFP4_CODEC,
        x,
        ids,
        w,
        x=codes,
        x_sf=sf,
        topk_idx_out=idx_out,
        topk_weights_out=w_out,
        quantize_input=True,
        scales=None,
    )
    ref_codes, ref_sf = mxfp4_quantize_torch(x)
    assert rows == 37
    assert torch.equal(codes[:37], ref_codes) and torch.equal(sf[:37], ref_sf)
    assert bool((codes[37:] == 0xAB).all()) and bool((sf[37:] == 0xCD).all())
    assert torch.equal(idx_out[:37], ids) and torch.equal(w_out[:37], w)
