"""E2M1 weights with fractional E4M3 scales shared by 32 K elements."""

import pytest
import torch

pytest.importorskip("cutlass")

from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
    _normalize_source_format_for_quant_mode,
)
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
    _validate_e4m3_k32_scales,
)


def test_k32_rejects_w4a4():
    with pytest.raises(ValueError, match="requires quant_mode='w4a16'"):
        _normalize_source_format_for_quant_mode("modelopt_e4m3_k32", "nvfp4")


@pytest.mark.parametrize("shape", [(2, 64, 2), (1, 64, 4), (1, 32, 2), (64, 2)])
def test_k32_rejects_wrong_scale_shape(shape):
    with pytest.raises(ValueError, match="must"):
        _validate_e4m3_k32_scales(
            torch.empty(shape, dtype=torch.float8_e4m3fn),
            experts=1,
            rows=64,
            cols=64,
            name="scale",
        )


def test_k32_rejects_non_e4m3_scales():
    with pytest.raises(TypeError, match="float8_e4m3fn"):
        _validate_e4m3_k32_scales(
            torch.empty((1, 64, 2)), experts=1, rows=64, cols=64, name="scale"
        )


def _weights(experts, rows, cols, generator):
    codes = torch.randint(
        0,
        16,
        (experts, rows, cols),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    packed = (codes[..., ::2] | (codes[..., 1::2] << 4)).contiguous()
    scales = (
        torch.randint(
            0, 8, (experts, rows, cols // 32), device="cuda", generator=generator
        ).float()
        / 4
    ).to(torch.float8_e4m3fn)
    global_scale = torch.full((experts,), 1 / (cols**0.5 * 3), device="cuda")
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    reference = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1, 1)
    reference *= scales.float().repeat_interleave(32, -1) * global_scale[:, None, None]
    return packed, scales, global_scale, reference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [17, 129])
def test_k32_public_api_matches_reference(tokens):
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        pytest.skip("SM12x required")
    from flashinfer.fused_moe import b12x_fused_moe

    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device="cuda").manual_seed(42)
    w13, s13, g13, r13 = _weights(4, 1024, 512, gen)
    w2, s2, g2, r2 = _weights(4, 512, 512, gen)
    x = torch.randn(tokens, 512, device="cuda", generator=gen).bfloat16()
    ids = torch.rand(tokens, 4, device="cuda", generator=gen).topk(2, -1).indices.int()
    weights = torch.full((tokens, 2), 0.5, device="cuda")
    out = b12x_fused_moe(
        x,
        w13,
        s13,
        w2,
        s2,
        ids,
        weights,
        4,
        2,
        w1_alpha=g13,
        w2_alpha=g2,
        quant_mode="w4a16",
        source_format="modelopt_e4m3_k32",
    )
    reference = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(4):
        tokens_, slots = (ids == expert).nonzero(as_tuple=True)
        hidden = (x[tokens_].float() @ r13[expert].T).bfloat16().float()
        up, gate = hidden.chunk(2, -1)
        activation = (torch.nn.functional.silu(gate).bfloat16() * up.bfloat16()).float()
        y = (activation @ r2[expert].T) * weights[tokens_, slots, None]
        reference.index_add_(0, tokens_, y.bfloat16().float())
    relative_l2 = (out.float() - reference).norm() / reference.norm()
    assert torch.isfinite(out).all()
    assert relative_l2 < 0.015, relative_l2.item()
