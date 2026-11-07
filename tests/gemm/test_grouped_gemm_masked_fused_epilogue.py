"""Tests for the fused epilogues of the masked grouped GEMM (SM100/SM103)."""

import pytest
import torch

from flashinfer.cute_dsl.utils import is_cute_dsl_available
from flashinfer.gemm import grouped_gemm_nt_masked
from flashinfer.tllm_enums import ActivationType
from flashinfer.utils import get_compute_capability

SF_VEC = 16
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _skip_unless_sm100():
    if not is_cute_dsl_available():
        pytest.skip("Please `pip install nvidia-cutlass-dsl`")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3)):
        pytest.skip("Fused masked GEMM epilogues require SM100/SM103")


def _swizzle_sf(sf: torch.Tensor) -> torch.Tensor:
    """(l, rows, k/16) E4M3 -> masked GEMM layout (32, 4, rm, 4, rk, l)."""
    l, rows, k_sf = sf.shape
    phys = sf.view(l, rows // 128, 4, 32, k_sf // 4, 4).permute(0, 1, 4, 3, 2, 5)
    return phys.contiguous().permute(3, 4, 1, 5, 2, 0)


def _nvfp4_operand(l: int, rows: int, k: int, device):
    """Random NVFP4 operand as (packed (rows, k/2, l), SF, dequantized (l, rows, k))."""
    codes = torch.randint(0, 16, (l, rows, k), dtype=torch.uint8, device=device)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    lut = torch.tensor(E2M1_VALUES, device=device)
    lut = torch.cat([lut, -lut])
    sf = (torch.rand(l, rows, k // SF_VEC, device=device) * 1.5 + 0.25).to(
        torch.float8_e4m3fn
    )
    dequant = lut[codes.long()] * sf.float().repeat_interleave(SF_VEC, dim=-1)
    return packed.permute(1, 2, 0), _swizzle_sf(sf), dequant


def _swiglu_reference(acc, gate_scale, up_scale, alpha, beta, limit):
    """acc is (l, m, n) with 64-row up/gate blocks; returns (l, m, n/2)."""
    l, m, n = acc.shape
    blocks = acc.view(l, m, n // 128, 2, 64)
    up = blocks[:, :, :, 0].reshape(l, m, n // 2) * up_scale
    gate = blocks[:, :, :, 1].reshape(l, m, n // 2) * gate_scale
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + beta)


@pytest.mark.parametrize("l,m", [(2, 256), (5, 128)])
@pytest.mark.parametrize("n,k", [(512, 1024), (2048, 512)])
@pytest.mark.parametrize("mma_tiler_mn", [(128, 128), (128, 256)])
@pytest.mark.parametrize(
    "swiglu", [None, (1.702, 1.0, 7.0)], ids=["default", "clamped"]
)
@pytest.mark.parametrize("separate_up_alpha", [False, True])
@pytest.mark.parametrize("per_token_scale", [False, True])
def test_gated_activation(
    l, m, n, k, mma_tiler_mn, swiglu, separate_up_alpha, per_token_scale
):
    _skip_unless_sm100()
    torch.manual_seed(0)
    device = torch.device("cuda")
    a, sfa, a_ref = _nvfp4_operand(l, m, k, device)
    b, sfb, b_ref = _nvfp4_operand(l, n, k, device)
    acc = torch.bmm(a_ref, b_ref.transpose(1, 2))
    masked_m = torch.randint(0, m + 1, (l,), dtype=torch.int32, device=device)
    masked_m[0] = m

    # Normalize the activation input to O(1) so the clamp is exercised.
    std = acc.std().item()
    alpha = torch.rand(l, device=device) * 2 / std + 0.5 / std
    alpha_up = torch.rand(l, device=device) * 2 / std + 0.5 / std
    token_scale = torch.rand(l, m, device=device) + 0.5
    kwargs = {}
    if swiglu is not None:
        kwargs.update(swiglu_alpha=swiglu[0], swiglu_beta=swiglu[1])
        kwargs.update(swiglu_limit=swiglu[2])
    if separate_up_alpha:
        kwargs["alpha_up"] = alpha_up
    if per_token_scale:
        kwargs["a_per_token_scale"] = token_scale

    out = torch.full((l, m, n // 2), float("nan"), dtype=torch.bfloat16, device=device)
    grouped_gemm_nt_masked(
        (a, sfa),
        (b, sfb),
        out.permute(1, 2, 0),
        masked_m,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16",
        sf_vec_size=SF_VEC,
        mma_tiler_mn=mma_tiler_mn,
        alpha=alpha,
        alpha_dtype="float32",
        activation_type=ActivationType.Swiglu,
        **kwargs,
    )

    gate_scale = alpha.view(l, 1, 1)
    up_scale = (alpha_up if separate_up_alpha else alpha).view(l, 1, 1)
    if per_token_scale:
        gate_scale = gate_scale * token_scale.unsqueeze(-1)
        up_scale = up_scale * token_scale.unsqueeze(-1)
    act_alpha, beta, limit = swiglu or (1.0, 0.0, torch.finfo(torch.float32).max)
    ref = _swiglu_reference(acc, gate_scale, up_scale, act_alpha, beta, limit)
    for i in range(l):
        rows = masked_m[i].item()
        torch.testing.assert_close(
            out[i, :rows].float(), ref[i, :rows], atol=2e-2, rtol=1e-2
        )


def _run_invalid(**overrides):
    device = torch.device("cuda")
    l, m, n, k = 2, 128, 256, 256
    a, sfa, _ = _nvfp4_operand(l, m, k, device)
    b, sfb, _ = _nvfp4_operand(l, n, k, device)
    args = dict(
        out=torch.empty(l, m, n // 2, dtype=torch.bfloat16, device=device).permute(
            1, 2, 0
        ),
        c_dtype="bfloat16",
        activation_type=ActivationType.Swiglu,
    )
    args.update(overrides)
    grouped_gemm_nt_masked(
        (a, sfa),
        (b, sfb),
        args.pop("out"),
        torch.full((l,), m, dtype=torch.int32, device=device),
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        sf_vec_size=SF_VEC,
        **args,
    )


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"activation_type": ActivationType.Relu2}, NotImplementedError),
        ({"is_swap_ab": True}, NotImplementedError),
        ({"mma_tiler_mn": (256, 128), "cluster_shape_mn": (2, 1)}, ValueError),
        ({"c_dtype": "float8_e4m3fn"}, ValueError),
        ({"swiglu_limit": -1.0}, ValueError),
        ({"a_per_token_scale": torch.ones(2, 64)}, ValueError),
        ({"activation_type": None, "alpha_up": torch.ones(2)}, ValueError),
    ],
)
def test_gated_activation_rejects_unsupported_arguments(overrides, error):
    _skip_unless_sm100()
    overrides = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in overrides.items()
    }
    with pytest.raises(error):
        _run_invalid(**overrides)


def test_gated_activation_rejects_full_width_output():
    _skip_unless_sm100()
    out = torch.empty(2, 128, 256, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="out of shape"):
        _run_invalid(out=out.permute(1, 2, 0))
