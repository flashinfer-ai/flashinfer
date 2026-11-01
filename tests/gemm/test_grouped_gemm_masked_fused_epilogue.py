"""Tests for the fused epilogues of the masked grouped GEMM (SM100/SM103)."""

import pytest
import torch

from flashinfer.cute_dsl.utils import is_cute_dsl_available
from flashinfer.gemm import grouped_gemm_nt_masked
from flashinfer.quantization.fp4_quantization import scaled_fp4_grouped_quantize
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


def _unswizzle_sf(sf: torch.Tensor, l: int, rows: int, k_sf: int) -> torch.Tensor:
    """Swizzled E4M3 bytes of any shape -> (l, rows, k_sf) raw bytes."""
    phys = sf.contiguous().view(torch.uint8).reshape(-1)
    if sf.dim() == 6:
        # The quantizer returns the permuted (32, 4, rm, 4, rk, l) view.
        phys = sf.permute(5, 2, 4, 0, 1, 3).contiguous().view(torch.uint8)
    phys = phys.reshape(l, rows // 128, k_sf // 4, 32, 4, 4)
    return phys.permute(0, 1, 4, 3, 2, 5).reshape(l, rows, k_sf)


def _run_gemm(a, sfa, b, sfb, masked_m, out, c_dtype, **kwargs):
    grouped_gemm_nt_masked(
        (a, sfa),
        (b, sfb),
        out,
        masked_m,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype=c_dtype,
        sf_vec_size=SF_VEC,
        **kwargs,
    )


def _nvfp4_case(l, m, n, k, gated, mma_tiler_mn, device):
    a, sfa, a_ref = _nvfp4_operand(l, m, k, device)
    b, sfb, b_ref = _nvfp4_operand(l, n, k, device)
    std = torch.bmm(a_ref, b_ref.transpose(1, 2)).std().item()
    masked_m = torch.randint(1, m + 1, (l,), dtype=torch.int32, device=device)
    masked_m[0] = m
    kwargs = dict(
        mma_tiler_mn=mma_tiler_mn,
        alpha=torch.rand(l, device=device) * 2 / std + 0.5 / std,
        alpha_dtype="float32",
    )
    if gated:
        kwargs.update(
            activation_type=ActivationType.Swiglu,
            swiglu_alpha=1.702,
            swiglu_beta=1.0,
            swiglu_limit=7.0,
            a_per_token_scale=torch.rand(l, m, device=device) + 0.5,
        )
    return (a, sfa, b, sfb, masked_m), kwargs


@pytest.mark.parametrize("l,m", [(2, 256), (5, 128)])
@pytest.mark.parametrize("n,k", [(512, 1024), (1536, 512)])
@pytest.mark.parametrize("mma_tiler_mn", [(128, 128), (128, 256)])
@pytest.mark.parametrize("gated", [False, True])
def test_nvfp4_output_matches_separate_quantize(l, m, n, k, mma_tiler_mn, gated):
    _skip_unless_sm100()
    torch.manual_seed(0)
    device = torch.device("cuda")
    operands, kwargs = _nvfp4_case(l, m, n, k, gated, mma_tiler_mn, device)
    masked_m = operands[4]
    n_out = n // 2 if gated else n

    out_bf16 = torch.empty(l, m, n_out, dtype=torch.bfloat16, device=device)
    _run_gemm(*operands, out_bf16.permute(1, 2, 0), "bfloat16", **kwargs)
    amax = out_bf16.float().abs().amax(dim=(1, 2))
    global_scale = (448.0 * 6.0 / amax).contiguous()
    ref_q, ref_sf = scaled_fp4_grouped_quantize(out_bf16, masked_m, global_scale)

    out_q = torch.zeros(l, m, n_out // 2, dtype=torch.uint8, device=device)
    out_sf = torch.zeros(
        l * ((m + 127) // 128) * 128 * (n_out // SF_VEC),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    _run_gemm(
        *operands,
        out_q.permute(1, 2, 0),
        "float4_e2m1fn",
        out_scale=out_sf,
        global_scale=global_scale,
        **kwargs,
    )

    ref_q = ref_q.view(torch.uint8).permute(2, 0, 1)
    rows_pad = (m + 127) // 128 * 128
    ref_sf = _unswizzle_sf(ref_sf, l, rows_pad, n_out // SF_VEC)
    out_sf = _unswizzle_sf(out_sf, l, rows_pad, n_out // SF_VEC)
    for i in range(l):
        rows = masked_m[i].item()
        assert torch.equal(out_q[i, :rows], ref_q[i, :rows])
        assert torch.equal(out_sf[i, :rows], ref_sf[i, :rows])


def test_nvfp4_output_cuda_graph():
    _skip_unless_sm100()
    torch.manual_seed(0)
    device = torch.device("cuda")
    l, m, n, k = 4, 256, 1024, 512
    operands, kwargs = _nvfp4_case(l, m, n, k, True, (128, 256), device)
    masked_m = operands[4]
    global_scale = torch.full((l,), 100.0, device=device)
    outs = []
    for _ in range(2):
        outs.append(
            (
                torch.zeros(l, m, n // 4, dtype=torch.uint8, device=device),
                torch.zeros(l * m * n // 32, dtype=torch.uint8, device=device),
            )
        )

    def run(out_q, out_sf):
        _run_gemm(
            *operands,
            out_q.permute(1, 2, 0),
            "float4_e2m1fn",
            out_scale=out_sf,
            global_scale=global_scale,
            **kwargs,
        )

    run(*outs[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(*outs[1])
    masked_m.copy_(torch.tensor([256, 1, 130, 0], dtype=torch.int32))
    run(*outs[0])
    graph.replay()
    torch.cuda.synchronize()
    rows_pad = m
    sf = [_unswizzle_sf(o[1], l, rows_pad, n // 2 // SF_VEC) for o in outs]
    for i, rows in enumerate(masked_m.tolist()):
        assert torch.equal(outs[0][0][i, :rows], outs[1][0][i, :rows])
        assert torch.equal(sf[0][i, :rows], sf[1][i, :rows])


def _run_invalid_nvfp4(out_shape=(2, 128, 128), out_dtype=torch.uint8, **overrides):
    device = torch.device("cuda")
    l, m, n, k = 2, 128, 256, 256
    a, sfa, _ = _nvfp4_operand(l, m, k, device)
    b, sfb, _ = _nvfp4_operand(l, n, k, device)
    args = dict(
        c_dtype="float4_e2m1fn",
        out_scale=torch.empty(l * m * n // 16, dtype=torch.uint8, device=device),
        global_scale=torch.ones(l, device=device),
    )
    args.update(overrides)
    out = torch.empty(out_shape, dtype=out_dtype, device=device).permute(1, 2, 0)
    _run_gemm(
        a,
        sfa,
        b,
        sfb,
        torch.full((l,), m, dtype=torch.int32, device=device),
        out,
        args.pop("c_dtype"),
        **args,
    )


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"out_scale": None}, ValueError),
        ({"global_scale": torch.ones(1)}, ValueError),
        ({"out_scale": torch.empty(100, dtype=torch.uint8)}, ValueError),
        ({"out_shape": (2, 128, 256)}, ValueError),
        ({"mma_tiler_mn": (256, 128), "cluster_shape_mn": (2, 1)}, ValueError),
        ({"is_swap_ab": True}, NotImplementedError),
        (
            {
                "c_dtype": "bfloat16",
                "out_shape": (2, 128, 256),
                "out_dtype": torch.bfloat16,
            },
            ValueError,
        ),
    ],
)
def test_nvfp4_output_rejects_unsupported_arguments(overrides, error):
    _skip_unless_sm100()
    overrides = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in overrides.items()
    }
    with pytest.raises(error):
        _run_invalid_nvfp4(**overrides)


@pytest.mark.parametrize(
    "env_var",
    [
        "FLASHINFER_NVFP4_4OVER6",
        "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
        "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH",
    ],
)
def test_nvfp4_output_rejects_non_default_recipe(monkeypatch, env_var):
    _skip_unless_sm100()
    monkeypatch.setenv(env_var, "1")
    with pytest.raises(ValueError, match=env_var):
        _run_invalid_nvfp4()
