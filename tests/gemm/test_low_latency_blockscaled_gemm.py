"""Kernel-level tests for the low-latency block-scaled GEMM."""

import pytest
import torch

cutlass = pytest.importorskip("cutlass")

from flashinfer.gemm.gemm_base import _cutedsl_low_latency_blockscaled_gemm_runner
from flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm import (
    LowLatencyBlockscaledGemmKernel,
    _decode_ab_to_f32,
    make_blockscaled_tensors,
)


_SUPPORTED_FORMATS = [
    (
        cutlass.Float4E2M1FN,
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
    ),
    *[
        (a_dtype, b_dtype, cutlass.Float8E8M0FNU, 32)
        for a_dtype in (
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        )
        for b_dtype in (
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        )
    ],
]

_GPT_OSS_120B_KERNEL_SHAPES = [
    (1280, tokens, 2880, 1)  # qkv_proj, TP=4
    for tokens in (1, 4, 8)
] + [
    (2880, tokens, 1024, 1)  # o_proj, TP=4
    for tokens in (1, 4, 8)
]

_DEEPSEEK_V3_KERNEL_SHAPES = [
    (7168, tokens, 2048, 1)  # o_proj, TP=8
    for tokens in (1, 4, 8)
] + [
    (3072, tokens, 1536, 1)  # q_b_proj, TP=8
    for tokens in (1, 4, 8)
]

_EDGE_KERNEL_SHAPES = [
    (1, 1, 128, 1),
    (127, 7, 128, 2),
    (128, 8, 256, 1),
    (129, 3, 384, 2),
]


def _require_supported_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("low-latency block-scaled GEMM requires SM100 or SM103")


def _reference_scales(scales, sf_vec_size, k):
    return torch.repeat_interleave(scales.cpu().to(torch.float32), sf_vec_size, dim=1)[
        :, :k
    ]


def _reference_output(a, b, sfa, sfb, sf_vec_size, mnkl, alpha, bias):
    return (
        torch.einsum(
            "mkl,nkl->mnl",
            _decode_ab_to_f32(a) * _reference_scales(sfa, sf_vec_size, mnkl[2]),
            _decode_ab_to_f32(b) * _reference_scales(sfb, sf_vec_size, mnkl[2]),
        )
        * alpha.cpu()
        + bias.cpu().to(torch.float32)[:, None, None]
    ).to(torch.bfloat16)


def _make_runner():
    return _cutedsl_low_latency_blockscaled_gemm_runner(100, False)


def _run_all_tactics_and_check(runner, mnkl, fmt, workspace, alpha):
    a_dtype, b_dtype, sf_dtype, sf_vec_size = fmt
    a, b, sfa, sfb, out, sfa_simple, sfb_simple = make_blockscaled_tensors(
        mnkl,
        a_dtype,
        b_dtype,
        sf_dtype,
        sf_vec_size,
        cutlass.BFloat16,
    )
    bias = torch.linspace(-0.5, 0.5, mnkl[0], dtype=torch.bfloat16, device="cuda")
    inputs = [a, b, sfa, sfb, out, workspace, mnkl, alpha, bias]
    tactics = runner.get_valid_tactics(inputs, None)
    assert tactics, f"expected at least one tactic for {mnkl} and {fmt}"
    reference = _reference_output(
        a, b, sfa_simple, sfb_simple, sf_vec_size, mnkl, alpha, bias
    )
    for tactic in tactics:
        out.zero_()
        runner.forward(inputs, tactic=tactic)
        torch.testing.assert_close(out.cpu(), reference, atol=1e-1, rtol=1e-3)
    return tactics


@pytest.mark.parametrize(
    "mnkl,a_dtype,b_dtype,sf_dtype,sf_vec_size,c_dtype,cta_k,ab_stages,split_k,expected",
    [
        (
            (129, 7, 256, 2),
            cutlass.Float4E2M1FN,
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.BFloat16,
            64,
            12,
            8,
            True,
        ),
        (
            (3072, 8, 4096, 6),
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float16,
            512,
            2,
            4,
            True,
        ),
        (
            (128, 8, 128, 1),
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float32,
            128,
            8,
            2,
            True,
        ),
        # MX layouts require CTA-K to contain complete four-MMA-K atoms.
        (
            (128, 8, 128, 1),
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E8M0FNU,
            32,
            cutlass.Float32,
            64,
            8,
            2,
            False,
        ),
        # NVFP4 scale factors are only valid with two FP4 operands and vec16.
        (
            (128, 8, 128, 1),
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            16,
            cutlass.BFloat16,
            128,
            8,
            1,
            False,
        ),
        (
            (128, 8, 256, 1),
            cutlass.Float4E2M1FN,
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.BFloat16,
            256,
            13,
            1,
            False,
        ),
        (
            (128, 8, 256, 1),
            cutlass.Float4E2M1FN,
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            16,
            cutlass.BFloat16,
            256,
            8,
            16,
            False,
        ),
    ],
)
def test_low_latency_blockscaled_gemm_can_implement(
    mnkl,
    a_dtype,
    b_dtype,
    sf_dtype,
    sf_vec_size,
    c_dtype,
    cta_k,
    ab_stages,
    split_k,
    expected,
):
    assert (
        LowLatencyBlockscaledGemmKernel.can_implement(
            mnkl,
            a_dtype,
            b_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            mma_tiler_mnk=(128, 8, cta_k),
            num_ab_stage=ab_stages,
            num_sfb_tmem_stage=1,
            split_k=split_k,
        )
        is expected
    )


@pytest.mark.parametrize("fmt", _SUPPORTED_FORMATS)
def test_low_latency_blockscaled_gemm_all_tactics_correctness(fmt):
    """Every generated tactic is correct, then exercise model and edge shapes."""
    _require_supported_gpu()
    a_dtype, b_dtype, sf_dtype, sf_vec_size = fmt
    runner = _make_runner()
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    # K=256 exposes three NVFP4 CTA-K divisors and two MX CTA-K divisors while
    # keeping this exhaustive compile-and-run sweep reasonably sized.
    mnkl = (129, 7, 256, 2)
    a, b, sfa, sfb, out, sfa_simple, sfb_simple = make_blockscaled_tensors(
        mnkl,
        a_dtype,
        b_dtype,
        sf_dtype,
        sf_vec_size,
        cutlass.BFloat16,
    )
    alpha = torch.tensor(0.125, dtype=torch.float32, device="cuda")
    bias = torch.linspace(-0.5, 0.5, mnkl[0], dtype=torch.bfloat16, device="cuda")
    inputs = [a, b, sfa, sfb, out, workspace, mnkl, alpha, bias]
    tactics = runner.get_valid_tactics(inputs, None)

    assert len(tactics) == len(set(tactics))
    assert {tactic[3] for tactic in tactics} == {1, 2, 4, 8}
    assert {tactic[2] for tactic in tactics} == {1}
    assert {tactic[1] for tactic in tactics}.issuperset({1, 4, 8})
    assert len({tactic[0] for tactic in tactics}) > 1
    assert all(
        LowLatencyBlockscaledGemmKernel.can_implement(
            mnkl,
            a_dtype,
            b_dtype,
            sf_dtype,
            sf_vec_size,
            cutlass.BFloat16,
            mma_tiler_mnk=(128, 8, tactic[0]),
            num_ab_stage=tactic[1],
            num_sfb_tmem_stage=tactic[2],
            split_k=tactic[3],
        )
        for tactic in tactics
    )

    reference = _reference_output(
        a, b, sfa_simple, sfb_simple, sf_vec_size, mnkl, alpha, bias
    )
    gemm_only = _reference_output(
        a, b, sfa_simple, sfb_simple, sf_vec_size, mnkl, alpha, torch.zeros_like(bias)
    )
    assert gemm_only.abs().max() > 1.0
    for tactic in tactics:
        out.zero_()
        runner.forward(inputs, tactic=tactic)
        torch.testing.assert_close(out.cpu(), reference, atol=1e-1, rtol=1e-3)

    # Standard 128x4 scales only match the direct SFB layout for one narrow
    # kernel-N tile. Public dispatchers reach this layout by swapping A/B.
    wide_inputs = [
        a,
        b,
        sfa,
        sfb,
        out,
        workspace,
        (129, 9, 256, 2),
        alpha,
        bias,
    ]
    assert runner.get_valid_tactics(wide_inputs, None) == []

    model_shapes = []
    if sf_dtype is cutlass.Float8E4M3FN:
        model_shapes = _GPT_OSS_120B_KERNEL_SHAPES + _DEEPSEEK_V3_KERNEL_SHAPES
    elif a_dtype is cutlass.Float4E2M1FN and b_dtype is cutlass.Float8E4M3FN:
        # MX block scales require K divisible by 128.
        model_shapes = [
            (m, n, 2944 if k == 2880 else k, l)
            for m, n, k, l in (_GPT_OSS_120B_KERNEL_SHAPES + _DEEPSEEK_V3_KERNEL_SHAPES)
        ]
    elif a_dtype is b_dtype is cutlass.Float8E4M3FN:
        model_shapes = [
            shape
            for shape in _GPT_OSS_120B_KERNEL_SHAPES + _DEEPSEEK_V3_KERNEL_SHAPES
            if shape[2] % 128 == 0
        ]

    edge_shapes = [
        (m, n, k if sf_vec_size == 16 else ((k + 127) // 128 * 128), l)
        for m, n, k, l in _EDGE_KERNEL_SHAPES
    ]
    for shape in model_shapes[:1] + model_shapes[-1:] + edge_shapes[:1]:
        _run_all_tactics_and_check(runner, shape, fmt, workspace, alpha)
