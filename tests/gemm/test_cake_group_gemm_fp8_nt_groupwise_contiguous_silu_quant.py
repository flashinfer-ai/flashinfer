# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the generated SM100a fused grouped FP8 gate_up GEMM + SwiGLU + FP8 quantization program."""

import pytest
import torch

from flashinfer.activation import silu_and_mul
from flashinfer.gemm import (
    group_gemm_fp8_nt_groupwise_contiguous,
    prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant,
)
from flashinfer.gemm.cake_grouped_fp8_fused_silu_quant import (
    ACT_ROUTE,
    FUSED_ROUTE,
    GEMM_BACKEND_CAKE,
    GEMM_BACKEND_CUTE,
    SMALL_M_MAX,
    is_group_gemm_fp8_nt_groupwise_contiguous_silu_quant_prepared_available,
    launch_plan,
    small_m_gemm_backend,
)
from flashinfer.quantization import per_token_group_quant_8bit

GROUP_SIZE = 128
EPS = 1e-10
# FP8 outputs: dequantized 0.1/0.1 per element against the chain; scales exact.
ATOL = RTOL = 0.1


@pytest.fixture(autouse=True)
def _require_generated_program():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (10, 0):
        pytest.skip("generated fused grouped FP8 gate_up programs target SM100a only")
    if not is_group_gemm_fp8_nt_groupwise_contiguous_silu_quant_prepared_available(
        device
    ):
        pytest.skip("no generated fused grouped FP8 gate_up program is registered")


def _make_inputs(group_counts, n2, k, *, seed, device, arbitrary_scales=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    groups = len(group_counts)
    m = sum(group_counts)
    a = torch.randn((m, k), generator=generator, device=device).to(torch.float8_e4m3fn)
    b = torch.randn((groups, n2, k), generator=generator, device=device).to(
        torch.float8_e4m3fn
    )
    if arbitrary_scales:
        a_scale = (
            torch.rand((m, k // 128), generator=generator, device=device) * 0.5 + 0.25
        )
        b_scale = (
            torch.rand(
                (groups, n2 // 128, k // 128), generator=generator, device=device
            )
            * 0.5
            + 0.25
        )
    else:
        a_scale = torch.pow(
            2.0,
            torch.randint(
                -8, 1, (m, k // 128), generator=generator, device=device
            ).float(),
        )
        b_scale = torch.pow(
            2.0,
            torch.randint(
                -8, 1, (groups, n2 // 128, k // 128), generator=generator, device=device
            ).float(),
        )
    counts = torch.tensor(group_counts, dtype=torch.int64, device=device)
    m_indices = torch.repeat_interleave(
        torch.arange(groups, dtype=torch.int32, device=device), counts
    )
    return a, b, a_scale.contiguous(), b_scale.contiguous(), m_indices.contiguous()


def _reference_gemm(a, b, a_scale, b_scale, m_indices):
    """K128 partial dots, post-dot scaling, ordered FP32 accumulation, BF16 output."""
    m, k = a.shape
    groups, n2, _ = b.shape
    a32 = a.float()
    b32 = b.float()
    out = torch.zeros((m, n2), dtype=torch.float32, device=a.device)
    for g in range(groups):
        rows = m_indices == g
        if not bool(rows.any()):
            continue
        acc = torch.zeros((int(rows.sum()), n2), dtype=torch.float32, device=a.device)
        for q in range(k // 128):
            partial = (
                a32[rows, q * 128 : (q + 1) * 128]
                @ b32[g, :, q * 128 : (q + 1) * 128].T
            )
            scale = a_scale[rows, q].reshape(-1, 1) * b_scale[
                g, :, q
            ].repeat_interleave(128).reshape(1, n2)
            acc = acc + partial * scale
        out[rows] = acc
    return out.to(torch.bfloat16)


def _reference_activation(y_bf16):
    """FlashInfer chain arithmetic in torch: BF16 gate/up halves, FP32 SwiGLU, BF16 activation."""
    h = y_bf16.shape[1] // 2
    g = y_bf16[:, :h].float()
    u = y_bf16[:, h:].float()
    act = (g * torch.sigmoid(g) * u).to(torch.bfloat16).float()
    return g, u, act


def _reference_group_scales(act):
    m, h = act.shape
    absmax = act.reshape(m, h // GROUP_SIZE, GROUP_SIZE).abs().amax(dim=-1)
    return (absmax.clamp_min(EPS) / 448.0).contiguous()


def _e4m3_spacing(q):
    """Distance between adjacent E4M3 values at |q| (subnormal spacing 2**-9 below 2**-6)."""
    _, exponent = torch.frexp(q.float().abs().clamp_min(2.0**-6))
    return torch.pow(2.0, exponent.float() - 4.0)


def _assert_quantizes_reference(out_q, out_s, g, u, act):
    """Definition check against the torch reference, independent of any FlashInfer kernel.

    The reference GEMM accumulates in a different order, so its BF16 gate/up halves may differ
    from the kernel's by one BF16 ulp: the group scales move by at most 2**-7 relative and the
    activation by 2**-6 * |g| * |u| + 2**-7 * |act|.  Round-to-nearest quantization then places
    the dequantized value within half an E4M3 spacing of the activation, whichever side of a
    rounding boundary the kernel lands on.
    """
    m, h = out_q.shape
    ref_s = _reference_group_scales(act)
    assert torch.isfinite(out_s).all()
    torch.testing.assert_close(
        out_s, ref_s.reshape(out_s.shape), atol=0.0, rtol=2.0**-7
    )
    s = (
        out_s.reshape(m, h // GROUP_SIZE, 1)
        .expand(m, h // GROUP_SIZE, GROUP_SIZE)
        .reshape(m, h)
    )
    dequantized = out_q.float() * s
    assert torch.isfinite(dequantized).all()
    bound = (
        0.5 * _e4m3_spacing(out_q) * s
        + 2.0**-6 * g.abs() * u.abs()
        + 2.0**-7 * act.abs()
    )
    excess = (dequantized - act).abs() - bound
    violations = int((excess > 0).sum())
    assert violations == 0, (
        f"{violations} elements exceed the quantization bound "
        f"(max excess {float(excess.max()):.3e})"
    )


def _chain(a, b, a_scale, b_scale, m_indices):
    """The three-kernel FlashInfer chain the fused program replaces."""
    y = group_gemm_fp8_nt_groupwise_contiguous(a, b, a_scale, b_scale, m_indices)
    act = silu_and_mul(y)
    return per_token_group_quant_8bit(act, GROUP_SIZE, EPS, torch.float8_e4m3fn)


def _dequantize(q, s):
    m, h = q.shape
    return (
        q.float().reshape(m, h // GROUP_SIZE, GROUP_SIZE)
        * s.reshape(m, h // GROUP_SIZE, 1)
    ).reshape(m, h)


def _assert_matches(out_q, out_s, ref_q, ref_s):
    assert torch.isfinite(out_s).all()
    torch.testing.assert_close(out_s, ref_s.reshape(out_s.shape), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        _dequantize(out_q, out_s), _dequantize(ref_q, ref_s), atol=ATOL, rtol=RTOL
    )


# (group_counts, 2H, K, arbitrary_scales): 128-aligned routing incl. empties, odd block
# counts (dummy CTA), a partial final block, one block, and the two MoE perf shapes.
ROUTING_CASES = [
    pytest.param([128], 256, 512, False, id="one_block_min_shape"),
    pytest.param([128, 128], 256, 512, True, id="two_experts_one_block_each"),
    pytest.param([256], 512, 1024, False, id="one_pair"),
    pytest.param([384, 128, 0, 640], 512, 512, True, id="odd_blocks_with_empty"),
    pytest.param(
        [0, 256, 0, 0, 128, 100],
        256,
        1024,
        False,
        id="leading_internal_empty_partial_tail",
    ),
    pytest.param(
        [128, 0, 0, 256, 32], 768, 2048, True, id="consecutive_empty_partial_tail"
    ),
    pytest.param([0] * 16 + [96], 256, 512, False, id="groups_gt_blocks_partial_only"),
    pytest.param([256] * 16, 2048, 4096, False, id="wide_ep32_gate_up_uniform"),
    pytest.param(
        [256, 0, 512, 0, 384, 640, 128, 384, 256, 0, 128, 128, 384, 384, 512, 0],
        2048,
        4096,
        False,
        id="wide_ep32_gate_up_random_aligned",
    ),
    pytest.param(
        [384] * 8 + [128] * 8, 2048, 4096, False, id="wide_ep32_gate_up_all_odd"
    ),
    pytest.param([512] * 16, 2048, 4096, False, id="m8192_max"),
]


@pytest.mark.parametrize("group_counts,n2,k,arbitrary_scales", ROUTING_CASES)
def test_prepared_matches_torch_reference_chain(group_counts, n2, k, arbitrary_scales):
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts,
        n2,
        k,
        seed=662 + len(group_counts) + n2 + k,
        device=device,
        arbitrary_scales=arbitrary_scales,
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    )
    m = sum(group_counts)
    if m < SMALL_M_MAX:
        assert prepared.route == ACT_ROUTE
        assert prepared.num_kernels == 2
        assert prepared.gemm_backend == small_m_gemm_backend(m, n2, k)
        assert prepared.gemm_backend == (
            GEMM_BACKEND_CAKE if (m % 128 and k >= 1024) else GEMM_BACKEND_CUTE
        )
    else:
        assert prepared.route == FUSED_ROUTE
        assert prepared.num_kernels == 1
        assert prepared.gemm_backend is None
    out_q, out_s = prepared.launch()
    torch.cuda.synchronize()
    g, u, act = _reference_activation(
        _reference_gemm(a, b, a_scale, b_scale, m_indices)
    )
    _assert_quantizes_reference(out_q, out_s, g, u, act)
    # Contents may change between launches of the same prepared object.
    a.copy_(torch.randn(a.shape, device=device).to(torch.float8_e4m3fn))
    out_q2, out_s2 = prepared()
    torch.cuda.synchronize()
    assert (
        out_q2.data_ptr() == out_q.data_ptr() and out_s2.data_ptr() == out_s.data_ptr()
    )
    g, u, act = _reference_activation(
        _reference_gemm(a, b, a_scale, b_scale, m_indices)
    )
    _assert_quantizes_reference(out_q2, out_s2, g, u, act)


@pytest.mark.parametrize("group_counts,n2,k,arbitrary_scales", ROUTING_CASES)
def test_prepared_matches_flashinfer_chain(group_counts, n2, k, arbitrary_scales):
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts,
        n2,
        k,
        seed=663 + len(group_counts) + n2 + k,
        device=device,
        arbitrary_scales=arbitrary_scales,
    )
    try:
        chain_q, chain_s = _chain(a, b, a_scale, b_scale, m_indices)
    except (ValueError, ImportError, RuntimeError, NotImplementedError) as exc:
        pytest.skip(f"FlashInfer gate_up chain unavailable: {exc}")
    out_q, out_s = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices
    ).launch()
    torch.cuda.synchronize()
    _assert_matches(out_q, out_s, chain_q, chain_s)


@pytest.mark.parametrize(
    "group_counts,n2,k,expected_route",
    [
        pytest.param([256] * 16, 2048, 4096, FUSED_ROUTE, id="fused_wide"),
        pytest.param([256] * 8, 256, 512, FUSED_ROUTE, id="fused_at_threshold"),
        pytest.param([256] * 4 + [128, 100], 512, 1024, ACT_ROUTE, id="small_m"),
    ],
)
def test_launch_plan_matches_prepared(group_counts, n2, k, expected_route):
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts, n2, k, seed=664, device=device
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices
    )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    m = sum(group_counts)
    route, grid = launch_plan(m, n2, sm_count=sm_count)
    assert route == expected_route
    assert (prepared.route, prepared.grid) == (route, grid)
    if route == FUSED_ROUTE:
        assert grid[0] % 2 == 0 and grid[0] <= 128
    else:
        items = m * (n2 // 2 // GROUP_SIZE)
        assert 1 <= grid[0] <= max(1, -(-items // 4))
        assert grid[0] <= 16 * sm_count


@pytest.mark.parametrize(
    "group_counts,n2,k",
    [
        pytest.param([128, 128, 100], 256, 1024, id="small_m_partial_tail"),
        pytest.param([256, 256], 256, 512, id="small_m_aligned"),
        pytest.param([256] * 8, 256, 512, id="fused"),
    ],
)
def test_cuda_graph_replay_after_first_launch(group_counts, n2, k):
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts, n2, k, seed=665, device=device
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices
    )
    prepared.launch()  # initializes the private descriptor storage
    torch.cuda.synchronize()
    ref_q, ref_s = _chain(a, b, a_scale, b_scale, m_indices)
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        prepared.launch()
    prepared.out_q.view(torch.uint8).fill_(0xFF)
    prepared.out_s.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches(prepared.out_q, prepared.out_s, ref_q, ref_s)


def test_rejects_invalid_inputs():
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        [128, 128], 256, 512, seed=666, device=device
    )
    with pytest.raises(ValueError, match="K must be a positive multiple of 512"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            a[:, :384].contiguous(),
            b[:, :, :384].contiguous(),
            a_scale[:, :3].contiguous(),
            b_scale[:, :, :3].contiguous(),
            m_indices,
        )
    with pytest.raises(ValueError, match="2H must be a positive multiple of 256"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            a, b[:, :128].contiguous(), a_scale, b_scale[:, :1].contiguous(), m_indices
        )
    with pytest.raises(ValueError, match="a_scale must have shape"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            a, b, a_scale[:, :1].contiguous(), b_scale, m_indices
        )
    bad = m_indices.clone()
    bad[0] = 1
    with pytest.raises(ValueError, match="m_indices must be sorted"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            a, b, a_scale, b_scale, bad, validate_indices=True
        )
    unaligned = m_indices.clone()
    unaligned[100:] = 1  # boundary at row 100 is not a multiple of 128
    with pytest.raises(ValueError, match="multiple of 128 rows"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            a, b, a_scale, b_scale, unaligned, validate_indices=True
        )
    with pytest.raises(ValueError, match="M must be at most 8192"):
        big_a, big_b, big_as, big_bs, big_idx = _make_inputs(
            [8320], 256, 512, seed=667, device=device
        )
        prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            big_a, big_b, big_as, big_bs, big_idx
        )
