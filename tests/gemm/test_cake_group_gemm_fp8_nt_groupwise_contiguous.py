# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the generated SM100a contiguous grouped FP8 GEMM programs."""

import pytest
import torch

from flashinfer.gemm import (
    group_gemm_fp8_nt_groupwise_contiguous,
    prepare_group_gemm_fp8_nt_groupwise_contiguous,
)
from flashinfer.gemm.cake_grouped_fp8_gemm import (
    DEEPK_C2_SCALAR_OUTPUT_ROUTE,
    is_group_gemm_fp8_nt_groupwise_contiguous_prepared_available,
    launch_plan,
)

ATOL = RTOL = 3e-2


@pytest.fixture(autouse=True)
def _require_generated_program():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (10, 0):
        pytest.skip("generated contiguous grouped FP8 GEMM programs target SM100a only")
    if not is_group_gemm_fp8_nt_groupwise_contiguous_prepared_available(device):
        pytest.skip("no generated contiguous grouped FP8 GEMM program is registered")


def _make_inputs(group_counts, n, k, *, seed, device, arbitrary_scales=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    groups = len(group_counts)
    m = sum(group_counts)
    a = torch.randn((m, k), generator=generator, device=device).to(torch.float8_e4m3fn)
    b = torch.randn((groups, n, k), generator=generator, device=device).to(
        torch.float8_e4m3fn
    )
    if arbitrary_scales:
        a_scale = torch.rand((m, k // 128), generator=generator, device=device) + 0.25
        b_scale = (
            torch.rand((groups, n // 128, k // 128), generator=generator, device=device)
            + 0.25
        )
        a_scale[::3] *= -1.0
        b_scale[:, ::2] *= -1.0
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
                -8, 1, (groups, n // 128, k // 128), generator=generator, device=device
            ).float(),
        )
    counts = torch.tensor(group_counts, dtype=torch.int64, device=device)
    m_indices = torch.repeat_interleave(
        torch.arange(groups, dtype=torch.int32, device=device), counts
    )
    return a, b, a_scale.contiguous(), b_scale.contiguous(), m_indices.contiguous()


def _reference(a, b, a_scale, b_scale, m_indices):
    """K128 partial dots, post-dot scaling, ordered FP32 accumulation."""
    m, k = a.shape
    groups, n, _ = b.shape
    a32 = a.float()
    b32 = b.float()
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)
    for g in range(groups):
        rows = m_indices == g
        if not bool(rows.any()):
            continue
        acc = torch.zeros((int(rows.sum()), n), dtype=torch.float32, device=a.device)
        for q in range(k // 128):
            partial = (
                a32[rows, q * 128 : (q + 1) * 128]
                @ b32[g, :, q * 128 : (q + 1) * 128].T
            )
            scale = a_scale[rows, q].reshape(-1, 1) * b_scale[
                g, :, q
            ].repeat_interleave(128).reshape(1, n)
            acc = acc + partial * scale
        out[rows] = acc
    return out.to(torch.bfloat16)


def _assert_close(out, expected):
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), expected.float(), atol=ATOL, rtol=RTOL)


# (group_counts, N, K, arbitrary_scales): one row per host route plus boundary cases.
ROUTE_CASES = [
    pytest.param([128, 256, 128, 256], 256, 128, False, id="k128"),
    pytest.param([64], 256, 384, False, id="k384_tail64"),
    pytest.param([128], 128, 512, False, id="deepk_c2_exact_m128"),
    pytest.param([128, 1], 384, 1024, False, id="deepk_c2_128_1"),
    pytest.param([129], 256, 512, False, id="ab6_scale1_cross_m128"),
    pytest.param([255], 640, 640, True, id="kg1_tail255_k640"),
    pytest.param([128, 127], 256, 1152, True, id="kg1_final127_k1152"),
    pytest.param([0, 128, 1], 256, 384, False, id="leading_empty"),
    pytest.param([128, 128, 1, 0], 256, 1024, False, id="trailing_empty"),
    pytest.param([128, 0, 0, 256, 1, 0], 384, 4096, True, id="consecutive_empty_deep"),
    pytest.param([0] * 16 + [1], 128, 512, False, id="groups_gt_rows"),
    pytest.param([64, 1], 128, 128, True, id="unaligned_transition"),
]


@pytest.mark.parametrize("group_counts,n,k,arbitrary_scales", ROUTE_CASES)
def test_prepared_matches_reference(group_counts, n, k, arbitrary_scales):
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts,
        n,
        k,
        seed=4734 + len(group_counts) + n + k,
        device=device,
        arbitrary_scales=arbitrary_scales,
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    )
    out = prepared.launch()
    torch.cuda.synchronize()
    _assert_close(out, _reference(a, b, a_scale, b_scale, m_indices))
    # Contents may change between launches of the same prepared object.
    a.copy_(torch.randn(a.shape, device=device).to(torch.float8_e4m3fn))
    out2 = prepared()
    torch.cuda.synchronize()
    assert out2.data_ptr() == out.data_ptr()
    _assert_close(out2, _reference(a, b, a_scale, b_scale, m_indices))


@pytest.mark.parametrize("group_counts,n,k,arbitrary_scales", ROUTE_CASES)
def test_prepared_matches_cute_dsl(group_counts, n, k, arbitrary_scales):
    boundaries = torch.tensor(group_counts[:-1]).cumsum(0)
    if bool((boundaries % 128).any()):
        pytest.skip(
            "the CuTe-DSL kernel requires 128-aligned internal expert boundaries"
        )
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts,
        n,
        k,
        seed=4735 + len(group_counts) + n + k,
        device=device,
        arbitrary_scales=arbitrary_scales,
    )
    try:
        expected = group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_scale, b_scale, m_indices, validate_indices=True
        )
    except (ValueError, ImportError, RuntimeError) as exc:
        pytest.skip(f"CuTe-DSL contiguous grouped GEMM unavailable: {exc}")
    out = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices
    ).launch()
    torch.cuda.synchronize()
    _assert_close(out, expected)


def test_scalar_output_route_for_unaligned_output():
    device = torch.device("cuda")
    group_counts, n, k = [128, 64], 128, 128
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts, n, k, seed=4736, device=device, arbitrary_scales=True
    )
    m = sum(group_counts)
    storage = torch.empty(m * n + 8, dtype=torch.bfloat16, device=device)
    out = storage[1 : 1 + m * n].view(m, n)
    assert out.data_ptr() % 16 != 0 and out.is_contiguous()
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices, out=out
    )
    assert prepared.route == DEEPK_C2_SCALAR_OUTPUT_ROUTE
    result = prepared.launch()
    torch.cuda.synchronize()
    assert result.data_ptr() == out.data_ptr()
    _assert_close(out, _reference(a, b, a_scale, b_scale, m_indices))


@pytest.mark.parametrize(
    "groups,rows_per_group,n,k",
    [
        pytest.param(16, 256, 2048, 4096, id="wide_ep32_gate_up"),
        pytest.param(16, 256, 4096, 1024, id="wide_ep32_down"),
        pytest.param(64, 256, 2048, 4096, id="ep8_gate_up"),
        pytest.param(64, 256, 4096, 1024, id="ep8_down"),
        pytest.param(64, 256, 2048, 2048, id="n256_three_panel"),
    ],
)
def test_two_cta_routes_match_reference(groups, rows_per_group, n, k):
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        [rows_per_group] * groups, n, k, seed=4737 + n + k, device=device
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices
    )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    route, grid = launch_plan(
        groups * rows_per_group, n, k, sm_count=sm_count, scalar_output=False
    )
    assert (prepared.route, prepared.grid) == (route, grid)
    out = prepared.launch()
    torch.cuda.synchronize()
    _assert_close(out, _reference(a, b, a_scale, b_scale, m_indices))


def test_cuda_graph_replay_after_first_launch():
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        [128, 128], 256, 1024, seed=4738, device=device
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices
    )
    prepared.launch()  # initializes the private descriptor storage
    torch.cuda.synchronize()
    expected = _reference(a, b, a_scale, b_scale, m_indices)
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        prepared.launch()
    prepared.out.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(prepared.out, expected)


def test_rejects_invalid_inputs():
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        [128], 256, 512, seed=4739, device=device
    )
    with pytest.raises(ValueError, match="K must be a positive multiple of 128"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a[:, :200].contiguous(), b, a_scale, b_scale, m_indices
        )
    with pytest.raises(ValueError, match="a_scale must have shape"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_scale[:, :1].contiguous(), b_scale, m_indices
        )
    bad = m_indices.clone()
    bad[0] = 5
    with pytest.raises(ValueError, match="m_indices must"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_scale, b_scale, bad, validate_indices=True
        )
