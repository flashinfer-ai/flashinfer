# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the generated Blackwell contiguous grouped FP8 GEMM programs."""

import functools
import random

import pytest
import torch

from flashinfer.gemm import (
    group_gemm_fp8_nt_groupwise_contiguous,
    prepare_group_gemm_fp8_nt_groupwise_contiguous,
)
from flashinfer.gemm.cake_grouped_fp8_gemm import (
    BS_N128_ROUTE,
    BS_N128_RUN32_ROUTE,
    BS_N256_ROUTE,
    BS_N128_S16_ROUTE,
    BS_N256_RUN32_ROUTE,
    BS_N256_S16_ROUTE,
    BS_ROUTES,
    BS_STORE16_ROUTES,
    BS_SWAP_C1_ROUTE,
    BS_SWAP_CARRY_ROUTE,
    BS_SWAP_CARRY_S16_ROUTE,
    BS_SWAP_ROUTE,
    BS_SWAP_ROUTES,
    BS_SWAP_RP32_DENSE_ARCHS,
    BS_SWAP_RP32_ROUTE,
    BS_SWAP_RP32_S11_ROUTE,
    BS_SWAP_RUN32_ROUTE,
    BS_SWAP_S16_ROUTE,
    BS_SWAP_SM100_SMALL_ARCHS,
    BS_SWAP_SM100_SMALL_KEEP,
    BS_SWAP_SMALL_ARCHS,
    DEEPK_C2_AB6_SCALE1_DIRECT_STORE_ROUTE,
    DEEPK_C2_AB6_SCALE1_ROUTE,
    DEEPK_C2_KG1_DIRECT_STORE_ROUTE,
    DEEPK_C2_KG1_EXPERT_ALIGNED_ROUTE,
    DEEPK_C2_KG1_N64_ROUTE,
    DEEPK_C2_KG1_ROUTE,
    DEEPK_C2_ROUTE,
    DEEPK_C2_SCALAR_OUTPUT_ROUTE,
    DEEPK_CG2_AB7_BSCALE_PREFETCH_ROUTE,
    DEEPK_CG2_FOUR_LOAD_GRID128_ROUTE,
    DEEPK_CG2_FOUR_LOAD_ROUTE,
    DEEPK_CG2_RECURRENCE_ROUTE,
    DEEPK_N256_ROUTE,
    EXPERT_ALIGNED_MAX_GROUPS,
    K128_ROUTE,
    K384_ROUTE,
    _launch_plan,
    _route,
    _tile_starts,
    _validate_indices,
    bs_route_for_shape,
    bs_store16,
    bs_swap_route_for_shape,
    bs_swap_small_route,
    device_sm_count,
    is_group_gemm_fp8_nt_groupwise_contiguous_prepared_available,
    launch_plan,
    route_for_shape,
    route_geometry,
)
from flashinfer.jit.gemm.cake_grouped_fp8_gemm import (
    PROGRAMS,
    ROUTES,
    SUPPORTED_COMPUTE_CAPABILITIES,
    device_arch,
)

ATOL = RTOL = 3e-2


@pytest.fixture(autouse=True)
def _require_generated_program():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip(
            "generated contiguous grouped FP8 GEMM programs target SM100a and SM103a"
        )
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


def _random_token_counts(rng, total, groups):
    """Random per-expert token counts summing to ``total`` (empty experts allowed)."""
    cuts = sorted(rng.randint(0, total) for _ in range(groups - 1))
    bounds = [0, *cuts, total]
    return [bounds[i + 1] - bounds[i] for i in range(groups)]


# --- block-scaled family (packed UE8M0 scales, compact layout) -----------------


def _pack_ue8m0_mn_major(scale):
    """DeepGEMM's packed activation-scale layout: ``(rows, ceil(k/4))`` int32, stride ``(1, rows)``."""
    rows, k = scale.shape
    exponents = (scale.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    cols = -(-k // 4)
    padded = torch.zeros((rows, 4 * cols), dtype=torch.uint8, device=scale.device)
    padded[:, :k] = exponents
    packed = torch.empty((cols, rows), dtype=torch.int32, device=scale.device).mT
    packed.copy_(padded.view(torch.int32))
    return packed


def _pack_ue8m0_row_repeated(scale):
    """sglang's ``transform_scale_ue8m0`` weight layout: ``(G, N, ceil(k/4))`` int32, stride ``(N*cols, 1, N)``."""
    groups, n_blocks, k = scale.shape
    exponents = (
        (scale.contiguous().view(torch.int32) >> 23)
        .to(torch.uint8)
        .reshape(groups * n_blocks, k)
    )
    cols = -(-k // 4)
    padded = torch.zeros(
        (groups * n_blocks, 4 * cols), dtype=torch.uint8, device=scale.device
    )
    padded[:, :k] = exponents
    words = (
        padded.view(torch.int32)
        .view(groups, n_blocks, cols)
        .repeat_interleave(128, dim=1)
    )
    packed = torch.empty(
        (groups, cols, n_blocks * 128), dtype=torch.int32, device=scale.device
    ).permute(0, 2, 1)
    packed.copy_(words)
    return packed


def _compact_layout(group_counts, alignment, device, *, tail=True):
    """``m_indices`` of the compact MoE layout: each expert's rows then -1 padding to ``alignment``,
    plus the engine's ``G * (alignment - 1)`` tail rounded up to 128 rows."""
    pieces = []
    for g, count in enumerate(group_counts):
        padded = -(-count // alignment) * alignment
        if count:
            pieces.append(torch.full((count,), g, dtype=torch.int32))
        if padded > count:
            pieces.append(torch.full((padded - count,), -1, dtype=torch.int32))
    m_indices = torch.cat(pieces)
    total = int(m_indices.numel())
    if tail:
        total = max(total, sum(group_counts) + len(group_counts) * (alignment - 1))
    total = -(-total // 128) * 128
    if total > m_indices.numel():
        m_indices = torch.cat(
            [m_indices, torch.full((total - m_indices.numel(),), -1, dtype=torch.int32)]
        )
    return m_indices.to(device).contiguous()


def _make_block_scaled_inputs(
    group_counts, n, k, *, alignment, seed, device, weight_rows_repeated=True
):
    """FP8 operands with power-of-two scales delivered as packed UE8M0 words on the compact layout."""
    m_indices = _compact_layout(group_counts, alignment, device)
    m = int(m_indices.numel())
    groups = len(group_counts)
    generator = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((m, k), generator=generator, device=device).to(torch.float8_e4m3fn)
    b = torch.randn((groups, n, k), generator=generator, device=device).to(
        torch.float8_e4m3fn
    )
    a_scale = torch.pow(
        2.0,
        torch.randint(-8, 1, (m, k // 128), generator=generator, device=device).float(),
    )
    b_scale = torch.pow(
        2.0,
        torch.randint(
            -8, 1, (groups, n // 128, k // 128), generator=generator, device=device
        ).float(),
    )
    a_packed = _pack_ue8m0_mn_major(a_scale)
    if weight_rows_repeated:
        b_packed = _pack_ue8m0_row_repeated(b_scale)
    else:
        b_packed = (
            _pack_ue8m0_mn_major(b_scale.reshape(-1, k // 128))
            .contiguous()
            .view(groups, n // 128, -1)
        )
    return a, b, a_scale, b_scale, a_packed, b_packed, m_indices


def _assert_block_scaled(out, expected, m_indices):
    """Routed rows match the reference; the output rows of ``-1`` entries are unspecified.

    A route writes each expert run's routed rows rounded up to 32 rows (16 on the
    store16 programs): the other padding rows stay untouched, the rest hold values
    of no meaning (DeepGEMM does the same per 128-row block).
    """
    valid = m_indices >= 0
    _assert_close(out[valid], expected[valid])


# (group_counts, N, K, arbitrary_scales): small deterministic rows covering the
# K classes, empty experts, partial blocks and unaligned expert boundaries.
SHAPE_CASES = [
    pytest.param([128, 256, 128, 256], 256, 128, False, id="k128"),
    pytest.param([64], 256, 384, False, id="k384_tail64"),
    pytest.param([128], 128, 512, False, id="exact_m128"),
    pytest.param([128, 1], 384, 1024, False, id="m128_plus_1"),
    pytest.param([129], 256, 512, False, id="cross_m128"),
    pytest.param([255], 640, 640, True, id="tail255_k640"),
    pytest.param([128, 127], 256, 1152, True, id="final127_k1152"),
    pytest.param([0, 128, 1], 256, 384, False, id="leading_empty"),
    pytest.param([128, 128, 1, 0], 256, 1024, False, id="trailing_empty"),
    pytest.param([128, 0, 0, 256, 1, 0], 384, 4096, True, id="consecutive_empty_deep"),
    pytest.param([0] * 16 + [1], 128, 512, False, id="groups_gt_rows"),
    pytest.param([64, 1], 128, 128, True, id="unaligned_transition"),
]

# Random token counts per expert (not 128-aligned, zeros allowed) around the
# token counts the host once treated as exact shapes (4096, 16384) and at
# arbitrary totals, for the K classes the routes distinguish.
RANDOM_TOKEN_CASES = [
    pytest.param(seed, total, groups, n, k, id=f"s{seed}_m{total}_g{groups}_n{n}_k{k}")
    for seed, total, groups, n, k in (
        (1, 4095, 16, 2048, 4096),
        (2, 4096, 16, 2048, 4096),
        (3, 4097, 16, 2048, 4096),
        (4, 4096, 16, 4096, 1024),
        (5, 4093, 16, 4096, 1024),
        (6, 16384, 64, 2048, 4096),
        (7, 16385, 64, 4096, 1024),
        (8, 16383, 64, 2048, 2048),
        (9, 777, 8, 256, 512),
        (10, 3001, 12, 512, 128),
        (11, 1234, 4, 384, 384),
        (12, 5000, 32, 1024, 640),
    )
]


@pytest.mark.parametrize("group_counts,n,k,arbitrary_scales", SHAPE_CASES)
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


@pytest.mark.parametrize("seed,total,groups,n,k", RANDOM_TOKEN_CASES)
def test_random_token_counts_match_reference(seed, total, groups, n, k):
    device = torch.device("cuda")
    rng = random.Random(seed)
    group_counts = _random_token_counts(rng, total, groups)
    assert sum(group_counts) == total
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts,
        n,
        k,
        seed=4740 + seed,
        device=device,
        arbitrary_scales=bool(seed % 2),
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    )
    assert prepared.num_ctas >= 1
    out = prepared.launch()
    torch.cuda.synchronize()
    _assert_close(out, _reference(a, b, a_scale, b_scale, m_indices))


@pytest.mark.parametrize("group_counts,n,k,arbitrary_scales", SHAPE_CASES)
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
    except (ValueError, ImportError, RuntimeError, NotImplementedError) as exc:
        pytest.skip(f"CuTe-DSL contiguous grouped GEMM unavailable: {exc}")
    out = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices
    ).launch()
    torch.cuda.synchronize()
    _assert_close(out, expected)


def test_unaligned_output_is_written_in_place():
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
        pytest.param(512, 256, 256, 4096, id="tp8_gate_up"),
        pytest.param(512, 256, 4096, 128, id="tp8_down"),
    ],
)
def test_moe_shapes_match_reference(groups, rows_per_group, n, k):
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        [rows_per_group] * groups, n, k, seed=4737 + n + k, device=device
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices
    )
    out = prepared.launch()
    torch.cuda.synchronize()
    _assert_close(out, _reference(a, b, a_scale, b_scale, m_indices))


def test_cuda_graph_capture_of_first_launch():
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        [128, 128], 256, 1024, seed=4738, device=device
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices
    )
    expected = _reference(a, b, a_scale, b_scale, m_indices)
    # Tensor maps travel by value: the very first launch is captured, no eager warm-up launch.
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        prepared.launch()
    prepared.out.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(prepared.out, expected)
    # New contents, same graph.
    a.copy_(torch.randn(a.shape, device=device).to(torch.float8_e4m3fn))
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(prepared.out, _reference(a, b, a_scale, b_scale, m_indices))


def test_first_launch_after_queued_frees_on_a_busy_side_stream():
    """Regression for the descriptor-upload race (sglang moe_fp8_grouped route).

    An earlier runner uploaded its tensor maps with a synchronous host-to-device
    copy into a caching-allocator block whose previous tenant's kernels were
    still queued on the stream; the first launch then read overwritten
    descriptors.  Tensor maps now travel by value, so a first launch (eager or
    graph-captured) right after queued work and frees on a non-default stream
    must match the clean result bitwise.
    """
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        [512] * 8, 256, 1024, seed=4746, device=device
    )
    clean = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices
    ).launch()
    torch.cuda.synchronize()
    clean = clean.clone()
    x = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
    side = torch.cuda.Stream(device=device)
    for _ in range(4):
        with torch.cuda.stream(side):
            junk = [
                torch.empty(384, dtype=torch.uint8, device=device) for _ in range(8)
            ]
            for _ in range(8):
                y = x @ x  # noqa: F841 queued work
            for j in junk:
                j.fill_(255)
            del junk  # blocks return to the allocator with their fills still queued
            prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
                a, b, a_scale, b_scale, m_indices
            )
            out = prepared.launch()  # first launch, nothing synchronized before it
            torch.cuda.synchronize()
            assert torch.equal(out, clean)
            graph = torch.cuda.CUDAGraph()
            captured = prepare_group_gemm_fp8_nt_groupwise_contiguous(
                a, b, a_scale, b_scale, m_indices
            )
            with torch.cuda.graph(graph, stream=side):
                captured.launch()
            captured.out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured.out, clean)
            del graph, prepared, captured


def test_re_preparing_per_step_retains_no_device_memory():
    """A new token count per step re-prepares the launch; nothing accumulates."""
    device = torch.device("cuda")
    rng = random.Random(4741)
    n, k, groups = 256, 512, 4
    b = torch.randn((groups, n, k), device=device).to(torch.float8_e4m3fn)
    b_scale = torch.ones((groups, n // 128, k // 128), device=device)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    for _ in range(16):
        total = rng.randint(1, 1024)
        counts = _random_token_counts(rng, total, groups)
        a = torch.randn((total, k), device=device).to(torch.float8_e4m3fn)
        a_scale = torch.ones((total, k // 128), device=device)
        m_indices = torch.repeat_interleave(
            torch.arange(groups, dtype=torch.int32, device=device),
            torch.tensor(counts, dtype=torch.int64, device=device),
        ).contiguous()
        prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_scale, b_scale, m_indices
        )
        out = prepared.launch()
        torch.cuda.synchronize()
        _assert_close(out, _reference(a, b, a_scale, b_scale, m_indices))
        del prepared, out, a, a_scale, m_indices
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(device) == baseline


def test_release_prepared_operands_frees_token_buffers():
    """A plan kept per layer must not pin the dispatcher's buffers: after
    ``release_prepared_operands`` only the weights stay referenced, a bare
    ``launch()`` is refused and a fully rebinding launch still works."""
    device = torch.device("cuda")
    counts = [100, 28, 128]
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        counts, 256, 1024, seed=4746, device=device
    )
    expected = _reference(a, b, a_scale, b_scale, m_indices)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated(device)
    # Token operands allocated after the baseline: they must be freed by the release.
    a2, a_scale2, m_indices2 = a.clone(), a_scale.clone(), m_indices.clone()
    out = torch.empty((a.shape[0], 256), dtype=torch.bfloat16, device=device)
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a2, b, a_scale2, b_scale, m_indices2, out=out
    )
    prepared.release_prepared_operands()
    with pytest.raises(ValueError, match="released"):
        prepared.launch()
    with pytest.raises(ValueError, match="released"):
        _ = prepared.out
    del a2, a_scale2, m_indices2, out
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(device) == baseline
    out = torch.empty((a.shape[0], 256), dtype=torch.bfloat16, device=device)
    result = prepared.launch(a=a, a_scale=a_scale, m_indices=m_indices, out=out)
    assert result is out
    torch.cuda.synchronize()
    _assert_close(out, expected)


def test_launch_rebinds_per_call_operands():
    """One prepared object per layer: the token operands are swapped per call without copies."""
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        [200, 56, 128], 256, 1024, seed=4743, device=device
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices
    )
    _assert_close(prepared.launch(), _reference(a, b, a_scale, b_scale, m_indices))
    a2, _, a_scale2, _, _ = _make_inputs(
        [200, 56, 128], 256, 1024, seed=4744, device=device
    )
    m_indices2 = torch.repeat_interleave(
        torch.arange(3, dtype=torch.int32, device=device),
        torch.tensor([64, 256, 64], dtype=torch.int64, device=device),
    ).contiguous()
    out2 = torch.empty_like(prepared.out)
    result = prepared.launch(a=a2, a_scale=a_scale2, m_indices=m_indices2, out=out2)
    assert result is out2
    torch.cuda.synchronize()
    _assert_close(out2, _reference(a2, b, a_scale2, b_scale, m_indices2))
    # The prepared operands are untouched and still bound.
    _assert_close(prepared.launch(), _reference(a, b, a_scale, b_scale, m_indices))
    with pytest.raises(ValueError, match="a must have shape"):
        prepared.launch(a=a2[:128].contiguous())
    with pytest.raises(ValueError, match="alignment class"):
        storage = torch.empty(
            prepared.m * prepared.n + 8, dtype=torch.bfloat16, device=device
        )
        prepared.launch(
            out=storage[1 : 1 + prepared.m * prepared.n].view(prepared.m, prepared.n)
        )


# --- validated layout: the expert-aligned route (a speed choice only) ---------


def _expert_aligned_registered(device):
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    program = ROUTES.get(DEEPK_C2_KG1_EXPERT_ALIGNED_ROUTE)
    return program is not None and arch in PROGRAMS[program]["arches"]


def _layout(group_counts, tile_m):
    """Host reference of the validated layout: (expert runs starting at a row that
    is not a multiple of ``tile_m``, expert runs starting after row 0, 128-row
    tiles when no tile spans two experts, the largest offset of such an inner
    start within its tile)."""
    m = sum(group_counts)
    starts, row = set(), 0
    for count in group_counts[:-1]:
        row += count
        if 0 < row < m:
            starts.add(row)
    offsets = [row % tile_m for row in starts if row % tile_m]
    tiles = sum((count + 127) // 128 for count in group_counts)
    return len(offsets), len(starts), tiles, max(offsets, default=0)


def _aligned(group_counts, tile_m=256):
    """Host reference of the validated alignment: every non-empty expert starts
    on a multiple of ``tile_m`` rows."""
    row = 0
    for count in group_counts:
        if count and row % tile_m:
            return False
        row += count
    return True


def _tile_starts_ref(group_counts):
    """Host reference of the per-tile expert starts: for 128- and 256-row M tiles,
    how many non-empty experts start inside each tile, off its first row."""
    m = sum(group_counts)
    out = {}
    for tile_m in (128, 256):
        per_tile, row = [0] * -(-m // tile_m), 0
        for count in group_counts:
            if count and row % tile_m:
                per_tile[row // tile_m] += 1
            row += count
        out[tile_m] = tuple(per_tile)
    return out


def _validated_plan(group_counts, n, k, sm_count):
    """The shape route's plan once the layout alignment is known (no expert-aligned
    program)."""
    return _launch_plan(
        sum(group_counts),
        n,
        k,
        sm_count=sm_count,
        scalar_output=False,
        block_scaled=False,
        alignment=128,
        num_groups=len(group_counts),
        arch=None,
        aligned=_aligned(group_counts),
        tile_starts=_tile_starts_ref(group_counts),
    )


LAYOUT_CASES = [
    [128, 256, 128, 256],
    [64, 1],
    [0, 128, 1],
    [129],
    [1],
    [0] * 16 + [1],
    [128, 0, 0, 256, 1, 0],
    [200, 56, 128],
    [256, 256, 128, 384],
    *(_random_token_counts(random.Random(seed), 3001, 12) for seed in range(4)),
]


@pytest.mark.parametrize("group_counts", LAYOUT_CASES)
@pytest.mark.parametrize("tile_m", [128, 256])
def test_validated_layout_counts_runs_and_tiles(group_counts, tile_m):
    device = torch.device("cuda")
    counts = torch.tensor(group_counts, dtype=torch.int64, device=device)
    m_indices = torch.repeat_interleave(
        torch.arange(len(group_counts), dtype=torch.int32, device=device), counts
    ).contiguous()
    assert _validate_indices(m_indices, len(group_counts)) == (None, None)
    layout, aligned = _validate_indices(
        m_indices, len(group_counts), layout_tile_m=tile_m
    )
    assert layout == _layout(group_counts, tile_m) and aligned is None
    assert _validate_indices(
        m_indices, len(group_counts), layout_tile_m=tile_m, read_alignment=True
    ) == (layout, _aligned(group_counts))
    assert _tile_starts(m_indices) == _tile_starts_ref(group_counts)


ROUTING_CASES = [
    pytest.param([128, 256, 128, 256], 256, 128, id="aligned_k128"),
    pytest.param([64, 1], 128, 128, id="unaligned_k128"),
    pytest.param([256] * 16, 2048, 4096, id="aligned_cg2"),
    pytest.param([257] + [256] * 14 + [255], 2048, 4096, id="unaligned_cg2"),
    pytest.param([256] * 7 + [257, 255] + [256] * 7, 2048, 4096, id="one_inner_cg2"),
    pytest.param([384, 128] * 8, 2048, 4096, id="aligned128_cg2"),
    pytest.param([129, 255, 128, 256], 256, 128, id="one_inner_of_three_k128"),
    pytest.param([200, 56, 128], 256, 1024, id="unaligned_ab6"),
    pytest.param([128, 127], 256, 1152, id="aligned_kg1"),
    pytest.param([60, 73, 711, 292, 620, 138, 1107], 512, 640, id="unaligned_kg1"),
    pytest.param([129, 128, 128, 128], 256, 128, id="one_row_in_k128"),
    pytest.param([192, 320] * 8, 2048, 4096, id="half_inner_cg2"),
    pytest.param(
        _random_token_counts(random.Random(4790), 4096, 16), 2048, 4096, id="random_cg2"
    ),
    pytest.param([100, 100, 100, 212], 1024, 512, id="inner_k512"),
]


@pytest.mark.parametrize("group_counts,n,k", ROUTING_CASES)
def test_validated_layout_selects_the_expert_aligned_route(group_counts, n, k):
    """Without ``validate_indices`` the route is the shape rule's; with it, a layout in
    which at least 90% of the expert runs start inside a tile of that route, one of them
    at least half a tile in, selects the expert-aligned program (when one is
    registered); every other layout keeps the shape rule's route and grid for its
    256-row alignment."""
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts, n, k, seed=4750 + len(group_counts) + n + k, device=device
    )
    m = sum(group_counts)
    sm_count = device_sm_count(torch.cuda.current_device())
    shape_plan = launch_plan(m, n, k, sm_count=sm_count, scalar_output=False)
    plain = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices
    )
    assert (plain.route, plain.grid) == shape_plan
    validated = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    )
    tile_m = route_geometry(route_for_shape(m, n, k))[0]
    inner, starts, tiles, max_offset = _layout(group_counts, tile_m)
    preferred = inner and 10 * inner >= 9 * starts and 2 * max_offset >= tile_m
    if preferred and _expert_aligned_registered(device):
        assert validated.route == DEEPK_C2_KG1_EXPERT_ALIGNED_ROUTE
        assert validated.grid == (min(tiles * (n // 128), sm_count), 1, 1)
    else:
        assert (validated.route, validated.grid) == _validated_plan(
            group_counts, n, k, sm_count
        )
    out = validated.launch()
    torch.cuda.synchronize()
    assert torch.equal(out, plain.launch())
    _assert_close(out, _reference(a, b, a_scale, b_scale, m_indices))


def test_validated_layout_keeps_the_shape_route_outside_the_program_contract():
    """More experts than the expert-aligned program holds, padding rows and a
    scalar-store output keep the shape rule's route."""
    device = torch.device("cuda")
    groups = EXPERT_ALIGNED_MAX_GROUPS + 1
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        [1] * groups, 128, 512, seed=4760, device=device
    )
    sm_count = device_sm_count(torch.cuda.current_device())
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    )
    assert (prepared.route, prepared.grid) == launch_plan(
        groups, 128, 512, sm_count=sm_count, scalar_output=False
    )
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        [64, 1], 128, 512, seed=4761, device=device
    )
    padded = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices, validate_indices=True, fill_padding=True
    )
    assert padded.route == route_for_shape(65, 128, 512)
    storage = torch.empty(65 * 128 + 8, dtype=torch.bfloat16, device=device)
    scalar = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a,
        b,
        a_scale,
        b_scale,
        m_indices,
        storage[1 : 1 + 65 * 128].view(65, 128),
        validate_indices=True,
    )
    assert scalar.route == DEEPK_C2_SCALAR_OUTPUT_ROUTE


@pytest.mark.parametrize("k", [128, 1024])
def test_route_choice_keeps_results_when_indices_change(k):
    """The route chosen from the prepared layout is a speed choice: rebinding
    ``m_indices`` to another layout, or rewriting it before a graph replay, gives
    the bits of a launch prepared for the new layout."""
    device = torch.device("cuda")
    aligned, unaligned = [128, 256, 0, 128], [200, 120, 72, 120]
    a, b, a_scale, b_scale, idx_aligned = _make_inputs(
        aligned, 256, k, seed=4770 + k, device=device, arbitrary_scales=True
    )
    idx_unaligned = torch.repeat_interleave(
        torch.arange(4, dtype=torch.int32, device=device),
        torch.tensor(unaligned, dtype=torch.int64, device=device),
    ).contiguous()
    # Expected bits: the shape rule's route, prepared without the layout read.
    expect_aligned = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, idx_aligned
    ).launch()
    expect_unaligned = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, idx_unaligned
    ).launch()
    torch.cuda.synchronize()
    _assert_close(expect_unaligned, _reference(a, b, a_scale, b_scale, idx_unaligned))
    for prepared_layout, other, expect_other, expect_own in (
        (idx_aligned, idx_unaligned, expect_unaligned, expect_aligned),
        (idx_unaligned, idx_aligned, expect_aligned, expect_unaligned),
    ):
        prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_scale, b_scale, prepared_layout.clone(), validate_indices=True
        )
        out = torch.empty_like(expect_own)
        prepared.launch(m_indices=other, out=out)
        torch.cuda.synchronize()
        assert torch.equal(out, expect_other), prepared.route
        indices = prepared_layout.clone()
        replay_out = torch.empty_like(expect_own)
        prepared.launch(m_indices=indices, out=replay_out)  # warm up outside capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prepared.launch(m_indices=indices, out=replay_out)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(replay_out, expect_own), prepared.route
        indices.copy_(other)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(replay_out, expect_other), prepared.route


def test_fill_padding_accepts_compact_layout():
    """-1 padding rows (compact MoE layout) are forward-filled onto the preceding expert."""
    device = torch.device("cuda")
    counts = [100, 128, 5]
    padded = [128, 128, 128]
    a, b, a_scale, b_scale, _ = _make_inputs(padded, 256, 512, seed=4745, device=device)
    blocks = []
    for g, (c, p) in enumerate(zip(counts, padded, strict=True)):
        blocks.append(torch.full((c,), g, dtype=torch.int32))
        blocks.append(torch.full((p - c,), -1, dtype=torch.int32))
    m_indices = torch.cat(blocks).to(device).contiguous()
    with pytest.raises(ValueError, match="fill_padding"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_scale, b_scale, m_indices, validate_indices=True
        )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices, validate_indices=True, fill_padding=True
    )
    filled = torch.clamp(torch.cummax(m_indices, 0).values, min=0)
    expected = _reference(a, b, a_scale, b_scale, filled)
    valid = m_indices >= 0
    out = prepared.launch()
    torch.cuda.synchronize()
    _assert_close(out[valid], expected[valid])
    # Graph capture includes the two forward-fill kernels.
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        prepared.launch()
    out.fill_(float("nan"))
    m_indices.copy_(
        torch.cat([torch.full((128,), 0), torch.full((256,), 2)]).to(torch.int32)
    )
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(out, _reference(a, b, a_scale, b_scale, m_indices))


BLOCK_SCALED_CASES = [
    # (group_counts, N, K, alignment): the compact layout at the engine's 128-row alignment
    # (single-run schedule) and the 32/64/96-row alignments (multi-run schedule).
    ([100, 0, 3, 128, 300, 1], 256, 512, 128),
    ([257, 640, 3, 0, 128], 1024, 2048, 128),
    ([5, 33, 0, 97, 32, 1, 129], 4096, 256, 128),
    ([100, 0, 3, 69, 128, 200, 1, 64], 256, 512, 64),
    ([5, 33, 0, 0, 97, 32, 1, 129], 1024, 256, 32),
    ([95, 97, 2, 193, 0, 100], 384, 128, 96),
]


@pytest.mark.parametrize("group_counts,n,k,alignment", BLOCK_SCALED_CASES)
def test_block_scaled_packed_scales_match_reference(group_counts, n, k, alignment):
    """Packed UE8M0 scales select the block-scaled family: native -1 padding, in-place packed layouts."""
    device = torch.device("cuda")
    a, b, a_scale, b_scale, a_packed, b_packed, m_indices = _make_block_scaled_inputs(
        group_counts, n, k, alignment=alignment, seed=4750 + alignment, device=device
    )
    expected = _reference(a, b, a_scale, b_scale, m_indices)
    out = torch.full(
        (int(m_indices.numel()), n), float("nan"), dtype=torch.bfloat16, device=device
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a,
        b,
        a_packed,
        b_packed,
        m_indices,
        out=out,
        validate_indices=True,
        alignment=alignment,
    )
    assert prepared.route in BS_ROUTES | BS_SWAP_ROUTES
    # The small-block programs hold one group's rows per 32-row block, so they
    # serve the single-run and the multi-run layouts; the rule names the program.
    small = bs_swap_small_route(
        int(m_indices.numel()),
        n,
        k,
        alignment=alignment,
        arch=device_arch(torch.cuda.current_device()),
        sm_count=device_sm_count(torch.cuda.current_device()),
    )
    if small is not None:
        assert prepared.route == small
    else:
        assert (
            prepared.route
            in (
                BS_N128_RUN32_ROUTE,
                BS_N256_RUN32_ROUTE,
                BS_SWAP_RUN32_ROUTE,
                BS_SWAP_RP32_ROUTE,
            )
        ) == (alignment % 128 != 0)
    assert (prepared.route in BS_STORE16_ROUTES.values()) == bs_store16(
        int(m_indices.numel()), alignment=alignment, num_groups=len(group_counts)
    )
    prepared.launch()
    torch.cuda.synchronize()
    _assert_block_scaled(out, expected, m_indices)


def test_block_scaled_accepts_block_rows_weight_scales():
    """Weight scales may be ``(G, N//128, cols)`` instead of the row-repeated ``(G, N, cols)``."""
    device = torch.device("cuda")
    a, b, a_scale, b_scale, a_packed, b_packed, m_indices = _make_block_scaled_inputs(
        [130, 64, 0, 200],
        512,
        1024,
        alignment=128,
        seed=4760,
        device=device,
        weight_rows_repeated=False,
    )
    assert tuple(b_packed.shape) == (4, 4, 2)
    out = torch.full(
        (int(m_indices.numel()), 512), float("nan"), dtype=torch.bfloat16, device=device
    )
    prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_packed, b_packed, m_indices, out=out
    ).launch()
    torch.cuda.synchronize()
    _assert_block_scaled(out, _reference(a, b, a_scale, b_scale, m_indices), m_indices)


def test_block_scaled_launch_rebinds_token_operands_and_captures():
    """Per-call rebinding of a / a_scale / m_indices / out (same packed layout) and CUDA-graph capture.

    K=1024 gives two packed scale columns, so a row-major copy of the MN-major scales is a
    genuinely different layout (at K<=512 the single column makes every stride contiguous).
    """
    device = torch.device("cuda")
    counts = [128, 100, 0, 256]
    a, b, a_scale, b_scale, a_packed, b_packed, m_indices = _make_block_scaled_inputs(
        counts, 1024, 1024, alignment=128, seed=4761, device=device
    )
    m = int(m_indices.numel())
    out = torch.full((m, 1024), float("nan"), dtype=torch.bfloat16, device=device)
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_packed, b_packed, m_indices, out=out
    )
    prepared.launch()
    torch.cuda.synchronize()
    _assert_block_scaled(out, _reference(a, b, a_scale, b_scale, m_indices), m_indices)
    a2, _, a_scale2, _, a_packed2, _, _ = _make_block_scaled_inputs(
        counts, 1024, 1024, alignment=128, seed=4762, device=device
    )
    m_indices2 = _compact_layout([64, 200, 5, 128], 128, device)
    assert int(m_indices2.numel()) == m
    out2 = torch.full_like(out, float("nan"))
    result = prepared.launch(a=a2, a_scale=a_packed2, m_indices=m_indices2, out=out2)
    assert result is out2
    torch.cuda.synchronize()
    _assert_block_scaled(
        out2, _reference(a2, b, a_scale2, b_scale, m_indices2), m_indices2
    )
    with pytest.raises(ValueError, match="packed layout"):
        prepared.launch(a_scale=a_packed2.contiguous())
    # First launch of a fresh prepared object inside a CUDA graph on a side stream.
    out3 = torch.full_like(out, float("nan"))
    fresh = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a2, b, a_packed2, b_packed, m_indices2, out=out3
    )
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        fresh.launch()
    torch.cuda.synchronize()
    out3.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_block_scaled(
        out3, _reference(a2, b, a_scale2, b_scale, m_indices2), m_indices2
    )


def test_block_scaled_route_rule():
    """BLOCK_N 256 unless a 256-column grid underfills the device or K is long with under three waves of M blocks (except single-run below 32 K blocks and two waves, and multi-run rows that fill a wave of 128-column tiles); run32 below 128-row alignment."""
    assert (
        bs_route_for_shape(48896, 1024, 2048, alignment=128, sm_count=152)
        == BS_N128_ROUTE
    )
    assert (
        bs_route_for_shape(48896, 1024, 2048, alignment=128, sm_count=148)
        == BS_N128_ROUTE
    )
    assert (
        bs_route_for_shape(36608, 1024, 2048, alignment=128, sm_count=148)
        == BS_N256_ROUTE
    )
    assert (
        bs_route_for_shape(16256, 512, 4096, alignment=32, sm_count=148, num_groups=512)
        == BS_N256_RUN32_ROUTE
    )
    assert (
        bs_route_for_shape(16256, 512, 4096, alignment=32, sm_count=148)
        == BS_N128_RUN32_ROUTE
    )
    assert (
        bs_route_for_shape(16064, 512, 4096, alignment=32, sm_count=148, num_groups=512)
        == BS_N128_RUN32_ROUTE
    )
    assert (
        bs_route_for_shape(32640, 1024, 2048, alignment=128, sm_count=148)
        == BS_N256_ROUTE
    )
    assert (
        bs_route_for_shape(65152, 512, 4096, alignment=128, sm_count=152)
        == BS_N256_ROUTE
    )
    assert (
        bs_route_for_shape(16256, 512, 4096, alignment=128, sm_count=152)
        == BS_N128_ROUTE
    )
    assert (
        bs_route_for_shape(1664, 256, 4096, alignment=128, sm_count=152)
        == BS_N128_ROUTE
    )
    assert (
        bs_route_for_shape(1152, 256, 512, alignment=64, sm_count=148)
        == BS_N128_RUN32_ROUTE
    )
    assert (
        bs_route_for_shape(8064, 1024, 2048, alignment=32, sm_count=152)
        == BS_N128_RUN32_ROUTE
    )
    assert (
        bs_route_for_shape(65280, 1024, 2048, alignment=128, sm_count=152)
        == BS_N256_ROUTE
    )
    assert (
        bs_route_for_shape(48896, 2048, 512, alignment=128, sm_count=152)
        == BS_N256_ROUTE
    )
    assert (
        bs_route_for_shape(48896, 384, 512, alignment=128, sm_count=152)
        == BS_N128_ROUTE
    )
    assert (
        bs_route_for_shape(32512, 1024, 2048, alignment=64, sm_count=148)
        == BS_N128_RUN32_ROUTE
    )
    assert (
        bs_route_for_shape(16256, 4096, 256, alignment=32, sm_count=148)
        == BS_N256_RUN32_ROUTE
    )
    for route in BS_ROUTES | BS_SWAP_ROUTES:
        assert route in ROUTES or any(
            route in table for table in ROUTES.values() if isinstance(table, dict)
        )
    with pytest.raises(ValueError, match="multiple of 32"):
        bs_route_for_shape(4096, 1024, 2048, alignment=48, sm_count=148)


@pytest.mark.parametrize(
    ("m", "n", "k", "num_groups", "alignment", "sm_100a", "sm_103a"),
    [
        # N not a multiple of 256: the non-swapped routes.
        (1664, 384, 128, 8, 128, None, None),
        # 32-row multi-run layouts swap.
        (16000, 4096, 256, 512, 32, BS_SWAP_RUN32_ROUTE, BS_SWAP_RUN32_ROUTE),
        # K <= 512: 2 <= rows <= 20, and 1 <= rows < 2 at K <= 256.
        (34560, 2048, 512, 256, 128, BS_SWAP_ROUTE, BS_SWAP_ROUTE),
        (48896, 2048, 512, 256, 128, None, None),
        (65664, 4096, 256, 512, 128, BS_SWAP_CARRY_ROUTE, BS_SWAP_CARRY_ROUTE),
        (65408, 4096, 256, 512, 128, None, None),
        # Longer K: keep below 0.75 rows, swap below 100 rows.
        (32640, 1024, 2048, 256, 128, None, None),
        (65408, 512, 4096, 512, 128, BS_SWAP_ROUTE, BS_SWAP_ROUTE),
        # From 100 rows the rule depends on the architecture.
        (65280, 1024, 2048, 256, 128, BS_SWAP_CARRY_ROUTE, None),
        (146944, 512, 4096, 512, 128, None, BS_SWAP_ROUTE),
        (163584, 1024, 2048, 256, 128, BS_SWAP_CARRY_ROUTE, BS_SWAP_CARRY_ROUTE),
        # sm_100a also swaps long K when the BLOCK_N 128 grid is at most 74 tiles.
        (1664, 256, 4096, 5, 128, BS_SWAP_ROUTE, BS_SWAP_ROUTE),
        (4736, 256, 4096, 5, 128, BS_SWAP_ROUTE, BS_SWAP_ROUTE),
        (4864, 256, 4096, 5, 128, None, BS_SWAP_ROUTE),
        (1664, 256, 1536, 5, 128, None, BS_SWAP_CARRY_ROUTE),
    ],
)
def test_block_scaled_swap_route_rule(m, n, k, num_groups, alignment, sm_100a, sm_103a):
    """Swap-AB routes from (m, n, k, num_groups, alignment, arch) only."""
    for arch, expected in (("sm_100a", sm_100a), ("sm_103a", sm_103a)):
        assert (
            bs_swap_route_for_shape(
                m, n, k, alignment=alignment, num_groups=num_groups, arch=arch
            )
            == expected
        )


def test_block_scaled_swap_route_needs_groups_and_arch():
    """Single-run layouts need the group count; other architectures never swap."""
    assert (
        bs_swap_route_for_shape(
            65280, 1024, 2048, alignment=128, num_groups=None, arch="sm_100a"
        )
        is None
    )
    assert (
        bs_swap_route_for_shape(
            16000, 4096, 256, alignment=32, num_groups=None, arch="sm_100a"
        )
        == BS_SWAP_RUN32_ROUTE
    )
    assert (
        bs_swap_route_for_shape(
            65280, 1024, 2048, alignment=128, num_groups=256, arch="sm_120a"
        )
        is None
    )


def test_block_scaled_swap_rp32_rule():
    """The 32-row-block program takes 32- and 64-row alignments of at most 1536 rows.

    Its tiles also meet the architecture's bound: on sm_100a
    ``ceil(m / 32) * n / 256`` 32 x 256 tiles of at most ``sm_count``, on
    sm_103a three times ``ceil(m / 128) * n / 256`` 128 x 256 tiles of at most
    ``sm_count``.  Without ``sm_count`` both keep the run32 program.  sm_100a
    sends these layouts to the small-block programs up to K 4096 (see
    test_block_scaled_swap_small_block_rule_on_sm_100a), so its cases use K
    6144.
    """
    rule = functools.partial(bs_swap_route_for_shape, num_groups=8)
    k_of = {"sm_100a": 6144, "sm_103a": 1024}
    # N 1024 on 148 SMs: 37 * 4 = 148 tiles of 32 rows fit sm_100a, 38 * 4 = 152
    # do not; sm_103a takes 3 * 10 * 4 = 120 for both.
    for m, sm_100a in ((1184, BS_SWAP_RP32_ROUTE), (1216, BS_SWAP_RUN32_ROUTE)):
        assert (
            rule(m, 1024, 6144, alignment=32, arch="sm_100a", sm_count=148) == sm_100a
        )
        assert (
            rule(m, 1024, 1024, alignment=32, arch="sm_103a", sm_count=148)
            == BS_SWAP_RP32_ROUTE
        )
    # N 2048 on 148 SMs: 3 * 6 * 8 = 144 tiles of 128 rows fit sm_103a,
    # 3 * 7 * 8 = 168 do not.
    for m, sm_103a in ((768, BS_SWAP_RP32_ROUTE), (800, BS_SWAP_RUN32_ROUTE)):
        assert (
            rule(m, 2048, 1024, alignment=32, arch="sm_103a", sm_count=148) == sm_103a
        )
    for arch, k in k_of.items():
        at_k = functools.partial(bs_swap_route_for_shape, k=k, num_groups=8)
        assert at_k(640, 1024, alignment=32, arch=arch) == BS_SWAP_RUN32_ROUTE
        # At most 1536 rows.
        assert (
            at_k(1536, 256, alignment=32, arch=arch, sm_count=148) == BS_SWAP_RP32_ROUTE
        )
        assert (
            at_k(1568, 256, alignment=32, arch=arch, sm_count=148)
            == BS_SWAP_RUN32_ROUTE
        )
        # Alignments of at most 64 rows.
        assert (
            at_k(1024, 512, alignment=64, arch=arch, sm_count=148) == BS_SWAP_RP32_ROUTE
        )
        assert (
            at_k(1024, 512, alignment=96, arch=arch, sm_count=148)
            == BS_SWAP_RUN32_ROUTE
        )


def test_block_scaled_swap_small_block_rule():
    """The small-block programs take sm_103a launches of one wave of CTAs.

    The 11-stage two-CTA program takes K >= 2048 when ``2 * ceil(m / 32) *
    n / 256`` CTAs fit ``sm_count``; the one-CTA program takes the 32-row
    alignment at 256 <= K <= 512 and N <= 512 when ``ceil(m / 32) * n / 128``
    CTAs fit.  An unknown ``sm_count`` keeps the other routes; sm_100a has its
    own rule (test_block_scaled_swap_small_block_rule_on_sm_100a).
    """
    rule = functools.partial(bs_swap_route_for_shape, num_groups=8)
    # K 4096, N 256: 2 * 76 = 152 CTAs of 32-row blocks fit 152 SMs, not 148.
    s11 = functools.partial(rule, 2432, 256, 4096, alignment=128)
    assert s11(arch="sm_103a", sm_count=152) == BS_SWAP_RP32_S11_ROUTE
    assert s11(arch="sm_103a", sm_count=148) == BS_SWAP_ROUTE
    assert s11(arch="sm_103a") == BS_SWAP_ROUTE
    assert s11(arch="sm_100a", sm_count=148) == BS_SWAP_ROUTE
    # sm_100a takes the 32-row-block program there (see the dense rule).
    assert s11(arch="sm_100a", sm_count=152) == BS_SWAP_RP32_ROUTE
    # Any alignment: 2 * 16 * 4 = 128 CTAs at K 2048, not at K 1920.
    for k, sm_103a in ((2048, BS_SWAP_RP32_S11_ROUTE), (1920, BS_SWAP_RP32_ROUTE)):
        assert rule(512, 1024, k, alignment=32, arch="sm_103a", sm_count=152) == sm_103a
    assert (
        rule(512, 1024, 2048, alignment=32, arch="sm_103a", sm_count=120)
        == BS_SWAP_RP32_ROUTE
    )
    # The one-CTA program: 36 * 4 = 144 CTAs fit 152 SMs, not 140.
    c1 = functools.partial(rule, 1152, 512, alignment=32)
    for k in (256, 512):
        assert c1(k, arch="sm_103a", sm_count=152) == BS_SWAP_C1_ROUTE
        assert c1(k, arch="sm_103a", sm_count=140) == BS_SWAP_RP32_ROUTE
        assert c1(k, arch="sm_103a") == BS_SWAP_RUN32_ROUTE
    # Its K range, its largest N and its alignment.
    for k in (128, 640, 1024):
        assert c1(k, arch="sm_103a", sm_count=152) == BS_SWAP_RP32_ROUTE
    assert (
        rule(384, 1024, 256, alignment=32, arch="sm_103a", sm_count=152)
        == BS_SWAP_RP32_ROUTE
    )
    assert (
        rule(1152, 512, 256, alignment=64, arch="sm_103a", sm_count=152)
        == BS_SWAP_RP32_ROUTE
    )
    assert sorted(BS_SWAP_SMALL_ARCHS) == ["sm_103a"]
    assert (
        bs_swap_small_route(1152, 512, 256, alignment=32, arch="sm_120a", sm_count=152)
        is None
    )
    assert {BS_SWAP_RP32_S11_ROUTE, BS_SWAP_C1_ROUTE} <= BS_SWAP_ROUTES


def test_block_scaled_swap_c1_rule_for_odd_multiples_of_128():
    """The one-CTA program takes N an odd multiple of 128 on sm_103a launches of one wave.

    The multi-run layouts (an alignment of 32, 64 or 96 rows modulo 128) at
    128 <= K <= 512 take it when ``ceil(m / 32) * n / 128`` CTAs fit
    ``sm_count``.  A single-run layout, a longer K, a second wave, an unknown
    ``sm_count`` and an architecture without swap-AB routes keep the
    non-swapped routes; N a multiple of 256 keeps its own rules.  sm_100a has
    its own rule (test_block_scaled_swap_small_block_rule_on_sm_100a).
    """
    rule = functools.partial(bs_swap_route_for_shape, num_groups=4)
    # (m, n, CTAs): 36 x 1, 36 x 3 and 12 x 9 tiles of 32 x 128.
    for m, n, ctas in ((1152, 128, 36), (1152, 384, 108), (384, 1152, 108)):
        for alignment in (32, 64, 96):
            c1 = functools.partial(rule, m, n, alignment=alignment)
            for k in (128, 256, 512):
                assert c1(k, arch="sm_103a", sm_count=152) == BS_SWAP_C1_ROUTE
            for k in (640, 1024):
                assert c1(k, arch="sm_103a", sm_count=152) is None
            # The wave edge: ``ctas`` CTAs fit ``ctas`` SMs, not ``ctas - 1``.
            assert c1(512, arch="sm_103a", sm_count=ctas) == BS_SWAP_C1_ROUTE
            assert c1(512, arch="sm_103a", sm_count=ctas - 1) is None
            assert c1(512, arch="sm_103a") is None
            assert c1(512, arch="sm_120a", sm_count=152) is None
            assert (
                bs_swap_small_route(
                    m, n, 128, alignment=alignment, arch="sm_120a", sm_count=152
                )
                is None
            )
        for alignment in (128, 256):
            assert (
                rule(m, n, 256, alignment=alignment, arch="sm_103a", sm_count=152)
                is None
            )
    # N a multiple of 256 keeps its rules: the one-CTA program at alignment 32 only.
    for alignment, route in (
        (32, BS_SWAP_C1_ROUTE),
        (64, BS_SWAP_RP32_ROUTE),
        (96, BS_SWAP_RUN32_ROUTE),
    ):
        assert (
            rule(1152, 512, 256, alignment=alignment, arch="sm_103a", sm_count=152)
            == route
        )
    assert (
        rule(1152, 512, 128, alignment=32, arch="sm_103a", sm_count=152)
        == BS_SWAP_RP32_ROUTE
    )
    # N not a multiple of 128 is never swapped.
    for n in (192, 320):
        assert (
            bs_swap_small_route(
                1152, n, 256, alignment=32, arch="sm_103a", sm_count=152
            )
            is None
        )
        assert rule(1152, n, 256, alignment=32, arch="sm_103a", sm_count=152) is None


def test_block_scaled_swap_small_block_rule_on_sm_100a():
    """sm_100a takes the small-block programs for up to three 32 x 128 tiles per SM.

    For N a multiple of 256, the multi-run layouts of an alignment of at most 64
    rows and at most 1536 rows: the 11-stage two-CTA program at K >= 2048 when
    its launch is one wave or the layout has at most 20 32-row blocks, the
    one-CTA program otherwise.  For N an odd multiple of 128, the one-CTA
    program for any alignment at 128 <= K <= 512.  K of at most 4096 and at
    most ``3 * sm_count`` tiles of 32 x 128 rows; an unknown ``sm_count`` and
    the layouts of ``BS_SWAP_SM100_SMALL_KEEP`` keep the other routes.
    """
    rule = functools.partial(bs_swap_route_for_shape, num_groups=8, arch="sm_100a")
    small = functools.partial(bs_swap_small_route, arch="sm_100a")
    # N 4096, 384 rows: 12 * 32 = 384 one-CTA tiles, more than one wave.
    for k in (128, 256, 1024, 1920):
        assert rule(384, 4096, k, alignment=32, sm_count=148) == BS_SWAP_C1_ROUTE
    # Three tiles per SM: 384 tiles fit 128 SMs, not 127.
    assert rule(384, 4096, 1024, alignment=32, sm_count=128) == BS_SWAP_C1_ROUTE
    assert rule(384, 4096, 1024, alignment=32, sm_count=127) == BS_SWAP_RUN32_ROUTE
    # K >= 2048: 20 blocks of 32 rows take the two-CTA program, 21 do not.
    assert rule(640, 2048, 2048, alignment=32, sm_count=148) == BS_SWAP_RP32_S11_ROUTE
    assert rule(672, 2048, 2048, alignment=32, sm_count=148) == BS_SWAP_C1_ROUTE
    # ... or one wave of two-CTA clusters: 2 * 36 = 72 CTAs fit 72 SMs, not 71.
    s11 = functools.partial(rule, 1152, 256, 4096, alignment=64)
    assert s11(sm_count=72) == BS_SWAP_RP32_S11_ROUTE
    assert s11(sm_count=71) == BS_SWAP_C1_ROUTE
    # K of at most 4096.
    assert rule(384, 4096, 4096, alignment=32, sm_count=148) == BS_SWAP_RP32_S11_ROUTE
    assert rule(384, 4096, 4224, alignment=32, sm_count=148) == BS_SWAP_RUN32_ROUTE
    # Alignments of at most 64 rows and at most 1536 rows.
    assert rule(1024, 512, 1024, alignment=64, sm_count=148) == BS_SWAP_C1_ROUTE
    assert rule(1024, 512, 1024, alignment=96, sm_count=148) == BS_SWAP_RUN32_ROUTE
    assert rule(1536, 256, 1024, alignment=32, sm_count=148) == BS_SWAP_C1_ROUTE
    assert rule(1568, 256, 1024, alignment=32, sm_count=148) == BS_SWAP_RUN32_ROUTE
    # Single-run layouts at N a multiple of 256 keep the other rules.
    assert small(1152, 512, 256, alignment=128, sm_count=148) is None
    # N an odd multiple of 128: any alignment at 128 <= K <= 512.
    for alignment in (32, 64, 96, 128):
        for k in (128, 256, 512):
            assert (
                small(1152, 384, k, alignment=alignment, sm_count=148)
                == BS_SWAP_C1_ROUTE
            )
        assert small(1152, 384, 640, alignment=alignment, sm_count=148) is None
    # 148 * 3 = 444 tiles of N 384 fit 148 SMs, 149 * 3 = 447 do not.
    assert small(4736, 384, 256, alignment=32, sm_count=148) == BS_SWAP_C1_ROUTE
    assert small(4768, 384, 256, alignment=32, sm_count=148) is None
    # An unknown sm_count keeps the other routes.
    assert small(384, 4096, 1024, alignment=32, sm_count=None) is None
    assert rule(384, 4096, 1024, alignment=32) == BS_SWAP_RUN32_ROUTE
    # The layouts of BS_SWAP_SM100_SMALL_KEEP keep the other swap-AB routes.
    for m, n, k, alignment in BS_SWAP_SM100_SMALL_KEEP:
        assert small(m, n, k, alignment=alignment, sm_count=148) is None
    assert rule(384, 3072, 1024, alignment=32, sm_count=148) == BS_SWAP_RP32_ROUTE
    assert rule(512, 2560, 2048, alignment=32, sm_count=148) == BS_SWAP_RUN32_ROUTE
    assert rule(512, 2048, 2048, alignment=64, sm_count=148) == BS_SWAP_RP32_ROUTE
    # ... on sm_100a only.
    assert (
        bs_swap_small_route(1152, 512, 256, alignment=32, arch="sm_103a", sm_count=152)
        == BS_SWAP_C1_ROUTE
    )
    # sm_103a keeps its one-wave rule.
    assert (
        bs_swap_route_for_shape(
            384, 4096, 1024, alignment=32, num_groups=8, arch="sm_103a", sm_count=152
        )
        == BS_SWAP_RP32_ROUTE
    )
    assert sorted(BS_SWAP_SM100_SMALL_ARCHS) == ["sm_100a"]


def test_block_scaled_swap_rp32_dense_rule():
    """sm_100a sends a dense single-run layout to the 32-row-block program while it is one pass.

    More than 128 rows per group, ``(m - num_groups * (alignment - 1)) /
    num_groups``, and ``2 * ceil(m / 32) * n / 256`` CTAs of at most
    ``sm_count``.  sm_103a, an unknown ``sm_count`` or group count and the
    multi-run layouts keep the other rules.
    """
    rule = functools.partial(bs_swap_route_for_shape, alignment=128)
    # N 256, K 4096, 8 groups: 2304 rows are 161 rows per group in 2 * 72 CTAs.
    dense = functools.partial(rule, 2304, 256, 4096, num_groups=8)
    for sm_count in (148, 144):
        assert dense(arch="sm_100a", sm_count=sm_count) == BS_SWAP_RP32_ROUTE
    assert dense(arch="sm_100a", sm_count=143) == BS_SWAP_ROUTE
    assert dense(arch="sm_100a") == BS_SWAP_ROUTE
    assert dense(arch="sm_103a", sm_count=148) == BS_SWAP_RP32_S11_ROUTE
    assert dense(arch="sm_103a") == BS_SWAP_ROUTE
    # 2432 rows: 2 * 76 = 152 CTAs fit 152 SMs, not 148.
    wide = functools.partial(rule, 2432, 256, 4096, num_groups=8, arch="sm_100a")
    assert wide(sm_count=152) == BS_SWAP_RP32_ROUTE
    assert wide(sm_count=148) == BS_SWAP_ROUTE
    # K 512: 2048 rows are 129 rows per group; 2040 rows are exactly 128.
    short = functools.partial(rule, n=256, k=512, num_groups=8, arch="sm_100a")
    assert short(2048, sm_count=148) == BS_SWAP_RP32_ROUTE
    assert short(2040, sm_count=148) is None
    # Any K and N: one group of 256 rows (129 rows) at N 2048, K 128.
    assert rule(256, 2048, 128, num_groups=1, arch="sm_100a", sm_count=148) == (
        BS_SWAP_RP32_ROUTE
    )
    assert rule(256, 2048, 128, num_groups=1, arch="sm_100a") is None
    # An alignment of 256 counts its own padding: 4 groups in 2048 rows (257 each).
    assert (
        bs_swap_route_for_shape(
            2048, 256, 1024, alignment=256, num_groups=4, arch="sm_100a", sm_count=148
        )
        == BS_SWAP_RP32_ROUTE
    )
    assert dense(num_groups=None, arch="sm_100a", sm_count=148) is None
    # The multi-run layouts keep their rule.
    assert (
        bs_swap_route_for_shape(
            2304, 256, 4096, alignment=64, num_groups=8, arch="sm_100a", sm_count=148
        )
        == BS_SWAP_RUN32_ROUTE
    )
    assert sorted(BS_SWAP_RP32_DENSE_ARCHS) == ["sm_100a"]


def test_launch_plan_takes_the_small_block_routes_on_sm_103a():
    """The small-block launches: two-CTA clusters of 32 x 256 tiles, one CTA per 32 x 128 tile."""
    plan = functools.partial(
        launch_plan, scalar_output=False, block_scaled=True, sm_count=152
    )
    route, grid = plan(512, 1024, 2048, alignment=32, num_groups=8, arch="sm_103a")
    assert route == BS_SWAP_RP32_S11_ROUTE and grid == (2 * 16 * 4, 1, 1)
    route, grid = plan(2432, 256, 4096, alignment=128, num_groups=8, arch="sm_103a")
    assert route == BS_SWAP_RP32_S11_ROUTE and grid == (2 * 76, 1, 1)
    route, grid = plan(1152, 512, 256, alignment=32, num_groups=12, arch="sm_103a")
    assert route == BS_SWAP_C1_ROUTE and grid == (36 * 4, 1, 1)
    assert route_geometry(BS_SWAP_RP32_S11_ROUTE) == (32, 256, 2)
    assert route_geometry(BS_SWAP_C1_ROUTE) == (32, 128, 1)
    route, _ = plan(1152, 512, 256, alignment=32, num_groups=12, arch=None)
    assert route != BS_SWAP_C1_ROUTE


def test_launch_plan_takes_the_one_cta_program_for_odd_multiples_of_128():
    """One CTA per 32 x 128 tile for N an odd multiple of 128 while it is one wave.

    On sm_103a a second wave and no swap-AB arch keep the 128-column run32
    route; a single-run layout keeps the non-swapped routes.
    """
    plan = functools.partial(
        launch_plan, scalar_output=False, block_scaled=True, num_groups=5
    )
    for alignment in (32, 64, 96):
        shape = functools.partial(plan, 1152, 384, 256, alignment=alignment)
        for sm_count in (152, 108):
            route, grid = shape(sm_count=sm_count, arch="sm_103a")
            assert route == BS_SWAP_C1_ROUTE and grid == (36 * 3, 1, 1)
        route, _ = shape(sm_count=107, arch="sm_103a")
        assert route == BS_N128_RUN32_ROUTE
        route, _ = shape(sm_count=152, arch=None)
        assert route == BS_N128_RUN32_ROUTE
    route, _ = plan(1152, 384, 256, sm_count=152, alignment=128, arch="sm_103a")
    assert route not in BS_SWAP_ROUTES


def test_launch_plan_takes_the_small_block_routes_on_sm_100a():
    """More 32 x 128 tiles than SMs: every SM runs a CTA (or half a two-CTA cluster)."""
    plan = functools.partial(
        launch_plan, scalar_output=False, block_scaled=True, arch="sm_100a"
    )
    # 12 * 32 = 384 one-CTA tiles on 148 CTAs.
    route, grid = plan(384, 4096, 1024, sm_count=148, alignment=32, num_groups=8)
    assert route == BS_SWAP_C1_ROUTE and grid == (148, 1, 1)
    # 20 * 8 = 160 tiles of 32 x 256 rows on 74 two-CTA clusters.
    route, grid = plan(640, 2048, 2048, sm_count=148, alignment=32, num_groups=8)
    assert route == BS_SWAP_RP32_S11_ROUTE and grid == (2 * 74, 1, 1)
    # N an odd multiple of 128: 52 * 3 = 156 tiles of a single-run layout.
    route, grid = plan(1664, 384, 128, sm_count=148, alignment=128, num_groups=8)
    assert route == BS_SWAP_C1_ROUTE and grid == (148, 1, 1)
    route, grid = plan(1152, 384, 256, sm_count=148, alignment=32, num_groups=5)
    assert route == BS_SWAP_C1_ROUTE and grid == (36 * 3, 1, 1)
    route, _ = plan(
        384, 4096, 1024, sm_count=148, alignment=32, num_groups=8, arch=None
    )
    assert route not in BS_SWAP_ROUTES


def test_launch_plan_takes_the_32_row_block_program_for_dense_layouts_on_sm_100a():
    """Two-CTA clusters of 32 x 256 tiles, one tile per cluster."""
    plan = functools.partial(
        launch_plan, scalar_output=False, block_scaled=True, alignment=128
    )
    route, grid = plan(2304, 256, 4096, num_groups=8, sm_count=148, arch="sm_100a")
    assert route == BS_SWAP_RP32_ROUTE and grid == (2 * 72, 1, 1)
    route, grid = plan(1152, 512, 2048, num_groups=4, sm_count=148, arch="sm_100a")
    assert route == BS_SWAP_RP32_ROUTE and grid == (2 * 36 * 2, 1, 1)
    route, _ = plan(2432, 256, 4096, num_groups=8, sm_count=148, arch="sm_100a")
    assert route == BS_SWAP_ROUTE
    route, _ = plan(2304, 256, 4096, num_groups=8, sm_count=148, arch=None)
    assert route not in BS_SWAP_ROUTES


@pytest.mark.parametrize(
    ("m", "n", "k", "num_groups", "alignment", "sm_100a", "sm_103a"),
    [
        # At most 128 routed rows per group: the store16 program of either family.
        (1408, 256, 512, 6, 128, BS_N128_S16_ROUTE, BS_N128_S16_ROUTE),
        (65408, 512, 4096, 512, 128, BS_SWAP_S16_ROUTE, BS_SWAP_S16_ROUTE),
        (65280, 1024, 2048, 256, 128, BS_SWAP_CARRY_S16_ROUTE, BS_N256_S16_ROUTE),
        # More rows per group, and the multi-run layouts: the 32-row stores.
        (163584, 1024, 2048, 256, 128, BS_SWAP_CARRY_ROUTE, BS_SWAP_CARRY_ROUTE),
        (16000, 4096, 256, 512, 32, BS_SWAP_RUN32_ROUTE, BS_SWAP_RUN32_ROUTE),
    ],
)
def test_block_scaled_store16_rule(m, n, k, num_groups, alignment, sm_100a, sm_103a):
    """Store16 programs from (m, num_groups, alignment, arch) only: single-run layouts of at most 128 routed rows per group."""
    plan = functools.partial(
        launch_plan,
        sm_count=148,
        scalar_output=False,
        block_scaled=True,
        alignment=alignment,
        num_groups=num_groups,
    )
    for arch, expected in (("sm_100a", sm_100a), ("sm_103a", sm_103a)):
        assert plan(m, n, k, arch=arch)[0] == expected
    for arch in (None, "sm_120a"):
        assert plan(m, n, k, arch=arch)[0] not in BS_STORE16_ROUTES.values()


def test_block_scaled_store16_needs_single_run_and_groups():
    """The bound is inclusive: 128 routed rows per group store16, 128.5 do not."""
    assert bs_store16(65280, alignment=128, num_groups=256)
    assert not bs_store16(65408, alignment=128, num_groups=256)
    assert not bs_store16(65280, alignment=128, num_groups=None)
    assert not bs_store16(16000, alignment=32, num_groups=512)


def test_launch_plan_takes_the_swap_routes_with_the_architecture():
    """``arch`` admits the swap-AB routes; a two-CTA cluster takes two 128-row weight tiles."""
    for arch in ("sm_100a", "sm_103a"):
        route, grid = launch_plan(
            65408,
            512,
            4096,
            sm_count=148,
            scalar_output=False,
            block_scaled=True,
            alignment=128,
            num_groups=512,
            arch=arch,
        )
        assert route == BS_SWAP_S16_ROUTE
        assert grid == (2 * min(148 // 2, (65408 // 128) * (512 // 256)), 1, 1)
    multi_run = functools.partial(
        launch_plan,
        scalar_output=False,
        block_scaled=True,
        alignment=32,
        num_groups=8,
    )
    # sm_100a takes the small-block programs up to K 4096: K 6144 shows its
    # other multi-run routes.
    k_of = {"sm_100a": 6144, "sm_103a": 256}
    for arch, k in k_of.items():
        # 20 * 4 tiles of 32 x 256 rows on 76 two-CTA clusters.
        route, grid = multi_run(640, 1024, k, sm_count=152, arch=arch)
        assert route == BS_SWAP_RP32_ROUTE and grid == (2 * 76, 1, 1)
        # More than 1536 rows: 14 * 4 tiles of 128 x 256 rows.
        route, grid = multi_run(1792, 1024, k, sm_count=152, arch=arch)
        assert route == BS_SWAP_RUN32_ROUTE and grid == (2 * 14 * 4, 1, 1)
    # launch_plan passes its sm_count: 80 tiles of 32 rows exceed 64 SMs on
    # sm_100a; 3 * 20 tiles of 128 rows fit 64 SMs on sm_103a, not 56.
    route, grid = multi_run(640, 1024, 6144, sm_count=64, arch="sm_100a")
    assert route == BS_SWAP_RUN32_ROUTE and grid == (2 * 5 * 4, 1, 1)
    route, grid = multi_run(640, 1024, 256, sm_count=64, arch="sm_103a")
    assert route == BS_SWAP_RP32_ROUTE and grid == (2 * 32, 1, 1)
    route, grid = multi_run(640, 1024, 256, sm_count=56, arch="sm_103a")
    assert route == BS_SWAP_RUN32_ROUTE and grid == (2 * 5 * 4, 1, 1)
    route, _ = launch_plan(
        65408,
        512,
        4096,
        sm_count=148,
        scalar_output=False,
        block_scaled=True,
        alignment=128,
        num_groups=512,
    )
    assert route not in BS_SWAP_ROUTES


# (group_counts, N, K, alignment, route): compact layouts the swap-AB rule sends to
# each swap route on both architectures with at least 80 SMs (multi-run of at
# most 1536 rows within the rp32 tile bound; longer multi-run; K > 512 below
# 100 rows per group; K <= 256 at 2..20 rows per group).  A dict names the
# route per architecture: sm_100a sends the multi-run layouts of at most 1536
# rows to the small-block programs up to K 4096 (148 SMs), except the layouts
# of BS_SWAP_SM100_SMALL_KEEP (the rp32 case: 640 rows, N 1024, K 1024).
BLOCK_SCALED_SWAP_CASES = [
    pytest.param(
        [5, 33, 0, 0, 97, 32, 1, 129],
        1024,
        1024,
        32,
        BS_SWAP_RP32_ROUTE,
        id="rp32",
    ),
    pytest.param(
        [5, 33, 0, 0, 97, 32, 1, 129],
        1024,
        6144,
        32,
        {"sm_100a": BS_SWAP_RP32_ROUTE},
        id="rp32_k6144",
    ),
    pytest.param(
        [20, 0, 31, 7, 64, 1, 2, 40],
        512,
        1024,
        64,
        {"sm_100a": BS_SWAP_C1_ROUTE, "sm_103a": BS_SWAP_RP32_ROUTE},
        id="rp32_a64",
    ),
    pytest.param(
        [200, 300, 0, 150, 400, 1, 260, 129],
        1024,
        1024,
        32,
        BS_SWAP_RUN32_ROUTE,
        id="run32",
    ),
    pytest.param(
        [20, 0, 31, 7, 64, 1, 2, 40], 512, 1024, 128, BS_SWAP_ROUTE, id="single_run"
    ),
    pytest.param(
        [10, 3, 0, 17, 20, 9, 12, 9], 1024, 256, 128, BS_SWAP_CARRY_ROUTE, id="carry"
    ),
]


@pytest.mark.parametrize("group_counts,n,k,alignment,route", BLOCK_SCALED_SWAP_CASES)
def test_block_scaled_swap_routes_match_the_non_swapped_routes(
    group_counts, n, k, alignment, route
):
    """A swap-AB route writes the bits of the non-swapped route on the same problem.

    The block-rows weight scales ``(G, N//128, cols)`` are not the swap-AB
    programs' layout, so they keep the non-swapped route on the same words.
    """
    device = torch.device("cuda")
    arch = device_arch(torch.cuda.current_device())
    if isinstance(route, dict):
        if arch not in route:
            pytest.skip(f"the case names its route on {sorted(route)} only")
        if arch in BS_SWAP_SM100_SMALL_ARCHS and (
            device_sm_count(torch.cuda.current_device()) < 148
        ):
            pytest.skip("the sm_100a routes are sized for 148 SMs")
        route = route[arch]
    inputs = _make_block_scaled_inputs(
        group_counts, n, k, alignment=alignment, seed=4770 + k, device=device
    )
    a, b, a_scale, b_scale, a_packed, b_packed, m_indices = inputs
    m = int(m_indices.numel())
    assert (
        bs_swap_route_for_shape(
            m,
            n,
            k,
            alignment=alignment,
            num_groups=len(group_counts),
            arch=arch,
            sm_count=device_sm_count(torch.cuda.current_device()),
        )
        == route
    )
    block_rows = _make_block_scaled_inputs(
        group_counts,
        n,
        k,
        alignment=alignment,
        seed=4770 + k,
        device=device,
        weight_rows_repeated=False,
    )[5]
    outs = []
    for weight_scale, expected_route in ((b_packed, route), (block_rows, None)):
        out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
        prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_packed, weight_scale, m_indices, out=out, alignment=alignment
        )
        if expected_route is None:
            assert prepared.route in BS_ROUTES
        elif bs_store16(m, alignment=alignment, num_groups=len(group_counts)):
            assert prepared.route == BS_STORE16_ROUTES[expected_route]
        else:
            assert prepared.route == expected_route
        prepared.launch()
        outs.append(out)
    torch.cuda.synchronize()
    expected = _reference(a, b, a_scale, b_scale, m_indices)
    _assert_block_scaled(outs[0], expected, m_indices)
    routed = m_indices >= 0
    assert torch.equal(
        outs[0][routed].view(torch.int16), outs[1][routed].view(torch.int16)
    )


# (group_counts, N, K, alignment, route): compact layouts the small-block rule
# sends to each small-block route on sm_103a (at most 80 CTAs: one wave) and on
# sm_100a with 148 SMs; a dict names the route per architecture (the sm_100a
# cases: more than one wave, K 4096, a single-run layout at N 384).
BLOCK_SCALED_SMALL_CASES = [
    pytest.param(
        [5, 33, 0, 0, 97, 32, 1, 129],
        256,
        2048,
        32,
        BS_SWAP_RP32_S11_ROUTE,
        id="rp32_s11",
    ),
    pytest.param(
        [20, 0, 31, 7, 64, 1, 2, 40],
        256,
        2048,
        128,
        {"sm_103a": BS_SWAP_RP32_S11_ROUTE},
        id="rp32_s11_single_run",
    ),
    pytest.param([10, 3, 0, 17, 20, 9, 12, 9], 512, 512, 32, BS_SWAP_C1_ROUTE, id="c1"),
    pytest.param(
        [100, 0, 3, 69, 128, 200, 1, 64],
        256,
        256,
        32,
        BS_SWAP_C1_ROUTE,
        id="c1_k256",
    ),
    # N an odd multiple of 128 on the multi-run layouts.
    pytest.param(
        [10, 3, 0, 17, 20, 9, 12, 9],
        384,
        128,
        32,
        BS_SWAP_C1_ROUTE,
        id="c1_n384_a32_k128",
    ),
    pytest.param(
        [20, 0, 31, 7, 64, 1, 2, 40],
        128,
        512,
        64,
        BS_SWAP_C1_ROUTE,
        id="c1_n128_a64_k512",
    ),
    pytest.param([5, 33, 0, 40], 640, 256, 96, BS_SWAP_C1_ROUTE, id="c1_n640_a96"),
    # sm_100a: 12 * 32 = 384 one-CTA tiles; 12 blocks of K 2048 on the two-CTA
    # program (192 CTAs); K 4096 on 24 * 8 = 192 tiles; a single-run N 384.
    pytest.param(
        [40, 77, 3, 100],
        4096,
        1024,
        32,
        {"sm_100a": BS_SWAP_C1_ROUTE},
        id="c1_sm100_waves",
    ),
    pytest.param(
        [40, 77, 3, 100],
        2048,
        2048,
        32,
        {"sm_100a": BS_SWAP_RP32_S11_ROUTE},
        id="rp32_s11_sm100_blocks",
    ),
    pytest.param(
        [20, 0, 31, 7, 64, 1, 2, 40],
        1024,
        4096,
        64,
        {"sm_100a": BS_SWAP_C1_ROUTE},
        id="c1_sm100_k4096",
    ),
    pytest.param(
        [100, 0, 3, 128, 300, 1],
        384,
        128,
        128,
        {"sm_100a": BS_SWAP_C1_ROUTE},
        id="c1_sm100_n384_single_run",
    ),
]


@pytest.mark.parametrize("group_counts,n,k,alignment,route", BLOCK_SCALED_SMALL_CASES)
def test_block_scaled_small_block_routes_match_the_non_swapped_routes(
    group_counts, n, k, alignment, route
):
    """A small-block route writes the bits of the non-swapped route on the same problem."""
    device = torch.device("cuda")
    arch = device_arch(torch.cuda.current_device())
    if arch not in BS_SWAP_SMALL_ARCHS and arch not in BS_SWAP_SM100_SMALL_ARCHS:
        pytest.skip("the small-block swap-AB rules serve sm_100a and sm_103a only")
    if arch in BS_SWAP_SM100_SMALL_ARCHS and (
        device_sm_count(torch.cuda.current_device()) < 148
    ):
        pytest.skip("the sm_100a cases are sized for 148 SMs")
    if isinstance(route, dict):
        if arch not in route:
            pytest.skip(f"the case names its route on {sorted(route)} only")
        route = route[arch]
    inputs = _make_block_scaled_inputs(
        group_counts, n, k, alignment=alignment, seed=4790 + k, device=device
    )
    a, b, a_scale, b_scale, a_packed, b_packed, m_indices = inputs
    m = int(m_indices.numel())
    block_rows = _make_block_scaled_inputs(
        group_counts,
        n,
        k,
        alignment=alignment,
        seed=4790 + k,
        device=device,
        weight_rows_repeated=False,
    )[5]
    outs = []
    for weight_scale, expected_route in ((b_packed, route), (block_rows, None)):
        out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
        prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_packed, weight_scale, m_indices, out=out, alignment=alignment
        )
        if expected_route is None:
            assert prepared.route in BS_ROUTES
        else:
            assert prepared.route == expected_route
        prepared.launch()
        outs.append(out)
    torch.cuda.synchronize()
    expected = _reference(a, b, a_scale, b_scale, m_indices)
    _assert_block_scaled(outs[0], expected, m_indices)
    routed = m_indices >= 0
    assert torch.equal(
        outs[0][routed].view(torch.int16), outs[1][routed].view(torch.int16)
    )


# (group_counts, N, K): dense single-run layouts (alignment 128) the sm_100a rule
# sends to the 32-row-block program on 148 SMs (more than 128 rows per group,
# at most 148 CTAs).
BLOCK_SCALED_DENSE_CASES = [
    pytest.param([300, 260, 0, 280], 256, 512, id="n256_k512"),
    pytest.param([300, 260, 0, 280], 256, 4096, id="n256_k4096"),
    pytest.param([200, 0, 150, 180], 512, 2048, id="n512_k2048"),
]


@pytest.mark.parametrize("group_counts,n,k", BLOCK_SCALED_DENSE_CASES)
def test_block_scaled_rp32_dense_routes_match_the_non_swapped_routes(
    group_counts, n, k
):
    """On sm_100a the 32-row-block route writes the bits of the non-swapped route."""
    device = torch.device("cuda")
    if device_arch(torch.cuda.current_device()) not in BS_SWAP_RP32_DENSE_ARCHS:
        pytest.skip("the dense 32-row-block rule is sm_100a only")
    if device_sm_count(torch.cuda.current_device()) < 148:
        pytest.skip("the cases are sized for 148 SMs")
    inputs = _make_block_scaled_inputs(
        group_counts, n, k, alignment=128, seed=4800 + k, device=device
    )
    a, b, a_scale, b_scale, a_packed, b_packed, m_indices = inputs
    m = int(m_indices.numel())
    block_rows = _make_block_scaled_inputs(
        group_counts,
        n,
        k,
        alignment=128,
        seed=4800 + k,
        device=device,
        weight_rows_repeated=False,
    )[5]
    outs = []
    for weight_scale, expected_route in (
        (b_packed, BS_SWAP_RP32_ROUTE),
        (block_rows, None),
    ):
        out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device=device)
        prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_packed, weight_scale, m_indices, out=out, alignment=128
        )
        if expected_route is None:
            assert prepared.route in BS_ROUTES
        else:
            assert prepared.route == expected_route
        prepared.launch()
        outs.append(out)
    torch.cuda.synchronize()
    expected = _reference(a, b, a_scale, b_scale, m_indices)
    _assert_block_scaled(outs[0], expected, m_indices)
    routed = m_indices >= 0
    assert torch.equal(
        outs[0][routed].view(torch.int16), outs[1][routed].view(torch.int16)
    )


def test_block_scaled_swap_route_rebinds_token_operands_and_captures():
    """Per-call rebinding and a first launch inside a CUDA graph on a swap-AB route."""
    device = torch.device("cuda")
    counts = [20, 0, 31, 7, 64, 1, 2, 40]
    a, b, a_scale, b_scale, a_packed, b_packed, m_indices = _make_block_scaled_inputs(
        counts, 512, 1024, alignment=128, seed=4780, device=device
    )
    m = int(m_indices.numel())
    out = torch.full((m, 512), float("nan"), dtype=torch.bfloat16, device=device)
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_packed, b_packed, m_indices, out=out
    )
    assert prepared.route == BS_SWAP_S16_ROUTE
    a2, _, a_scale2, _, a_packed2, _, _ = _make_block_scaled_inputs(
        counts, 512, 1024, alignment=128, seed=4781, device=device
    )
    m_indices2 = _compact_layout([2, 60, 0, 5, 33, 8, 1, 39], 128, device)
    assert int(m_indices2.numel()) == m
    out2 = torch.full_like(out, float("nan"))
    assert (
        prepared.launch(a=a2, a_scale=a_packed2, m_indices=m_indices2, out=out2) is out2
    )
    torch.cuda.synchronize()
    _assert_block_scaled(
        out2, _reference(a2, b, a_scale2, b_scale, m_indices2), m_indices2
    )
    out3 = torch.full_like(out, float("nan"))
    fresh = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_packed, b_packed, m_indices, out=out3
    )
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        fresh.launch()
    torch.cuda.synchronize()
    out3.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_block_scaled(out3, _reference(a, b, a_scale, b_scale, m_indices), m_indices)


def test_block_scaled_rejects_invalid_inputs():
    device = torch.device("cuda")
    a, b, a_scale, b_scale, a_packed, b_packed, m_indices = _make_block_scaled_inputs(
        [128, 64], 256, 512, alignment=128, seed=4763, device=device
    )
    with pytest.raises(ValueError, match="fill_padding"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_packed, b_packed, m_indices, fill_padding=True
        )
    with pytest.raises(ValueError, match="packed int32 b_scale|int32 tensor"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_packed, b_scale, m_indices
        )
    with pytest.raises(ValueError, match="multiple of 32"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a, b, a_packed, b_packed, m_indices, alignment=100
        )
    with pytest.raises(ValueError, match="16-byte-aligned output"):
        storage = torch.empty(
            int(m_indices.numel()) * 256 + 8, dtype=torch.bfloat16, device=device
        )
        prepare_group_gemm_fp8_nt_groupwise_contiguous(
            a,
            b,
            a_packed,
            b_packed,
            m_indices,
            out=storage[1 : 1 + int(m_indices.numel()) * 256].view(-1, 256),
        )


def test_launch_plan_below_128_ctas_uses_a_registered_route():
    """(4096, 4096, 1024) selects the four-load schedule, exported only at 128 CTAs: a device with
    fewer than 128 SMs plans the same-geometry recurrence schedule instead of an unregistered route."""
    route, grid = launch_plan(4096, 4096, 1024, sm_count=148, scalar_output=False)
    assert (route, grid) == (DEEPK_CG2_FOUR_LOAD_GRID128_ROUTE, (128, 1, 1))
    for sm_count in (2, 64, 96, 127):
        route, grid = launch_plan(
            4096, 4096, 1024, sm_count=sm_count, scalar_output=False
        )
        assert grid == ((sm_count // 2) * 2, 1, 1)
        assert route == DEEPK_CG2_RECURRENCE_ROUTE
        assert route in ROUTES


def test_route_rule_reads_the_layout_alignment():
    aligned = [256] * 16
    unaligned = [255, 257] + [256] * 14
    assert _aligned(aligned) and _aligned([256, 0, 512, 0, 255])
    assert not _aligned(unaligned) and not _aligned([0, 128, 128])
    # Selected shapes: a layout unknown or aligned to the 256-row tiles keeps the two-CTA routes.
    for known in (None, True):
        assert (
            _route(4096, 2048, 4096, aligned=known)
            == DEEPK_CG2_AB7_BSCALE_PREFETCH_ROUTE
        )
        assert _route(4096, 4096, 1024, aligned=known) == DEEPK_CG2_FOUR_LOAD_ROUTE
    assert route_for_shape(16384, 2048, 4096) == DEEPK_CG2_AB7_BSCALE_PREFETCH_ROUTE
    assert route_for_shape(16384, 4096, 1024) == DEEPK_CG2_RECURRENCE_ROUTE
    # A known unaligned layout without its per-tile starts runs a single-CTA route on all SMs.
    assert _route(4096, 2048, 4096, aligned=False) == DEEPK_C2_AB6_SCALE1_ROUTE
    assert _route(4096, 4096, 1024, aligned=False) == DEEPK_N256_ROUTE
    plan = _launch_plan(
        4096,
        2048,
        4096,
        sm_count=148,
        scalar_output=False,
        block_scaled=False,
        alignment=128,
        num_groups=16,
        arch=None,
        aligned=False,
    )
    assert plan == (DEEPK_C2_AB6_SCALE1_ROUTE, (148, 1, 1))
    # The block-scaled family keeps its plan, including the 128-CTA cap of the selected shapes.
    route, grid = launch_plan(
        4096, 4096, 1024, sm_count=148, scalar_output=False, block_scaled=True
    )
    assert route in BS_ROUTES and grid[0] <= 128
    # An aligned layout one row off the selected M keeps the two-CTA route.
    assert _route(4095, 2048, 4096, aligned=True) == DEEPK_CG2_AB7_BSCALE_PREFETCH_ROUTE
    assert _route(16385, 4096, 1024, aligned=True) == DEEPK_CG2_RECURRENCE_ROUTE
    assert route_for_shape(4095, 2048, 4096) == DEEPK_C2_AB6_SCALE1_ROUTE
    # K=128 runs the K-loop K-group 1 route for at most sm_count 128x128 tiles
    # (one wave), else its static-K route; K=384 runs the K-loop route for at
    # most four 128x128 tiles.
    assert _route(1, 128, 128, sm_count=148) == DEEPK_C2_KG1_ROUTE
    assert _route(512, 4736, 128, sm_count=148) == DEEPK_C2_KG1_ROUTE
    assert _route(512, 4864, 128, sm_count=148) == K128_ROUTE
    assert _route(512, 4864, 128, sm_count=152) == DEEPK_C2_KG1_ROUTE
    assert route_for_shape(1, 128, 128) == K128_ROUTE
    assert _route(131072, 4096, 128, sm_count=152) == K128_ROUTE
    assert _route(64, 256, 384, sm_count=148) == DEEPK_C2_KG1_ROUTE
    assert _route(256, 256, 384, sm_count=152) == DEEPK_C2_KG1_ROUTE
    assert _route(256, 384, 384, sm_count=148) == K384_ROUTE
    assert _route(1234, 384, 384, sm_count=148) == K384_ROUTE
    assert _route(16384, 4096, 384, sm_count=148) == K384_ROUTE
    assert route_for_shape(64, 256, 384) == K384_ROUTE
    assert route_for_shape(128, 128, 512) == DEEPK_C2_AB6_SCALE1_ROUTE
    assert route_for_shape(256, 384, 1280) == DEEPK_C2_KG1_ROUTE
    # K=512 keeps the K-multiple-of-512 route from two 128x128 tiles on, except at
    # N=256 or M=4096, with or without the SM count.
    for sm_count in (None, 148, 152):
        assert _route(64, 1536, 512, sm_count=sm_count) == DEEPK_C2_ROUTE
        assert _route(1024, 1024, 512, sm_count=sm_count) == DEEPK_C2_ROUTE
        assert _route(128, 256, 512, sm_count=sm_count) == DEEPK_C2_AB6_SCALE1_ROUTE
        assert _route(129, 256, 512, sm_count=sm_count) == DEEPK_C2_AB6_SCALE1_ROUTE
        assert _route(4096, 1024, 512, sm_count=sm_count) == DEEPK_C2_AB6_SCALE1_ROUTE
    assert route_for_shape(129, 128, 512) == DEEPK_C2_ROUTE
    assert route_for_shape(1024, 1024, 1024) == DEEPK_C2_AB6_SCALE1_ROUTE


UNALIGNED_CG2_CASES = [
    # A start off both tile heights adds a segment to both routes: the two-CTA route stays.
    pytest.param(
        16384, 4096, 1024, [64, 16320], DEEPK_CG2_RECURRENCE_ROUTE, id="pair_kept"
    ),
    # The pair route stays when the single-CTA route saves its busiest worker only one
    # segment (14 against 15), and leaves at two (14 against 16).
    pytest.param(
        16384,
        4096,
        1024,
        [128, 16256],
        DEEPK_CG2_RECURRENCE_ROUTE,
        id="pair_kept_one_fewer",
    ),
    pytest.param(
        16384, 4096, 1024, [128, 960, 15296], DEEPK_N256_ROUTE, id="pair_to_n256"
    ),
    pytest.param(
        4096, 4096, 1024, [64, 4032], DEEPK_CG2_FOUR_LOAD_ROUTE, id="four_kept"
    ),
    pytest.param(
        4096, 2048, 4096, [64, 4032], DEEPK_CG2_AB7_BSCALE_PREFETCH_ROUTE, id="ab7_kept"
    ),
    # A start on a 128-row boundary only burdens the 256-row tiles: the single-CTA route.
    pytest.param(4096, 4096, 1024, [128, 3968], DEEPK_N256_ROUTE, id="four_to_n256"),
    pytest.param(
        4096, 2048, 4096, [128, 3968], DEEPK_C2_AB6_SCALE1_ROUTE, id="ab7_to_ab6"
    ),
]


@pytest.mark.parametrize("m,n,k,group_counts,expected", UNALIGNED_CG2_CASES)
@pytest.mark.parametrize("sm_count", [148, 152])
def test_unaligned_layout_keeps_the_two_cta_route_unless_one_cta_is_shorter(
    m, n, k, group_counts, expected, sm_count
):
    """At a shape whose shape route is two-CTA, a known unaligned layout takes the
    single-CTA route only when its busiest worker runs fewer expert segments (at
    least two fewer for the pair route; at N=2048, K=4096: at most 1.5 times as
    many)."""
    assert sum(group_counts) == m and not _aligned(group_counts)
    route = _route(
        m,
        n,
        k,
        sm_count=sm_count,
        aligned=False,
        tile_starts=_tile_starts_ref(group_counts),
    )
    assert route == expected


def test_unaligned_layout_rule_reads_the_sm_count():
    """The same layout can keep the two-CTA route on 148 SMs and leave it on 152."""
    group_counts = [257] + [256] * 14 + [255]
    starts = _tile_starts_ref(group_counts)
    plans = {
        sm_count: _route(
            4096, 2048, 4096, sm_count=sm_count, aligned=False, tile_starts=starts
        )
        for sm_count in (148, 152)
    }
    assert plans == {
        148: DEEPK_CG2_AB7_BSCALE_PREFETCH_ROUTE,
        152: DEEPK_C2_AB6_SCALE1_ROUTE,
    }


def test_routing_aware_routes_are_bitwise_equal():
    """The layout selects among programs with one accumulation order: identical output bits."""
    device = torch.device("cuda")
    rng = random.Random(4796)
    group_counts = _random_token_counts(rng, 4096, 16)
    assert not _aligned(group_counts)
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts, 4096, 1024, seed=4796, device=device, arbitrary_scales=True
    )
    unknown = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices
    )
    validated = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    )
    if torch.cuda.get_device_properties(device).multi_processor_count >= 128:
        assert unknown.route == DEEPK_CG2_FOUR_LOAD_GRID128_ROUTE
    assert validated.route in (
        DEEPK_N256_ROUTE,
        DEEPK_C2_KG1_EXPERT_ALIGNED_ROUTE,
        unknown.route,
    )
    outs = [p.launch() for p in (unknown, validated)]
    torch.cuda.synchronize()
    _assert_close(outs[1], _reference(a, b, a_scale, b_scale, m_indices))
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))


def test_direct_store_route_only_when_every_tile_is_small():
    # At most 32 rows on at most 16 CTAs: the direct-store program of the KG1 / AB6 route.
    for (m, n, k), plan in (
        ((1, 128, 256), (DEEPK_C2_KG1_DIRECT_STORE_ROUTE, (1, 1, 1))),
        ((32, 256, 384), (DEEPK_C2_KG1_DIRECT_STORE_ROUTE, (2, 1, 1))),
        ((32, 2048, 640), (DEEPK_C2_KG1_DIRECT_STORE_ROUTE, (16, 1, 1))),
        ((32, 128, 128), (DEEPK_C2_KG1_DIRECT_STORE_ROUTE, (1, 1, 1))),
        ((1, 128, 512), (DEEPK_C2_AB6_SCALE1_DIRECT_STORE_ROUTE, (1, 1, 1))),
        ((32, 2048, 4096), (DEEPK_C2_AB6_SCALE1_DIRECT_STORE_ROUTE, (16, 1, 1))),
    ):
        assert launch_plan(m, n, k, sm_count=148, scalar_output=False) == plan
        assert plan[0] in ROUTES
    # More than 32 rows on the KG1 route take its 128 x 64 program when those
    # tiles fit in one wave (see test_kg1_n64_route_when_its_tiles_fit_in_one_wave).
    for (m, n, k), plan in (
        ((33, 128, 256), (DEEPK_C2_KG1_N64_ROUTE, (2, 1, 1))),
        ((64, 256, 384), (DEEPK_C2_KG1_N64_ROUTE, (4, 1, 1))),
    ):
        assert launch_plan(m, n, k, sm_count=148, scalar_output=False) == plan
    # More than 16 CTAs or another route keeps the route's program.
    for (m, n, k), plan in (
        ((32, 2176, 640), (DEEPK_C2_KG1_ROUTE, (17, 1, 1))),
        ((129, 256, 512), (DEEPK_C2_AB6_SCALE1_ROUTE, (4, 1, 1))),
        ((32, 4096, 512), (DEEPK_C2_ROUTE, (32, 1, 1))),
        ((32, 19072, 128), (K128_ROUTE, (75, 1, 1))),
    ):
        assert launch_plan(m, n, k, sm_count=148, scalar_output=False) == plan
    # K=128 past one wave of 128x128 tiles takes the static-K route, which has no
    # direct-store program, on at most 16 CTAs too.
    assert launch_plan(32, 2560, 128, sm_count=16, scalar_output=False) == (
        K128_ROUTE,
        (10, 1, 1),
    )
    assert launch_plan(32, 128, 256, sm_count=148, scalar_output=True) == (
        DEEPK_C2_SCALAR_OUTPUT_ROUTE,
        (1, 1, 1),
    )


@pytest.mark.parametrize(
    "group_counts,n,k,full_route",
    [
        pytest.param([1], 256, 640, DEEPK_C2_KG1_N64_ROUTE, id="kg1_m1"),
        pytest.param(
            [5, 0, 27], 2048, 640, DEEPK_C2_KG1_N64_ROUTE, id="kg1_m32_three_experts"
        ),
        pytest.param([16], 256, 512, DEEPK_C2_AB6_SCALE1_ROUTE, id="ab6_m16"),
        pytest.param(
            [3, 29], 256, 4096, DEEPK_C2_AB6_SCALE1_ROUTE, id="ab6_m32_two_experts"
        ),
    ],
)
def test_direct_store_programs_match_the_route_programs(group_counts, n, k, full_route):
    """A launch of at most 32 rows runs the direct-store program; its output equals
    the same rows of a 33-row launch, which runs the route's TMA-store program (on
    the KG1 route, its 128 x 64 program: those tiles fit in one wave)."""
    device = torch.device("cuda")
    m = sum(group_counts)
    longer = [*group_counts[:-1], group_counts[-1] + 33 - m]
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        longer, n, k, seed=4800 + m + n + k, device=device, arbitrary_scales=True
    )
    short = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a[:m], b, a_scale[:m], b_scale, m_indices[:m]
    )
    full = prepare_group_gemm_fp8_nt_groupwise_contiguous(
        a, b, a_scale, b_scale, m_indices
    )
    assert full.route == full_route
    assert (
        short.route
        == {
            DEEPK_C2_KG1_N64_ROUTE: DEEPK_C2_KG1_DIRECT_STORE_ROUTE,
            DEEPK_C2_AB6_SCALE1_ROUTE: DEEPK_C2_AB6_SCALE1_DIRECT_STORE_ROUTE,
        }[full.route]
    )
    out = short.launch()
    expect = full.launch()
    torch.cuda.synchronize()
    assert torch.equal(out, expect[:m])
    _assert_close(out, _reference(a[:m], b, a_scale[:m], b_scale, m_indices[:m]))


def test_kg1_n64_route_when_its_tiles_fit_in_one_wave():
    """A KG1 launch of more than 32 rows takes the KG1 program with 128 x 64 tiles
    when ceil(M / 128) * ceil(N / 64) <= the SM count, one CTA per tile; at most 32
    rows, or more tiles than SMs, keep the 128 x 128 program."""
    assert route_geometry(DEEPK_C2_KG1_N64_ROUTE) == (128, 64, 1)
    assert DEEPK_C2_KG1_N64_ROUTE in ROUTES
    for (m, n, k, sm_count), plan in (
        ((256, 4736, 640, 148), (DEEPK_C2_KG1_N64_ROUTE, (148, 1, 1))),
        ((256, 4864, 640, 148), (DEEPK_C2_KG1_ROUTE, (76, 1, 1))),
        ((256, 4864, 640, 152), (DEEPK_C2_KG1_N64_ROUTE, (152, 1, 1))),
        ((33, 2176, 640, 148), (DEEPK_C2_KG1_N64_ROUTE, (34, 1, 1))),
        ((32, 2176, 640, 148), (DEEPK_C2_KG1_ROUTE, (17, 1, 1))),
        ((127, 128, 256, 148), (DEEPK_C2_KG1_N64_ROUTE, (2, 1, 1))),
    ):
        assert launch_plan(m, n, k, sm_count=sm_count, scalar_output=False) == plan


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
    two_group_a, two_group_b, two_as, two_bs, two_idx = _make_inputs(
        [64, 64], 256, 512, seed=4742, device=device
    )
    swapped = torch.flip(two_idx, dims=(0,)).contiguous()
    with pytest.raises(ValueError, match="sorted"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous(
            two_group_a, two_group_b, two_as, two_bs, swapped, validate_indices=True
        )
