# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the generated Blackwell contiguous grouped FP8 GEMM programs."""

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
    BS_N256_RUN32_ROUTE,
    BS_ROUTES,
    DEEPK_CG2_FOUR_LOAD_GRID128_ROUTE,
    DEEPK_CG2_RECURRENCE_ROUTE,
    bs_route_for_shape,
    is_group_gemm_fp8_nt_groupwise_contiguous_prepared_available,
    launch_plan,
)
from flashinfer.jit.gemm.cake_grouped_fp8_gemm import (
    ROUTES,
    SUPPORTED_COMPUTE_CAPABILITIES,
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
    """Routed rows match the reference; padding is skipped per 32-row sub-block.

    Every row of a 32-row sub-block whose leading ``m_indices`` entry is ``-1`` stays
    untouched (NaN-filled before the launch); ``-1`` rows sharing a sub-block with
    routed rows hold finite values of no meaning (DeepGEMM does the same per 128-row
    block).
    """
    valid = m_indices >= 0
    _assert_close(out[valid], expected[valid])
    leading = m_indices.view(-1, 32)[:, :1].expand(-1, 32).reshape(-1)
    untouched = leading < 0
    if bool(untouched.any()):
        assert torch.isnan(out[untouched].float()).all()
    touched_padding = (~valid) & ~untouched
    if bool(touched_padding.any()):
        assert torch.isfinite(out[touched_padding].float()).all()


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
    assert prepared.route in BS_ROUTES
    assert (prepared.route in (BS_N128_RUN32_ROUTE, BS_N256_RUN32_ROUTE)) == (
        alignment % 128 != 0
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
    """BLOCK_N 256 for short K or at least three waves of M blocks, 128 otherwise; run32 below 128-row alignment."""
    assert (
        bs_route_for_shape(48896, 1024, 2048, alignment=128, sm_count=152)
        == BS_N128_ROUTE
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
    for route in BS_ROUTES:
        assert route in ROUTES or any(
            route in table for table in ROUTES.values() if isinstance(table, dict)
        )
    with pytest.raises(ValueError, match="multiple of 32"):
        bs_route_for_shape(4096, 1024, 2048, alignment=48, sm_count=148)


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
