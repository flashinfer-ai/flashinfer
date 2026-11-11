# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the generated Blackwell fused grouped FP8 gate_up GEMM + SwiGLU + FP8 quantization programs."""

import random

import pytest
import torch

from flashinfer.activation import silu_and_mul
from flashinfer.gemm import (
    group_gemm_fp8_nt_groupwise_contiguous,
    prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant,
)
from flashinfer.gemm.cake_grouped_fp8_fused_silu_quant import (
    ACT_ROUTE,
    ACT_ROUTES,
    ACT_WIDE_MIN_ITEMS,
    ACT_WIDE_ROUTE,
    CTA_CAP,
    FUSED_MIXED_ROUTE,
    FUSED_ROUTE,
    FUSED_ROUTES,
    GEMM_BACKEND_CAKE,
    GEMM_BACKEND_CUTE,
    MIXED_CTA_CAPS,
    ROUTE_MODEL_GEOMETRY,
    ROUTE_MODELS_CAP,
    SMALL_M_MAX,
    TINY_MIXED_MAX_K,
    TINY_MIXED_MAX_UNITS,
    act_items,
    fused_tile_counts,
    is_group_gemm_fp8_nt_groupwise_contiguous_silu_quant_prepared_available,
    launch_plan,
    mixed_clusters,
    routing_blocks,
    select_route,
    select_route_and_cap,
    small_m_gemm_backend,
    tail_launch_grid,
)
from flashinfer.jit.gemm.cake_grouped_fp8_fused_silu_quant import (
    SUPPORTED_COMPUTE_CAPABILITIES,
    device_arch,
)
from flashinfer.quantization import per_token_group_quant_8bit

GROUP_SIZE = 128
EPS = 1e-10
# FP8 outputs: dequantized 0.1/0.1 per element against the chain; scales exact.
ATOL = RTOL = 0.1


def _small_m_route(n2, k, group_blocks):
    """Route of a known small-M routing: the mixed schedule for a tiny routing (no
    pair tile, at most TINY_MIXED_MAX_UNITS odd-tail units, K <= TINY_MIXED_MAX_K),
    else the GEMM + act route."""
    pair_tiles, odd_units = fused_tile_counts(n2, group_blocks)
    tiny = (
        pair_tiles == 0
        and 0 < odd_units <= TINY_MIXED_MAX_UNITS
        and k <= TINY_MIXED_MAX_K
    )
    return FUSED_MIXED_ROUTE if tiny else ACT_ROUTE


@pytest.fixture(autouse=True)
def _require_generated_program():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip(
            "generated fused grouped FP8 gate_up programs target SM100a and SM103a"
        )
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


def _reference_halves(y_bf16):
    """The BF16 gate and up halves of the reference GEMM output, as FP32 tensors."""
    h = y_bf16.shape[1] // 2
    return y_bf16[:, :h].float(), y_bf16[:, h:].float()


def _bf16_step(x, direction):
    """The BF16 neighbour of every element of ``x`` (bfloat16, finite) one ulp
    toward +inf (``direction`` 1) or -inf (``direction`` -1)."""
    bits = x.contiguous().view(torch.int16)
    if direction > 0:
        stepped = torch.where(bits >= 0, bits + 1, bits - 1)
        # -0.0 steps to the smallest positive value
        stepped = torch.where(bits == -(2**15), torch.ones_like(bits), stepped)
    else:
        stepped = torch.where(bits > 0, bits - 1, bits + 1)
        # +0.0 steps to the smallest negative value
        stepped = torch.where(bits == 0, torch.full_like(bits, -(2**15 - 1)), stepped)
    return stepped.view(torch.bfloat16)


def _activation_envelope(g, u):
    """Per element, the BF16 interval ``[lo, hi]`` of activations a kernel can
    produce whose gate and up halves each lie within one BF16 ulp of the
    reference halves ``g`` and ``u``.

    The kernel accumulates its FP32 dot products in a different order from
    :func:`_reference_gemm`, so each of its BF16 halves is the reference value
    or one of its two BF16 neighbours (nine gate/up candidates).  Its FP32
    SwiGLU (``ex2.approx`` / ``rcp.approx``) can round a candidate's product to
    a neighbouring BF16 activation, so the envelope extends one BF16 ulp
    beyond the candidates' extremes.  (A half that is tiny through cancellation
    can differ by more ulps of its own magnitude, but it lies far below the
    group's quantization step and cannot set the group's absmax.)
    """
    g16, u16 = g.to(torch.bfloat16), u.to(torch.bfloat16)
    gates = (_bf16_step(g16, -1), g16, _bf16_step(g16, 1))
    ups = (_bf16_step(u16, -1), u16, _bf16_step(u16, 1))
    lo = hi = None
    for gate in gates:
        gate = gate.float()
        silu = gate * torch.sigmoid(gate)
        for up in ups:
            act = (silu * up.float()).to(torch.bfloat16)
            lo = act if lo is None else torch.minimum(lo, act)
            hi = act if hi is None else torch.maximum(hi, act)
    return _bf16_step(lo, -1).float(), _bf16_step(hi, 1).float()


def _e4m3_spacing(q):
    """Distance between adjacent E4M3 values at |q| (subnormal spacing 2**-9 below 2**-6)."""
    _, exponent = torch.frexp(q.float().abs().clamp_min(2.0**-6))
    return torch.pow(2.0, exponent.float() - 4.0)


def _assert_quantizes_reference(out_q, out_s, g, u):
    """Definition check against the torch reference, independent of any FlashInfer kernel.

    :func:`_activation_envelope` encloses every BF16 activation of a kernel whose
    gate and up halves are within one BF16 ulp of the reference halves.  The
    group scale ``max(absmax, EPS) / 448`` (an FP32 division, as in the kernel)
    therefore lies between the scales of the envelope's smallest and largest
    magnitudes, and round-to-nearest quantization ``q = act * (1 / scale)``
    places the dequantized value within half an E4M3 spacing of the envelope
    plus the two FP32 roundings of the reciprocal-multiply.
    """
    m, h = out_q.shape
    lo, hi = _activation_envelope(g, u)
    straddles_zero = (lo <= 0) & (hi >= 0)
    magnitude_lo = torch.where(
        straddles_zero, torch.zeros_like(lo), torch.minimum(lo.abs(), hi.abs())
    )
    magnitude_hi = torch.maximum(lo.abs(), hi.abs())
    grouped = (m, h // GROUP_SIZE, GROUP_SIZE)
    scale_lo = magnitude_lo.reshape(grouped).amax(dim=-1).clamp_min(EPS) / 448.0
    scale_hi = magnitude_hi.reshape(grouped).amax(dim=-1).clamp_min(EPS) / 448.0
    assert torch.isfinite(out_s).all()
    s = out_s.reshape(m, h // GROUP_SIZE)
    outside = (s < scale_lo) | (s > scale_hi)
    excess_rel = torch.maximum((scale_lo - s) / scale_lo, (s - scale_hi) / scale_hi)
    assert not bool(outside.any()), (
        f"{int(outside.sum())} group scales outside the one-ulp envelope "
        f"(max relative excess {float(excess_rel.max()):.3e})"
    )
    s = s.reshape(m, h // GROUP_SIZE, 1).expand(grouped).reshape(m, h)
    dequantized = out_q.float() * s
    assert torch.isfinite(dequantized).all()
    distance = torch.clamp(lo - dequantized, min=0.0) + torch.clamp(
        dequantized - hi, min=0.0
    )
    bound = (0.5 * _e4m3_spacing(out_q) + 2.0**-22 * out_q.float().abs()) * s
    excess = distance - bound
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


def _chain_or_skip(a, b, a_scale, b_scale, m_indices):
    """The chain reference, or a skip when one of its kernels cannot run here.

    ``per_token_group_quant_8bit`` is a cuTile kernel: without the ``tileiras``
    compiler it raises ``FileNotFoundError`` at JIT time.  The torch-reference
    tests cover the prepared program on such configurations.
    """
    try:
        from flashinfer.cutile import is_cuda_tile_available
    except ImportError:
        is_cuda_tile_available = None
    if is_cuda_tile_available is not None and not is_cuda_tile_available():
        pytest.skip(
            "cuTile unavailable: the chain reference per_token_group_quant_8bit "
            "needs cuda-tile with the tileiras compiler"
        )
    try:
        return _chain(a, b, a_scale, b_scale, m_indices)
    except (
        ValueError,
        ImportError,
        RuntimeError,
        NotImplementedError,
        FileNotFoundError,
    ) as exc:
        pytest.skip(f"FlashInfer gate_up chain unavailable: {exc}")


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
    # odd-tail coverage: every expert one block (all odd tails, M = 2048 route boundary) and a mixed routing with
    # twelve odd-block experts plus two empty experts
    pytest.param([128] * 16, 2048, 4096, False, id="wide_ep32_gate_up_all_one_block"),
    pytest.param(
        [640, 128, 384, 0, 896, 128, 384, 256, 128, 0, 384, 128, 256, 128, 128, 128],
        2048,
        4096,
        True,
        id="wide_ep32_gate_up_mixed_tail",
    ),
    # two odd-block experts: 16 odd-tail units fit on the free SMs, so the fused route (pair + tail kernels) keeps the row
    pytest.param(
        [128, 384] + [256] * 14, 2048, 4096, False, id="wide_ep32_gate_up_two_odd"
    ),
]


@pytest.mark.parametrize("group_counts,n2,k,arbitrary_scales", ROUTING_CASES)
def test_prepared_matches_torch_reference_chain(group_counts, n2, k, arbitrary_scales):
    device = torch.device("cuda")
    seed = 662 + len(group_counts) + n2 + k
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts,
        n2,
        k,
        seed=seed,
        device=device,
        arbitrary_scales=arbitrary_scales,
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    )
    m = sum(group_counts)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    # validate_indices=True hands the routing to the routing-aware rule: large rows take the route the fitted makespan
    # model picks (pair + tail, mixed schedule or GEMM + wide act) unless the legacy odd-tail rule's route is within
    # ROUTE_MODEL_MARGIN of it
    expected_route = select_route(
        m, n2, sm_count=sm_count, group_blocks=routing_blocks(group_counts), k=k
    )
    assert prepared.route == expected_route
    # GEMM + act kernel, or pair kernel + PDL tail kernel; the mixed-schedule route is one kernel
    assert prepared.num_kernels == (1 if prepared.route == FUSED_MIXED_ROUTE else 2)
    if m < SMALL_M_MAX:
        assert prepared.route == _small_m_route(n2, k, routing_blocks(group_counts))
    if prepared.route in ACT_ROUTES:
        assert prepared.gemm_backend == small_m_gemm_backend(m, n2, k)
        assert prepared.gemm_backend == (
            GEMM_BACKEND_CAKE if (m % 128 and k >= 1024) else GEMM_BACKEND_CUTE
        )
        assert prepared.tail_grid is None
    elif prepared.route == FUSED_ROUTE:
        assert prepared.gemm_backend is None
        assert prepared.tail_grid is not None and prepared.tail_grid[0] >= 1
    else:
        assert prepared.route == FUSED_MIXED_ROUTE
        assert prepared.gemm_backend is None
        assert prepared.tail_grid is None and set(prepared.stage_grids) == {"main"}
    out_q, out_s = prepared.launch()
    torch.cuda.synchronize()
    g, u = _reference_halves(_reference_gemm(a, b, a_scale, b_scale, m_indices))
    _assert_quantizes_reference(out_q, out_s, g, u)
    # Contents may change between launches of the same prepared object.
    refill = torch.Generator(device=device).manual_seed(seed + 1)
    a.copy_(
        torch.randn(a.shape, generator=refill, device=device).to(torch.float8_e4m3fn)
    )
    out_q2, out_s2 = prepared()
    torch.cuda.synchronize()
    assert (
        out_q2.data_ptr() == out_q.data_ptr() and out_s2.data_ptr() == out_s.data_ptr()
    )
    g, u = _reference_halves(_reference_gemm(a, b, a_scale, b_scale, m_indices))
    _assert_quantizes_reference(out_q2, out_s2, g, u)


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
    chain_q, chain_s = _chain_or_skip(a, b, a_scale, b_scale, m_indices)
    out_q, out_s = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices
    ).launch()
    torch.cuda.synchronize()
    _assert_matches(out_q, out_s, chain_q, chain_s)
    # with the routing known, the routing-aware rule may pick the GEMM + act or the mixed-schedule route instead
    out_q, out_s = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    ).launch()
    torch.cuda.synchronize()
    _assert_matches(out_q, out_s, chain_q, chain_s)


@pytest.mark.parametrize(
    "group_counts,n2,k,expected_route",
    [
        pytest.param([256] * 16, 2048, 4096, FUSED_ROUTE, id="fused_wide"),
        pytest.param([256] * 8, 256, 512, FUSED_ROUTE, id="fused_at_threshold"),
        pytest.param([256] * 4 + [128, 100], 512, 1024, ACT_ROUTE, id="small_m"),
        # routing-aware rule: the expected route is whatever the makespan-model rule selects
        pytest.param(
            [256, 0, 512, 0, 384, 640, 128, 384, 256, 0, 128, 128, 384, 384, 512, 0],
            2048,
            4096,
            None,
            id="wide_random_aligned",
        ),
        pytest.param([384] * 8 + [128] * 8, 2048, 4096, None, id="wide_all_odd"),
        pytest.param([128] * 16, 2048, 4096, None, id="wide_all_one_block"),
        pytest.param(
            [
                640,
                128,
                384,
                0,
                896,
                128,
                384,
                256,
                128,
                0,
                384,
                128,
                256,
                128,
                128,
                128,
            ],
            2048,
            4096,
            None,
            id="wide_mixed_tail",
        ),
    ],
)
def test_launch_plan_matches_prepared(group_counts, n2, k, expected_route):
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts, n2, k, seed=664, device=device
    )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    arch = device_arch(torch.cuda.current_device())
    m = sum(group_counts)
    blocks = routing_blocks(group_counts)
    # without the routing the plan is shape-only; with it (and K) the routing-aware rule applies,
    # and with the device's architecture the CTA-cap rule of the mixed-schedule route
    shape_route, _ = launch_plan(m, n2, sm_count=sm_count, k=k, arch=arch)
    assert shape_route == select_route(m, n2, sm_count=sm_count, k=k)
    if expected_route is not None:
        assert shape_route == expected_route
    route, grid = launch_plan(
        m, n2, sm_count=sm_count, group_blocks=blocks, k=k, arch=arch
    )
    rule_route, cta_cap = select_route_and_cap(
        m, n2, sm_count=sm_count, group_blocks=blocks, k=k, arch=arch
    )
    assert route == rule_route
    if cta_cap == CTA_CAP:
        assert route == select_route(m, n2, sm_count=sm_count, group_blocks=blocks, k=k)
    else:
        assert route == FUSED_MIXED_ROUTE
        assert cta_cap in MIXED_CTA_CAPS and cta_cap <= sm_count
    if m < SMALL_M_MAX:
        assert route == _small_m_route(n2, k, blocks)
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    )
    assert (prepared.route, prepared.grid) == (route, grid)
    unvalidated = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices
    )
    assert (unvalidated.route, unvalidated.grid) == launch_plan(
        m, n2, sm_count=sm_count, k=k
    )
    if route == FUSED_ROUTE:
        assert grid[0] % 2 == 0 and grid[0] <= 128
        # with the routing known the tail grid covers the odd-tail units (at least
        # one CTA); without it, every block
        odd_units = fused_tile_counts(n2, blocks)[1]
        tail = tail_launch_grid(m, n2, sm_count=sm_count, odd_units=odd_units)
        assert prepared.tail_grid == tail
        assert tail[0] == max(1, min(odd_units, sm_count))
        assert set(prepared.stage_grids) == {"pair", "tail"}
        if unvalidated.route == FUSED_ROUTE:
            assert unvalidated.tail_grid == tail_launch_grid(m, n2, sm_count=sm_count)
    elif route == FUSED_MIXED_ROUTE:
        pair_tiles, odd_units = fused_tile_counts(n2, blocks)
        clusters = mixed_clusters(
            pair_tiles, odd_units, sm_count=sm_count, cta_cap=cta_cap
        )
        assert clusters >= 1  # one cluster per pair tile or solo unit up to the cap
        assert grid == (2 * clusters, 1, 1) and grid[0] <= cta_cap
        assert prepared.tail_grid is None
        assert set(prepared.stage_grids) == {"main"}
        assert prepared.num_kernels == 1
    else:
        assert prepared.tail_grid is None
        assert route in ACT_ROUTES
        items = act_items(m, n2)
        if route == ACT_WIDE_ROUTE:
            assert m >= SMALL_M_MAX and items >= ACT_WIDE_MIN_ITEMS
            assert 1 <= grid[0] <= max(1, -(-items // 16))
            assert grid[0] <= 10 * sm_count
        else:
            assert 1 <= grid[0] <= max(1, -(-items // 4))
            assert grid[0] <= 16 * sm_count


def test_routing_blocks_and_fused_tile_counts():
    counts = [256, 0, 512, 0, 384, 640, 128, 384, 256, 0, 128, 128, 384, 384, 512, 0]
    blocks = routing_blocks(counts)
    assert blocks == (2, 0, 4, 0, 3, 5, 1, 3, 2, 0, 1, 1, 3, 3, 4, 0)
    assert sum(blocks) == sum(counts) // 128
    # N2 = 2048 -> eight N256 tiles: 12 complete pairs and 8 odd tail blocks
    assert fused_tile_counts(2048, blocks) == (96, 64)
    assert fused_tile_counts(2048, routing_blocks([256] * 16)) == (128, 0)
    assert fused_tile_counts(2048, routing_blocks([128] * 16)) == (0, 128)
    assert routing_blocks([100]) == (1,)  # a partial final block counts as one block
    with pytest.raises(ValueError):
        routing_blocks([128, -1])
    with pytest.raises(ValueError):
        select_route(4096, 2048, sm_count=148, group_blocks=(1, 1))
    assert (
        select_route(1024, 2048, sm_count=148, group_blocks=routing_blocks([128] * 8))
        == ACT_ROUTE
    )
    assert select_route(4096, 2048, sm_count=148) == FUSED_ROUTE
    # B200 rule at the model's calibration geometry (2H = 2048, K = 4096): the fitted makespan model ranks pair + tail,
    # mixed schedule and GEMM + act, guarded by the legacy odd-tail rule (units beyond the 20 SMs the 128-CTA pair grid
    # leaves free leave the pair + tail route)
    wide = dict(sm_count=148, k=4096)
    assert (
        select_route(4096, 2048, group_blocks=routing_blocks([256] * 16), **wide)
        == FUSED_ROUTE
    )
    two_odd = routing_blocks(
        [128, 384] + [256] * 14
    )  # 16 odd-tail units <= 20 free SMs
    assert select_route(4096, 2048, group_blocks=two_odd, **wide) == FUSED_ROUTE
    # diverted wide rows: the mixed-schedule route (random_aligned: 96 pair tiles + 64 solo units over 64 clusters,
    # mixed_tail: 80 + 96), as under the legacy rule
    assert fused_tile_counts(2048, blocks) == (96, 64)
    assert mixed_clusters(96, 64, sm_count=148) == 64
    assert select_route(4096, 2048, group_blocks=blocks, **wide) == FUSED_MIXED_ROUTE
    mixed_tail = routing_blocks(
        [640, 128, 384, 0, 896, 128, 384, 256, 128, 0, 384, 128, 256, 128, 128, 128]
    )
    assert fused_tile_counts(2048, mixed_tail) == (80, 96)
    assert (
        select_route(4096, 2048, group_blocks=mixed_tail, **wide) == FUSED_MIXED_ROUTE
    )
    # all_odd (64 pair tiles = one per cluster) and all_one_block (no pair tile at all) left the legacy rule's
    # round-10 envelope on the GEMM + wide act route; the makespan model sends them to the mixed schedule (measured
    # 0.97x and 0.87x the act route on B200)
    all_odd = routing_blocks([384] * 8 + [128] * 8)
    assert fused_tile_counts(2048, all_odd) == (64, 128)
    assert select_route(4096, 2048, group_blocks=all_odd, **wide) == FUSED_MIXED_ROUTE
    all_one_block = routing_blocks([128] * 16)
    assert fused_tile_counts(2048, all_one_block) == (0, 128)
    assert (
        select_route(2048, 2048, group_blocks=all_one_block, **wide)
        == FUSED_MIXED_ROUTE
    )
    # the margin guard keeps the legacy route where the model is within 5 % of its pick: twelve odd experts at
    # M = 3072 stay on the wide act route; the four-odd M = 8192 routing goes back to pair + tail (legacy: mixed)
    twelve_odd_3072 = routing_blocks([384] * 6 + [128] * 6 + [0] * 4)
    assert fused_tile_counts(2048, twelve_odd_3072) == (48, 96)
    assert (
        select_route(3072, 2048, group_blocks=twelve_odd_3072, **wide) == ACT_WIDE_ROUTE
    )
    four_odd_8192 = routing_blocks([640, 640, 384, 384] + [512] * 12)
    assert fused_tile_counts(2048, four_odd_8192) == (240, 32)
    assert select_route(8192, 2048, group_blocks=four_odd_8192, **wide) == FUSED_ROUTE
    assert {FUSED_ROUTE, FUSED_MIXED_ROUTE} == FUSED_ROUTES
    assert act_items(2048, 2048) == ACT_WIDE_MIN_ITEMS
    # outside the calibration geometry the legacy rule applies unchanged: 16 odd-tail units (<= 20 free SMs) stay on
    # the pair + tail route, 32 units below the item threshold take the one-warp-per-group act kernel
    assert (
        select_route(2048, 256, group_blocks=all_one_block, sm_count=148, k=512)
        == FUSED_ROUTE
    )
    assert act_items(2048, 512) < ACT_WIDE_MIN_ITEMS
    assert (
        select_route(2048, 512, group_blocks=all_one_block, sm_count=148, k=1024)
        == ACT_ROUTE
    )
    # without K (existing callers) or at another K the same routing keeps the legacy route
    assert (
        select_route(4096, 2048, group_blocks=all_odd, sm_count=148) == ACT_WIDE_ROUTE
    )
    assert (
        select_route(4096, 2048, group_blocks=all_odd, sm_count=148, k=2048)
        == ACT_WIDE_ROUTE
    )
    # grids: the mixed-schedule route on the wide random_aligned row and on all_one_block (solo units only), the two
    # act routes on the twelve-odd M = 3072 row and a small row
    _, mixed_grid = launch_plan(4096, 2048, group_blocks=blocks, **wide)
    assert mixed_grid == (
        128,
        1,
        1,
    )  # 64 clusters of two CTAs (160 units capped at the 128-CTA grid)
    _, solo_grid = launch_plan(2048, 2048, group_blocks=all_one_block, **wide)
    assert solo_grid == (128, 1, 1)  # 128 solo units capped at the 64 clusters
    _, wide_grid = launch_plan(3072, 2048, group_blocks=twelve_odd_3072, **wide)
    assert wide_grid == (
        1480,
        1,
        1,
    )  # ceil(24576 / 16) = 1536 warps-of-four capped at 10 CTAs x 148 SMs
    _, small_grid = launch_plan(
        1024, 2048, group_blocks=routing_blocks([128] * 8), **wide
    )
    assert small_grid == (2048, 1, 1)  # 8192 items, one warp each, four per CTA


# Picks of the CTA-cap rule on the measured MoE rows at [2H = 2048, K = 4096] per device model; rows not listed
# keep select_route's route at CTA_CAP.
CTA_CAP_RULE_PICKS: dict[tuple[str, int], dict[str, tuple[str, int]]] = {
    ("sm_100a", 148): {
        "m8192_sixteen_odd": (FUSED_MIXED_ROUTE, 148),
    },
    ("sm_103a", 152): {
        "m8192_sixteen_odd": (FUSED_MIXED_ROUTE, 148),
    },
}
CTA_CAP_RULE_ROWS = {
    "wide_ep32_gate_up_uniform": [256] * 16,
    "wide_ep32_gate_up_random_aligned": [256, 0, 512, 0, 384, 640, 128, 384]
    + [256, 0, 128, 128, 384, 384, 512, 0],
    "wide_ep32_gate_up_all_odd": [384] * 8 + [128] * 8,
    "wide_ep32_gate_up_all_one_block": [128] * 16,
    "wide_ep32_gate_up_mixed_tail": [640, 128, 384, 0, 896, 128, 384, 256]
    + [128, 0, 384, 128, 256, 128, 128, 128],
    "wide_ep32_gate_up_two_odd": [128, 384] + [256] * 14,
    "m8192_max": [512] * 16,
    "m8192_sixteen_odd": [640] * 8 + [384] * 8,
}


def test_select_route_and_cap_rule():
    """The CTA-cap rule moves a routing only to the mixed-schedule route above 128 CTAs, only on a
    device with a model, at the model's geometry with the routing known; every other call keeps
    select_route's route at 128 CTAs."""
    rng = random.Random(1356)
    routings = []
    for _ in range(400):
        blocks = [0] * 16
        for _ in range(rng.randint(8, 64)):
            blocks[rng.randrange(16)] += 1
        n2, k = rng.choice([(2048, 4096)] * 6 + [(2048, 2048), (1024, 4096)])
        routings.append(([128 * b for b in blocks], n2, k))
    routings += [(counts, 2048, 4096) for counts in CTA_CAP_RULE_ROWS.values()]
    devices = dict.fromkeys([*ROUTE_MODELS_CAP, ("sm_100a", 148), ("sm_103a", 152)])
    for arch, sm_count in devices:
        changed = 0
        for counts, n2, k in routings:
            m = sum(counts)
            blocks = routing_blocks(counts)
            route, cap = select_route_and_cap(
                m, n2, sm_count=sm_count, group_blocks=blocks, k=k, arch=arch
            )
            base = select_route(m, n2, sm_count=sm_count, group_blocks=blocks, k=k)
            if cap == CTA_CAP:
                assert route == base
            else:
                changed += 1
                assert (arch, sm_count) in ROUTE_MODELS_CAP
                assert route == FUSED_MIXED_ROUTE
                assert cap in MIXED_CTA_CAPS and cap <= sm_count
                assert m >= SMALL_M_MAX and (n2, k) == ROUTE_MODEL_GEOMETRY
                pair_tiles, odd_units = fused_tile_counts(n2, blocks)
                clusters = mixed_clusters(
                    pair_tiles, odd_units, sm_count=sm_count, cta_cap=cap
                )
                assert launch_plan(
                    m, n2, sm_count=sm_count, group_blocks=blocks, k=k, arch=arch
                ) == (FUSED_MIXED_ROUTE, (2 * clusters, 1, 1))
            # no architecture, no routing, no K or another SM count: no CTA-cap rule
            assert select_route_and_cap(
                m, n2, sm_count=sm_count, group_blocks=blocks, k=k
            ) == (base, CTA_CAP)
            assert select_route_and_cap(
                m, n2, sm_count=sm_count, group_blocks=blocks, arch=arch
            ) == (select_route(m, n2, sm_count=sm_count, group_blocks=blocks), CTA_CAP)
            assert select_route_and_cap(m, n2, sm_count=sm_count, k=k, arch=arch) == (
                select_route(m, n2, sm_count=sm_count, k=k),
                CTA_CAP,
            )
            assert select_route_and_cap(
                m, n2, sm_count=sm_count - 16, group_blocks=blocks, k=k, arch=arch
            ) == (
                select_route(m, n2, sm_count=sm_count - 16, group_blocks=blocks, k=k),
                CTA_CAP,
            )
        if (arch, sm_count) not in ROUTE_MODELS_CAP:
            assert changed == 0
    for (arch, sm_count), picks in CTA_CAP_RULE_PICKS.items():
        for name, counts in CTA_CAP_RULE_ROWS.items():
            m = sum(counts)
            blocks = routing_blocks(counts)
            base = select_route(m, 2048, sm_count=sm_count, group_blocks=blocks, k=4096)
            assert select_route_and_cap(
                m, 2048, sm_count=sm_count, group_blocks=blocks, k=4096, arch=arch
            ) == picks.get(name, (base, CTA_CAP))


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
    # Tensor maps travel by value: the very first launch is captured, no eager warm-up launch.
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        prepared.launch()
    prepared.out_q.view(torch.uint8).fill_(0xFF)
    prepared.out_s.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    replay_q = prepared.out_q.clone()
    replay_s = prepared.out_s.clone()
    prepared.launch()
    torch.cuda.synchronize()
    # the replay reproduces the eager launch bit for bit (same kernels, same inputs) ...
    assert torch.equal(prepared.out_q.view(torch.uint8), replay_q.view(torch.uint8))
    assert torch.equal(prepared.out_s, replay_s)
    # ... and matches the chain where the chain can run
    ref_q, ref_s = _chain_or_skip(a, b, a_scale, b_scale, m_indices)
    _assert_matches(prepared.out_q, prepared.out_s, ref_q, ref_s)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
@pytest.mark.parametrize(
    "group_counts,n2,k,expected_route",
    [
        pytest.param([256, 256], 256, 512, ACT_ROUTE, id="act_cute_gemm"),
        pytest.param([128, 128, 100], 256, 1024, ACT_ROUTE, id="act_cake_gemm"),
        pytest.param([128], 256, 512, FUSED_MIXED_ROUTE, id="tiny_mixed"),
        pytest.param([256] * 8 + [128], 256, 512, FUSED_ROUTE, id="pair_tail"),
    ],
)
def test_launch_on_non_current_device(group_counts, n2, k, expected_route):
    """launch() enters no device or stream context: with cuda:0 current, a launch of
    a cuda:1 problem runs on cuda:1's current stream (eager, on a side stream and
    under a graph capture) and writes the same bits as a launch under
    ``torch.cuda.device(1)``."""
    dev1 = torch.device("cuda", 1)
    if torch.cuda.get_device_capability(dev1) not in SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip("cuda:1 is not an SM100a / SM103a device")
    with torch.cuda.device(0):
        a, b, a_scale, b_scale, m_indices = _make_inputs(
            group_counts, n2, k, seed=669, device=dev1
        )
        prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            a, b, a_scale, b_scale, m_indices, group_counts=group_counts
        )
        assert prepared.route == expected_route
        out_q, out_s = prepared.out_q, prepared.out_s

        def poison():
            out_q.view(torch.uint8).fill_(0xFF)
            out_s.fill_(float("nan"))
            torch.cuda.synchronize(dev1)

        def assert_same_bits(ref_q, ref_s):
            assert torch.equal(out_q.view(torch.uint8), ref_q.view(torch.uint8))
            assert torch.equal(out_s.view(torch.int32), ref_s.view(torch.int32))

        with torch.cuda.device(1):
            prepared.launch()
            torch.cuda.synchronize()
            g, u = _reference_halves(_reference_gemm(a, b, a_scale, b_scale, m_indices))
            _assert_quantizes_reference(out_q, out_s, g, u)
        ref_q, ref_s = out_q.clone(), out_s.clone()

        # eager, cuda:0 current
        poison()
        prepared.launch()
        assert torch.cuda.current_device() == 0
        torch.cuda.synchronize(dev1)
        assert_same_bits(ref_q, ref_s)

        # side stream of cuda:1 made current there, cuda:0 current again
        poison()
        side = torch.cuda.Stream(device=dev1)
        side.wait_stream(torch.cuda.current_stream(dev1))
        with torch.cuda.stream(side):
            torch.cuda.set_device(0)
            prepared.launch()
            torch.cuda.set_device(1)
        side.synchronize()
        assert_same_bits(ref_q, ref_s)

        # graph capture on cuda:1 with cuda:0 current during the launch
        graph = torch.cuda.CUDAGraph()
        capture = torch.cuda.Stream(device=dev1)
        with torch.cuda.graph(graph, stream=capture):
            torch.cuda.set_device(0)
            prepared.launch()
            torch.cuda.set_device(1)
        poison()
        graph.replay()
        torch.cuda.synchronize(dev1)
        assert_same_bits(ref_q, ref_s)
        del graph


def _random_aligned_counts(rng, groups, max_blocks):
    """Random rows per expert under the routing contract: 128-row multiples (empties allowed) and
    one optional partial final block; the total never exceeds ``max_blocks`` blocks."""
    blocks = [0] * groups
    for _ in range(rng.randint(1, max_blocks)):
        blocks[rng.randrange(groups)] += 1
    counts = [b * 128 for b in blocks]
    last = max(i for i, c in enumerate(counts) if c)
    counts[last] -= rng.choice(
        (0, 0, rng.randint(1, 127))
    )  # partial final block on some rows
    return counts


RANDOM_TOKEN_CASES = [
    pytest.param(
        seed,
        groups,
        max_blocks,
        n2,
        k,
        id=f"s{seed}_g{groups}_b{max_blocks}_n{n2}_k{k}",
    )
    for seed, groups, max_blocks, n2, k in (
        (1, 4, 8, 256, 512),
        (2, 8, 16, 512, 1024),
        (3, 16, 24, 2048, 4096),
        (4, 16, 40, 2048, 4096),
        (5, 16, 64, 2048, 4096),
        (6, 6, 12, 768, 2048),
        (7, 32, 64, 1024, 1024),
        (8, 16, 33, 2048, 4096),
    )
]


@pytest.mark.parametrize("seed,groups,max_blocks,n2,k", RANDOM_TOKEN_CASES)
def test_random_token_counts_match_torch_reference(seed, groups, max_blocks, n2, k):
    device = torch.device("cuda")
    rng = random.Random(seed)
    group_counts = _random_aligned_counts(rng, groups, max_blocks)
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts,
        n2,
        k,
        seed=670 + seed,
        device=device,
        arbitrary_scales=bool(seed % 2),
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    )
    out_q, out_s = prepared.launch()
    torch.cuda.synchronize()
    g, u = _reference_halves(_reference_gemm(a, b, a_scale, b_scale, m_indices))
    _assert_quantizes_reference(out_q, out_s, g, u)


@pytest.mark.parametrize(
    "group_counts,n2,k",
    [
        pytest.param([256] * 16, 2048, 4096, id="uniform"),
        pytest.param([384] * 8 + [128] * 8, 2048, 4096, id="all_odd"),
        pytest.param([128] * 16, 2048, 4096, id="all_one_block"),
        pytest.param([128, 128, 100], 256, 1024, id="small_m_partial_tail"),
    ],
)
def test_group_counts_route_without_device_work(group_counts, n2, k):
    """Caller-supplied rows per expert select the routing-aware route without touching m_indices."""
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts, n2, k, seed=668, device=device
    )
    validated = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices, validate_indices=True
    )
    from_counts = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices, group_counts=group_counts
    )
    assert (from_counts.route, from_counts.grid, from_counts.stage_grids) == (
        validated.route,
        validated.grid,
        validated.stage_grids,
    )
    both = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a,
        b,
        a_scale,
        b_scale,
        m_indices,
        group_counts=group_counts,
        validate_indices=True,
    )
    assert both.route == validated.route
    out_q, out_s = from_counts.launch()
    torch.cuda.synchronize()
    g, u = _reference_halves(_reference_gemm(a, b, a_scale, b_scale, m_indices))
    _assert_quantizes_reference(out_q, out_s, g, u)


def test_caller_owned_outputs_and_workspace():
    device = torch.device("cuda")
    group_counts, n2, k = [128, 128, 100], 256, 1024
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts, n2, k, seed=669, device=device
    )
    m, h = sum(group_counts), n2 // 2
    out_q = torch.empty((m, h), dtype=torch.float8_e4m3fn, device=device)
    out_s = torch.empty((m, h // GROUP_SIZE), dtype=torch.float32, device=device)
    workspace = torch.empty((m, n2), dtype=torch.bfloat16, device=device)
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices, out_q=out_q, out_s=out_s, workspace=workspace
    )
    assert prepared.route in ACT_ROUTES
    q, s = prepared.launch()
    torch.cuda.synchronize()
    assert q.data_ptr() == out_q.data_ptr() and s.data_ptr() == out_s.data_ptr()
    g, u = _reference_halves(_reference_gemm(a, b, a_scale, b_scale, m_indices))
    _assert_quantizes_reference(out_q, out_s, g, u)
    torch.testing.assert_close(
        workspace.float(),
        _reference_gemm(a, b, a_scale, b_scale, m_indices).float(),
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.parametrize(
    "group_counts,n2,k",
    [
        pytest.param([128, 128, 100], 256, 1024, id="small_m_cake_gemm"),
        pytest.param([128, 256], 256, 512, id="small_m_cute_gemm"),
        pytest.param([256] * 8, 256, 512, id="fused"),
        pytest.param([128] * 16, 2048, 4096, id="wide_all_one_block"),
    ],
)
def test_launch_rebinds_per_call_operands(group_counts, n2, k):
    """One prepared object per layer: the token operands and outputs are swapped per call without copies."""
    device = torch.device("cuda")
    a, b, a_scale, b_scale, m_indices = _make_inputs(
        group_counts, n2, k, seed=672, device=device
    )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices, group_counts=group_counts
    )
    a2, _, a_scale2, _, _ = _make_inputs(group_counts, n2, k, seed=673, device=device)
    m_indices2 = m_indices.clone()
    out_q2 = torch.empty_like(prepared.out_q)
    out_s2 = torch.empty_like(prepared.out_s)
    rebound = dict(
        a=a2, a_scale=a_scale2, m_indices=m_indices2, out_q=out_q2, out_s=out_s2
    )
    q, s = prepared.launch(**rebound)
    assert q is out_q2 and s is out_s2
    torch.cuda.synchronize()
    g, u = _reference_halves(_reference_gemm(a2, b, a_scale2, b_scale, m_indices2))
    _assert_quantizes_reference(out_q2, out_s2, g, u)
    # The prepared operands are untouched and still bound.
    q, s = prepared.launch()
    assert q is prepared.out_q and s is prepared.out_s
    torch.cuda.synchronize()
    g, u = _reference_halves(_reference_gemm(a, b, a_scale, b_scale, m_indices))
    _assert_quantizes_reference(q, s, g, u)
    # A captured rebound launch replays onto the rebound tensors.
    expected_q, expected_s = out_q2.clone(), out_s2.clone()
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        prepared.launch(**rebound)
    out_q2.view(torch.uint8).fill_(0xFF)
    out_s2.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_q2.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(out_s2, expected_s)
    with pytest.raises(ValueError, match="a must have shape"):
        prepared.launch(a=a2[:128].contiguous())
    with pytest.raises(ValueError, match="out_s must have shape"):
        prepared.launch(out_s=out_s2[:, :0].contiguous())


@pytest.mark.parametrize(
    "real_counts,n2,k",
    [
        pytest.param([100, 128, 5, 0, 250], 256, 512, id="small_m"),
        pytest.param(
            [200, 0, 37, 256, 129, 300, 1, 255, 128, 64, 99, 180],
            256,
            512,
            id="large_m",
        ),
    ],
)
def test_fill_padding_accepts_compact_layout(real_counts, n2, k):
    """-1 padding rows (compact MoE layout) are forward-filled onto the preceding expert."""
    device = torch.device("cuda")
    padded = [-(-c // 128) * 128 for c in real_counts]
    a, b, a_scale, b_scale, _ = _make_inputs(padded, n2, k, seed=674, device=device)
    rows = []
    for g, (c, p) in enumerate(zip(real_counts, padded, strict=True)):
        rows += [g] * c + [-1] * (p - c)
    m_indices = torch.tensor(rows, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="fill_padding"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            a, b, a_scale, b_scale, m_indices, validate_indices=True
        )
    unsorted = m_indices.clone()
    unsorted[: real_counts[0]] = len(real_counts) - 1
    with pytest.raises(ValueError, match="m_indices must be sorted"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            a,
            b,
            a_scale,
            b_scale,
            unsorted,
            validate_indices=True,
            fill_padding=True,
        )
    prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices, validate_indices=True, fill_padding=True
    )
    # group_counts count the padding rows that follow each expert
    from_counts = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
        a, b, a_scale, b_scale, m_indices, group_counts=padded, fill_padding=True
    )
    assert from_counts.route == prepared.route
    filled = torch.clamp(torch.cummax(m_indices, 0).values, min=0)
    valid = m_indices >= 0
    out_q, out_s = prepared.launch()
    torch.cuda.synchronize()
    g, u = _reference_halves(_reference_gemm(a, b, a_scale, b_scale, filled))
    _assert_quantizes_reference(out_q[valid], out_s[valid], g[valid], u[valid])
    # Graph capture includes the two forward-fill kernels.
    expected_q, expected_s = out_q.clone(), out_s.clone()
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        prepared.launch()
    out_q.view(torch.uint8).fill_(0xFF)
    out_s.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(
        out_q[valid].view(torch.uint8), expected_q[valid].view(torch.uint8)
    )
    assert torch.equal(out_s[valid], expected_s[valid])


def test_re_preparing_per_step_retains_no_device_memory():
    """A new routing per step re-prepares the launch on the fused routes; nothing accumulates."""
    device = torch.device("cuda")
    rng = random.Random(671)
    n2, k, groups = 256, 512, 8
    b = torch.randn((groups, n2, k), device=device).to(torch.float8_e4m3fn)
    b_scale = torch.ones((groups, n2 // 128, k // 128), device=device)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    for _ in range(12):
        counts = _random_aligned_counts(rng, groups, 32)
        counts = [
            c - c % 128 for c in counts
        ]  # fused routes: whole blocks, M >= SMALL_M_MAX
        while sum(counts) < SMALL_M_MAX:
            counts[rng.randrange(groups)] += 128
        m = sum(counts)
        a = torch.randn((m, k), device=device).to(torch.float8_e4m3fn)
        a_scale = torch.ones((m, k // 128), device=device)
        m_indices = torch.repeat_interleave(
            torch.arange(groups, dtype=torch.int32, device=device),
            torch.tensor(counts, dtype=torch.int64, device=device),
        ).contiguous()
        prepared = prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            a, b, a_scale, b_scale, m_indices, group_counts=counts
        )
        assert prepared.route in FUSED_ROUTES
        out_q, out_s = prepared.launch()
        torch.cuda.synchronize()
        assert torch.isfinite(out_s).all()
        del prepared, out_q, out_s, a, a_scale, m_indices
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated(device) == baseline


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
    with pytest.raises(ValueError, match="group_counts must"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            a, b, a_scale, b_scale, m_indices, group_counts=[100, 156]
        )
    with pytest.raises(ValueError, match="disagree with m_indices"):
        prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            a,
            b,
            a_scale,
            b_scale,
            m_indices,
            group_counts=[256, 0],
            validate_indices=True,
        )
    with pytest.raises(ValueError, match="M must be at most 8192"):
        big_a, big_b, big_as, big_bs, big_idx = _make_inputs(
            [8320], 256, 512, seed=667, device=device
        )
        prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
            big_a, big_b, big_as, big_bs, big_idx
        )
