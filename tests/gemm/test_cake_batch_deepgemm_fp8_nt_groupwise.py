# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the generated Cake programs behind ``batch_deepgemm_fp8_nt_groupwise(backend="cake")``."""

import ctypes
import functools
import gc
import os
import time

import pytest
import torch

from flashinfer import cake_locality
from flashinfer.gemm import batch_deepgemm_fp8_nt_groupwise
from flashinfer.gemm.cake_batch_deepgemm_fp8 import (
    DOMAIN_SERVING_ROUTE,
    PACKED_ARCHS,
    PACKED_GROUPS,
    PACKED_NK,
    PAIR_COUNTER_WORDS,
    SUPPORTED_NK,
    domain_serving_program,
    domain_serving_routed,
    expected_m_class,
    is_batch_deepgemm_fp8_nt_groupwise_cake_available,
    localize_weights,
    prepare_batch_deepgemm_fp8_nt_groupwise,
    select_route,
    serving_config,
    serving_plan,
    serving_program_key,
    supported_nk,
)
from flashinfer.jit.gemm.cake_batch_deepgemm_fp8 import (
    DOMAIN_SERVING_CELLS,
    DOMAIN_SERVING_PROGRAMS,
    SERVING_PROGRAMS,
    _SM100A_SM103A_MMSETUP_TWINS,
    _SM107A_CELLS,
    _SM107A_MMSETUP_TWINS,
    _TABLE,
    _cell_tag,
    _em4,
    _matches,
    PROGRAMS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    device_arch,
    device_sm_count,
)
from flashinfer.utils import BackendSupportedError

ATOL = RTOL = 3e-2


def _arch():
    return device_arch(torch.cuda.current_device())


def _require_nk(nk):
    if nk not in supported_nk(_arch()):
        pytest.skip(f"{_arch()} owns no route for (N, K) = {nk}")


def _require_packed():
    if _arch() not in PACKED_ARCHS:
        pytest.skip(f"no packed-scale serving program ships for {_arch()}")


def _packed_serving(test):
    """A test of the packed-scale serving routes alone: on SM107a they take their
    grid from the SM count, so they run on every SM count of that architecture;
    elsewhere they need the device's exported route chain as before."""
    test.packed_serving = True
    return test


def _host_only(test):
    """A test of host-side selection rules: it launches no kernel and runs on any
    device (the CUDA tensors it builds only carry metadata)."""
    test.host_only = True
    return test


@pytest.fixture(autouse=True)
def _require_generated_program(request):
    if getattr(request.function, "host_only", False):
        return
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip(
            "generated Cake batch DeepGEMM FP8 programs target SM100a, SM103a and SM107a"
        )
    if getattr(request.function, "packed_serving", False) and _arch() == "sm_107a":
        _require_packed()
    elif not is_batch_deepgemm_fp8_nt_groupwise_cake_available(device):
        pytest.skip(
            "no generated Cake batch DeepGEMM FP8 program is registered for this device"
        )


def _quantize(value, block_mn, device):
    """Power-of-two block scales (exact in FP32 and UE8M0) of a float32 tensor."""
    batch, rows, k = value.shape
    blocks = value.reshape(batch, rows // block_mn, block_mn, k // 128, 128)
    amax = blocks.abs().amax(dim=(2, 4)).clamp(1e-4)
    scale = torch.pow(
        2.0, torch.ceil(torch.log2(amax / torch.finfo(torch.float8_e4m3fn).max))
    )
    expanded = scale.repeat_interleave(block_mn, dim=1).repeat_interleave(128, dim=2)
    quantized = (value / expanded).to(torch.float8_e4m3fn)
    return quantized.contiguous(), scale.float().contiguous()


def _make_inputs(groups, m, n, k, *, seed, device, masked=None):
    generator = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((groups, m, k), generator=generator, device=device)
    b = torch.randn((groups, n, k), generator=generator, device=device)
    a_fp8, a_scale = _quantize(a, 1, device)
    b_fp8, b_scale = _quantize(b, 128, device)
    if masked is None:
        masked_m = torch.randint(
            0, m + 1, (groups,), generator=generator, device=device, dtype=torch.int32
        )
    else:
        masked_m = torch.tensor(masked, device=device, dtype=torch.int32)
    expected_m = min(int(masked_m.float().mean().item()) + 1, m)
    return a_fp8, b_fp8, a_scale, b_scale, masked_m, expected_m


def _reference(a_fp8, b_fp8, a_scale, b_scale, masked_m):
    groups, m, k = a_fp8.shape
    n = b_fp8.shape[1]
    out = torch.zeros((groups, m, n), dtype=torch.float32, device=a_fp8.device)
    rows_per_group = masked_m.tolist()
    for g, rows in enumerate(rows_per_group):
        if rows <= 0:
            continue
        a = a_fp8[g, :rows].float() * a_scale[g, :rows].repeat_interleave(128, dim=-1)
        b = b_fp8[g].float() * b_scale[g].repeat_interleave(
            128, dim=0
        ).repeat_interleave(128, dim=1)
        out[g, :rows] = a @ b.T
    return out


def _assert_valid_rows_close(out, ref, masked_m):
    for g, rows in enumerate(masked_m.tolist()):
        if rows > 0:
            torch.testing.assert_close(
                out[g, :rows].float(), ref[g, :rows], atol=ATOL, rtol=RTOL
            )


def _pack_ue8m0(scale):
    """DeepGEMM's MN-major packed UE8M0 ABI of power-of-two float32 scales ``[B, MN, K/128]``."""
    batch, mn, sf_k = scale.shape
    aligned_mn = ((mn + 3) // 4) * 4
    aligned_k = ((sf_k + 3) // 4) * 4
    exponents = (scale.view(torch.int32) >> 23).to(torch.uint8)
    padded = torch.zeros(
        (batch, aligned_mn, aligned_k), device=scale.device, dtype=torch.uint8
    )
    padded[:, :mn, :sf_k] = exponents
    packed = padded.view(torch.int32).reshape(batch, aligned_mn, aligned_k // 4)
    out = torch.empty_strided(
        (batch, aligned_mn, aligned_k // 4),
        (aligned_mn * (aligned_k // 4), 1, aligned_mn),
        device=scale.device,
        dtype=torch.int32,
    )
    out.copy_(packed)
    return out[:, :mn, :]


@pytest.mark.parametrize("nk", sorted(SUPPORTED_NK))
@pytest.mark.parametrize("groups,m", [(1, 128), (4, 256), (8, 128)])
def test_prepared_matches_reference(nk, groups, m):
    _require_nk(nk)
    device = torch.device("cuda")
    n, k = nk
    a, b, a_scale, b_scale, masked_m, expected_m = _make_inputs(
        groups, m, n, k, seed=923, device=device
    )
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, masked_m, expected_m
    )
    out = prepared.launch()
    torch.cuda.synchronize()
    assert prepared.programs and all(p in PROGRAMS for p in prepared.programs)
    _assert_valid_rows_close(
        out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )


def test_empty_and_full_groups_match_reference():
    device = torch.device("cuda")
    a, b, a_scale, b_scale, masked_m, expected_m = _make_inputs(
        4, 256, 7168, 2048, seed=924, device=device, masked=[0, 256, 1, 129]
    )
    out = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, masked_m, expected_m
    ).launch()
    torch.cuda.synchronize()
    _assert_valid_rows_close(
        out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )


@pytest.mark.parametrize("nk", [(128, 512), (4096, 7168)])
def test_public_api_backend_cake(nk):
    device = torch.device("cuda")
    n, k = nk
    a, b, a_scale, b_scale, masked_m, expected_m = _make_inputs(
        8, 256, n, k, seed=925, device=device
    )
    out = batch_deepgemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, masked_m, expected_m, backend="cake"
    )
    torch.cuda.synchronize()
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (8, 256, n)
    _assert_valid_rows_close(
        out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )


def test_cuda_graph_capture_of_first_launch():
    _require_nk((4096, 2048))
    device = torch.device("cuda")
    # A swap-AB route (scale-pack kernel + GEMM, packed-scale workspace): the
    # DeepGEMM expected-M 230 profile on (4096, 2048).
    a, b, a_scale, b_scale, masked_m, _ = _make_inputs(
        2, 4096, 4096, 2048, seed=926, device=device, masked=[230, 1000]
    )
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, masked_m, 230
    )
    expected = _reference(a, b, a_scale, b_scale, masked_m)
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    # Tensor maps travel by value and the workspace is bound at preparation: the
    # very first launch is captured, no eager warm-up launch.
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        prepared.launch()
    prepared.out.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    _assert_valid_rows_close(prepared.out, expected, masked_m)
    a.copy_(torch.randn(a.shape, device=device).to(torch.float8_e4m3fn))
    graph.replay()
    torch.cuda.synchronize()
    _assert_valid_rows_close(
        prepared.out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )


def test_public_api_first_call_is_graph_capturable():
    device = torch.device("cuda")
    a, b, a_scale, b_scale, masked_m, expected_m = _make_inputs(
        4, 128, 7168, 2048, seed=927, device=device
    )
    out = torch.empty((4, 128, 7168), dtype=torch.bfloat16, device=device)
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        batch_deepgemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, masked_m, expected_m, out=out, backend="cake"
        )
    out.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    _assert_valid_rows_close(
        out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )


@pytest.mark.parametrize("nk", [(4096, 2048), (4096, 7168)])
def test_launch_allocates_nothing(nk):
    _require_nk(nk)
    device = torch.device("cuda")
    a, b, a_scale, b_scale, masked_m, expected_m = _make_inputs(
        2, 4096, *nk, seed=928, device=device, masked=[230, 230]
    )
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, masked_m, expected_m
    )
    prepared.launch()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    prepared.launch()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    assert after["allocation.all.freed"] == before["allocation.all.freed"]


@_packed_serving
def test_packed_ue8m0_scales_match_reference():
    _require_packed()
    device = torch.device("cuda")
    groups, m, n, k = 32, 256, 7168, 2048
    a, b, a_scale, b_scale, masked_m, _ = _make_inputs(
        groups, m, n, k, seed=929, device=device, masked=[3, 0, 7, 1] * 8
    )
    a_packed = _pack_ue8m0(a_scale)
    b_packed = _pack_ue8m0(b_scale.repeat_interleave(128, dim=1))
    assert a_packed.dtype == torch.int32 and a_packed.stride() == (m * (k // 512), 1, m)
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_packed, b_packed, masked_m, 3
    )
    assert prepared.route == "seed_serving_m32_packed"
    out = prepared.launch()
    torch.cuda.synchronize()
    _assert_valid_rows_close(
        out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )
    public = batch_deepgemm_fp8_nt_groupwise(
        a, b, a_packed, b_packed, masked_m, 3, backend="cake"
    )
    torch.cuda.synchronize()
    _assert_valid_rows_close(
        public, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )


@_packed_serving
@pytest.mark.parametrize("groups", sorted(PACKED_GROUPS))
@pytest.mark.parametrize("nk", sorted(PACKED_NK))
def test_packed_serving_expected_m_one_with_multi_block_groups(nk, groups):
    """``expected_m`` is only a hint: at ``expected_m=1`` a group can still hold
    more than one 32-row M block (here 40 and 33 rows) on the serving route."""
    _require_packed()
    device = torch.device("cuda")
    n, k = nk
    m = 256
    masked = [g % 5 for g in range(groups)]
    masked[9], masked[20] = 40, 33
    a, b, a_scale, b_scale, masked_m, _ = _make_inputs(
        groups, m, n, k, seed=930, device=device, masked=masked
    )
    a_packed = _pack_ue8m0(a_scale)
    b_packed = _pack_ue8m0(b_scale.repeat_interleave(128, dim=1))
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_packed, b_packed, masked_m, 1
    )
    assert prepared.route == "seed_serving_m32_packed"
    out = batch_deepgemm_fp8_nt_groupwise(
        a, b, a_packed, b_packed, masked_m, 1, backend="cake"
    )
    torch.cuda.synchronize()
    _assert_valid_rows_close(
        out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )


_N512_CELL_BN32 = "seed_masked_bn__n512_k128_cell_bn32_k128"


@pytest.mark.parametrize(
    "groups,m,full,route,route_sm107",
    [
        (1, 1024, (0,), "seed_n512_k128__b1_bn32", _N512_CELL_BN32),
        (1, 2048, (0,), "seed_n512_k128__b1_mbn32", _N512_CELL_BN32),
        (4, 1024, (0, 3), "seed_n512_k128__b2_8_bn32", _N512_CELL_BN32),
        (4, 2048, (0, 3), "seed_n512_k128__b2_8_mbn32", _N512_CELL_BN32),
        (16, 1024, (2, 15), "seed_n512_k128__b9_bn128", _N512_CELL_BN32),
        (16, 2048, (2, 15), "seed_n512_k128__b9_mbn128", _N512_CELL_BN32),
        (
            40,
            2048,
            (2, 39),
            "seed_n512_k128__b9_mbn192",
            "seed_masked_bn__n512_k128_cell_bn192_once",
        ),
    ],
)
def test_n512_k128_groups_far_above_expected_m(groups, m, full, route, route_sm107):
    """``expected_m`` is only a hint: groups that fill all M rows at ``expected_m=1``
    on the N=512, K=128 routes still have every valid row written."""
    device = torch.device("cuda")
    masked = [g % 3 for g in range(groups)]
    for g in full:
        masked[g] = m
    a, b, a_scale, b_scale, masked_m, _ = _make_inputs(
        groups, m, 512, 128, seed=933, device=device, masked=masked
    )
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, masked_m, 1
    )
    if _arch() in ("sm_100a", "sm_103a"):  # the N=512, K=128 routes' architectures
        assert prepared.route == route
    elif _arch() == "sm_107a":  # the masked BLOCK_N cells of the same geometry
        assert prepared.parent == "seed_masked_bn"
        assert prepared.route == route_sm107
    else:
        assert prepared.parent == "seed_masked_bn"
    out = prepared.launch()
    torch.cuda.synchronize()
    _assert_valid_rows_close(
        out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )


def test_out_of_band_problems_raise():
    device = torch.device("cuda")
    a, b, a_scale, b_scale, masked_m, expected_m = _make_inputs(
        2, 256, 256, 256, seed=930, device=device
    )
    assert (256, 256) not in SUPPORTED_NK
    assert (
        supported_nk("sm_107a") < SUPPORTED_NK
        and supported_nk("sm_100a") == SUPPORTED_NK
    )
    with pytest.raises(ValueError, match=r"\(N, K\)"):
        batch_deepgemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, masked_m, expected_m, backend="cake"
        )
    with pytest.raises(ValueError, match=r"\(N, K\)"):
        prepare_batch_deepgemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, masked_m, expected_m
        )
    # An in-band geometry with a non-128-aligned M.
    a, b, a_scale, b_scale, masked_m, expected_m = _make_inputs(
        2, 128, 512, 128, seed=931, device=device
    )
    with pytest.raises(ValueError, match="128-aligned"):
        prepare_batch_deepgemm_fp8_nt_groupwise(
            a[:, :64].contiguous(),
            b,
            a_scale[:, :64].contiguous(),
            b_scale,
            masked_m,
            1,
        )
    # Packed scales outside the serving boundary.
    a, b, a_scale, b_scale, masked_m, expected_m = _make_inputs(
        2, 128, 7168, 2048, seed=932, device=device
    )
    assert 2 not in PACKED_GROUPS and (7168, 2048) in PACKED_NK
    assert "sm_107a" in PACKED_ARCHS
    with pytest.raises(ValueError, match="packed UE8M0"):
        prepare_batch_deepgemm_fp8_nt_groupwise(
            a,
            b,
            _pack_ue8m0(a_scale),
            _pack_ue8m0(b_scale.repeat_interleave(128, dim=1)),
            masked_m,
            expected_m,
        )
    # The float32 path never admits a mixed scale dtype pair.
    with pytest.raises(ValueError, match="int32"):
        prepare_batch_deepgemm_fp8_nt_groupwise(
            a, b, _pack_ue8m0(a_scale), b_scale, masked_m, expected_m
        )
    # Architectures the programs are not built for are rejected by the backend requirement.
    assert not batch_deepgemm_fp8_nt_groupwise.is_backend_supported("cake", 90)
    assert not batch_deepgemm_fp8_nt_groupwise.is_backend_supported("cake", 120)
    assert batch_deepgemm_fp8_nt_groupwise.is_backend_supported("cake", 100)
    assert batch_deepgemm_fp8_nt_groupwise.is_backend_supported("cake", 103)
    assert batch_deepgemm_fp8_nt_groupwise.is_backend_supported("cake", 107)
    # A geometry another architecture owns is out of the band on SM107a.
    if _arch() == "sm_107a":
        a, b, a_scale, b_scale, masked_m, expected_m = _make_inputs(
            2, 256, 4096, 2048, seed=934, device=device
        )
        with pytest.raises(ValueError, match=r"\(N, K\)"):
            prepare_batch_deepgemm_fp8_nt_groupwise(
                a, b, a_scale, b_scale, masked_m, expected_m
            )
    with pytest.raises(BackendSupportedError):
        batch_deepgemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, masked_m, expected_m, backend="nonexistent"
        )


# The 256-byte L2 promotion of the N=4096, K=7168 band on sm_103a, scoped to problems of at most two M tiles per group:
# arch -> (promoted program, the band's program, in-class (groups, m, expected_m), just outside).
_L2_SCOPE = {
    "sm_103a": (
        "cake_batch_deepgemm_fp8_d5517b631a2ffbb3e9eb",
        "cake_batch_deepgemm_fp8_0db53ce746a8da419343",
        ((4, 768, 192), (12, 768, 192)),
        ((2, 1024, 512),),
    ),
}


def test_l2_promotion_takes_only_problems_of_at_most_two_m_tiles():
    device = torch.device("cuda")
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = device_arch(index)
    sm_count = device_sm_count(index)
    if arch not in _L2_SCOPE:
        pytest.skip(f"no L2-promoted band program on {arch}")
    promoted, band, inside, outside = _L2_SCOPE[arch]
    for groups, m, expected_m in inside + outside:
        plan = select_route(arch, sm_count, groups, m, 4096, 7168, expected_m)
        want = promoted if (groups, m, expected_m) in inside else band
        assert plan is not None and [s.program for s in plan.stages] == [want], (
            groups,
            m,
            expected_m,
        )
        a, b, a_scale, b_scale, masked_m, _ = _make_inputs(
            groups, m, 4096, 7168, seed=926, device=device, masked=[expected_m] * groups
        )
        prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, masked_m, expected_m
        )
        assert prepared.programs == (want,)
        out = prepared.launch()
        torch.cuda.synchronize()
        _assert_valid_rows_close(
            out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
        )


# The swap-AB BLOCK_M 128 points: arch -> (the program they take, the (B, M, N, K) points). On SM100a the build that
# prefetches masked_m into L2 at entry; on SM103a the build without it.
_SWAP_BM_POINTS = {
    "sm_100a": (
        "cake_batch_deepgemm_fp8_c65607eda6318e4ba622",
        (
            (1, 1024, 7168, 2048),
            (4, 128, 7168, 2048),
            (4, 256, 7168, 2048),
            (8, 128, 4096, 7168),
            (8, 128, 7168, 2048),
            (8, 256, 4096, 7168),
            (64, 128, 4096, 7168),
            (64, 256, 4096, 7168),
            (64, 256, 7168, 2048),
            (128, 128, 4096, 7168),
            (128, 128, 7168, 2048),
        ),
    ),
    "sm_103a": (
        "cake_batch_deepgemm_fp8_f239d5b14db9706be968",
        (
            (1, 8192, 4096, 7168),
            (8, 128, 4096, 7168),
            (8, 256, 7168, 2048),
            (64, 256, 4096, 7168),
            (64, 256, 7168, 2048),
        ),
    ),
}

# The expected_m bands where a swap-AB BLOCK_M 128 point takes another program: arch -> {(B, M, N, K): (the lowest
# expected_m of the band, its program)}.  On SM103a the B=1, M=8192, N=4096, K=7168 point takes the BLOCK_M 192 build
# from expected_m 5890 on.
_SWAP_BM_EM_BANDS = {
    "sm_103a": {
        (1, 8192, 4096, 7168): (5890, "cake_batch_deepgemm_fp8_802de3de969a71fa24ba"),
    },
}


def _swap_bm_program(arch, point, expected_m):
    band = _SWAP_BM_EM_BANDS.get(arch, {}).get(point)
    if band is not None and expected_m >= band[0]:
        return band[1]
    return _SWAP_BM_POINTS[arch][0]


def test_swap_bm_points_take_the_architecture_program():
    device = torch.device("cuda")
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = device_arch(index)
    sm_count = device_sm_count(index)
    if arch not in _SWAP_BM_POINTS:
        pytest.skip(f"no swap-AB BLOCK_M 128 point on {arch}")
    program, points = _SWAP_BM_POINTS[arch]
    bands = _SWAP_BM_EM_BANDS.get(arch, {})
    assert arch in PROGRAMS[program]["arches"]
    for _, band_program in bands.values():
        assert arch in PROGRAMS[band_program]["arches"]
    for point in points:
        groups, m, n, k = point
        edges = (bands[point][0] - 1, bands[point][0]) if point in bands else ()
        for expected_m in sorted({1, m // 2, m, *edges}):
            want = _swap_bm_program(arch, point, expected_m)
            plan = select_route(arch, sm_count, groups, m, n, k, expected_m)
            assert plan is not None and [s.program for s in plan.stages] == [want], (
                groups,
                m,
                n,
                k,
                expected_m,
            )
    # The first point at random rows, and every band at its lowest expected_m.
    cases = [(points[0], None)]
    cases += [(point, [edge] * point[0]) for point, (edge, _) in bands.items()]
    for (groups, m, n, k), masked in cases:
        a, b, a_scale, b_scale, masked_m, expected_m = _make_inputs(
            groups, m, n, k, seed=937, device=device, masked=masked
        )
        if masked is not None:
            expected_m = masked[0]
        prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, masked_m, expected_m
        )
        assert prepared.programs == (
            _swap_bm_program(arch, (groups, m, n, k), expected_m),
        )
        out = prepared.launch()
        torch.cuda.synchronize()
        _assert_valid_rows_close(
            out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
        )


def test_route_chain_covers_inventory_with_registered_programs():
    device = torch.device("cuda")
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = device_arch(index)
    sm_count = device_sm_count(index)
    for n, k in sorted(supported_nk(arch)):
        for groups, m in ((1, 128), (4, 1024), (64, 256), (256, 128)):
            for expected_m in (1, m // 2, m):
                plan = select_route(arch, sm_count, groups, m, n, k, expected_m)
                assert plan is not None, (n, k, groups, m, expected_m)
                for stage in plan.stages:
                    assert arch in PROGRAMS[stage.program]["arches"]
                    assert all(int(v) > 0 for v in stage.grid)
    # Geometry outside the inventory is a named gap, never a fallback.
    assert select_route(arch, sm_count, 1, 128, 256, 256, 1) is None


def _band_points(row):
    """The CTA counts at the ends of each of the row's intervals on SM107a (an open
    lower end at 1, an open upper end 32 CTAs past the lower end, at least 320)."""
    points = set()
    for lo, hi in row.where["ctas"]["sm_107a"]:
        points.add(1 if lo is None else lo)
        points.add(max(320, (lo or 1) + 32) if hi is None else hi)
    return sorted(points)


@pytest.mark.parametrize(
    "row",
    [row for row in _TABLE.get("sm_107a", ()) if row.form == "band"],
    ids=lambda row: row.route,
)
def test_sm107a_band_boundaries_match_reference(row):
    """Every end of every SM107a BLOCK_N band selects that band's program, or the
    (B, em4) cell or the one-wave (B, CTA, M) row that takes the problem first, and
    matches the reference."""
    if _arch() != "sm_107a":
        pytest.skip("the SM107a bands run on SM107a")
    device = torch.device("cuda")
    n, k = row.where["nk"]
    for cta in _band_points(row):
        # One group of ``cta`` 128-row M tiles: CTA = ceil(expected_m / 128).
        m = 128 * cta
        rows = m - 5
        a, b, a_scale, b_scale, masked_m, _ = _make_inputs(
            1, m, n, k, seed=935 + cta, device=device, masked=[rows]
        )
        prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, masked_m, rows
        )
        tag = _cell_tag(n, k, 1, rows)
        earlier = _TABLE["sm_107a"][: _TABLE["sm_107a"].index(row)]
        wave = next(
            (
                other
                for other in earlier
                if other.form == "bcta" and _matches(other, "sm_107a", 1, m, n, k, rows)
            ),
            None,
        )
        if tag is None and wave is None:
            assert prepared.route == row.route, (cta, prepared.route)
            assert prepared.programs == (row.stages[0][0],)
        elif tag is None:
            assert prepared.route == wave.route, (cta, prepared.route)
            assert prepared.programs == (wave.stages[0][0],)
        else:
            assert prepared.route == f"seed_masked_bn__n{n}_k{k}_cell_{tag}", (
                cta,
                prepared.route,
            )
        out = prepared.launch()
        torch.cuda.synchronize()
        _assert_valid_rows_close(
            out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
        )
        del a, b, out
        torch.cuda.empty_cache()


def _cell_problem(nk, tag):
    """The (B, expected_m) of the cell of ``tag`` with the fewest rows in all (B * em4),
    with the smallest expected_m that falls in its em4 bin."""
    groups, em4 = min(_SM107A_CELLS[nk][tag], key=lambda c: (c[0] * c[1], c))
    expected_m = next(e for e in range(max(1, em4 - 2), em4 + 4) if _em4(e) == em4)
    return groups, expected_m


@pytest.mark.parametrize(
    "row",
    [row for row in _TABLE.get("sm_107a", ()) if row.form == "cell"],
    ids=lambda row: row.route,
)
def test_sm107a_cells_match_reference(row):
    """One cell of every SM107a (B, em4) cell row selects that row's program and matches the reference."""
    if _arch() != "sm_107a":
        pytest.skip("the SM107a cells run on SM107a")
    device = torch.device("cuda")
    nk, tag = row.where["nk"], row.where["tag"]
    n, k = nk
    groups, expected_m = _cell_problem(nk, tag)
    m = 128 * ((expected_m + 127) // 128)
    # Every group but the last holds expected_m rows; the last holds all M rows.
    masked = [expected_m] * (groups - 1) + [m]
    a, b, a_scale, b_scale, masked_m, _ = _make_inputs(
        groups, m, n, k, seed=950, device=device, masked=masked
    )
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, masked_m, expected_m
    )
    assert prepared.route == row.route, (groups, expected_m, prepared.route)
    assert prepared.programs == (row.stages[0][0],)
    out = prepared.launch()
    torch.cuda.synchronize()
    _assert_valid_rows_close(
        out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )


def test_sm107a_cell_keys_and_nearest_fallback():
    """Each cell key selects its tag; a key without a cell takes the nearest cell on
    N=128 K=512 and N=512 K=128 and no cell on N=4096 K=7168 and N=7168 K=2048."""
    for nk, cells in _SM107A_CELLS.items():
        for tag, keys in cells.items():
            for groups, em4 in keys:
                expected_m = next(
                    e for e in range(max(1, em4 - 2), em4 + 4) if _em4(e) == em4
                )
                assert _cell_tag(*nk, groups, expected_m) == tag, (nk, groups, em4)
    # (64, 1024) is as near (64, 128) as (8, 1024) with the same B * em4: the larger B wins.
    assert (64, 1024) not in {
        key for keys in _SM107A_CELLS[(128, 512)].values() for key in keys
    }
    assert _cell_tag(128, 512, 64, 1024) == _cell_tag(128, 512, 64, 128)
    assert _cell_tag(128, 512, 1000, 1) is not None
    assert _cell_tag(7168, 2048, 1000, 1) is None
    assert _cell_tag(4096, 7168, 1, 128) is None
    assert _cell_tag(4096, 7168, 1000, 1) is None


_BN144 = "cake_batch_deepgemm_fp8_fa7c913fa26439d68c52"
_BN192_L2 = "cake_batch_deepgemm_fp8_e73ad8a03db49514daf6"
_BN240_K128 = "cake_batch_deepgemm_fp8_55bb778b259bbabdf1d9"
_CELL240 = "seed_masked_bn__n7168_k2048_cell_bn240_k128"


def test_sm107a_mmsetup_twins_take_n7168_k2048_keys_of_at_least_64_groups():
    """On SM107a an N=7168 K=2048 key with at least 64 groups (at most 128 for the
    BLOCK_N 240 program) whose route picks a program of ``_SM107A_MMSETUP_TWINS``
    runs that program's masked_m setup twin under the route's name with
    ``_mmsetup`` appended; other group counts and the N=4096 K=7168 keys of the
    same programs keep the parent."""
    for groups, n, k, expected_m, route, program, twinned in (
        (64, 7168, 2048, 2, "seed_masked_bn__n7168_k2048_cell_bn144", _BN144, True),
        (64, 7168, 2048, 100, "seed_masked_bn__n7168_k2048_bn192", _BN192_L2, True),
        (32, 7168, 2048, 1, "seed_masked_bn__n7168_k2048_cell_bn144", _BN144, False),
        (32, 7168, 2048, 100, "seed_masked_bn__n7168_k2048_bn192", _BN192_L2, False),
        (64, 4096, 7168, 2, "seed_masked_bn__n4096_k7168_cell_bn144", _BN144, False),
        (64, 4096, 7168, 100, "seed_masked_bn__n4096_k7168_bn192", _BN192_L2, False),
        (64, 7168, 2048, 54, _CELL240, _BN240_K128, True),
        (128, 7168, 2048, 108, _CELL240, _BN240_K128, True),
        (144, 7168, 2048, 108, _CELL240, _BN240_K128, False),
    ):
        plan = select_route("sm_107a", 216, groups, 128, n, k, expected_m)
        twin = _SM107A_MMSETUP_TWINS[program][0]
        want = (f"{route}_mmsetup", twin) if twinned else (route, program)
        assert (plan.route, plan.stages[0].program) == want, (groups, n, expected_m)
        assert len(plan.stages) == 1 and "sm_107a" in PROGRAMS[want[1]]["arches"]
    for program, (twin, b_min, b_max) in _SM107A_MMSETUP_TWINS.items():
        assert b_min == 64 and twin != program
        assert b_max == (128 if program == _BN240_K128 else None)
        assert PROGRAMS[twin]["arches"] == ["sm_107a"]


@pytest.mark.parametrize(
    "expected_m,route",
    [
        (2, "seed_masked_bn__n7168_k2048_cell_bn144_mmsetup"),
        (100, "seed_masked_bn__n7168_k2048_bn192_mmsetup"),
        (54, "seed_masked_bn__n7168_k2048_cell_bn240_k128_mmsetup"),
    ],
)
def test_sm107a_mmsetup_twins_match_reference(expected_m, route):
    """Each SM107a masked_m setup twin, with an empty group and a full one, matches
    the reference."""
    if _arch() != "sm_107a":
        pytest.skip("the SM107a masked_m setup twins run on SM107a")
    device = torch.device("cuda")
    groups, m, n, k = 64, 128, 7168, 2048
    masked = [expected_m] * (groups - 2) + [0, m]
    a, b, a_scale, b_scale, masked_m, _ = _make_inputs(
        groups, m, n, k, seed=970 + expected_m, device=device, masked=masked
    )
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, masked_m, expected_m
    )
    assert prepared.route == route, prepared.route
    out = prepared.launch()
    torch.cuda.synchronize()
    _assert_valid_rows_close(
        out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )
    del a, b, out
    torch.cuda.empty_cache()


_BN192 = "cake_batch_deepgemm_fp8_20c8e0726c5c51b098d9"
_N512_BN192 = "seed_n512_k128__b9_mbn192"
_N512_BN128 = "seed_n512_k128__b9_bn128"
_N512_PROGRAM = "cake_batch_deepgemm_fp8_c9415f6c743e6bb72fc7"


def test_sm100a_sm103a_mmsetup_twin_takes_n512_k128_keys_of_at_least_64_groups():
    """On SM100a and SM103a an N=512 K=128 key with at least 64 groups whose
    route picks the BLOCK_N 192 program runs its masked_m setup twin under the
    route's name with ``_mmsetup`` appended; fewer groups and the keys of other
    programs keep their route."""
    assert list(_SM100A_SM103A_MMSETUP_TWINS) == [_BN192]
    twin, b_min, b_max = _SM100A_SM103A_MMSETUP_TWINS[_BN192]
    assert (b_min, b_max) == (64, None) and twin != _BN192
    assert PROGRAMS[twin]["arches"] == ["sm_100a", "sm_103a"]
    for arch, sm_count in (("sm_100a", 148), ("sm_103a", 152)):
        for groups, m, expected_m, route, program, twinned in (
            (64, 2048, 1000, _N512_BN192, _BN192, True),
            (256, 4096, 3067, _N512_BN192, _BN192, True),
            (48, 2048, 1000, _N512_BN192, _BN192, False),
            (64, 1024, 1000, _N512_BN128, _N512_PROGRAM, False),
        ):
            plan = select_route(arch, sm_count, groups, m, 512, 128, expected_m)
            want = (f"{route}_mmsetup", twin) if twinned else (route, program)
            assert (plan.route, plan.stages[0].program) == want, (arch, groups, m)
            assert len(plan.stages) == 1 and arch in PROGRAMS[want[1]]["arches"]


def test_sm100a_sm103a_mmsetup_twin_matches_reference():
    """The SM100a / SM103a masked_m setup twin, with an empty group and a full one,
    matches the reference."""
    if _arch() not in ("sm_100a", "sm_103a"):
        pytest.skip("the masked_m setup twin runs on SM100a and SM103a")
    device = torch.device("cuda")
    groups, m, n, k, expected_m = 64, 2048, 512, 128, 1000
    masked = [expected_m] * (groups - 2) + [0, m]
    a, b, a_scale, b_scale, masked_m, _ = _make_inputs(
        groups, m, n, k, seed=1356, device=device, masked=masked
    )
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, masked_m, expected_m
    )
    assert prepared.route == f"{_N512_BN192}_mmsetup", prepared.route
    out = prepared.launch()
    torch.cuda.synchronize()
    _assert_valid_rows_close(
        out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )
    del a, b, out
    torch.cuda.empty_cache()


_BN32_S8 = "cake_batch_deepgemm_fp8_929a5561abc99caa7f4f"
_EARLY_BN32_S8 = "cake_batch_deepgemm_fp8_927fea695cd92359d999"


def test_sm100a_sm103a_n512_k128_mbn32_routes_take_the_masked_m_once_build():
    """On SM100a and SM103a the N=512 K=128 keys with M > 1024, at most eight
    groups and at most eight CTAs (seven for one group) run the masked_m-once form
    of the BLOCK_N 32 eight-stage build; the neighbouring keys keep their programs
    and the parent stays shipped for the N=4096 K=7168 one-CTA band."""
    assert PROGRAMS[_EARLY_BN32_S8]["arches"] == ["sm_100a", "sm_103a"]
    assert PROGRAMS[_BN32_S8]["arches"] == ["sm_100a", "sm_103a"]
    b1, b2_8 = "seed_n512_k128__b1_mbn32", "seed_n512_k128__b2_8_mbn32"
    for arch, sm_count in (("sm_100a", 148), ("sm_103a", 152)):
        for groups, m, expected_m, route, program in (
            (1, 1152, 1, b1, _EARLY_BN32_S8),
            (1, 16384, 896, b1, _EARLY_BN32_S8),
            (2, 2048, 512, b2_8, _EARLY_BN32_S8),
            (8, 8192, 128, b2_8, _EARLY_BN32_S8),
            (1, 1024, 896, "seed_n512_k128__b1_bn32", None),
            (1, 2048, 897, "seed_n512_k128__b1_bn192", None),
            (8, 2048, 129, "seed_n512_k128__b2_8_bn128", None),
        ):
            plan = select_route(arch, sm_count, groups, m, 512, 128, expected_m)
            assert plan.route == route, (arch, groups, m, expected_m, plan.route)
            assert len(plan.stages) == 1
            got = plan.stages[0].program
            if program is None:
                assert got not in (_EARLY_BN32_S8, _BN32_S8), (arch, route, got)
            else:
                assert got == program, (arch, route, got)


@pytest.mark.parametrize("groups,m,expected_m", [(1, 2048, 700), (8, 4096, 100)])
def test_sm100a_sm103a_masked_m_once_bn32_matches_reference(groups, m, expected_m):
    """The SM100a / SM103a masked_m-once BLOCK_N 32 eight-stage build, with an
    empty group and a full one where there are two or more, matches the
    reference."""
    if _arch() not in ("sm_100a", "sm_103a"):
        pytest.skip("the masked_m-once BLOCK_N 32 build runs on SM100a and SM103a")
    device = torch.device("cuda")
    masked = [expected_m] * groups
    if groups > 1:
        masked[-2:] = [0, m]
    a, b, a_scale, b_scale, masked_m, _ = _make_inputs(
        groups, m, 512, 128, seed=1356, device=device, masked=masked
    )
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, masked_m, expected_m
    )
    assert prepared.route.endswith("_mbn32"), prepared.route
    out = prepared.launch()
    torch.cuda.synchronize()
    _assert_valid_rows_close(
        out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )
    del a, b, out
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# The locality-domain twins of the SM107a serving programs (dual-die GPUs).

_SERVING_ROUTE = "seed_serving_m32_packed"
_SERVING_M = 256
# One expected_m per expected_m class.
_CLASS_EXPECTED_M = (1, 4, 8, 16)
_RACE_ENV = "FLASHINFER_CAKE_DOMAIN_RACE_LAUNCHES"
_RACE_PLACEMENTS = ("stream", "occupier-die0", "occupier-die1", "one-die0", "one-die1")

# The cells of the twins' source rule, keyed by serving key: a (die class,
# expected_m class) cell is listed where the twin ran below 0.98 x the fastest
# whole-GPU build of the same N, K and groups on every timed 212-SM GPU of that
# die class.  DOMAIN_SERVING_CELLS must route exactly these.
_SOURCE_ROUTES = {
    "sm_107a:n4096_k7168_g32_s5e2_evict_normal_l8": {
        ("even", "em8-15"),
        ("uneven", "em8-15"),
    },
    "sm_107a:n4096_k7168_g64_s5e2_evict_normal_l8": {
        ("even", "em8-15"),
        ("uneven", "em8-15"),
    },
    "sm_107a:n7168_k2048_g32_s5e2_evict_normal_l8": {
        ("even", "em8-15"),
        ("uneven", "em8-15"),
    },
    "sm_107a:n7168_k2048_g64_s5e2_evict_normal_l8": {
        ("even", "em8-15"),
        ("uneven", "em8-15"),
    },
    "sm_107a:n4096_k7168_g64_s6e1_evict_normal_l2": {
        ("even", "em<=1"),
        ("uneven", "em<=1"),
    },
}
_SOURCE_SM_COUNT = 212
# Probe vectors of the source rule: die splits at both sides of the even bound
# (a gap of 4 SMs is even), and expected_m at both sides of every class bound.
_PROBE_DIES = (
    (106, 106),
    (108, 104),
    (104, 108),
    (105, 109),
    (104, 109),
    (102, 110),
    (112, 100),
)
_PROBE_EXPECTED_M = (0, 1, 2, 7, 8, 15, 16, 64)


def _source_twin(key, dies, expected_m):
    """The twin the source rule launches for serving key ``key`` (qualified)."""
    if expected_m <= 1:
        em = "em<=1"
    elif expected_m < 8:
        em = "em2-7"
    elif expected_m < 16:
        em = "em8-15"
    else:
        em = "em16+"
    die = "even" if abs(dies[0] - dies[1]) <= 4 else "uneven"
    if (die, em) in _SOURCE_ROUTES.get(key, ()):
        return DOMAIN_SERVING_PROGRAMS[key]
    return None


def _cell_id(cell):
    (n, k), groups, expected_m = cell
    return f"n{n}_k{k}_g{groups}_em{expected_m}"


def _serving_key(arch, sm_count, nk, groups, expected_m):
    n, k = nk
    stages, epi, hint, _, l2 = serving_config(_SERVING_M, k, expected_m, sm_count, arch)
    return f"{arch}:{serving_program_key(n, k, groups, stages, epi, hint, l2)}"


def _listed(cell):
    """True when some SM count and die class route ``cell`` to a twin."""
    nk, groups, expected_m = cell
    for sm_count, cells in DOMAIN_SERVING_CELLS.items():
        key = _serving_key("sm_107a", sm_count, nk, groups, expected_m)
        for (_, em_class), keys in cells.items():
            if em_class == expected_m_class(expected_m) and key in keys:
                return True
    return False


# ((N, K), groups, expected_m) of every serving shape, one expected_m per class;
# the twin tests take the cells some die class routes to a twin.
_SHAPE_CELLS = [
    (nk, groups, expected_m)
    for nk in sorted(PACKED_NK)
    for groups in sorted(PACKED_GROUPS)
    for expected_m in _CLASS_EXPECTED_M
]
_DOMAIN_CELLS = [cell for cell in _SHAPE_CELLS if _listed(cell)]


def _whole_program(key):
    """The whole serving program of an architecture-qualified key (an unqualified
    key serves every architecture)."""
    return SERVING_PROGRAMS.get(key, SERVING_PROGRAMS.get(key.split(":", 1)[1]))


def _require_domain():
    """The device and its SM table: a dual-die SM107a GPU, a CUDA 13.5 toolkit and
    driver, and an SM count with listed cells; past the skips, a build or probe
    failure is a test failure."""
    device = torch.cuda.current_device()
    if _arch() != "sm_107a":
        pytest.skip("the locality-domain twins run on SM107a")
    from flashinfer.jit.cpp_ext import get_cuda_version
    from flashinfer.locality_domain import locality_domain_unsupported_reason

    version = get_cuda_version()
    if (version.major, version.minor) < cake_locality.LOCALITY_CUDA:
        pytest.skip(f"the twins need a CUDA 13.5 toolkit, found {version}")
    reason = locality_domain_unsupported_reason(device)
    if reason is not None:
        pytest.skip(f"CUDA locality domains are unavailable: {reason}")
    driver = int(cake_locality._module().driver_version())
    if (driver // 1000, driver % 1000 // 10) < cake_locality.LOCALITY_CUDA:
        pytest.skip(f"the twins need a CUDA 13.5 driver, found {driver}")
    sm_count = device_sm_count(device)
    if sm_count not in DOMAIN_SERVING_CELLS:
        pytest.skip(f"no locality-domain cell is listed for {sm_count} SMs")
    table = cake_locality.sm_table(device)
    assert table is not None, "the SM table probe failed on a dual-die SM107a GPU"
    assert sum(table.counts) == sm_count
    return device, table


def _require_routed(cell, device, table):
    """The twin program of ``cell`` on this device; skips a cell its die class
    does not route."""
    nk, groups, expected_m = cell
    sm_count = device_sm_count(device)
    key = _serving_key("sm_107a", sm_count, nk, groups, expected_m)
    twin = domain_serving_program(
        "sm_107a", sm_count, key.split(":", 1)[1], expected_m, table.counts
    )
    if twin is None:
        pytest.skip(
            f"{_cell_id(cell)} is not routed to a twin for the dies {table.counts}"
        )
    return twin


def _home(b, table, g_cut=None):
    """``b`` copied into a split allocation cut after group ``g_cut`` (default:
    the dies' SM-proportional share), whatever :func:`localize_weights` decides."""
    groups = int(b.shape[0])
    if g_cut is None:
        g_cut = cake_locality.split_units(groups, table.counts)[0]
    homed = cake_locality.split_empty(
        tuple(b.shape), b.dtype, g_cut * b[0].numel() * b.element_size(), b.device
    )
    homed.copy_(b)
    return homed


@functools.lru_cache(maxsize=1)
def _serving_problem(nk, groups, expected_m):
    """Packed-scale serving inputs of one cell (masks around ``expected_m``, an
    empty, a full and multi-block groups) and their float32 reference."""
    n, k = nk
    m = _SERVING_M
    if expected_m <= 1:
        masked = [g % 2 for g in range(groups)]
    else:
        masked = [(g * 7) % (2 * expected_m + 1) for g in range(groups)]
    masked[3], masked[5], masked[9], masked[20] = m, 0, 40, 33
    a, b, a_scale, b_scale, masked_m, _ = _make_inputs(
        groups, m, n, k, seed=941, device=torch.device("cuda"), masked=masked
    )
    a_packed = _pack_ue8m0(a_scale)
    b_packed = _pack_ue8m0(b_scale.repeat_interleave(128, dim=1))
    reference = _reference(a, b, a_scale, b_scale, masked_m)
    valid = torch.arange(m, device=a.device)[None, :] < masked_m[:, None]
    return a, b, a_scale, b_scale, a_packed, b_packed, masked_m, reference, valid


def _valid_bits(out, valid):
    return out.view(torch.int16)[valid].clone()


@_host_only
def test_domain_serving_cells_follow_the_source_rule():
    """DOMAIN_SERVING_CELLS route exactly the source rule's cells over its probe
    vectors, for every sm_107a serving key of the packed-scale shapes."""
    assert set(DOMAIN_SERVING_CELLS) == {_SOURCE_SM_COUNT}
    assert set(_SOURCE_ROUTES) <= set(DOMAIN_SERVING_PROGRAMS)
    assert all(_whole_program(key) for key in DOMAIN_SERVING_PROGRAMS)
    for dies, die in (((105, 109), "even"), ((104, 109), "uneven")):
        assert cake_locality.die_class(dies) == die
    for expected_m, em in ((0, "em<=1"), (1, "em<=1"), (2, "em2-7"), (7, "em2-7")):
        assert expected_m_class(expected_m) == em
    for expected_m, em in ((8, "em8-15"), (15, "em8-15"), (16, "em16+"), (64, "em16+")):
        assert expected_m_class(expected_m) == em
    keys = set(DOMAIN_SERVING_PROGRAMS) | {
        _serving_key("sm_107a", _SOURCE_SM_COUNT, nk, groups, expected_m)
        for nk in PACKED_NK
        for groups in PACKED_GROUPS
        for expected_m in range(0, 70)
    }
    reached = set()
    for dies in _PROBE_DIES:
        for expected_m in _PROBE_EXPECTED_M:
            for key in sorted(keys):
                got = domain_serving_program(
                    "sm_107a", _SOURCE_SM_COUNT, key.split(":", 1)[1], expected_m, dies
                )
                assert got == _source_twin(key, dies, expected_m), (
                    key,
                    dies,
                    expected_m,
                )
                if got is not None:
                    cell = (cake_locality.die_class(dies), expected_m_class(expected_m))
                    reached.add((key, *cell))
    listed = {
        (key, *cell)
        for cell, cell_keys in DOMAIN_SERVING_CELLS[_SOURCE_SM_COUNT].items()
        for key in cell_keys
    }
    assert listed == reached


@_host_only
def test_every_sm107a_serving_key_has_a_twin():
    """Every serving program an SM107a packed-scale problem selects, at any
    expected_m, has a locality-domain twin of the same key."""
    keys = {
        _serving_key("sm_107a", _SOURCE_SM_COUNT, nk, groups, expected_m)
        for nk in PACKED_NK
        for groups in PACKED_GROUPS
        for expected_m in range(0, 70)
    }
    assert sorted(keys - set(DOMAIN_SERVING_PROGRAMS)) == []
    assert sorted(set(DOMAIN_SERVING_PROGRAMS) - keys) == []
    for key, twin in DOMAIN_SERVING_PROGRAMS.items():
        assert PROGRAMS[twin]["arches"] == ["sm_107a"], key
        assert PROGRAMS[twin]["arg_plan"] == "plan8", key


@_host_only
def test_domain_serving_plan_of_listed_cells():
    """A listed cell plans its twin (one CTA per SM, a counter pair); the same
    problem without an SM split, at an unmeasured SM count or off SM107a plans
    the whole program."""
    for counts in ((104, 108), (108, 104), (102, 110), (112, 100)):
        sm_count = sum(counts)
        for cell in _SHAPE_CELLS:
            nk, groups, expected_m = cell
            key = _serving_key("sm_107a", sm_count, nk, groups, expected_m)
            plan = serving_plan(
                "sm_107a", sm_count, groups, _SERVING_M, *nk, expected_m, counts=counts
            )
            twin = _source_twin(key, counts, expected_m)
            whole = serving_plan(
                "sm_107a", sm_count, groups, _SERVING_M, *nk, expected_m
            )
            assert whole.route == _SERVING_ROUTE and whole.workspaces == {}
            assert whole.stages[0].program == _whole_program(key)
            if twin is None:
                assert plan.route == _SERVING_ROUTE, cell
                continue
            assert plan.route == DOMAIN_SERVING_ROUTE
            assert plan.parent == _SERVING_ROUTE
            assert [stage.program for stage in plan.stages] == [twin]
            assert plan.stages[0].grid == (sm_count, 1, 1)
            assert plan.workspaces == {
                "counters0": (PAIR_COUNTER_WORDS,),
                "counters1": (PAIR_COUNTER_WORDS,),
            }
    for arch, sm_count, counts in (
        ("sm_107a", 216, (108, 108)),
        ("sm_100a", 148, (74, 74)),
    ):
        for cell in _DOMAIN_CELLS:
            nk, groups, expected_m = cell
            plan = serving_plan(
                arch, sm_count, groups, _SERVING_M, *nk, expected_m, counts=counts
            )
            assert plan.route == _SERVING_ROUTE


@_host_only
def test_domain_serving_routed_needs_every_expected_m_routed(monkeypatch):
    """Weights are localizable only where every expected_m runs a twin."""
    import flashinfer.gemm.cake_batch_deepgemm_fp8 as module

    for dies in _PROBE_DIES:
        for nk in sorted(PACKED_NK):
            for groups in sorted(PACKED_GROUPS):
                want = all(
                    _source_twin(
                        _serving_key("sm_107a", _SOURCE_SM_COUNT, nk, groups, em),
                        dies,
                        em,
                    )
                    for em in range(0, 70)
                )
                got = domain_serving_routed(
                    "sm_107a", _SOURCE_SM_COUNT, dies, *nk, groups
                )
                assert got == want, (dies, nk, groups)
    for arch, sm_count, n, k, groups in (
        ("sm_107a", _SOURCE_SM_COUNT, 512, 128, 64),
        ("sm_107a", _SOURCE_SM_COUNT, 4096, 7168, 16),
        ("sm_107a", 216, 4096, 7168, 64),
        ("sm_103a", _SOURCE_SM_COUNT, 4096, 7168, 64),
    ):
        assert not domain_serving_routed(arch, sm_count, (106, 106), n, k, groups)
    # Every cell keeps its shape localizable only while it is listed.
    for (die, em_class), keys in DOMAIN_SERVING_CELLS[_SOURCE_SM_COUNT].items():
        dies = (106, 106) if die == "even" else (102, 110)
        for key in keys:
            n, k, groups = (
                int(part[1:]) for part in key.split(":", 1)[1].split("_")[:3]
            )
            cells = {
                count: {
                    cell: tuple(
                        other
                        for other in cell_keys
                        if (cell, other) != ((die, em_class), key)
                    )
                    for cell, cell_keys in table.items()
                }
                for count, table in DOMAIN_SERVING_CELLS.items()
            }
            with monkeypatch.context() as patch:
                patch.setattr(module, "DOMAIN_SERVING_CELLS", cells)
                assert not domain_serving_routed(
                    "sm_107a", _SOURCE_SM_COUNT, dies, n, k, groups
                ), (key, die, em_class)


@_host_only
def test_unlisted_sm_counts_of_sm100a_sm103a_still_raise(monkeypatch):
    """The packed-scale serving route runs on every SM count of SM107a only:
    SM100a and SM103a keep refusing an SM count without a route chain."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import flashinfer.gemm.cake_batch_deepgemm_fp8 as module

    device = torch.device("cuda")
    a = torch.empty((1, 1, 1), dtype=torch.float8_e4m3fn, device=device)
    scale = torch.empty((1, 1, 1), dtype=torch.int32, device=device)
    masked_m = torch.zeros((1,), dtype=torch.int32, device=device)
    for arch, sm_count in (("sm_103a", 148), ("sm_103a", 160), ("sm_100a", 132)):
        patch_arch = functools.partial(lambda _index, value: value, value=arch)
        patch_sms = functools.partial(lambda _index, value: value, value=sm_count)
        with monkeypatch.context() as patch:
            patch.setattr(module, "device_arch", patch_arch)
            patch.setattr(module, "device_sm_count", patch_sms)
            with pytest.raises(ValueError, match=f"this device has {sm_count}$"):
                prepare_batch_deepgemm_fp8_nt_groupwise(a, a, scale, scale, masked_m, 0)
    with monkeypatch.context() as patch:
        patch.setattr(module, "device_arch", lambda _index: "sm_107a")
        patch.setattr(module, "device_sm_count", lambda _index: 212)
        with pytest.raises(ValueError, match="aligned M, N and K"):
            prepare_batch_deepgemm_fp8_nt_groupwise(a, a, scale, scale, masked_m, 0)


@_host_only
def test_domain_counters_must_not_overlap():
    """Caller-owned counters of the two dies must not share a byte."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from flashinfer.gemm.cake_batch_deepgemm_fp8 import _check_workspace

    (nk, groups, expected_m) = _DOMAIN_CELLS[0]
    counts = (106, 106)
    plan = serving_plan(
        "sm_107a", sum(counts), groups, _SERVING_M, *nk, expected_m, counts=counts
    )
    assert plan.route == DOMAIN_SERVING_ROUTE
    device = torch.device("cuda", torch.cuda.current_device())
    words = torch.zeros(4, dtype=torch.int32, device=device)
    for first, second in ((words[0:2], words[0:2]), (words[0:2], words[1:3])):
        with pytest.raises(ValueError, match="overlap"):
            _check_workspace(plan, {"counters0": first, "counters1": second}, device)
        with pytest.raises(ValueError, match="overlap"):
            _check_workspace(plan, {"counters0": second, "counters1": first}, device)
    _check_workspace(plan, {"counters0": words[2:4], "counters1": words[0:2]}, device)


@_host_only
def test_counters_fall_back_to_unhomed_zeros(monkeypatch):
    """Without FlashInfer's locality-domain pools (another allocator backend, no
    MemPool, no locality domains) the counters are ordinary zeroed tensors."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from flashinfer import locality_domain

    def refuse(_index):
        raise locality_domain.LocalityDomainError("backend:cudaMallocAsync")

    monkeypatch.setattr(locality_domain, "_domain_pools", refuse)
    first, second = cake_locality.counters(PAIR_COUNTER_WORDS)
    for tensor in (first, second):
        assert tensor.dtype == torch.int32 and tuple(tensor.shape) == (
            PAIR_COUNTER_WORDS,
        )
        assert tensor.device.type == "cuda"
        assert int(tensor.abs().sum()) == 0
    assert first.data_ptr() != second.data_ptr()


@_packed_serving
@pytest.mark.parametrize("cell", _DOMAIN_CELLS, ids=_cell_id)
def test_domain_twin_matches_reference(cell):
    """(a) The twin over the shape list: the reference within the family tolerance,
    the public API the same bits; the whole program is checked the same way.
    ``localize_weights`` homes the weights exactly when every expected_m of
    their shape is routed to a twin."""
    device, table = _require_domain()
    twin_program = _require_routed(cell, device, table)
    nk, groups, expected_m = cell
    a, b, _, _, a_packed, b_packed, masked_m, reference, valid = _serving_problem(*cell)
    sm_count = device_sm_count(device)
    key = _serving_key("sm_107a", sm_count, nk, groups, expected_m)
    whole = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_packed, b_packed, masked_m, expected_m
    )
    assert whole.route == _SERVING_ROUTE
    assert whole.programs == (_whole_program(key),)
    whole_out = whole.launch()
    localized = localize_weights(b)
    if domain_serving_routed("sm_107a", sm_count, table.counts, *nk, groups):
        assert localized is not b
        assert cake_locality.homing(localized) is not None
        homed = localized
    else:
        assert localized is b
        homed = _home(b, table)
    assert torch.equal(homed.view(torch.uint8), b.view(torch.uint8))
    twin = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, homed, a_packed, b_packed, masked_m, expected_m
    )
    assert twin.route == DOMAIN_SERVING_ROUTE
    assert twin.programs == (twin_program,) == (DOMAIN_SERVING_PROGRAMS[key],)
    assert twin.grids == ((sm_count // 2 * 2, 1, 1),)
    assert twin.g_cut == cake_locality.split_units(groups, table.counts)[0]
    assert not twin.capture_only
    twin_out = twin.launch()
    torch.cuda.synchronize()
    _assert_valid_rows_close(twin_out, reference, masked_m)
    _assert_valid_rows_close(whole_out, reference, masked_m)
    public = batch_deepgemm_fp8_nt_groupwise(
        a, homed, a_packed, b_packed, masked_m, expected_m, backend="cake"
    )
    torch.cuda.synchronize()
    assert torch.equal(_valid_bits(public, valid), _valid_bits(twin_out, valid))
    same = torch.equal(_valid_bits(whole_out, valid), _valid_bits(twin_out, valid))
    print(
        f"TWIN {_cell_id(cell)} counts={table.counts} g_cut={twin.g_cut} "
        f"bitwise_equal_to_shipped_whole_program={same}"
    )


@_packed_serving
def test_localize_weights_refuses_what_could_run_slower(monkeypatch):
    """``localize_weights`` returns ``b`` itself for weights of a non-serving
    shape or dtype, for a shape with an expected_m whose cell is not listed, and
    for a cut off the allocation granularity; it raises for a ``g_cut`` outside
    ``[0, B]``."""
    import flashinfer.gemm.cake_batch_deepgemm_fp8 as module

    device, table = _require_domain()
    sm_count = device_sm_count(device)
    for shape, dtype in (
        ((64, 512, 128), torch.float8_e4m3fn),
        ((16, 7168, 2048), torch.float8_e4m3fn),
        ((32, 7168, 2048), torch.uint8),
    ):
        weights = torch.zeros(shape, dtype=dtype, device=f"cuda:{device}")
        assert localize_weights(weights) is weights
        del weights
    routed = [
        (nk, groups)
        for nk in sorted(PACKED_NK)
        for groups in sorted(PACKED_GROUPS)
        if domain_serving_routed("sm_107a", sm_count, table.counts, *nk, groups)
    ]
    shapes = {(nk, groups) for nk in PACKED_NK for groups in PACKED_GROUPS}
    for nk, groups in sorted(shapes - set(routed)):
        weights = torch.zeros(
            (groups, *nk), dtype=torch.float8_e4m3fn, device=f"cuda:{device}"
        )
        assert localize_weights(weights) is weights
        del weights
    if not routed:
        pytest.skip(f"no serving shape is localizable on the dies {table.counts}")
    nk, groups = min(routed, key=lambda item: item[0][0] * item[0][1] * item[1])
    weights = torch.zeros(
        (groups, *nk), dtype=torch.float8_e4m3fn, device=f"cuda:{device}"
    )
    homed = localize_weights(weights)
    assert homed is not weights and cake_locality.homing(homed) is not None
    del homed
    with pytest.raises(ValueError, match="g_cut"):
        localize_weights(weights, groups + 1)
    with monkeypatch.context() as patch:
        patch.setattr(module, "DOMAIN_SERVING_CELLS", {})
        assert localize_weights(weights) is weights
    with monkeypatch.context() as patch:
        # A granularity of 3 MiB: a cut of an odd number of 14 or 28 MiB groups
        # is off it.
        patch.setattr(cake_locality, "granularity", lambda device=None: 3 << 20)
        assert localize_weights(weights, 1) is weights
        assert localize_weights(weights, 0) is not weights


@_packed_serving
@pytest.mark.parametrize("cell", _DOMAIN_CELLS, ids=_cell_id)
def test_domain_twin_bits_do_not_depend_on_the_cut(cell):
    """(b) Empty-die cuts both ways, an uneven cut and the SM-proportional cuts of
    both die orders give the same bits."""
    device, table = _require_domain()
    _require_routed(cell, device, table)
    _, groups, expected_m = cell
    a, b, _, _, a_packed, b_packed, masked_m, reference, valid = _serving_problem(*cell)
    cuts = sorted(
        {
            0,
            groups,
            groups // 4,
            cake_locality.split_units(groups, table.counts)[0],
            cake_locality.split_units(groups, table.counts[::-1])[0],
        }
    )
    first = None
    for g_cut in cuts:
        homed = _home(b, table, g_cut)
        prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
            a, homed, a_packed, b_packed, masked_m, expected_m
        )
        assert prepared.route == DOMAIN_SERVING_ROUTE
        assert prepared.g_cut == g_cut
        out = prepared.launch()
        torch.cuda.synchronize()
        bits = _valid_bits(out, valid)
        if first is None:
            _assert_valid_rows_close(out, reference, masked_m)
            first = bits
        else:
            assert torch.equal(bits, first), f"g_cut={g_cut} changed the result"
        del prepared, homed, out


@_packed_serving
@pytest.mark.parametrize("cell", _DOMAIN_CELLS[:2], ids=_cell_id)
def test_domain_twin_counters_zero_and_reused(cell, monkeypatch):
    """(c) The claim counters are zero after every launch and reused by the next;
    a launch allocates nothing; a caller-owned counter pair is bound as given;
    without the locality-domain pools the counters are unhomed and the result
    is the same bits."""
    device, table = _require_domain()
    _require_routed(cell, device, table)
    _, _, expected_m = cell
    a, b, _, _, a_packed, b_packed, masked_m, reference, valid = _serving_problem(*cell)
    homed = _home(b, table)
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, homed, a_packed, b_packed, masked_m, expected_m
    )
    assert prepared.route == DOMAIN_SERVING_ROUTE
    counters = (prepared.workspace["counters0"], prepared.workspace["counters1"])
    for tensor in counters:
        assert tensor.dtype == torch.int32 and tuple(tensor.shape) == (
            PAIR_COUNTER_WORDS,
        )
    prepared.launch()
    torch.cuda.synchronize()
    first = _valid_bits(prepared.out, valid)
    _assert_valid_rows_close(prepared.out, reference, masked_m)
    for _ in range(3):
        assert all(int(t.abs().sum()) == 0 for t in counters)
        prepared.out.fill_(float("nan"))
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()
        prepared.launch()
        torch.cuda.synchronize()
        after = torch.cuda.memory_stats()
        assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
        assert after["allocation.all.freed"] == before["allocation.all.freed"]
        assert torch.equal(_valid_bits(prepared.out, valid), first)
    assert all(int(t.abs().sum()) == 0 for t in counters)
    owned = dict(
        zip(
            ("counters0", "counters1"),
            cake_locality.counters(PAIR_COUNTER_WORDS),
            strict=True,
        )
    )
    given = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, homed, a_packed, b_packed, masked_m, expected_m, workspace=owned
    )
    assert given.workspace["counters0"] is owned["counters0"]
    given.launch()
    torch.cuda.synchronize()
    assert torch.equal(_valid_bits(given.out, valid), first)
    assert all(int(t.abs().sum()) == 0 for t in owned.values())
    from flashinfer import locality_domain

    def refuse(_index):
        raise locality_domain.LocalityDomainError("backend:cudaMallocAsync")

    with monkeypatch.context() as patch:
        patch.setattr(locality_domain, "_domain_pools", refuse)
        unhomed = prepare_batch_deepgemm_fp8_nt_groupwise(
            a, homed, a_packed, b_packed, masked_m, expected_m
        )
        assert unhomed.route == DOMAIN_SERVING_ROUTE
        unhomed.launch()
        public = batch_deepgemm_fp8_nt_groupwise(
            a, homed, a_packed, b_packed, masked_m, expected_m, backend="cake"
        )
        torch.cuda.synchronize()
    assert torch.equal(_valid_bits(unhomed.out, valid), first)
    assert torch.equal(_valid_bits(public, valid), first)
    assert all(int(t.abs().sum()) == 0 for t in unhomed.workspace.values())


@_packed_serving
@pytest.mark.parametrize("cell", _DOMAIN_CELLS[:2], ids=_cell_id)
def test_domain_weights_and_counters_homed_per_die(cell):
    """(d) ``pointer_domain`` of the first and last bytes of each die's groups of
    ``b``, and of each die's counters."""
    device, table = _require_domain()
    _require_routed(cell, device, table)
    nk, groups, expected_m = cell
    a, b, _, _, a_packed, b_packed, masked_m, _, _ = _serving_problem(*cell)
    group_bytes = nk[0] * nk[1]
    for g_cut in sorted(
        {cake_locality.split_units(groups, table.counts)[0], 0, groups, groups // 4}
    ):
        homed = _home(b, table, g_cut)
        record = cake_locality.homing(homed)
        assert record is not None and record.cut == g_cut * group_bytes
        base = homed.data_ptr()
        if g_cut > 0:
            assert cake_locality.pointer_domain(base) == 0
            assert cake_locality.pointer_domain(base + g_cut * group_bytes - 1) == 0
        if g_cut < groups:
            assert cake_locality.pointer_domain(base + g_cut * group_bytes) == 1
            assert cake_locality.pointer_domain(base + groups * group_bytes - 1) == 1
        assert torch.equal(homed.view(torch.uint8), b.view(torch.uint8))
        del homed
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, _home(b, table), a_packed, b_packed, masked_m, expected_m
    )
    assert prepared.route == DOMAIN_SERVING_ROUTE
    for die in range(2):
        counter = prepared.workspace[f"counters{die}"]
        assert cake_locality.pointer_domain(counter.data_ptr()) == die


@_packed_serving
def test_domain_fallbacks_take_the_whole_route(monkeypatch):
    """(e) An unhomed ``b``, a copy of homed weights, no listed cell and no SM
    table take the whole program; ``localize_weights`` then returns ``b``."""
    import flashinfer.gemm.cake_batch_deepgemm_fp8 as module

    device, table = _require_domain()
    cell = next(
        (
            cell
            for cell in _DOMAIN_CELLS
            if domain_serving_program(
                "sm_107a",
                device_sm_count(device),
                _serving_key("sm_107a", device_sm_count(device), *cell).split(":")[1],
                cell[2],
                table.counts,
            )
        ),
        None,
    )
    if cell is None:
        pytest.skip(f"no cell is routed to a twin for the dies {table.counts}")
    nk, groups, expected_m = cell
    a, b, a_scale, b_scale, a_packed, b_packed, masked_m, reference, _ = (
        _serving_problem(*cell)
    )
    homed = _home(b, table)
    unhomed = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_packed, b_packed, masked_m, expected_m
    )
    assert unhomed.route == _SERVING_ROUTE and unhomed.g_cut is None
    # A copy of homed weights is not homed.
    copied = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, homed.clone(), a_packed, b_packed, masked_m, expected_m
    )
    assert copied.route == _SERVING_ROUTE
    for cells in ({}, {device_sm_count(device): {}}):
        with monkeypatch.context() as patch:
            patch.setattr(module, "DOMAIN_SERVING_CELLS", cells)
            assert localize_weights(b) is b
            prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
                a, homed, a_packed, b_packed, masked_m, expected_m
            )
            assert prepared.route == _SERVING_ROUTE
            out = prepared.launch()
            torch.cuda.synchronize()
            _assert_valid_rows_close(out, reference, masked_m)
    with monkeypatch.context() as patch:
        patch.setattr(cake_locality, "sm_table", lambda device=None: None)
        assert localize_weights(b) is b
        no_table = prepare_batch_deepgemm_fp8_nt_groupwise(
            a, homed, a_packed, b_packed, masked_m, expected_m
        )
        assert no_table.route == _SERVING_ROUTE
        out = no_table.launch()
        torch.cuda.synchronize()
        _assert_valid_rows_close(out, reference, masked_m)
    # Float32 scales on a device without a route chain still raise.
    if device_sm_count(torch.cuda.current_device()) not in (216,):
        with pytest.raises(ValueError, match="SMs"):
            prepare_batch_deepgemm_fp8_nt_groupwise(
                a, homed, a_scale, b_scale, masked_m, expected_m
            )


@_packed_serving
@pytest.mark.parametrize("cell", _DOMAIN_CELLS[:2], ids=_cell_id)
def test_domain_twin_cuda_graph_capture_and_replay(cell):
    """(f) The first launch of a prepared twin is captured and replayed; replays
    follow new activations and leave the counters zero; the public API's call is
    capturable too; a twin prepared under capture launches only under capture."""
    device_index, table = _require_domain()
    _require_routed(cell, device_index, table)
    _, _, expected_m = cell
    a0, b, a_scale, b_scale, a_packed, b_packed, masked_m, reference, valid = (
        _serving_problem(*cell)
    )
    device = a0.device
    a = a0.clone()
    homed = _home(b, table)
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, homed, a_packed, b_packed, masked_m, expected_m
    )
    assert prepared.route == DOMAIN_SERVING_ROUTE
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        prepared.launch()
    prepared.out.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    _assert_valid_rows_close(prepared.out, reference, masked_m)
    first = _valid_bits(prepared.out, valid)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_valid_bits(prepared.out, valid), first)
    counters = (prepared.workspace["counters0"], prepared.workspace["counters1"])
    assert all(int(t.abs().sum()) == 0 for t in counters)
    a.copy_(torch.randn(a.shape, device=device).to(torch.float8_e4m3fn))
    graph.replay()
    torch.cuda.synchronize()
    _assert_valid_rows_close(
        prepared.out, _reference(a, b, a_scale, b_scale, masked_m), masked_m
    )
    assert all(int(t.abs().sum()) == 0 for t in counters)
    out = torch.empty_like(prepared.out)
    public_graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(public_graph, stream=stream):
        batch_deepgemm_fp8_nt_groupwise(
            a0, homed, a_packed, b_packed, masked_m, expected_m, out=out, backend="cake"
        )
    out.fill_(float("nan"))
    torch.cuda.synchronize()
    public_graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_valid_bits(out, valid), first)
    captured_graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(captured_graph, stream=stream):
        captured = prepare_batch_deepgemm_fp8_nt_groupwise(
            a0, homed, a_packed, b_packed, masked_m, expected_m
        )
        captured.launch()
    assert captured.route == DOMAIN_SERVING_ROUTE and captured.capture_only
    with pytest.raises(RuntimeError, match="capture"):
        captured.launch()
    captured.out.fill_(float("nan"))
    torch.cuda.synchronize()
    captured_graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_valid_bits(captured.out, valid), first)


# The race's occupier: one 1024-thread block per SM (its whole opt-in shared
# memory, so no two share an SM), each spinning until released (or its cycle
# budget runs out); every block records its SM and counts itself in.
_OCCUPIER = "cake_domain_race_occupier"
_OCCUPIER_SOURCE = f"""\
extern "C" __global__ void __launch_bounds__(1024, 1) {_OCCUPIER}(const int* release, int* state, int* smids,
                                                                 long long budget) {{
    extern __shared__ int pad[];
    if (threadIdx.x == 0) {{
        unsigned sm;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        smids[blockIdx.x] = (int)sm;
        pad[0] = (int)sm;
        atomicAdd(state, 1);
        long long start = clock64();
        while (*((volatile const int*)release) == 0) {{
            if (clock64() - start > budget) {{ atomicAdd(state + 1, 1); break; }}
            __nanosleep(1000);
        }}
    }}
    __syncthreads();
}}
"""
_OCCUPIER_BUDGET_CYCLES = 50_000_000_000
_OCCUPIER_WAIT_S = 10.0


def _checked(result, what):
    error, *values = result
    if int(error) != 0:
        raise RuntimeError(f"{what} failed: {error}")
    return values[0] if len(values) == 1 else tuple(values)


@functools.cache
def _occupier(device):
    from cuda.bindings import driver, nvrtc

    program = _checked(
        nvrtc.nvrtcCreateProgram(
            _OCCUPIER_SOURCE.encode(), f"{_OCCUPIER}.cu".encode(), 0, [], []
        ),
        "nvrtcCreateProgram",
    )
    error = nvrtc.nvrtcCompileProgram(program, 1, [b"--gpu-architecture=sm_107a"])[0]
    if int(error) != 0:
        size = _checked(nvrtc.nvrtcGetProgramLogSize(program), "nvrtcGetProgramLogSize")
        log = b" " * size
        nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError(f"occupier build failed: {log.decode(errors='replace')}")
    size = _checked(nvrtc.nvrtcGetCUBINSize(program), "nvrtcGetCUBINSize")
    image = b" " * size
    _checked(nvrtc.nvrtcGetCUBIN(program, image), "nvrtcGetCUBIN")
    smem = int(torch.cuda.get_device_properties(device).shared_memory_per_block_optin)
    with torch.cuda.device(device):
        torch.cuda.current_stream().synchronize()
        module = _checked(driver.cuModuleLoadData(image), "cuModuleLoadData")
        function = _checked(
            driver.cuModuleGetFunction(module, _OCCUPIER.encode()),
            "cuModuleGetFunction",
        )
        attribute = (
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        )
        _checked(
            driver.cuFuncSetAttribute(function, attribute, smem), "cuFuncSetAttribute"
        )
    return module, function, smem


def _die_of(table, sm):
    word = int(table.entries[sm])
    return (word >> cake_locality.ENTRY_DOMAIN_SHIFT) & (
        (1 << cake_locality.ENTRY_DOMAIN_BITS) - 1
    )


def _run_beside_occupier(device, die, domains, table, launch):
    """``launch()`` on the current stream while the occupier holds half of die
    ``die``'s SMs on that die's green context; returns why the run does not prove
    that the whole launch ran beside it."""
    from cuda.bindings import driver

    _, function, smem = _occupier(device)
    blocks = domains.sm_counts[die] // 2
    release = torch.zeros(1, dtype=torch.int32, device=f"cuda:{device}")
    state = torch.zeros(2, dtype=torch.int32, device=f"cuda:{device}")
    smids = torch.full((blocks,), -1, dtype=torch.int32, device=f"cuda:{device}")
    torch.cuda.synchronize()
    # No garbage collection while the occupier holds SMs: freeing a split
    # allocation synchronizes the device, which waits for the occupier.
    gc.collect()
    gc.disable()
    try:
        occupier_stream = driver.CUstream(domains.streams[die].cuda_stream)
        args = (
            (
                release.data_ptr(),
                state.data_ptr(),
                smids.data_ptr(),
                _OCCUPIER_BUDGET_CYCLES,
            ),
            (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong),
        )
        _checked(
            driver.cuLaunchKernel(
                function, blocks, 1, 1, 1024, 1, 1, smem, occupier_stream, args, 0
            ),
            "cuLaunchKernel(occupier)",
        )
        reader = torch.cuda.Stream(device=device)
        started = time.monotonic()
        resident = 0
        while time.monotonic() - started < _OCCUPIER_WAIT_S:
            with torch.cuda.stream(reader):
                seen = state.to("cpu", non_blocking=False)
            resident = int(seen[0])
            if resident >= blocks:
                break
            time.sleep(1e-3)
        launch()
        release.fill_(1)
        _checked(driver.cuStreamSynchronize(occupier_stream), "cuStreamSynchronize")
        torch.cuda.synchronize()
    finally:
        gc.enable()
    held, timed_out = smids.tolist(), int(state[1])
    failures = []
    if resident != blocks:
        failures.append(f"only {resident} of {blocks} occupier blocks were resident")
    if timed_out:
        failures.append(f"{timed_out} occupier blocks ran out of their budget")
    if len(set(held)) != blocks or {_die_of(table, sm) for sm in held} != {die}:
        failures.append(
            f"the occupier held SMs {sorted(held)}, not {blocks} of die {die}"
        )
    return failures


def _run_on_die(domains, die, launch):
    """``launch()`` on die ``die``'s green-context stream, ordered after and before
    the current stream."""
    current = torch.cuda.current_stream()
    stream = domains.streams[die]
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        launch()
    current.wait_stream(stream)


@_packed_serving
@pytest.mark.parametrize(
    "cell,placement",
    [(cell, placement) for cell in _DOMAIN_CELLS for placement in _RACE_PLACEMENTS],
    ids=[
        f"{_cell_id(cell)}-{placement}"
        for cell in _DOMAIN_CELLS
        for placement in _RACE_PLACEMENTS
    ],
)
def test_domain_twin_race(cell, placement):
    """(g) A stress gate (``FLASHINFER_CAKE_DOMAIN_RACE_LAUNCHES`` launches per
    placement): on the test's stream, beside an occupier on each die, and on each
    die's green context.  The output is poisoned before every launch and must
    equal the first launch's bits; the counters must be zero after every launch."""
    launches = os.environ.get(_RACE_ENV)
    if not launches:
        pytest.skip(f"a stress gate: set {_RACE_ENV} to the launches per placement")
    device, table = _require_domain()
    _require_routed(cell, device, table)
    from flashinfer.locality_domain import get_localized_domains

    _, _, expected_m = cell
    a, b, _, _, a_packed, b_packed, masked_m, reference, valid = _serving_problem(*cell)
    homed = _home(b, table)
    prepared = prepare_batch_deepgemm_fp8_nt_groupwise(
        a, homed, a_packed, b_packed, masked_m, expected_m
    )
    assert prepared.route == DOMAIN_SERVING_ROUTE
    counters = (prepared.workspace["counters0"], prepared.workspace["counters1"])
    domains = get_localized_domains(device, sm_split="strict")
    if placement.startswith("occupier"):
        _occupier(device)
    prepared.out.fill_(float("nan"))
    prepared.launch()
    torch.cuda.synchronize()
    _assert_valid_rows_close(prepared.out, reference, masked_m)
    first = _valid_bits(prepared.out, valid)
    failures, wrong = [], 0
    for index in range(int(launches)):
        prepared.out.fill_(float("nan"))
        if placement.startswith("occupier"):
            found = _run_beside_occupier(
                device, int(placement[-1]), domains, table, prepared.launch
            )
        elif placement.startswith("one-die"):
            _run_on_die(domains, int(placement[-1]), prepared.launch)
            found = []
        else:
            prepared.launch()
            found = []
        torch.cuda.synchronize()
        if not torch.equal(_valid_bits(prepared.out, valid), first):
            found.append("the output differs bitwise from the first launch")
        if any(int(t.abs().sum()) != 0 for t in counters):
            found.append(f"counters left {[t.tolist() for t in counters]}")
        wrong += bool(found)
        if found and len(failures) < 20:
            failures += [f"launch {index + 1}: {failure}" for failure in found]
    print(
        f"RACE {_cell_id(cell)} {placement} counts={table.counts} "
        f"launches={launches} wrong={wrong}"
    )
    assert not wrong, "\n".join(failures)
