# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the generated Cake programs behind ``batch_deepgemm_fp8_nt_groupwise(backend="cake")``."""

import pytest
import torch

from flashinfer.gemm import batch_deepgemm_fp8_nt_groupwise
from flashinfer.gemm.cake_batch_deepgemm_fp8 import (
    PACKED_ARCHS,
    PACKED_GROUPS,
    PACKED_NK,
    SUPPORTED_NK,
    is_batch_deepgemm_fp8_nt_groupwise_cake_available,
    prepare_batch_deepgemm_fp8_nt_groupwise,
    select_route,
    supported_nk,
)
from flashinfer.jit.gemm.cake_batch_deepgemm_fp8 import (
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


@pytest.fixture(autouse=True)
def _require_generated_program():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip(
            "generated Cake batch DeepGEMM FP8 programs target SM100a, SM103a and SM107a"
        )
    if not is_batch_deepgemm_fp8_nt_groupwise_cake_available(device):
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
