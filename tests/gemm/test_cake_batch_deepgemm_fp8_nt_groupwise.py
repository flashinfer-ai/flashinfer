# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the generated Cake programs behind ``batch_deepgemm_fp8_nt_groupwise(backend="cake")``."""

import pytest
import torch

from flashinfer.gemm import batch_deepgemm_fp8_nt_groupwise
from flashinfer.gemm.cake_batch_deepgemm_fp8 import (
    PACKED_GROUPS,
    PACKED_NK,
    SUPPORTED_NK,
    is_batch_deepgemm_fp8_nt_groupwise_cake_available,
    prepare_batch_deepgemm_fp8_nt_groupwise,
    select_route,
)
from flashinfer.jit.gemm.cake_batch_deepgemm_fp8 import (
    PROGRAMS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    device_arch,
    device_sm_count,
)
from flashinfer.utils import BackendSupportedError

ATOL = RTOL = 3e-2


@pytest.fixture(autouse=True)
def _require_generated_program():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip(
            "generated Cake batch DeepGEMM FP8 programs target SM100a and SM103a"
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


def test_launch_allocates_nothing():
    device = torch.device("cuda")
    a, b, a_scale, b_scale, masked_m, expected_m = _make_inputs(
        2, 4096, 4096, 2048, seed=928, device=device, masked=[230, 230]
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


def test_out_of_band_problems_raise():
    device = torch.device("cuda")
    a, b, a_scale, b_scale, masked_m, expected_m = _make_inputs(
        2, 256, 256, 256, seed=930, device=device
    )
    assert (256, 256) not in SUPPORTED_NK
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
    with pytest.raises(BackendSupportedError):
        batch_deepgemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, masked_m, expected_m, backend="nonexistent"
        )


def test_route_chain_covers_inventory_with_registered_programs():
    device = torch.device("cuda")
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = device_arch(index)
    sm_count = device_sm_count(index)
    for n, k in sorted(SUPPORTED_NK):
        for groups, m in ((1, 128), (4, 1024), (64, 256), (256, 128)):
            for expected_m in (1, m // 2, m):
                plan = select_route(arch, sm_count, groups, m, n, k, expected_m)
                assert plan is not None, (n, k, groups, m, expected_m)
                for stage in plan.stages:
                    assert arch in PROGRAMS[stage.program]["arches"]
                    assert all(int(v) > 0 for v in stage.grid)
    # Geometry outside the inventory is a named gap, never a fallback.
    assert select_route(arch, sm_count, 1, 128, 256, 256, 1) is None
