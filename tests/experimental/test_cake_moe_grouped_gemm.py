"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import pytest
import torch

from flashinfer.experimental.cake_moe_grouped_gemm import (
    grouped_gemm_dgrad,
    grouped_gemm_fwd,
    grouped_gemm_wgrad,
)
from flashinfer.experimental.cake_moe_grouped_gemm import cake_backend as cb
from flashinfer.experimental.cake_moe_grouped_gemm.cake_backend import (
    SUPPORTED_COMPUTE_CAPABILITIES,
    prepare_grouped_gemm_dgrad,
    prepare_grouped_gemm_fwd,
    prepare_grouped_gemm_wgrad,
)
from flashinfer.experimental.cake_moe_grouped_gemm.cake_jit import HOST_PLAN_CONSTANTS

ATOL = RTOL = 1e-2  # bf16 outputs
FP32_REL_L2 = 1e-4  # fp32 weight gradient vs the fp64 reference

# Group size lists: an empty group, a one-row group, sizes that are not tile
# multiples, and a long-reduction case that can select the 512 k tile.
RAGGED_E4 = [300, 0, 1, 700]
RAGGED_E6 = [1, 257, 511, 4099, 0, 33]
TINY_E5 = [1, 2, 3, 0, 4]
LONG_E3 = [6145, 0, 12289]


def _require_program(op, out_dtype=None):
    if not torch.cuda.is_available():
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    device = torch.device("cuda", 0)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    if arch is None:
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    if not cb.generated_program_available(device, op, out_dtype):
        pytest.skip(
            f"no generated ragged grouped GEMM program for {op} registered for {arch}"
        )
    return device


def _offs(sizes, device):
    return torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )


def _inputs(sizes, n, k, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    sum_m = sum(sizes)
    x = torch.randn(sum_m, k, generator=gen, device=device, dtype=torch.float32)
    w = torch.randn(len(sizes), n, k, generator=gen, device=device, dtype=torch.float32)
    g = torch.randn(sum_m, n, generator=gen, device=device, dtype=torch.float32)
    x = (x / math.sqrt(k)).to(torch.bfloat16)
    w = (w / math.sqrt(k)).to(torch.bfloat16)
    g = g.to(torch.bfloat16)
    return x, w, g, _offs(sizes, device)


def _bounds(sizes):
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return list(zip([0] + ends[:-1], ends, strict=True))


def _reference(op, a, b, sizes):
    """Per-group fp64 reference in fp32: fwd (x, w), dgrad (g, w), wgrad (g, x)."""
    if op == "fwd":
        out = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32, device=a.device)
        for e, (s, t) in enumerate(_bounds(sizes)):
            if t > s:
                out[s:t] = (a[s:t].double() @ b[e].double().T).float()
    elif op == "dgrad":
        out = torch.zeros(a.shape[0], b.shape[2], dtype=torch.float32, device=a.device)
        for e, (s, t) in enumerate(_bounds(sizes)):
            if t > s:
                out[s:t] = (a[s:t].double() @ b[e].double()).float()
    else:
        out = torch.zeros(
            len(sizes), a.shape[1], b.shape[1], dtype=torch.float32, device=a.device
        )
        for e, (s, t) in enumerate(_bounds(sizes)):
            if t > s:
                out[e] = (a[s:t].double().T @ b[s:t].double()).float()
    return out


def _assert_bf16_close(out, ref):
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Host plan (CPU)
# ---------------------------------------------------------------------------


def _constants():
    if not HOST_PLAN_CONSTANTS:
        pytest.skip("no host plan constants registered in this checkout")
    return HOST_PLAN_CONSTANTS


@pytest.mark.parametrize("sm_count", [148, 160, 212])
@pytest.mark.parametrize(
    "sizes", [RAGGED_E4, RAGGED_E6, LONG_E3, [131072] * 1, [4096] * 32]
)
def test_plan_invariants(sm_count, sizes):
    c = _constants()
    sum_m, e = sum(sizes), len(sizes)
    cta_group = int(c["cta_group"])
    for op, n, k in (("fwd", 4096, 6144), ("dgrad", 4096, 6144)):
        plan = cb.plan_fwd_dgrad(
            op, sum_m=sum_m, num_groups=e, n=n, k=k, sm_count=sm_count, constants=c
        )
        grid_x = plan["grid"][0]
        assert (
            grid_x % cta_group == 0
            and cta_group <= grid_x <= (sm_count // cta_group) * cta_group
        )
        assert plan["launches"] == 1 and plan["tile_k"] is None
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for n, k in ((4096, 6144), (6144, 2048), (256, 256)):
            tile_k = cb.select_wgrad_tile(k, arch, sum_m, e, c)
            assert tile_k in (256, 512) and k % tile_k == 0
            if tile_k == 512:
                assert (
                    arch in c["k512_archs"]
                    and sum_m / e >= c["k512_min_rows_per_group"]
                )
            plan = cb.plan_wgrad(
                sum_m=sum_m,
                num_groups=e,
                n=n,
                k=k,
                tile_k=tile_k,
                sm_count=sm_count,
                reduce_threads=256,
                constants=c,
            )
            assert plan["grid"][0] == plan["clusters"] * cta_group
            assert 0 <= plan["num_tail"] < plan["clusters"]
            assert 1 <= plan["tail_splits"] <= int(c["tail_plan"]["max_splits"])
            assert plan["raster_rows"] in (0, 1)
            assert plan["launches"] == 1 + int(plan["tail_splits"] > 1)
            assert (plan["tail_reduce_grid"] is None) == (plan["tail_splits"] == 1)
            if plan["tail_reduce_grid"] is not None:
                assert plan["tail_reduce_grid"][0] > 0
            if plan["num_tail"] == 0:
                assert plan["tail_splits"] == 1


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sizes,n,k", [(RAGGED_E4, 512, 256), (RAGGED_E6, 1024, 512), (TINY_E5, 256, 64)]
)
def test_fwd_matches_reference(sizes, n, k):
    device = _require_program("fwd")
    x, w, _g, offs = _inputs(sizes, n, k, device)
    out = grouped_gemm_fwd(x, w, offs)
    assert out.shape == (sum(sizes), n)
    _assert_bf16_close(out, _reference("fwd", x, w, sizes))


@pytest.mark.parametrize(
    "sizes,n,k", [(RAGGED_E4, 256, 512), (RAGGED_E6, 1024, 512), (TINY_E5, 64, 256)]
)
def test_dgrad_matches_reference(sizes, n, k):
    device = _require_program("dgrad")
    _x, w, g, offs = _inputs(sizes, n, k, device)
    out = grouped_gemm_dgrad(g, w, offs)
    assert out.shape == (sum(sizes), k)
    _assert_bf16_close(out, _reference("dgrad", g, w, sizes))


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "sizes,n,k", [(RAGGED_E4, 512, 256), (RAGGED_E6, 512, 256), (TINY_E5, 256, 512)]
)
def test_wgrad_matches_reference(sizes, n, k, out_dtype):
    device = _require_program("wgrad", out_dtype)
    x, _w, g, offs = _inputs(sizes, n, k, device)
    launch = prepare_grouped_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
    out = launch.launch()
    assert out.shape == (len(sizes), n, k) and out.dtype == out_dtype
    ref = _reference("wgrad", g, x, sizes)
    if out_dtype == torch.bfloat16:
        _assert_bf16_close(out, ref)
    else:
        rel_l2 = (out - ref).norm() / ref.norm()
        assert rel_l2.item() <= FP32_REL_L2, (
            f"fp32 weight gradient rel-L2 {rel_l2.item():.3e}"
        )
    for e, (s, t) in enumerate(_bounds(sizes)):
        if t == s:
            assert torch.equal(out[e], torch.zeros_like(out[e])), (
                f"empty group {e} must be exact zeros"
            )
    # Deterministic: a second launch into the same output is bitwise identical.
    first = out.clone()
    out.fill_(float("nan"))
    launch.launch()
    torch.cuda.synchronize()
    assert torch.equal(out, first)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_long_reduction_selects_a_valid_tile(out_dtype):
    device = _require_program("wgrad", out_dtype)
    n, k = 512 if out_dtype == torch.bfloat16 else 256, 1024
    x, _w, g, offs = _inputs(LONG_E3, n, k, device, seed=3)
    launch = prepare_grouped_gemm_wgrad(g, x, offs, out_dtype=out_dtype)
    assert launch.plan["tile_k"] in (256, 512) and k % launch.plan["tile_k"] == 0
    out = launch.launch()
    ref = _reference("wgrad", g, x, LONG_E3)
    if out_dtype == torch.bfloat16:
        _assert_bf16_close(out, ref)
    else:
        assert ((out - ref).norm() / ref.norm()).item() <= FP32_REL_L2
    assert torch.equal(out[1], torch.zeros_like(out[1]))


def test_prepared_launch_reuses_buffers_without_allocation():
    device = _require_program("fwd")
    x, w, _g, offs = _inputs(RAGGED_E6, 1024, 512, device)
    out = torch.full(
        (sum(RAGGED_E6), 1024), float("nan"), dtype=torch.bfloat16, device=device
    )
    launch = prepare_grouped_gemm_fwd(x, w, offs, out=out)
    assert launch.launch() is out
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats(device)
    launch.launch()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats(device)
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    _assert_bf16_close(out, _reference("fwd", x, w, RAGGED_E6))


def test_prepared_launch_follows_new_offsets():
    """The same prepared launch serves new group boundaries written into offs."""
    device = _require_program("dgrad")
    first, second = [300, 0, 1, 700], [1, 500, 250, 250]
    _x, w, g, offs = _inputs(first, 256, 512, device)
    launch = prepare_grouped_gemm_dgrad(g, w, offs)
    _assert_bf16_close(launch.launch(), _reference("dgrad", g, w, first))
    offs.copy_(_offs(second, device))
    _assert_bf16_close(launch.launch(), _reference("dgrad", g, w, second))


def test_padded_outputs_leave_the_padding_untouched():
    device = _require_program("fwd")
    device = _require_program("wgrad", torch.bfloat16)
    x, w, g, offs = _inputs(RAGGED_E4, 512, 256, device)
    sum_m, n, k = sum(RAGGED_E4), 512, 256
    padded = torch.full(
        (sum_m, n + 64), float("nan"), dtype=torch.bfloat16, device=device
    )
    out = grouped_gemm_fwd(x, w, offs, out=padded[:, :n])
    _assert_bf16_close(out, _reference("fwd", x, w, RAGGED_E4))
    assert torch.isnan(padded[:, n:]).all()
    padded = torch.full(
        (len(RAGGED_E4), n, k + 32), float("nan"), dtype=torch.bfloat16, device=device
    )
    out = grouped_gemm_wgrad(g, x, offs, out=padded[:, :, :k])
    _assert_bf16_close(out, _reference("wgrad", g, x, RAGGED_E4))
    assert torch.isnan(padded[:, :, k:]).all()


def test_empty_batch():
    device = _require_program("fwd")
    device = _require_program("wgrad", torch.float32)
    x, w, g, offs = _inputs([0, 0], 256, 256, device)
    y = torch.full((0, 256), 0.0, dtype=torch.bfloat16, device=device)
    assert grouped_gemm_fwd(x, w, offs, out=y).shape == (0, 256)
    dw = torch.full((2, 256, 256), float("nan"), dtype=torch.float32, device=device)
    assert torch.equal(grouped_gemm_wgrad(g, x, offs, out=dw), torch.zeros_like(dw))


def test_rejects_misaligned_shapes():
    device = _require_program("fwd")
    x, w, _g, offs = _inputs(RAGGED_E4, 384, 256, device)  # N % 256 != 0
    with pytest.raises(ValueError):
        prepare_grouped_gemm_fwd(x, w, offs)
    with pytest.raises(ValueError):
        prepare_grouped_gemm_fwd(x, w, offs.to(torch.int64))


# ---------------------------------------------------------------------------
# Stable-API opt-in, the m_indptr form and the autograd wrapper
# ---------------------------------------------------------------------------


def _m_indptr(sizes, device):
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return torch.tensor([0] + ends, dtype=torch.int32, device=device)


def test_grouped_mm_bf16_cake_backend_matches_reference():
    device = _require_program("fwd")
    from flashinfer.grouped_mm import grouped_mm_bf16

    x, w, _g, _offs = _inputs(RAGGED_E6, 1024, 512, device)
    out = grouped_mm_bf16(x, w, _m_indptr(RAGGED_E6, device), backend="cake")
    assert out.shape == (sum(RAGGED_E6), 1024) and out.dtype == torch.bfloat16
    _assert_bf16_close(out, _reference("fwd", x, w, RAGGED_E6))
    # rows past m_indptr[-1] belong to no group and are left untouched
    padded_x = torch.cat([x, torch.zeros(37, 512, dtype=x.dtype, device=device)])
    out = torch.full(
        (sum(RAGGED_E6) + 37, 1024), float("nan"), dtype=torch.bfloat16, device=device
    )
    grouped_mm_bf16(padded_x, w, _m_indptr(RAGGED_E6, device), out=out, backend="cake")
    _assert_bf16_close(out[: sum(RAGGED_E6)], _reference("fwd", x, w, RAGGED_E6))
    assert torch.isnan(out[sum(RAGGED_E6) :]).all()


def test_grouped_mm_bf16_cake_backend_rejects_unsupported_options():
    device = _require_program("fwd")
    from flashinfer.grouped_mm import grouped_mm_bf16

    x, w, _g, _offs = _inputs(RAGGED_E4, 512, 256, device)
    m_indptr = _m_indptr(RAGGED_E4, device)
    with pytest.raises(ValueError):
        grouped_mm_bf16(x, w, m_indptr, out_dtype=torch.float32, backend="cake")
    with pytest.raises(ValueError):
        grouped_mm_bf16(x, w, m_indptr, backend="cake", tactic=0)


def test_m_indptr_form_matches_offs_form():
    device = _require_program("fwd")
    device = _require_program("dgrad")
    device = _require_program("wgrad", torch.bfloat16)
    x, w, g, offs = _inputs(RAGGED_E6, 512, 256, device)
    m_indptr = _m_indptr(RAGGED_E6, device)
    assert torch.equal(grouped_gemm_fwd(x, w, m_indptr), grouped_gemm_fwd(x, w, offs))
    assert torch.equal(
        grouped_gemm_dgrad(g, w, m_indptr), grouped_gemm_dgrad(g, w, offs)
    )
    assert torch.equal(
        grouped_gemm_wgrad(g, x, m_indptr, num_groups=len(RAGGED_E6)),
        grouped_gemm_wgrad(g, x, offs),
    )
    with pytest.raises(ValueError):  # neither E nor E + 1 entries
        grouped_gemm_wgrad(g, x, m_indptr[:-2], num_groups=len(RAGGED_E6))


def test_autograd_wrapper_matches_reference_and_is_deterministic():
    device = _require_program("fwd")
    device = _require_program("dgrad")
    device = _require_program("wgrad", torch.bfloat16)
    from flashinfer.experimental.cake_moe_grouped_gemm import cake_grouped_mm

    x, w, g, offs = _inputs(RAGGED_E6, 512, 256, device)
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    y = cake_grouped_mm(x, w, offs)
    assert y.shape == (sum(RAGGED_E6), 512) and y.dtype == torch.bfloat16
    _assert_bf16_close(y.detach(), _reference("fwd", x.detach(), w.detach(), RAGGED_E6))
    y.backward(g)
    assert x.grad.dtype == torch.bfloat16 and w.grad.dtype == torch.bfloat16
    _assert_bf16_close(x.grad, _reference("dgrad", g, w.detach(), RAGGED_E6))
    _assert_bf16_close(w.grad, _reference("wgrad", g, x.detach(), RAGGED_E6))
    for e, (s, t) in enumerate(_bounds(RAGGED_E6)):
        if t == s:
            assert torch.equal(w.grad[e], torch.zeros_like(w.grad[e])), (
                f"empty group {e} must receive an exactly zero weight gradient"
            )
    dx, dw = x.grad.clone(), w.grad.clone()
    x.grad = None
    w.grad = None
    # Same gradients from the m_indptr form; bitwise reproducible.
    cake_grouped_mm(x, w, _m_indptr(RAGGED_E6, device), deterministic=True).backward(g)
    torch.cuda.synchronize()
    assert torch.equal(x.grad, dx) and torch.equal(w.grad, dw)
    with pytest.raises(ValueError):
        cake_grouped_mm(x.float(), w, offs)
