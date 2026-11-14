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

The persistent Cake program behind ``BlockSparseAttentionWrapper(backend=
"vsa_sm120_blk64")``: the plan's work table (CPU), and on SM120/SM121 the rows
the plan splits into parts (merged by the last part to arrive), ragged per-head
selections with empty and full rows, GQA, caller-provided ``out`` / ``lse``, and
CUDA graph replay of one plan (the split counters rewind in the kernel).
"""

import math

import pytest
import torch

from flashinfer import cake_vsa_sm120
from flashinfer.sparse import BlockSparseAttentionWrapper
from flashinfer.utils import is_sm12x_supported

R = C = 64
HEAD_DIM = 128

requires_sm12x = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")),
    reason="vsa_sm120_blk64 backend requires SM120/SM121 GPU",
)


def _random_counts(num_heads, mb, nb, generator):
    counts = torch.randint(0, nb + 1, (num_heads, mb), generator=generator)
    counts[0, 0] = 0
    counts[-1, -1] = nb
    return counts


KERNELS = ("vsa_sm120_blk64", "vsa_sm120_blk64_pair")
# The virtual CTAs (128-thread groups) of one CTA of each kernel.
GROUPS = {"vsa_sm120_blk64": 1, "vsa_sm120_blk64_pair": 2}


@pytest.mark.parametrize("group_size", [1, 4])
@pytest.mark.parametrize("num_sms", [1, 7, 188])
@pytest.mark.parametrize("kernel", [None, *KERNELS])
@pytest.mark.parametrize("pack", [None, False, True])
def test_work_table_covers_every_row(group_size, num_sms, kernel, pack):
    generator = torch.Generator().manual_seed(group_size * 100 + num_sms)
    num_heads, mb, nb = 8, 9, 40
    counts = _random_counts(num_heads, mb, nb, generator)
    first = (torch.cumsum(counts.reshape(-1), 0) - counts.reshape(-1)).view_as(counts)
    plan = cake_vsa_sm120.build_work(
        counts, first, group_size, num_sms, kernel=kernel, pack=pack
    )
    uniform = cake_vsa_sm120.build_work(
        counts, first, group_size, num_sms, kernel=kernel, pack=False
    )
    work = plan["work"].long()
    packed = not torch.equal(plan["work"], uniform["work"])
    # Packing only below _PACK_ROWS rows per virtual CTA (72 rows: on 188 SMs).
    assert not packed or num_heads * mb < cake_vsa_sm120._PACK_ROWS * 2 * num_sms
    assert plan["total_work"] == work.shape[0]
    assert plan["kernel"] == kernel or (kernel is None and plan["kernel"] in KERNELS)
    ctas = 2 * num_sms // GROUPS[plan["kernel"]]
    assert plan["grid"] == (max(1, min(work.shape[0], ctas)), 1, 1)
    sizes = work[:, 3] - work[:, 2]
    if packed and pack is None:
        rows = [plan["work"].tolist(), uniform["work"].tolist()]
        resident = 2 * num_sms
        assert cake_vsa_sm120._pack_time(
            rows[0], resident, plan["kernel"]
        ) < cake_vsa_sm120._pack_time(rows[1], resident, uniform["kernel"])
    if not packed:
        assert torch.all(sizes[:-1] >= sizes[1:]), "items are ordered longest first"
    groups, slots = set(), set()
    for h in range(num_heads):
        for b in range(mb):
            rows = work[(work[:, 0] == h) & (work[:, 1] == b)]
            rows = rows[torch.argsort(rows[:, 2])]
            c, f = int(counts[h, b]), int(first[h, b])
            assert int(rows[0, 2]) == f and int(rows[-1, 3]) == f + c
            assert torch.equal(rows[1:, 2], rows[:-1, 3]), "parts are contiguous"
            parts = rows.shape[0]
            assert torch.all(rows[:, 5] == parts)
            part_sizes = rows[:, 3] - rows[:, 2]
            assert packed or int(part_sizes.max() - part_sizes.min()) <= 1
            if parts == 1:
                assert int(rows[0, 4]) == -1
                continue
            assert len(set(rows[:, 4].tolist())) == 1
            groups.add(int(rows[0, 4]))
            assert sorted(rows[:, 6].tolist()) == list(range(parts))
            row_slots = set(rows[:, 7].tolist())
            assert len(row_slots) == parts and not row_slots & slots
            slots |= row_slots
    assert groups == set(range(len(groups)))
    assert slots == set(range(len(slots)))
    assert plan["split_counters"].numel() == max(1, len(groups))
    assert plan["partial_ws"].numel() >= max(1, len(slots)) * 68 * 128


def test_plan_picks_the_kernel_by_its_cost():
    """Few short rows (under a wave) run on vsa_sm120_blk64; many long rows
    (many waves) on vsa_sm120_blk64_pair, which streams a block faster."""
    small = torch.full((4, 16), 4, dtype=torch.int64)
    large = torch.full((24, 2160), 216, dtype=torch.int64)
    for counts, kernel in ((small, KERNELS[0]), (large, KERNELS[1])):
        first = (torch.cumsum(counts.reshape(-1), 0) - counts.reshape(-1)).view_as(
            counts
        )
        assert cake_vsa_sm120.build_work(counts, first, 1, 188)["kernel"] == kernel


def _blocks(plan, resident):
    """The KV blocks each of ``resident`` virtual CTAs runs under ``plan``."""
    work = plan["work"].long()
    grid = max(1, min(work.shape[0], resident))
    return torch.zeros(resident, dtype=torch.int64).index_add_(
        0, torch.arange(work.shape[0]) % grid, work[:, 3] - work[:, 2]
    )


@pytest.mark.parametrize("rows_per_head", [32, 64])
def test_packing_fills_the_last_wave(rows_per_head):
    """Equal rows of 32 blocks, 0.68 and 1.36 rows per virtual CTA on 188 SMs:
    the packed plan cuts rows so that every virtual CTA has work and the
    longest one has fewer blocks than in the uniform plan, deterministically."""
    counts = torch.full((8, rows_per_head), 32, dtype=torch.int64)
    first = (torch.cumsum(counts.reshape(-1), 0) - counts.reshape(-1)).view_as(counts)
    plans = [cake_vsa_sm120.build_work(counts, first, 1, 188) for _ in range(2)]
    uniform = cake_vsa_sm120.build_work(counts, first, 1, 188, pack=False)
    assert torch.equal(plans[0]["work"], plans[1]["work"])
    work = plans[0]["work"].long()
    assert uniform["total_work"] == 8 * rows_per_head < work.shape[0]
    assert int((work[:, 4] >= 0).sum()) > 0, "rows are cut"
    packed_blocks, uniform_blocks = _blocks(plans[0], 376), _blocks(uniform, 376)
    assert int(packed_blocks.min()) > 0
    assert int(packed_blocks.max()) < int(uniform_blocks.max())
    # K and V of up to the rows' _PACK_KV_BLOCKS limit over the 8 KV heads,
    # and beyond.
    limit = cake_vsa_sm120._PACK_KV_BLOCKS[8 * rows_per_head >= 376] // 8
    for num_kv_blocks, expected in (
        (32, plans[0]),
        (limit, plans[0]),
        (limit + 1, uniform),
    ):
        plan = cake_vsa_sm120.build_work(
            counts, first, 1, 188, num_kv_blocks=num_kv_blocks
        )
        assert torch.equal(plan["work"], expected["work"])


def test_split_cost_past_the_kv_limit():
    """Equal rows of 40 blocks, 0.68 rows per virtual CTA on 188 SMs: while the
    K and V of the 8 KV heads span at most _SPLIT_KV_BLOCKS blocks, the plan
    splits every row in two; past that, a split part costs _FAR_SPLIT_COST and
    the plan keeps the rows whole."""
    counts = torch.full((8, 32), 40, dtype=torch.int64)
    first = (torch.cumsum(counts.reshape(-1), 0) - counts.reshape(-1)).view_as(counts)
    limit = cake_vsa_sm120._SPLIT_KV_BLOCKS // 8
    near, far = (
        cake_vsa_sm120.build_work(counts, first, 1, 188, num_kv_blocks=n)
        for n in (limit, limit + 1)
    )
    assert near["total_work"] == 2 * counts.numel()
    assert far["total_work"] == counts.numel()


def test_split_cost_of_a_packed_plan_past_its_kv_limit():
    """Equal rows of 32 blocks, 1.02 rows per virtual CTA on 188 SMs, a plan
    that may be packed: while the K and V of the 8 KV heads span at most
    _PACK_SPLIT_KV_BLOCKS blocks, the packing starts from every row split in
    two; past that, a split part costs _FAR_SPLIT_COST and the packing starts
    from the whole rows, so it cuts only the rows past the first wave.  A plan
    that is not packed keeps the _SPLIT_KV_BLOCKS limit."""
    counts = torch.full((8, 48), 32, dtype=torch.int64)
    first = (torch.cumsum(counts.reshape(-1), 0) - counts.reshape(-1)).view_as(counts)
    limit = cake_vsa_sm120._PACK_SPLIT_KV_BLOCKS // 8
    assert limit < cake_vsa_sm120._SPLIT_KV_BLOCKS // 8

    def split_rows(plan):
        group = plan["work"][:, 4]
        return int(torch.unique(group[group >= 0]).numel())

    near, far = (
        cake_vsa_sm120.build_work(counts, first, 1, 188, num_kv_blocks=n)
        for n in (limit, limit + 1)
    )
    assert split_rows(near) == counts.numel()
    assert 0 < split_rows(far) <= counts.numel() - 376
    uniform = [
        cake_vsa_sm120.build_work(counts, first, 1, 188, pack=False, num_kv_blocks=n)
        for n in (limit, limit + 1)
    ]
    assert torch.equal(uniform[0]["work"], uniform[1]["work"])
    assert uniform[0]["total_work"] == 2 * counts.numel()


def _bsr(mb, nb, generator):
    counts = torch.randint(0, nb + 1, (mb,), generator=generator)
    indptr = torch.cat((torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, 0)))
    indices = torch.cat(
        [torch.randperm(nb, generator=generator)[: int(c)] for c in counts]
    )
    return indptr, indices


@pytest.mark.parametrize(
    "case", ["long_indptr", "short_indptr", "past_indices", "decreasing"]
)
def test_plan_bsr_refuses_rows_outside_the_selection(case):
    generator = torch.Generator().manual_seed(3)
    mb, nb, hq = 6, 10, 4
    indptr, indices = _bsr(mb, nb, generator)
    if case == "long_indptr":
        indptr = torch.cat((indptr, indptr[-1:]))
    elif case == "short_indptr":
        indptr = indptr[:-1]
    elif case == "past_indices":
        indptr[-1] = indices.numel() + 1
    else:
        low, high = int(indptr[2]), int(indptr[3])
        indptr[2], indptr[3] = high + 1, low
    with pytest.raises(ValueError, match="indptr"):
        cake_vsa_sm120.plan_bsr(
            indptr, indices, mb, hq, nb, 1, 188, torch.device("cpu")
        )


@pytest.mark.parametrize(
    "tensor", ["q_rows", "q_heads", "k_rows", "v_heads", "q_dim", "v_dtype", "fp32"]
)
def test_run_refuses_tensors_unlike_the_plan(tensor):
    generator = torch.Generator().manual_seed(5)
    mb, nb, hq, hkv = 4, 6, 4, 2
    indptr, indices = _bsr(mb, nb, generator)
    plan = cake_vsa_sm120.plan_bsr(
        indptr, indices, mb, hq, nb, hq // hkv, 188, torch.device("cpu")
    )
    q = torch.zeros(R * mb, hq, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.zeros(C * nb, hkv, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    if tensor == "q_rows":
        q = q[:-R]
    elif tensor == "q_heads":
        q = q[:, :-1]
    elif tensor == "k_rows":
        k = torch.cat((k, k[:C]))
    elif tensor == "v_heads":
        v = v[:, :1]
    elif tensor == "q_dim":
        q = torch.zeros(R * mb, hq, 2 * HEAD_DIM, dtype=torch.bfloat16)
    elif tensor == "v_dtype":
        v = v.half()
    else:
        q, k, v = q.float(), k.float(), v.float()
    with pytest.raises(ValueError, match="as planned|one dtype"):
        cake_vsa_sm120.run(plan, q, k, v, None, None, None, False)


@pytest.mark.parametrize("empty", ["kv", "q"])
def test_run_without_kv_or_query_blocks_gives_empty_rows(empty):
    """No KV block (or no query block): every row is empty, O = 0 and lse =
    -inf, without a launch."""
    mb, nb, hq, hkv = (3, 0, 4, 2) if empty == "kv" else (0, 5, 4, 2)
    block_mask = torch.zeros(hq, mb, nb, dtype=torch.bool)
    plan = cake_vsa_sm120.plan_block_mask(
        block_mask, hq // hkv, 188, torch.device("cpu")
    )
    q = torch.randn(R * mb, hq, HEAD_DIM).bfloat16()
    k = torch.randn(C * nb, hkv, HEAD_DIM).bfloat16()
    v = torch.randn_like(k)
    out = torch.full_like(q, float("nan"))
    lse = torch.full((R * mb, hq), float("nan"))
    got_out, got_lse = cake_vsa_sm120.run(plan, q, k, v, None, out, lse, True)
    assert got_out is out and got_lse is lse
    assert torch.all(out == 0) and torch.all(lse == float("-inf"))
    fresh = cake_vsa_sm120.run(plan, q, k, v, None, None, None, False)
    assert fresh.shape == q.shape and torch.all(fresh == 0)


def _reference(q, k, v, block_mask, sm_scale):
    """FP32 dense attention under the per-QO-head block mask; empty rows give
    O = 0 and lse = -inf."""
    m, hq, _ = q.shape
    n, hkv, _ = k.shape
    group = hq // hkv
    out = torch.zeros(m, hq, HEAD_DIM, dtype=torch.float32, device=q.device)
    lse = torch.full((m, hq), float("-inf"), dtype=torch.float32, device=q.device)
    for h in range(hq):
        mask = block_mask[h].repeat_interleave(R, 0).repeat_interleave(C, 1)
        scores = (q[:, h].float() @ k[:, h // group].float().T) * sm_scale
        scores = scores.masked_fill(~mask, float("-inf"))
        row_lse = torch.logsumexp(scores, dim=-1)
        live = torch.isfinite(row_lse)
        probs = torch.exp(scores - torch.where(live, row_lse, 0.0)[:, None])
        out[:, h] = torch.where(live[:, None], probs @ v[:, h // group].float(), 0.0)
        lse[:, h] = row_lse
    return out, lse


def _ragged_mask(hq, mb, nb, generator, device):
    """Per-head random selections, one empty and one full row: with the full
    row far longer than the mean, the plan splits it into parts."""
    mask = torch.zeros(hq, mb, nb, dtype=torch.bool)
    counts = _random_counts(hq, mb, nb, generator)
    counts = torch.minimum(counts, torch.full_like(counts, max(2, nb // 6)))
    counts[0, 0] = 0
    counts[-1, -1] = nb
    for h in range(hq):
        for b in range(mb):
            mask[h, b, torch.randperm(nb, generator=generator)[: int(counts[h, b])]] = (
                True
            )
    return mask.to(device)


@requires_sm12x
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hq,hkv", [(4, 4), (8, 2)])
def test_ragged_split_rows(dtype, hq, hkv):
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(hq * 10 + hkv)
    m, n = 1024, 4096
    mb, nb = m // R, n // C
    q = torch.randn(m, hq, HEAD_DIM, generator=generator).to(dtype).to(device)
    k = torch.randn(n, hkv, HEAD_DIM, generator=generator).to(dtype).to(device)
    v = torch.randn(n, hkv, HEAD_DIM, generator=generator).to(dtype).to(device)
    block_mask = _ragged_mask(hq, mb, nb, generator, device)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="vsa_sm120_blk64")
    wrapper.plan(
        None, None, m, n, R, C, hq, hkv, HEAD_DIM,
        q_data_type=dtype, block_mask=block_mask,
    )  # fmt: skip
    assert (wrapper._vsa_sm120_plan["work"][:, 4] >= 0).any(), "a row is split"

    out = torch.full_like(q, float("nan"))
    lse = torch.full((m, hq), float("nan"), dtype=torch.float32, device=device)
    got_out, got_lse = wrapper.run(q, k, v, out=out, lse=lse, return_lse=True)
    assert got_out.data_ptr() == out.data_ptr() and got_lse.data_ptr() == lse.data_ptr()
    ref_out, ref_lse = _reference(q, k, v, block_mask, sm_scale)
    torch.testing.assert_close(out.float(), ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-2)
    assert torch.equal(
        wrapper._vsa_sm120_plan["split_counters"],
        torch.zeros_like(wrapper._vsa_sm120_plan["split_counters"]),
    ), "the last part rewinds its counter"


@requires_sm12x
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_sms,part,pack", [(7, 2, None), (None, None, False), (None, None, True)]
)
def test_both_kernels_give_the_same_bits(dtype, num_sms, part, pack):
    """One plan's work table on either kernel (several waves of split rows on 7
    SMs' worth of virtual CTAs, and the device's own plan, uniform and packed):
    bitwise-equal out and lse, within the FP32 reference's tolerance."""
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(11)
    hq, hkv, m, n = 8, 2, 1024, 4096
    q = torch.randn(m, hq, HEAD_DIM, generator=generator).to(dtype).to(device)
    k = torch.randn(n, hkv, HEAD_DIM, generator=generator).to(dtype).to(device)
    v = torch.randn(n, hkv, HEAD_DIM, generator=generator).to(dtype).to(device)
    block_mask = _ragged_mask(hq, m // R, n // C, generator, device)
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    got = {}
    table = cake_vsa_sm120.plan_block_mask(
        block_mask, hq // hkv, num_sms, device, part=part, kernel=KERNELS[0], pack=pack
    )
    if pack:
        uniform = cake_vsa_sm120.plan_block_mask(
            block_mask, hq // hkv, num_sms, device, kernel=KERNELS[0], pack=False
        )
        assert not torch.equal(table["work"], uniform["work"]), "the plan is packed"
    for kernel in KERNELS:
        ctas = 2 * num_sms // GROUPS[kernel]
        grid = (max(1, min(table["total_work"], ctas)), 1, 1)
        plan = dict(table, kernel=kernel, grid=grid)
        got[kernel] = cake_vsa_sm120.run(plan, q, k, v, None, None, None, True)
        assert torch.equal(
            plan["split_counters"], torch.zeros_like(plan["split_counters"])
        )
    (out_a, lse_a), (out_b, lse_b) = got.values()
    assert torch.equal(out_a, out_b) and torch.equal(lse_a, lse_b)
    ref_out, ref_lse = _reference(q, k, v, block_mask, 1.0 / math.sqrt(HEAD_DIM))
    torch.testing.assert_close(out_a.float(), ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse_a, ref_lse, atol=1e-2, rtol=1e-2)


@requires_sm12x
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kernel", KERNELS)
def test_packed_plan_of_each_kernel(dtype, kernel):
    """Each kernel's own packed plan (its pieces sized by that kernel's item
    cost) on the device's SMs: within the FP32 reference's tolerance."""
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(13)
    hq, hkv, m, n = 8, 2, 1024, 4096
    q = torch.randn(m, hq, HEAD_DIM, generator=generator).to(dtype).to(device)
    k = torch.randn(n, hkv, HEAD_DIM, generator=generator).to(dtype).to(device)
    v = torch.randn(n, hkv, HEAD_DIM, generator=generator).to(dtype).to(device)
    block_mask = _ragged_mask(hq, m // R, n // C, generator, device)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    plans = [
        cake_vsa_sm120.plan_block_mask(
            block_mask, hq // hkv, num_sms, device, kernel=kernel, pack=pack
        )
        for pack in (True, False)
    ]
    assert plans[0]["kernel"] == kernel
    assert not torch.equal(plans[0]["work"], plans[1]["work"]), "the plan is packed"
    out, lse = cake_vsa_sm120.run(plans[0], q, k, v, None, None, None, True)
    assert torch.equal(
        plans[0]["split_counters"], torch.zeros_like(plans[0]["split_counters"])
    )
    ref_out, ref_lse = _reference(q, k, v, block_mask, 1.0 / math.sqrt(HEAD_DIM))
    torch.testing.assert_close(out.float(), ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-2)


@requires_sm12x
def test_strided_out_and_lse_get_the_same_bits():
    """A caller's ``out`` / ``lse`` that are head slices of larger buffers (not
    contiguous) receive the bits of a launch into contiguous ones."""
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(13)
    dtype, hq, hkv, m, n = torch.bfloat16, 4, 2, 512, 2048
    q = torch.randn(m, hq, HEAD_DIM, generator=generator).to(dtype).to(device)
    k = torch.randn(n, hkv, HEAD_DIM, generator=generator).to(dtype).to(device)
    v = torch.randn(n, hkv, HEAD_DIM, generator=generator).to(dtype).to(device)
    block_mask = _ragged_mask(hq, m // R, n // C, generator, device)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    plan = cake_vsa_sm120.plan_block_mask(block_mask, hq // hkv, num_sms, device)
    want_out, want_lse = cake_vsa_sm120.run(plan, q, k, v, None, None, None, True)
    out_wide = torch.full((m, 2 * hq, HEAD_DIM), 7.0, dtype=dtype, device=device)
    lse_wide = torch.full((m, 2 * hq), 7.0, dtype=torch.float32, device=device)
    out, lse = out_wide[:, :hq], lse_wide[:, hq:]
    assert not out.is_contiguous() and not lse.is_contiguous()
    got_out, got_lse = cake_vsa_sm120.run(plan, q, k, v, None, out, lse, True)
    assert got_out is out and got_lse is lse
    assert torch.equal(out, want_out) and torch.equal(lse, want_lse)
    assert torch.all(out_wide[:, hq:] == 7.0) and torch.all(lse_wide[:, :hq] == 7.0)


@requires_sm12x
@pytest.mark.parametrize(
    "case", ["q", "k", "v", "lse_in_q", "lse_in_out", "out_unaligned"]
)
def test_out_and_lse_get_the_out_of_place_bits(case):
    """An ``out`` that is ``q``, ``k`` or ``v`` itself, an ``lse`` in ``q``'s or
    in a strided ``out``'s memory, or an ``out`` that is not 16-byte aligned
    receives the bits of an out-of-place launch (main computed into a buffer
    of its own and copied out, then lse).  The plan runs on one SM in many
    waves with split rows, so a store into memory that a later work item
    still reads would show."""
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(17)
    dtype, h, m = torch.bfloat16, 4, 1024
    q = torch.randn(m, h, HEAD_DIM, generator=generator).to(dtype).to(device)
    k = torch.randn(m, h, HEAD_DIM, generator=generator).to(dtype).to(device)
    v = torch.randn(m, h, HEAD_DIM, generator=generator).to(dtype).to(device)
    block_mask = _ragged_mask(h, m // R, m // C, generator, device)
    plan = cake_vsa_sm120.plan_block_mask(block_mask, 1, 1, device, part=2)
    assert (plan["work"][:, 4] >= 0).any(), "a row is split"
    want_out, want_lse = cake_vsa_sm120.run(plan, q, k, v, None, None, None, True)
    inputs = {"q": q.clone(), "k": k.clone(), "v": v.clone()}
    out = lse = None
    if case in inputs:
        out = inputs[case]
    elif case == "lse_in_q":
        lse = inputs["q"].view(torch.float32).view(-1)[: m * h].view(m, h)
    elif case == "lse_in_out":
        base = torch.empty(h * m * HEAD_DIM, dtype=dtype, device=device)
        out = base.view(h, m, HEAD_DIM).transpose(0, 1)
        lse = base.view(torch.float32)[: m * h].view(m, h)
    else:
        buffer = torch.empty(m * h * HEAD_DIM + 4, dtype=dtype, device=device)
        out = buffer[4:].view(m, h, HEAD_DIM)
        assert out.is_contiguous() and out.data_ptr() % 16
    got_out, got_lse = cake_vsa_sm120.run(
        plan, inputs["q"], inputs["k"], inputs["v"], None, out, lse, True
    )
    assert out is None or got_out is out
    assert lse is None or got_lse is lse
    assert torch.equal(got_lse, want_lse)
    if case == "lse_in_out":
        # lse is written last over head 0's first rows of out
        assert torch.equal(got_out[:, 1:], want_out[:, 1:])
    else:
        assert torch.equal(got_out, want_out)


@requires_sm12x
def test_graph_replay_matches_eager():
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(7)
    dtype, hq, hkv, m, n = torch.bfloat16, 4, 4, 512, 2048
    q = torch.randn(m, hq, HEAD_DIM, generator=generator).to(dtype).to(device)
    k = torch.randn(n, hkv, HEAD_DIM, generator=generator).to(dtype).to(device)
    v = torch.randn(n, hkv, HEAD_DIM, generator=generator).to(dtype).to(device)
    block_mask = _ragged_mask(hq, m // R, n // C, generator, device)
    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="vsa_sm120_blk64")
    wrapper.plan(
        None, None, m, n, R, C, hq, hkv, HEAD_DIM,
        q_data_type=dtype, block_mask=block_mask,
    )  # fmt: skip
    eager = wrapper.run(q, k, v).clone()

    out = torch.empty_like(q)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        wrapper.run(q, k, v, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k, v, out=out)
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)
