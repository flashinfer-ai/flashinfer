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

from flashinfer import cake_vsa
from flashinfer.sparse import BlockSparseAttentionWrapper

BLOCK = 64
HEAD_DIM = 128


def _is_sm100_or_sm103() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() in ((10, 0), (10, 3))


requires_gpu = pytest.mark.skipif(
    not _is_sm100_or_sm103(), reason="Cake VSA requires SM100 or SM103"
)


def _counters_are_zero(counters):
    """The ticket counter (``uint32``) is all zero; read through an ``int32``
    view, since ``uint32`` lacks reductions on some devices."""
    return not bool(counters.view(torch.int32).any())


def _shared_mask(heads, mb, nb, counts, device, seed=0):
    """One selection shared by every head: ``counts[i]`` random blocks of row i."""
    generator = torch.Generator().manual_seed(seed)
    mask = torch.zeros((mb, nb), dtype=torch.bool)
    for row, count in enumerate(counts):
        mask[row, torch.randperm(nb, generator=generator)[:count]] = True
    return mask.to(device).unsqueeze(0).expand(heads, -1, -1).contiguous()


def _ragged_counts(mb, nb, step=13):
    return [1 + (row * step) % nb for row in range(mb)]


def _plan(wrapper, mask, heads, M, N, **kw):
    wrapper.plan(
        None,
        None,
        M,
        N,
        BLOCK,
        BLOCK,
        heads,
        heads,
        HEAD_DIM,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        block_mask=mask,
        **kw,
    )


def _inputs(M, N, heads, device, seed=0):
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((M, heads, HEAD_DIM), generator=generator, device=device)
    k = torch.randn((N, heads, HEAD_DIM), generator=generator, device=device)
    v = torch.randn((N, heads, HEAD_DIM), generator=generator, device=device)
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def _dense_reference(q, k, v, mask):
    scores = torch.einsum("mhd,nhd->hmn", q.float(), k.float()) / math.sqrt(HEAD_DIM)
    token_mask = mask.repeat_interleave(BLOCK, 1).repeat_interleave(BLOCK, 2)
    scores.masked_fill_(~token_mask, float("-inf"))
    output = torch.einsum("hmn,nhd->mhd", torch.softmax(scores, dim=-1), v.float())
    return output.bfloat16(), torch.logsumexp(scores, dim=-1).transpose(0, 1)


def _forced(plan, q, k, v, return_lse):
    """The balanced program launched directly from the plan's metadata, with a
    counter of its own."""
    import tvm_ffi

    from flashinfer.jit import cake_vsa_balanced
    from flashinfer.jit.cake_program import CAPABILITY_TARGET

    program = cake_vsa_balanced.VSA_BLK64_BALANCED_BF16
    entry = program.entry(CAPABILITY_TARGET[torch.cuda.get_device_capability()])
    metadata = plan["blk64_balanced"]
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    counters = torch.zeros((4,), dtype=torch.uint32, device=q.device)
    tiles = plan["mb"] * plan["num_qo_heads"]
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = cake_vsa.cake_vsa_balanced_launch_grid(tiles, sm_count)
    values = {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "lse": lse,
        "indptr": metadata["indptr"],
        "indices": metadata["indices"],
        "sequence_q": plan["M"],
        "query_blocks": plan["mb"],
        "total_tiles": tiles,
        "num_heads": plan["num_qo_heads"],
        "softmax_scale_log2": (1.0 / math.sqrt(HEAD_DIM)) / math.log(2.0),
        "return_lse": int(return_lse),
        "queue_counters": counters,
        "order_debug": torch.empty((tiles,), dtype=torch.int32, device=q.device),
        "debug_order": 0,
        "grid_x": grid[0],
        "grid_y": grid[1],
        "grid_z": grid[2],
    }
    with tvm_ffi.use_torch_stream():
        entry(*[values[name] for name in program.args])
    torch.cuda.synchronize()
    assert _counters_are_zero(counters)
    return out, lse


# --------------------------------------------------------------------------- plan-time route (no GPU)


def _plan_on_cpu(monkeypatch, mask, sm_count=148, arch="sm_100a", **overrides):
    monkeypatch.setattr(cake_vsa, "_arch_for_device", lambda _device: arch)
    monkeypatch.setattr(cake_vsa, "_device_index", lambda _device: 0)
    monkeypatch.setattr(cake_vsa, "_sm_count", lambda _index: sm_count)
    heads, mb, nb = mask.shape
    kwargs = {
        "indptr": None,
        "indices": None,
        "block_mask": mask,
        "kv_block_lens": None,
        "q2k_indices": None,
        "q2k_num": None,
        "M": mb * BLOCK,
        "N": nb * BLOCK,
        "R": BLOCK,
        "C": BLOCK,
        "num_qo_heads": heads,
        "num_kv_heads": heads,
        "head_dim": HEAD_DIM,
        "q_data_type": torch.bfloat16,
        "sm_scale": None,
        "device": torch.device("cpu"),
    }
    kwargs.update(overrides)
    return cake_vsa.plan_cake_vsa(**kwargs)


def test_shared_multi_wave_selection_plans_the_balanced_program(monkeypatch):
    counts = _ragged_counts(64, 64)
    mask = _shared_mask(8, 64, 64, counts, "cpu")
    plan = _plan_on_cpu(monkeypatch, mask)
    metadata = plan["blk64_balanced"]
    assert metadata is not None
    expected_ptr = torch.tensor([0, *counts], dtype=torch.int32).cumsum(0)
    torch.testing.assert_close(metadata["indptr"], expected_ptr.to(torch.int32))
    torch.testing.assert_close(
        metadata["indices"], mask[0].nonzero()[:, 1].to(torch.int32)
    )
    assert metadata["queue_counters"].dtype == torch.uint32
    assert tuple(metadata["queue_counters"].shape) == (4,)
    assert _counters_are_zero(metadata["queue_counters"])
    assert tuple(metadata["order_debug"].shape) == (64 * 8,)
    # the per-head plan is kept as it was beside the balanced metadata
    assert plan["blk64_profile"] in ("blk64_persistent", "blk64_persistent_ws_m64n256")


_WS = "blk64_persistent_ws_m64n256"


@pytest.mark.parametrize("arch,sm_count", [("sm_100a", 148), ("sm_103a", 152)])
@pytest.mark.parametrize(
    "mb,heads,counts,routed",
    [
        # fewer than seven blocks per query block on average: the per-head programs
        (16, 8, [5] * 16, False),
        (16, 8, [6] * 16, False),
        (16, 8, [7] * 16, True),
        (16, 8, [4, 10] * 8, True),  # seven on average
        (16, 8, [4, 9] * 8, False),  # six and a half
        (64, 8, [4] * 64, False),  # four blocks per query block: the ordinary kernel's
        # the per-head plan is the ordinary program: any grid of 8 tiles and up
        (1, 8, [16], True),  # 8 tiles
        (8, 1, [7] * 8, True),  # 8 tiles
        (7, 1, [16] * 7, False),  # 7 tiles
        (4, 1, [7] * 4, False),  # 4 tiles
        (14, 8, [16] * 14, True),  # 112 tiles, under one wave
        (16, 8, [8] * 16, True),  # 128 tiles: the S = 1024 standings rows
        (32, 8, [16] * 32, True),  # 256 tiles: the S = 2048 standings rows
        (16, 8, [65] * 16, True),  # not whole groups
        (16, 8, [8, 40] * 8, True),  # whole groups, 24 blocks on average
    ],
)
def test_route_needs_seven_blocks_and_eight_tiles(
    monkeypatch, arch, sm_count, mb, heads, counts, routed
):
    assert cake_vsa._BLK64_BALANCED_MIN_AVERAGE_BLOCKS == 7
    assert cake_vsa._BLK64_BALANCED_MIN_TILES == 8
    mask = _shared_mask(heads, mb, 256, counts, "cpu")
    plan = _plan_on_cpu(monkeypatch, mask, sm_count=sm_count, arch=arch)
    assert plan["blk64_profile"] != _WS
    assert (plan["blk64_balanced"] is not None) == routed


@pytest.mark.parametrize(
    "mb,heads,nb,counts,sm_count,sm100a,sm103a",
    [
        # the per-head plan is the M64N256 WS program (rows of whole groups of four,
        # more than 24 blocks on average). sm_100a: from 0.8 tiles per SM, rows in
        # one group-count bucket up to 64 blocks, rows in several buckets from 1.05
        # waves (156 tiles on 148 SMs, 160 on 152). sm_103a: every such selection.
        (14, 8, 64, [40] * 14, 148, False, True),  # 112 tiles: 0.76 waves
        (1, 8, 64, [40], 148, False, True),  # 8 tiles
        (15, 8, 64, [40] * 15, 148, True, True),  # 120 tiles: 0.81 waves on 148 SMs
        (15, 8, 64, [40] * 15, 152, False, True),  # 0.79 waves on 152 SMs
        (16, 8, 128, [64] * 16, 148, True, True),  # the cap (one wave of 128 tiles)
        (16, 8, 128, [68] * 16, 148, False, True),  # longer rows in one bucket
        (64, 8, 256, [96] * 64, 148, False, True),  # one bucket, 512 tiles
        (32, 8, 256, [252, 256] * 16, 148, False, True),  # both in the last bucket
        (32, 8, 256, [248, 256] * 16, 148, True, True),  # buckets 62 and 63
        (16, 8, 128, [32, 64] * 8, 148, False, True),  # several buckets in one wave
        (38, 4, 256, [64, 128] * 19, 148, False, True),  # 152 tiles: 1.03 waves
        (39, 4, 256, [64, 128] * 19 + [64], 148, True, True),  # 156 tiles: 1.05 waves
        (39, 4, 256, [64, 128] * 19 + [64], 152, False, True),  # 1.03 waves on 152 SMs
        (7, 1, 64, [40] * 7, 148, False, False),  # 7 tiles
    ],
)
def test_route_keeps_the_ws_program_where_it_is_faster(
    monkeypatch, mb, heads, nb, counts, sm_count, sm100a, sm103a
):
    assert cake_vsa._BLK64_BALANCED_WS_MIN_WAVES == 0.8
    assert cake_vsa._BLK64_BALANCED_WS_MAX_ROW_BLOCKS == 64
    assert cake_vsa._BLK64_BALANCED_WS_RAGGED_MIN_WAVES == 1.05
    mask = _shared_mask(heads, mb, nb, counts, "cpu")
    for arch, routed in (("sm_100a", sm100a), ("sm_103a", sm103a)):
        plan = _plan_on_cpu(monkeypatch, mask, sm_count=sm_count, arch=arch)
        assert plan["blk64_profile"] == _WS
        assert (plan["blk64_balanced"] is not None) == routed, arch


def test_balanced_beats_per_head_boundaries():
    def beats(
        profile,
        fewest,
        most,
        selected=None,
        mb=1,
        tiles=128,
        sm_count=148,
        arch="sm_100a",
    ):
        return cake_vsa._blk64_balanced_beats_per_head(
            arch=arch,
            per_head_profile=profile,
            selected=most if selected is None else selected,
            mb=mb,
            fewest=fewest,
            most=most,
            tiles=tiles,
            sm_count=sm_count,
        )

    assert beats("blk64_persistent", 256, 256)
    assert beats("blk64_persistent", 7, 7) and not beats("blk64_persistent", 6, 6)
    assert not beats("blk64_persistent", 4, 12, selected=6)
    assert beats("blk64_persistent", 8, 8, tiles=8)
    assert not beats("blk64_persistent", 8, 8, tiles=7)
    assert beats(_WS, 64, 64) and not beats(_WS, 68, 68)
    assert not beats(_WS, 64, 64, tiles=118) and beats(_WS, 64, 64, tiles=119)
    assert not beats(_WS, 252, 256, tiles=512)  # both in the last bucket
    assert beats(_WS, 248, 256, tiles=512)
    assert not beats(_WS, 32, 64, tiles=155) and beats(_WS, 32, 64, tiles=156)
    assert not beats(_WS, 32, 64, tiles=159, sm_count=152)
    assert beats(_WS, 32, 64, tiles=160, sm_count=152)
    for most, tiles in ((68, 128), (256, 512), (64, 8), (64, 155)):
        assert beats(_WS, 32, most, tiles=tiles, sm_count=152, arch="sm_103a")
    assert not beats(_WS, 64, 64, tiles=7, sm_count=152, arch="sm_103a")
    buckets = [cake_vsa._blk64_group_bucket(c) for c in (1, 4, 5, 248, 252, 256)]
    assert buckets == [1, 1, 2, 62, 63, 63]


def test_per_head_selections_and_partial_blocks_keep_the_per_head_programs(
    monkeypatch,
):
    mask = _shared_mask(8, 64, 64, [16] * 64, "cpu")
    other = mask.clone()
    other[3, 0] = ~other[3, 0]
    other[3, 0, 0] = True
    assert _plan_on_cpu(monkeypatch, other)["blk64_balanced"] is None
    kv_block_lens = torch.full((64,), BLOCK, dtype=torch.int32)
    kv_block_lens[5] = 63
    plan = _plan_on_cpu(monkeypatch, mask, kv_block_lens=kv_block_lens)
    assert plan["blk64_balanced"] is None
    q2k_indices = torch.arange(16, dtype=torch.int32).expand(8, 64, 16).contiguous()
    plan = _plan_on_cpu(monkeypatch, mask, block_mask=None, q2k_indices=q2k_indices)
    assert plan["blk64_balanced"] is None


def test_zero_block_rows_are_refused(monkeypatch):
    counts = [16] * 64
    counts[7] = 0
    with pytest.raises(ValueError, match="at least one block"):
        _plan_on_cpu(monkeypatch, _shared_mask(8, 64, 64, counts, "cpu"))


def test_launch_grid_is_one_cta_per_sm():
    assert cake_vsa.cake_vsa_balanced_launch_grid(512, 148) == (148, 1, 1)
    assert cake_vsa.cake_vsa_balanced_launch_grid(100, 148) == (100, 1, 1)
    assert cake_vsa.cake_vsa_balanced_launch_grid(0, 148) == (1, 1, 1)


# --------------------------------------------------------------------------- the program on the GPU


@requires_gpu
@pytest.mark.parametrize(
    "seq,counts_of",
    [
        (4096, lambda mb, nb: [16] * mb),  # uniform: the static stream
        (4096, lambda mb, nb: [48] * mb),
        (4096, _ragged_counts),  # ragged: the ticket queue and partial groups
        (8192, lambda mb, nb: _ragged_counts(mb, nb, 7)),
    ],
)
@pytest.mark.parametrize("return_lse", [False, True])
def test_balanced_route_matches_the_reference_and_the_forced_program(
    seq, counts_of, return_lse
):
    device = torch.device("cuda")
    heads, mb = 8, seq // BLOCK
    mask = _shared_mask(heads, mb, mb, counts_of(mb, mb), device, seed=seq)
    q, k, v = _inputs(seq, seq, heads, device, seed=1)
    workspace = torch.empty((1,), dtype=torch.uint8, device=device)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")
    _plan(wrapper, mask, heads, seq, seq)
    plan = wrapper._cake_vsa_plan
    assert plan["blk64_balanced"] is not None

    result = wrapper.run(q, k, v, return_lse=return_lse)
    output, lse = result if return_lse else (result, None)
    torch.cuda.synchronize()
    assert _counters_are_zero(plan["blk64_balanced"]["queue_counters"])

    forced_output, forced_lse = _forced(plan, q, k, v, return_lse)
    assert torch.equal(output, forced_output)
    if return_lse:
        assert torch.equal(lse, forced_lse)

    for _ in range(3):
        again = wrapper.run(q, k, v, return_lse=return_lse)
        again_output = again[0] if return_lse else again
        assert torch.equal(again_output, output)
    assert _counters_are_zero(plan["blk64_balanced"]["queue_counters"])

    if seq <= 4096:
        reference, reference_lse = _dense_reference(q, k, v, mask)
        torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
        if return_lse:
            torch.testing.assert_close(lse, reference_lse, atol=1e-2, rtol=1e-2)


@requires_gpu
@pytest.mark.parametrize(
    "mb,nb,counts",
    [
        (1, 256, [8]),  # 8 tiles, the smallest balanced grid
        (2, 256, [7, 7]),
        (2, 256, [16, 16]),
        (3, 64, [7] * 3),
        (19, 256, [5 + (3 * row) % 5 for row in range(19)]),  # ragged, about one wave
    ],
)
def test_sub_wave_balanced_route_matches_the_reference_and_the_forced_program(
    mb, nb, counts
):
    device = torch.device("cuda")
    heads = 8
    seq_q, seq_kv = mb * BLOCK, nb * BLOCK
    mask = _shared_mask(heads, mb, nb, counts, device, seed=mb)
    q, k, v = _inputs(seq_q, seq_kv, heads, device, seed=7)
    wrapper = BlockSparseAttentionWrapper(
        torch.empty((1,), dtype=torch.uint8, device=device), backend="cake"
    )
    _plan(wrapper, mask, heads, seq_q, seq_kv)
    plan = wrapper._cake_vsa_plan
    assert plan["blk64_balanced"] is not None

    output, lse = wrapper.run(q, k, v, return_lse=True)
    torch.cuda.synchronize()
    assert _counters_are_zero(plan["blk64_balanced"]["queue_counters"])

    forced_output, forced_lse = _forced(plan, q, k, v, True)
    assert torch.equal(output, forced_output)
    assert torch.equal(lse, forced_lse)

    reference, reference_lse = _dense_reference(q, k, v, mask)
    torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, reference_lse, atol=1e-2, rtol=1e-2)


@requires_gpu
def test_graph_replays_equal_eager_and_reset_the_counter():
    device = torch.device("cuda")
    heads, seq = 8, 4096
    mb = seq // BLOCK
    mask = _shared_mask(heads, mb, mb, _ragged_counts(mb, mb), device, seed=3)
    q, k, v = _inputs(seq, seq, heads, device, seed=2)
    wrapper = BlockSparseAttentionWrapper(
        torch.empty((1,), dtype=torch.uint8, device=device), backend="cake"
    )
    _plan(wrapper, mask, heads, seq, seq)
    counters = wrapper._cake_vsa_plan["blk64_balanced"]["queue_counters"]
    out = torch.empty_like(q)
    lse = torch.empty((seq, heads), dtype=torch.float32, device=device)
    eager_out, eager_lse = wrapper.run(q, k, v, return_lse=True)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        wrapper.run(q, k, v, out=out, lse=lse, return_lse=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, k, v, out=out, lse=lse, return_lse=True)
    for _ in range(5):
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _counters_are_zero(counters)
        assert torch.equal(out, eager_out)
        assert torch.equal(lse, eager_lse)
    # an eager launch after the replays starts from the reset counter too
    assert torch.equal(wrapper.run(q, k, v), eager_out)


@requires_gpu
def test_two_wrappers_on_two_streams_keep_their_own_counters():
    device = torch.device("cuda")
    heads, seq = 8, 4096
    mb = seq // BLOCK
    masks = (
        _shared_mask(heads, mb, mb, _ragged_counts(mb, mb, 13), device, seed=4),
        _shared_mask(heads, mb, mb, _ragged_counts(mb, mb, 29), device, seed=5),
    )
    q, k, v = _inputs(seq, seq, heads, device, seed=3)
    wrappers, expected = [], []
    for mask in masks:
        wrapper = BlockSparseAttentionWrapper(
            torch.empty((1,), dtype=torch.uint8, device=device), backend="cake"
        )
        _plan(wrapper, mask, heads, seq, seq)
        assert wrapper._cake_vsa_plan["blk64_balanced"] is not None
        wrappers.append(wrapper)
        expected.append(wrapper.run(q, k, v))
    torch.cuda.synchronize()
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    outputs = [[], []]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(8):
        for index, (stream, wrapper) in enumerate(zip(streams, wrappers, strict=True)):
            with torch.cuda.stream(stream):
                outputs[index].append(wrapper.run(q, k, v))
    torch.cuda.synchronize()
    for index, wrapper in enumerate(wrappers):
        counters = wrapper._cake_vsa_plan["blk64_balanced"]["queue_counters"]
        assert _counters_are_zero(counters)
        for output in outputs[index]:
            assert torch.equal(output, expected[index])


@requires_gpu
def test_unrouted_plans_keep_the_per_head_programs():
    device = torch.device("cuda")
    heads, seq = 8, 896
    mb = seq // BLOCK
    # six blocks per query block: fewer than seven on both targets
    mask = _shared_mask(heads, mb, mb, [6] * mb, device, seed=6)
    q, k, v = _inputs(seq, seq, heads, device, seed=4)
    wrapper = BlockSparseAttentionWrapper(
        torch.empty((1,), dtype=torch.uint8, device=device), backend="cake"
    )
    _plan(wrapper, mask, heads, seq, seq)
    assert wrapper._cake_vsa_plan["blk64_balanced"] is None
    output = wrapper.run(q, k, v)
    reference, _ = _dense_reference(q, k, v, mask)
    torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
    # a selection that differs between heads
    wrapper = BlockSparseAttentionWrapper(
        torch.empty((1,), dtype=torch.uint8, device=device), backend="cake"
    )
    big = 4096
    per_head = _shared_mask(heads, big // BLOCK, big // BLOCK, [16] * 64, device)
    per_head[2, 3] = False
    per_head[2, 3, :8] = True
    q, k, v = _inputs(big, big, heads, device, seed=5)
    _plan(wrapper, per_head, heads, big, big)
    assert wrapper._cake_vsa_plan["blk64_balanced"] is None
    reference, _ = _dense_reference(q, k, v, per_head)
    torch.testing.assert_close(wrapper.run(q, k, v), reference, atol=1e-2, rtol=1e-2)
