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

import random

import pytest
import torch

from flashinfer.decode import prepare_balanced_batch_decode_with_kv_cache
from flashinfer.experimental.balanced_gqa_decode import cake_backend, cake_bounds
from flashinfer.experimental.balanced_gqa_decode.cake_backend import (
    GROUP_RATIO,
    HEAD_DIM,
    PAGE_SIZE,
    SUPPORTED_COMPUTE_CAPABILITIES,
    balanced_gqa_decode_workspace_size,
    validate_balanced_gqa_decode_inputs,
    workspace_layout,
)
from flashinfer.experimental.balanced_gqa_decode.cake_bounds import (
    MAX_BALANCE_FACTOR,
    MAX_REQUESTS,
    NUM_BUCKETS,
    PAIR_TOKENS,
    max_items_bound,
    mtp_max_items_bound,
    mtp_n_rows,
    uses_packed_mtp,
    workspace_bounds,
)
from tests.test_helpers.cake_balanced_gqa_plan import (
    chunk_pairs_for,
    length_bucket,
    mtp_chunk_pairs,
    mtp_device_plan,
    plan_balanced_work,
    simulate_greedy_makespan,
)

ATOL = RTOL = 1e-2

# The AgentX-derived ragged KV pattern of flashinfer-ai/flashinfer#4832.
AGENTX_LENGTHS = [
    8193,
    57345,
    73729,
    81921,
    98305,
    106497,
    114689,
    131073,
    139265,
    147457,
    163841,
    180225,
    196609,
    212993,
    229377,
    237569,
]


def _ceil_div(a, b):
    return -(-a // b)


# ---------------------------------------------------------------------------
# Host plan mirror (CPU)
# ---------------------------------------------------------------------------


def _check_plan_invariants(seq_lens, work_plan):
    max_items, max_tiles = workspace_bounds(work_plan.num_ctas)
    assert work_plan.num_split_items <= max_items
    assert work_plan.num_split_tiles <= max_tiles
    assert work_plan.bucket_ends[-1] == work_plan.num_items
    assert list(work_plan.bucket_ends) == sorted(work_plan.bucket_ends)
    assert work_plan.num_items <= max_items_bound(
        len(seq_lens), work_plan.q_len, work_plan.num_kv_heads, work_plan.num_ctas
    )
    for ticket, item in enumerate(work_plan.items):
        assert item.ticket == ticket
        lo = work_plan.bucket_ends[item.bucket - 1] if item.bucket else 0
        assert lo <= ticket < work_plan.bucket_ends[item.bucket]
        assert item.block_begin < item.block_end
        assert item.block_end == _ceil_div(item.seqlen_row, cake_bounds.BLOCK_N) or (
            item.block_end - item.block_begin == 2 * work_plan.chunk_pairs
        )
        assert (item.slot >= 0) == item.is_split
        assert (item.counter >= 0) == item.is_split
    # Every (row, kv head) tile is covered exactly once over its causal length.
    coverage = {}
    for item in work_plan.items:
        key = (item.batch, item.q_row, item.kv_head)
        coverage.setdefault(key, []).append((item.block_begin, item.block_end))
    for b, seq_len in enumerate(seq_lens):
        for j in range(work_plan.q_len):
            for h in range(work_plan.num_kv_heads):
                spans = sorted(coverage[(b, j, h)])
                expected_end = _ceil_div(
                    seq_len - (work_plan.q_len - 1 - j), cake_bounds.BLOCK_N
                )
                assert spans[0][0] == 0 and spans[-1][1] == expected_end
                for (_, e0), (s1, _) in zip(spans, spans[1:], strict=False):
                    assert e0 == s1
    # Split slots and counters are unique.
    slots = [item.slot for item in work_plan.items if item.is_split]
    assert len(slots) == len(set(slots)) == work_plan.num_split_items
    counters = {item.counter for item in work_plan.items if item.is_split}
    assert len(counters) == work_plan.num_split_tiles


@pytest.mark.parametrize(
    "seq_lens,q_len,num_kv_heads,num_ctas",
    [
        (AGENTX_LENGTHS, 1, 1, 160),
        (AGENTX_LENGTHS * 12, 1, 1, 160),
        (AGENTX_LENGTHS, 7, 1, 148),
        ([8193], 7, 1, 148),
        ([8193, 8194, 8320], 7, 1, 148),
        ([257, 300, 7, 8, 256, 255, 511, 513], 7, 2, 16),
        ([1] * 5, 1, 8, 148),
        ([16384] * 8, 1, 8, 148),
        ([4096] * 128, 1, 8, 160),
    ],
)
def test_plan_invariants(seq_lens, q_len, num_kv_heads, num_ctas):
    work_plan = plan_balanced_work(
        seq_lens, q_len=q_len, num_kv_heads=num_kv_heads, num_ctas=num_ctas
    )
    _check_plan_invariants(seq_lens, work_plan)


@pytest.mark.parametrize("balance_factor", [None, 1, 2, MAX_BALANCE_FACTOR])
def test_random_plans(balance_factor):
    rng = random.Random(578)
    for _ in range(40):
        batch = rng.randint(1, 96)
        q_len = rng.choice([1, 1, 2, 7])
        num_kv_heads = rng.choice([1, 2, 8])
        num_ctas = rng.choice([132, 148, 160])
        seq_lens = [
            rng.randint(q_len, rng.choice([600, 9000, 70000])) for _ in range(batch)
        ]
        work_plan = plan_balanced_work(
            seq_lens,
            q_len=q_len,
            num_kv_heads=num_kv_heads,
            num_ctas=num_ctas,
            balance_factor=balance_factor,
        )
        _check_plan_invariants(seq_lens, work_plan)


def test_uniform_large_batch_never_splits():
    work_plan = plan_balanced_work([4096] * 512, num_kv_heads=8, num_ctas=160)
    assert work_plan.num_split_items == 0
    assert work_plan.num_items == 512 * 8


def test_small_uniform_batch_splits_evenly():
    work_plan = plan_balanced_work([65536] * 4, num_kv_heads=8, num_ctas=160)
    assert work_plan.balance_factor == 0  # even-split rule
    assert all(item.is_split for item in work_plan.items)
    assert work_plan.num_items <= 160


def test_multi_wave_uniform_batch_follows_launch_cost_model():
    # 512 tiles of 128 pairs on 148 CTAs: 3 full waves plus a 68-CTA last
    # wave, below SATURATING_CTAS -> two chunks per tile (6 full waves plus a
    # saturated partial wave).  1024 tiles leave 136 CTAs streaming in the
    # last wave and keep whole tiles.
    split = plan_balanced_work([32768] * 64, num_kv_heads=8, num_ctas=148)
    assert split.balance_factor == 0 and split.chunk_pairs == 64
    assert all(item.n_chunks == 2 for item in split.items)
    whole = plan_balanced_work([32768] * 128, num_kv_heads=8, num_ctas=148)
    assert whole.num_split_items == 0 and whole.num_items == 1024


def test_agentx_ragged_batch_splits_and_orders_by_length():
    work_plan = plan_balanced_work(AGENTX_LENGTHS, num_kv_heads=1, num_ctas=160)
    assert work_plan.num_split_items > 0
    lengths = [item.block_end - item.block_begin for item in work_plan.items]
    full = 2 * work_plan.chunk_pairs
    first_short = next((i for i, n in enumerate(lengths) if n < full), len(lengths))
    assert all(n == full for n in lengths[:first_short])
    buckets = [item.bucket for item in work_plan.items]
    assert buckets == sorted(buckets)
    ideal = sum(_ceil_div(s, PAIR_TOKENS) for s in AGENTX_LENGTHS) / 160
    assert simulate_greedy_makespan(work_plan) <= 1.1 * ideal + work_plan.chunk_pairs


def test_length_bucket_edges():
    L = 64
    assert length_bucket(64, L) == 0
    assert length_bucket(63, L) == 1
    assert length_bucket(32, L) == 1
    assert length_bucket(31, L) == 2
    assert length_bucket(16, L) == 2
    assert length_bucket(15, L) == NUM_BUCKETS - 1
    assert length_bucket(1, L) == NUM_BUCKETS - 1
    with pytest.raises(ValueError):
        length_bucket(65, L)


def test_chunk_pairs_respects_minimum():
    chunk_pairs, k = chunk_pairs_for([1] * 3, q_len=1, num_kv_heads=1, num_ctas=160)
    assert chunk_pairs >= cake_bounds.DEFAULT_PAIRS_MIN
    assert k >= 0


def test_plan_rejects_bad_inputs():
    with pytest.raises(ValueError):
        plan_balanced_work([], num_kv_heads=1, num_ctas=160)
    with pytest.raises(ValueError):
        plan_balanced_work([0, 5], num_kv_heads=1, num_ctas=160)
    with pytest.raises(ValueError):
        plan_balanced_work([3], q_len=7, num_kv_heads=1, num_ctas=160)
    with pytest.raises(ValueError):
        plan_balanced_work([5] * (MAX_REQUESTS + 1), num_kv_heads=1, num_ctas=160)


# ---------------------------------------------------------------------------
# Host layer (CPU)
# ---------------------------------------------------------------------------


def test_workspace_layout_is_shape_independent():
    layout = workspace_layout(160)
    assert layout["page_table"] == (layout["page_table"][0], 0)
    total = balanced_gqa_decode_workspace_size(num_sms=160)
    assert total == layout["total"]
    assert total == balanced_gqa_decode_workspace_size(
        num_sms=160, batch=1024, max_pages=64
    )
    padded = balanced_gqa_decode_workspace_size(num_sms=160, batch=4, max_pages=9)
    assert padded == total + cake_backend._align(4 * 16 * 4)
    for name in ("partial_o", "partial_stats", "tile_counters", "queue_counters"):
        assert layout[name][0] % cake_backend.WORKSPACE_ALIGN == 0


def _host_tensors(batch=2, num_kv_heads=1, q_len=1, num_pages=64, max_pages=16):
    num_q_heads = GROUP_RATIO * num_kv_heads
    query = torch.zeros(batch * q_len, num_q_heads, HEAD_DIM, dtype=torch.bfloat16)
    k_cache = torch.zeros(
        num_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM, dtype=torch.bfloat16
    )
    v_cache = torch.zeros_like(k_cache)
    block_tables = torch.zeros(batch, max_pages, dtype=torch.int32)
    seq_lens = torch.ones(batch, dtype=torch.int32)
    return query, k_cache, v_cache, block_tables, seq_lens


def test_validate_inputs_accepts_contract_shapes():
    query, k_cache, v_cache, block_tables, seq_lens = _host_tensors(q_len=7)
    assert validate_balanced_gqa_decode_inputs(
        query, k_cache, v_cache, block_tables, seq_lens, q_len_per_req=7
    ) == (2, 8, 1, 64)


@pytest.mark.parametrize(
    "index,mutate,message",
    [
        (0, lambda t: t.to(torch.float16), "bfloat16"),
        (0, lambda t: t[:, :4].contiguous(), "query heads per KV head"),
        (0, lambda t: t[:1], "batch \\* q_len_per_req"),
        (1, lambda t: t[:, :, :8].contiguous(), "k_cache"),
        (3, lambda t: t.to(torch.int64), "block_tables"),
        (4, lambda t: t[:1], "seq_lens"),
    ],
)
def test_validate_inputs_rejects(index, mutate, message):
    tensors = list(_host_tensors())
    tensors[index] = mutate(tensors[index])
    with pytest.raises(ValueError, match=message):
        validate_balanced_gqa_decode_inputs(*tensors, q_len_per_req=1)


def test_kv_cache_tuple_required():
    query, k_cache, v_cache, block_tables, seq_lens = _host_tensors()
    stacked = torch.stack([k_cache, v_cache], dim=1)
    with pytest.raises(ValueError, match="separate contiguous"):
        cake_backend._split_kv_cache(stacked, "HND")
    with pytest.raises(ValueError, match="HND"):
        cake_backend._split_kv_cache((k_cache, v_cache), "NHD")
    assert cake_backend._split_kv_cache((k_cache, v_cache), "HND") == (k_cache, v_cache)


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------


def _gpu_arch():
    if not torch.cuda.is_available():
        return None
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(0))


def _require_program(q_len=1):
    arch = _gpu_arch()
    if arch is None:
        pytest.skip("balanced GQA decode requires an SM100/SM103 GPU")
    if not cake_backend.generated_program_available(torch.device("cuda", 0), q_len):
        pytest.skip(
            f"no generated balanced GQA decode program ({cake_backend.program_kind(q_len)}) "
            f"registered for {arch}"
        )


def reference_decode(
    query, k_cache, v_cache, block_tables, seq_lens, *, q_len, sm_scale
):
    """FP32 paged GQA reference: ``[batch * q_len, num_q_heads, 128]`` bf16."""
    batch = int(block_tables.shape[0])
    num_q_heads = int(query.shape[1])
    num_kv_heads = int(k_cache.shape[1])
    out = torch.empty_like(query, dtype=torch.float32)
    lens = seq_lens.tolist()
    for b in range(batch):
        seq_len = int(lens[b])
        pages = block_tables[b, : _ceil_div(seq_len, PAGE_SIZE)].long()
        k = (
            k_cache[pages]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, HEAD_DIM)[:, :seq_len]
        )
        v = (
            v_cache[pages]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, HEAD_DIM)[:, :seq_len]
        )
        k = k.float().repeat_interleave(GROUP_RATIO, dim=0)  # [Hq, S, D]
        v = v.float().repeat_interleave(GROUP_RATIO, dim=0)
        q = (
            query[b * q_len : (b + 1) * q_len].float().permute(1, 0, 2)
        )  # [Hq, q_len, D]
        scores = torch.einsum("hqd,hkd->hqk", q, k) * sm_scale
        positions = torch.arange(seq_len, device=query.device)
        row_limit = seq_len - q_len + 1 + torch.arange(q_len, device=query.device)
        mask = positions[None, :] < row_limit[:, None]
        scores = scores.masked_fill(~mask[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[b * q_len : (b + 1) * q_len] = torch.einsum("hqk,hkd->qhd", probs, v)
    assert num_q_heads == GROUP_RATIO * num_kv_heads
    return out.to(torch.bfloat16)


def make_inputs(
    seq_lens, num_kv_heads, *, q_len=1, seed=0, pad_pages=True, device="cuda"
):
    """Deterministic BF16 paged inputs with a shuffled page table."""
    gen = torch.Generator(device=device).manual_seed(seed)
    batch = len(seq_lens)
    num_q_heads = GROUP_RATIO * num_kv_heads
    max_pages = _ceil_div(max(seq_lens), PAGE_SIZE)
    if pad_pages:
        max_pages = (max_pages + 7) // 8 * 8
    num_pages = batch * max_pages
    query = torch.randn(
        (batch * q_len, num_q_heads, HEAD_DIM), generator=gen, device=device
    ).to(torch.bfloat16)
    k_cache = torch.randn(
        (num_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM), generator=gen, device=device
    ).to(torch.bfloat16)
    v_cache = torch.randn(
        (num_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM), generator=gen, device=device
    ).to(torch.bfloat16)
    block_tables = (
        torch.randperm(num_pages, generator=gen, device=device)
        .to(torch.int32)
        .view(batch, max_pages)
    )
    seq_lens_dev = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return query, k_cache, v_cache, block_tables, seq_lens_dev


def _workspace(device):
    return torch.empty(
        balanced_gqa_decode_workspace_size(device, batch=MAX_REQUESTS, max_pages=9),
        dtype=torch.uint8,
        device=device,
    )


def _expected_device_plan(seq_lens, *, q_len, num_kv_heads, num_ctas):
    """``(chunk_pairs, tickets)`` the selected program publishes for this batch."""
    if uses_packed_mtp(q_len):
        return mtp_device_plan(
            seq_lens, q_len=q_len, num_kv_heads=num_kv_heads, num_ctas=num_ctas
        )
    mirror = plan_balanced_work(
        seq_lens, q_len=q_len, num_kv_heads=num_kv_heads, num_ctas=num_ctas
    )
    return mirror.chunk_pairs, mirror.num_items


def _run_and_check(seq_lens, num_kv_heads, *, q_len=1, seed=0, pad_pages=True):
    query, k_cache, v_cache, block_tables, seq_lens_dev = make_inputs(
        seq_lens, num_kv_heads, q_len=q_len, seed=seed, pad_pages=pad_pages
    )
    sm_scale = HEAD_DIM**-0.5
    workspace = _workspace(query.device)
    out = torch.full_like(query, float("nan"))
    runner = prepare_balanced_batch_decode_with_kv_cache(
        query,
        (k_cache, v_cache),
        block_tables,
        seq_lens_dev,
        workspace,
        sm_scale=sm_scale,
        q_len_per_req=q_len,
        out=out,
    )
    assert runner.block_tables_padded == (block_tables.shape[1] % 8 != 0)
    result = runner()
    assert result is out
    torch.cuda.synchronize()
    expected = reference_decode(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens_dev,
        q_len=q_len,
        sm_scale=sm_scale,
    )
    torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)
    # The device planner published the same plan as the host mirror, and the
    # self-resetting counters are back at zero.
    assert runner.device_plan() == _expected_device_plan(
        seq_lens, q_len=q_len, num_kv_heads=num_kv_heads, num_ctas=runner.num_ctas
    )
    counters = runner.main_kwargs["queue_counters"].tolist()
    assert counters[0] == 0 and counters[1] == 0
    assert int(runner.main_kwargs["tile_counters"].sum().item()) == 0
    return runner, (query, k_cache, v_cache, block_tables, seq_lens_dev, out)


@pytest.mark.parametrize(
    "seq_lens,num_kv_heads,q_len",
    [
        ([130, 8000, 519, 4096, 1024, 2048, 3000], 8, 1),  # ragged, GQA 64/8
        ([512] * 4, 8, 1),  # short uniform, one item per tile
        ([65536] * 4, 8, 1),  # long uniform small batch: even split
        ([4096] * 200, 1, 1),  # 1.35 waves of 16-pair tiles: launch-cost model splits
        ([8192] * 40, 8, 1),  # 2.2 waves, sparse last wave: launch-cost model splits
        (AGENTX_LENGTHS, 1, 1),  # #4832 AgentX pattern
        ([300, 257, 5000, 777], 2, 7),  # MTP verify rows, ragged (packed 64-row tile)
        ([8193, 8194, 8320], 1, 7),  # MTP rows straddling a 128-block edge
        ([1, 17, 129, 4097], 1, 1),  # single-token and one-past-page lengths
        ([4099, 1027], 2, 4),  # packed 32-row tile (q_len <= 4)
        (
            [3018, 3609, 238, 1491, 2736, 1785, 2709, 1441],
            8,
            3,
        ),  # consecutive chunk items per CTA
        ([60008], 1, 8),  # longest packed tile, one 60k request
        ([130, 8000, 519, 4096], 8, 2),  # q_len 2 stays on the row-tile program
        ([64, 3000], 1, 9),  # q_len 9 stays on the row-tile program
    ],
)
def test_balanced_decode_matches_reference(seq_lens, num_kv_heads, q_len):
    _require_program(q_len)
    _run_and_check(seq_lens, num_kv_heads, q_len=q_len, seed=len(seq_lens))


def test_unpadded_block_table_is_copied():
    _require_program()
    _run_and_check([1000, 3000, 2000], 1, seed=3, pad_pages=False)


def test_graph_replay_follows_device_lengths():
    """Capture once, replay with new lengths written into ``seq_lens``."""
    _require_program()
    seq_lens = [130, 8000, 519, 4096, 1024, 2048, 3000, 70000]
    runner, (query, k_cache, v_cache, block_tables, seq_lens_dev, out) = _run_and_check(
        seq_lens, 1, seed=11
    )
    sm_scale = HEAD_DIM**-0.5
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner()
    torch.cuda.synchronize()
    rng = random.Random(4832)
    for _ in range(3):
        new_lens = [rng.randint(1, s) for s in seq_lens]
        seq_lens_dev.copy_(torch.tensor(new_lens, dtype=torch.int32))
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        expected = reference_decode(
            query,
            k_cache,
            v_cache,
            block_tables,
            seq_lens_dev,
            q_len=1,
            sm_scale=sm_scale,
        )
        torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)
        mirror = plan_balanced_work(new_lens, num_kv_heads=1, num_ctas=runner.num_ctas)
        assert runner.device_plan() == (mirror.chunk_pairs, mirror.num_items)


def test_graph_replay_follows_device_lengths_mtp():
    """q_len_per_req = 7 on the packed-row program: capture once, replay new lengths."""
    _require_program(7)
    seq_lens = [3018, 3609, 238, 1491, 2736, 1785, 2709, 1441]
    runner, (query, k_cache, v_cache, block_tables, seq_lens_dev, out) = _run_and_check(
        seq_lens, 8, q_len=7, seed=606
    )
    assert runner.module_name.endswith("sm_100a") or runner.module_name.endswith(
        "sm_103a"
    )
    assert "mtp64" in runner.module_name
    sm_scale = HEAD_DIM**-0.5
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner()
    torch.cuda.synchronize()
    rng = random.Random(606)
    for _ in range(3):
        new_lens = [rng.randint(7, s) for s in seq_lens]
        seq_lens_dev.copy_(torch.tensor(new_lens, dtype=torch.int32))
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        expected = reference_decode(
            query,
            k_cache,
            v_cache,
            block_tables,
            seq_lens_dev,
            q_len=7,
            sm_scale=sm_scale,
        )
        torch.testing.assert_close(out, expected, atol=ATOL, rtol=RTOL)
        assert runner.device_plan() == mtp_device_plan(
            new_lens, q_len=7, num_kv_heads=8, num_ctas=runner.num_ctas
        )


def test_packed_mtp_bounds_and_chunk_length():
    """Host-side facts of the packed-row program (no GPU)."""
    assert [uses_packed_mtp(q) for q in (1, 2, 3, 8, 9)] == [
        False,
        False,
        True,
        True,
        False,
    ]
    assert (
        mtp_n_rows(3) == 32
        and mtp_n_rows(4) == 32
        and mtp_n_rows(5) == 64
        and mtp_n_rows(8) == 64
    )
    assert mtp_max_items_bound(16, 7, 1, 148) == 16 + 16 * 148 + 8 * 148 * 14
    agentx = AGENTX_LENGTHS[:16]
    # 8516 pairs on 148 CTAs: 155 coarse tickets would leave a 2x tail; three waves of L=20 do not.
    assert mtp_chunk_pairs(agentx, q_len=7, num_kv_heads=1, num_ctas=148) == 20
    assert mtp_chunk_pairs([4096] * 8, q_len=7, num_kv_heads=8, num_ctas=148) == 8
    assert (
        mtp_chunk_pairs([1024] * 64, q_len=7, num_kv_heads=1, num_ctas=148) == 2
    )  # two-chunk tiles fold in place: one wave of 2-pair chunks, no tickets
    assert (
        mtp_chunk_pairs([60007], q_len=7, num_kv_heads=1, num_ctas=148) == 4
    )  # coarser chunks halve the fourteen merge tickets' fold
    chunk, tickets = mtp_device_plan(
        [300, 257, 5000, 777], q_len=7, num_kv_heads=2, num_ctas=148
    )
    # 28 chunk items; the 5000-token tile (10 chunks) takes 14 tickets per kv
    # head, the 777-token tile (2 chunks) is folded in place.
    assert (chunk, tickets) == (2, 28 + 2 * 14)


def test_launch_makes_no_allocation():
    _require_program()
    runner, _ = _run_and_check([2048, 300, 9000], 2, seed=5)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] - before["allocation.all.allocated"] == 0


def test_prepare_rejects_small_workspace_and_wrong_device():
    _require_program()
    query, k_cache, v_cache, block_tables, seq_lens_dev = make_inputs([600, 900], 1)
    small = torch.empty(1024, dtype=torch.uint8, device=query.device)
    with pytest.raises(ValueError, match="workspace_buffer needs"):
        prepare_balanced_batch_decode_with_kv_cache(
            query, (k_cache, v_cache), block_tables, seq_lens_dev, small
        )
    workspace = _workspace(query.device)
    with pytest.raises(ValueError, match="one CUDA device"):
        prepare_balanced_batch_decode_with_kv_cache(
            query, (k_cache, v_cache), block_tables.cpu(), seq_lens_dev, workspace
        )
    with pytest.raises(ValueError, match="backend"):
        prepare_balanced_batch_decode_with_kv_cache(
            query,
            (k_cache, v_cache),
            block_tables,
            seq_lens_dev,
            workspace,
            backend="xqa",
        )
