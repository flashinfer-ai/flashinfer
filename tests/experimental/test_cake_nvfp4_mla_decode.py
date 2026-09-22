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

import hashlib
import math
from collections import Counter
from dataclasses import replace

import pytest
import torch

from flashinfer.experimental.nvfp4_mla_decode import cake_backend
from flashinfer.experimental.nvfp4_mla_decode.cake_backend import (
    BALANCED_MIN_KV,
    CLUSTER_PAIR,
    DSV4_Q_LEN,
    FLAG_DIRECT_OUT,
    FLAG_SEED_SINK,
    HEAD_DIM,
    ITEM_FIELDS,
    MAX_SPLITS,
    PAGE_SIZE,
    ROW_BYTES,
    ROWS_PER_TILE,
    SF_ROW_BYTES,
    SF_VEC,
    SUPPORTED_COMPUTE_CAPABILITIES,
    V_HALVES,
    build_work_plan,
    check_pairs,
    kv_tiles,
    max_nvfp4_mla_decode_workspace_size,
    nvfp4_mla_decode_workspace_size,
    quantize_nvfp4,
    work_table_rows,
    workspace_layout,
)
from flashinfer.mla import prepare_nvfp4_batch_decode_with_kv_cache_mla

ATOL = RTOL = 0.1
REL_L2_MAX = 0.05
LSE_ATOL = LSE_RTOL = 0.05
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ---------------------------------------------------------------------------
# Torch reference: FP32 attention over exactly dequantized NVFP4 operands
# ---------------------------------------------------------------------------


def dequantize_nvfp4(packed, scale):
    """Exact FP32 decode of packed E2M1 codes with UE4M3 block-16 scales."""
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.stack((lo, hi), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    magnitude = table[(codes & 0x7).long()]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    values = magnitude * sign
    blocks = values.reshape(*values.shape[:-1], values.shape[-1] // SF_VEC, SF_VEC)
    scale = scale.view(torch.float8_e4m3fn).float()
    return (blocks * scale.unsqueeze(-1)).reshape(*values.shape)


def _gather_dequant(cache, scale, block_table_row, kv_len):
    """Dequantize the first ``kv_len`` tokens of one request: [kv_len, 512]."""
    pages = block_table_row[: kv_tiles(kv_len)].long()
    rows = dequantize_nvfp4(cache[pages], scale[pages])  # [pages, page, D]
    return rows.reshape(-1, rows.shape[-1])[:kv_len]


def reference(
    query,
    query_scale,
    kv_cache,
    kv_scale,
    block_tables,
    kv_lens,
    q_len,
    sm_scale,
    sinks,
):
    """Returns ``(O bf16 [total_q, H, 512], LSE f32 [total_q, H])``."""
    batch = len(kv_lens)
    q_all = dequantize_nvfp4(query, query_scale)  # [total_q, H, D]
    num_heads = q_all.shape[1]
    O = torch.empty(
        (batch * q_len, num_heads, HEAD_DIM), dtype=torch.float32, device=q_all.device
    )
    LSE = torch.empty(
        (batch * q_len, num_heads), dtype=torch.float32, device=q_all.device
    )
    for b in range(batch):
        kv_len = kv_lens[b]
        k = _gather_dequant(kv_cache, kv_scale, block_tables[b], kv_len)  # [kv_len, D]
        v = k
        q = q_all[b * q_len : (b + 1) * q_len]  # [q_len, H, D]
        logits = torch.einsum("rhd,nd->hrn", q, k) * sm_scale  # [H, q_len, kv_len]
        positions = torch.arange(kv_len, device=q.device)
        row_limit = kv_len - q_len + torch.arange(q_len, device=q.device) + 1
        mask = positions[None, :] < row_limit[:, None]  # [q_len, kv_len]
        logits = logits.masked_fill(~mask[None], float("-inf"))
        row_max = logits.amax(dim=-1)  # [H, q_len]
        if sinks is not None:
            sink = sinks[:, None].expand(num_heads, q_len)
            row_max = torch.maximum(row_max, sink)
        probs = torch.exp(logits - row_max[..., None])
        denom = probs.sum(dim=-1)
        if sinks is not None:
            denom = denom + torch.exp(sink - row_max)
        out = torch.einsum("hrn,nd->hrd", probs, v) / denom[..., None]
        lse = row_max + torch.log(denom)
        O[b * q_len : (b + 1) * q_len] = out.permute(1, 0, 2)
        LSE[b * q_len : (b + 1) * q_len] = lse.permute(1, 0)
    return O.to(torch.bfloat16), LSE


def make_inputs(kv_lens, num_heads, *, q_len, enable_sink, device, seed=0):
    """Deterministic NVFP4 paged decode inputs with a peaked softmax."""
    gen = torch.Generator(device=device).manual_seed(seed)
    batch = len(kv_lens)
    pages_per_seq = [kv_tiles(kv) for kv in kv_lens]
    total_pages = sum(pages_per_seq)
    max_pages = max(pages_per_seq)
    k_full = torch.randn(
        (total_pages, PAGE_SIZE, HEAD_DIM), generator=gen, device=device
    )
    permutation = torch.randperm(total_pages, generator=gen, device=device).to(
        torch.int32
    )
    block_tables = torch.zeros((batch, max_pages), dtype=torch.int32, device=device)
    offset = 0
    for b, count in enumerate(pages_per_seq):
        block_tables[b, :count] = permutation[offset : offset + count]
        offset += count
    q = torch.randn((batch * q_len, num_heads, HEAD_DIM), generator=gen, device=device)
    for b in range(batch):
        for i in range(q_len):
            visible = kv_lens[b] - q_len + i + 1
            target = int(
                torch.randint(0, visible, (1,), generator=gen, device=device).item()
            )
            page = int(block_tables[b, target // PAGE_SIZE].item())
            q[b * q_len + i] += 0.2 * k_full[page, target % PAGE_SIZE]
    query, query_scale = quantize_nvfp4(q)
    kv_cache, kv_scale = quantize_nvfp4(k_full)
    sinks = None
    if enable_sink:
        sinks = torch.randn((num_heads,), generator=gen, device=device) * 0.5 + 1.0
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    return dict(
        query=query,
        query_scale=query_scale,
        kv_cache=kv_cache,
        kv_scale=kv_scale,
        block_tables=block_tables,
        seq_lens=seq_lens,
        sinks=sinks,
        q_len=q_len,
        sm_scale=HEAD_DIM**-0.5,
    )


def check_outputs(out, lse, ref_out, ref_lse):
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out, ref_out, atol=ATOL, rtol=RTOL)
    rel = (out.float() - ref_out.float()).norm(dim=-1) / ref_out.float().norm(
        dim=-1
    ).clamp_min(1e-6)
    assert float(rel.max()) <= REL_L2_MAX, float(rel.max())
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(lse, ref_lse, atol=LSE_ATOL, rtol=LSE_RTOL)


def _plan_digest(plan):
    return hashlib.sha256(
        repr([tuple(int(v) for v in it) for it in plan.items]).encode()
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Host-only tests
# ---------------------------------------------------------------------------


def test_public_entry_point_is_experimental():
    assert prepare_nvfp4_batch_decode_with_kv_cache_mla.is_experimental is True


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        prepare_nvfp4_batch_decode_with_kv_cache_mla(
            None, None, None, None, None, None, None, sm_scale=1.0, backend="unknown"
        )


def test_rejects_host_tensors_and_bad_shapes():
    inputs = make_inputs([64, 100], 2, q_len=6, enable_sink=False, device="cpu")
    workspace = torch.empty(1 << 20, dtype=torch.uint8)
    with pytest.raises(ValueError, match="CUDA"):
        prepare_nvfp4_batch_decode_with_kv_cache_mla(
            inputs["query"],
            inputs["query_scale"],
            inputs["kv_cache"],
            inputs["kv_scale"],
            inputs["block_tables"],
            inputs["seq_lens"],
            workspace,
            sm_scale=inputs["sm_scale"],
        )
    args = (
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        inputs["seq_lens"],
    )
    with pytest.raises(ValueError, match="query must be"):
        cake_backend.validate_nvfp4_mla_decode_inputs(
            inputs["query"][..., :128], inputs["query_scale"], *args
        )
    with pytest.raises(ValueError, match="batch \\* q_len"):
        cake_backend.validate_nvfp4_mla_decode_inputs(
            inputs["query"][:-1], inputs["query_scale"][:-1], *args
        )
    with pytest.raises(ValueError, match="int32"):
        cake_backend.validate_nvfp4_mla_decode_inputs(
            inputs["query"],
            inputs["query_scale"],
            inputs["kv_cache"],
            inputs["kv_scale"],
            inputs["block_tables"].to(torch.int64),
            inputs["seq_lens"],
        )


@pytest.mark.parametrize("q_len", [1, 6])
def test_q_len_is_derived_from_query_rows(q_len):
    inputs = make_inputs(
        [64, 100, 130], 4, q_len=q_len, enable_sink=False, device="cpu"
    )
    batch, num_heads, num_pages, got = cake_backend.validate_nvfp4_mla_decode_inputs(
        inputs["query"],
        inputs["query_scale"],
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        inputs["seq_lens"],
    )
    assert (batch, num_heads, num_pages, got) == (
        3,
        4,
        inputs["kv_cache"].shape[0],
        q_len,
    )


# Expected values computed once from the source host plan module for the same
# shapes (cluster of two CTAs per unit; balanced schedule over num_sms // 2
# units; uniform schedule pairs consecutive items). Columns: schedule, kv_lens,
# q_len, num_heads, num_sms, tiles_per_split, enable_sink -> (num_items,
# num_units, grid_x, max_splits, tiles_per_split, item digest).
PLAN_EXPECTATIONS = [
    (
        "balanced",
        [256, 300],
        6,
        64,
        148,
        None,
        True,
        (16, 3, 6, 2, None, "e1be1df6a1068135"),
    ),
    (
        "balanced",
        [8192] * 4,
        6,
        64,
        148,
        None,
        False,
        (168, 74, 148, 7, None, "bce7f7a800a4cb91"),
    ),
    # contract row partial_pages_h8
    (
        "balanced",
        [1000, 8192, 5000],
        6,
        8,
        148,
        None,
        True,
        (56, 27, 54, 16, None, "3592220b611a0c8b"),
    ),
    (
        "balanced",
        [131072, 64],
        1,
        64,
        148,
        None,
        False,
        (76, 37, 74, 37, None, "a98e3fabd5f652ba"),
    ),
    # contract row bs32_q6_kv8k (auto -> balanced at the 8K threshold)
    (
        "auto",
        [8192] * 32,
        6,
        64,
        148,
        None,
        True,
        (336, 74, 148, 2, None, "d05e6b3d8872ab59"),
    ),
    (
        "uniform",
        [64, 4096],
        6,
        64,
        148,
        32,
        False,
        (18, 9, 18, 2, 32, "9bcf8e5442f91a15"),
    ),
    # contract row smoke, forced to two splits (tiles_per_split = ceil(5 / 2))
    ("auto", [256, 300], 6, 64, 148, 3, True, (24, 12, 24, 2, 3, "2dd204fdabaf9bf2")),
    (
        "uniform",
        [8192] * 32,
        6,
        64,
        148,
        None,
        False,
        (192, 96, 192, 1, 128, "2549a4d3ff721985"),
    ),
    (
        "uniform",
        [777, 65],
        1,
        64,
        148,
        None,
        True,
        (4, 2, 4, 1, 13, "7e8e47b4623f60b9"),
    ),
    # contract row no_sink_q1 (auto -> uniform)
    (
        "auto",
        [64, 65, 4096, 777],
        1,
        64,
        148,
        None,
        False,
        (8, 4, 8, 1, 64, "95fe7983a25d234f"),
    ),
    (
        "uniform",
        [16384, 12000],
        6,
        64,
        148,
        None,
        False,
        (12, 6, 12, 1, 256, "73cfcd1186a5db80"),
    ),
]


@pytest.mark.parametrize(
    "schedule,kv_lens,q_len,num_heads,num_sms,tiles_per_split,enable_sink,expected",
    PLAN_EXPECTATIONS,
)
def test_plan_matches_source_host_plan(
    schedule, kv_lens, q_len, num_heads, num_sms, tiles_per_split, enable_sink, expected
):
    plan = build_work_plan(
        kv_lens,
        num_heads=num_heads,
        num_sms=num_sms,
        q_len=q_len,
        enable_sink=enable_sink,
        schedule=schedule,
        tiles_per_split=tiles_per_split,
    )
    got = (
        plan.num_items,
        plan.num_units,
        plan.grid[0],
        plan.max_splits,
        plan.tiles_per_split,
        _plan_digest(plan),
    )
    assert got == expected
    assert plan.grid == (CLUSTER_PAIR * plan.num_units, 1, 1)
    if schedule == "auto":
        assert plan.schedule == (
            "balanced" if max(kv_lens) >= BALANCED_MIN_KV else "uniform"
        )
    if plan.schedule == "uniform":
        assert plan.unit_first == tuple(range(0, plan.num_items + 1, CLUSTER_PAIR))


def test_balanced_unit_first_matches_source():
    plan = build_work_plan(
        [256, 300],
        num_heads=64,
        num_sms=148,
        q_len=6,
        enable_sink=True,
        schedule="balanced",
    )
    assert plan.unit_first == (0, 6, 12, 16)
    plan = build_work_plan(
        [8192] * 32,
        num_heads=64,
        num_sms=148,
        q_len=6,
        enable_sink=True,
        schedule="balanced",
    )
    assert plan.unit_first[:4] == (0, 4, 8, 12) and plan.unit_first[-3:] == (
        328,
        332,
        336,
    )


def test_pair_invariant_holds_and_is_enforced():
    plan = build_work_plan(
        [1000, 8192, 5000], num_heads=8, num_sms=148, q_len=6, enable_sink=True
    )
    check_pairs(plan)
    items = list(plan.items)
    items[0], items[1] = items[1], items[0]
    with pytest.raises(ValueError, match="v_half pair"):
        check_pairs(replace(plan, items=tuple(items)))
    with pytest.raises(ValueError, match="even count"):
        check_pairs(replace(plan, unit_first=(0, 1, plan.num_items)))


def test_uniform_plan_partitions_every_request():
    plan = build_work_plan(
        [256, 300],
        num_heads=64,
        num_sms=148,
        q_len=6,
        enable_sink=True,
        schedule="uniform",
    )
    m_tiles = math.ceil(6 * 64 / ROWS_PER_TILE)
    for b, kv in enumerate([256, 300]):
        ranges = sorted({(it[3], it[4]) for it in plan.items if it[0] == b})
        assert ranges[0][0] == 0 and ranges[-1][1] == kv_tiles(kv)
        for (_, e0), (s1, _) in zip(ranges, ranges[1:], strict=False):
            assert e0 == s1
        assert (
            sum(1 for it in plan.items if it[0] == b)
            == len(ranges) * m_tiles * V_HALVES
        )
    assert all(
        (it[7] & FLAG_SEED_SINK) == (FLAG_SEED_SINK if it[5] == 0 else 0)
        for it in plan.items
    )
    assert all(((it[7] & FLAG_DIRECT_OUT) != 0) == (it[6] == 1) for it in plan.items)


@pytest.mark.parametrize(
    "kv_len,want_splits", [(8192, 1), (16384, 2), (32768, 2), (65536, 2), (131072, 2)]
)
def test_uniform_split_policy_matches_measured_winners(kv_len, want_splits):
    plan = build_work_plan(
        [kv_len] * 32, num_heads=64, num_sms=148, q_len=6, schedule="uniform"
    )
    assert plan.max_splits == want_splits
    assert all(it[4] - it[3] >= 8 for it in plan.items)


@pytest.mark.parametrize(
    "kv_lens,num_heads,num_sms",
    [
        ([131072] * 32, 64, 148),
        ([8192] * 32, 64, 148),
        ([256, 300], 64, 148),
        ([1000, 8192, 5000], 8, 148),
        ([131072], 64, 148),
    ],
)
def test_balanced_plan_covers_every_page_once(kv_lens, num_heads, num_sms):
    plan = build_work_plan(
        kv_lens,
        num_heads=num_heads,
        num_sms=num_sms,
        q_len=6,
        enable_sink=True,
        schedule="balanced",
    )
    assert plan.max_splits <= MAX_SPLITS and plan.num_units <= num_sms // CLUSTER_PAIR
    assert plan.unit_first[0] == 0 and plan.unit_first[-1] == plan.num_items
    covered = Counter()
    for it in plan.items:
        assert 0 <= it[3] < it[4]
        for t in range(it[3], it[4]):
            covered[(it[0], it[1], it[2], t)] += 1
    m_tiles = math.ceil(6 * num_heads / ROWS_PER_TILE)
    want = sum(kv_tiles(kv) for kv in kv_lens) * m_tiles * V_HALVES
    assert len(covered) == want and max(covered.values()) == 1
    pages = [
        sum(it[4] - it[3] for it in plan.items[a:b])
        for a, b in zip(plan.unit_first, plan.unit_first[1:], strict=False)
    ]
    assert max(pages) - min(pages) <= 2


def test_auto_schedule_threshold():
    assert (
        build_work_plan(
            [BALANCED_MIN_KV - 1] * 32, num_heads=64, num_sms=148, q_len=6
        ).schedule
        == "uniform"
    )
    assert (
        build_work_plan(
            [BALANCED_MIN_KV] * 32, num_heads=64, num_sms=148, q_len=6
        ).schedule
        == "balanced"
    )
    with pytest.raises(ValueError, match="kv_len >= q_len"):
        build_work_plan([5], num_heads=64, num_sms=148, q_len=6)


def test_work_table_clears_direct_out_when_any_request_splits():
    single = build_work_plan(
        [64, 65, 4096, 777],
        num_heads=64,
        num_sms=148,
        q_len=1,
        schedule="uniform",
        tiles_per_split=4096,
    )
    assert single.max_splits == 1
    assert bool((work_table_rows(single)[:, 7] & FLAG_DIRECT_OUT).all())
    mixed = build_work_plan(
        [64, 4096],
        num_heads=64,
        num_sms=148,
        q_len=6,
        schedule="uniform",
        tiles_per_split=32,
    )
    table = work_table_rows(mixed)
    assert mixed.max_splits == 2 and table.shape == (mixed.num_items, ITEM_FIELDS)
    assert not bool((table[:, 7] & FLAG_DIRECT_OUT).any())
    assert torch.equal(table[:, 0].unique(), torch.tensor([0, 1], dtype=torch.int32))


def test_workspace_sizing():
    kv_lens = [8192] * 32
    plan = build_work_plan(
        kv_lens, num_heads=64, num_sms=148, q_len=6, enable_sink=True
    )
    layout = workspace_layout(plan, batch=32, num_heads=64, q_len=6)
    total_q = 32 * 6
    assert plan.max_splits == 2
    assert layout["partial_o"][1] == total_q * 64 * plan.max_splits * HEAD_DIM * 4
    assert layout["partial_lse"][1] == total_q * 64 * plan.max_splits * 4
    assert layout["work_table"][1] == plan.num_items * ITEM_FIELDS * 4
    assert layout["unit_first"][1] == (plan.num_units + 1) * 4
    offsets = [
        layout[k][0]
        for k in (
            "partial_o",
            "partial_lse",
            "work_table",
            "unit_first",
            "row_splits",
            "q_indptr",
            "sinks",
        )
    ]
    assert offsets == sorted(offsets) and all(o % 256 == 0 for o in offsets)
    assert (
        nvfp4_mla_decode_workspace_size(
            kv_lens, 64, num_sms=148, q_len=6, enable_sink=True
        )
        == layout["total"]
    )
    assert layout["total"] <= max_nvfp4_mla_decode_workspace_size(32, 64, q_len=6)
    single = build_work_plan([64, 65, 4096, 777], num_heads=64, num_sms=148, q_len=1)
    single_layout = workspace_layout(single, batch=4, num_heads=64, q_len=1)
    assert (
        single.max_splits == 1
        and single_layout["partial_o"][1] == 4
        and single_layout["partial_lse"][1] == 4
    )
    assert max_nvfp4_mla_decode_workspace_size(
        32, 64, max_splits=1
    ) < max_nvfp4_mla_decode_workspace_size(32, 64, max_splits=2)


def test_quantize_nvfp4_roundtrip_and_saturation():
    x = torch.randn((4, 3, HEAD_DIM))
    packed, scale = quantize_nvfp4(x)
    assert packed.shape == (4, 3, ROW_BYTES) and packed.dtype == torch.uint8
    assert scale.shape == (4, 3, SF_ROW_BYTES) and scale.dtype == torch.uint8
    decoded = dequantize_nvfp4(packed, scale)
    # The coarsest E2M1 bin (4 -> 6) has a half-step of one normalized unit,
    # so every element is within one decoded block scale of its source.
    block_scale = (
        scale.view(torch.float8_e4m3fn)
        .float()
        .unsqueeze(-1)
        .expand(-1, -1, -1, SF_VEC)
        .reshape_as(x)
    )
    assert ((decoded - x).abs() <= block_scale + 1e-6).all()
    values = torch.tensor([0.0, 1.0, 2688.0, 4096.0, -4096.0], dtype=torch.float32)
    packed, scale = quantize_nvfp4(values[:, None].expand(-1, 32).contiguous())
    scales = scale.view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(
        scales,
        torch.tensor([2.0**-9, 0.171875, 448.0, 448.0, 448.0])[:, None].expand(-1, 2),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        packed[:, 0],
        torch.tensor([0x00, 0x77, 0x77, 0x77, 0xFF], dtype=torch.uint8),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("q_len,enable_sink", [(6, False), (6, True), (1, True)])
def test_reference_matches_masked_softmax(q_len, enable_sink):
    """The ported reference equals a plain softmax with the sink as an extra logit."""
    kv_lens = [70, 130]
    inputs = make_inputs(
        kv_lens, 4, q_len=q_len, enable_sink=enable_sink, device="cpu", seed=3
    )
    out, lse = reference(
        inputs["query"],
        inputs["query_scale"],
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        kv_lens,
        q_len,
        inputs["sm_scale"],
        inputs["sinks"],
    )
    q_all = dequantize_nvfp4(inputs["query"], inputs["query_scale"])
    for b, kv_len in enumerate(kv_lens):
        k = _gather_dequant(
            inputs["kv_cache"], inputs["kv_scale"], inputs["block_tables"][b], kv_len
        )
        for i in range(q_len):
            visible = kv_len - q_len + i + 1
            logits = (
                q_all[b * q_len + i] @ k[:visible].T * inputs["sm_scale"]
            )  # [H, visible]
            if enable_sink:
                logits = torch.cat([logits, inputs["sinks"][:, None]], dim=-1)
            probs = torch.softmax(logits, dim=-1)
            expected = probs[:, :visible] @ k[:visible]
            torch.testing.assert_close(
                out[b * q_len + i].float(), expected, atol=2e-2, rtol=2e-2
            )
            torch.testing.assert_close(
                lse[b * q_len + i],
                torch.logsumexp(logits, dim=-1),
                atol=1e-5,
                rtol=1e-5,
            )


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


def _gpu_skip_reason():
    if not torch.cuda.is_available():
        return "CUDA required"
    if torch.cuda.get_device_capability() not in SUPPORTED_COMPUTE_CAPABILITIES:
        return "SM100 or SM103 required"
    if not cake_backend.generated_program_available(torch.device("cuda")):
        return "generated program not registered in this checkout (flashinfer-ai/flashinfer#5403)"
    return None


def test_rejects_unsupported_compute_capability():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() in SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip("device is supported; nothing to reject")
    inputs = make_inputs([64], 2, q_len=1, enable_sink=False, device="cuda")
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="compute capability"):
        prepare_nvfp4_batch_decode_with_kv_cache_mla(
            inputs["query"],
            inputs["query_scale"],
            inputs["kv_cache"],
            inputs["kv_scale"],
            inputs["block_tables"],
            inputs["seq_lens"],
            workspace,
            sm_scale=inputs["sm_scale"],
        )


@pytest.mark.parametrize(
    "label,kv_lens,q_len,num_heads,enable_sink,tiles_per_split",
    [
        ("smoke_forced_two_splits", [256, 300], 6, 64, True, 3),
        ("partial_pages_h8", [1000, 8192, 5000], 6, 8, True, None),
        ("no_sink_q1", [64, 65, 4096, 777], 1, 64, False, None),
        ("bs32_q6_kv8k", [8192] * 32, 6, 64, True, None),
        ("bs2_kv16k_mixed", [16384, 12000], 6, 64, False, None),
    ],
)
def test_nvfp4_mla_decode(
    label, kv_lens, q_len, num_heads, enable_sink, tiles_per_split
):
    reason = _gpu_skip_reason()
    if reason:
        pytest.skip(reason)
    inputs = make_inputs(
        kv_lens, num_heads, q_len=q_len, enable_sink=enable_sink, device="cuda"
    )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if tiles_per_split is None:
        nbytes = nvfp4_mla_decode_workspace_size(
            kv_lens, num_heads, num_sms=num_sms, q_len=q_len, enable_sink=enable_sink
        )
    else:
        nbytes = max_nvfp4_mla_decode_workspace_size(
            len(kv_lens), num_heads, q_len=q_len
        )
    workspace = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    out = torch.empty(
        (len(kv_lens) * q_len, num_heads, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    lse = torch.full(
        (len(kv_lens) * q_len, num_heads),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    decode = cake_backend.prepare_nvfp4_batch_decode_with_kv_cache_mla(
        inputs["query"],
        inputs["query_scale"],
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        inputs["seq_lens"],
        workspace,
        sm_scale=inputs["sm_scale"],
        sinks=inputs["sinks"],
        out=out,
        lse=lse,
        return_lse=True,
        tiles_per_split=tiles_per_split,
    )
    assert decode.main_kwargs["grid"] == (CLUSTER_PAIR * decode.plan.num_units, 1, 1)
    assert decode.main_kwargs["q_len"] == q_len
    if tiles_per_split is not None:
        assert decode.plan.max_splits >= 2 and decode.reduce_kwargs is not None
    result = decode()
    assert result[0] is out and result[1] is lse
    torch.cuda.synchronize()
    ref_out, ref_lse = reference(
        inputs["query"],
        inputs["query_scale"],
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        kv_lens,
        q_len,
        inputs["sm_scale"],
        inputs["sinks"],
    )
    check_outputs(out, lse, ref_out, ref_lse)
    snapshot_out, snapshot_lse = out.clone(), lse.clone()
    out.zero_()
    lse.fill_(float("nan"))
    decode()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, snapshot_out, atol=0, rtol=0)
    torch.testing.assert_close(lse, snapshot_lse, atol=0, rtol=0)


def test_public_api_returns_out_without_lse():
    reason = _gpu_skip_reason()
    if reason:
        pytest.skip(reason)
    kv_lens = [512, 300]
    inputs = make_inputs(kv_lens, 8, q_len=DSV4_Q_LEN, enable_sink=False, device="cuda")
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    workspace = torch.empty(
        nvfp4_mla_decode_workspace_size(kv_lens, 8, num_sms=num_sms, q_len=DSV4_Q_LEN),
        dtype=torch.uint8,
        device="cuda",
    )
    decode = prepare_nvfp4_batch_decode_with_kv_cache_mla(
        inputs["query"],
        inputs["query_scale"],
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        inputs["seq_lens"],
        workspace,
        sm_scale=inputs["sm_scale"],
        seq_lens_cpu=inputs["seq_lens"].cpu(),
    )
    out = decode()
    assert isinstance(out, torch.Tensor) and out.shape == (
        len(kv_lens) * DSV4_Q_LEN,
        8,
        HEAD_DIM,
    )
    torch.cuda.synchronize()
    ref_out, ref_lse = reference(
        inputs["query"],
        inputs["query_scale"],
        inputs["kv_cache"],
        inputs["kv_scale"],
        inputs["block_tables"],
        kv_lens,
        DSV4_Q_LEN,
        inputs["sm_scale"],
        None,
    )
    check_outputs(out, decode.lse, ref_out, ref_lse)
