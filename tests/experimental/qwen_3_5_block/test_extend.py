# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The EXTEND form against the fp32 oracle: a packed prefill step (empty prefixes) and a step behind cached prefixes, pads in the
middle of the batch, the bounded token count, both cache layouts, the 32 / 2 shard, the small-capacity plans' live-row rule and a
cached step under ``set_sync_debug_mode("error")``."""

import pytest
import torch

from tests.test_helpers.qwen_3_5_block_cases import PagedCase

pytestmark = [pytest.mark.arch_rubin, pytest.mark.usefixtures("rubin")]


def _geometry():
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry

    return Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)


def _padded(case, capacity):
    """The step's token tensors at the plan's CAPACITY: the live rows first, finite pad rows (slot -1) after them."""
    t, dm, rope = case.t, case.geometry.d_model, case.geometry.rope_dim
    h = torch.zeros(capacity, dm, dtype=torch.bfloat16, device="cuda")
    cos = torch.ones(capacity, rope, dtype=torch.bfloat16, device="cuda")
    sin = torch.zeros(capacity, rope, dtype=torch.bfloat16, device="cuda")
    slots = torch.full((capacity,), -1, dtype=torch.int32, device="cuda")
    h[:t], cos[:t], sin[:t], slots[:t] = (
        case.hidden_states,
        case.rope_cos,
        case.rope_sin,
        case.slot_mapping,
    )
    return h, cos, sin, slots


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_extend_prefill_step_with_empty_prefixes(layout):
    """Every sequence starts at position 0 (the prefill case of the extend form): lens 128 / 200 / pad / 60 at a 512-token capacity."""
    from flashinfer.qwen_3_5_block import qwen_3_5_block_bf16_extend

    g = _geometry()
    lens = (128, 200, 0, 60)
    case = PagedCase(g, lens, lens, page_size=16, layout=layout, seed=5)
    cap = 512
    h, cos, sin, slots = _padded(case, cap)
    out = qwen_3_5_block_bf16_extend(
        h,
        cos,
        sin,
        case.weights,
        case.w_o,
        geometry=g,
        query_start_loc=case.query_start_loc,
        kv_cache=case.kv_cache,
        slot_mapping=slots,
        block_table=case.block_table,
        seq_lens=case.seq_lens,
        kv_layout=layout,
        num_actual_tokens=case.t,
    )
    torch.cuda.synchronize()
    assert out.shape == (cap, g.d_model)
    case.check_out(out[: case.t], label=f"extend prefill {layout}")
    case.check_written_rows(label=f"extend prefill {layout}")
    assert (case.kv_cache[0] == 0).all(), "the NULL page must never be written"


def test_extend_behind_cached_prefixes_and_the_live_row_count():
    """New tokens behind prefixes (300 / 0 / 77 / 1000-token histories) at a 256-token capacity: the oracle over the full history;
    the live rows handed at EXACTLY num_actual_tokens rows give bitwise the capacity launch's rows."""
    from flashinfer.qwen_3_5_block import prepare_qwen_3_5_block_bf16

    g = _geometry()
    case = PagedCase(
        g,
        (300, 0, 77, 1000),
        (64, 0, 32, 100),
        page_size=16,
        layout="NHD",
        seed=6,
        table_width=80,
    )
    cap = 256
    plan = prepare_qwen_3_5_block_bf16(
        "extend",
        geometry=g,
        device="cuda",
        weights=case.weights,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=16,
        batch_size=4,
        token_capacity=cap,
        block_table=case.block_table,
    )
    assert plan.declaration["stages"] and plan.table_row_stride == 80
    h, cos, sin, slots = _padded(case, cap)
    args = dict(case.extend_args(), slot_mapping=slots)
    out = plan.run(
        h, cos, sin, case.weights, case.w_o, num_actual_tokens=case.t, **args
    )
    torch.cuda.synchronize()
    case.check_out(out[: case.t], label="extend behind prefixes")
    case.check_written_rows(label="extend behind prefixes")
    live = plan.run(
        case.hidden_states,
        case.rope_cos,
        case.rope_sin,
        case.weights,
        case.w_o,
        num_actual_tokens=case.t,
        **case.extend_args(),
    )
    torch.cuda.synchronize()
    assert live.shape == (case.t, g.d_model) and torch.equal(live, out[: case.t]), (
        "the live-row launch differs from the capacity launch"
    )
    # a step past the capacity is a typed decline naming the bound
    with pytest.raises(ValueError, match="num_actual_tokens"):
        plan.run(h, cos, sin, case.weights, case.w_o, num_actual_tokens=cap + 1, **args)
    with pytest.raises(ValueError, match="hidden_states"):
        plan.run(
            h[: case.t - 1],
            cos,
            sin,
            case.weights,
            case.w_o,
            num_actual_tokens=case.t,
            **args,
        )


def test_extend_two_chunks_match_the_single_oracle():
    """A 400-token sequence as two chunks on ONE plan (capacity 256): chunk 2 reads chunk 1's pool rows."""
    from flashinfer.qwen_3_5_block import prepare_qwen_3_5_block_bf16

    g = _geometry()
    full = PagedCase(
        g, (400,), (400,), page_size=16, layout="NHD", seed=7
    )  # the oracle over the whole sequence; its pools are the prefix source
    ref = full.refs[0]
    cap = 256
    kv = full.kv_cache.clone().zero_()
    plan = prepare_qwen_3_5_block_bf16(
        "extend",
        geometry=g,
        device="cuda",
        weights=full.weights,
        w_o=full.w_o,
        kv_cache=kv,
        page_size=16,
        batch_size=1,
        token_capacity=cap,
        block_table=full.block_table,
    )
    outs = []
    for lo, hi in ((0, 250), (250, 400)):
        n = hi - lo
        h = torch.zeros(cap, g.d_model, dtype=torch.bfloat16, device="cuda")
        cos = torch.ones(cap, g.rope_dim, dtype=torch.bfloat16, device="cuda")
        sin = torch.zeros_like(cos)
        slots = torch.full((cap,), -1, dtype=torch.int32, device="cuda")
        h[:n] = full.h_full[0][0, lo:hi]
        pos = torch.arange(lo, hi, device="cuda")
        from tests.test_helpers.qwen_3_5_block_reference import build_rope_tables

        c, s = build_rope_tables(
            n,
            g.rope_dim,
            base=full.ref_geom.rope_base,
            batch=1,
            device="cuda",
            dtype=torch.bfloat16,
            positions=pos[None],
        )
        cos[:n], sin[:n] = c[0], s[0]
        pages = torch.tensor(full.pages[0], device="cuda")
        slots[:n] = (pages[pos // 16] * 16 + pos % 16).to(torch.int32)
        out = plan.run(
            h,
            cos,
            sin,
            full.weights,
            full.w_o,
            query_start_loc=torch.tensor([0, n], dtype=torch.int32, device="cuda"),
            kv_cache=kv,
            slot_mapping=slots,
            block_table=full.block_table,
            seq_lens=torch.tensor([hi], dtype=torch.int32, device="cuda"),
            num_actual_tokens=n,
        )
        torch.cuda.synchronize()
        outs.append(out[:n].clone())
    from tests.test_helpers.qwen_3_5_block_reference import cosine

    got = torch.cat(outs)
    c = cosine(got, ref.out[0])
    print(f"[extend two chunks] vs the single-sequence oracle: cos {c:.6f}")
    assert torch.isfinite(got.float()).all() and c > 0.999


def test_extend_32_2_shard_page_32_hnd():
    """The 397B-class shard (32 / 2 at d_model 4096) behind prefixes with a pad in the middle, page 32, HND, at a 128-token capacity."""
    from flashinfer.qwen_3_5_block import (
        Qwen35BlockGeometry,
        qwen_3_5_block_bf16_extend,
    )

    g = Qwen35BlockGeometry(d_model=4096, num_qo_heads=32, num_kv_heads=2)
    case = PagedCase(
        g,
        (300, 0, 77),
        (40, 0, 24),
        page_size=32,
        layout="HND",
        seed=26,
        table_width=40,
    )
    cap = 128
    h, cos, sin, slots = _padded(case, cap)
    out = qwen_3_5_block_bf16_extend(
        h,
        cos,
        sin,
        case.weights,
        case.w_o,
        geometry=g,
        query_start_loc=case.query_start_loc,
        kv_cache=case.kv_cache,
        slot_mapping=slots,
        block_table=case.block_table,
        seq_lens=case.seq_lens,
        kv_layout="HND",
        num_actual_tokens=case.t,
    )
    torch.cuda.synchronize()
    case.check_out(out[: case.t], label="extend 32/2 p32 HND")
    case.check_written_rows(label="extend 32/2 p32 HND")


def _plan_extend(case, g, cap, **kw):
    from flashinfer.qwen_3_5_block import prepare_qwen_3_5_block_bf16

    return prepare_qwen_3_5_block_bf16(
        "extend",
        geometry=g,
        device="cuda",
        weights=case.weights,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=16,
        batch_size=case.b,
        token_capacity=cap,
        block_table=case.block_table,
        **kw,
    )


def test_small_capacity_plans_launch_token_rows_in_multiples_of_the_out_projection_vector():
    """At a token capacity of at most 128 the out projection runs the small-capacity (swap-AB) GEMM tile, whose token rows are
    the kernel's epilogue-vector axis: every live count must be a multiple of ``plan.live_rows_multiple`` (the widest vector that
    divides the capacity: 32 on 64 tokens, 16 on 48, in bf16); the plan says so in its declaration and refuses the rest typed,
    before any launch.  Above 128 any count binds."""
    g = _geometry()
    case48 = PagedCase(
        g,
        (300, 0, 77),
        (16, 0, 32),
        page_size=16,
        layout="NHD",
        seed=32,
        table_width=70,
    )  # 48 live rows
    plan = _plan_extend(case48, g, 64)
    d = plan.declaration
    assert d["out_proj_tile"] and d["out_proj_tile"].endswith("_swapAB"), d[
        "out_proj_tile"
    ]
    assert plan.live_rows_multiple == d["live_rows_multiple"] == 32, d
    h, cos, sin, slots = _padded(case48, 64)
    args = dict(case48.extend_args(), slot_mapping=slots)
    # the capacity launch (the token tensors at the capacity, pad rows finite with slot -1) serves any live total
    out = plan.run(h, cos, sin, case48.weights, case48.w_o, **args)
    torch.cuda.synchronize()
    case48.check_out(out[: case48.t], label="extend cap 64 capacity launch")
    case48.check_written_rows(label="extend cap 64 capacity launch")
    # 48 live rows are not a multiple of 32: a typed decline naming the rule and the remedies, in both admitted spellings
    with pytest.raises(
        ValueError, match="num_actual_tokens=48 must be a multiple of 32"
    ):
        plan.run(h, cos, sin, case48.weights, case48.w_o, num_actual_tokens=48, **args)
    with pytest.raises(
        ValueError, match="num_actual_tokens=48 must be a multiple of 32"
    ):
        plan.run(
            case48.hidden_states,
            case48.rope_cos,
            case48.rope_sin,
            case48.weights,
            case48.w_o,
            num_actual_tokens=48,
            **case48.extend_args(),
        )
    assert not plan.supports(
        hidden_states=h, rope_cos=cos, rope_sin=sin, num_actual_tokens=48, **args
    )
    # a capacity that is not a multiple of 32 compiles with the widest epilogue vector that divides it (16 on 48 tokens): the
    # multiple always divides the capacity, and the live-row rule follows it
    plan48 = _plan_extend(case48, g, 48)
    assert plan48.live_rows_multiple == 16 and 48 % plan48.live_rows_multiple == 0, (
        plan48.declaration
    )
    h48, cos48, sin48, slots48 = _padded(case48, 48)
    args48 = dict(case48.extend_args(), slot_mapping=slots48)
    out48 = plan48.run(h48, cos48, sin48, case48.weights, case48.w_o, **args48)
    live48 = plan48.run(
        h48, cos48, sin48, case48.weights, case48.w_o, num_actual_tokens=48, **args48
    )
    torch.cuda.synchronize()
    case48.check_out(out48, label="extend cap 48 capacity launch")
    assert torch.equal(live48, out48)
    with pytest.raises(
        ValueError, match="num_actual_tokens=40 must be a multiple of 16"
    ):
        plan48.run(
            h48,
            cos48,
            sin48,
            case48.weights,
            case48.w_o,
            num_actual_tokens=40,
            **args48,
        )
    # 32 live rows on a 64-token plan: both spellings launch and agree bitwise with the capacity launch's rows
    case32 = PagedCase(
        g, (300, 77), (16, 16), page_size=16, layout="NHD", seed=34, table_width=70
    )
    plan32 = _plan_extend(case32, g, 64)
    assert plan32.live_rows_multiple == 32
    h, cos, sin, slots = _padded(case32, 64)
    args = dict(case32.extend_args(), slot_mapping=slots)
    full = plan32.run(h, cos, sin, case32.weights, case32.w_o, **args)
    live = plan32.run(
        h, cos, sin, case32.weights, case32.w_o, num_actual_tokens=32, **args
    )
    exact = plan32.run(
        case32.hidden_states,
        case32.rope_cos,
        case32.rope_sin,
        case32.weights,
        case32.w_o,
        num_actual_tokens=32,
        **case32.extend_args(),
    )
    torch.cuda.synchronize()
    case32.check_out(full[:32], label="extend cap 64 capacity launch (32 live)")
    assert torch.equal(live[:32], full[:32]) and torch.equal(exact, full[:32])
    # above the small-capacity route (the forced 256-wide tile takes the rows as M): 48 live rows bind
    plan256 = _plan_extend(case48, g, 256)
    assert plan256.live_rows_multiple == 1 and not plan256.declaration[
        "out_proj_tile"
    ].endswith("_swapAB")
    h, cos, sin, slots = _padded(case48, 256)
    out = plan256.run(
        h,
        cos,
        sin,
        case48.weights,
        case48.w_o,
        num_actual_tokens=48,
        **dict(case48.extend_args(), slot_mapping=slots),
    )
    torch.cuda.synchronize()
    case48.check_out(out[:48], label="extend cap 256 live 48")


def test_a_cached_extend_step_runs_without_a_host_sync():
    """A prepared extend plan's run under ``set_sync_debug_mode("error")``: the host checks read no device data."""
    g = _geometry()
    case = PagedCase(
        g,
        (300, 0, 77),
        (64, 0, 32),
        page_size=16,
        layout="NHD",
        seed=35,
        table_width=70,
    )
    plan = _plan_extend(case, g, 256)
    h, cos, sin, slots = _padded(case, 256)
    args = dict(case.extend_args(), slot_mapping=slots, num_actual_tokens=case.t)
    first = plan.run(h, cos, sin, case.weights, case.w_o, **args)
    torch.cuda.synchronize()
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        again = plan.run(h, cos, sin, case.weights, case.w_o, **args)
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    torch.cuda.synchronize()
    assert torch.equal(first[: case.t], again[: case.t])
