# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The EXTEND form against the fp32 oracle: a packed prefill step (empty prefixes) and a step behind cached prefixes, pads in the
middle of the batch, the bounded token count, both cache layouts."""

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
