# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph capture of the three forms: plan outside capture, the FIRST run captured with no eager warm-up under
``set_sync_debug_mode("error")``, the replay bitwise the eager run; on decode also a replay over new lengths / tables / slots /
inputs; ``prepare`` under capture refused typed; the functional decode captured through the plan cache replays bitwise after
the LRU evicted its plan (the plan retained, ``clear_plan_cache()`` the release)."""

import pytest
import torch

from tests.test_helpers.qwen_3_5_block_cases import PagedCase

pytestmark = [pytest.mark.arch_rubin, pytest.mark.usefixtures("rubin")]


def test_decode_first_run_captured_replays_bitwise_eager():
    from flashinfer.qwen_3_5_block import (
        Qwen35BlockGeometry,
        prepare_qwen_3_5_block_bf16,
    )

    g = Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)
    case = PagedCase(
        g,
        (300, 1000, 77),
        (2, 2, 2),
        page_size=16,
        layout="NHD",
        seed=9,
        table_width=70,
    )
    plan = prepare_qwen_3_5_block_bf16(
        "decode",
        geometry=g,
        device="cuda",
        weights=case.weights,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=16,
        batch_size=3,
        q_len_per_req=2,
        block_table=case.block_table,
    )
    out = torch.full(
        (case.t, g.d_model), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    # the captured tensors: the step's own buffers (addresses are baked)
    h, cos, sin, slots = (
        case.hidden_states.clone(),
        case.rope_cos.clone(),
        case.rope_sin.clone(),
        case.slot_mapping.clone(),
    )
    table, lens = case.block_table.clone(), case.seq_lens.clone()
    written_before = int((case.kv_cache != 0).any(dim=-1).sum())
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                plan.run(
                    h,
                    cos,
                    sin,
                    case.weights,
                    case.w_o,
                    kv_cache=case.kv_cache,
                    slot_mapping=slots,
                    block_table=table,
                    seq_lens=lens,
                    out=out,
                )
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert (
            torch.isnan(out).all()
            and int((case.kv_cache != 0).any(dim=-1).sum()) == written_before
        ), "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        case.check_out(replayed, label="replay")
        out.fill_(float("nan"))
        plan.run(
            h,
            cos,
            sin,
            case.weights,
            case.w_o,
            kv_cache=case.kv_cache,
            slot_mapping=slots,
            block_table=table,
            seq_lens=lens,
            out=out,
        )
        torch.cuda.synchronize()
        assert torch.equal(out, replayed), (
            "the replay of the captured first run differs from the eager run"
        )
        # a NEW step through the captured tensors: new lengths, a re-laid-out table, new slots and inputs
        sib = PagedCase(
            g,
            (301, 1000, 0),
            (2, 2, 0),
            page_size=16,
            layout="NHD",
            seed=10,
            table_width=70,
            extra_pages=case.num_pages - 1 - sum(-(-L // 16) for L in (301, 1000, 0)),
        )
        assert sib.kv_cache.shape == case.kv_cache.shape
        case.kv_cache.copy_(sib.kv_cache)
        table.copy_(sib.block_table)
        lens.copy_(sib.seq_lens)
        slots[:4].copy_(sib.slot_mapping)
        slots[4:] = -1
        h[:4].copy_(sib.hidden_states)
        h[4:] = 0
        cos[:4].copy_(sib.rope_cos)
        cos[4:] = 1
        sin[:4].copy_(sib.rope_sin)
        sin[4:] = 0
        case.weights.w_qkvg.copy_(sib.weights.w_qkvg)
        case.w_o.copy_(sib.w_o)
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed2 = out.clone()
        assert torch.isfinite(replayed2.float()).all() and (replayed2[4:] == 0).all(), (
            "the padded entry must replay as exact zeros"
        )
        sib.check_out(replayed2[:4], label="replay over a new step")
        out.fill_(float("nan"))
        plan.run(
            h,
            cos,
            sin,
            case.weights,
            case.w_o,
            kv_cache=case.kv_cache,
            slot_mapping=slots,
            block_table=table,
            seq_lens=lens,
            out=out,
        )
        torch.cuda.synchronize()
        assert torch.equal(out, replayed2), (
            "the replay over new lengths / tables / slots differs from the eager run over them"
        )
    finally:
        graph.reset()


def test_prepare_under_capture_is_refused_typed():
    from flashinfer.qwen_3_5_block import (
        Qwen35BlockGeometry,
        prepare_qwen_3_5_block_bf16,
    )

    g = Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)
    case = PagedCase(g, (64,), (1,), page_size=16, layout="NHD", seed=11)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with (
        pytest.raises(RuntimeError, match="capture"),
        torch.cuda.graph(graph, stream=stream),
    ):
        prepare_qwen_3_5_block_bf16(
            "decode",
            geometry=g,
            device="cuda",
            weights=case.weights,
            w_o=case.w_o,
            kv_cache=case.kv_cache,
            page_size=16,
            batch_size=1,
            block_table=case.block_table,
        )


def _capture_first_run(run, out, kv_cache, rows=None):
    """Capture ``run()`` as the FIRST launch of its plan under ``set_sync_debug_mode("error")``; assert nothing ran at capture; replay.
    ``rows`` bounds the comparison on a bounded (extend) step: rows past the live ones are untouched by contract (they keep their NaN)."""
    rows = out.shape[0] if rows is None else rows
    written_before = int((kv_cache != 0).any(dim=-1).sum())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                run()
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert (
            torch.isnan(out).all()
            and int((kv_cache != 0).any(dim=-1).sum()) == written_before
        ), "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        assert torch.isnan(replayed[rows:]).all(), (
            "rows past the live rows must stay untouched"
        )
        out.fill_(float("nan"))
        run()
        torch.cuda.synchronize()
        assert torch.equal(out[:rows], replayed[:rows]), (
            "the replay of the captured first run differs from the eager run"
        )
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[:rows], replayed[:rows]), (
            "a second replay differs from the first"
        )
        return replayed
    finally:
        graph.reset()


def test_extend_first_run_captured_replays_bitwise_eager():
    """The extend form at a 64-token capacity (the small-capacity out-projection tile) with 32 live rows: the first run captured,
    the replay bitwise the eager run, the pool rows as the oracle predicts."""
    from flashinfer.qwen_3_5_block import (
        Qwen35BlockGeometry,
        prepare_qwen_3_5_block_bf16,
    )

    g = Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)
    case = PagedCase(
        g,
        (300, 0, 77),
        (16, 0, 16),
        page_size=16,
        layout="NHD",
        seed=24,
        table_width=70,
    )
    cap, t = 64, case.t
    plan = prepare_qwen_3_5_block_bf16(
        "extend",
        geometry=g,
        device="cuda",
        weights=case.weights,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=16,
        batch_size=3,
        token_capacity=cap,
        block_table=case.block_table,
    )
    assert t % plan.live_rows_multiple == 0
    h = torch.zeros(cap, g.d_model, dtype=torch.bfloat16, device="cuda")
    cos = torch.ones(cap, g.rope_dim, dtype=torch.bfloat16, device="cuda")
    sin = torch.zeros_like(cos)
    slots = torch.full((cap,), -1, dtype=torch.int32, device="cuda")
    h[:t], cos[:t], sin[:t], slots[:t] = (
        case.hidden_states,
        case.rope_cos,
        case.rope_sin,
        case.slot_mapping,
    )
    out = torch.full(
        (cap, g.d_model), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    args = dict(
        query_start_loc=case.query_start_loc.clone(),
        kv_cache=case.kv_cache,
        slot_mapping=slots,
        block_table=case.block_table.clone(),
        seq_lens=case.seq_lens.clone(),
        num_actual_tokens=t,
    )
    replayed = _capture_first_run(
        lambda: plan.run(h, cos, sin, case.weights, case.w_o, out=out, **args),
        out,
        case.kv_cache,
        rows=t,
    )
    case.check_out(replayed[:t], label="extend replay")
    case.check_written_rows(label="extend replay")


def test_prefill_first_run_captured_replays_bitwise_eager():
    from flashinfer.qwen_3_5_block import (
        Qwen35BlockGeometry,
        prepare_qwen_3_5_block_bf16,
    )
    from tests.test_helpers.qwen_3_5_block_cases import (
        new_kv_cache,
        random_weights,
        ref_geometry,
    )
    from tests.test_helpers.qwen_3_5_block_reference import (
        build_rope_tables,
        cosine,
        gated_attention_block_reference,
    )

    g = Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)
    weights, w_o = random_weights(g, device="cuda", seed=27)
    batch, seq, page = 2, 96, 16
    gen = torch.Generator(device="cuda").manual_seed(27)
    h = torch.randn(
        batch, seq, g.d_model, generator=gen, device="cuda", dtype=torch.float32
    ).to(torch.bfloat16)
    rg = ref_geometry(g)
    cos, sin = build_rope_tables(
        seq,
        g.rope_dim,
        base=rg.rope_base,
        batch=batch,
        device="cuda",
        dtype=torch.bfloat16,
    )
    ref = gated_attention_block_reference(
        h, weights.w_qkvg, weights.w_q_norm, weights.w_k_norm, cos, sin, w_o, rg
    )
    pages_per_seq = -(-seq // page)
    kv = new_kv_cache(
        batch * pages_per_seq + 3,
        g.num_kv_heads,
        page,
        g.head_dim,
        "NHD",
        device="cuda",
    )
    pos = torch.arange(seq, device="cuda")
    slots = torch.cat(
        [
            ((1 + b * pages_per_seq + pos // page) * page + pos % page)
            for b in range(batch)
        ]
    ).to(torch.int32)
    plan = prepare_qwen_3_5_block_bf16(
        "prefill",
        geometry=g,
        device="cuda",
        weights=weights,
        w_o=w_o,
        kv_cache=kv,
        page_size=page,
        batch_size=batch,
        seq_len=seq,
    )
    out = torch.full(
        (batch, seq, g.d_model), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    replayed = _capture_first_run(
        lambda: plan.run(
            h, cos, sin, weights, w_o, kv_cache=kv, slot_mapping=slots, out=out
        ),
        out,
        kv,
    )
    c = cosine(replayed, ref.out)
    print(f"[prefill replay] vs oracle cos {c:.6f}")
    assert torch.isfinite(replayed.float()).all() and c > 0.999


def test_functional_decode_captured_survives_plan_cache_eviction():
    """The plan-cache lifetime under capture: bucket A's functional decode is captured (a cache hit), then 33 more buckets go
    through the same entry point (A's plan evicted from the 32-entry LRU) and storage of the workspace's size is allocated and
    filled with garbage; A's replay is bitwise its eager run, the garbage untouched, A's plan and workspace alive (retained) and
    no live allocation overlaps the workspace the graph still addresses; the bucket's next call revives that plan; with the graph
    gone, ``clear_plan_cache()`` releases both."""
    import gc
    import weakref

    from flashinfer.experimental.qwen_3_5_block import api
    from flashinfer.qwen_3_5_block import (
        Qwen35BlockGeometry,
        qwen_3_5_block_bf16_decode,
    )

    api.clear_plan_cache()
    g = Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)
    case = PagedCase(
        g,
        (300, 1000, 77),
        (2, 2, 2),
        page_size=16,
        layout="NHD",
        seed=9,
        table_width=70,
    )
    out = torch.full(
        (case.t, g.d_model), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    w = (case.hidden_states, case.rope_cos, case.rope_sin, case.weights, case.w_o)
    args = dict(geometry=g, q_len_per_req=2, out=out, **case.decode_args())
    qwen_3_5_block_bf16_decode(
        *w, **args
    )  # the warm-up: bucket A's plan compiles (a miss, outside capture)
    torch.cuda.synchronize()
    eager = out.clone()
    case.check_out(eager, label="functional decode, eager")
    ((key_a, plan_a),) = api._PLANS.items()
    ws_ptr, ws_size = plan_a._workspace.data_ptr(), plan_a.workspace_size
    refs = weakref.ref(plan_a), weakref.ref(plan_a._workspace)
    del plan_a

    def overlaps(t):
        return (
            t.data_ptr() < ws_ptr + ws_size
            and ws_ptr < t.data_ptr() + t.numel() * t.element_size()
        )

    out.fill_(float("nan"))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                qwen_3_5_block_bf16_decode(*w, **args)  # the cache hit under capture
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(out).all(), "the capture launched work"
        assert len(api._PLANS) == 1 and key_a in api._PLANS, (
            "the capture must be a hit on bucket A's cached plan"
        )
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), (
            "the replay before the churn differs from the eager run"
        )
        # the churn: PLAN_CACHE_SIZE + 1 further buckets of the same step (distinct logical KV bounds are distinct buckets)
        scratch = torch.empty_like(out)
        for i in range(api.PLAN_CACHE_SIZE + 1):
            qwen_3_5_block_bf16_decode(
                *w, **{**args, "out": scratch, "max_seq_len": 1000 + i}
            )
        torch.cuda.synchronize()
        assert len(api._PLANS) == api.PLAN_CACHE_SIZE and key_a not in api._PLANS, (
            "the churn must evict bucket A"
        )
        gc.collect()
        assert refs[0]() is not None and refs[1]() is not None, (
            "the eviction freed the plan / workspace a captured graph still addresses"
        )
        garbage = [
            torch.full((ws_size,), 255, dtype=torch.uint8, device="cuda")
            for _ in range(4)
        ]
        for t in garbage + [p._workspace for p in api._PLANS.values()]:
            assert not overlaps(t), (
                "an allocation overlaps the workspace of the captured plan: it was freed"
            )
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), (
            "the replay after the churn differs from the eager run"
        )
        assert all(bool((t == 255).all()) for t in garbage), (
            "the replay wrote into storage it no longer owns"
        )
        assert [k for k, _ in api._RETAINED] == [key_a] and api._RETAINED[0][1] is refs[
            0
        ]()
        # the bucket asked for again: the retained plan itself, back in the LRU (no second compile, no second workspace)
        out.fill_(float("nan"))
        qwen_3_5_block_bf16_decode(*w, **args)
        torch.cuda.synchronize()
        assert torch.equal(out, eager)
        assert api._PLANS.get(key_a) is refs[0]() and not api._RETAINED
        assert refs[0]()._workspace.data_ptr() == ws_ptr
    finally:
        graph.reset()
    # the documented release: with the graph gone, clear_plan_cache() frees the retained plan and its workspace
    api.clear_plan_cache()
    gc.collect()
    assert refs[0]() is None and refs[1]() is None, (
        "clear_plan_cache() must be the release of a captured plan"
    )
