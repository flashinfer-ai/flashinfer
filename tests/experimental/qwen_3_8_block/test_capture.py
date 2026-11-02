# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph capture of the decode form: the FIRST run of a fresh plan captured with no eager warm-up under
``set_sync_debug_mode("error")``, replayed bitwise the eager run, then replayed over a NEW step (new lengths / slots / ring slots /
hidden states written in place, the pad rows garbage but finite) bitwise an eager run of that step; ``prepare`` under capture refused;
the functional decode captured as a plan-cache hit survives the LRU's eviction."""

import pytest
import torch

from tests.test_helpers.qwen_3_8_block_cases import Qsa38PagedCase, bits

pytestmark = [pytest.mark.arch_rubin, pytest.mark.usefixtures("rubin")]


def _geometry():
    from flashinfer.qwen_3_8_block import Qwen38BlockGeometry

    return Qwen38BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)


def _prepare(case, g, **kw):
    from flashinfer.qwen_3_8_block import prepare_qwen_3_8_block_bf16

    return prepare_qwen_3_8_block_bf16(
        "decode",
        geometry=g,
        device="cuda",
        weights=case.weights38,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=case.page_size,
        batch_size=case.b,
        q_len_per_req=case.q_len,
        block_table=case.block_table,
        **case.plan_args(),
        **kw,
    )


def _pages(lens, page_size=16):
    return sum(-(-L // page_size) for L in lens)


def test_decode_first_run_captured_replays_bitwise_eager_and_over_a_new_step():
    g = _geometry()
    lens_a, lens_b = (303, 2300, 0, 77), (304, 2301, 1000, 0)
    # the two steps share the pools: the first case carries the spare pages the second step's longer sequences need
    case = Qsa38PagedCase(
        g,
        lens_a,
        4,
        page_size=16,
        seed=21,
        table_width=160,
        extra_pages=_pages(lens_b) - _pages(lens_a) + 2,
    )
    case.prime()
    plan = _prepare(case, g, ring_slots=True)
    t = case.b * case.q_len
    out = torch.full((t, g.d_model), float("nan"), dtype=torch.bfloat16, device="cuda")
    ids = torch.full((t, g.top_k), -7, dtype=torch.int32, device="cuda")
    # the captured tensors: the step's own buffers (addresses are baked)
    h, slots, ring_slots = (
        case.hidden_states.clone(),
        case.slot_mapping.clone(),
        case.ring_slots.to(case.slot_mapping.dtype).clone(),
    )
    table, lens, ring_table = (
        case.block_table.clone(),
        case.seq_lens.clone(),
        case.index_state_block_table.clone(),
    )
    state_before = (
        bits(case.kv_cache).clone(),
        bits(case.index_state_cache).clone(),
        bits(case.index_kv_cache).clone(),
    )
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    args = dict(
        kv_cache=case.kv_cache,
        slot_mapping=slots,
        block_table=table,
        seq_lens=lens,
        index_kv_cache=case.index_kv_cache,
        index_state_cache=case.index_state_cache,
        index_state_block_table=ring_table,
        index_state_slot_mapping=ring_slots,
        out=out,
        block_ids_out=ids,
    )
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                plan.run(h, None, None, case.weights38, case.w_o, **args)
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(out).all() and all(
            torch.equal(a, b)
            for a, b in zip(
                state_before,
                (
                    bits(case.kv_cache),
                    bits(case.index_state_cache),
                    bits(case.index_kv_cache),
                ),
                strict=True,
            )
        ), "the capture launched work"
        graph.replay()
        torch.cuda.synchronize()
        replayed, ids_replayed = out.clone(), ids.clone()
        state_after = (
            bits(case.kv_cache).clone(),
            bits(case.index_state_cache).clone(),
            bits(case.index_kv_cache).clone(),
        )
        case.check_out_from_list(replayed, ids_replayed, label="replay")
        # the eager run of the same step over the state BEFORE the step (the step writes the ring / cache / pools: restore them first)
        for dst, src in zip(
            (case.kv_cache, case.index_state_cache, case.index_kv_cache),
            state_before,
            strict=True,
        ):
            dst.view(torch.int16).copy_(src)
        out.fill_(float("nan"))
        plan.run(h, None, None, case.weights38, case.w_o, **args)
        torch.cuda.synchronize()
        assert torch.equal(out, replayed) and torch.equal(ids, ids_replayed), (
            "the replay of the captured first run differs from the eager run"
        )
        assert all(
            torch.equal(a, b)
            for a, b in zip(
                state_after,
                (
                    bits(case.kv_cache),
                    bits(case.index_state_cache),
                    bits(case.index_kv_cache),
                ),
                strict=True,
            )
        ), "the replay's state writes differ from the eager run's"
        # a NEW step through the captured tensors: new lengths / slots / ring slots / inputs (the 77-token entry padded away, the padded one live)
        sib = Qsa38PagedCase(
            g,
            lens_b,
            4,
            page_size=16,
            seed=22,
            table_width=160,
            weights=(case.weights38, case.w_o),
            extra_pages=2,
        )
        sib.prime()
        assert (
            sib.kv_cache.shape == case.kv_cache.shape
            and sib.index_kv_cache.shape == case.index_kv_cache.shape
        )
        case.kv_cache.copy_(sib.kv_cache)
        case.index_kv_cache.copy_(sib.index_kv_cache)
        case.index_state_cache.copy_(sib.index_state_cache)
        table.copy_(sib.block_table)
        lens.copy_(sib.seq_lens)
        ring_table.copy_(sib.index_state_block_table)
        slots.copy_(sib.slot_mapping)
        ring_slots.copy_(sib.ring_slots.to(ring_slots.dtype))
        h.copy_(sib.hidden_states)
        h[12:] = (
            7.0  # the padded entry's rows: garbage but FINITE (the GEMM reads them; the dead-row rule writes out = 0 regardless)
        )
        state_before2 = (
            bits(case.kv_cache).clone(),
            bits(case.index_state_cache).clone(),
            bits(case.index_kv_cache).clone(),
        )
        out.fill_(float("nan"))
        ids.fill_(-7)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        replayed2, ids2 = out.clone(), ids.clone()
        assert (
            torch.isfinite(replayed2.float()).all() and (replayed2[12:] == 0).all()
        ), "the padded entry must replay as exact zeros"
        sib.kv_cache, sib.index_kv_cache, sib.index_state_cache = (
            case.kv_cache,
            case.index_kv_cache,
            case.index_state_cache,
        )
        sib.check_out_from_list(replayed2, ids2, label="replay over a new step")
        state_after2 = (
            bits(case.kv_cache).clone(),
            bits(case.index_state_cache).clone(),
            bits(case.index_kv_cache).clone(),
        )
        for dst, src in zip(
            (case.kv_cache, case.index_state_cache, case.index_kv_cache),
            state_before2,
            strict=True,
        ):
            dst.view(torch.int16).copy_(src)
        out.fill_(float("nan"))
        plan.run(h, None, None, case.weights38, case.w_o, **args)
        torch.cuda.synchronize()
        assert torch.equal(out, replayed2) and torch.equal(ids, ids2), (
            "the replay over new lengths / slots / inputs differs from the eager run over them"
        )
        assert all(
            torch.equal(a, b)
            for a, b in zip(
                state_after2,
                (
                    bits(case.kv_cache),
                    bits(case.index_state_cache),
                    bits(case.index_kv_cache),
                ),
                strict=True,
            )
        )
    finally:
        graph.reset()


def test_prepare_under_capture_is_refused_typed():
    g = _geometry()
    case = Qsa38PagedCase(g, (64,), 1, page_size=16, seed=23)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with (
        pytest.raises(RuntimeError, match="capture"),
        torch.cuda.graph(graph, stream=stream),
    ):
        _prepare(case, g)


def test_functional_decode_captured_survives_plan_cache_eviction():
    """Bucket A's functional decode captured (a cache hit), 33 more buckets through the same entry (A's plan evicted from the 32-entry
    LRU): A's replay is bitwise its eager run, A's plan retained and revived; ``clear_plan_cache()`` the release."""
    import gc
    import weakref

    from flashinfer.experimental.qwen_3_8_block import api
    from flashinfer.qwen_3_8_block import qwen_3_8_block_bf16_decode

    api.clear_plan_cache()
    g = _geometry()
    case = Qsa38PagedCase(g, (300, 1000, 77), 2, page_size=16, seed=24, table_width=70)
    case.prime()
    out = torch.full(
        (case.t, g.d_model), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    w = (case.hidden_states, None, None, case.weights38, case.w_o)
    args = dict(
        geometry=g,
        q_len_per_req=2,
        out=out,
        position_tables=case.position_tables,
        **case.decode_args(),
    )
    state0 = (
        bits(case.kv_cache).clone(),
        bits(case.index_state_cache).clone(),
        bits(case.index_kv_cache).clone(),
    )

    def restore():
        for dst, src in zip(
            (case.kv_cache, case.index_state_cache, case.index_kv_cache),
            state0,
            strict=True,
        ):
            dst.view(torch.int16).copy_(src)

    qwen_3_8_block_bf16_decode(
        *w, **args
    )  # the warm-up: bucket A's plan compiles (a miss, outside capture)
    torch.cuda.synchronize()
    eager = out.clone()
    ((key_a, plan_a),) = api._PLANS.items()
    refs = weakref.ref(plan_a), weakref.ref(plan_a._workspace)
    del plan_a
    restore()
    out.fill_(float("nan"))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.graph(graph, stream=stream):
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                qwen_3_8_block_bf16_decode(*w, **args)  # the cache hit under capture
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        torch.cuda.synchronize()
        assert torch.isnan(out).all() and len(api._PLANS) == 1 and key_a in api._PLANS
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)
        scratch = torch.empty_like(out)
        for i in range(
            api.PLAN_CACHE_SIZE + 1
        ):  # the churn: distinct logical KV bounds are distinct buckets
            restore()
            qwen_3_8_block_bf16_decode(
                *w, **{**args, "out": scratch, "max_seq_len": 1000 + i}
            )
        torch.cuda.synchronize()
        assert len(api._PLANS) == api.PLAN_CACHE_SIZE and key_a not in api._PLANS
        gc.collect()
        assert refs[0]() is not None and refs[1]() is not None, (
            "the eviction freed the plan / workspace a captured graph still addresses"
        )
        restore()
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), (
            "the replay after the churn differs from the eager run"
        )
        assert [k for k, _ in api._RETAINED] == [key_a]
        restore()
        qwen_3_8_block_bf16_decode(*w, **args)
        torch.cuda.synchronize()
        assert (
            torch.equal(out, eager)
            and api._PLANS.get(key_a) is refs[0]()
            and not api._RETAINED
        )
    finally:
        graph.reset()
    api.clear_plan_cache()
    gc.collect()
    assert refs[0]() is None and refs[1]() is None, (
        "clear_plan_cache() must be the release of a captured plan"
    )
