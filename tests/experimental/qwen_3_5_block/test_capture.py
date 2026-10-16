# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph capture of the DECODE form: plan outside capture, the FIRST run captured with no eager warm-up under
``set_sync_debug_mode("error")``, the replay bitwise the eager run, a replay over new lengths / tables / slots / inputs."""

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
