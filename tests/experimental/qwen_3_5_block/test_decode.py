# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The DECODE form against the fp32 oracle: the two Qwen3.5 TP-1 shard geometries, q_len 1 and 4, a short / a long / a padded
sequence, both FlashInfer cache layouts, the functional entry point and the wrapper."""

import pytest
import torch

from tests.test_helpers.qwen_3_5_block_cases import PagedCase

pytestmark = [pytest.mark.arch_rubin, pytest.mark.usefixtures("rubin")]

GEOMETRIES = {
    "24_2": dict(d_model=2560, num_qo_heads=24, num_kv_heads=2),
    "32_2": dict(d_model=4096, num_qo_heads=32, num_kv_heads=2),
}
LENS = (
    100,
    500,
    0,
)  # one KV tile with a tail, four tiles with a tail, a PADDED entry (block 0, slot -1, seq_len 0)


def _geometry(name):
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry

    return Qwen35BlockGeometry(**GEOMETRIES[name])


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
@pytest.mark.parametrize("q_len", [1, 4])
def test_decode_matches_the_oracle(name, q_len):
    from flashinfer.qwen_3_5_block import qwen_3_5_block_bf16_decode

    g = _geometry(name)
    case = PagedCase(
        g, LENS, [q_len if L else 0 for L in LENS], page_size=16, layout="NHD", seed=1
    )
    # a padded entry's rows: present in the batch (uniform q_len), zero-length in the case -- hand it finite rows
    b = len(LENS)
    h = torch.zeros(b * q_len, g.d_model, dtype=torch.bfloat16, device="cuda")
    cos = torch.ones(b * q_len, g.rope_dim, dtype=torch.bfloat16, device="cuda")
    sin = torch.zeros_like(cos)
    slots = torch.full((b * q_len,), -1, dtype=torch.int32, device="cuda")
    live = [i for i, L in enumerate(LENS) if L]
    for j, i in enumerate(live):
        h[i * q_len : (i + 1) * q_len] = case.hidden_states[j * q_len : (j + 1) * q_len]
        cos[i * q_len : (i + 1) * q_len] = case.rope_cos[j * q_len : (j + 1) * q_len]
        sin[i * q_len : (i + 1) * q_len] = case.rope_sin[j * q_len : (j + 1) * q_len]
        slots[i * q_len : (i + 1) * q_len] = case.slot_mapping[
            j * q_len : (j + 1) * q_len
        ]
    out = qwen_3_5_block_bf16_decode(
        h,
        cos,
        sin,
        case.weights,
        case.w_o,
        geometry=g,
        kv_cache=case.kv_cache,
        slot_mapping=slots,
        block_table=case.block_table,
        seq_lens=case.seq_lens,
        q_len_per_req=q_len,
    )
    torch.cuda.synchronize()
    assert out.shape == (b * q_len, g.d_model) and out.dtype == torch.bfloat16
    padded = [i for i, L in enumerate(LENS) if L == 0]
    for i in padded:
        assert (out[i * q_len : (i + 1) * q_len] == 0).all(), (
            "a padded entry (seq_len 0) must yield out rows exactly 0"
        )
    rows = torch.cat([out[i * q_len : (i + 1) * q_len] for i in live])
    case.check_out(rows, label=f"decode {name} q{q_len}")
    case.check_written_rows(label=f"decode {name} q{q_len}")
    assert (case.kv_cache[0] == 0).all(), (
        "the NULL page (block 0) must never be written"
    )


def test_decode_layouts_are_bitwise():
    """The same step through an NHD and an HND cache binds without a repack, two launches are bitwise, and the two layouts give
    bitwise the same output."""
    from flashinfer.qwen_3_5_block import qwen_3_5_block_bf16_decode

    g = _geometry("24_2")
    outs = {}
    for layout in ("NHD", "HND"):
        case = PagedCase(g, (300, 77), (2, 2), page_size=16, layout=layout, seed=2)
        args = dict(geometry=g, kv_layout=layout, q_len_per_req=2, **case.decode_args())
        out = qwen_3_5_block_bf16_decode(
            case.hidden_states,
            case.rope_cos,
            case.rope_sin,
            case.weights,
            case.w_o,
            **args,
        )
        torch.cuda.synchronize()
        case.check_out(out, label=f"decode {layout}")
        again = qwen_3_5_block_bf16_decode(
            case.hidden_states,
            case.rope_cos,
            case.rope_sin,
            case.weights,
            case.w_o,
            **args,
        )
        torch.cuda.synchronize()
        assert torch.equal(out, again), "two launches of the same step differ"
        outs[layout] = out.clone()
    assert torch.equal(outs["NHD"], outs["HND"]), (
        "NHD and HND pools holding the same values must give bitwise the same output"
    )


def test_wrapper_plan_run_and_the_declaration_facts():
    from flashinfer.qwen_3_5_block import Qwen35BlockWrapper

    g = _geometry("24_2")
    case = PagedCase(
        g, (257, 16), (1, 1), page_size=32, layout="NHD", seed=3, table_width=64
    )
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w = Qwen35BlockWrapper(ws, "NHD", geometry=g)
    w.plan(
        "decode",
        weights=case.weights,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=32,
        batch_size=2,
        q_len_per_req=1,
        block_table=case.block_table,
    )
    plan = w.plan_info
    assert (
        plan.form == "decode"
        and plan.page_size == 32
        and plan.table_row_stride == 64
        and plan.max_pages == 64
    )
    assert (
        plan.workspace_size <= ws.numel()
        and plan.declaration["decode_route"]["tile"] == "decode_d256_f16"
    )
    assert (
        plan.declaration["gate_fused"]
        and plan.declaration["cache_write_fused"]
        and plan.declaration["decode_split_kv"] >= 2
    )
    out = w.run(
        case.hidden_states,
        case.rope_cos,
        case.rope_sin,
        case.weights,
        case.w_o,
        **case.decode_args(),
    )
    torch.cuda.synchronize()
    case.check_out(out, label="wrapper decode p32")
    assert plan.supports(
        hidden_states=case.hidden_states,
        rope_cos=case.rope_cos,
        rope_sin=case.rope_sin,
        **case.decode_args(),
    )
    assert not plan.supports(
        hidden_states=case.hidden_states[:1],
        rope_cos=case.rope_cos,
        rope_sin=case.rope_sin,
        **case.decode_args(),
    )
    # a workspace too small for the plan is a typed ValueError naming the size
    with pytest.raises(ValueError, match="workspace_size"):
        Qwen35BlockWrapper(
            torch.empty(16, dtype=torch.uint8, device="cuda"), "NHD", geometry=g
        ).plan(
            "decode",
            weights=case.weights,
            w_o=case.w_o,
            kv_cache=case.kv_cache,
            page_size=32,
            batch_size=2,
            q_len_per_req=1,
            block_table=case.block_table,
        )


def test_declines_at_plan_time_are_typed_and_named():
    from flashinfer.qwen_3_5_block import (
        Qwen35BlockGeometry,
        prepare_qwen_3_5_block_bf16,
    )

    g = _geometry("24_2")
    case = PagedCase(g, (64,), (1,), page_size=16, layout="NHD", seed=4)
    common = dict(
        geometry=g,
        device="cuda",
        weights=case.weights,
        w_o=case.w_o,
        kv_cache=case.kv_cache,
        page_size=16,
        batch_size=1,
        block_table=case.block_table,
    )
    with pytest.raises(NotImplementedError, match="decode envelope"):
        prepare_qwen_3_5_block_bf16("decode", q_len_per_req=33, **common)
    with pytest.raises(ValueError, match="page_size"):
        prepare_qwen_3_5_block_bf16("decode", **dict(common, page_size=8))
    with pytest.raises(
        ValueError, match="kv_cache"
    ):  # the cache's page size disagrees with the plan's
        prepare_qwen_3_5_block_bf16("decode", **dict(common, page_size=32))
    with pytest.raises(NotImplementedError, match="head_dim=256"):
        prepare_qwen_3_5_block_bf16(
            "decode",
            **dict(
                common,
                geometry=Qwen35BlockGeometry(
                    d_model=2560, num_qo_heads=24, num_kv_heads=2, head_dim=128
                ),
            ),
        )
    with pytest.raises(NotImplementedError, match="bfloat16"):
        _fp16_entry(common)


def _fp16_entry(common):
    """The bf16 entry refuses f16 weights by name (the dtype is the entry's, not an argument)."""
    from flashinfer.experimental.qwen_3_5_block import api as _api

    _api.prepare("decode", dtype=torch.float16, **common)
