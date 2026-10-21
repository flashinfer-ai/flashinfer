# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The dense PREFILL form against the fp32 oracle: a uniform [batch, seq] step with write-through into a FlashInfer cache."""

import pytest
import torch

from tests.test_helpers.qwen_3_5_block_cases import (
    cache_rows,
    new_kv_cache,
    random_weights,
    ref_geometry,
)
from tests.test_helpers.qwen_3_5_block_reference import (
    build_rope_tables,
    cosine,
    gated_attention_block_reference,
)

pytestmark = [pytest.mark.arch_rubin, pytest.mark.usefixtures("rubin")]


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("batch, seq", [(2, 128), (1, 300)])
def test_prefill_matches_the_oracle_and_writes_the_pools(layout, batch, seq):
    from flashinfer.qwen_3_5_block import (
        Qwen35BlockGeometry,
        qwen_3_5_block_bf16_prefill,
    )

    g = Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)
    rg = ref_geometry(g)
    weights, w_o = random_weights(g, device="cuda", seed=8)
    gen = torch.Generator(device="cuda").manual_seed(8)
    h = torch.randn(
        batch, seq, g.d_model, generator=gen, device="cuda", dtype=torch.float32
    ).to(torch.bfloat16)
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
    page = 16
    pages_per_seq = -(-seq // page)
    kv = new_kv_cache(
        batch * pages_per_seq + 3,
        g.num_kv_heads,
        page,
        g.head_dim,
        layout,
        device="cuda",
    )
    # slots: sequence b takes pages [1 + b * pages_per_seq, ...); every 9th token padded (-1) to exercise the no-write slot
    pos = torch.arange(seq, device="cuda")
    slots = torch.cat(
        [
            ((1 + b * pages_per_seq + pos // page) * page + pos % page)
            for b in range(batch)
        ]
    ).to(torch.int32)
    slots[8::9] = -1
    out = qwen_3_5_block_bf16_prefill(
        h,
        cos,
        sin,
        weights,
        w_o,
        geometry=g,
        kv_cache=kv,
        slot_mapping=slots,
        kv_layout=layout,
    )
    torch.cuda.synchronize()
    assert out.shape == h.shape
    c = cosine(out, ref.out)
    print(f"[prefill {layout} {batch}x{seq}] vs oracle cos {c:.6f}")
    assert torch.isfinite(out.float()).all() and c > 0.999
    live = slots >= 0
    k_rows, v_rows = cache_rows(kv, layout, slots.to(torch.long).clamp_min(0), page)
    want_k, want_v = (
        ref.k.reshape(-1, g.num_kv_heads, g.head_dim),
        ref.v.reshape(-1, g.num_kv_heads, g.head_dim),
    )
    torch.testing.assert_close(
        v_rows[live].float(), want_v[live].float(), rtol=2.0**-7, atol=1e-2
    )
    torch.testing.assert_close(
        k_rows[live].float(), want_k[live].float(), rtol=2.0**-7, atol=1e-2
    )
    assert (kv[0] == 0).all(), "page 0 was never addressed and must stay untouched"
    # the padded slots wrote nothing: their page rows past the live ones stay zero
    assert int((kv[1:] != 0).any(dim=-1).sum()) <= int(live.sum()) * 2 * g.num_kv_heads
    # a 2-D token matrix is one sequence
    out2 = qwen_3_5_block_bf16_prefill(
        h[0],
        cos[0],
        sin[0],
        weights,
        w_o,
        geometry=g,
        kv_cache=kv,
        slot_mapping=slots[:seq],
        kv_layout=layout,
    )
    torch.cuda.synchronize()
    assert out2.shape == (seq, g.d_model) and cosine(out2, ref.out[0]) > 0.999
