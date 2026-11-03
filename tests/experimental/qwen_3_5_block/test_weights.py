# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU: the weight packer against an independent spelling of the permutation at TP 1 / 2 / 4 / 8, and bitwise the vendored
cuDNN Frontend loader where the CuTe DSL is importable."""

import pytest
import torch

GEOMS = {
    "32_2": dict(d_model=4096, num_qo_heads=32, num_kv_heads=2),
    "24_2": dict(d_model=2560, num_qo_heads=24, num_kv_heads=2),
}


def _hf_layer(name, seed=0, dtype=torch.bfloat16):
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry

    g = Qwen35BlockGeometry(**GEOMS[name])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d, hq, hkv, dm = g.head_dim, g.num_qo_heads, g.num_kv_heads, g.d_model
    randn = lambda *s, std: (
        torch.randn(*s, generator=gen, dtype=torch.float32) * std
    ).to(dtype)  # noqa: E731
    return (
        g,
        randn(2 * hq * d, dm, std=0.02),
        randn(hkv * d, dm, std=0.02),
        randn(hkv * d, dm, std=0.02),
        torch.randn(d, generator=gen) * 0.1,
        torch.randn(d, generator=gen) * 0.1,
    )


def _rank(q, k, v, g, tp, rank):
    """The rows a serving stack's QKV linear gives rank ``rank`` of ``tp`` and the rank's geometry (one KV head per rank, replicated, at tp >= H_kv)."""
    from flashinfer.qwen_3_5_block import Qwen35BlockGeometry

    d = g.head_dim
    hq_r = g.num_qo_heads // tp
    if tp >= g.num_kv_heads:
        hkv_r, replicas = 1, tp // g.num_kv_heads
    else:
        hkv_r, replicas = g.num_kv_heads // tp, 1
    kv_rank = rank // replicas
    g_r = Qwen35BlockGeometry(
        d_model=g.d_model,
        num_qo_heads=hq_r,
        num_kv_heads=hkv_r,
        head_dim=d,
        rope_dim=g.rope_dim,
    )
    return (
        q[rank * 2 * hq_r * d : (rank + 1) * 2 * hq_r * d],
        k[kv_rank * hkv_r * d : (kv_rank + 1) * hkv_r * d],
        v[kv_rank * hkv_r * d : (kv_rank + 1) * hkv_r * d],
        g_r,
    )


def _permute_rows(merged, g):
    """An independent spelling: split [2 q | kv | kv], view the Q block per head, cut [:D] / [D:], concatenate Q | GATE | K | V."""
    d, hq, hkv, dm = g.head_dim, g.num_qo_heads, g.num_kv_heads, g.d_model
    q_gate, w_k, w_v = torch.split(merged, [2 * hq * d, hkv * d, hkv * d], dim=0)
    per_head = q_gate.reshape(hq, 2 * d, dm)
    return torch.cat(
        [
            per_head[:, :d].reshape(hq * d, dm),
            per_head[:, d:].reshape(hq * d, dm),
            w_k,
            w_v,
        ],
        dim=0,
    ).contiguous()


@pytest.mark.parametrize(
    "name, tp",
    [
        ("32_2", 1),
        ("32_2", 2),
        ("32_2", 4),
        ("32_2", 8),
        ("24_2", 1),
        ("24_2", 2),
        ("24_2", 4),
    ],
)
def test_pack_weights_is_the_permutation_at_every_rank(name, tp):
    from flashinfer.qwen_3_5_block import qwen_3_5_block_pack_weights

    g, q, k, v, qn, kn = _hf_layer(name)
    for rank in range(tp):
        q_r, k_r, v_r, g_r = _rank(q, k, v, g, tp, rank)
        merged = torch.cat([q_r, k_r, v_r], dim=0).contiguous()
        got = qwen_3_5_block_pack_weights(merged, qn, kn, g_r)
        assert (
            got.w_qkvg.shape == (g_r.n_qkvg, g.d_model)
            and got.w_qkvg.numel() == merged.numel()
            and got.w_qkvg.is_contiguous()
        )
        assert got.w_qkvg.dtype == torch.bfloat16 and g_r.n_qkvg % 256 == 0
        assert torch.equal(got.w_qkvg, _permute_rows(merged, g_r))
        assert torch.equal(
            got.w_q_norm, (1.0 + qn.float()).to(torch.bfloat16)
        ) and torch.equal(got.w_k_norm, (1.0 + kn.float()).to(torch.bfloat16))
        # K and V rows untouched; the Q / GATE bands a reordering of the q_proj rows
        o_k = 2 * g_r.num_qo_heads * g_r.head_dim
        assert torch.equal(got.w_qkvg[o_k:], merged[o_k:])
        assert torch.equal(
            got.w_qkvg[:o_k]
            .view(2, g_r.num_qo_heads, g_r.head_dim, g.d_model)
            .transpose(0, 1)
            .reshape(-1, g.d_model),
            merged[:o_k],
        )


def test_pack_weights_refusals_are_typed():
    from flashinfer.qwen_3_5_block import (
        Qwen35BlockGeometry,
        qwen_3_5_block_pack_weights,
    )

    g, q, k, v, qn, kn = _hf_layer("24_2")
    merged = torch.cat([q, k, v], dim=0)
    with pytest.raises(ValueError, match="rows"):
        qwen_3_5_block_pack_weights(merged[:-1], qn, kn, g)
    with pytest.raises(ValueError, match="columns"):
        qwen_3_5_block_pack_weights(merged[:, :-1], qn, kn, g)
    with pytest.raises(ValueError, match="q_norm_weight is required"):
        qwen_3_5_block_pack_weights(merged, None, kn, g)
    with pytest.raises(ValueError, match="head_dim=256"):
        qwen_3_5_block_pack_weights(merged, qn[:-1], kn, g)
    g_nonorm = Qwen35BlockGeometry(
        d_model=2560, num_qo_heads=24, num_kv_heads=2, qk_norm=False
    )
    with pytest.raises(ValueError, match="qk_norm=False"):
        qwen_3_5_block_pack_weights(merged, qn, kn, g_nonorm)
    got = qwen_3_5_block_pack_weights(merged, None, None, g_nonorm)
    assert got.w_q_norm is None and got.w_k_norm is None


def test_pack_weights_is_bitwise_the_vendored_loader():
    """Where the CuTe DSL is importable (the vendored tree imports it), the pure-torch packer equals cuDNN Frontend's ``qkvg_from_vllm``."""
    pytest.importorskip("cutlass")
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block import (
        qkvg_from_vllm,
    )
    from flashinfer.qwen_3_5_block import qwen_3_5_block_pack_weights

    for name in GEOMS:
        g, q, k, v, qn, kn = _hf_layer(name)
        merged = torch.cat([q, k, v], dim=0).contiguous()
        ours = qwen_3_5_block_pack_weights(merged, qn, kn, g)
        theirs = qkvg_from_vllm(merged, qn, kn, g.frost())
        for a, b in zip(ours, theirs, strict=False):
            assert torch.equal(a, b)
