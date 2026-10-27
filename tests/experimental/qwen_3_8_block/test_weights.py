# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU: the weight packer on both spellings (the indexer band appended to the merged weight, or a separate indexer projection) at the
TP 1 / 2 / 4 shard geometries, the four (1 + w) norms, the typed refusals, and bitwise the vendored cuDNN Frontend loader where the
CuTe DSL is importable."""

import pytest
import torch

GEOMS = {
    "24_2": dict(d_model=2560, num_qo_heads=24, num_kv_heads=2),  # TP 1
    "12_1": dict(d_model=2560, num_qo_heads=12, num_kv_heads=1),  # TP 2
    "6_1": dict(d_model=2560, num_qo_heads=6, num_kv_heads=1),  # TP 4
}


def _layer(name, seed=0, dtype=torch.bfloat16):
    """A rank's tensors: the merged [q (per-head [q_h | gate_h]) | k | v] weight, the indexer projection [640, d_model], the four norms."""
    from flashinfer.qwen_3_8_block import Qwen38BlockGeometry

    g = Qwen38BlockGeometry(**GEOMS[name])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d, hq, hkv, dm = g.head_dim, g.num_qo_heads, g.num_kv_heads, g.d_model
    randn = lambda *s, std: (
        torch.randn(*s, generator=gen, dtype=torch.float32) * std
    ).to(dtype)  # noqa: E731
    merged = torch.cat(
        [
            randn(2 * hq * d, dm, std=0.02),
            randn(hkv * d, dm, std=0.02),
            randn(hkv * d, dm, std=0.02),
        ],
        dim=0,
    ).contiguous()
    band = randn(g.index_band_cols, dm, std=0.02)
    norms = [torch.randn(d, generator=gen) * 0.1, torch.randn(d, generator=gen) * 0.1]
    inorms = [
        torch.randn(g.index_head_dim, generator=gen) * 0.1,
        torch.randn(g.index_head_dim, generator=gen) * 0.1,
    ]
    return g, merged, band, norms, inorms


def _permute_rows(merged, g):
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
    )


@pytest.mark.parametrize("name", sorted(GEOMS))
def test_pack_weights_both_spellings_are_the_permutation_plus_the_band(name):
    from flashinfer.qwen_3_8_block import qwen_3_8_block_pack_weights

    g, merged, band, (qn, kn), (iqn, ikn) = _layer(name)
    # spelling 1: the band appended to the merged weight (a stack that packs the replicated indexer projection into the same GEMM)
    packed = torch.cat([merged, band], dim=0).contiguous()
    a = qwen_3_8_block_pack_weights(
        packed, qn, kn, g, index_q_norm_weight=iqn, index_k_norm_weight=ikn
    )
    # spelling 2: the band as its own projection
    b = qwen_3_8_block_pack_weights(
        merged,
        qn,
        kn,
        g,
        index_qk_proj_weight=band,
        index_q_norm_weight=iqn,
        index_k_norm_weight=ikn,
    )
    for got in (a, b):
        assert (
            got.w_qkvg.shape == (g.n_qkvg, g.d_model)
            and got.w_qkvg.is_contiguous()
            and got.w_qkvg.dtype == torch.bfloat16
        )
        assert got.w_qkvg.numel() == packed.numel(), (
            "a row permutation keeps the byte count"
        )
        assert torch.equal(got.w_qkvg[: g.n_qkvg_dense], _permute_rows(merged, g))
        assert torch.equal(got.w_qkvg[g.n_qkvg_dense :], band), (
            "the indexer band rides below V as the checkpoint stores it"
        )
        assert torch.equal(
            got.w_q_norm, (1.0 + qn.float()).to(torch.bfloat16)
        ) and torch.equal(got.w_k_norm, (1.0 + kn.float()).to(torch.bfloat16))
        assert torch.equal(
            got.w_iq_norm, (1.0 + iqn.float()).to(torch.bfloat16)
        ) and torch.equal(got.w_ik_norm, (1.0 + ikn.float()).to(torch.bfloat16))
        assert (
            got.w_iq_norm.shape == (g.index_head_dim,) and got.w_ik_norm.is_contiguous()
        )
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y), "the two spellings pack the same bytes"


def test_pack_weights_refusals_are_typed():
    from flashinfer.qwen_3_8_block import (
        Qwen38BlockGeometry,
        qwen_3_8_block_pack_weights,
    )

    g, merged, band, (qn, kn), (iqn, ikn) = _layer("24_2")
    common = dict(index_q_norm_weight=iqn, index_k_norm_weight=ikn)
    with pytest.raises(ValueError, match="given twice"):
        qwen_3_8_block_pack_weights(
            torch.cat([merged, band]), qn, kn, g, index_qk_proj_weight=band, **common
        )
    with pytest.raises(ValueError, match="index_qk_proj_weight is required"):
        qwen_3_8_block_pack_weights(merged, qn, kn, g, **common)
    with pytest.raises(ValueError, match="rows"):
        qwen_3_8_block_pack_weights(
            merged[:-1], qn, kn, g, index_qk_proj_weight=band, **common
        )
    with pytest.raises(ValueError, match="columns"):
        qwen_3_8_block_pack_weights(
            merged[:, :-1], qn, kn, g, index_qk_proj_weight=band[:, :-1], **common
        )
    with pytest.raises(ValueError, match=r"index_qk_proj_weight must be \[640, 2560\]"):
        qwen_3_8_block_pack_weights(
            merged, qn, kn, g, index_qk_proj_weight=band[:-1], **common
        )
    with pytest.raises(ValueError, match="index_q_norm_weight is required"):
        qwen_3_8_block_pack_weights(
            merged, qn, kn, g, index_qk_proj_weight=band, index_k_norm_weight=ikn
        )
    with pytest.raises(ValueError, match=r"index_k_norm_weight must be \[128\]"):
        qwen_3_8_block_pack_weights(
            merged,
            qn,
            kn,
            g,
            index_qk_proj_weight=band,
            index_q_norm_weight=iqn,
            index_k_norm_weight=ikn[:-1],
        )
    with pytest.raises(ValueError, match="q_norm_weight is required"):
        qwen_3_8_block_pack_weights(
            merged, None, kn, g, index_qk_proj_weight=band, **common
        )
    g_nonorm = Qwen38BlockGeometry(
        d_model=2560, num_qo_heads=24, num_kv_heads=2, qk_norm=False
    )
    with pytest.raises(ValueError, match="qk_norm=False"):
        qwen_3_8_block_pack_weights(
            merged, qn, kn, g_nonorm, index_qk_proj_weight=band, **common
        )
    got = qwen_3_8_block_pack_weights(
        merged, None, None, g_nonorm, index_qk_proj_weight=band, **common
    )
    assert got.w_q_norm is None and got.w_k_norm is None and got.w_iq_norm is not None


def test_pack_weights_is_bitwise_the_vendored_loader():
    """Where the CuTe DSL is importable (the vendored tree imports it), the pure-torch packer equals cuDNN Frontend's ``qkvg_from_vllm``
    on both spellings (the band appended; ``index_qk_proj_weight=``), and the indexer norms its ``(1 + w)`` rule."""
    pytest.importorskip("cutlass")
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block import (
        qkvg_from_vllm,
    )
    from flashinfer.qwen_3_8_block import qwen_3_8_block_pack_weights

    for name in GEOMS:
        g, merged, band, (qn, kn), (iqn, ikn) = _layer(name)
        fg = g.frost()
        ours_a = qwen_3_8_block_pack_weights(
            torch.cat([merged, band]),
            qn,
            kn,
            g,
            index_q_norm_weight=iqn,
            index_k_norm_weight=ikn,
        )
        theirs_a = qkvg_from_vllm(torch.cat([merged, band]), qn, kn, fg)
        ours_b = qwen_3_8_block_pack_weights(
            merged,
            qn,
            kn,
            g,
            index_qk_proj_weight=band,
            index_q_norm_weight=iqn,
            index_k_norm_weight=ikn,
        )
        theirs_b = qkvg_from_vllm(merged, qn, kn, fg, index_qk_proj_weight=band)
        for ours, theirs in ((ours_a, theirs_a), (ours_b, theirs_b)):
            for a, b in zip(ours[:3], theirs, strict=True):
                assert torch.equal(a, b)
