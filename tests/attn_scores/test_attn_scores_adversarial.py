# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adversarial "try to break it" tests for paged MQA logits.

These deliberately hit boundary/degenerate/skewed inputs. They only exercise
*valid* (well-formed) inputs — cases that would trigger an out-of-bounds read
(and poison the CUDA context) are covered by review, not run here.
"""

import pytest
import torch

from flashinfer.utils import is_sm100a_supported

# Reuse the validated helpers + references from the main test module.
from tests.attn_scores.test_attn_scores import (
    _calc_cosine_diff,
    _cast_back_from_fp4,
    _ceil_to_ue8m0_fp,
    _kv_cache_cast_to_fp4,
    _make_fused_kv_fp8,
    _make_paged_kv,
    _per_token_cast_to_fp4,
    _ref_fp4_paged_mqa_logits,
    _ref_fp8_paged_mqa_logits,
    _valid_causal_mask,
)

DEVICE = "cuda"


def _skip_if_not_sm100():
    if not is_sm100a_supported(torch.device(DEVICE)):
        pytest.skip("paged MQA logits requires SM100a (B200)")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build valid packed inputs from given context_lens
# ─────────────────────────────────────────────────────────────────────────────


def _build_fp8(context_lens, next_n, block_size, seed=0, num_heads=64, head_dim=128):
    torch.manual_seed(seed)
    B = context_lens.shape[0]
    block_table, ntb = _make_paged_kv(B, block_size, context_lens, DEVICE)
    q = torch.randn(B, next_n, num_heads, head_dim, device=DEVICE).to(
        torch.float8_e4m3fn
    )
    kv = torch.randn(ntb, block_size, head_dim, device=DEVICE)
    amax = kv.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
    scale = _ceil_to_ue8m0_fp(amax / 448.0).squeeze(-1)
    kv_fp8 = (kv / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    weights = torch.randn(B * next_n, num_heads, device=DEVICE)
    kv_fused = _make_fused_kv_fp8(kv_fp8, scale, block_size, head_dim)
    return q, kv_fp8, scale, weights, kv_fused, block_table


def _cmp_fp8(
    q, kv_fp8, scale, weights, kv_fused, cl, bt, max_ml, next_n, tol=0.02, atol=2e-4
):
    from flashinfer import fp8_paged_mqa_logits

    ref = _ref_fp8_paged_mqa_logits(
        q,
        kv_fp8,
        scale,
        weights,
        cl,
        bt,
        max_ml,
        kv_fused.shape[1],
        out_dtype=torch.float32,
    )
    out = fp8_paged_mqa_logits(q, kv_fused, weights, cl, bt, max_ml)
    valid = _valid_causal_mask(cl, next_n, max_ml, DEVICE)
    o = out.float().masked_fill(~valid, 0)
    r = ref.float().masked_fill(~valid, 0)
    fin = torch.isfinite(o) & torch.isfinite(r)
    v = valid & fin
    diff = _calc_cosine_diff(o.masked_fill(~fin, 0), r.masked_fill(~fin, 0))
    max_abs = (o[v] - r[v]).abs().max().item() if v.any() else 0.0
    return diff, max_abs, out


# ─────────────────────────────────────────────────────────────────────────────
# A) Schedule kernel: GPU vs CPU bit-exact under adversarial batch/context
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("batch_size", [1, 7, 31, 32, 33, 63, 64, 65, 127, 128, 256])
@pytest.mark.parametrize("dist", ["uniform", "skewed", "minimal", "twotier", "random"])
def test_adv_schedule_gpu_vs_cpu(batch_size, dist):
    """The on-GPU schedule kernel must be BIT-EXACT vs the CPU reference for every
    batch size (incl. non-multiples of 32) and adversarial context distributions."""
    _skip_if_not_sm100()
    from flashinfer.attn_scores.attn_scores import (
        _cached_num_sms,
        _compute_schedule_metadata,
        compute_paged_mqa_logits_schedule,
    )
    from flashinfer.utils import get_device_index

    torch.manual_seed(batch_size)
    if dist == "uniform":
        cl = torch.full((batch_size,), 4096, dtype=torch.int32, device=DEVICE)
    elif dist == "skewed":
        cl = torch.ones(batch_size, dtype=torch.int32, device=DEVICE)
        cl[0] = 500000  # one huge sequence, rest length-1
    elif dist == "minimal":
        cl = torch.ones(batch_size, dtype=torch.int32, device=DEVICE)
    elif dist == "twotier":
        cl = torch.where(
            torch.arange(batch_size, device=DEVICE) % 2 == 0,
            torch.tensor(131072, device=DEVICE),
            torch.tensor(128, device=DEVICE),
        ).to(torch.int32)
    else:
        cl = torch.randint(1, 200000, (batch_size,), dtype=torch.int32, device=DEVICE)

    num_sms = _cached_num_sms(get_device_index(torch.device(DEVICE)))
    ref = _compute_schedule_metadata(cl.cpu(), num_sms).to(DEVICE)
    gpu = compute_paged_mqa_logits_schedule(cl, use_gpu_kernel=True)
    torch.cuda.synchronize()
    assert torch.equal(ref, gpu), (
        f"GPU != CPU schedule for B={batch_size} dist={dist}: "
        f"{(ref != gpu).sum().item()} mismatched entries"
    )


# ─────────────────────────────────────────────────────────────────────────────
# B) Context lengths at block / SPLIT_KV boundaries (± 1)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("next_n", [1, 2, 3])
@pytest.mark.parametrize(
    "ctx",
    [1, 2, 3, 33, 64, 127, 128, 129, 255, 256, 257, 383, 384, 512, 640, 1024, 1025],
)
def test_adv_context_boundaries_fp8(block_size, next_n, ctx):
    """Boundary context lengths (block/128/256 multiples and ±1) must stay correct."""
    _skip_if_not_sm100()
    if ctx < next_n:
        pytest.skip("ctx must be >= next_n for a valid causal window")
    cl = torch.full((4,), ctx, dtype=torch.int32, device=DEVICE)
    max_ml = max(ctx + 8, 512)
    q, kv_fp8, scale, w, kv_fused, bt = _build_fp8(cl, next_n, block_size, seed=ctx)
    diff, max_abs, _ = _cmp_fp8(q, kv_fp8, scale, w, kv_fused, cl, bt, max_ml, next_n)
    assert diff < 0.02, f"cosine diff {diff:.3e} (max_abs {max_abs:.3e}) ctx={ctx}"


# ─────────────────────────────────────────────────────────────────────────────
# C) Skewed / variable context lengths (schedule load imbalance)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("next_n", [1, 2])
def test_adv_skewed_varlen_fp8(next_n):
    """One very long sequence + many short ones (schedule stress)."""
    _skip_if_not_sm100()
    B, block_size = 16, 64
    cl = torch.full((B,), 128, dtype=torch.int32, device=DEVICE)
    cl[0] = 32768
    cl[1] = 16384
    max_ml = 32768 + 8
    q, kv_fp8, scale, w, kv_fused, bt = _build_fp8(cl, next_n, block_size, seed=7)
    diff, max_abs, _ = _cmp_fp8(q, kv_fp8, scale, w, kv_fused, cl, bt, max_ml, next_n)
    assert diff < 0.02, f"cosine diff {diff:.3e} max_abs {max_abs:.3e}"


# ─────────────────────────────────────────────────────────────────────────────
# D) Degenerate values
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode", ["zero_q", "zero_weights", "neg_weights", "large_weights", "zero_kv"]
)
def test_adv_degenerate_values_fp8(mode):
    """Degenerate inputs must not NaN and must match the reference."""
    _skip_if_not_sm100()
    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(3)
    B, next_n, block_size, H, D = 4, 2, 64, 64, 128
    ctx = 2048
    cl = torch.full((B,), ctx, dtype=torch.int32, device=DEVICE)
    max_ml = ctx + 8
    block_table, ntb = _make_paged_kv(B, block_size, cl, DEVICE)

    q_f = torch.randn(B, next_n, H, D, device=DEVICE)
    kv = torch.randn(ntb, block_size, D, device=DEVICE)
    weights = torch.randn(B * next_n, H, device=DEVICE)

    if mode == "zero_q":
        q_f.zero_()
    elif mode == "zero_weights":
        weights.zero_()
    elif mode == "neg_weights":
        weights = -weights.abs() * 5.0
    elif mode == "large_weights":
        weights = weights * 1000.0
    elif mode == "zero_kv":
        kv.zero_()

    q = q_f.to(torch.float8_e4m3fn)
    amax = kv.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
    scale = _ceil_to_ue8m0_fp(amax / 448.0).squeeze(-1)
    kv_fp8 = (kv / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    kv_fused = _make_fused_kv_fp8(kv_fp8, scale, block_size, D)

    ref = _ref_fp8_paged_mqa_logits(
        q, kv_fp8, scale, weights, cl, block_table, max_ml, block_size
    )
    out = fp8_paged_mqa_logits(q, kv_fused, weights, cl, block_table, max_ml)

    valid = _valid_causal_mask(cl, next_n, max_ml, DEVICE)
    assert torch.isfinite(out.float()[valid]).all(), (
        f"non-finite output in valid region ({mode})"
    )
    o = out.float().masked_fill(~valid, 0)
    r = ref.float().masked_fill(~valid, 0)
    fin = torch.isfinite(o) & torch.isfinite(r)
    diff = _calc_cosine_diff(o.masked_fill(~fin, 0), r.masked_fill(~fin, 0))
    # zero_weights/zero_q -> all-zero logits -> cosine is 0 by construction
    assert diff < 0.02, f"cosine diff {diff:.3e} ({mode})"


# ─────────────────────────────────────────────────────────────────────────────
# E) Determinism (required for CUDA-graph replay correctness)
# ─────────────────────────────────────────────────────────────────────────────


def test_adv_determinism_fp8():
    """Identical inputs must yield bitwise-identical output across calls."""
    _skip_if_not_sm100()
    from flashinfer import fp8_paged_mqa_logits

    cl = torch.randint(500, 4096, (8,), dtype=torch.int32, device=DEVICE)
    max_ml = 4096
    q, kv_fp8, scale, w, kv_fused, bt = _build_fp8(cl, 2, 64, seed=11)
    o1 = fp8_paged_mqa_logits(q, kv_fused, w, cl, bt, max_ml).clone()
    o2 = fp8_paged_mqa_logits(q, kv_fused, w, cl, bt, max_ml).clone()
    valid = _valid_causal_mask(cl, 2, max_ml, DEVICE)
    assert torch.equal(o1[valid], o2[valid]), (
        "kernel is non-deterministic in the valid region"
    )


# ─────────────────────────────────────────────────────────────────────────────
# F) FP4 correctness across block_size (flat SF vs online transpose)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("next_n", [1, 2, 3])
@pytest.mark.parametrize("ctx", [128, 129, 256, 257, 1024])
def test_adv_fp4_boundaries(block_size, next_n, ctx):
    """FP4: boundary contexts across all block_size (SF layout correctness)."""
    _skip_if_not_sm100()
    from flashinfer import fp4_paged_mqa_logits

    if ctx < next_n:
        pytest.skip("ctx must be >= next_n")
    torch.manual_seed(ctx + block_size)
    B, H, D = 4, 64, 128
    cl = torch.full((B,), ctx, dtype=torch.int32, device=DEVICE)
    max_ml = max(ctx + 8, 512)
    block_table, ntb = _make_paged_kv(B, block_size, cl, DEVICE)
    q_bf = torch.randn(B, next_n, H, D, device=DEVICE, dtype=torch.bfloat16)
    kv_cache = torch.randn(ntb, block_size, 1, D, device=DEVICE, dtype=torch.bfloat16)
    weights = torch.randn(B * next_n, H, device=DEVICE)
    q_pk, sf_qp = _per_token_cast_to_fp4(q_bf.view(-1, D), gran_k=32)
    q_fp4 = q_pk.view(torch.uint8).view(B, next_n, H, D // 2)
    sf_q = sf_qp.view(torch.int32).view(B, next_n, H)
    kv_fused, kv_sim = _kv_cache_cast_to_fp4(kv_cache)
    q_sim = (
        _cast_back_from_fp4(q_pk, sf_qp, gran_k=32)
        .view(B, next_n, H, D)
        .to(torch.bfloat16)
    )

    ref = _ref_fp4_paged_mqa_logits(
        q_sim.float(), kv_sim.float(), weights, cl, block_table, max_ml
    )
    out = fp4_paged_mqa_logits(
        q_fp4,
        sf_q,
        kv_fused,
        weights,
        cl,
        block_table,
        max_ml,
        output_dtype=torch.float32,
    )
    valid = _valid_causal_mask(cl, next_n, max_ml, DEVICE)
    assert torch.isfinite(out.float()[valid]).all(), (
        "non-finite FP4 output in valid region"
    )
    o = out.float().masked_fill(~valid, 0)
    r = ref.float().masked_fill(~valid, 0)
    fin = torch.isfinite(o) & torch.isfinite(r)
    diff = _calc_cosine_diff(o.masked_fill(~fin, 0), r.masked_fill(~fin, 0))
    assert diff < 0.05, f"FP4 cosine diff {diff:.3e} ctx={ctx} block_size={block_size}"


# ─────────────────────────────────────────────────────────────────────────────
# G) next_n=4 with small / boundary contexts (FP4 atom-split)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ctx", [8, 64, 128, 129, 256, 257])
def test_adv_fp4_next_n4_small_ctx(ctx):
    """FP4 next_n=4 atom-split with small/boundary contexts (ctx-2 crosses blocks)."""
    _skip_if_not_sm100()
    from flashinfer import fp4_paged_mqa_logits

    torch.manual_seed(ctx)
    B, next_n, H, D, block_size = 4, 4, 64, 128, 64
    cl = torch.full((B,), ctx, dtype=torch.int32, device=DEVICE)
    max_ml = max(ctx + 8, 512)
    block_table, ntb = _make_paged_kv(B, block_size, cl, DEVICE)
    q_bf = torch.randn(B, next_n, H, D, device=DEVICE, dtype=torch.bfloat16)
    kv_cache = torch.randn(ntb, block_size, 1, D, device=DEVICE, dtype=torch.bfloat16)
    weights = torch.randn(B * next_n, H, device=DEVICE)
    q_pk, sf_qp = _per_token_cast_to_fp4(q_bf.view(-1, D), gran_k=32)
    q_fp4 = q_pk.view(torch.uint8).view(B, next_n, H, D // 2)
    sf_q = sf_qp.view(torch.int32).view(B, next_n, H)
    kv_fused, kv_sim = _kv_cache_cast_to_fp4(kv_cache)
    q_sim = (
        _cast_back_from_fp4(q_pk, sf_qp, gran_k=32)
        .view(B, next_n, H, D)
        .to(torch.bfloat16)
    )

    ref = _ref_fp4_paged_mqa_logits(
        q_sim.float(), kv_sim.float(), weights, cl, block_table, max_ml
    )
    out = fp4_paged_mqa_logits(
        q_fp4,
        sf_q,
        kv_fused,
        weights,
        cl,
        block_table,
        max_ml,
        output_dtype=torch.float32,
    )
    valid = _valid_causal_mask(cl, next_n, max_ml, DEVICE)
    o = out.float().masked_fill(~valid, 0)
    r = ref.float().masked_fill(~valid, 0)
    fin = torch.isfinite(o) & torch.isfinite(r)
    diff = _calc_cosine_diff(o.masked_fill(~fin, 0), r.masked_fill(~fin, 0))
    assert diff < 0.05, f"FP4 next_n=4 cosine diff {diff:.3e} ctx={ctx}"


# ─────────────────────────────────────────────────────────────────────────────
# H) Shared physical blocks across rows (block_table reuse)
# ─────────────────────────────────────────────────────────────────────────────


def test_adv_shared_physical_blocks_fp8():
    """All rows point at the SAME physical blocks (KV sharing) — must be correct."""
    _skip_if_not_sm100()
    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(5)
    B, next_n, block_size, H, D = 4, 2, 64, 64, 128
    ctx = 1024
    cl = torch.full((B,), ctx, dtype=torch.int32, device=DEVICE)
    max_ml = ctx + 8
    n_blk = (ctx + block_size - 1) // block_size
    kv = torch.randn(n_blk + 2, block_size, D, device=DEVICE)
    amax = kv.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
    scale = _ceil_to_ue8m0_fp(amax / 448.0).squeeze(-1)
    kv_fp8 = (kv / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    kv_fused = _make_fused_kv_fp8(kv_fp8, scale, block_size, D)
    # Every row uses the identical block table (shared KV).
    bt = torch.arange(n_blk, dtype=torch.int32, device=DEVICE).unsqueeze(0).repeat(B, 1)
    q = torch.randn(B, next_n, H, D, device=DEVICE).to(torch.float8_e4m3fn)
    weights = torch.randn(B * next_n, H, device=DEVICE)

    ref = _ref_fp8_paged_mqa_logits(
        q, kv_fp8, scale, weights, cl, bt, max_ml, block_size
    )
    out = fp8_paged_mqa_logits(q, kv_fused, weights, cl, bt, max_ml)
    valid = _valid_causal_mask(cl, next_n, max_ml, DEVICE)
    o = out.float().masked_fill(~valid, 0)
    r = ref.float().masked_fill(~valid, 0)
    fin = torch.isfinite(o) & torch.isfinite(r)
    diff = _calc_cosine_diff(o.masked_fill(~fin, 0), r.masked_fill(~fin, 0))
    assert diff < 0.02, f"shared-block cosine diff {diff:.3e}"


# ─────────────────────────────────────────────────────────────────────────────
# I) Guard rails (must raise cleanly, never silently corrupt / crash)
# ─────────────────────────────────────────────────────────────────────────────


def test_adv_guards_raise():
    """Adversarial malformed inputs on the guarded paths must raise ValueError."""
    _skip_if_not_sm100()
    from flashinfer import padded_context_len, fp8_paged_mqa_logits

    B, next_n, block_size = 2, 1, 64
    ctx = 1024
    cl = torch.full((B,), ctx, dtype=torch.int32, device=DEVICE)
    max_ml = ctx + 300  # not a multiple of 256 -> padded > max_ml
    q, kv_fp8, scale, w, kv_fused, bt = _build_fp8(cl, next_n, block_size, seed=1)

    # Undersized out= (would OOB-write) -> ValueError
    bad = torch.empty((B * next_n, max_ml), device=DEVICE, dtype=torch.float32)
    assert padded_context_len(max_ml) > max_ml
    with pytest.raises(ValueError, match="padded_context_len"):
        fp8_paged_mqa_logits(q, kv_fused, w, cl, bt, max_ml, out=bad)

    # FP8 bf16 output unsupported
    with pytest.raises(ValueError, match="float32, float16"):
        fp8_paged_mqa_logits(
            q, kv_fused, w, cl, bt, max_ml, output_dtype=torch.bfloat16
        )

    # CPU / int64 index tensors
    with pytest.raises(ValueError, match="context_lens"):
        fp8_paged_mqa_logits(q, kv_fused, w, cl.cpu(), bt, max_ml)
    with pytest.raises(ValueError, match="block_table"):
        fp8_paged_mqa_logits(q, kv_fused, w, cl, bt.to(torch.int64), max_ml)


def test_adv_new_guards_raise():
    """Guards added after the adversarial review must raise ValueError (not OOB/crash)."""
    _skip_if_not_sm100()
    from flashinfer import (
        compute_paged_mqa_logits_schedule,
        fp4_paged_mqa_logits,
        fp8_paged_mqa_logits,
    )

    B, next_n, block_size = 4, 2, 64
    ctx = 2048
    cl = torch.full((B,), ctx, dtype=torch.int32, device=DEVICE)
    max_ml = ctx + 8
    q, kv_fp8, scale, w, kv_fused, bt = _build_fp8(cl, next_n, block_size, seed=2)

    # (#2) fp8 e5m2 q rejected
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        fp8_paged_mqa_logits(q.view(torch.float8_e5m2), kv_fused, w, cl, bt, max_ml)

    # (#3) kv_fused wrong last dim (misreads scale region)
    bad_kv = kv_fused[..., :-1].contiguous()
    with pytest.raises(ValueError, match="kv_fused"):
        fp8_paged_mqa_logits(q, bad_kv, w, cl, bt, max_ml)

    # (#1) context_lens / block_table row count != B
    with pytest.raises(ValueError, match="batch_size"):
        fp8_paged_mqa_logits(q, kv_fused, w, cl[:-1], bt, max_ml)
    with pytest.raises(ValueError, match="batch_size"):
        fp8_paged_mqa_logits(q, kv_fused, w, cl, bt[:-1].contiguous(), max_ml)

    # (#4) caller schedule_meta of the wrong size
    bad_sched = torch.zeros((4, 2), dtype=torch.int32, device=DEVICE)
    with pytest.raises(ValueError, match="schedule_meta"):
        fp8_paged_mqa_logits(q, kv_fused, w, cl, bt, max_ml, schedule_meta=bad_sched)
    # correct schedule_meta passes the guard
    good = compute_paged_mqa_logits_schedule(cl)
    fp8_paged_mqa_logits(q, kv_fused, w, cl, bt, max_ml, schedule_meta=good)

    # (#6/#9) num_epi_subtiles < 1
    with pytest.raises(ValueError, match="num_epi_subtiles"):
        fp8_paged_mqa_logits(q, kv_fused, w, cl, bt, max_ml, num_epi_subtiles=0)

    # FP4-specific guards: q must be uint8, sf_q must be int32 [B, next_n, H]
    q_bf = torch.randn(B, next_n, 64, 128, device=DEVICE, dtype=torch.bfloat16)
    q_pk, sf_qp = _per_token_cast_to_fp4(q_bf.view(-1, 128), gran_k=32)
    q_fp4 = q_pk.view(torch.uint8).view(B, next_n, 64, 64)
    sf_q = sf_qp.view(torch.int32).view(B, next_n, 64)
    kv_cache = torch.randn(
        kv_fp8.shape[0], block_size, 1, 128, device=DEVICE, dtype=torch.bfloat16
    )
    kv_fused4, _ = _kv_cache_cast_to_fp4(kv_cache)
    with pytest.raises(ValueError, match="uint8"):
        fp4_paged_mqa_logits(q_fp4.float(), sf_q, kv_fused4, w, cl, bt, max_ml)
    with pytest.raises(ValueError, match="sf_q"):
        fp4_paged_mqa_logits(
            q_fp4, sf_q[..., :-1].contiguous(), kv_fused4, w, cl, bt, max_ml
        )
