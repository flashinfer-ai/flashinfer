"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for Minimax Sparse Attention (MSA) operations on SM120/SM121.
"""

import math

import pytest
import torch

from flashinfer.utils import is_sm12x_supported

BLK_KV = 128


def _skip_if_unsupported():
    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("MSA ops require SM120 or SM121 and CUDA >= 12.8")


# ---------------------------------------------------------------------------
# msa_sparse_attention (sparse prefill)
# ---------------------------------------------------------------------------


def _ref_sparse_attention(q, k, v, idx, cu_q, cu_k, causal, scale):
    total_q, Hq, _ = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    out = torch.zeros_like(q, dtype=torch.float32)
    B = cu_q.numel() - 1
    for b in range(B):
        q_lo, q_hi = int(cu_q[b]), int(cu_q[b + 1])
        k_lo, k_hi = int(cu_k[b]), int(cu_k[b + 1])
        seqlen_k, seqlen_q = k_hi - k_lo, q_hi - q_lo
        nb = (seqlen_k + BLK_KV - 1) // BLK_KV
        for qi in range(q_lo, q_hi):
            q_pos = qi - q_lo
            for hq in range(Hq):
                hkv = hq // G
                sel = idx[hkv, qi]
                sel = sel[(sel >= 0) & (sel < nb)].unique()
                cols = []
                for blk in sel.tolist():
                    lo = blk * BLK_KV
                    hi = min(lo + BLK_KV, seqlen_k)
                    cols.extend(range(lo, hi))
                if causal:
                    limit = q_pos + seqlen_k - seqlen_q
                    cols = [c for c in cols if c <= limit]
                if not cols:
                    continue
                kk = k[k_lo + torch.tensor(cols), hkv].float()
                vv = v[k_lo + torch.tensor(cols), hkv].float()
                p = torch.softmax((q[qi, hq].float() @ kk.T) * scale, dim=-1)
                out[qi, hq] = p @ vv
    return out


@pytest.mark.parametrize(
    "B,Hq,Hkv,topk,seqs_q,seqs_k,causal",
    [
        (1, 1, 1, 4, [70], [1000], False),
        (2, 4, 2, 16, [100, 37], [2048, 700], False),
        (2, 2, 2, 16, [130, 64], [1024, 512], True),
        (1, 8, 2, 8, [200], [4096], False),
        (1, 16, 1, 16, [333], [3000], False),
        (2, 8, 8, 16, [97, 211], [1111, 2222], True),
    ],
)
def test_sparse_attention(B, Hq, Hkv, topk, seqs_q, seqs_k, causal):
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_attention

    torch.manual_seed(7)
    dev, dtype = "cuda", torch.bfloat16
    cu_q = torch.tensor(
        [0] + list(torch.tensor(seqs_q).cumsum(0)), dtype=torch.int32, device=dev
    )
    cu_k = torch.tensor(
        [0] + list(torch.tensor(seqs_k).cumsum(0)), dtype=torch.int32, device=dev
    )
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(total_q, Hq, 128, dtype=dtype, device=dev) / 3
    k = torch.randn(total_k, Hkv, 128, dtype=dtype, device=dev) / 3
    v = torch.randn(total_k, Hkv, 128, dtype=dtype, device=dev) / 3

    idx = torch.full((Hkv, total_q, topk), -1, dtype=torch.int32, device=dev)
    for b in range(B):
        nb = (seqs_k[b] + BLK_KV - 1) // BLK_KV
        lo, hi = int(cu_q[b]), int(cu_q[b + 1])
        n = min(topk, nb)
        for h in range(Hkv):
            for qi in range(lo, hi):
                nsel = torch.randint(0, n + 1, (1,)).item()
                if nsel > 0:
                    sel = torch.randperm(nb)[:nsel].sort().values.to(torch.int32)
                    idx[h, qi, :nsel] = sel.to(dev)

    scale = 1.0 / math.sqrt(128)
    out = msa_sparse_attention(
        q, k, v, idx, cu_q, cu_k, causal=causal, softmax_scale=scale
    )
    torch.cuda.synchronize()
    ref = _ref_sparse_attention(
        q.cpu(), k.cpu(), v.cpu(), idx.cpu(), cu_q.cpu(), cu_k.cpu(), causal, scale
    )
    err = (out.float().cpu() - ref).abs().max().item()
    assert err < 2.5e-2, f"max abs error {err}"


# ---------------------------------------------------------------------------
# paged KV, LSE output, fused combine, fp16/topk32 coverage
# ---------------------------------------------------------------------------


def _make_case(B, Hq, Hkv, topk, seqs_q, seqs_k, dtype, seed, min_sel=0):
    torch.manual_seed(seed)
    dev = "cuda"
    cu_q = torch.tensor(
        [0] + list(torch.tensor(seqs_q).cumsum(0)), dtype=torch.int32, device=dev
    )
    cu_k = torch.tensor(
        [0] + list(torch.tensor(seqs_k).cumsum(0)), dtype=torch.int32, device=dev
    )
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(total_q, Hq, 128, dtype=dtype, device=dev) / 3
    k = torch.randn(total_k, Hkv, 128, dtype=dtype, device=dev) / 3
    v = torch.randn(total_k, Hkv, 128, dtype=dtype, device=dev) / 3
    idx = torch.full((Hkv, total_q, topk), -1, dtype=torch.int32, device=dev)
    for b in range(B):
        nb = (seqs_k[b] + BLK_KV - 1) // BLK_KV
        lo, hi = int(cu_q[b]), int(cu_q[b + 1])
        n = min(topk, nb)
        for h in range(Hkv):
            for qi in range(lo, hi):
                nsel = torch.randint(min_sel, n + 1, (1,)).item()
                if nsel > 0:
                    sel = torch.randperm(nb)[:nsel].sort().values.to(torch.int32)
                    idx[h, qi, :nsel] = sel.to(dev)
    return q, k, v, idx, cu_q, cu_k


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [16, 32])
def test_sparse_attention_dtypes_topk(dtype, topk):
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_attention

    q, k, v, idx, cu_q, cu_k = _make_case(
        2, 4, 2, topk, [100, 64], [2048, 1024], dtype, seed=20 + topk
    )
    scale = 1.0 / math.sqrt(128)
    out = msa_sparse_attention(q, k, v, idx, cu_q, cu_k, softmax_scale=scale)
    torch.cuda.synchronize()
    ref = _ref_sparse_attention(
        q.cpu(), k.cpu(), v.cpu(), idx.cpu(), cu_q.cpu(), cu_k.cpu(), False, scale
    )
    err = (out.float().cpu() - ref).abs().max().item()
    assert err < 3.5e-2, f"max abs error {err}"


def test_sparse_attention_lse():
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_attention

    q, k, v, idx, cu_q, cu_k = _make_case(
        2, 4, 2, 16, [80, 50], [1024, 640], torch.bfloat16, seed=30
    )
    scale = 1.0 / math.sqrt(128)
    _, lse = msa_sparse_attention(
        q, k, v, idx, cu_q, cu_k, softmax_scale=scale, return_softmax_lse=True
    )
    torch.cuda.synchronize()
    Hq, Hkv = q.shape[1], k.shape[1]
    G = Hq // Hkv
    seqs_k = [1024, 640]
    for _ in range(40):
        qi = torch.randint(0, q.shape[0], (1,)).item()
        hq = torch.randint(0, Hq, (1,)).item()
        b = 0 if qi < int(cu_q[1]) else 1
        nb = (seqs_k[b] + BLK_KV - 1) // BLK_KV
        sel = idx[hq // G, qi]
        sel = sel[(sel >= 0) & (sel < nb)]
        if sel.numel() == 0:
            assert lse[qi, hq].item() == float("-inf")
            continue
        cols = torch.cat(
            [torch.arange(s * BLK_KV, (s + 1) * BLK_KV) for s in sel.tolist()]
        )
        kk = k[int(cu_k[b]) + cols, hq // G].float()
        s = (q[qi, hq].float() @ kk.T) * scale
        ref_lse = torch.logsumexp(s, dim=-1).item()
        assert abs(lse[qi, hq].item() - ref_lse) < 1e-2


def test_sparse_attention_paged():
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_attention

    seqs_q, seqs_k = [150, 90], [2048, 1280]  # seqs_k multiples of 128
    q, k, v, idx, cu_q, cu_k = _make_case(
        2, 8, 2, 16, seqs_q, seqs_k, torch.bfloat16, seed=40
    )
    scale = 1.0 / math.sqrt(128)
    out_flat = msa_sparse_attention(q, k, v, idx, cu_q, cu_k, softmax_scale=scale)

    Hkv = k.shape[1]
    num_pages_per = [s // BLK_KV for s in seqs_k]
    total_pages = sum(num_pages_per)
    perm = torch.randperm(total_pages)
    k_pg = torch.zeros(total_pages, Hkv, BLK_KV, 128, dtype=k.dtype, device=k.device)
    v_pg = torch.zeros_like(k_pg)
    ptab = torch.full((2, max(num_pages_per)), -1, dtype=torch.int32, device=k.device)
    pi = 0
    for b in range(2):
        for blk in range(num_pages_per[b]):
            pg = int(perm[pi])
            pi += 1
            ptab[b, blk] = pg
            rows = slice(int(cu_k[b]) + blk * BLK_KV, int(cu_k[b]) + (blk + 1) * BLK_KV)
            k_pg[pg] = k[rows].transpose(0, 1)
            v_pg[pg] = v[rows].transpose(0, 1)
    seqused = torch.tensor(seqs_k, dtype=torch.int32, device=k.device)
    out_paged = msa_sparse_attention(
        q,
        k_pg.contiguous(),
        v_pg.contiguous(),
        idx,
        cu_q,
        page_table=ptab,
        seqused_k=seqused,
        softmax_scale=scale,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_paged, out_flat)


def _combine_partials_torch(
    o_partial, lse_partial, split_counts, group_size, out_dtype
):
    """LSE-weighted reduction over each query's split slots (torch reference
    for the CUDA combine kernel). ``lse_partial`` is in the log2 domain."""
    topk = o_partial.shape[0]
    counts = split_counts.repeat_interleave(group_size, dim=1)  # [total_q, Hq]
    slots = torch.arange(topk, device=o_partial.device).view(topk, 1, 1)
    mask = slots < counts.unsqueeze(0)  # [topk, total_q, Hq]
    neg_inf = torch.finfo(torch.float32).min
    lse = torch.where(mask, lse_partial, neg_inf)
    lse_max = lse.max(dim=0, keepdim=True).values
    w = torch.exp2(lse - lse_max)
    w = torch.where(mask, w, 0.0)
    denom = w.sum(dim=0)
    # Mask the partials too: slots >= count hold uninitialized memory, and
    # 0 * inf/NaN would poison the sum.
    o_masked = torch.where(mask.unsqueeze(-1), o_partial.float(), 0.0)
    out = (o_masked * w.unsqueeze(-1)).sum(dim=0)
    out = out / denom.unsqueeze(-1)
    out = torch.nan_to_num(out, nan=0.0)  # queries with zero valid splits
    return out.to(out_dtype)


def test_fused_combine_matches_torch():
    _skip_if_unsupported()
    import importlib

    # The msa_sparse_attention *function* shadows the submodule on attribute import.
    sa = importlib.import_module("flashinfer.msa_ops.sparse_decode")

    torch.manual_seed(50)
    dev = "cuda"
    topk, total_q, Hq, G, d = 16, 200, 8, 4, 128
    Hkv = Hq // G
    o_p = torch.randn(topk, total_q, Hq, d, dtype=torch.bfloat16, device=dev)
    lse_p = torch.randn(topk, total_q, Hq, dtype=torch.float32, device=dev) * 4
    counts = torch.randint(0, topk + 1, (total_q, Hkv), dtype=torch.int32, device=dev)
    ref = _combine_partials_torch(o_p, lse_p, counts, G, torch.bfloat16)
    got = sa._combine_partials(o_p, lse_p, counts, G, torch.bfloat16)
    torch.cuda.synchronize()
    err = (got.float() - ref.float()).abs().max().item()
    assert err < 1e-2, f"combine mismatch {err}"


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_sparse_attention_fp8_kv(q_dtype, causal):
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_attention

    torch.manual_seed(60)
    dev = "cuda"
    B, Hq, Hkv, topk = 2, 8, 2, 16
    seqs_q, seqs_k = [150, 90], [2048, 1280]
    cu_q = torch.tensor(
        [0] + list(torch.tensor(seqs_q).cumsum(0)), dtype=torch.int32, device=dev
    )
    cu_k = torch.tensor(
        [0] + list(torch.tensor(seqs_k).cumsum(0)), dtype=torch.int32, device=dev
    )
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(total_q, Hq, 128, dtype=q_dtype, device=dev) / 3
    k8 = (torch.randn(total_k, Hkv, 128, device=dev) / 3).to(torch.float8_e4m3fn)
    v8 = (torch.randn(total_k, Hkv, 128, device=dev) / 3).to(torch.float8_e4m3fn)
    idx = torch.full((Hkv, total_q, topk), -1, dtype=torch.int32, device=dev)
    for b in range(B):
        nb = seqs_k[b] // BLK_KV
        lo, hi = int(cu_q[b]), int(cu_q[b + 1])
        for h in range(Hkv):
            for qi in range(lo, hi):
                nsel = torch.randint(0, min(topk, nb) + 1, (1,)).item()
                if nsel > 0:
                    sel = torch.randperm(nb)[:nsel].sort().values.to(torch.int32)
                    idx[h, qi, :nsel] = sel.to(dev)
    scale = 1.0 / math.sqrt(128)
    out = msa_sparse_attention(
        q, k8, v8, idx, cu_q, cu_k, causal=causal, softmax_scale=scale
    )
    torch.cuda.synchronize()
    # Reference uses the dequantized K/V so the quantization error cancels.
    ref = _ref_sparse_attention(
        q.cpu(),
        k8.to(q_dtype).cpu(),
        v8.to(q_dtype).cpu(),
        idx.cpu(),
        cu_q.cpu(),
        cu_k.cpu(),
        causal,
        scale,
    )
    err = (out.float().cpu() - ref).abs().max().item()
    assert err < 3.5e-2, f"max abs error {err}"


# ---------------------------------------------------------------------------
# sparse decode
# ---------------------------------------------------------------------------


def _make_decode_case(B, sq, Hq, Hkv, topk, kv_dtype, paged, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    total_q = B * sq
    seqused = (torch.randint(2, 16, (B,), device=dev) * BLK_KV).to(torch.int32)
    cu_k = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    cu_k[1:] = seqused.cumsum(0)
    total_k = int(cu_k[-1])
    q = torch.randn(total_q, Hq, 128, dtype=torch.bfloat16, device=dev) / 3
    k_flat = (torch.randn(total_k, Hkv, 128, device=dev) / 3).to(kv_dtype)
    v_flat = (torch.randn(total_k, Hkv, 128, device=dev) / 3).to(kv_dtype)
    idx = torch.full((Hkv, total_q, topk), -1, dtype=torch.int32, device=dev)
    for b in range(B):
        nb = int(seqused[b]) // BLK_KV
        for i in range(sq):
            qi = b * sq + i
            for h in range(Hkv):
                nsel = torch.randint(0, min(topk, nb) + 1, (1,)).item()
                if nsel > 0:
                    sel = torch.randperm(nb)[:nsel].sort().values.to(torch.int32)
                    idx[h, qi, :nsel] = sel.to(dev)
    if not paged:
        return q, k_flat, v_flat, idx, seqused, cu_k, None, None
    npg = [int(s) // BLK_KV for s in seqused]
    perm = torch.randperm(sum(npg))
    k_pg = torch.zeros(sum(npg), Hkv, BLK_KV, 128, dtype=kv_dtype, device=dev)
    v_pg = torch.zeros_like(k_pg)
    ptab = torch.full((B, max(npg)), -1, dtype=torch.int32, device=dev)
    pi = 0
    for b in range(B):
        for blk in range(npg[b]):
            pg = int(perm[pi])
            pi += 1
            ptab[b, blk] = pg
            rows = slice(int(cu_k[b]) + blk * BLK_KV, int(cu_k[b]) + (blk + 1) * BLK_KV)
            k_pg[pg] = k_flat[rows].transpose(0, 1)
            v_pg[pg] = v_flat[rows].transpose(0, 1)
    return q, k_flat, v_flat, idx, seqused, cu_k, (k_pg, v_pg), ptab


@pytest.mark.parametrize(
    "B,sq,Hq,Hkv,topk,kv_dtype,paged",
    [
        (8, 1, 8, 2, 16, torch.bfloat16, False),
        (8, 1, 8, 2, 16, torch.float8_e4m3fn, True),
        (4, 2, 16, 2, 16, torch.bfloat16, True),
        (16, 1, 4, 4, 8, torch.float8_e4m3fn, True),
    ],
)
def test_sparse_decode(B, sq, Hq, Hkv, topk, kv_dtype, paged):
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_decode_attention

    q, k_flat, v_flat, idx, seqused, cu_k, pg, ptab = _make_decode_case(
        B, sq, Hq, Hkv, topk, kv_dtype, paged, seed=80 + B + sq
    )
    scale = 1.0 / math.sqrt(128)
    if paged:
        out = msa_sparse_decode_attention(
            q,
            pg[0].contiguous(),
            pg[1].contiguous(),
            idx,
            page_table=ptab,
            seqused_k=seqused,
            seqlen_q=sq,
            causal=True,
            softmax_scale=scale,
        )
    else:
        out = msa_sparse_decode_attention(
            q,
            k_flat,
            v_flat,
            idx,
            cu_seqlens_k=cu_k,
            seqlen_q=sq,
            causal=True,
            softmax_scale=scale,
        )
    torch.cuda.synchronize()
    cu_q = torch.arange(0, B * sq + 1, sq, dtype=torch.int32)
    k_ref = k_flat.to(torch.bfloat16) if kv_dtype == torch.float8_e4m3fn else k_flat
    v_ref = v_flat.to(torch.bfloat16) if kv_dtype == torch.float8_e4m3fn else v_flat
    ref = _ref_sparse_attention(
        q.cpu(), k_ref.cpu(), v_ref.cpu(), idx.cpu(), cu_q, cu_k.cpu(), True, scale
    )
    err = (out.float().cpu() - ref).abs().max().item()
    assert err < 2.5e-2, f"decode max abs error {err}"


@pytest.mark.parametrize(
    "B,sq,Hq,Hkv,topk,kv_dtype,paged",
    [
        (8, 1, 8, 2, 16, torch.bfloat16, False),
        (4, 2, 16, 2, 16, torch.bfloat16, True),
        (8, 1, 4, 4, 8, torch.float8_e4m3fn, False),
        (6, 1, 16, 1, 16, torch.float8_e4m3fn, True),
    ],
)
def test_sparse_decode_fused(B, sq, Hq, Hkv, topk, kv_dtype, paged):
    """Fused decode (force_fused=True: one CTA per token, no combine) must match
    the per-block split+combine path (output and LSE) and the torch oracle."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_decode_attention

    q, k_flat, v_flat, idx, seqused, cu_k, pg, ptab = _make_decode_case(
        B, sq, Hq, Hkv, topk, kv_dtype, paged, seed=70 + B + sq
    )
    scale = 1.0 / math.sqrt(128)

    def run(force_fused):
        kw = dict(
            seqlen_q=sq,
            causal=True,
            softmax_scale=scale,
            return_softmax_lse=True,
            force_fused=force_fused,
        )
        if paged:
            return msa_sparse_decode_attention(
                q,
                pg[0].contiguous(),
                pg[1].contiguous(),
                idx,
                page_table=ptab,
                seqused_k=seqused,
                **kw,
            )
        return msa_sparse_decode_attention(
            q, k_flat, v_flat, idx, cu_seqlens_k=cu_k, **kw
        )

    out_split, lse_split = run(False)
    out_fused, lse_fused = run(True)
    torch.cuda.synchronize()

    cu_q = torch.arange(0, B * sq + 1, sq, dtype=torch.int32)
    k_ref = k_flat.to(torch.bfloat16) if kv_dtype == torch.float8_e4m3fn else k_flat
    v_ref = v_flat.to(torch.bfloat16) if kv_dtype == torch.float8_e4m3fn else v_flat
    ref = _ref_sparse_attention(
        q.cpu(), k_ref.cpu(), v_ref.cpu(), idx.cpu(), cu_q, cu_k.cpu(), True, scale
    )
    assert (out_fused.float().cpu() - ref).abs().max().item() < 2.5e-2
    # Fused must agree with the split path to ~bf16 precision.
    assert (out_fused.float() - out_split.float()).abs().max().item() < 5e-3
    # Non-finite (empty-selection) slots must agree exactly; masking first would
    # hide a path that goes NaN/-inf where the other stays finite.
    assert torch.equal(torch.isfinite(lse_split), torch.isfinite(lse_fused))
    finite = torch.isfinite(lse_split)
    assert (lse_fused[finite] - lse_split[finite]).abs().max().item() < 5e-3


def test_sparse_decode_cuda_graph():
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_decode_attention

    q, _, _, idx, seqused, cu_k, pg, ptab = _make_decode_case(
        8, 1, 8, 2, 16, torch.float8_e4m3fn, True, seed=90
    )
    k_pg, v_pg = pg[0].contiguous(), pg[1].contiguous()
    scale = 1.0 / math.sqrt(128)
    call = lambda: msa_sparse_decode_attention(
        q,
        k_pg,
        v_pg,
        idx,
        page_table=ptab,
        seqused_k=seqused,
        seqlen_q=1,
        causal=True,
        softmax_scale=scale,
    )
    call()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = call()
    torch.manual_seed(91)
    q.copy_(torch.randn_like(q) / 3)
    g.replay()
    torch.cuda.synchronize()
    fresh = call()
    torch.cuda.synchronize()
    assert torch.equal(out, fresh)


# ---------------------------------------------------------------------------
# NVFP4 KV cache
# ---------------------------------------------------------------------------


def _msa_nvfp4_dequant(packed_u8, sf_u8, global_scale, rows, d):
    """Reference NVFP4 dequant: e2m1 nibbles + e4m3 block scales in the
    cuBLAS 128x4 tiled layout."""
    dev = packed_u8.device
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=dev,
    )
    lo = (packed_u8 & 0x0F).long()
    hi = (packed_u8 >> 4).long()
    vals = torch.empty(rows, d, dtype=torch.float32, device=dev)
    vals[:, 0::2] = lut[lo]
    vals[:, 1::2] = lut[hi]
    cols = d // 16
    r = torch.arange(rows, device=dev)[:, None]
    c = torch.arange(cols, device=dev)[None, :]
    off = (
        ((r // 128) * (-(-cols // 4)) + c // 4) * 512
        + (r % 128 % 32) * 16
        + (r % 128 // 32) * 4
        + c % 4
    )
    sc = (
        sf_u8.reshape(-1)[off.reshape(-1)]
        .reshape(rows, cols)
        .view(torch.float8_e4m3fn)
        .float()
    )
    return vals * sc.repeat_interleave(16, dim=1) * global_scale


def _nvfp4_quant(x2d):
    from flashinfer import nvfp4_quantize

    amax = x2d.float().abs().max()
    gsf = (448.0 * 6.0) / amax
    xq, sf = nvfp4_quantize(x2d, gsf.to(x2d.device), sf_vec_size=16)
    return xq.view(torch.uint8), sf.view(torch.uint8), float(1.0 / gsf)


def _nvfp4_dequant_kv(kq, ksf, kg, vq, vsf, vg, total_k, Hkv):
    """Dequantize packed K/V back to bf16 (total_k, Hkv, 128) for the oracle."""
    k_deq = (
        _msa_nvfp4_dequant(kq, ksf, kg, total_k * Hkv, 128)
        .reshape(total_k, Hkv, 128)
        .to(torch.bfloat16)
    )
    v_deq = (
        _msa_nvfp4_dequant(vq, vsf, vg, total_k * Hkv, 128)
        .reshape(total_k, Hkv, 128)
        .to(torch.bfloat16)
    )
    return k_deq, v_deq


@pytest.mark.parametrize("causal", [False, True])
def test_sparse_attention_nvfp4(causal):
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_attention

    torch.manual_seed(100)
    dev = "cuda"
    B, Hq, Hkv, topk = 2, 8, 2, 16
    seqs_q, seqs_k = [150, 90], [2048, 1280]
    cu_q = torch.tensor(
        [0] + list(torch.tensor(seqs_q).cumsum(0)), dtype=torch.int32, device=dev
    )
    cu_k = torch.tensor(
        [0] + list(torch.tensor(seqs_k).cumsum(0)), dtype=torch.int32, device=dev
    )
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(total_q, Hq, 128, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(total_k, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3
    v = torch.randn(total_k, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3
    idx = torch.full((Hkv, total_q, topk), -1, dtype=torch.int32, device=dev)
    for b in range(B):
        nb = seqs_k[b] // BLK_KV
        lo, hi = int(cu_q[b]), int(cu_q[b + 1])
        for h in range(Hkv):
            for qi in range(lo, hi):
                nsel = torch.randint(0, min(topk, nb) + 1, (1,)).item()
                if nsel > 0:
                    sel = torch.randperm(nb)[:nsel].sort().values.to(torch.int32)
                    idx[h, qi, :nsel] = sel.to(dev)
    scale = 1.0 / math.sqrt(128)

    kq, ksf, kg = _nvfp4_quant(k.reshape(-1, 128))
    vq, vsf, vg = _nvfp4_quant(v.reshape(-1, 128))
    out = msa_sparse_attention(
        q,
        kq.reshape(total_k, Hkv, 64),
        vq.reshape(total_k, Hkv, 64),
        idx,
        cu_q,
        cu_k,
        causal=causal,
        softmax_scale=scale,
        k_scale=ksf,
        v_scale=vsf,
        k_global_scale=kg,
        v_global_scale=vg,
    )
    torch.cuda.synchronize()
    k_deq, v_deq = _nvfp4_dequant_kv(kq, ksf, kg, vq, vsf, vg, total_k, Hkv)
    ref = _ref_sparse_attention(
        q.cpu(),
        k_deq.cpu(),
        v_deq.cpu(),
        idx.cpu(),
        cu_q.cpu(),
        cu_k.cpu(),
        causal,
        scale,
    )
    err = (out.float().cpu() - ref).abs().max().item()
    assert err < 2.5e-2, f"nvfp4 max abs error {err}"


def test_sparse_decode_nvfp4():
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_decode_attention

    torch.manual_seed(110)
    dev = "cuda"
    B, sq, Hq, Hkv, topk = 8, 1, 8, 2, 16
    total_q = B * sq
    seqused = (torch.randint(3, 16, (B,), device=dev) * BLK_KV).to(torch.int32)
    cu_k = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    cu_k[1:] = seqused.cumsum(0)
    total_k = int(cu_k[-1])
    q = torch.randn(total_q, Hq, 128, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(total_k, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3
    v = torch.randn(total_k, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3
    idx = torch.full((Hkv, total_q, topk), -1, dtype=torch.int32, device=dev)
    for b in range(B):
        nb = int(seqused[b]) // BLK_KV
        for h in range(Hkv):
            nsel = torch.randint(1, min(topk, nb) + 1, (1,)).item()
            sel = torch.randperm(nb)[:nsel].sort().values.to(torch.int32)
            idx[h, b, :nsel] = sel.to(dev)
    scale = 1.0 / math.sqrt(128)

    kq, ksf, kg = _nvfp4_quant(k.reshape(-1, 128))
    vq, vsf, vg = _nvfp4_quant(v.reshape(-1, 128))
    call = lambda: msa_sparse_decode_attention(
        q,
        kq.reshape(total_k, Hkv, 64),
        vq.reshape(total_k, Hkv, 64),
        idx,
        cu_seqlens_k=cu_k,
        seqlen_q=sq,
        causal=True,
        softmax_scale=scale,
        k_scale=ksf,
        v_scale=vsf,
        k_global_scale=kg,
        v_global_scale=vg,
    )
    out = call()
    torch.cuda.synchronize()
    k_deq, v_deq = _nvfp4_dequant_kv(kq, ksf, kg, vq, vsf, vg, total_k, Hkv)
    cu_q = torch.arange(0, total_q + 1, sq, dtype=torch.int32)
    ref = _ref_sparse_attention(
        q.cpu(), k_deq.cpu(), v_deq.cpu(), idx.cpu(), cu_q, cu_k.cpu(), True, scale
    )
    err = (out.float().cpu() - ref).abs().max().item()
    assert err < 2.5e-2, f"nvfp4 decode max abs error {err}"

    # CUDA graph capture must still work with the nvfp4 path
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_g = call()
    torch.manual_seed(111)
    q.copy_(torch.randn_like(q) / 3)
    g.replay()
    torch.cuda.synchronize()
    fresh = call()
    torch.cuda.synchronize()
    assert torch.equal(out_g, fresh)


@pytest.mark.parametrize("paged", [False, True])
def test_sparse_decode_fused_nvfp4(paged):
    """Fused decode with packed NVFP4 KV must match the split+combine path
    (output and LSE) and the dequant torch oracle."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_decode_attention

    B, sq, Hq, Hkv, topk = 8, 1, 8, 2, 16
    q, k_flat, v_flat, idx, seqused, cu_k, pg, ptab = _make_decode_case(
        B, sq, Hq, Hkv, topk, torch.bfloat16, paged, seed=115
    )
    total_q, total_k = B * sq, k_flat.shape[0]
    scale = 1.0 / math.sqrt(128)

    # The paged cache is a page permutation of the flat rows, so both share one
    # global amax and quantize identically row-for-row; the flat dequant serves
    # as the oracle for either layout.
    kq, ksf, kg = _nvfp4_quant(k_flat.reshape(-1, 128))
    vq, vsf, vg = _nvfp4_quant(v_flat.reshape(-1, 128))
    if paged:
        npages = pg[0].shape[0]
        kq_p, ksf_p, kg_p = _nvfp4_quant(pg[0].reshape(-1, 128))
        vq_p, vsf_p, vg_p = _nvfp4_quant(pg[1].reshape(-1, 128))

    def run(force_fused):
        kw = dict(
            seqlen_q=sq,
            causal=True,
            softmax_scale=scale,
            return_softmax_lse=True,
            force_fused=force_fused,
        )
        if paged:
            return msa_sparse_decode_attention(
                q,
                kq_p.reshape(npages, Hkv, BLK_KV, 64),
                vq_p.reshape(npages, Hkv, BLK_KV, 64),
                idx,
                page_table=ptab,
                seqused_k=seqused,
                k_scale=ksf_p,
                v_scale=vsf_p,
                k_global_scale=kg_p,
                v_global_scale=vg_p,
                **kw,
            )
        return msa_sparse_decode_attention(
            q,
            kq.reshape(total_k, Hkv, 64),
            vq.reshape(total_k, Hkv, 64),
            idx,
            cu_seqlens_k=cu_k,
            k_scale=ksf,
            v_scale=vsf,
            k_global_scale=kg,
            v_global_scale=vg,
            **kw,
        )

    out_split, lse_split = run(False)
    out_fused, lse_fused = run(True)
    torch.cuda.synchronize()

    k_deq, v_deq = _nvfp4_dequant_kv(kq, ksf, kg, vq, vsf, vg, total_k, Hkv)
    cu_q = torch.arange(0, total_q + 1, sq, dtype=torch.int32)
    ref = _ref_sparse_attention(
        q.cpu(), k_deq.cpu(), v_deq.cpu(), idx.cpu(), cu_q, cu_k.cpu(), True, scale
    )
    assert (out_fused.float().cpu() - ref).abs().max().item() < 2.5e-2
    # Fused must agree with the split path to ~bf16 precision.
    assert (out_fused.float() - out_split.float()).abs().max().item() < 5e-3
    # Non-finite (empty-selection) slots must agree exactly; masking first would
    # hide a path that goes NaN/-inf where the other stays finite.
    assert torch.equal(torch.isfinite(lse_split), torch.isfinite(lse_fused))
    finite = torch.isfinite(lse_split)
    assert (lse_fused[finite] - lse_split[finite]).abs().max().item() < 5e-3


def test_sparse_decode_nvfp4_intermediate_chunks(monkeypatch):
    """The split kernel must handle intermediate chunking for NVFP4
    (1 < num_chunks < topk: several dequant blocks per chunk, online-softmax
    carry across them). The default heuristic picks per-block for NVFP4, so
    pin num_chunks and check against the per-block split and the dequant
    oracle."""
    _skip_if_unsupported()
    import flashinfer.msa_ops.sparse_decode as sd

    B, sq, Hq, Hkv, topk = 8, 1, 8, 2, 16
    q, k_flat, v_flat, idx, seqused, cu_k, _, _ = _make_decode_case(
        B, sq, Hq, Hkv, topk, torch.bfloat16, False, seed=117
    )
    total_q, total_k = B * sq, k_flat.shape[0]
    scale = 1.0 / math.sqrt(128)
    kq, ksf, kg = _nvfp4_quant(k_flat.reshape(-1, 128))
    vq, vsf, vg = _nvfp4_quant(v_flat.reshape(-1, 128))

    def run(force_fused=None):
        return sd.msa_sparse_decode_attention(
            q,
            kq.reshape(total_k, Hkv, 64),
            vq.reshape(total_k, Hkv, 64),
            idx,
            cu_seqlens_k=cu_k,
            seqlen_q=sq,
            causal=True,
            softmax_scale=scale,
            k_scale=ksf,
            v_scale=vsf,
            k_global_scale=kg,
            v_global_scale=vg,
            force_fused=force_fused,
        )

    monkeypatch.setattr(sd, "_decode_num_chunks", lambda *a: 4)  # 16 topk / 4 chunks
    out_mid = run()
    out_per_block = run(force_fused=False)
    torch.cuda.synchronize()

    k_deq, v_deq = _nvfp4_dequant_kv(kq, ksf, kg, vq, vsf, vg, total_k, Hkv)
    cu_q = torch.arange(0, total_q + 1, sq, dtype=torch.int32)
    ref = _ref_sparse_attention(
        q.cpu(), k_deq.cpu(), v_deq.cpu(), idx.cpu(), cu_q, cu_k.cpu(), True, scale
    )
    assert (out_mid.float().cpu() - ref).abs().max().item() < 2.5e-2
    assert (out_mid.float() - out_per_block.float()).abs().max().item() < 5e-3


def test_sparse_decode_nvfp4_scale_size_guard():
    """An undersized k_scale/v_scale (not the 128-row-padded swizzled layout)
    must be rejected before launch instead of read out of bounds."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_decode_attention

    B, sq, Hq, Hkv, topk = 4, 1, 8, 2, 16
    q, k_flat, v_flat, idx, seqused, cu_k, _, _ = _make_decode_case(
        B, sq, Hq, Hkv, topk, torch.bfloat16, False, seed=118
    )
    total_k = k_flat.shape[0]
    kq, ksf, kg = _nvfp4_quant(k_flat.reshape(-1, 128))
    vq, vsf, vg = _nvfp4_quant(v_flat.reshape(-1, 128))
    with pytest.raises(ValueError, match="swizzled scale"):
        msa_sparse_decode_attention(
            q,
            kq.reshape(total_k, Hkv, 64),
            vq.reshape(total_k, Hkv, 64),
            idx,
            cu_seqlens_k=cu_k,
            seqlen_q=sq,
            k_scale=ksf.reshape(-1)[:-64],
            v_scale=vsf,
            k_global_scale=kg,
            v_global_scale=vg,
        )


# ---------------------------------------------------------------------------
# top-k selection and edge-case coverage
# ---------------------------------------------------------------------------


# P=128 exercises the count-rank kernel, P=256 the radix kernel (crossover at 128).
@pytest.mark.parametrize("P", [128, 256])
def test_msa_topk_select_forced_and_clamped(P):
    """Forced begin/end blocks are always selected within the topk budget;
    num_valid_pages clamps the candidate range."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_topk_select

    torch.manual_seed(120)
    dev = "cuda"
    H, S = 2, 64
    topk, nvp, fb, fe = 16, P - 56, 3, 2
    max_score = torch.randn(H, P, S, dtype=torch.float32, device=dev)
    max_score[:, nvp:, :] = float("-inf")
    # Give the forced regions the WORST scores: they must still be selected.
    max_score[:, :fb, :] = -1e10
    max_score[:, nvp - fe : nvp, :] = -1e10

    out = msa_topk_select(
        max_score,
        topk,
        num_valid_pages=nvp,
        force_begin_blocks=fb,
        force_end_blocks=fe,
    )
    torch.cuda.synchronize()
    forced = set(range(fb)) | set(range(nvp - fe, nvp))
    for h in range(H):
        for qi in range(S):
            row = out[qi, h]
            valid = row[row >= 0]
            assert valid.numel() == topk
            assert (valid.diff() > 0).all(), "indices must be ascending"
            assert (valid < nvp).all(), "num_valid_pages clamp violated"
            sel = set(valid.tolist())
            assert forced <= sel, f"forced blocks missing: q={qi} h={h}"
            # The rest must be the top-(topk - forced) of the middle region.
            rest = sorted(sel - forced)
            mid_scores = max_score[h, fb : nvp - fe, qi]
            expect = torch.topk(mid_scores, topk - fb - fe).indices + fb
            assert rest == sorted(expect.tolist()), f"q={qi} h={h}"

    # Clamping alone, no forced blocks.
    out2 = msa_topk_select(max_score, topk, num_valid_pages=nvp)
    torch.cuda.synchronize()
    v = out2[out2 >= 0]
    assert (v < nvp).all()


def test_msa_topk_select_large_max_k_tiles():
    """Radix top-k at large max_k_tiles (16384 blocks = 2M-token context),
    including scores that share the top key bits so every radix stage's
    threshold bin overflows the staging buffer and falls through to the next.
    Exact ties make index sets ambiguous, so compare the selected VALUES; the
    kernel drops key bits 0-1, so allow 3 ulp per value."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_topk_select

    torch.manual_seed(140)
    dev = "cuda"
    H, P, S, topk = 2, 16384, 32, 16
    random_scores = torch.randn(H, P, S, dtype=torch.float32, device=dev)
    # All values in [1.0, 1.0 + 2^-13): identical key bits 31..12, so all P
    # items land in one bin at stage 1 AND stage 2.
    tied_scores = 1.0 + torch.rand(H, P, S, device=dev) * (2.0**-13)
    for score in (random_scores, tied_scores):
        out = msa_topk_select(score, topk)
        torch.cuda.synchronize()
        assert (out >= 0).all() and (out < P).all()
        assert (out.diff(dim=-1) > 0).all(), "indices must be ascending"
        ref_vals = torch.topk(score.permute(2, 0, 1), topk, dim=-1).values
        for q in range(0, S, 5):
            for h in range(H):
                got = score[h, out[q, h].long(), q].sort(descending=True).values
                ulp = (got.view(torch.int32) - ref_vals[q, h].view(torch.int32)).abs()
                assert ulp.max().item() <= 3, f"q={q} h={h} ulp {ulp.max().item()}"


def test_msa_topk_select_input_guards():
    """Out-of-range num_valid_pages and oversized forced regions must be rejected
    (the radix path does not clamp internally, so the wrapper validates)."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_topk_select

    dev = "cuda"
    H, P, S, topk = 2, 64, 8, 16
    max_score = torch.randn(H, P, S, dtype=torch.float32, device=dev)
    with pytest.raises(ValueError, match="num_valid_pages"):
        msa_topk_select(max_score, topk, num_valid_pages=P + 1)
    with pytest.raises(ValueError, match="num_valid_pages"):
        msa_topk_select(max_score, topk, num_valid_pages=0)
    with pytest.raises(ValueError, match="topk"):
        msa_topk_select(max_score, topk, force_begin_blocks=10, force_end_blocks=10)
    with pytest.raises(ValueError, match="num_valid_pages"):
        msa_topk_select(
            max_score, topk, num_valid_pages=4, force_begin_blocks=3, force_end_blocks=3
        )


def test_q2k_indices_must_be_contiguous():
    """Strided q2k (the natural bare permute of msa_topk_select's output) must
    be rejected with a clear message."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import (
        msa_sparse_attention,
        msa_sparse_decode_attention,
    )

    dev, dtype = "cuda", torch.bfloat16
    B, sq, Hq, Hkv, topk = 2, 1, 4, 2, 8
    cu_k = torch.tensor([0, 512, 1024], dtype=torch.int32, device=dev)
    total_q, total_k = B * sq, 1024
    q = torch.randn(total_q, Hq, 128, dtype=dtype, device=dev)
    k = torch.randn(total_k, Hkv, 128, dtype=dtype, device=dev)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 1, 2], dtype=torch.int32, device=dev)
    # (total_q, Hkv, topk) permuted to (Hkv, total_q, topk): right shape, strided
    idx_strided = torch.zeros(
        total_q, Hkv, topk, dtype=torch.int32, device=dev
    ).permute(1, 0, 2)
    assert not idx_strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        msa_sparse_decode_attention(
            q, k, v, idx_strided, cu_seqlens_k=cu_k, seqlen_q=sq, causal=True
        )
    with pytest.raises(ValueError, match="contiguous"):
        msa_sparse_attention(q, k, v, idx_strided, cu_q, cu_k, causal=True)


def test_fuzz_random_shapes():
    _skip_if_unsupported()
    import random

    from flashinfer.msa_ops import msa_sparse_attention

    rng = random.Random(2026)
    dev, dtype = "cuda", torch.bfloat16
    for it in range(12):
        B = rng.randint(1, 3)
        Hkv = rng.choice([1, 2])
        G = rng.choice([1, 2, 4, 8])
        Hq = Hkv * G
        topk = rng.choice([8, 16])
        causal = rng.random() < 0.5
        seqs_q = [rng.randint(1, 80) for _ in range(B)]
        seqs_k = [rng.randint(150, 2304) for _ in range(B)]
        torch.manual_seed(1000 + it)
        cu_q = torch.tensor(
            [0] + list(torch.tensor(seqs_q).cumsum(0)), dtype=torch.int32, device=dev
        )
        cu_k = torch.tensor(
            [0] + list(torch.tensor(seqs_k).cumsum(0)), dtype=torch.int32, device=dev
        )
        total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
        q = torch.randn(total_q, Hq, 128, dtype=dtype, device=dev) / 3
        k = torch.randn(total_k, Hkv, 128, dtype=dtype, device=dev) / 3
        v = torch.randn(total_k, Hkv, 128, dtype=dtype, device=dev) / 3
        idx = torch.full((Hkv, total_q, topk), -1, dtype=torch.int32, device=dev)
        for b in range(B):
            nb = (seqs_k[b] + BLK_KV - 1) // BLK_KV
            lo, hi = int(cu_q[b]), int(cu_q[b + 1])
            for h in range(Hkv):
                for qi in range(lo, hi):
                    nsel = rng.randint(0, min(topk, nb))
                    if nsel > 0:
                        sel = torch.randperm(nb)[:nsel].sort().values
                        idx[h, qi, :nsel] = sel.to(torch.int32).to(dev)
        scale = 1.0 / math.sqrt(128)
        ref = _ref_sparse_attention(
            q.cpu(), k.cpu(), v.cpu(), idx.cpu(), cu_q.cpu(), cu_k.cpu(), causal, scale
        )
        tag = f"it={it} B={B} Hq={Hq} Hkv={Hkv} topk={topk} causal={causal}"
        out = msa_sparse_attention(
            q, k, v, idx, cu_q, cu_k, causal=causal, softmax_scale=scale
        )
        torch.cuda.synchronize()
        assert torch.isfinite(out.float()).all(), f"NaN/Inf {tag}"
        err = (out.float().cpu() - ref).abs().max().item()
        assert err < 2.5e-2, f"{tag}: err={err}"


def test_fully_masked_selected_blocks():
    """Queries whose entire selection is above the causal diagonal must produce
    exact zeros and -inf LSE, with no NaN/Inf anywhere."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_attention

    torch.manual_seed(130)
    dev, dtype = "cuda", torch.bfloat16
    Hq, Hkv, topk = 2, 1, 16
    seqlen = 1024  # seqlen_q == seqlen_k -> q_loc i attends k <= i
    nb = seqlen // BLK_KV
    cu_q = torch.tensor([0, seqlen], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, seqlen], dtype=torch.int32, device=dev)
    q = torch.randn(seqlen, Hq, 128, dtype=dtype, device=dev) / 3
    k = torch.randn(seqlen, Hkv, 128, dtype=dtype, device=dev) / 3
    v = torch.randn(seqlen, Hkv, 128, dtype=dtype, device=dev) / 3
    idx = torch.full((Hkv, seqlen, topk), -1, dtype=torch.int32, device=dev)
    # First 64 tokens select ONLY the last two blocks -> fully causally masked.
    idx[0, :64, 0] = nb - 2
    idx[0, :64, 1] = nb - 1
    # Remaining tokens select a mix (some masked, some visible).
    for qi in range(64, seqlen):
        nsel = min((qi % topk) + 1, nb)
        sel = torch.randperm(nb)[:nsel].sort().values
        idx[0, qi, :nsel] = sel.to(torch.int32).to(dev)
    scale = 1.0 / math.sqrt(128)

    out_kv, lse = msa_sparse_attention(
        q,
        k,
        v,
        idx,
        cu_q,
        cu_k,
        causal=True,
        softmax_scale=scale,
        return_softmax_lse=True,
    )
    torch.cuda.synchronize()
    for name, out in [("prefill", out_kv)]:
        assert torch.isfinite(out.float()).all(), f"NaN/Inf in {name}"
        assert (out[:64].float() == 0).all(), f"{name}: masked rows must be zero"
    assert (lse[:64] == float("-inf")).all(), "LSE of masked rows must be -inf"
    # Rows 64+ select >=1 block, but some selections may be fully causally masked
    # (-> LSE -inf). The real invariant is "no NaN": LSE must be finite or -inf.
    assert not torch.isnan(lse[64:]).any(), "LSE must be finite or -inf, never NaN"
    ref = _ref_sparse_attention(
        q.cpu(), k.cpu(), v.cpu(), idx.cpu(), cu_q.cpu(), cu_k.cpu(), True, scale
    )
    for name, out in [("prefill", out_kv)]:
        err = (out.float().cpu() - ref).abs().max().item()
        assert err < 2.5e-2, f"{name}: err={err}"


# ---------------------------------------------------------------------------
# msa_proxy_score (dense proxy)
# ---------------------------------------------------------------------------


def _ref_proxy_score(q, k, cu_q, cu_k, causal, mkt):
    total_q, Hq, _ = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    out = torch.full((Hq, mkt, total_q), float("-inf"), dtype=torch.float32)
    B = cu_q.numel() - 1
    for b in range(B):
        qlo, qhi = int(cu_q[b]), int(cu_q[b + 1])
        klo, khi = int(cu_k[b]), int(cu_k[b + 1])
        sq, sk = qhi - qlo, khi - klo
        nb = (sk + BLK_KV - 1) // BLK_KV
        for h in range(Hq):
            s = q[qlo:qhi, h].float() @ k[klo:khi, h // G].float().T  # unscaled
            if causal:
                qi = torch.arange(sq).unsqueeze(1) + (sk - sq)
                ki = torch.arange(sk).unsqueeze(0)
                s = s.masked_fill(ki > qi, float("-inf"))
            for t in range(nb):
                out[h, t, qlo:qhi] = s[:, t * BLK_KV : (t + 1) * BLK_KV].amax(dim=1)
    return out


@pytest.mark.parametrize(
    "B,Hq,Hkv,seqs_q,seqs_k,causal",
    [
        (2, 4, 2, [100, 37], [2048, 700], True),
        (2, 2, 2, [130, 64], [1024, 512], False),
        (3, 2, 1, [1, 33, 7], [300, 1111, 256], True),
    ],
)
def test_msa_proxy_score(B, Hq, Hkv, seqs_q, seqs_k, causal):
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_proxy_score

    torch.manual_seed(150 + B)
    dev = "cuda"
    cu_q = torch.tensor(
        [0] + list(torch.tensor(seqs_q).cumsum(0)), dtype=torch.int32, device=dev
    )
    cu_k = torch.tensor(
        [0] + list(torch.tensor(seqs_k).cumsum(0)), dtype=torch.int32, device=dev
    )
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(total_q, Hq, 128, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(total_k, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3
    out = msa_proxy_score(q, k, cu_q, cu_k, causal=causal)
    torch.cuda.synchronize()
    ref = _ref_proxy_score(
        q.cpu(), k.cpu(), cu_q.cpu(), cu_k.cpu(), causal, out.shape[1]
    )
    got = out.cpu()
    assert ((got == float("-inf")) == (ref == float("-inf"))).all(), "-inf pattern"
    fin = ref != float("-inf")
    if fin.any():
        assert (got[fin] - ref[fin]).abs().max().item() < 1e-2


@pytest.mark.parametrize(
    "B,Hq,Hkv,seqlen_q,seqlen_k,causal",
    [
        (4, 4, 1, 1, 8192, True),  # group 4, q_len 1 -> packed (4 x 16 tile)
        (2, 4, 1, 16, 4096, True),  # group 4, q_len at the gate edge (16)
        (3, 8, 2, 8, 2048, True),  # group 4 (Hq/Hkv), q_len 8
        (2, 4, 1, 16, 4096, False),  # non-causal packed
    ],
)
def test_msa_proxy_score_decode_packed(B, Hq, Hkv, seqlen_q, seqlen_k, causal):
    """Short-q decode dispatches the head-fused packed bf16 kernel; group 4 with
    q_len <= 16 is the MiniMax-M3 indexer shape."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_proxy_score

    torch.manual_seed(170 + B)
    dev = "cuda"
    # max_seqlen_q == seqlen_q so the packed decode gate fires
    cu_q = torch.arange(0, (B + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=dev)
    cu_k = torch.arange(0, (B + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=dev)
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(total_q, Hq, 128, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(total_k, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3
    out = msa_proxy_score(q, k, cu_q, cu_k, causal=causal)
    torch.cuda.synchronize()
    ref = _ref_proxy_score(
        q.cpu(), k.cpu(), cu_q.cpu(), cu_k.cpu(), causal, out.shape[1]
    )
    got = out.cpu()
    assert ((got == float("-inf")) == (ref == float("-inf"))).all(), "-inf pattern"
    fin = ref != float("-inf")
    if fin.any():
        assert (got[fin] - ref[fin]).abs().max().item() < 1e-2


@pytest.mark.parametrize(
    "Hq,Hkv,paged,explicit_qoff",
    [
        (4, 1, False, False),  # M3 shape, ragged flat
        (8, 2, False, False),  # multi kv-head
        (1, 1, False, False),  # group 1 (below the packed gate)
        (8, 1, True, False),  # group 8 (stream upper gate), paged
        (4, 1, False, True),  # explicit q_offset masks mid-sequence
        (4, 1, True, True),
    ],
)
def test_msa_proxy_score_decode_stream(Hq, Hkv, paged, explicit_qoff):
    """Single-token decode dispatches the stream kernel; cover ragged varlen
    with non-128 tails, an empty sequence, multi kv-head, group sizes 1-8,
    paged KV, and an explicit causal q_offset."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_proxy_score

    torch.manual_seed(190 + Hq + Hkv)
    dev = "cuda"
    seqs_k = [700, 2048, 129, 0, 4096]
    B = len(seqs_k)
    cu_k = torch.tensor(
        [0] + list(torch.tensor(seqs_k).cumsum(0)), dtype=torch.int32, device=dev
    )
    cu_q = torch.arange(B + 1, dtype=torch.int32, device=dev)
    total_k = int(cu_k[-1])
    q = torch.randn(B, Hq, 128, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(total_k, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3
    qoff = None
    if explicit_qoff:
        # Positions strictly inside each sequence so the causal limit bites.
        qoff = torch.tensor(
            [max(0, s // 2) for s in seqs_k], dtype=torch.int32, device=dev
        )

    if paged:
        npg = [-(-s // BLK_KV) for s in seqs_k]
        perm = torch.randperm(sum(npg))
        k_pg = torch.zeros(sum(npg), Hkv, BLK_KV, 128, dtype=torch.bfloat16, device=dev)
        ptab = torch.full((B, max(max(npg), 1)), -1, dtype=torch.int32, device=dev)
        pi = 0
        for b in range(B):
            for blk in range(npg[b]):
                pg = int(perm[pi])
                pi += 1
                ptab[b, blk] = pg
                lo = int(cu_k[b]) + blk * BLK_KV
                hi = min(lo + BLK_KV, int(cu_k[b + 1]))
                k_pg[pg, :, : hi - lo] = k[lo:hi].transpose(0, 1)
        seqused = torch.tensor(seqs_k, dtype=torch.int32, device=dev)
        out = msa_proxy_score(
            q,
            k_pg,
            cu_q,
            page_table=ptab,
            seqused_k=seqused,
            causal=True,
            q_offset=qoff,
        )
    else:
        out = msa_proxy_score(q, k, cu_q, cu_k, causal=True, q_offset=qoff)
    torch.cuda.synchronize()

    mkt = out.shape[1]
    G = Hq // Hkv
    ref = torch.full((Hq, mkt, B), float("-inf"), dtype=torch.float32)
    for b in range(B):
        sk = seqs_k[b]
        if sk == 0:
            continue
        lim = min((int(qoff[b]) if qoff is not None else sk - 1) + 1, sk)
        for h in range(Hq):
            kb = k[int(cu_k[b]) : int(cu_k[b + 1]), h // G].float().cpu()
            s = q[b, h].float().cpu() @ kb.T
            s[lim:] = float("-inf")
            for t in range(-(-sk // BLK_KV)):
                blk = s[t * BLK_KV : (t + 1) * BLK_KV]
                ref[h, t, b] = blk.amax()
    got = out.cpu()
    assert ((got == float("-inf")) == (ref == float("-inf"))).all(), "-inf pattern"
    fin = ref != float("-inf")
    assert (got[fin] - ref[fin]).abs().max().item() < 1e-2


@pytest.mark.parametrize("Hq,Hkv", [(4, 1), (8, 2)])
def test_msa_proxy_score_reduce_heads(Hq, Hkv):
    """reduce_heads=True must equal amax(dim=0) over the per-head output
    bit-exactly, and leave the default per-head path unchanged."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_proxy_score

    torch.manual_seed(151)
    dev = "cuda"
    total_q, total_k = 300, 4096
    cu_q = torch.tensor([0, total_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, total_k], dtype=torch.int32, device=dev)
    q = torch.randn(total_q, Hq, 128, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(total_k, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3

    per_head = msa_proxy_score(q, k, cu_q, cu_k, causal=True)
    reduced = msa_proxy_score(q, k, cu_q, cu_k, causal=True, reduce_heads=True)
    torch.cuda.synchronize()

    assert per_head.shape == (Hq, per_head.shape[1], total_q)
    assert reduced.shape == (1, per_head.shape[1], total_q)
    assert torch.equal(reduced, per_head.amax(dim=0, keepdim=True))

    # A caller-provided output of the reduced shape is honored.
    out = torch.empty_like(reduced)
    ret = msa_proxy_score(q, k, cu_q, cu_k, causal=True, reduce_heads=True, output=out)
    torch.cuda.synchronize()
    assert ret.data_ptr() == out.data_ptr()
    assert torch.equal(ret, reduced)


def test_msa_proxy_score_paged_fp8():
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_proxy_score

    torch.manual_seed(160)
    dev = "cuda"
    B, Hq, Hkv = 2, 4, 2
    seqs_q, seqs_k = [150, 90], [2048, 1280]
    cu_q = torch.tensor(
        [0] + list(torch.tensor(seqs_q).cumsum(0)), dtype=torch.int32, device=dev
    )
    cu_k = torch.tensor(
        [0] + list(torch.tensor(seqs_k).cumsum(0)), dtype=torch.int32, device=dev
    )
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(total_q, Hq, 128, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(total_k, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3
    out_flat = msa_proxy_score(q, k, cu_q, cu_k, causal=True)
    # fp8 == same kernel on dequantized K (bit-identical)
    k8 = k.to(torch.float8_e4m3fn)
    out_fp8 = msa_proxy_score(q, k8, cu_q, cu_k, causal=True)
    out_deq = msa_proxy_score(q, k8.to(torch.bfloat16), cu_q, cu_k, causal=True)
    torch.cuda.synchronize()
    assert torch.equal(out_fp8, out_deq)
    # Paged (permuted pages) == flat, bit-identical.
    npg = [s // BLK_KV for s in seqs_k]
    tp = sum(npg)
    perm = torch.randperm(tp)
    k_pg = torch.zeros(tp, Hkv, BLK_KV, 128, dtype=torch.bfloat16, device=dev)
    ptab = torch.full((B, max(npg)), -1, dtype=torch.int32, device=dev)
    pi = 0
    for b in range(B):
        for blk in range(npg[b]):
            pg = int(perm[pi])
            pi += 1
            ptab[b, blk] = pg
            rows = slice(int(cu_k[b]) + blk * BLK_KV, int(cu_k[b]) + (blk + 1) * BLK_KV)
            k_pg[pg] = k[rows].transpose(0, 1)
    seqused = torch.tensor(seqs_k, dtype=torch.int32, device=dev)
    out_paged = msa_proxy_score(
        q, k_pg.contiguous(), cu_q, page_table=ptab, seqused_k=seqused, causal=True
    )
    torch.cuda.synchronize()
    assert torch.equal(out_paged, out_flat)


def test_e2e_full_pipeline_from_raw_tensors():
    """Full pipeline: proxy scores -> top-k selection -> sparse prefill."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import (
        msa_proxy_score,
        msa_sparse_attention,
        msa_topk_select,
    )

    torch.manual_seed(170)
    dev = "cuda"
    Hq, Hkv, topk = 8, 2, 16
    seqlen_q, seqlen_k = 200, 4096
    cu_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=dev)
    q = torch.randn(seqlen_q, Hq, 128, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(seqlen_k, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3
    v = torch.randn(seqlen_k, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3
    # Proxy Q with one head per KV head.
    proxy_q = torch.randn(seqlen_q, Hkv, 128, dtype=torch.bfloat16, device=dev) / 3

    max_score = msa_proxy_score(proxy_q, k, cu_q, cu_k, causal=True)
    idx_qmajor = msa_topk_select(max_score.contiguous(), topk)
    idx = idx_qmajor.permute(1, 0, 2).contiguous()  # (Hkv, total_q, topk)
    torch.cuda.synchronize()
    # Causal proxy => selected blocks never start beyond the query position.
    for qi in range(0, seqlen_q, 37):
        q_pos = qi + seqlen_k - seqlen_q
        sel = idx[:, qi][idx[:, qi] >= 0]
        assert (sel * BLK_KV <= q_pos).all(), "selected fully-masked block"
    scale = 1.0 / math.sqrt(128)
    out = msa_sparse_attention(
        q, k, v, idx, cu_q, cu_k, causal=True, softmax_scale=scale
    )
    torch.cuda.synchronize()
    ref = _ref_sparse_attention(
        q.cpu(), k.cpu(), v.cpu(), idx.cpu(), cu_q.cpu(), cu_k.cpu(), True, scale
    )
    err = (out.float().cpu() - ref).abs().max().item()
    assert err < 2.5e-2, f"e2e pipeline max abs error {err}"


def _ref_sparse_attention_qoff(q, k, v, idx, cu_q, cu_k, scale, q_offsets):
    """Reference with explicit per-batch causal offsets (MSA q_offset)."""
    total_q, Hq, _ = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    out = torch.zeros_like(q, dtype=torch.float32)
    for b in range(cu_q.numel() - 1):
        q_lo, q_hi = int(cu_q[b]), int(cu_q[b + 1])
        k_lo, k_hi = int(cu_k[b]), int(cu_k[b + 1])
        seqlen_k = k_hi - k_lo
        nb = (seqlen_k + BLK_KV - 1) // BLK_KV
        for qi in range(q_lo, q_hi):
            limit = (qi - q_lo) + int(q_offsets[b])
            for hq in range(Hq):
                sel = idx[hq // G, qi]
                sel = sel[(sel >= 0) & (sel < nb)]
                cols = [
                    c
                    for blk in sel.tolist()
                    for c in range(blk * BLK_KV, min((blk + 1) * BLK_KV, seqlen_k))
                    if c <= limit
                ]
                if not cols:
                    continue
                kk = k[k_lo + torch.tensor(cols), hq // G].float()
                vv = v[k_lo + torch.tensor(cols), hq // G].float()
                p = torch.softmax((q[qi, hq].float() @ kk.T) * scale, dim=-1)
                out[qi, hq] = p @ vv
    return out


def test_q_offset_override():
    """Queries positioned mid-sequence (MSA q_offset): many selected blocks
    end up partially or fully above the causal diagonal."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import (
        msa_proxy_score,
        msa_sparse_attention,
    )

    torch.manual_seed(180)
    dev, dtype = "cuda", torch.bfloat16
    B, Hq, Hkv, topk = 2, 4, 2, 16
    seqs_q, seqs_k = [256, 256], [8192, 4096]
    q_offsets = torch.tensor([256, 128], dtype=torch.int32, device=dev)
    cu_q = torch.tensor(
        [0] + list(torch.tensor(seqs_q).cumsum(0)), dtype=torch.int32, device=dev
    )
    cu_k = torch.tensor(
        [0] + list(torch.tensor(seqs_k).cumsum(0)), dtype=torch.int32, device=dev
    )
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(total_q, Hq, 128, dtype=dtype, device=dev) / 3
    k = torch.randn(total_k, Hkv, 128, dtype=dtype, device=dev) / 3
    v = torch.randn(total_k, Hkv, 128, dtype=dtype, device=dev) / 3
    idx = torch.full((Hkv, total_q, topk), -1, dtype=torch.int32, device=dev)
    for b in range(B):
        nb = seqs_k[b] // BLK_KV
        lo, hi = int(cu_q[b]), int(cu_q[b + 1])
        for h in range(Hkv):
            for qi in range(lo, hi):
                nsel = torch.randint(1, 9, (1,)).item()
                sel = torch.randperm(nb)[:nsel].sort().values
                idx[h, qi, :nsel] = sel.to(torch.int32).to(dev)
    scale = 1.0 / math.sqrt(128)
    ref = _ref_sparse_attention_qoff(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        idx.cpu(),
        cu_q.cpu(),
        cu_k.cpu(),
        scale,
        q_offsets.cpu(),
    )
    for name, fn in [
        ("prefill", msa_sparse_attention),
    ]:
        out = fn(
            q,
            k,
            v,
            idx,
            cu_q,
            cu_k,
            causal=True,
            softmax_scale=scale,
            q_offset=q_offsets,
        )
        torch.cuda.synchronize()
        assert torch.isfinite(out.float()).all(), name
        err = (out.float().cpu() - ref).abs().max().item()
        assert err < 2.5e-2, f"{name}: err={err}"

    # Proxy with offsets: -inf exactly where the block is fully masked.
    ms = msa_proxy_score(q, k, cu_q, cu_k, causal=True, q_offset=q_offsets)
    torch.cuda.synchronize()
    for b in range(B):
        lo = int(cu_q[b])
        limit0 = int(q_offsets[b])
        first_masked_tile = limit0 // BLK_KV + 1
        if first_masked_tile < ms.shape[1]:
            assert (ms[:, first_masked_tile:, lo] == float("-inf")).all()
        assert torch.isfinite(ms[:, : limit0 // BLK_KV, lo]).all()


def test_temperature_lse():
    """return_temperature_lse: LSE computed with the exponent multiplied by
    lse_temperature_scale (MSA semantics), merged across splits."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_attention

    torch.manual_seed(200)
    T = 0.7
    q, k, v, idx, cu_q, cu_k = _make_case(
        2, 4, 2, 16, [80, 50], [1024, 640], torch.bfloat16, seed=200
    )
    scale = 1.0 / math.sqrt(128)
    _, lse, lse_t = msa_sparse_attention(
        q,
        k,
        v,
        idx,
        cu_q,
        cu_k,
        causal=False,
        softmax_scale=scale,
        return_temperature_lse=True,
        lse_temperature_scale=T,
    )
    torch.cuda.synchronize()
    Hq, Hkv = q.shape[1], k.shape[1]
    G = Hq // Hkv
    seqs_k = [1024, 640]
    checked = 0
    for _ in range(60):
        qi = torch.randint(0, q.shape[0], (1,)).item()
        hq = torch.randint(0, Hq, (1,)).item()
        b = 0 if qi < int(cu_q[1]) else 1
        nb = (seqs_k[b] + BLK_KV - 1) // BLK_KV
        sel = idx[hq // G, qi]
        sel = sel[(sel >= 0) & (sel < nb)]
        if sel.numel() == 0:
            assert lse[qi, hq].item() == float("-inf")
            assert lse_t[qi, hq].item() == float("-inf")
            continue
        cols = torch.cat(
            [torch.arange(s * BLK_KV, (s + 1) * BLK_KV) for s in sel.tolist()]
        )
        kk = k[int(cu_k[b]) + cols, hq // G].float()
        s = (q[qi, hq].float() @ kk.T) * scale
        assert abs(lse[qi, hq].item() - torch.logsumexp(s, -1).item()) < 1e-2
        assert abs(lse_t[qi, hq].item() - torch.logsumexp(s * T, -1).item()) < 1e-2
        checked += 1
    assert checked > 20


def test_fp8_q_decode():
    """All-fp8 decode inputs (Q, K, V all e4m3): Q is upconverted in-kernel;
    the reference uses the dequantized tensors."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_sparse_decode_attention

    torch.manual_seed(210)
    dev = "cuda"
    Hq, Hkv, topk = 8, 2, 16
    B, seqs_q = 8, [1] * 8
    seqs_k = [int(x) * BLK_KV for x in torch.randint(3, 16, (B,))]
    cu_q = torch.tensor(
        [0] + list(torch.tensor(seqs_q).cumsum(0)), dtype=torch.int32, device=dev
    )
    cu_k = torch.tensor(
        [0] + list(torch.tensor(seqs_k).cumsum(0)), dtype=torch.int32, device=dev
    )
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    q8 = (torch.randn(total_q, Hq, 128, device=dev) / 3).to(torch.float8_e4m3fn)
    k8 = (torch.randn(total_k, Hkv, 128, device=dev) / 3).to(torch.float8_e4m3fn)
    v8 = (torch.randn(total_k, Hkv, 128, device=dev) / 3).to(torch.float8_e4m3fn)
    idx = torch.full((Hkv, total_q, topk), -1, dtype=torch.int32, device=dev)
    for b in range(B):
        nb = (seqs_k[b] + BLK_KV - 1) // BLK_KV
        lo, hi = int(cu_q[b]), int(cu_q[b + 1])
        for h in range(Hkv):
            for qi in range(lo, hi):
                nsel = torch.randint(1, min(topk, nb) + 1, (1,)).item()
                sel = torch.randperm(nb)[:nsel].sort().values
                idx[h, qi, :nsel] = sel.to(torch.int32).to(dev)
    scale = 1.0 / math.sqrt(128)
    out = msa_sparse_decode_attention(
        q8,
        k8,
        v8,
        idx,
        cu_seqlens_k=cu_k,
        seqlen_q=1,
        causal=True,
        softmax_scale=scale,
    )
    torch.cuda.synchronize()
    assert out.dtype == torch.bfloat16
    ref = _ref_sparse_attention(
        q8.to(torch.bfloat16).cpu(),
        k8.to(torch.bfloat16).cpu(),
        v8.to(torch.bfloat16).cpu(),
        idx.cpu(),
        cu_q.cpu(),
        cu_k.cpu(),
        True,
        scale,
    )
    err = (out.float().cpu() - ref).abs().max().item()
    assert err < 2.5e-2, f"err={err}"


def test_msa_topk_select_countrank_matches_radix_on_nan():
    """Both kernels must produce the same, deterministic selection even when the
    proxy emits NaN scores (count-rank ranks on the radix bit-key)."""
    _skip_if_unsupported()
    from flashinfer.msa_ops.sparse_topk_select import _get_compiled_topk

    dev = "cuda"
    H, P, S, topk = 2, 100, 64, 16
    torch.manual_seed(3)
    score = torch.randn(H, P, S, device=dev, dtype=torch.float32)
    score[0, 5, :] = float("nan")
    score[1, 40, ::3] = float("nan")
    outs = []
    for small in (True, False, True):
        out = torch.empty(S, H, topk, dtype=torch.int32, device=dev)
        _get_compiled_topk(topk, small)(score, out, P, 0, 0, S, H)
        outs.append(out.cpu())
    assert torch.equal(outs[0], outs[2]), "count-rank nondeterministic on NaN"
    assert torch.equal(outs[0], outs[1]), "count-rank != radix on NaN scores"
