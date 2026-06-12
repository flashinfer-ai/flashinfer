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
# Phase 2: build_k2q_csr
# ---------------------------------------------------------------------------


def _ref_build_k2q_csr(q2k, cu_q, cu_k, blk_kv):
    H, S_Q, topk = q2k.shape
    B = cu_q.numel() - 1
    seq_k = (cu_k[1:] - cu_k[:-1]).tolist()
    rows_per_b = [(s + blk_kv - 1) // blk_kv for s in seq_k]
    total_rows = sum(rows_per_b)
    max_kv = max(rows_per_b) if rows_per_b else 0
    row_map = {}
    row_linear = 0
    for level in range(max_kv):
        for b in range(B):
            if rows_per_b[b] > level:
                row_map[(b, level)] = row_linear
                row_linear += 1
    row_ptr = torch.zeros(H, total_rows + 1, dtype=torch.int64)
    q_lists = [[[] for _ in range(total_rows)] for _ in range(H)]
    q2k_c, cu_q_c = q2k.cpu(), cu_q.cpu()
    for h in range(H):
        for b in range(B):
            for qi in range(int(cu_q_c[b]), int(cu_q_c[b + 1])):
                qloc = qi - int(cu_q_c[b])
                for t in range(topk):
                    kvb = int(q2k_c[h, qi, t])
                    if 0 <= kvb < rows_per_b[b]:
                        q_lists[h][row_map[(b, kvb)]].append(qloc)
    for h in range(H):
        for r in range(total_rows):
            row_ptr[h, r + 1] = row_ptr[h, r] + len(q_lists[h][r])
    return row_ptr, q_lists, total_rows


@pytest.mark.parametrize(
    "B,H,topk,seqs_q,seqs_k",
    [
        (1, 2, 16, [37], [1024]),
        (3, 4, 16, [17, 64, 5], [512, 2048, 300]),
        (2, 1, 8, [100, 33], [4096, 700]),
        (2, 2, 32, [50, 21], [8192, 5000]),
        (4, 3, 4, [1, 2, 3, 4], [128, 256, 384, 129]),
    ],
)
def test_build_k2q_csr(B, H, topk, seqs_q, seqs_k):
    _skip_if_unsupported()
    from flashinfer.msa_ops import build_k2q_csr

    torch.manual_seed(0)
    dev = "cuda"
    cu_q = torch.tensor(
        [0] + list(torch.tensor(seqs_q).cumsum(0)), dtype=torch.int32, device=dev
    )
    cu_k = torch.tensor(
        [0] + list(torch.tensor(seqs_k).cumsum(0)), dtype=torch.int32, device=dev
    )
    S_Q = int(cu_q[-1])
    q2k = torch.full((H, S_Q, topk), -1, dtype=torch.int32, device=dev)
    for b in range(B):
        nb = (seqs_k[b] + BLK_KV - 1) // BLK_KV
        lo, hi = int(cu_q[b]), int(cu_q[b + 1])
        n = min(topk, nb)
        for h in range(H):
            for qi in range(lo, hi):
                sel = torch.randperm(nb)[:n].sort().values.to(torch.int32)
                q2k[h, qi, :n] = sel.to(dev)

    row_ptr, q_idx = build_k2q_csr(q2k, cu_q, cu_k, blk_kv=BLK_KV)
    torch.cuda.synchronize()

    ref_ptr, ref_lists, total_rows = _ref_build_k2q_csr(q2k, cu_q, cu_k, BLK_KV)
    assert row_ptr.shape == (H, total_rows + 1)
    assert torch.equal(row_ptr.cpu().long(), ref_ptr)
    q_idx_c = q_idx.cpu()
    for h in range(H):
        for r in range(total_rows):
            lo, hi = int(ref_ptr[h, r]), int(ref_ptr[h, r + 1])
            got = q_idx_c[h, lo:hi].tolist()
            assert got == sorted(got), f"row not q-ascending h={h} r={r}"
            assert sorted(got) == sorted(ref_lists[h][r]), f"content h={h} r={r}"


# ---------------------------------------------------------------------------
# Phase 3a: sparse_attention (q-major SM12x kernel)
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
    ],
)
def test_sparse_attention(B, Hq, Hkv, topk, seqs_q, seqs_k, causal):
    _skip_if_unsupported()
    from flashinfer.msa_ops import sparse_attention

    torch.manual_seed(42)
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
                # vary the number of selected blocks; leave some rows empty
                nsel = torch.randint(0, n + 1, (1,)).item()
                if nsel > 0:
                    sel = torch.randperm(nb)[:nsel].sort().values.to(torch.int32)
                    idx[h, qi, :nsel] = sel.to(dev)

    scale = 1.0 / math.sqrt(128)
    out = sparse_attention(q, k, v, idx, cu_q, cu_k, causal=causal, softmax_scale=scale)
    torch.cuda.synchronize()
    ref = _ref_sparse_attention(
        q.cpu(), k.cpu(), v.cpu(), idx.cpu(), cu_q.cpu(), cu_k.cpu(), causal, scale
    )
    err = (out.float().cpu() - ref).abs().max().item()
    assert err < 2.5e-2, f"max abs error {err}"


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
def test_sparse_attention_kvmajor(B, Hq, Hkv, topk, seqs_q, seqs_k, causal):
    _skip_if_unsupported()
    from flashinfer.msa_ops import sparse_attention_kvmajor

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
    out = sparse_attention_kvmajor(
        q, k, v, idx, cu_q, cu_k, causal=causal, softmax_scale=scale
    )
    torch.cuda.synchronize()
    ref = _ref_sparse_attention(
        q.cpu(), k.cpu(), v.cpu(), idx.cpu(), cu_q.cpu(), cu_k.cpu(), causal, scale
    )
    err = (out.float().cpu() - ref).abs().max().item()
    assert err < 2.5e-2, f"max abs error {err}"


# ---------------------------------------------------------------------------
# Phase 3c: paged KV, LSE output, fused combine, fp16/topk32 coverage
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
def test_sparse_attention_kvmajor_dtypes_topk(dtype, topk):
    _skip_if_unsupported()
    from flashinfer.msa_ops import sparse_attention_kvmajor

    q, k, v, idx, cu_q, cu_k = _make_case(
        2, 4, 2, topk, [100, 64], [2048, 1024], dtype, seed=20 + topk
    )
    scale = 1.0 / math.sqrt(128)
    out = sparse_attention_kvmajor(q, k, v, idx, cu_q, cu_k, softmax_scale=scale)
    torch.cuda.synchronize()
    ref = _ref_sparse_attention(
        q.cpu(), k.cpu(), v.cpu(), idx.cpu(), cu_q.cpu(), cu_k.cpu(), False, scale
    )
    err = (out.float().cpu() - ref).abs().max().item()
    assert err < 3.5e-2, f"max abs error {err}"


def test_sparse_attention_kvmajor_lse():
    _skip_if_unsupported()
    from flashinfer.msa_ops import sparse_attention_kvmajor

    q, k, v, idx, cu_q, cu_k = _make_case(
        2, 4, 2, 16, [80, 50], [1024, 640], torch.bfloat16, seed=30
    )
    scale = 1.0 / math.sqrt(128)
    out, lse = sparse_attention_kvmajor(
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


def test_sparse_attention_kvmajor_paged():
    _skip_if_unsupported()
    from flashinfer.msa_ops import sparse_attention_kvmajor

    seqs_q, seqs_k = [150, 90], [2048, 1280]  # multiples of 128
    q, k, v, idx, cu_q, cu_k = _make_case(
        2, 8, 2, 16, seqs_q, seqs_k, torch.bfloat16, seed=40
    )
    scale = 1.0 / math.sqrt(128)
    out_flat = sparse_attention_kvmajor(q, k, v, idx, cu_q, cu_k, softmax_scale=scale)

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
    out_paged = sparse_attention_kvmajor(
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


def test_fused_combine_matches_torch():
    _skip_if_unsupported()
    import importlib

    # the sparse_attention *function* shadows the submodule on attribute import
    sa = importlib.import_module("flashinfer.msa_ops.sparse_attention")

    torch.manual_seed(50)
    dev = "cuda"
    topk, total_q, Hq, G, d = 16, 200, 8, 4, 128
    Hkv = Hq // G
    o_p = torch.randn(topk, total_q, Hq, d, dtype=torch.bfloat16, device=dev)
    lse_p = torch.randn(topk, total_q, Hq, dtype=torch.float32, device=dev) * 4
    counts = torch.randint(0, topk + 1, (total_q, Hkv), dtype=torch.int32, device=dev)
    ref = sa._combine_partials_torch(o_p, lse_p, counts, G, torch.bfloat16)
    got = sa._combine_partials(o_p, lse_p, counts, G, torch.bfloat16)
    torch.cuda.synchronize()
    err = (got.float() - ref.float()).abs().max().item()
    assert err < 1e-2, f"combine mismatch {err}"


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_sparse_attention_kvmajor_fp8_kv(q_dtype, causal):
    _skip_if_unsupported()
    from flashinfer.msa_ops import sparse_attention_kvmajor

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
    out = sparse_attention_kvmajor(
        q, k8, v8, idx, cu_q, cu_k, causal=causal, softmax_scale=scale
    )
    torch.cuda.synchronize()
    # reference uses the dequantized K/V (quantization error cancels)
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
# End-to-end: sparse_topk_select -> build_k2q_csr_schedule -> kv-major fwd
# ---------------------------------------------------------------------------


def test_e2e_pipeline():
    _skip_if_unsupported()
    from flashinfer.msa_ops import sparse_attention_kvmajor, sparse_topk_select

    torch.manual_seed(70)
    dev, dtype = "cuda", torch.bfloat16
    Hq, Hkv, topk = 8, 2, 16
    seqlen_q, seqlen_k = 200, 4096  # single sequence
    nb = seqlen_k // BLK_KV
    cu_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=dev)

    # Phase 1: proxy max-scores per KV head -> top-K block indices
    max_score = torch.randn(Hkv, nb, seqlen_q, dtype=torch.float32, device=dev)
    idx_q_major = sparse_topk_select(max_score, topk)  # (total_q, Hkv, topk)
    torch.cuda.synchronize()

    # selection must match torch.topk as a set, ascending, -1-free here
    ref_sel = torch.topk(max_score, topk, dim=1).indices  # (Hkv, topk, total_q)
    for _ in range(30):
        qi = torch.randint(0, seqlen_q, (1,)).item()
        h = torch.randint(0, Hkv, (1,)).item()
        got = idx_q_major[qi, h]
        assert (got >= 0).all()
        assert (got.diff() > 0).all(), "indices must be ascending"
        assert set(got.tolist()) == set(ref_sel[h, :, qi].tolist())

    # Phase 2 + 3: head-major indices -> CSR schedule -> KV-major forward
    idx = idx_q_major.permute(1, 0, 2).contiguous()  # (Hkv, total_q, topk)
    q = torch.randn(seqlen_q, Hq, 128, dtype=dtype, device=dev) / 3
    k = torch.randn(seqlen_k, Hkv, 128, dtype=dtype, device=dev) / 3
    v = torch.randn(seqlen_k, Hkv, 128, dtype=dtype, device=dev) / 3
    scale = 1.0 / math.sqrt(128)
    out = sparse_attention_kvmajor(q, k, v, idx, cu_q, cu_k, softmax_scale=scale)
    torch.cuda.synchronize()

    ref = _ref_sparse_attention(
        q.cpu(), k.cpu(), v.cpu(), idx.cpu(), cu_q.cpu(), cu_k.cpu(), False, scale
    )
    err = (out.float().cpu() - ref).abs().max().item()
    assert err < 2.5e-2, f"e2e max abs error {err}"


# ---------------------------------------------------------------------------
# Phase 4: sparse decode
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
    from flashinfer.msa_ops import sparse_decode_attention

    q, k_flat, v_flat, idx, seqused, cu_k, pg, ptab = _make_decode_case(
        B, sq, Hq, Hkv, topk, kv_dtype, paged, seed=80 + B + sq
    )
    scale = 1.0 / math.sqrt(128)
    if paged:
        out = sparse_decode_attention(
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
        out = sparse_decode_attention(
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


def test_sparse_decode_cuda_graph():
    _skip_if_unsupported()
    from flashinfer.msa_ops import sparse_decode_attention

    q, _, _, idx, seqused, cu_k, pg, ptab = _make_decode_case(
        8, 1, 8, 2, 16, torch.float8_e4m3fn, True, seed=90
    )
    k_pg, v_pg = pg[0].contiguous(), pg[1].contiguous()
    scale = 1.0 / math.sqrt(128)
    call = lambda: sparse_decode_attention(
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
    # mutate inputs in place; replay must track them
    torch.manual_seed(91)
    q.copy_(torch.randn_like(q) / 3)
    g.replay()
    torch.cuda.synchronize()
    fresh = call()
    torch.cuda.synchronize()
    assert torch.equal(out, fresh)


# ---------------------------------------------------------------------------
# Phase 5: NVFP4 KV cache
# ---------------------------------------------------------------------------


def _msa_nvfp4_dequant(packed_u8, sf_u8, global_scale, rows, d):
    """Reference dequant per the MSA contract (quantize.py)."""
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


@pytest.mark.parametrize("causal", [False, True])
def test_sparse_attention_kvmajor_nvfp4(causal):
    _skip_if_unsupported()
    from flashinfer.msa_ops import sparse_attention_kvmajor

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
    out = sparse_attention_kvmajor(
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
    from flashinfer.msa_ops import sparse_decode_attention

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
    call = lambda: sparse_decode_attention(
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
