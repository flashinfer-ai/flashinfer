"""Tests for the union-tile sparse-attention prefill kernel
(``sparse_fwd_union_sm12x``): query-tile-major + in-kernel online softmax + direct
output (no GMEM partials, no combine). Validated against the KV-major
``msa_sparse_attention`` and an fp32 torch oracle on the same causal per-query
block selection."""

import math

import pytest
import torch

from flashinfer.utils import is_sm12x_supported

BLK = 128


def _skip():
    if not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("union-tile prefill requires SM120/SM121")


def _rand_q2k_causal(cu_q, cu_k, Hkv, topk, blk, dev, seed):
    """Per-query causal top-k selection (ascending, -1 padded): token t sees
    ``offset + t + 1`` tokens -> nb blocks -> min(topk, nb) random ones."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cuq, cuk = cu_q.tolist(), cu_k.tolist()
    out = torch.full((Hkv, cuq[-1], topk), -1, dtype=torch.int32)
    for b in range(len(cuq) - 1):
        qs, qe = cuq[b], cuq[b + 1]
        sq, sk = qe - qs, cuk[b + 1] - cuk[b]
        off = sk - sq
        for h in range(Hkv):
            for t in range(sq):
                nb = (off + t + 1 + blk - 1) // blk
                cnt = min(topk, nb)
                sel = torch.randperm(nb, generator=g)[:cnt].sort().values
                out[h, qs + t, :cnt] = sel.to(torch.int32)
    return out.to(dev)


def _torch_oracle(q, k, v, q2k, cu_q, cu_k, blk, G, scale):
    dev = q.device
    cuq, cuk = cu_q.tolist(), cu_k.tolist()
    Hq = q.shape[1]
    out = torch.zeros_like(q, dtype=torch.float32)
    for b in range(len(cuq) - 1):
        qs, qe = cuq[b], cuq[b + 1]
        ks, sk = cuk[b], cuk[b + 1] - cuk[b]
        sq = qe - qs
        off = sk - sq
        for hq in range(Hq):
            h = hq // G
            for t in range(sq):
                blocks = [x for x in q2k[h, qs + t].tolist() if x >= 0]
                climit = off + t
                cols = [
                    bid * blk + j
                    for bid in blocks
                    for j in range(blk)
                    if bid * blk + j < sk and bid * blk + j <= climit
                ]
                if not cols:
                    continue
                ci = torch.tensor(cols, device=dev)
                s = (q[qs + t, hq].float() @ k[ks + ci, h].float().T) * scale
                out[qs + t, hq] = torch.softmax(s, -1) @ v[ks + ci, h].float()
    return out


def _compile_union(G, causal, m_block, num_threads, hd, return_lse):
    import cutlass
    import cutlass.cute as cute

    from flashinfer.msa_ops.cute_dsl.sparse_fwd_union_sm12x import (
        SparseAttentionUnionFwdSm12x,
    )
    from flashinfer.msa_ops.sparse_attention import _cutlass_dtype, _fake

    cdt = _cutlass_dtype(torch.bfloat16)
    i32 = _cutlass_dtype(torch.int32)
    f32 = _cutlass_dtype(torch.float32)
    s = [cute.sym_int() for _ in range(11)]
    obj = SparseAttentionUnionFwdSm12x(
        head_dim=hd,
        m_block_size=m_block,
        n_block_size=128,
        group_size=G,
        num_threads=num_threads,
        is_causal=causal,
        return_softmax_lse=return_lse,
    )
    return cute.compile(
        obj,
        _fake(cdt, (s[0], s[1], hd)),
        _fake(cdt, (s[2], s[3], hd)),
        _fake(cdt, (s[2], s[3], hd)),
        _fake(cdt, (s[0], s[1], hd)),
        _fake(f32, (s[1], s[0]), align=4),
        _fake(i32, (s[6], s[7]), align=4),
        _fake(i32, (s[6], s[7]), align=4),
        _fake(i32, (s[6],), align=4),
        _fake(i32, (s[6], 3), align=4),
        _fake(i32, (s[8],), align=4),
        _fake(i32, (s[4],), align=4),
        _fake(i32, (s[4],), align=4),
        _fake(i32, (s[5],), align=4),
        _fake(i32, (s[9], s[10]), align=4),
        cutlass.Float32(1.0),
        cutlass.Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _canon_union_meta(ub, um, uc, wm, n):
    """Map a union-metadata tuple to {(batch, q_tile, kv_head): (blocks, masks)}
    over its non-empty work items, so two builders can be compared regardless of
    work-item emission order or empty-item padding."""
    ub, um, uc, wm = (t.cpu() for t in (ub, um, uc, wm))
    out = {}
    for i in range(n):
        cnt = int(uc[i])
        if cnt == 0:
            continue
        key = tuple(int(x) for x in wm[i].tolist())
        out[key] = (
            tuple(int(x) for x in ub[i, :cnt].tolist()),
            tuple(int(x) for x in um[i, :cnt].tolist()),
        )
    return out


@pytest.mark.parametrize("tpt", [8, 16])
@pytest.mark.parametrize("B,S", [(1, 2048), (3, 640)])
def test_union_metadata_device_matches_host(tpt, B, S):
    """The on-device union-metadata builder produces the same per-(batch, q-tile,
    kv-head) unions and membership masks as the host reference builder."""
    _skip()
    from flashinfer.msa_ops._union_metadata import (
        build_msa_union_metadata,
        build_msa_union_metadata_device,
    )

    dev = "cuda"
    Hkv, topk = 4, 16
    cu = torch.tensor([S * i for i in range(B + 1)], dtype=torch.int32, device=dev)
    q2k = _rand_q2k_causal(cu, cu, Hkv, topk, BLK, dev, seed=13)

    host = build_msa_union_metadata(q2k, cu, tpt, topk)
    devb = build_msa_union_metadata_device(q2k, cu, tpt, topk)
    torch.cuda.synchronize()

    assert _canon_union_meta(*host) == _canon_union_meta(*devb)


@pytest.mark.parametrize("m_block,num_threads", [(64, 128), (128, 256)])
@pytest.mark.parametrize("B,S", [(1, 1024), (2, 512)])
def test_union_prefill_matches_kvmajor_and_oracle(m_block, num_threads, B, S):
    _skip()
    import cutlass

    from flashinfer.msa_ops import msa_sparse_attention
    from flashinfer.msa_ops._union_metadata import build_msa_union_metadata
    from flashinfer.msa_ops.sparse_attention import _q_offset_tensor

    dev = "cuda"
    Hq, Hkv, topk, hd = 64, 4, 16, 128
    G = Hq // Hkv
    tpt = m_block // G
    scale = 1.0 / math.sqrt(hd)
    cu = torch.tensor([S * i for i in range(B + 1)], dtype=torch.int32, device=dev)
    torch.manual_seed(7)
    q = torch.randn(B * S, Hq, hd, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    v = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    q2k = _rand_q2k_causal(cu, cu, Hkv, topk, BLK, dev, seed=3)

    ref = msa_sparse_attention(q, k, v, q2k, cu, cu, causal=True, softmax_scale=scale)
    oracle = _torch_oracle(q, k, v, q2k, cu, cu, BLK, G, scale)

    ub, um, uc, wm, n = build_msa_union_metadata(q2k, cu, tpt, topk)
    qoff = _q_offset_tensor(None, cu, cu, dev)
    wc = torch.tensor([n], dtype=torch.int32, device=dev)
    out = torch.empty(B * S, Hq, hd, dtype=torch.bfloat16, device=dev)
    lse = torch.empty(Hq, B * S, dtype=torch.float32, device=dev)
    ptab_dummy = torch.zeros((1, 1), dtype=torch.int32, device=dev)
    compiled = _compile_union(G, True, m_block, num_threads, hd, False)
    compiled(
        q,
        k,
        v,
        out,
        lse,
        ub,
        um,
        uc,
        wm,
        wc,
        cu,
        cu,
        qoff,
        ptab_dummy,
        cutlass.Float32(scale),
        cutlass.Int32(n),
    )
    torch.cuda.synchronize()

    mae_oracle = (out.float() - oracle).abs().mean().item()
    mae_kv_oracle = (ref.float() - oracle).abs().mean().item()
    mae_vs_kv = (out.float() - ref.float()).abs().mean().item()
    # union accumulates the whole softmax in fp32 (no bf16 partial round-trip), so
    # it should track the oracle at least as well as kvmajor, and agree with kvmajor
    # to bf16 precision.
    assert mae_oracle <= mae_kv_oracle * 1.5 + 1e-6, (mae_oracle, mae_kv_oracle)
    assert mae_vs_kv < 5e-4, mae_vs_kv


@pytest.mark.parametrize("B,S", [(1, 1024), (2, 512)])
def test_union_paged_matches_flat(B, S):
    """Paged-KV union (page_size == block == 128, one page per KV block) matches
    the flat union path bit-for-bit on identical data -- the paged path only
    remaps the K/V block address through the page table."""
    _skip()
    from flashinfer.msa_ops import msa_sparse_attention

    dev = "cuda"
    Hq, Hkv, topk, hd = 64, 4, 16, 128
    scale = 1.0 / math.sqrt(hd)
    cu = torch.tensor([S * i for i in range(B + 1)], dtype=torch.int32, device=dev)
    torch.manual_seed(17)
    q = torch.randn(B * S, Hq, hd, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    v = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    q2k = _rand_q2k_causal(cu, cu, Hkv, topk, BLK, dev, seed=9)

    flat = msa_sparse_attention(
        q, k, v, q2k, cu, cu, union=True, causal=True, softmax_scale=scale
    )

    # relay flat K/V into a randomly-permuted paged cache (one 128-token page per
    # KV block) and build the page table, then run the same union path paged.
    npages = [S // BLK] * B
    perm = torch.randperm(sum(npages))
    k_pg = torch.zeros(sum(npages), Hkv, BLK, hd, dtype=k.dtype, device=dev)
    v_pg = torch.zeros_like(k_pg)
    ptab = torch.full((B, max(npages)), -1, dtype=torch.int32, device=dev)
    pi = 0
    for b in range(B):
        for blk in range(npages[b]):
            pg = int(perm[pi])
            pi += 1
            ptab[b, blk] = pg
            rows = slice(b * S + blk * BLK, b * S + (blk + 1) * BLK)
            k_pg[pg] = k[rows].transpose(0, 1)
            v_pg[pg] = v[rows].transpose(0, 1)
    seqused = torch.tensor([S] * B, dtype=torch.int32, device=dev)
    paged = msa_sparse_attention(
        q,
        k_pg.contiguous(),
        v_pg.contiguous(),
        q2k,
        cu,
        page_table=ptab,
        seqused_k=seqused,
        union=True,
        causal=True,
        softmax_scale=scale,
    )
    torch.cuda.synchronize()
    assert torch.equal(paged, flat)


@pytest.mark.parametrize("B,S", [(1, 1024), (3, 384)])
@pytest.mark.parametrize("return_lse", [False, True])
def test_union_public_api(B, S, return_lse):
    """msa_sparse_attention(union=True) matches the default KV-major path (output
    and LSE) end to end through the public wrapper."""
    _skip()
    from flashinfer.msa_ops import msa_sparse_attention

    dev = "cuda"
    Hq, Hkv, topk, hd = 64, 4, 16, 128
    G = Hq // Hkv
    scale = 1.0 / math.sqrt(hd)
    cu = torch.tensor([S * i for i in range(B + 1)], dtype=torch.int32, device=dev)
    torch.manual_seed(11)
    q = torch.randn(B * S, Hq, hd, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    v = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    q2k = _rand_q2k_causal(cu, cu, Hkv, topk, BLK, dev, seed=5)
    oracle = _torch_oracle(q, k, v, q2k, cu, cu, BLK, G, scale)

    kw = dict(causal=True, softmax_scale=scale, return_softmax_lse=return_lse)
    ref = msa_sparse_attention(q, k, v, q2k, cu, cu, **kw)
    got = msa_sparse_attention(q, k, v, q2k, cu, cu, union=True, **kw)
    if return_lse:
        ref, ref_lse = ref
        got, got_lse = got
        assert got_lse.shape == (B * S, Hq)
        # compare LSE only where the KV-major path produced a finite value
        finite = torch.isfinite(ref_lse)
        dl = (got_lse[finite] - ref_lse[finite]).abs().max().item()
        assert dl < 5e-2, dl
    torch.cuda.synchronize()

    assert got.shape == (B * S, Hq, hd)
    mae_o = (got.float() - oracle).abs().mean().item()
    mae_kv = (ref.float() - oracle).abs().mean().item()
    assert mae_o <= mae_kv * 1.5 + 1e-6, (mae_o, mae_kv)
    assert (got.float() - ref.float()).abs().mean().item() < 5e-4
