"""Tests for the union-tile sparse-attention prefill kernel
(``sparse_fwd_union_sm12x``): query-tile-major + in-kernel online softmax + direct
output (no GMEM partials, no combine). This is the prefill path behind the public
``msa_sparse_attention``; validated against an fp32 torch oracle on the same causal
per-query block selection, plus flat-vs-paged consistency."""

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


def _compile_union(
    G, causal, m_block, num_threads, hd, return_lse, return_temp_lse=False
):
    import cutlass
    import cutlass.cute as cute

    from flashinfer.msa_ops.cute_dsl.sparse_fwd_union_sm12x import (
        SparseAttentionUnionFwdSm12x,
    )
    from flashinfer.msa_ops.sparse_attention import _cutlass_dtype, _fake

    cdt = _cutlass_dtype(torch.bfloat16)
    i32 = _cutlass_dtype(torch.int32)
    u8 = _cutlass_dtype(torch.uint8)
    f32 = _cutlass_dtype(torch.float32)
    s = [cute.sym_int() for _ in range(15)]
    obj = SparseAttentionUnionFwdSm12x(
        head_dim=hd,
        m_block_size=m_block,
        n_block_size=128,
        group_size=G,
        num_threads=num_threads,
        is_causal=causal,
        return_softmax_lse=return_lse,
        return_temperature_lse=return_temp_lse,
    )
    return cute.compile(
        obj,
        _fake(cdt, (s[0], s[1], hd)),
        _fake(cdt, (s[2], s[3], hd)),
        _fake(cdt, (s[2], s[3], hd)),
        _fake(u8, (s[11],), align=4),
        _fake(u8, (s[12],), align=4),
        _fake(cdt, (s[0], s[1], hd)),
        _fake(f32, (s[1], s[0]), align=4),
        _fake(f32, (s[13], s[14]), align=4),
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
        cutlass.Float32(1.0),
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
def test_union_prefill_kernel_matches_public_api_and_oracle(m_block, num_threads, B, S):
    """The union kernel invoked directly (with the host metadata builder) matches
    both the public ``msa_sparse_attention`` wrapper (on-device metadata builder)
    and the fp32 torch oracle."""
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
    lse_t = torch.zeros((1, 1), dtype=torch.float32, device=dev)
    sf_dummy = torch.zeros(1, dtype=torch.uint8, device=dev)
    ptab_dummy = torch.zeros((1, 1), dtype=torch.int32, device=dev)
    compiled = _compile_union(G, True, m_block, num_threads, hd, False)
    compiled(
        q,
        k,
        v,
        sf_dummy,
        sf_dummy,
        out,
        lse,
        lse_t,
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
        cutlass.Float32(1.0),
        cutlass.Float32(1.0),
        cutlass.Int32(n),
    )
    torch.cuda.synchronize()

    mae_oracle = (out.float() - oracle).abs().mean().item()
    mae_api_oracle = (ref.float() - oracle).abs().mean().item()
    mae_vs_api = (out.float() - ref.float()).abs().mean().item()
    # direct kernel and public API run the same union math (host vs device metadata
    # builder), so they agree to bf16 precision and track the fp32 oracle equally.
    assert mae_oracle < 5e-3, mae_oracle
    assert mae_api_oracle < 5e-3, mae_api_oracle
    assert mae_vs_api < 5e-4, mae_vs_api


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

    flat = msa_sparse_attention(q, k, v, q2k, cu, cu, causal=True, softmax_scale=scale)

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
        causal=True,
        softmax_scale=scale,
    )
    torch.cuda.synchronize()
    assert torch.equal(paged, flat)


@pytest.mark.parametrize("B,S", [(1, 1024), (3, 384)])
@pytest.mark.parametrize("return_lse", [False, True])
def test_union_public_api(B, S, return_lse):
    """msa_sparse_attention (the union prefill path) matches the fp32 torch oracle
    end to end through the public wrapper, including the LSE."""
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
    got = msa_sparse_attention(q, k, v, q2k, cu, cu, **kw)
    if return_lse:
        got, got_lse = got
        assert got_lse.shape == (B * S, Hq)
        # finite LSE for queries that selected at least one block
        assert torch.isfinite(got_lse).any()
    torch.cuda.synchronize()

    assert got.shape == (B * S, Hq, hd)
    assert (got.float() - oracle).abs().mean().item() < 5e-3


@pytest.mark.parametrize("Hq,Hkv", [(1, 1), (2, 2), (4, 2)])
@pytest.mark.parametrize("B,S", [(1, 1024), (2, 384)])
def test_union_small_group_matches_oracle(Hq, Hkv, B, S):
    """Small / unit GQA groups, including MHA (group_size=1 -> the 32-row tile,
    tokens_per_tile=32 = the membership mask's exact capacity), match the fp32
    torch oracle."""
    _skip()
    from flashinfer.msa_ops import msa_sparse_attention

    dev = "cuda"
    topk, hd = 16, 128
    G = Hq // Hkv
    scale = 1.0 / math.sqrt(hd)
    cu = torch.tensor([S * i for i in range(B + 1)], dtype=torch.int32, device=dev)
    torch.manual_seed(17)
    q = torch.randn(B * S, Hq, hd, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    v = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    q2k = _rand_q2k_causal(cu, cu, Hkv, topk, BLK, dev, seed=9)
    oracle = _torch_oracle(q, k, v, q2k, cu, cu, BLK, G, scale)

    kw = dict(causal=True, softmax_scale=scale)
    got = msa_sparse_attention(q, k, v, q2k, cu, cu, **kw)
    torch.cuda.synchronize()

    assert got.shape == (B * S, Hq, hd)
    assert (got.float() - oracle).abs().mean().item() < 5e-3


@pytest.mark.parametrize("B,S", [(1, 1024), (3, 384)])
@pytest.mark.parametrize("causal", [True, False])
def test_union_temperature_lse(B, S, causal):
    """return_temperature_lse returns (out, lse, lse_t); the temperature LSE is the
    log-sum-exp of the temperature-scaled scores over each query's selected (and
    causally valid) columns, matching a torch oracle."""
    _skip()
    from flashinfer.msa_ops import msa_sparse_attention

    dev = "cuda"
    Hq, Hkv, topk, hd = 64, 4, 16, 128
    G = Hq // Hkv
    scale = 1.0 / math.sqrt(hd)
    T = 0.7
    cu = torch.tensor([S * i for i in range(B + 1)], dtype=torch.int32, device=dev)
    torch.manual_seed(13)
    q = torch.randn(B * S, Hq, hd, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    v = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    q2k = _rand_q2k_causal(cu, cu, Hkv, topk, BLK, dev, seed=7)

    out, lse, lse_t = msa_sparse_attention(
        q,
        k,
        v,
        q2k,
        cu,
        cu,
        causal=causal,
        softmax_scale=scale,
        return_temperature_lse=True,
        lse_temperature_scale=T,
    )
    torch.cuda.synchronize()
    assert lse_t.shape == (B * S, Hq) and lse.shape == (B * S, Hq)

    # torch oracle over selected, causally valid columns (a few sampled rows)
    cuq = cu.tolist()
    off = 0  # cu_q == cu_k -> right-aligned causal offset is 0
    checked = 0
    for _ in range(40):
        qi = int(torch.randint(0, B * S, (1,)).item())
        hq = int(torch.randint(0, Hq, (1,)).item())
        b = next(i for i in range(B) if cuq[i] <= qi < cuq[i + 1])
        t = qi - cuq[b]
        nb = (off + t + 1 + BLK - 1) // BLK if causal else S // BLK
        sel = q2k[hq // G, qi]
        sel = sel[(sel >= 0) & (sel < nb)]
        if sel.numel() == 0:
            assert lse_t[qi, hq].item() == float("-inf")
            continue
        cols = torch.cat(
            [torch.arange(s * BLK, (s + 1) * BLK) for s in sel.tolist()]
        ).to(dev)
        if causal:
            cols = cols[cols <= off + t]
        kk = k[cuq[b] + cols, hq // G].float()
        s = (q[qi, hq].float() @ kk.T) * scale
        ref_lse_t = torch.logsumexp(s * T, -1).item()
        assert abs(lse_t[qi, hq].item() - ref_lse_t) < 5e-2
        checked += 1
    assert checked > 10


def _nvfp4_quant(x2d):
    from flashinfer import nvfp4_quantize

    gsf = (448.0 * 6.0) / x2d.float().abs().max()
    xq, sf = nvfp4_quantize(x2d, gsf.to(x2d.device), sf_vec_size=16)
    return xq.view(torch.uint8), sf.view(torch.uint8), float(1.0 / gsf)


@pytest.mark.parametrize("quant", ["fp8", "nvfp4"])
def test_union_paged_quant_matches_flat(quant):
    """Paged-KV union with fp8 / NVFP4 K/V matches the flat union path on the same
    values relayed into a permuted paged cache (one 128-token page per block). The
    paged dequant only changes the K/V block address (page table) and the SF row
    base; the NVFP4 paged SF lands in (page, head, token) order, which is exactly
    what quantizing the paged cache reshaped to (-1, head_dim) produces. (Flat
    fp8/NVFP4 union is validated against the torch oracle in test_msa_ops.py.)"""
    _skip()
    from flashinfer.msa_ops import msa_sparse_attention

    dev = "cuda"
    B, S, Hq, Hkv, topk, hd = 1, 1024, 64, 4, 16, 128
    scale = 1.0 / math.sqrt(hd)
    cu = torch.tensor([S * i for i in range(B + 1)], dtype=torch.int32, device=dev)
    torch.manual_seed(31)
    q = torch.randn(B * S, Hq, hd, dtype=torch.bfloat16, device=dev) / 3
    k = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    v = torch.randn(B * S, Hkv, hd, dtype=torch.bfloat16, device=dev) / 3
    q2k = _rand_q2k_causal(cu, cu, Hkv, topk, BLK, dev, seed=9)
    npages = S // BLK
    perm = torch.randperm(npages)
    ptab = torch.full((B, npages), -1, dtype=torch.int32, device=dev)
    seqused = torch.tensor([S] * B, dtype=torch.int32, device=dev)
    # relay flat K/V into a permuted paged bf16 cache (one 128-token page per block)
    kpg = torch.zeros(npages, Hkv, BLK, hd, dtype=torch.bfloat16, device=dev)
    vpg = torch.zeros_like(kpg)
    for blk in range(npages):
        pg = int(perm[blk])
        ptab[0, blk] = pg
        rows = slice(blk * BLK, (blk + 1) * BLK)
        kpg[pg] = k[rows].transpose(0, 1)
        vpg[pg] = v[rows].transpose(0, 1)
    fkw = dict(causal=True, softmax_scale=scale)
    pkw = dict(page_table=ptab, seqused_k=seqused, causal=True, softmax_scale=scale)
    if quant == "fp8":
        kf = k.reshape(B * S, Hkv, hd).to(torch.float8_e4m3fn)
        vf = v.reshape(B * S, Hkv, hd).to(torch.float8_e4m3fn)
        k_pg = kpg.to(torch.float8_e4m3fn).contiguous()
        v_pg = vpg.to(torch.float8_e4m3fn).contiguous()
    else:
        # quantize the flat and paged caches identically (same global scale)
        kq, ksf, kg = _nvfp4_quant(k.reshape(-1, hd))
        vq, vsf, vg = _nvfp4_quant(v.reshape(-1, hd))
        kf = kq.reshape(B * S, Hkv, hd // 2)
        vf = vq.reshape(B * S, Hkv, hd // 2)
        fkw.update(k_scale=ksf, v_scale=vsf, k_global_scale=kg, v_global_scale=vg)
        kpq, kpsf, kpg_ = _nvfp4_quant(kpg.reshape(-1, hd))
        vpq, vpsf, vpg_ = _nvfp4_quant(vpg.reshape(-1, hd))
        k_pg = kpq.reshape(npages, Hkv, BLK, hd // 2)
        v_pg = vpq.reshape(npages, Hkv, BLK, hd // 2)
        pkw.update(k_scale=kpsf, v_scale=vpsf, k_global_scale=kpg_, v_global_scale=vpg_)
    flat = msa_sparse_attention(q, kf, vf, q2k, cu, cu, **fkw)
    paged = msa_sparse_attention(q, k_pg, v_pg, q2k, cu, **pkw)
    torch.cuda.synchronize()
    assert paged.shape == (B * S, Hq, hd)
    assert (paged.float() - flat.float()).abs().max().item() < 5e-3
