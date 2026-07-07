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

Tests for the NVFP4 MSA proxy (``msa_proxy_score_fp4``): torch-dequant
correctness, selection overlap vs the bf16 proxy, and reduce_heads parity.
"""

import pytest
import torch

from flashinfer.utils import is_sm12x_supported

BLK_KV = 128
# e2m1 nibble -> value LUT (sign bit is the high bit of the nibble).
_E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)


def _skip_if_unsupported():
    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("MSA ops require SM120 or SM121 and CUDA >= 12.8")


def _dequant_128x4(xq, sf_flat, mul, rows, d=128):
    """Decode packed e2m1 + e4m3 scales in the cuBLAS 128x4 tiled layout to f32,
    scaled by `mul`. SF byte offset matches the kernel's `_sf_offset`."""
    dev = xq.device
    lo = (xq & 0xF).long()
    hi = (xq >> 4).long()
    lut = _E2M1.to(dev)
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
    sc = sf_flat.view(torch.float8_e4m3fn)[off.reshape(-1)].reshape(rows, cols).float()
    return vals * sc.repeat_interleave(16, dim=1) * mul


def _ref_proxy_fp4(q_deq, k_deq, seqlen_q, seqlen_k, group_size, causal):
    """Block-max proxy on pre-dequantized bf16 Q/K, single sequence -> [Hq, nb, Sq]."""
    Hq = q_deq.shape[1]
    nb = (seqlen_k + BLK_KV - 1) // BLK_KV
    q_off = seqlen_k - seqlen_q  # right-aligned causal
    out = torch.full((Hq, nb, seqlen_q), -float("inf"), dtype=torch.float32)
    for h in range(Hq):
        kv = h // group_size
        scores = q_deq[:, h].float() @ k_deq[:, kv].float().t()  # [Sq, Sk]
        for qi in range(seqlen_q):
            col_limit = min(qi + q_off + 1, seqlen_k) if causal else seqlen_k
            for t in range(nb):
                lo = t * BLK_KV
                hi = min(lo + BLK_KV, col_limit)
                if hi > lo:
                    out[h, t, qi] = scores[qi, lo:hi].max()
    return out


def _dequant_qk(q_fp4, q_scale, inv_q, k_fp4, k_scale, inv_k):
    """Dequant packed Q/K to bf16; both global scales fold into Q, as in the kernel."""
    Sq, Hq, _ = q_fp4.shape
    Sk, Hkv, _ = k_fp4.shape
    q_deq = (
        _dequant_128x4(q_fp4.reshape(-1, 64), q_scale, inv_q * inv_k, Sq * Hq)
        .reshape(Sq, Hq, 128)
        .to(torch.bfloat16)
    )
    k_deq = (
        _dequant_128x4(k_fp4.reshape(-1, 64), k_scale, 1.0, Sk * Hkv)
        .reshape(Sk, Hkv, 128)
        .to(torch.bfloat16)
    )
    return q_deq, k_deq


@pytest.mark.parametrize("Hq,Hkv", [(4, 1), (8, 2)])
@pytest.mark.parametrize("causal", [True, False])
def test_proxy_fp4_correctness(Hq, Hkv, causal):
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_proxy_score_fp4
    from flashinfer.msa_ops.proxy_score import _quantize_qk_to_nvfp4

    torch.manual_seed(11)
    dev = "cuda"
    seqlen_q, seqlen_k = 200, 1024
    group_size = Hq // Hkv
    cu_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=dev)

    q = torch.randn(seqlen_q, Hq, 128, dtype=torch.bfloat16, device=dev) * 2
    k = torch.randn(seqlen_k, Hkv, 128, dtype=torch.bfloat16, device=dev) * 2
    q_fp4, q_scale, inv_q = _quantize_qk_to_nvfp4(q)
    k_fp4, k_scale, inv_k = _quantize_qk_to_nvfp4(k)

    out = msa_proxy_score_fp4(
        q_fp4, k_fp4, q_scale, k_scale, inv_q, inv_k, cu_q, cu_k, causal=causal
    )
    torch.cuda.synchronize()

    q_deq, k_deq = _dequant_qk(
        q_fp4.cpu(), q_scale.cpu(), inv_q, k_fp4.cpu(), k_scale.cpu(), inv_k
    )
    ref = _ref_proxy_fp4(q_deq, k_deq, seqlen_q, seqlen_k, group_size, causal)
    got = out.float().cpu()
    assert got.shape == ref.shape
    finite = torch.isfinite(ref)
    assert (torch.isfinite(got) == finite).all(), "finite/-inf mask mismatch"
    diff = (got[finite] - ref[finite]).abs()
    scale = ref[finite].abs().clamp_min(1.0)
    rel = (diff / scale).max().item()
    assert rel < 5e-2, f"max rel err {rel}"


def test_proxy_fp4_selection_overlap_vs_bf16():
    _skip_if_unsupported()
    from flashinfer.msa_ops import (
        msa_proxy_score,
        msa_proxy_score_fp4,
        msa_topk_select,
    )
    from flashinfer.msa_ops.proxy_score import _quantize_qk_to_nvfp4

    torch.manual_seed(22)
    dev = "cuda"
    Hq, Hkv, topk = 4, 1, 16
    seqlen_q, seqlen_k = 256, 4096
    cu_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=dev)

    q = torch.randn(seqlen_q, Hq, 128, dtype=torch.bfloat16, device=dev) / 2
    k = torch.randn(seqlen_k, Hkv, 128, dtype=torch.bfloat16, device=dev) / 2

    bf16_score = msa_proxy_score(q, k, cu_q, cu_k, causal=True)
    q_fp4, q_scale, inv_q = _quantize_qk_to_nvfp4(q)
    k_fp4, k_scale, inv_k = _quantize_qk_to_nvfp4(k)
    fp4_score = msa_proxy_score_fp4(
        q_fp4, k_fp4, q_scale, k_scale, inv_q, inv_k, cu_q, cu_k, causal=True
    )
    torch.cuda.synchronize()

    sel_bf16 = msa_topk_select(bf16_score, topk)  # (total_q, Hq, topk)
    sel_fp4 = msa_topk_select(fp4_score, topk)
    torch.cuda.synchronize()

    # Set overlap over (query, head) pairs where the full topk is available.
    overlaps = []
    for qi in range(0, seqlen_q, 7):
        for h in range(Hq):
            a = set(x for x in sel_bf16[qi, h].tolist() if x >= 0)
            b = set(x for x in sel_fp4[qi, h].tolist() if x >= 0)
            if len(a) >= topk:
                overlaps.append(len(a & b) / len(a))
    mean_overlap = sum(overlaps) / len(overlaps)
    # Unstructured Gaussian Q/K is the worst case for selection overlap (near-tied
    # block scores, so fp4 rounding flips many boundary blocks). The bar only needs
    # to separate a working kernel (~0.89) from a broken one (chance ~ topk/nb = 0.5).
    assert mean_overlap > 0.8, f"mean topk overlap {mean_overlap} too low"


def test_proxy_fp4_paged():
    """group_size 4 dispatches the general, non-packed fp4-MMA schedule."""
    _skip_if_unsupported()
    from flashinfer import nvfp4_quantize
    from flashinfer.msa_ops import msa_proxy_score_fp4
    from flashinfer.msa_ops.proxy_score import _quantize_qk_to_nvfp4

    torch.manual_seed(44)
    dev = "cuda"
    Hq, Hkv = 4, 1
    group_size = Hq // Hkv
    seqlen_q, seqlen_k = 160, 1024  # seqlen_k multiple of 128
    nb = seqlen_k // BLK_KV
    cu_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=dev)
    seqused = torch.tensor([seqlen_k], dtype=torch.int32, device=dev)

    q = torch.randn(seqlen_q, Hq, 128, dtype=torch.bfloat16, device=dev) * 2
    k = torch.randn(seqlen_k, Hkv, 128, dtype=torch.bfloat16, device=dev) * 2
    q_fp4, q_scale, inv_q = _quantize_qk_to_nvfp4(q)

    # Scatter logical K into shuffled pages, then quantize page-major so the SF
    # lands in the (page*Hkv+head)*128+token row order the paged kernel reads.
    perm = torch.randperm(nb)
    k_pg_bf16 = torch.zeros(nb, Hkv, BLK_KV, 128, dtype=torch.bfloat16, device=dev)
    ptab = torch.zeros((1, nb), dtype=torch.int32, device=dev)
    for blk in range(nb):
        pg = int(perm[blk])
        ptab[0, blk] = pg
        k_pg_bf16[pg] = k[blk * BLK_KV : (blk + 1) * BLK_KV].transpose(0, 1)
    gsf_k = (448.0 * 6.0) / k.float().abs().max()
    inv_k = 1.0 / float(gsf_k)
    kq, ksf = nvfp4_quantize(
        k_pg_bf16.reshape(-1, 128), gsf_k.reshape(1).to(dev), sf_vec_size=16
    )
    k_pg = kq.view(torch.uint8).reshape(nb, Hkv, BLK_KV, 64)
    k_pg_scale = ksf.view(torch.uint8).reshape(-1)

    out = msa_proxy_score_fp4(
        q_fp4,
        k_pg,
        q_scale,
        k_pg_scale,
        inv_q,
        inv_k,
        cu_q,
        page_table=ptab,
        seqused_k=seqused,
        causal=True,
    )
    torch.cuda.synchronize()

    # Reference: dequant page-major K, gather into logical block order.
    k_pg_deq = (
        _dequant_128x4(
            k_pg.reshape(-1, 64).cpu(), k_pg_scale.cpu(), 1.0, nb * Hkv * BLK_KV
        )
        .reshape(nb, Hkv, BLK_KV, 128)
        .to(torch.bfloat16)
    )
    k_deq = torch.empty(seqlen_k, Hkv, 128, dtype=torch.bfloat16)
    for blk in range(nb):
        k_deq[blk * BLK_KV : (blk + 1) * BLK_KV] = k_pg_deq[
            int(ptab[0, blk])
        ].transpose(0, 1)
    q_deq = (
        _dequant_128x4(
            q_fp4.reshape(-1, 64).cpu(), q_scale.cpu(), inv_q * inv_k, seqlen_q * Hq
        )
        .reshape(seqlen_q, Hq, 128)
        .to(torch.bfloat16)
    )
    ref = _ref_proxy_fp4(q_deq, k_deq, seqlen_q, seqlen_k, group_size, True)
    got = out.float().cpu()
    finite = torch.isfinite(ref)
    assert (torch.isfinite(got) == finite).all()
    rel = ((got[finite] - ref[finite]).abs() / ref[finite].abs().clamp_min(1.0)).max()
    assert rel < 5e-2, f"paged max rel err {rel}"


@pytest.mark.parametrize("B,seqlen_q", [(1, 8), (2, 5), (3, 1)])
def test_proxy_fp4_decode_packed(B, seqlen_q):
    """group_size 16, q_len <= 8 dispatches the packed fp4 tensor-core kernel."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_proxy_score_fp4
    from flashinfer.msa_ops.proxy_score import _quantize_qk_to_nvfp4

    torch.manual_seed(55 + B)
    dev = "cuda"
    Hkv = 2
    Hq = Hkv * 16  # group_size == 16 triggers the packed path
    seqlen_k = 1024
    nb = seqlen_k // BLK_KV
    total_q = B * seqlen_q
    total_k = B * seqlen_k
    cu_q = torch.arange(0, (B + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=dev)
    cu_k = torch.arange(0, (B + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=dev)

    q = torch.randn(total_q, Hq, 128, dtype=torch.bfloat16, device=dev) * 2
    k = torch.randn(total_k, Hkv, 128, dtype=torch.bfloat16, device=dev) * 2
    q_fp4, q_scale, inv_q = _quantize_qk_to_nvfp4(q)
    k_fp4, k_scale, inv_k = _quantize_qk_to_nvfp4(k)

    out = msa_proxy_score_fp4(
        q_fp4,
        k_fp4,
        q_scale,
        k_scale,
        inv_q,
        inv_k,
        cu_q,
        cu_k,
        causal=True,
    )
    torch.cuda.synchronize()
    assert out.shape == (Hq, nb, total_q)

    got = out.float().cpu()
    # SF row = token*heads+head spans all batches, so dequant whole then slice per batch.
    q_deq, k_deq = _dequant_qk(
        q_fp4.cpu(), q_scale.cpu(), inv_q, k_fp4.cpu(), k_scale.cpu(), inv_k
    )
    for b in range(B):
        qsl = slice(b * seqlen_q, (b + 1) * seqlen_q)
        ksl = slice(b * seqlen_k, (b + 1) * seqlen_k)
        ref = _ref_proxy_fp4(q_deq[qsl], k_deq[ksl], seqlen_q, seqlen_k, 16, True)
        sub = got[:, :, qsl]
        finite = torch.isfinite(ref)
        assert (torch.isfinite(sub) == finite).all(), f"mask mismatch b={b}"
        rel = (
            (sub[finite] - ref[finite]).abs() / ref[finite].abs().clamp_min(1.0)
        ).max()
        assert rel < 5e-2, f"packed b={b} max rel err {rel}"


@pytest.mark.parametrize("B,seqlen_q", [(1, 1), (4, 1), (2, 8), (1, 32)])
def test_proxy_fp4_decode_packed_group4(B, seqlen_q):
    """MiniMax-M3 indexer shape (group_size 4): q_len <= 32 dispatches the
    head-fused packed fp4 kernel (4 heads x 32 q-slots per 128-row tile)."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_proxy_score_fp4
    from flashinfer.msa_ops.proxy_score import _quantize_qk_to_nvfp4

    torch.manual_seed(77 + B + seqlen_q)
    dev = "cuda"
    Hkv = 1
    Hq = Hkv * 4  # group_size == 4 (M3): q_len<=32 -> packed (4 x 32 tile)
    seqlen_k = 1024
    nb = seqlen_k // BLK_KV
    total_q = B * seqlen_q
    total_k = B * seqlen_k
    cu_q = torch.arange(0, (B + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=dev)
    cu_k = torch.arange(0, (B + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=dev)

    q = torch.randn(total_q, Hq, 128, dtype=torch.bfloat16, device=dev) * 2
    k = torch.randn(total_k, Hkv, 128, dtype=torch.bfloat16, device=dev) * 2
    q_fp4, q_scale, inv_q = _quantize_qk_to_nvfp4(q)
    k_fp4, k_scale, inv_k = _quantize_qk_to_nvfp4(k)

    out = msa_proxy_score_fp4(
        q_fp4, k_fp4, q_scale, k_scale, inv_q, inv_k, cu_q, cu_k, causal=True
    )
    torch.cuda.synchronize()
    assert out.shape == (Hq, nb, total_q)

    got = out.float().cpu()
    q_deq, k_deq = _dequant_qk(
        q_fp4.cpu(), q_scale.cpu(), inv_q, k_fp4.cpu(), k_scale.cpu(), inv_k
    )
    for b in range(B):
        qsl = slice(b * seqlen_q, (b + 1) * seqlen_q)
        ksl = slice(b * seqlen_k, (b + 1) * seqlen_k)
        ref = _ref_proxy_fp4(q_deq[qsl], k_deq[ksl], seqlen_q, seqlen_k, 4, True)
        sub = got[:, :, qsl]
        finite = torch.isfinite(ref)
        assert (torch.isfinite(sub) == finite).all(), f"mask mismatch b={b}"
        rel = (
            (sub[finite] - ref[finite]).abs() / ref[finite].abs().clamp_min(1.0)
        ).max()
        assert rel < 5e-2, f"packed-group4 b={b} max rel err {rel}"


def test_proxy_fp4_paged_packed():
    """group_size 16, q_len <= 8 decode with a paged K cache: the packed schedule
    plus the page_table indirection."""
    _skip_if_unsupported()
    from flashinfer import nvfp4_quantize
    from flashinfer.msa_ops import msa_proxy_score_fp4
    from flashinfer.msa_ops.proxy_score import _quantize_qk_to_nvfp4

    torch.manual_seed(66)
    dev = "cuda"
    Hkv = 1
    Hq = Hkv * 16  # group_size == 16 triggers the packed schedule
    seqlen_q, seqlen_k = 8, 1024  # seqlen_k multiple of 128
    nb = seqlen_k // BLK_KV
    cu_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=dev)
    seqused = torch.tensor([seqlen_k], dtype=torch.int32, device=dev)

    q = torch.randn(seqlen_q, Hq, 128, dtype=torch.bfloat16, device=dev) * 2
    k = torch.randn(seqlen_k, Hkv, 128, dtype=torch.bfloat16, device=dev) * 2
    q_fp4, q_scale, inv_q = _quantize_qk_to_nvfp4(q)

    # Scatter logical K into shuffled pages and quantize page-major, as in
    # test_proxy_fp4_paged.
    perm = torch.randperm(nb)
    k_pg_bf16 = torch.zeros(nb, Hkv, BLK_KV, 128, dtype=torch.bfloat16, device=dev)
    ptab = torch.zeros((1, nb), dtype=torch.int32, device=dev)
    for blk in range(nb):
        pg = int(perm[blk])
        ptab[0, blk] = pg
        k_pg_bf16[pg] = k[blk * BLK_KV : (blk + 1) * BLK_KV].transpose(0, 1)
    gsf_k = (448.0 * 6.0) / k.float().abs().max()
    inv_k = 1.0 / float(gsf_k)
    kq, ksf = nvfp4_quantize(
        k_pg_bf16.reshape(-1, 128), gsf_k.reshape(1).to(dev), sf_vec_size=16
    )
    k_pg = kq.view(torch.uint8).reshape(nb, Hkv, BLK_KV, 64)
    k_pg_scale = ksf.view(torch.uint8).reshape(-1)

    out = msa_proxy_score_fp4(
        q_fp4,
        k_pg,
        q_scale,
        k_pg_scale,
        inv_q,
        inv_k,
        cu_q,
        page_table=ptab,
        seqused_k=seqused,
        causal=True,
    )
    torch.cuda.synchronize()
    assert out.shape == (Hq, nb, seqlen_q)

    k_pg_deq = (
        _dequant_128x4(
            k_pg.reshape(-1, 64).cpu(), k_pg_scale.cpu(), 1.0, nb * Hkv * BLK_KV
        )
        .reshape(nb, Hkv, BLK_KV, 128)
        .to(torch.bfloat16)
    )
    k_deq = torch.empty(seqlen_k, Hkv, 128, dtype=torch.bfloat16)
    for blk in range(nb):
        k_deq[blk * BLK_KV : (blk + 1) * BLK_KV] = k_pg_deq[
            int(ptab[0, blk])
        ].transpose(0, 1)
    q_deq = (
        _dequant_128x4(
            q_fp4.reshape(-1, 64).cpu(), q_scale.cpu(), inv_q * inv_k, seqlen_q * Hq
        )
        .reshape(seqlen_q, Hq, 128)
        .to(torch.bfloat16)
    )
    ref = _ref_proxy_fp4(q_deq, k_deq, seqlen_q, seqlen_k, 16, True)
    got = out.float().cpu()
    finite = torch.isfinite(ref)
    assert (torch.isfinite(got) == finite).all()
    rel = ((got[finite] - ref[finite]).abs() / ref[finite].abs().clamp_min(1.0)).max()
    assert rel < 5e-2, f"paged-packed max rel err {rel}"


def test_proxy_split_k_heuristic():
    """The kv-block split factor is 1 once the base grid fills the SMs and grows
    as the base grid shrinks (low-batch decode), capped by max_k_tiles."""
    _skip_if_unsupported()
    from flashinfer.msa_ops.proxy_score import _proxy_split_k_fp4

    dev = torch.device("cuda")
    sm = torch.cuda.get_device_properties(dev).multi_processor_count
    # Large base grid -> no split.
    assert _proxy_split_k_fp4(4 * sm, 512, dev) == 1
    # Trivial / degenerate -> no split.
    assert _proxy_split_k_fp4(8, 1, dev) == 1
    assert _proxy_split_k_fp4(0, 512, dev) == 1
    # Small base grid -> split enough to reach ~2 CTAs/SM, capped by max_k_tiles.
    assert _proxy_split_k_fp4(8, 512, dev) == -(-2 * sm // 8)
    assert _proxy_split_k_fp4(4, 8, dev) == 8  # clamped to max_k_tiles


@pytest.mark.parametrize("Hq,Hkv", [(4, 1), (32, 2)])
def test_proxy_fp4_split_k_decode(Hq, Hkv):
    """Low-batch long-context decode forces split-K; covers the general (Hq=4)
    and 16-head packed (Hq=32) schedules."""
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_proxy_score_fp4
    from flashinfer.msa_ops.proxy_score import _quantize_qk_to_nvfp4
    from flashinfer.msa_ops.proxy_score import _proxy_split_k_fp4

    torch.manual_seed(88 + Hq)
    dev = "cuda"
    group_size = Hq // Hkv
    seqlen_q, seqlen_k = 1, 8192  # decode, 64 kv blocks -> many splits at B=1
    nb = seqlen_k // BLK_KV
    cu_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=dev)

    # The base grid is << 2*SMs here, so the split-K heuristic must pick split > 1.
    # fp4-MMA uses a 128-row q-tile for both the general and packed schedules.
    base = (1 if group_size == 16 else -(-seqlen_q // 128)) * (
        Hkv if group_size == 16 else Hq
    )
    assert _proxy_split_k_fp4(base, nb, torch.device(dev)) > 1, "test must split"

    q = torch.randn(seqlen_q, Hq, 128, dtype=torch.bfloat16, device=dev) * 2
    k = torch.randn(seqlen_k, Hkv, 128, dtype=torch.bfloat16, device=dev) * 2
    q_fp4, q_scale, inv_q = _quantize_qk_to_nvfp4(q)
    k_fp4, k_scale, inv_k = _quantize_qk_to_nvfp4(k)

    out = msa_proxy_score_fp4(
        q_fp4,
        k_fp4,
        q_scale,
        k_scale,
        inv_q,
        inv_k,
        cu_q,
        cu_k,
        causal=True,
    )
    torch.cuda.synchronize()
    assert out.shape == (Hq, nb, seqlen_q)

    q_deq, k_deq = _dequant_qk(
        q_fp4.cpu(), q_scale.cpu(), inv_q, k_fp4.cpu(), k_scale.cpu(), inv_k
    )
    ref = _ref_proxy_fp4(q_deq, k_deq, seqlen_q, seqlen_k, group_size, True)
    got = out.float().cpu()
    finite = torch.isfinite(ref)
    assert (torch.isfinite(got) == finite).all(), "finite/-inf mask mismatch"
    rel = ((got[finite] - ref[finite]).abs() / ref[finite].abs().clamp_min(1.0)).max()
    assert rel < 5e-2, f"split-K max rel err {rel}"


def test_proxy_fp4_reduce_heads():
    _skip_if_unsupported()
    from flashinfer.msa_ops import msa_proxy_score_fp4
    from flashinfer.msa_ops.proxy_score import _quantize_qk_to_nvfp4

    torch.manual_seed(33)
    dev = "cuda"
    Hq, Hkv = 4, 1
    seqlen_q, seqlen_k = 128, 1024
    cu_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=dev)

    q = torch.randn(seqlen_q, Hq, 128, dtype=torch.bfloat16, device=dev)
    k = torch.randn(seqlen_k, Hkv, 128, dtype=torch.bfloat16, device=dev)
    q_fp4, q_scale, inv_q = _quantize_qk_to_nvfp4(q)
    k_fp4, k_scale, inv_k = _quantize_qk_to_nvfp4(k)

    per_head = msa_proxy_score_fp4(
        q_fp4, k_fp4, q_scale, k_scale, inv_q, inv_k, cu_q, cu_k, causal=True
    )
    reduced = msa_proxy_score_fp4(
        q_fp4,
        k_fp4,
        q_scale,
        k_scale,
        inv_q,
        inv_k,
        cu_q,
        cu_k,
        causal=True,
        reduce_heads=True,
    )
    torch.cuda.synchronize()
    assert reduced.shape[0] == 1
    ref = torch.amax(per_head, dim=0, keepdim=True)
    # -inf-safe compare
    assert torch.equal(
        torch.nan_to_num(reduced, neginf=-1e30),
        torch.nan_to_num(ref, neginf=-1e30),
    )
