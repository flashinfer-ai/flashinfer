"""DSV4 sparse-MLA decode DUAL-CACHE numeric reference + test-case harness.

This is the P7c ground-truth oracle for the SM120 DSV4 *extra-cache* (dual
KV) decode path. It extends :mod:`dsv4_ref` (which it imports and reuses
verbatim — same quantizer, same UE8M0 path, same attention math) to the
two-section gather that FlashInfer's ``decode_dsv4_kernel.cuh`` performs when
``has_extra_cache`` is set.

WHY A SECOND SECTION
--------------------
A DSV4 decode attends over the UNION of two independently paged KV pools:

  * the MAIN paged topk-selected KV  (page_block_size, ``indices``,
    ``topk_length``), and
  * a SECOND "extra" KV pool         (``extra_kv_cache``, ``extra_indices``,
    ``extra_topk_length``, page size ``pbs_extra`` — typically the most-recent
    tokens with a small page block size, e.g. 2).

Both pools use the IDENTICAL DSV4 packed byte layout (nope FP8 448B + rope
bf16 128B per token, then a UE8M0 scale footer per block — see dsv4_ref).
The only per-section differences the kernel sees are:

  * a different base pointer            (KV_cache vs extra_KV_cache),
  * a different per-block stride         (stride_kv_block vs stride_extra_kv_block),
  * a different page block size          (pbs vs pbs_extra),
  * a different index array + row stride (indices[t,:topk] vs
    extra_indices[t,:extra_topk]),
  * a different valid length             (topk_length vs extra_topk_length).

A SINGLE online softmax + output accumulator spans BOTH sections; the kernel
processes the ``num_main_chunks = ceil(topk/BI)`` main chunks first (gathering
from the MAIN cache) then the ``num_extra_chunks = ceil(extra_topk/BI)`` extra
chunks (gathering from the EXTRA cache), with
``num_splits = num_main_chunks + num_extra_chunks`` spanning both.

THE ORACLE TRICK (matches FlashInfer test_sparse_mla_sm120.py verbatim)
----------------------------------------------------------------------
Because both sections feed ONE softmax over disjoint candidate sets, the dual
result is *exactly* the single-cache result over the concatenated pool with
the extra indices shifted into a disjoint slot range. This module therefore:

  1. dequantizes each packed cache with ``dsv4_ref.dequantize_kv_dsv4``,
  2. concatenates ``[main_dequant; extra_dequant]`` into one virtual pool
     (main slots occupy ``[0, main_s_kv)``, extra slots
     ``[main_s_kv, main_s_kv + extra_s_kv)``),
  3. masks the extra window by ``extra_topk_length`` and propagates ``-1``
     sentinels, then SHIFTS valid extra indices by ``main_s_kv``
     (``where(idx<0, idx, idx+main_s_kv)`` — sentinels stay negative),
  4. concatenates ``[main_idx, extra_idx_shifted]`` along the topk axis and
     calls the EXISTING :func:`dsv4_ref.dsv4_decode_reference` over the union.

With ``extra_topk == 0`` this reduces *bit-for-bit* to the single-cache
``dsv4_decode_reference`` (the concat is a no-op), which is the byte-identical
elision the kernel must honor under ``const_expr(has_extra_cache=False)``.

Public entry points
--------------------
  - dsv4_extra_decode_reference(q, main_kv_cache, main_indices, sm_scale,
        extra_kv_cache, extra_indices, *, page_block_size=64, pbs_extra=2,
        d_v=512, attn_sink=None, topk_length=None, extra_topk_length=None,
        main_kv_dequant=None, extra_kv_dequant=None)
        -> (O[num_tokens, num_heads, d_v] bf16, lse_log2[num_tokens, num_heads])
  - make_dsv4_extra_decode_case(num_heads=128, topk=64, extra_topk=128,
        num_blocks=64, page_block_size=64, pbs_extra=2, invalidate_half=True,
        seed=0, ...) -> dict with BOTH caches + indices + expected_O/lse

Run ``python -m tests.dsv4_extra_ref`` to execute the internal self-tests.
"""

from __future__ import annotations

import math

import torch

from b12x.testing.reference import dsv4_ref
from b12x.testing.reference.dsv4_ref import (  # noqa: F401  (re-exported traits used by callers/tests)
    DSV4_D_QK,
    DSV4_D_V,
    DSV4_DECODE_PAGE_BLOCK_SIZE,
    DSV4_KV_GMEM_STRIDE,
    dequantize_kv_dsv4,
    dsv4_decode_reference,
    quantize_kv_dsv4,
)

# BI = KV partition tile size in candidates (BLOCK_SIZE_N), DSV4_CAND_WINDOW in
# the kernel. num_splits = ceil(topk/BI) + ceil(extra_topk/BI).
DSV4_BI = 64


def dsv4_num_splits_dual(topk: int, extra_topk: int, *, bi: int = DSV4_BI) -> int:
    """num_splits the launcher computes for a dual-cache decode.

    Mirrors sparse_mla_sm120.py: ``ceil(topk/BI) + ceil(extra_topk/BI)``. With
    ``extra_topk == 0`` this equals the single-cache ``ceil(topk/BI)``.
    """
    return (topk + bi - 1) // bi + (extra_topk + bi - 1) // bi


def _build_virtual_pool(
    main_kv_dequant: torch.Tensor,
    extra_kv_dequant: torch.Tensor,
    main_indices: torch.Tensor,
    extra_indices: torch.Tensor,
    *,
    extra_topk_length: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Concatenate the two dequantized pools + indices into ONE virtual pool.

    Returns ``(virtual_kv_dequant, virtual_idx, main_s_kv)`` where
    ``virtual_kv_dequant`` is shaped (num_slots, 1, 1, d_qk) so it can be fed to
    :func:`dsv4_ref.dsv4_decode_reference` as a precomputed dequant pool with
    ``page_block_size=1`` (the slot id is then the absolute row index).

    Main slots occupy ``[0, main_s_kv)``; extra slots
    ``[main_s_kv, main_s_kv + extra_s_kv)``. Extra entries past
    ``extra_topk_length`` (or already -1) are forced to the -1 sentinel BEFORE
    the shift so they never select a real virtual slot.
    """
    d_qk = main_kv_dequant.shape[-1]
    main_s_kv = main_kv_dequant.shape[0] * main_kv_dequant.shape[1]

    virtual_kv = torch.cat(
        [
            main_kv_dequant.reshape(-1, d_qk),
            extra_kv_dequant.reshape(-1, d_qk),
        ],
        dim=0,
    ).reshape(-1, 1, 1, d_qk)

    ref_extra_idx = extra_indices.clone()
    if extra_topk_length is not None:
        extra_topk = ref_extra_idx.shape[-1]
        ar = torch.arange(extra_topk, device=ref_extra_idx.device).unsqueeze(0)
        past_len = ar >= extra_topk_length.clamp(min=0, max=extra_topk).unsqueeze(-1)
        ref_extra_idx = torch.where(
            past_len, torch.full_like(ref_extra_idx, -1), ref_extra_idx
        )
    extra_idx_shifted = torch.where(
        ref_extra_idx < 0, ref_extra_idx, ref_extra_idx + main_s_kv
    )
    virtual_idx = torch.cat([main_indices, extra_idx_shifted], dim=-1)
    return virtual_kv, virtual_idx, main_s_kv


def dsv4_extra_decode_reference(
    q: torch.Tensor,
    main_kv_cache: torch.Tensor,
    main_indices: torch.Tensor,
    sm_scale: float,
    extra_kv_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    *,
    page_block_size: int = DSV4_DECODE_PAGE_BLOCK_SIZE,
    pbs_extra: int = 2,
    d_v: int = DSV4_D_V,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    main_kv_dequant: torch.Tensor | None = None,
    extra_kv_dequant: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DSV4 dual-cache decode oracle: ONE softmax over main + extra topk rows.

    Args:
      q:              [num_tokens, num_heads, d_qk=512] bf16 (or float).
      main_kv_cache:  (main_nb, page_block_size, 1, 584) uint8, quantize_kv_dsv4.
      main_indices:   [num_tokens, topk] int32 slot ids into the MAIN pool
                      (slot = block*page_block_size + local). -1 = invalid.
      sm_scale:       softmax scale (typically d_qk**-0.5).
      extra_kv_cache: (extra_nb, pbs_extra, 1, 584) uint8, or None for
                      single-cache (then this is exactly dsv4_decode_reference).
      extra_indices:  [num_tokens, extra_topk] int32 slot ids into the EXTRA
                      pool (slot = block*pbs_extra + local). -1 = invalid.
      page_block_size: MAIN cache page block size (decode = 64).
      pbs_extra:      EXTRA cache page block size (DSv4 C128A passes 2).
      attn_sink:      optional [num_heads] fp32 per-head sink (single softmax
                      over the WHOLE union).
      topk_length:    optional [num_tokens] int32 MAIN valid length.
      extra_topk_length: optional [num_tokens] int32 EXTRA valid length.
      main_kv_dequant / extra_kv_dequant: optional precomputed dequant pools
                      (the case factory passes them to avoid recompute).

    Returns:
      (O[num_tokens, num_heads, d_v] bf16, lse_log2[num_tokens, num_heads] fp32).

    NOTE the slot-id base differs per section ONLY in its page block size: the
    kernel decodes ``block = idx / pbs`` and ``local = idx - block*pbs`` for the
    main pool (pbs=page_block_size) and the extra pool (pbs=pbs_extra). The byte
    layout WITHIN a block (IO_STRIDE=576, SCALE_BYTES_PER_TOKEN=8 footer) is the
    same; only the per-block stride scales with the page block size
    (stride = pbs * 584). This reference dequantizes each pool with its own
    page block size and then fuses by row index, so the pbs difference is
    captured by the cache shapes themselves.
    """
    if main_kv_dequant is None:
        main_kv_dequant = dequantize_kv_dsv4(main_kv_cache)

    # No extra section → byte-identical to the single-cache reference.
    if extra_kv_cache is None or extra_indices is None or extra_indices.shape[-1] == 0:
        return dsv4_decode_reference(
            q,
            main_kv_cache,
            main_indices,
            sm_scale,
            page_block_size=page_block_size,
            d_v=d_v,
            attn_sink=attn_sink,
            topk_length=topk_length,
            kv_dequant=main_kv_dequant,
        )

    if extra_kv_dequant is None:
        extra_kv_dequant = dequantize_kv_dsv4(extra_kv_cache)
    assert extra_kv_dequant.shape[1] == pbs_extra, (
        f"extra cache page_block_size {extra_kv_dequant.shape[1]} != pbs_extra "
        f"{pbs_extra}"
    )

    virtual_kv, virtual_idx, _ = _build_virtual_pool(
        main_kv_dequant,
        extra_kv_dequant,
        main_indices,
        extra_indices,
        extra_topk_length=extra_topk_length,
    )

    # Fuse main length-masking into the unified topk_length: the single-cache
    # reference masks main_idx positions >= topk_length, and the extra section
    # was already -1-masked in the virtual pool. Pass the precomputed virtual
    # dequant with page_block_size=1 so slot id == absolute row.
    if topk_length is not None:
        topk = main_indices.shape[-1]
        ar = torch.arange(topk, device=q.device).unsqueeze(0)
        main_past = ar >= topk_length.clamp(min=0, max=topk).unsqueeze(-1)
        virtual_idx = virtual_idx.clone()
        virtual_idx[:, :topk] = torch.where(
            main_past, torch.full_like(virtual_idx[:, :topk], -1), virtual_idx[:, :topk]
        )

    return dsv4_decode_reference(
        q,
        None,  # packed cache unused — kv_dequant supplied
        virtual_idx,
        sm_scale,
        page_block_size=1,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=None,  # already folded into virtual_idx sentinels
        kv_dequant=virtual_kv,
    )


def make_dsv4_extra_decode_case(
    num_heads: int = 128,
    topk: int = 64,
    extra_topk: int = 128,
    *,
    num_tokens: int = 1,
    num_blocks: int = 64,
    page_block_size: int = DSV4_DECODE_PAGE_BLOCK_SIZE,
    pbs_extra: int = 2,
    extra_num_blocks: int | None = None,
    with_sink: bool = False,
    invalidate_half: bool = True,
    with_extra_topk_length: bool = False,
    seed: int = 0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Build one self-consistent DSV4 DUAL-CACHE decode test case on `device`.

    Mirrors test_sparse_mla_sm120.py dual-cache decode inputs (randn/10
    clamp(-1,1), sm_scale = d_qk**-0.5, half-invalid topk on both sections).
    The MAIN and EXTRA pools are quantized with the SAME quantize_kv_dsv4 but
    have INDEPENDENT slot pools (each index is a slot id into its OWN pool).

    Returns a dict of torch tensors mirroring sparse_mla_sm120.py's extra_* arg
    contract:
      q                  [num_tokens, num_heads, 512] bf16
      kv_cache           (num_blocks, page_block_size, 1, 584) uint8  (MAIN)
      kv_dequant         (num_blocks, page_block_size, 1, 512) bf16
      topk_indices       [num_tokens, topk] int32 (-1 back half if invalidate_half)
      extra_kv_cache     (extra_num_blocks, pbs_extra, 1, 584) uint8  (EXTRA)
      extra_kv_dequant   (extra_num_blocks, pbs_extra, 1, 512) bf16
      extra_indices      [num_tokens, extra_topk] int32 (-1 back half)
      extra_topk_length  [num_tokens] int32 or None
      sm_scale           float
      attn_sink          [num_heads] fp32 or None
      page_block_size    int   (MAIN)
      pbs_extra          int   (EXTRA)
      topk               int
      extra_topk         int
      num_splits         int   (ceil(topk/BI) + ceil(extra_topk/BI))
      expected_O         [num_tokens, num_heads, 512] bf16
      expected_lse       [num_tokens, num_heads] fp32  (log2 LSE)
    """
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    d_qk, d_v = DSV4_D_QK, DSV4_D_V

    if extra_num_blocks is None:
        # enough extra blocks to address every extra slot.
        extra_num_blocks = max((extra_topk + pbs_extra - 1) // pbs_extra, 1)

    main_s_kv = num_blocks * page_block_size
    extra_s_kv = extra_num_blocks * pbs_extra

    main_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk,
            device=device, dtype=torch.bfloat16, generator=gen,
        )
        / 10.0
    ).clamp(-1, 1)
    extra_bf16 = (
        torch.randn(
            extra_num_blocks, pbs_extra, 1, d_qk,
            device=device, dtype=torch.bfloat16, generator=gen,
        )
        / 10.0
    ).clamp(-1, 1)
    main_packed = quantize_kv_dsv4(main_bf16)
    extra_packed = quantize_kv_dsv4(extra_bf16)
    main_dequant = dequantize_kv_dsv4(main_packed)
    extra_dequant = dequantize_kv_dsv4(extra_packed)

    q = (
        torch.randn(
            num_tokens, num_heads, d_qk,
            device=device, dtype=dtype, generator=gen,
        )
        / 10.0
    ).clamp(-1, 1)

    main_idx = torch.randint(
        0, main_s_kv, (num_tokens, topk), device=device, dtype=torch.int32, generator=gen
    )
    extra_idx = torch.randint(
        0, extra_s_kv, (num_tokens, extra_topk),
        device=device, dtype=torch.int32, generator=gen,
    )
    if invalidate_half:
        if topk >= 2:
            main_idx[:, topk // 2 :] = -1
        if extra_topk >= 2:
            extra_idx[:, extra_topk // 2 :] = -1

    extra_topk_length = None
    if with_extra_topk_length and extra_topk >= 2:
        # truncate the extra window to a fraction; past-length extra_indices stay
        # pointed at valid slots so the reference / kernel must mask via length.
        extra_topk_length = torch.full(
            (num_tokens,), max(extra_topk // 2, 1), dtype=torch.int32, device=device
        )

    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32, generator=gen) * 2.0
        if with_sink
        else None
    )

    sm_scale = d_qk**-0.5

    expected_O, expected_lse = dsv4_extra_decode_reference(
        q,
        main_packed,
        main_idx,
        sm_scale,
        extra_packed,
        extra_idx,
        page_block_size=page_block_size,
        pbs_extra=pbs_extra,
        d_v=d_v,
        attn_sink=attn_sink,
        extra_topk_length=extra_topk_length,
        main_kv_dequant=main_dequant,
        extra_kv_dequant=extra_dequant,
    )

    return {
        "q": q,
        "kv_cache": main_packed,
        "kv_dequant": main_dequant,
        "topk_indices": main_idx,
        "extra_kv_cache": extra_packed,
        "extra_kv_dequant": extra_dequant,
        "extra_indices": extra_idx,
        "extra_topk_length": extra_topk_length,
        "sm_scale": sm_scale,
        "attn_sink": attn_sink,
        "page_block_size": page_block_size,
        "pbs_extra": pbs_extra,
        "topk": topk,
        "extra_topk": extra_topk,
        "num_splits": dsv4_num_splits_dual(topk, extra_topk),
        "expected_O": expected_O,
        "expected_lse": expected_lse,
    }


# ── Internal self-tests ──────────────────────────────────────────────────────
def _self_test(device: str | torch.device = "cuda") -> None:
    device = torch.device(device)
    torch.manual_seed(0)

    # (1) extra_topk == 0 reduces BIT-FOR-BIT to the single-cache reference.
    #     (the dual oracle's concat/shift is a no-op when there is no extra
    #     section -> the const_expr(has_extra_cache=False) elision invariant.)
    single = dsv4_ref.make_dsv4_decode_case(
        num_heads=8, topk=64, num_tokens=2, num_blocks=4,
        invalidate_half=True, with_sink=False, device=device, seed=3,
    )
    O_dual, lse_dual = dsv4_extra_decode_reference(
        single["q"],
        single["kv_cache"],
        single["topk_indices"],
        single["sm_scale"],
        None,  # no extra cache
        None,
        page_block_size=single["page_block_size"],
        main_kv_dequant=single["kv_dequant"],
    )
    assert torch.equal(O_dual, single["expected_O"]), "extra_topk=0 O mismatch"
    assert torch.equal(
        torch.nan_to_num(lse_dual, neginf=-1e30),
        torch.nan_to_num(single["expected_lse"], neginf=-1e30),
    ), "extra_topk=0 LSE not bit-identical to single-cache reference"

    # (2) Brute-force dense cross-check on a TINY dual case (no online softmax,
    #     plain float matmul over the dequantized UNION of main+extra rows).
    case = make_dsv4_extra_decode_case(
        num_heads=4, topk=8, extra_topk=6, num_tokens=2,
        num_blocks=2, page_block_size=64, pbs_extra=2,
        invalidate_half=False, with_sink=False, device=device, seed=4,
    )
    q = case["q"].float()
    main_pool = case["kv_dequant"].reshape(-1, DSV4_D_QK).float()  # [main_s_kv, 512]
    extra_pool = case["extra_kv_dequant"].reshape(-1, DSV4_D_QK).float()
    main_s_kv = main_pool.shape[0]
    union_pool = torch.cat([main_pool, extra_pool], dim=0)  # [tot, 512]
    midx = case["topk_indices"].long()
    eidx = case["extra_indices"].long()
    eidx_shift = torch.where(eidx < 0, eidx, eidx + main_s_kv)
    union_idx = torch.cat([midx, eidx_shift], dim=-1)  # [t, topk+extra_topk]
    sm_scale = case["sm_scale"]
    nt, nh, _ = q.shape
    bruteO = torch.zeros(nt, nh, DSV4_D_V, device=device)
    bruteLSE = torch.zeros(nt, nh, device=device)
    for t in range(nt):
        valid = union_idx[t] >= 0
        rows = union_pool.index_select(0, union_idx[t].clamp(min=0))  # [K, 512]
        for h in range(nh):
            logits = (q[t, h] @ rows.t()) * sm_scale  # [K]
            logits = logits.masked_fill(~valid, float("-inf"))
            m = logits.max()
            w = torch.exp(logits - m)
            denom = w.sum()
            bruteO[t, h] = (w @ rows[:, :DSV4_D_V]) / denom
            bruteLSE[t, h] = (m + torch.log(denom)) / math.log(2.0)
    torch.testing.assert_close(case["expected_O"].float(), bruteO, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(case["expected_lse"], bruteLSE, atol=1e-3, rtol=1e-3)

    # (3) num_splits spans BOTH sections.
    assert case["num_splits"] == dsv4_num_splits_dual(case["topk"], case["extra_topk"])
    big = make_dsv4_extra_decode_case(
        num_heads=16, topk=128, extra_topk=2176, num_blocks=16,
        page_block_size=64, pbs_extra=2, device=device, seed=5,
    )
    # 128/64 + 2176/64 = 2 + 34 = 36 (>32, the old merge bound).
    assert big["num_splits"] == 36, big["num_splits"]
    assert torch.isfinite(big["expected_O"].float()).all()
    assert torch.isfinite(big["expected_lse"]).all()

    # (4) extra_topk_length truncation: shrinking the extra window changes the
    #     result, and past-length entries are masked even though they point at
    #     valid slots.  Cross-check the truncated case against a brute force.
    case_l = make_dsv4_extra_decode_case(
        num_heads=4, topk=8, extra_topk=8, num_tokens=2,
        num_blocks=2, page_block_size=64, pbs_extra=2,
        invalidate_half=False, with_extra_topk_length=True,
        with_sink=False, device=device, seed=6,
    )
    assert case_l["extra_topk_length"] is not None
    q = case_l["q"].float()
    main_pool = case_l["kv_dequant"].reshape(-1, DSV4_D_QK).float()
    extra_pool = case_l["extra_kv_dequant"].reshape(-1, DSV4_D_QK).float()
    main_s_kv = main_pool.shape[0]
    union_pool = torch.cat([main_pool, extra_pool], dim=0)
    midx = case_l["topk_indices"].long()
    eidx = case_l["extra_indices"].clone()
    elen = case_l["extra_topk_length"]
    etk = eidx.shape[-1]
    ar = torch.arange(etk, device=device).unsqueeze(0)
    eidx[ar >= elen.unsqueeze(-1)] = -1
    eidx = eidx.long()
    eidx_shift = torch.where(eidx < 0, eidx, eidx + main_s_kv)
    union_idx = torch.cat([midx, eidx_shift], dim=-1)
    sm_scale = case_l["sm_scale"]
    nt, nh, _ = q.shape
    bruteO = torch.zeros(nt, nh, DSV4_D_V, device=device)
    for t in range(nt):
        valid = union_idx[t] >= 0
        rows = union_pool.index_select(0, union_idx[t].clamp(min=0))
        for h in range(nh):
            logits = ((q[t, h] @ rows.t()) * sm_scale).masked_fill(~valid, float("-inf"))
            w = torch.exp(logits - logits.max())
            bruteO[t, h] = (w @ rows[:, :DSV4_D_V]) / w.sum()
    torch.testing.assert_close(case_l["expected_O"].float(), bruteO, atol=2e-2, rtol=2e-2)

    # (5) sink path over the UNION runs and stays finite (single softmax folds
    #     the sink mass once across both sections).
    case_s = make_dsv4_extra_decode_case(
        num_heads=4, topk=8, extra_topk=6, num_tokens=2,
        num_blocks=2, page_block_size=64, pbs_extra=2,
        invalidate_half=False, with_sink=True, device=device, seed=7,
    )
    assert case_s["attn_sink"] is not None
    assert torch.isfinite(case_s["expected_O"].float()).all()
    assert torch.isfinite(case_s["expected_lse"]).all()

    # (6) full-size dispatch shapes (num_heads=128) with pbs_extra=2.
    full = make_dsv4_extra_decode_case(
        num_heads=128, topk=64, extra_topk=128, device=device, seed=8,
    )
    assert full["q"].shape == (1, 128, DSV4_D_QK)
    assert full["expected_O"].shape == (1, 128, DSV4_D_V)
    assert full["expected_lse"].shape == (1, 128)
    assert full["kv_cache"].shape[-1] == DSV4_KV_GMEM_STRIDE
    assert full["extra_kv_cache"].shape[-1] == DSV4_KV_GMEM_STRIDE
    assert full["extra_kv_cache"].shape[1] == 2  # pbs_extra
    assert full["num_splits"] == 1 + 2  # ceil(64/64)+ceil(128/64)

    print(
        "dsv4_extra_ref self-tests PASSED "
        "(extra=0 elision/brute-force/num_splits/topk_length/sink/full-dispatch)"
    )


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _self_test(dev)
