"""DSV4 sparse-MLA decode numeric reference + test-case harness (pure PyTorch).

This module is the *ground-truth numeric reference* for the P5 SM120 DSV4 decode
port. It has NO dependency on the CuTeDSL kernel, on b12x._lib, or on FlashInfer
runtime — it is pure PyTorch so build/verification agents can compare kernel
register state and final O/LSE against a trusted oracle at every stage.

It ports two things verbatim from FlashInfer ground truth:

  1. The DSV4 packed-KV byte layout + quantizer/dequantizer, ported from
     flashinfer-other/tests/attention/test_sparse_mla_sm120.py::quantize_kv_dsv4
     and the UE8M0 pow2-ceil convention in
     include/.../common/fp8_quant.cuh (Step 3) +
     include/.../model/scale_convert.cuh (fp32_to_ue8m0 / ue8m0_to_fp32).

  2. The sparse-MLA attention math (dense SDPA over the gathered topk KV rows),
     ported from test_sparse_mla_sm120.py::_ref_sparse_attn, which is the exact
     oracle the FlashInfer kernel test asserts against. Note V_HAS_ROPE=true for
     DSV4: V == the full d_qk=512 KV row (nope 448 dequant + rope 64 bf16), so
     the PV output O is [num_heads, 512] and the rope component flows through PV
     unchanged. The kernel (decode_dsv4_kernel.cuh) splits this into acc_nope
     (S6) + acc_rope (S6b) but the fused result equals dense SDPA over the union
     row, which is what this reference computes.

DSV4 traits (verified_traits.md / make_unified_traits(DSV4, FP8, UE8M0_BYTE)):
  d_nope=448  d_rope=64  d_v=512  d_qk=512
  quant_tile=64  num_scales=7  n_v_chunks=7
  kv_gmem_stride=584 (448 fp8 + 128 rope-bf16 + 8 footer)
  page_block_size (decode test) = 64

Byte layout (authoritative = FlashInfer test + kernel IO, NOT a naive per-token
concatenation). Packed cache shape is (num_blocks, page_block_size, 1, 584)
uint8. Within ONE block of `bs` tokens the 584*bs bytes are:
    [0          : bs*576)   token DATA, token t at offset t*576:
                                [0   : 448)  nope  (448 e4m3 bytes)
                                [448 : 576)  rope  (64 bf16 = 128 bytes)
    [bs*576     : bs*584)   scale FOOTER, token t at offset bs*576 + t*8:
                                7 x UE8M0 scale bytes (one per 64-elem nope tile)
                                + 1 pad byte
The footer is grouped AFTER all token data within the block (it is NOT
interleaved immediately behind each token's data). This matches the kernel's
IO gather: scale_base = block*stride + pbs*IO_STRIDE + local*8, with
IO_STRIDE=576, and data_base = block*stride + local*576.

Public entry points (the build/verification agents call these):
  - quantize_kv_dsv4(kv_bf16) -> packed uint8 cache         (FlashInfer-identical)
  - dequantize_kv_dsv4(packed) -> bf16 KV                    (inverse)
  - dsv4_decode_reference(q, packed_kv_cache, topk_indices, sm_scale,
        page_block_size=64, d_v=512, attn_sink=None, topk_length=None)
        -> (O[num_tokens, num_heads, d_v] bf16, lse_log2[num_tokens, num_heads])
  - make_dsv4_decode_case(num_heads=128, topk=64, num_tokens=1, ...) -> dict

Run `python -m tests.dsv4_ref` to execute the internal self-tests.
"""

from __future__ import annotations

import math

import torch

# ── DSV4 traits (mirrors make_unified_traits(DSV4, FP8, UE8M0_BYTE)) ──────────
DSV4_D_NOPE = 448
DSV4_D_ROPE = 64
DSV4_D_V = 512
DSV4_D_QK = 512  # d_nope + d_rope
DSV4_QUANT_TILE = 64
DSV4_NUM_SCALES = 7  # 448 / 64
DSV4_N_V_CHUNKS = 7
DSV4_KV_GMEM_STRIDE = 584  # 448 fp8 + 128 rope-bf16 + 8 footer
DSV4_IO_STRIDE = 576  # 448 + 64*2  (per-token DATA stride within a block)
DSV4_SCALE_BYTES_PER_TOKEN = 8  # 7 UE8M0 + 1 pad
DSV4_DECODE_PAGE_BLOCK_SIZE = 64
FP8_MAX = 448.0  # e4m3 max magnitude (matches fp8_quant.cuh FP8_MAX)


# ── UE8M0 helpers (port of scale_convert.cuh + fp8_quant.cuh Step 3) ──────────
def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor) -> torch.Tensor:
    """Round an (inverse) scale up to the nearest power of two (FlashMLA conv.).

    Bit-exact with fp8_quant.cuh Step 3:
        raw = max(amax, 1e-4) / FP8_MAX
        if mantissa(raw) != 0:  raw = next power of two >= raw
    Implemented here as 2**ceil(log2(clamp_min(x, 1e-4))) which is the
    FlashInfer test helper. The two agree on all normal positive inputs because
    ceil(log2(2^e)) == e for exact powers of two and rounds up otherwise.
    """
    return torch.pow(2.0, torch.clamp_min(scales_inv, 1e-4).log2().ceil())


def _fp32_to_ue8m0_bytes(scale_fp32: torch.Tensor) -> torch.Tensor:
    """Extract the IEEE-754 exponent byte of an FP32 power-of-2 scale.

    Port of fp32_to_ue8m0 (scale_convert.cuh): byte = (float_bits >> 23) & 0xFF.
    """
    bits = scale_fp32.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def ue8m0_byte_to_fp32(ue8m0: torch.Tensor) -> torch.Tensor:
    """Port of ue8m0_to_fp32 (scale_convert.cuh): value = 2^(byte - 127).

    Note: bits = byte << 23 reconstructs the FP32 with that exponent and zero
    mantissa, i.e. exactly 2^(byte-127). We compute it via 2**(byte-127) which
    is identical for the byte range used here (avoids an int->float view dance).
    """
    return torch.pow(2.0, ue8m0.to(torch.float32) - 127.0)


def pow2_ceil_ue8m0_byte(value_fp32: torch.Tensor) -> torch.Tensor:
    """Bit-exact pow2-ceil then exponent-byte extraction (the Q-scale path).

    This is the *reference* for the kernel's NEW pow2_ceil_ue8m0 op (S0 Q quant).
    It mirrors fp8_quant.cuh exactly using integer bit-twiddling rather than
    log2 (which is why it is bit-exact at/near powers of two, unlike the
    lg2.approx-based cvt_f32_to_ue8m0 that the blueprint warns against):
        bits = float_as_uint(value)
        if (bits & 0x007FFFFF):  bits = (bits + 0x00800000) & 0x7F800000
        return (bits >> 23) & 0xFF
    """
    bits = value_fp32.to(torch.float32).view(torch.int32)
    mant_nonzero = (bits & 0x007FFFFF) != 0
    rounded = (bits + 0x00800000) & 0x7F800000
    bits = torch.where(mant_nonzero, rounded, bits)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


# ── Quantizer / dequantizer (ported verbatim from FlashInfer test) ───────────
def quantize_kv_dsv4(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into the DSV4 FP8 FOOTER format.

    Input  shape (nb, bs, 1, 512) bf16  (d_qk = d_nope 448 + d_rope 64).
    Output shape (nb, bs, 1, 584) uint8 — physical layout per block:
        [0          : bs*576)  token data (nope 448B FP8 + rope 128B BF16) per tok
        [bs*576     : bs*584)  scale footer (7 UE8M0 + 1 pad) per tok

    Bit-identical to test_sparse_mla_sm120.py::quantize_kv_dsv4.
    """
    d_nope, d_rope, tile_size, num_tiles = (
        DSV4_D_NOPE,
        DSV4_D_ROPE,
        DSV4_QUANT_TILE,
        DSV4_NUM_SCALES,
    )
    data_stride = d_nope + d_rope * 2  # 576
    scale_bytes = num_tiles + 1  # 8
    bpt = data_stride + scale_bytes  # 584
    nb, bs, hk, d = kv_bf16.shape
    assert d == DSV4_D_QK and hk == 1, f"expected (nb,bs,1,512), got {kv_bf16.shape}"
    kv = kv_bf16.squeeze(2)

    block_bytes = bs * bpt
    result_flat = torch.zeros(nb, block_bytes, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[..., ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / FP8_MAX)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        ue8m0 = _fp32_to_ue8m0_bytes(scale)

        for tok in range(bs):
            data_off = tok * data_stride + ti * tile_size
            result_flat[:, data_off : data_off + tile_size] = fp8[:, tok].view(torch.uint8)
            scale_off = bs * data_stride + tok * scale_bytes + ti
            result_flat[:, scale_off] = ue8m0[:, tok]

    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    rope = rope.reshape(nb, bs, d_rope * 2)
    for tok in range(bs):
        rope_off = tok * data_stride + d_nope
        result_flat[:, rope_off : rope_off + d_rope * 2] = rope[:, tok]

    return result_flat.view(nb, bs, 1, bpt)


def dequantize_kv_dsv4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack DSV4 FP8 FOOTER -> bf16. Inverse of :func:`quantize_kv_dsv4`.

    Output shape (nb, bs, 1, 512) bf16. Bit-identical to
    test_sparse_mla_sm120.py::dequantize_kv_dsv4.
    """
    d_nope, d_rope, tile_size, num_tiles = (
        DSV4_D_NOPE,
        DSV4_D_ROPE,
        DSV4_QUANT_TILE,
        DSV4_NUM_SCALES,
    )
    data_stride = d_nope + d_rope * 2
    scale_bytes = num_tiles + 1
    bpt = data_stride + scale_bytes
    nb, bs, _, _ = packed.shape
    result = torch.zeros(nb, bs, DSV4_D_QK, dtype=torch.bfloat16, device=packed.device)
    p = packed.view(nb, bs * bpt)

    for tok in range(bs):
        data_off = tok * data_stride
        scale_off = bs * data_stride + tok * scale_bytes
        for ti in range(num_tiles):
            fp8_off = data_off + ti * tile_size
            fp8 = p[:, fp8_off : fp8_off + tile_size].view(torch.float8_e4m3fn).float()
            ue8m0 = p[:, scale_off + ti]
            scale = ue8m0_byte_to_fp32(ue8m0)
            result[:, tok, ti * tile_size : (ti + 1) * tile_size] = (
                fp8 * scale.unsqueeze(-1)
            ).to(torch.bfloat16)
        rope_off = data_off + d_nope
        rope_bytes = p[:, rope_off : rope_off + d_rope * 2].contiguous()
        result[:, tok, d_nope:] = rope_bytes.view(torch.bfloat16).reshape(nb, d_rope)

    return result.view(nb, bs, 1, DSV4_D_QK)


# ── Sparse-MLA decode reference (ported from _ref_sparse_attn) ───────────────
def dsv4_decode_reference(
    q: torch.Tensor,
    packed_kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    sm_scale: float,
    *,
    page_block_size: int = DSV4_DECODE_PAGE_BLOCK_SIZE,
    d_v: int = DSV4_D_V,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    kv_dequant: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DSV4 decode oracle: dense SDPA over the sparse-gathered + dequantized KV.

    Args:
      q:            [num_tokens, num_heads, d_qk=512] bf16 (or float).
      packed_kv_cache: (nb, page_block_size, 1, 584) uint8 from quantize_kv_dsv4.
      topk_indices: [num_tokens, topk] int32, flat slot ids into the KV pool
                    (slot = block*page_block_size + local). -1 = invalid sentinel.
      sm_scale:     softmax scale (typically d_qk**-0.5).
      attn_sink:    optional [num_heads] fp32 per-head sink (FlashMLA V4 merge).
      topk_length:  optional [num_tokens] int32 per-token valid length.
      kv_dequant:   optional precomputed dequantized KV (skips dequant); the
                    self-tests pass it to avoid recomputation.

    Returns:
      (O[num_tokens, num_heads, d_v] bf16, lse_log2[num_tokens, num_heads] fp32).

    The output O[..., :d_v] is the softmax-weighted sum of the gathered rows'
    first d_v=512 dims — i.e. the FULL union row (nope+rope). This is the
    V_HAS_ROPE=true semantics: the rope tail is part of V. lse is returned in
    log2 space (log2(sum) + max in the kernel's base-2 convention), matching the
    kernel epilogue and the reused rs-1 base-2 merge.
    """
    if kv_dequant is None:
        kv_dequant = dequantize_kv_dsv4(packed_kv_cache)
    # Flatten to a (num_slots, d_qk) pool. The slot id = block*pbs + local.
    nb, bs, _, d_qk = kv_dequant.shape
    assert bs == page_block_size, (
        f"packed cache page_block_size {bs} != {page_block_size}"
    )
    kv_flat = kv_dequant.view(nb * bs, d_qk).float()

    num_tokens, num_heads, d_qk_q = q.shape
    assert d_qk_q == d_qk, f"q d_qk {d_qk_q} != cache d_qk {d_qk}"
    topk = topk_indices.shape[-1]
    q_f = q.float()

    idx_fixed = topk_indices.clamp(min=0).long()
    invalid = topk_indices < 0
    if topk_length is not None:
        ar = torch.arange(topk, device=q.device).unsqueeze(0)
        invalid = invalid | (ar >= topk_length.unsqueeze(-1))

    gathered = kv_flat.index_select(0, idx_fixed.view(-1)).view(num_tokens, topk, d_qk)
    # logits P[t, h, k] = (q[t,h] . K[t,k]) * sm_scale
    P = torch.einsum("thd,tkd->thk", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")

    lse_e = torch.logsumexp(P, dim=-1)  # natural-log LSE [t, h]
    lse_safe = lse_e.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out_f = torch.einsum("thk,tkd->thd", weights, gathered[..., :d_v])

    ln2 = math.log(2.0)
    lse_log2 = lse_e / ln2

    if attn_sink is not None:
        sink = attn_sink.float()  # [num_heads], natural-log domain
        sink_log2 = sink / ln2
        factor = torch.sigmoid(lse_e.float() - sink.unsqueeze(0))  # [t, h]
        out_f = out_f * factor.unsqueeze(-1)
        lse_log2 = torch.where(
            lse_log2 == float("-inf"),
            sink_log2.unsqueeze(0).expand_as(lse_log2),
            lse_log2 + torch.log2(1.0 + torch.exp2(sink_log2.unsqueeze(0) - lse_log2)),
        )

    return out_f.to(torch.bfloat16), lse_log2


# ── Test-case factory ────────────────────────────────────────────────────────
def make_dsv4_decode_case(
    num_heads: int = 128,
    topk: int = 64,
    *,
    num_tokens: int = 1,
    num_blocks: int = 64,
    page_block_size: int = DSV4_DECODE_PAGE_BLOCK_SIZE,
    with_sink: bool = False,
    invalidate_half: bool = True,
    seed: int = 0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Build one self-consistent DSV4 decode test case on `device`.

    Mirrors test_sparse_mla_sm120.py::test_sparse_mla_sm120_decode_dsv4 inputs
    (randn/10 clamp(-1,1), sm_scale = d_qk**-0.5, half-invalid topk).

    Returns a dict of torch tensors:
      q              [num_tokens, num_heads, 512] bf16
      kv_cache       (num_blocks, page_block_size, 1, 584) uint8 (packed)
      kv_dequant     (num_blocks, page_block_size, 1, 512) bf16
      topk_indices   [num_tokens, topk] int32  (-1 in the back half if invalidate_half)
      sm_scale       float
      attn_sink      [num_heads] fp32 or None
      page_block_size int
      expected_O     [num_tokens, num_heads, 512] bf16   (reference output)
      expected_lse   [num_tokens, num_heads] fp32        (reference log2 LSE)
    """
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    d_qk, d_v = DSV4_D_QK, DSV4_D_V
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk,
            device=device, dtype=torch.bfloat16, generator=gen,
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4(kv_bf16)
    kv_dequant = dequantize_kv_dsv4(kv_packed)

    q = (
        torch.randn(
            num_tokens, num_heads, d_qk,
            device=device, dtype=dtype, generator=gen,
        )
        / 10.0
    ).clamp(-1, 1)

    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32, generator=gen
    )
    if invalidate_half and topk >= 2:
        indices[:, topk // 2 :] = -1

    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32, generator=gen) * 2.0
        if with_sink
        else None
    )

    sm_scale = d_qk**-0.5

    expected_O, expected_lse = dsv4_decode_reference(
        q,
        kv_packed,
        indices,
        sm_scale,
        page_block_size=page_block_size,
        d_v=d_v,
        attn_sink=attn_sink,
        kv_dequant=kv_dequant,
    )

    return {
        "q": q,
        "kv_cache": kv_packed,
        "kv_dequant": kv_dequant,
        "topk_indices": indices,
        "sm_scale": sm_scale,
        "attn_sink": attn_sink,
        "page_block_size": page_block_size,
        "expected_O": expected_O,
        "expected_lse": expected_lse,
    }


# ── Internal self-tests ──────────────────────────────────────────────────────
def _self_test(device: str | torch.device = "cuda") -> None:
    device = torch.device(device)
    torch.manual_seed(0)

    # (1) UE8M0 pow2 round-trip: exponent-byte extraction of a pow2 scale, then
    #     reconstruction, is exact; and pow2_ceil_ue8m0_byte agrees with the
    #     log2-ceil helper used in the quantizer on positive normals.
    raw = torch.rand(10000, device=device) * 4.0 + 1e-3
    scale_log2ceil = _cast_scale_inv_to_ue8m0(raw)  # quantizer path (log2.ceil)
    byte_log2 = _fp32_to_ue8m0_bytes(scale_log2ceil)
    byte_bitexact = pow2_ceil_ue8m0_byte(raw)  # kernel S0 path (bit-twiddle)
    assert torch.equal(byte_log2, byte_bitexact), (
        "pow2_ceil_ue8m0_byte disagrees with quantizer log2-ceil exponent byte"
    )
    # reconstructing 2^(byte-127) reproduces the pow2 scale exactly.
    recon = ue8m0_byte_to_fp32(byte_bitexact)
    assert torch.allclose(recon, scale_log2ceil, rtol=0, atol=0), (
        "ue8m0 byte round-trip is not exact for pow2 scales"
    )

    # (2) dequant(quantize(x)) round-trips within e4m3 + per-tile-pow2 tolerance.
    nb, bs = 4, DSV4_DECODE_PAGE_BLOCK_SIZE
    kv_bf16 = (
        torch.randn(nb, bs, 1, DSV4_D_QK, device=device, dtype=torch.bfloat16) / 10.0
    ).clamp(-1, 1)
    packed = quantize_kv_dsv4(kv_bf16)
    assert packed.shape == (nb, bs, 1, DSV4_KV_GMEM_STRIDE), packed.shape
    assert packed.dtype == torch.uint8
    deq = dequantize_kv_dsv4(packed)
    orig = kv_bf16.squeeze(2).float()
    got = deq.squeeze(2).float()
    # rope tail is bf16-exact (no quant); nope is e4m3 with per-64 pow2 scale.
    rope_err = (got[..., DSV4_D_NOPE:] - orig[..., DSV4_D_NOPE:]).abs().max().item()
    assert rope_err < 1e-2, f"rope round-trip error too large: {rope_err}"
    nope_rel = (
        (got[..., : DSV4_D_NOPE] - orig[..., : DSV4_D_NOPE]).abs()
        / orig[..., : DSV4_D_NOPE].abs().clamp(min=1e-3)
    )
    # e4m3 has ~2 mantissa bits at this scale; per-element rel error < ~0.1.
    assert nope_rel.median().item() < 0.1, (
        f"nope round-trip median rel error too large: {nope_rel.median().item()}"
    )

    # (3) Reference matches an INDEPENDENT brute-force dense attention on a tiny
    #     case (no online softmax, plain float matmul) over the dequantized KV.
    case = make_dsv4_decode_case(
        num_heads=4, topk=8, num_tokens=2, num_blocks=2,
        invalidate_half=False, with_sink=False, device=device,
    )
    q = case["q"].float()
    deq = case["kv_dequant"].view(-1, DSV4_D_QK).float()
    idx = case["topk_indices"].long()
    sm_scale = case["sm_scale"]
    nt, nh, _ = q.shape
    bruteO = torch.zeros(nt, nh, DSV4_D_V, device=device)
    bruteLSE = torch.zeros(nt, nh, device=device)
    for t in range(nt):
        rows = deq.index_select(0, idx[t])  # [topk, d_qk]
        for h in range(nh):
            logits = (q[t, h] @ rows.t()) * sm_scale  # [topk]
            m = logits.max()
            w = torch.exp(logits - m)
            denom = w.sum()
            bruteO[t, h] = (w @ rows[:, :DSV4_D_V]) / denom
            bruteLSE[t, h] = (m + torch.log(denom)) / math.log(2.0)
    torch.testing.assert_close(
        case["expected_O"].float(), bruteO, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(case["expected_lse"], bruteLSE, atol=1e-3, rtol=1e-3)

    # (4) sink path: with a sink, output is scaled by sigmoid(lse_e - sink) and
    #     lse folds in the sink mass — sanity check it runs and the sink lowers
    #     the output magnitude (sigmoid factor < 1 when sink is comparable).
    case_s = make_dsv4_decode_case(
        num_heads=4, topk=8, num_tokens=2, num_blocks=2,
        invalidate_half=False, with_sink=True, device=device, seed=1,
    )
    assert case_s["expected_O"].shape == (2, 4, DSV4_D_V)
    assert torch.isfinite(case_s["expected_O"].float()).all()
    assert torch.isfinite(case_s["expected_lse"]).all()

    # (5) all-invalid token: lse must be -inf, output 0 (no sink) — matches the
    #     kernel's empty-block mid_lse=-inf / acc=0 epilogue.
    case_e = make_dsv4_decode_case(
        num_heads=4, topk=4, num_tokens=1, num_blocks=2,
        invalidate_half=False, with_sink=False, device=device, seed=2,
    )
    case_e["topk_indices"][:] = -1
    O_e, lse_e = dsv4_decode_reference(
        case_e["q"], case_e["kv_cache"], case_e["topk_indices"], case_e["sm_scale"],
        kv_dequant=case_e["kv_dequant"],
    )
    assert torch.all(lse_e == float("-inf")), "all-invalid LSE must be -inf"
    assert torch.all(O_e.float() == 0.0), "all-invalid output must be 0"

    # (6) topk sweep {64, 128, 512} produces finite, correctly-shaped outputs.
    for tk in (64, 128, 512):
        c = make_dsv4_decode_case(num_heads=128, topk=tk, device=device)
        assert c["q"].shape == (1, 128, DSV4_D_QK)
        assert c["expected_O"].shape == (1, 128, DSV4_D_V)
        assert c["expected_lse"].shape == (1, 128)
        assert c["kv_cache"].shape[-1] == DSV4_KV_GMEM_STRIDE
        assert torch.isfinite(c["expected_O"].float()).all()
        # valid heads always have at least one valid candidate (back half -1),
        # so lse is finite.
        assert torch.isfinite(c["expected_lse"]).all()

    print("dsv4_ref self-tests PASSED (pow2/round-trip/brute-force/sink/empty/topk-sweep)")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _self_test(dev)
